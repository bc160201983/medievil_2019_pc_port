// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include "video_core/texture_cache/readback_queue.h"
#include "video_core/texture_cache/readback_sources.h"
#include <chrono>
#include "gow3_toggles.h"
#include <optional>
#include <cstdlib>
#include <source_location>
#include "video_core/renderer_vulkan/vk_staging_buffer_pool.h"

#include <condition_variable>
#include <mutex>
#include <thread>
#include <unordered_set>
#include <boost/container/small_vector.hpp>
#include <queue>
#include <tsl/robin_map.h>

#include "common/lru_cache.h"
#include "common/slot_vector.h"
#include "shader_recompiler/resource.h"
#include "video_core/multi_level_page_table.h"
#include "video_core/texture_cache/blit_helper.h"
#include "video_core/texture_cache/image.h"
#include "video_core/texture_cache/image_view.h"
#include "video_core/texture_cache/sampler.h"
#include "video_core/texture_cache/tile_manager.h"

namespace AmdGpu {
struct Liverpool;
}

namespace Vulkan {
class Runtime;
}

namespace VideoCore {

class BufferCache;
class PageManager;

class TextureCache {
    // Default values for garbage collection
    static constexpr s64 DEFAULT_PRESSURE_GC_MEMORY = 1_GB + 512_MB;
    static constexpr s64 DEFAULT_CRITICAL_GC_MEMORY = 3_GB;
    static constexpr s64 TARGET_GC_THRESHOLD = 8_GB;

    using ImageIds = boost::container::small_vector<ImageId, 16>;

    struct Traits {
        using Entry = ImageIds;
        static constexpr size_t AddressSpaceBits = 40;
        static constexpr size_t FirstLevelBits = 10;
        static constexpr size_t PageBits = 20;
    };
    using PageTable = MultiLevelPageTable<Traits>;

public:
    enum class BindingType : u32 {
        Texture,
        Storage,
        RenderTarget,
        DepthTarget,
        VideoOut,
    };

    struct ImageDesc {
        ImageInfo info;
        ImageViewInfo view_info;
        BindingType type{BindingType::Texture};

        ImageDesc() = default;
        ImageDesc(const AmdGpu::Image& image, const Shader::ImageResource& desc)
            : info{image, desc}, view_info{image, desc},
              type{desc.is_written ? BindingType::Storage : BindingType::Texture} {}
        ImageDesc(const AmdGpu::ColorBuffer& buffer, AmdGpu::CbDbExtent hint)
            : info{buffer, hint}, view_info{buffer}, type{BindingType::RenderTarget} {}
        ImageDesc(const AmdGpu::DepthBuffer& buffer, AmdGpu::DepthView view,
                  AmdGpu::DepthControl ctl, VAddr htile_address, AmdGpu::CbDbExtent hint,
                  bool write_buffer = false)
            : info{buffer, view.NumSlices(), htile_address, hint, write_buffer},
              view_info{buffer, view, ctl}, type{BindingType::DepthTarget} {}
        ImageDesc(const Libraries::VideoOut::BufferAttributeGroup& group, VAddr cpu_address)
            : info{group, cpu_address}, type{BindingType::VideoOut} {}
    };

public:
    TextureCache(const Vulkan::Instance& instance, Vulkan::Scheduler& scheduler,
                 Vulkan::Runtime& runtime, AmdGpu::Liverpool* liverpool, BufferCache& buffer_cache,
                 PageManager& tracker);
    ~TextureCache();

    /// gow3: changes whenever an image is registered or unregistered.
    [[nodiscard]] u64 RegistryGeneration() const noexcept {
        return registry_generation.load(std::memory_order_acquire);
    }

    /// gow3 region stamps (T-S1a): a sum that grows whenever an image overlapping
    /// [address, address + size) is registered or unregistered, by 4 MiB region (a region shared
    /// through the table's wrap only adds misses). RegistryGeneration changes for any image: one
    /// texture streaming in invalidated every memoized texture set (~half of their misses).
    static constexpr u32 RegionShift = 22, RegionCount = 1u << 16;
    [[nodiscard]] u64 RegionStamp(VAddr address, u64 size) const noexcept {
        const u64 first = address >> RegionShift;
        const u64 last = (address + std::max<u64>(size, 1) - 1) >> RegionShift;
        u64 stamp = 0;
        for (u64 r = first; r <= last && r - first < RegionCount; ++r) {
            stamp += region_gens[r & (RegionCount - 1)].load(std::memory_order_relaxed);
        }
        return stamp;
    }

    /// gow3: FindImage's access tick for an image found by a memoized lookup. The LRU touch
    /// FindImage also does happens in UpdateImage (FindTexture), which every binding reaches.
    void MarkFound(ImageId image_id) {
        slot_images[image_id].tick_accessed_last = scheduler.CurrentTick();
    }

    TileManager& GetTileManager() noexcept {
        return tile_manager;
    }

    /// Invalidates any image in the logical page range.
    void InvalidateMemory(VAddr addr, size_t size);

    /// Marks an image as dirty if it exists at the provided address.
    void InvalidateMemoryFromGPU(VAddr address, size_t max_size);

    /// Evicts any images that overlap the unmapped range.
    void UnmapMemory(VAddr cpu_addr, size_t size);

    /// Schedules a copy of pending images for download back to CPU memory.
    void ProcessDownloadImages();

    /// gow3 R1: true once after a readback copy was queued with precise readbacks: the caller
    /// submits once the readback locks are released (Rasterizer::ProcessDownloadImages), so the
    /// GPU makes the copy soon after the work it reads instead of at the frame's end.
    bool TakeReadbackSubmitWanted() {
        return readback_submit_wanted && !(readback_submit_wanted = false);
    }
    bool ResolveReadbacks(VAddr address, u64 size, bool assume_locks = true,
                          ReadbackSource source = ReadbackSource::Unknown);
    void CancelReadbacks(VAddr address, u64 size);

    /// gow3 (MediEvil HZB remap, gpu/shim/medievil_hzb.cpp): numbers the readback copies of
    /// [address, address + size) from the first call on. Publishes the copies the GPU has
    /// finished (no wait) and copies the first `copy_size` bytes of guest memory into `dst`.
    /// `published` is the number of the copy those bytes come from (0: none numbered yet),
    /// `pushed` the newest queued. ok is false without deferred readbacks or a free slot.
    /// `restart`: numbering starts again from the next copy queued (copies queued before
    /// carry none).
    struct ReadbackSnapshot {
        u64 pushed = 0, published = 0;
        bool ok = false;
    };
    ReadbackSnapshot SnapshotReadback(VAddr address, u64 size, void* dst, u64 copy_size,
                                      bool restart = false);

    /// Retrieves the image handle of the image with the provided attributes.
    [[nodiscard]] ImageId FindImage(ImageDesc& desc, bool exact_fmt = false);

    /// Retrieves image whose address matches provided
    [[nodiscard]] ImageId FindImageFromRange(VAddr address, size_t size, bool ensure_valid = true);

    /// gow3: a FindView result remembered by the caller for the same image and view info.
    struct ViewMemo {
        ImageId image_id{};
        const void* backing = nullptr;
        ImageViewId view_id{};
    };

    /// Retrieves an image view with the properties of the specified image id.
    /// `refresh` false: an image found out of date is used as is (the texture binding helper
    /// checked it beforehand; only a guest write racing with the draw can make it so).
    [[nodiscard]] ImageView& FindTexture(ImageId image_id, const ImageDesc& desc,
                                         ViewMemo* memo = nullptr, bool refresh = true);

    /// Retrieves the render target with specified properties
    [[nodiscard]] ImageView& FindRenderTarget(ImageId image_id, const ImageDesc& desc);

    /// Retrieves the depth target with specified properties
    [[nodiscard]] ImageView& FindDepthTarget(ImageId image_id, const ImageDesc& desc);

    /// gow3: whether UpdateImage has nothing to do for this image (its fast path).
    [[nodiscard]] bool IsUpToDate(ImageId image_id) const {
        const Image& image = slot_images[image_id];
        const u32 flags = std::atomic_ref<u32>(const_cast<u32&>(reinterpret_cast<const u32&>(image.flags)))
                              .load(std::memory_order_acquire);
        constexpr u32 Dirty = static_cast<u32>(ImageFlagBits::Dirty);
        constexpr u32 Registered = static_cast<u32>(ImageFlagBits::Registered);
        constexpr u32 Aliased = static_cast<u32>(ImageFlagBits::Aliased);
        return (flags & (Dirty | Registered | Aliased)) == Registered &&
               image.track_addr == image.guest_begin && image.track_addr_end == image.guest_end &&
               image.lru_touched_tick == gc_tick;
    }

    /// Updates image contents if it was modified by CPU.
    void UpdateImage(ImageId image_id,
                     std::source_location caller = std::source_location::current()) {
        // gow3: a clean image already tracked and touched in this GC period needs nothing.
        // Every texture binding comes here; the mutex (shared with the fault handlers of the
        // guest threads) was ~3% of the GPU thread. Flags are read atomically: an invalidation
        // racing with this check races the same way with the locked path.
        if (!Gow3Toggle::Disabled(Gow3Toggle::UpdateImageFastPath)) {
            const Image& image = slot_images[image_id];
            const u32 flags = std::atomic_ref<u32>(const_cast<u32&>(reinterpret_cast<const u32&>(image.flags)))
                                  .load(std::memory_order_acquire);
            constexpr u32 Dirty = static_cast<u32>(ImageFlagBits::Dirty);
            constexpr u32 Registered = static_cast<u32>(ImageFlagBits::Registered);
            // Images sharing memory with another take the locked path: SynchronizeAlias.
            constexpr u32 Aliased = static_cast<u32>(ImageFlagBits::Aliased);
            if ((flags & (Dirty | Registered | Aliased)) == Registered &&
                image.track_addr == image.guest_begin && image.track_addr_end == image.guest_end &&
                image.lru_touched_tick == gc_tick) {
                return;
            }
        }
        scheduler.SyncWithDrawRecorder(); // before the lock: the draw recorder takes it too
        std::scoped_lock lock{mutex};
        Image& image = slot_images[image_id];
        TrackImage(image_id);
        TouchImage(image);
        RefreshImage(image, caller);
        SynchronizeAlias(image_id);
    }

    /// gow3: before a render / depth target binding: an image whose contents are undefined
    /// (Undefined: made at an address a guest unmap freed) is not uploaded first, the GPU renders
    /// into it. Any other first use uploads it as before (RefreshImage drops the flag).
    void DiscardUndefinedContents(ImageId image_id);

    /// gow3 (GOW3_CLEAR_SKIPS_UPLOAD): before a color target binding whose pass clears the whole
    /// image (a fast clear pending in its CMASK, one mip and layer): its memory is not uploaded
    /// first, the clear overwrites it. T-up2: 4K render-target-pool images uploaded 32-64 MB each
    /// right before being cleared, the large single-frame uploads while playing.
    void DiscardClearedContents(ImageId image_id);

    /// gow3: a GPU write to the image (render target, storage): images sharing its memory
    /// copy from it before their next read. From cuesta4/shadPS4 (eltutz), which fixed God of
    /// War III's exposure: it is measured into a small image and read through an alias.
    void PublishAliasWrite(ImageId image_id);

    /// Resolves overlap between existing cache image and pending merged image
    [[nodiscard]] std::tuple<ImageId, int, int> ResolveOverlap(const ImageInfo& info,
                                                               BindingType binding,
                                                               ImageId cache_img_id,
                                                               ImageId merged_image_id);

    /// Resolves depth overlap and either re-creates the image or returns existing one
    [[nodiscard]] ImageId ResolveDepthOverlap(const ImageInfo& requested_info, BindingType binding,
                                              ImageId cache_img_id);

    /// Creates a new image with provided image info and copies subresources from image_id
    [[nodiscard]] ImageId ExpandImage(const ImageInfo& info, ImageId image_id);

    /// Reuploads image contents.
    /// `caller` names the first use in the GOW3_TEXCACHE_STATS refresh list.
    void RefreshImage(Image& image,
                      std::source_location caller = std::source_location::current());

    /// Retrieves the sampler that matches the provided S# descriptor.
    /// extra_lod_bias: gow3, added to the S#'s bias (reduced scene rendering).
    [[nodiscard]] vk::Sampler GetSampler(const AmdGpu::Sampler& sampler,
                                         AmdGpu::BorderColorBuffer border_color_base,
                                         bool is_depth, float extra_lod_bias = 0.0f);

    /// gow3: the garbage collection period. A sampler GetSampler returned in this period stays
    /// alive for at least 16 more (the sampler GC), so its handle may be reused within it.
    [[nodiscard]] u64 GcTick() const noexcept {
        return std::atomic_ref<u64>(const_cast<u64&>(gc_tick)).load(std::memory_order_relaxed);
    }

    /// Retrieves the image with the specified id.
    Image* TryGetImage(ImageId id, u64 uid) {
        if (!slot_images.is_allocated(id)) return nullptr;
        auto& image = slot_images[id];
        return !uid || image.image_uid == uid ? &image : nullptr;
    }
    [[nodiscard]] Image& GetImage(ImageId id) {
        auto& image = slot_images[id];
        TouchImage(image);
        return image;
    }

    /// Retrieves the image view with the specified id.
    [[nodiscard]] ImageView& GetImageView(ImageId id) {
        return slot_image_views[id];
    }

    /// Get the associated depth stencil image if it is still valid.
    /// gow3 (from the Uncharted port): images that may stand for a stencil plane.
    static bool IsStencilProxyCandidate(const Image& image) {
        return image.info.pixel_format == vk::Format::eUndefined || image.info.num_bits == 8;
    }

    ImageId GetAssociatedDepth(Image& image) {
        if (!image.depth_id) {
            return {};
        }
        if (!IsStencilProxyCandidate(image)) {
            // A color texture reusing the stencil's memory reads its own data.
            image.DisassociateDepth();
            return {};
        }
        if (slot_images.is_allocated(image.depth_id)) {
            auto& depth_image = slot_images[image.depth_id];
            if (depth_image.image_uid == image.depth_uid &&
                depth_image.flags & ImageFlagBits::Registered) {
                return image.depth_id;
            }
        }
        // The linked depth image is no longer valid, disassociate it.
        image.DisassociateDepth();
        return {};
    }

    enum class MetaType {
        CMask,
        FMask,
        HTile,
    };

    /// Returns meta type if the specified address is a metadata surface.
    std::optional<MetaType> IsMeta(VAddr address) const {
        auto it = surface_metas.find(address);
        if (it != surface_metas.end()) {
            return it->second.type;
        }
        return std::nullopt;
    }

    /// Returns true if a slice of the specified metadata surface has been cleared.
    bool IsMetaCleared(VAddr address, u32 slice) const {
        const auto& it = surface_metas.find(address);
        if (it != surface_metas.end()) {
            return it.value().clear_mask & (1u << slice);
        }
        return false;
    }

    /// Clears all slices of the specified metadata surface.
    bool ClearMeta(VAddr address) {
        auto it = surface_metas.find(address);
        if (it != surface_metas.end()) {
            it.value().clear_mask = u32(-1);
            return true;
        }
        return false;
    }

    /// Updates the state of a slice of the specified metadata surface.
    bool TouchMeta(VAddr address, u32 slice, bool is_clear) {
        auto it = surface_metas.find(address);
        if (it != surface_metas.end()) {
            if (is_clear) {
                it.value().clear_mask |= 1u << slice;
            } else {
                it.value().clear_mask &= ~(1u << slice);
            }
            return true;
        }
        return false;
    }

    /// Runs the garbage collector.
    void RunGarbageCollector();

    template <typename Func>
    void ForEachImageInRegion(VAddr cpu_addr, size_t size, Func&& func) {
        using FuncReturn = typename std::invoke_result<Func, ImageId, Image&>::type;
        static constexpr bool BOOL_BREAK = std::is_same_v<FuncReturn, bool>;
        ImageIds images;
        ForEachPage(cpu_addr, size, [this, &images, cpu_addr, size, func](u64 page) {
            const auto it = page_table.find(page);
            if (it == nullptr) {
                if constexpr (BOOL_BREAK) {
                    return false;
                } else {
                    return;
                }
            }
            for (const ImageId image_id : *it) {
                Image& image = slot_images[image_id];
                if (image.flags & ImageFlagBits::Picked) {
                    continue;
                }
                if (!image.Overlaps(cpu_addr, size)) {
                    continue;
                }
                image.flags |= ImageFlagBits::Picked;
                images.push_back(image_id);
                if constexpr (BOOL_BREAK) {
                    if (func(image_id, image)) {
                        return true;
                    }
                } else {
                    func(image_id, image);
                }
            }
            if constexpr (BOOL_BREAK) {
                return false;
            }
        });
        for (const ImageId image_id : images) {
            slot_images[image_id].flags &= ~ImageFlagBits::Picked;
        }
    }

private:
    /// Iterate over all page indices in a range
    template <typename Func>
    static void ForEachPage(PAddr addr, size_t size, Func&& func) {
        static constexpr bool RETURNS_BOOL = std::is_same_v<std::invoke_result<Func, u64>, bool>;
        const u64 page_end = (addr + size - 1) >> Traits::PageBits;
        for (u64 page = addr >> Traits::PageBits; page <= page_end; ++page) {
            if constexpr (RETURNS_BOOL) {
                if (func(page)) {
                    break;
                }
            } else {
                func(page);
            }
        }
    }

    /// Copies image memory back to CPU.
    void DownloadImageMemory(ImageId image_id, bool sync = false);

    /// gow3: a recorded image-to-staging copy, written to guest memory after a GPU wait.
    struct PendingDownload {
        VAddr guest_address;
        u32 size;
        Vulkan::StagingBufferRef staging;
    };
    /// Records the copy of a GPU-modified image into a deferred staging allocation.
    std::optional<PendingDownload> RecordImageDownload(ImageId image_id);

    /// Thread function for copying downloaded images out to CPU memory.
    void DownloadedImagesThread(const std::stop_token& token);

    /// Create an image from the given parameters
    [[nodiscard]] ImageId InsertImage(const ImageInfo& info, VAddr cpu_addr);

    /// Register image in the page table
    void RegisterImage(ImageId image);

    /// Unregister image from the page table
    void UnregisterImage(ImageId image);

    /// Track CPU reads and writes for image
    void TrackImage(ImageId image_id);
    void TrackImageHead(ImageId image_id);
    void TrackImageTail(ImageId image_id);

    /// Stop tracking CPU reads and writes for image
    void UntrackImage(ImageId image_id);
    void UntrackImageHead(ImageId image_id);
    void UntrackImageTail(ImageId image_id);

    void MarkAsMaybeDirty(ImageId image_id, Image& image);

    /// Removes the image and any views/surface metas that reference it.
    void DeleteImage(ImageId image_id);

    /// Touch the image in the LRU cache.
    void TouchImage(const Image& image);

    /// gow3 GOW3_IMAGE_WATCH=WxH[:min_layers] (diagnostics, Mausoleum lighting): images of that
    /// size (and at least that many layers) get a timed log line for every event that can set or
    /// lose their contents (created, CPU / GPU dirty, uploaded from guest memory, bound as a render
    /// target or storage image, copied into, unmapped, freed); GOW3_IMAGE_WATCH_LINES caps them
    /// (default 4000).
    bool Watched(const ImageInfo& info) const;
    void WatchNote(const Image& image, const char* event, const std::string& detail = {});

    void FreeImage(ImageId image_id, u32 line = __builtin_LINE()) {
        const auto& image = slot_images[image_id];
        if (image_stats) {
            NoteImageChurn(false, line, image.info);
        }
        if (Watched(image.info)) {
            WatchNote(image, "freed", "at texture_cache line " + std::to_string(line));
        }
        // gow3 GOW3_UNDEFINED_GPU_FREED (default on; 0: off): the guest memory under a GPU-written
        // image never received the GPU's data (no write-back), so a new image made there holds
        // stale bytes, as after an unmap (C3): flagged Undefined at creation, a render / depth
        // target binding skips its upload. t-up3b: UE4's render-target pool reused memory without
        // an unmap; 4K targets uploaded 32-64 MB each right before being rendered into.
        static const bool gpu_freed = [] {
            const char* env = std::getenv("GOW3_UNDEFINED_GPU_FREED");
            return !(env && env[0] == '0');
        }();
        if (gpu_freed && True(image.flags & ImageFlagBits::GpuModified)) {
            if (gpu_freed_addresses.size() > 16384) {
                gpu_freed_addresses.clear();
            }
            gpu_freed_addresses.insert(image.info.guest_address);
        }
        ResolveReadbacks(image.info.guest_address, image.info.guest_size, true,
                         ReadbackSource::ImageFree);
        UntrackImage(image_id);
        UnregisterImage(image_id);
        DeleteImage(image_id);
    }

    void GarbageCollectImages();
    void GarbageCollectSamplers();

    /// GOW3_IMAGE_STATS=1: images registered and freed every 5 s (where, which sizes).
    void NoteImageChurn(bool registered, u32 line, const ImageInfo& info);
    const bool image_stats = std::getenv("GOW3_IMAGE_STATS") != nullptr;

private:
    const Vulkan::Instance& instance;
    Vulkan::Scheduler& scheduler;
    Vulkan::Runtime& runtime;
    AmdGpu::Liverpool* liverpool;
    BufferCache& buffer_cache;
    PageManager& tracker;
    BlitHelper blit_helper;
    TileManager tile_manager;
    Common::SlotVector<Image> slot_images;
    Common::SlotVector<ImageView> slot_image_views;
    tsl::robin_map<u64, Sampler> samplers;
    std::unordered_set<ImageId> download_images;
    u64 total_used_memory = 0;
    u64 gc_evictions = 0, gc_downloads = 0; ///< gow3: pressure report
    std::chrono::steady_clock::time_point gc_report_time{};
    u64 trigger_gc_memory = 0;
    u64 pressure_gc_memory = 0;
    u64 critical_gc_memory = 0;
    u64 total_used_samplers = 0;
    u64 trigger_gc_samplers = 0;
    u64 pressure_gc_samplers = 0;
    u64 critical_gc_samplers = 0;
    u64 gc_tick = 0;
    Common::LeastRecentlyUsedCache<ImageId, u64> lru_cache;
    Common::LeastRecentlyUsedCache<u64, u64> sampler_lru_cache;
    const bool readback_linear_images;
    const bool deferred_readbacks = [] {
        const char* value = std::getenv("GOW3_DEFERRED_READBACK");
        return value && value[0] == '1';
    }();
    /// gow3: GOW3_STALE_READBACK=1 (needs deferred readbacks). A pending copy only blocks CPU
    /// writes: CPU reads see the version the GPU published last (one frame late) instead of
    /// waiting for the GPU. Writes still wait, so no byte is lost.
    bool readback_submit_wanted = false;

    const bool stale_readbacks = deferred_readbacks && [] {
        const char* value = std::getenv("GOW3_STALE_READBACK");
        return value && value[0] == '1';
    }();
public:
    /// gow3 GOW3_CAPTURE_ON_TARGET (diagnostics, Mausoleum lighting): copies a float image of at
    /// most 256x256 back once (all mips and layers) and logs per mip its NaN and infinity counts
    /// and finite range, when the GPU is done. Call before the image's layout transition for a
    /// draw (the copy leaves it in the transfer layout).
    void LogTexelStats(Image& image);

private:
    template <bool track>
    void WatchReadback(VAddr address, u64 size);
    /// gow3 (plan-linux-deck.md "Mausoleum lighting"): in stale mode a CPU read of a range whose
    /// first copy is still pending sees bytes no copy wrote (nothing was published there yet).
    /// GOW3_READBACK_LOG=1 logs each first copy (queued, published, read by the CPU);
    /// First copies' CPU reads are watched too (GOW3_READBACK_FIRST_WAIT=0: not), so such a read
    /// waits for the GPU as without stale mode. Later copies stay stale (frame pacing unchanged).
    /// A copy read once at a level's load is a first copy: UE4's sky light average brightness
    /// (a 1x1 target at a new address each load) read as leftover bytes, 0 flooded the walls with
    /// reflections (Hall of Heroes teal, 01.00 white; research/patches/caches/readback-first-wait.md).
    const bool readback_log = [] {
        const char* value = std::getenv("GOW3_READBACK_LOG");
        return value && value[0] == '1';
    }();
    const bool readback_first_wait = stale_readbacks && [] {
        const char* value = std::getenv("GOW3_READBACK_FIRST_WAIT");
        return !(value && value[0] == '0');
    }();
    /// gow3 GOW3_TINY_TARGET_READBACK=1: tiny color targets and storage images (at most 64
    /// texels) are copied back to guest memory in their tiled layout too (texture_cache.cpp
    /// IsTinyTarget). Off by default: it copies many small per-frame targets back every frame
    /// (a large frame cost on the Deck), and it did not fix the Mausoleum (the GPU's own sky
    /// light result there is NaN; plan-linux-deck.md). Diagnostics only.
    const bool tiny_target_readback = [] {
        const char* value = std::getenv("GOW3_TINY_TARGET_READBACK");
        return value && value[0] == '1';
    }();
    std::unordered_set<VAddr> readback_published; ///< ranges with a published copy (readback_mutex)
    /// Drops the page watch a queued copy holds (a read watch too for a first copy).
    template <typename Entry>
    void UnwatchReadback(const Entry& entry);
    struct ReadbackImageStat {
        VAddr address;
        u32 width, height, bits, copies, cpu_accesses;
        vk::Format format;
        u64 Bytes() const { return u64{width} * height * bits / 8; }
    };
    std::array<ReadbackImageStat, 16> readback_images{};
    u64 readbacks_released = 0;
    ReadbackQueue<Vulkan::StagingBufferRef> pending_readbacks;
    std::mutex readback_mutex;
    struct TrackedReadback {
        VAddr address = 0;
        u64 size = 0, pushed = 0, published = 0;
    };
    std::array<TrackedReadback, 4> tracked_readbacks{}; ///< guarded by readback_mutex
    u64 NoteTrackedPush(VAddr address, u64 size);
    void NoteTrackedPublish(VAddr address, u64 size, u64 seq);
    std::atomic<bool> has_pending_readbacks{false};
    /// gow3 (impl-stage-b-bundle-2.md C5): the pages of the pending copies (ReadbackQueue::
    /// Spans), written under readback_mutex after every change that can add pages, read without
    /// it by the buffer cache's ResolveReadbacks calls (seq odd while it is written).
    static constexpr std::size_t ReadbackSpanCount = 4;
    struct ReadbackSpans {
        std::atomic<u32> seq{0}, count{0};
        std::array<std::atomic<u64>, ReadbackSpanCount> begin{}, end{};
    };
    ReadbackSpans readback_spans;
    void PublishReadbackSpans();
    [[nodiscard]] bool ReadbackSpansMiss(VAddr address, u64 size) const;
    u64 readbacks_queued = 0, readbacks_completed = 0, readbacks_waited = 0;
    u64 readbacks_canceled = 0, readbacks_peak_bytes = 0;
    std::chrono::steady_clock::time_point readback_report = std::chrono::steady_clock::now();
    ReadbackSourceStats readback_sources;
    void RetireReadbacks(bool polled = false);
    PageTable page_table;
    /// gow3: std::mutex that knows its owner, so the scheduler does not wait for the draw
    /// recording thread while this thread holds the lock that thread may need.
    struct OwnedMutex {
        void lock() {
            mutex.lock();
            owner.store(std::this_thread::get_id(), std::memory_order_relaxed);
        }
        bool try_lock() {
            if (!mutex.try_lock()) {
                return false;
            }
            owner.store(std::this_thread::get_id(), std::memory_order_relaxed);
            return true;
        }
        void unlock() {
            owner.store({}, std::memory_order_relaxed);
            mutex.unlock();
        }
        [[nodiscard]] bool HeldByCurrentThread() const noexcept {
            return owner.load(std::memory_order_relaxed) == std::this_thread::get_id();
        }
        std::mutex mutex;
        std::atomic<std::thread::id> owner{};
    };
    OwnedMutex mutex;

public:
    [[nodiscard]] bool LockHeldByCurrentThread() const noexcept {
        return mutex.HeldByCurrentThread();
    }

private:
    /// gow3: images with the same guest address (guarded by `mutex`). `backing` holds the
    /// complete contents; `writer` is a newer write not yet copied into it.
    struct AliasState {
        u64 backing_uid{};
        u64 writer_uid{};
        ImageId backing{};
        ImageId writer{};
        u32 members{};
        void ResetAuthority() {
            const u32 count = members;
            *this = {};
            members = count;
        }
    };
    tsl::robin_map<VAddr, AliasState> alias_states;
    std::atomic<u64> alias_generation{};
    [[nodiscard]] bool IsLiveImage(ImageId image_id, u64 image_uid) const {
        return image_id && slot_images.is_allocated(image_id) &&
               slot_images[image_id].image_uid == image_uid;
    }
    template <typename Func>
    void ForEachAlias(const Image& image, Func&& func);
    void SynchronizeAlias(ImageId image_id);
    [[nodiscard]] bool CommitAliasWriter(AliasState& state);
    void CopyAlias(ImageId src_id, ImageId dst_id, const Extent3D& extent);
    void InvalidateAlias(Image& image);
    // gow3: FindImage results for unchanged image registrations (guarded by `mutex`).
    struct FindImageCacheEntry {
        VAddr address = 0;
        u64 size = 0;
        Extent3D extent{};
        vk::Format format{};
        AmdGpu::ImageType type{};
        bool exact_fmt = false;
        BindingType binding{};
        u32 levels = 0;
        u32 layers = 0;
        u64 generation = ~0ULL;
        ImageId image_id{};
        int view_mip = -1;
        int view_slice = -1;
    };
    std::array<FindImageCacheEntry, 1024> find_image_cache{};
    std::atomic<u64> registry_generation{0};
    std::unique_ptr<std::atomic<u32>[]> region_gens{new std::atomic<u32>[RegionCount]{}};
    void BumpRegions(VAddr address, u64 size) noexcept {
        const u64 first = address >> RegionShift;
        const u64 last = (address + std::max<u64>(size, 1) - 1) >> RegionShift;
        for (u64 r = first; r <= last && r - first < RegionCount; ++r) {
            region_gens[r & (RegionCount - 1)].fetch_add(1, std::memory_order_release);
        }
    }
    /// gow3: GOW3_TEXCACHE_STATS: expands by image and reason since the last report (under mutex).
    std::unordered_map<std::string, u64> expand_sites;
    /// gow3: GOW3_TEXCACHE_STATS: images created since the last report, by description and binding;
    /// images freed by a guest unmap in the last ~30 s (address / size / format / extent key ->
    /// report number), to count how many come back as new images (under mutex).
    std::unordered_map<std::string, u64> create_sites;
    /// gow3: GOW3_TEXCACHE_STATS: images uploaded again (RefreshImage) since the last report, by
    /// description and the dirty flag that caused it -> {count, bytes} (own lock: refreshes can
    /// run outside `mutex`).
    std::unordered_map<std::string, std::pair<u64, u64>> refresh_sites;
    /// gow3: addresses of images freed by guest unmaps, not yet reused (under mutex).
    std::unordered_set<VAddr> unmapped_addresses;
    /// gow3: addresses of GPU-written images freed (GC, overlaps, unmaps), not yet reused
    /// (under mutex; GOW3_UNDEFINED_GPU_FREED).
    std::unordered_set<VAddr> gpu_freed_addresses;
    /// gow3 GOW3_PARK_UNMAPPED (default on; 0: off): render / depth targets a guest unmap frees,
    /// kept unregistered for the same target at the same address (under mutex). UE4's render
    /// target pool decommits a pass's targets while unused: the torch shadow maps (512x512 D16)
    /// and the 4K pair were freed and made again up to ~40 times a second (N-23).
    struct ParkedImage {
        ImageId id;
        std::chrono::steady_clock::time_point since;
    };
    std::vector<ParkedImage> parked_images;
    u64 parked_bytes = 0;
    std::atomic<u32> parked_count{0}; ///< parked_images.size(), read without the lock
    u64 create_ns = 0, create_max_ns = 0; ///< GOW3_TEXCACHE_STATS: new images (under mutex)
    bool ParkImage(ImageId image_id);
    ImageId TakeParkedImage(const ImageInfo& info);
    void EvictParkedImages(bool over_budget_only);
    std::mutex refresh_sites_mutex;
    std::unordered_map<u64, u64> unmapped_recently;
    u64 stats_reports = 0;
    std::mutex samplers_mutex;
    std::mutex download_images_mutex;
    struct MetaDataInfo {
        MetaType type;
        s32 clear_mask = -1;
    };
    tsl::robin_map<VAddr, MetaDataInfo> surface_metas;
};

} // namespace VideoCore
