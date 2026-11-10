// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "gow3_copy.h"
#include "gow3_resident_blocks.h"

#include <array>
#include <deque>
#include <source_location>
#include <boost/container/small_vector.hpp>

#include "common/interval_set.h"
#include "common/types.h"
#include "video_core/buffer_cache/buffer.h"
#include "video_core/buffer_cache/fault_manager.h"
#include "video_core/buffer_cache/range_set.h"
#include "video_core/renderer_vulkan/vk_semaphore.h"

namespace AmdGpu {
struct Liverpool;
}

namespace Core {
class MemoryManager;
}

namespace Vulkan {
class GraphicsPipeline;
struct SubmitInfo;
class Runtime;
class StagingBufferPool;
} // namespace Vulkan

namespace VideoCore {

class TextureCache;
class MemoryTracker;
class ResidencyPool;
class PageManager;
struct ArenaMappings;

class BufferCache {
    static constexpr u64 ADDRESS_SPACE_BITS = 40;
    // gow3: arena pages are 4 GiB unless the driver's buffer size limit is lower (AMD's
    // Windows driver: 2 GiB); an arena spans up to two pages, so they shrink to half the limit.
    static constexpr u64 MAX_ARENA_PAGE_BITS = 32;
    static constexpr u64 MIN_ARENA_PAGE_BITS = 30;
    static constexpr u64 MAX_ARENA_PAGES = u64{1} << (ADDRESS_SPACE_BITS - MIN_ARENA_PAGE_BITS);
    static constexpr u64 MIN_BLOCK_SIZE = 16_KB;

public:
    /// Read-only bindings up to this size are copied into a stream buffer (gow3: public for
    /// the draw pipeline's constant ring).
    static constexpr u64 STREAM_THRESHOLD = 16_KB;

    explicit BufferCache(const Vulkan::Instance& instance, Vulkan::Scheduler& scheduler,
                         Vulkan::Runtime& runtime, AmdGpu::Liverpool* liverpool,
                         TextureCache& texture_cache, PageManager& tracker);
    ~BufferCache();

    /// Returns a pointer to GDS device local buffer.
    [[nodiscard]] const Buffer* GetGdsBuffer() const noexcept {
        return &gds_buffer;
    }

    /// Retrieves the device local DBA page table buffer.
    [[nodiscard]] Buffer* GetBdaPageTableBuffer() noexcept {
        return bda_pagetable_buffer.get();
    }

    /// Retrieves the fault buffer.
    [[nodiscard]] Buffer* GetFaultBuffer() noexcept {
        return fault_manager->GetFaultBuffer();
    }

    /// Retrieves the stream buffer.
    StreamBuffer& GetStreamBuffer() noexcept {
        return stream_buffer;
    }

    /// Returns minimum granularity of a sparse memory bind.
    u32 GetSparsePageShift() const noexcept {
        return block_shift;
    }

    /// Invalidates any buffer in the logical page range.
    void InvalidateMemory(VAddr device_addr, u64 size, bool assume_locks = false);

    /// gow3: a guest write fault at `device_addr` was handled; unprotect its neighbourhood.
    void ExtendWriteFault(VAddr device_addr);

    /// Flushes any GPU modified buffer in the logical page range back to CPU memory.
    void ReadMemory(VAddr device_addr, u64 size, bool is_write = false, bool assume_locks = false);

    /// gow3: a guest libc write (memcpy / memset / memmove; src null: fill with value) into
    /// memory a shader wrote. Without it the write faults, the GPU's data in the page is read
    /// back and the whole GPU waited for. Here the bytes go into guest memory past the page
    /// protection and are queued for the GPU's copy (ProcessCpuWrites); the page stays
    /// GPU-modified, so the GPU's other data there is kept. Any thread.
    bool PreciseCpuWrite(VAddr addr, const void* src, int value, u64 size);
    /// Records the queued CPU writes into the GPU's copy: the recording thread, before a draw or
    /// dispatch is recorded, and before a download.
    void ProcessCpuWrites() {
        if (cpu_writes_pending.load(std::memory_order_acquire)) {
            std::scoped_lock lock{cpu_writes_mutex};
            ProcessCpuWritesLocked();
        }
    }

    /// gow3 GOW3_PERF_DIAG: prints and clears the write-fault download sites.
    static void ReportDownloadSites(double seconds);
    /// gow3 GOW3_FRAME_STATS: the GPU write twins' count and bytes, every stats window.
    static void ReportTwins();

    /// Finds a buffer for the specified region.
    [[nodiscard]] std::pair<const Buffer*, u64> ObtainBuffer(
        VAddr device_addr, u32 size, bool is_written, bool is_texel_buffer = false,
        std::source_location caller = std::source_location::current());

    /// gow3 GOW3_PERF_DIAG: the shader of a written buffer binding just obtained (BindBuffers), for
    /// the write-fault download report (who made a page GPU-written).
    static void NoteGpuWriterShader(VAddr device_addr, u64 size, u64 shader_hash);

    /// Attempts to obtain a buffer without modifying the cache contents.
    [[nodiscard]] std::pair<const Buffer*, u64> ObtainBufferForImage(VAddr device_addr, u32 size);

    /// Return true when a region is modified from the CPU
    [[nodiscard]] bool IsRegionCpuModified(VAddr addr, size_t size);

    /// Return true when a region is modified from the GPU
    [[nodiscard]] bool IsRegionGpuModified(VAddr addr, size_t size);

    /// Processes the fault buffer.
    void ProcessFaultBuffer();

    /// Synchronizes all buffers needed for DMA.
    void SynchronizeDmaBuffers();

    /// Commits pending sparse buffer memory binds. Must be called before every scheduler submit.
    void SubmitPendingArenaBinds(Vulkan::SubmitInfo& info);

    /// gow3 U2: a draw or dispatch reading memory through the page table (DMA) was recorded: it
    /// may read any resident block, so no direct arena write until the GPU is past it.
    void NoteDmaUse();

private:
    struct ArenaBinds {
        const Buffer* arena;
        boost::container::small_vector<vk::SparseMemoryBind, 32> binds;
    };

    ArenaBinds* BindsForArena(const Buffer* arena) {
        auto it = std::ranges::find(pending_binds, arena, &ArenaBinds::arena);
        if (it != pending_binds.end()) {
            return std::addressof(*it);
        }
        return &pending_binds.emplace_back(arena);
    }

    const Buffer* GetArena(u64 first_block, u64 last_block);

    void EnsureResident(const Buffer* arena, u64 first_block, u64 last_block);

    void DownloadMemory(const Buffer* arena, VAddr device_addr, u64 size);

    bool SynchronizeMemory(const Buffer* arena, VAddr device_addr, u32 size, bool is_written,
                           bool is_texel_buffer);

    /// gow3: batched small copies on the copy threads (Gow3Copy::QueueCopy).
    static void RunGuestCopy(const Gow3Copy::Item& item);
    void SmallGuestCopy(const Gow3Copy::Item& item);

    const Buffer* UploadCopies(const Buffer* arena, std::span<vk::BufferCopy> copies,
                               size_t total_size_bytes);

    /// gow3 U2 (GOW3_ARENA_HOST_VISIBLE): direct arena writes. The reason [addr, addr + size)
    /// cannot be written through the arena mapping now (Gow3Stats::ArenaFallback), or
    /// DirectWriteOk.
    static constexpr unsigned DirectWriteOk = ~0u;
    unsigned DirectWriteBlocked(VAddr addr, u64 size);
    /// Calls func(guest address, host address, size) over [addr, addr + size) in pieces
    /// contiguous in the mapping (DirectWriteBlocked returned DirectWriteOk).
    template <typename Func>
    void ForEachDirectPiece(VAddr addr, u64 size, Func&& func);
    /// Takes the upload copies (UploadCopies' input: srcOffset into the staging, dstOffset the
    /// guest address) that can be written directly out of `copies`, starts their guest copies
    /// into the arena mapping, and repacks the rest for the staging path; returns its size.
    size_t UploadDirect(boost::container::small_vector<vk::BufferCopy, 4>& copies,
                        size_t total_size_bytes);
    static void RunDirectCopy(const Gow3Copy::Item& item);
    /// The GPU uses [addr, addr + size) in the command buffer being recorded (`staged`: a copy
    /// from staging writes it).
    void NoteArenaUse(VAddr addr, u64 size, bool staged = false);

    bool SynchronizeMemoryFromImage(const Buffer* arena, VAddr device_addr, u32 size);

    const Vulkan::Instance& instance;
    Vulkan::Scheduler& scheduler;
    // gow3: texel buffer ranges known not to alias an image, per image registry generation.
    struct ImageMiss {
        VAddr address = 0;
        u32 size = 0;
        u64 generation = ~0ULL;
    };
    std::array<ImageMiss, 256> image_miss_cache{};
    Vulkan::Runtime& runtime;
    Vulkan::StagingBufferPool& staging_pool;
    AmdGpu::Liverpool* liverpool;
    Core::MemoryManager* memory;
    TextureCache& texture_cache;
    std::unique_ptr<MemoryTracker> memory_tracker;

    StreamBuffer stream_buffer;
    /// gow3 S2 (GOW3_STREAM_CLEAN=1): small read-only ranges uploaded into the arena once
    /// (value 1), or found CPU-written again after that (2: dynamic, always streamed).
    std::mutex stream_class_mutex;
    /// gow3: small read-only buffer states by address (0 new, 1 uploaded once, 2 dynamic), in a
    /// fixed direct-mapped table. It was an unordered_map cleared at 262,144 entries: the
    /// clear (and each rehash while it grew) took ~8-12 ms on stage B, about once a second
    /// while panning over new content (t-pan2: "obt_regions"). An evicted address starts again
    /// as new: at worst one more upload of a small buffer outside a render pass.
    struct StreamClass {
        VAddr address;
        u8 state;
    };
    std::array<StreamClass, 1u << 16> stream_class{};
    Buffer gds_buffer;
    RangeSet gpu_modified_ranges;
    void ProcessCpuWritesLocked();
    /// gow3: a download's data into guest memory, keeping guest bytes written since their twin.
    void WriteBackDownload(VAddr arena_base, std::span<const vk::BufferCopy> copies, const u8* data,
                           u64 data_offset);
    /// gow3: guest libc writes into shader-written memory, waiting for the recording thread.
    std::mutex cpu_writes_mutex;
    std::vector<std::pair<VAddr, std::vector<u8>>> cpu_writes;
    std::atomic<bool> cpu_writes_pending{false};

    std::unique_ptr<FaultManager> fault_manager;
    std::unique_ptr<Buffer> bda_pagetable_buffer;

    std::array<const Buffer*, MAX_ARENA_PAGES> address_space{};
    u64 arena_page_bits{MAX_ARENA_PAGE_BITS};
    u64 arena_page_size{u64{1} << MAX_ARENA_PAGE_BITS};
    u64 num_arena_pages{u64{1} << (ADDRESS_SPACE_BITS - MAX_ARENA_PAGE_BITS)};
    std::deque<Buffer> arenas;
    std::vector<ArenaBinds> pending_binds;
    Vulkan::Semaphore memory_semaphore;

    struct Backing : public Interval {
        vk::DeviceMemory memory;
        u64 offset;
        constexpr bool CanMergeWith(const Backing& other) const noexcept {
            return memory == other.memory && offset + (end - start) == other.offset;
        }
        constexpr Backing SubRange(u64 a, u64 b) const noexcept {
            return {{a, b}, memory, offset + (a - start)};
        }
    };
    IntervalList<Backing> resident_ranges;
    /// gow3 R2 (impl-stage-b-per-draw-2.md): the blocks of resident_ranges as a bitmap, set with
    /// every range added (whatever the toggle), for EnsureResident's "already resident" answer.
    Gow3Resident::Blocks resident_bitmap;
    /// gow3: device memory for resident blocks, allocated ahead in chunks (buffer_cache.cpp).
    std::unique_ptr<ResidencyPool> residency_pool;
    /// gow3 U2: host addresses of the resident blocks when the arena memory is mapped (null:
    /// GOW3_ARENA_HOST_VISIBLE off or no suitable memory type; direct writes unavailable).
    std::unique_ptr<ArenaMappings> arena_mappings;
    /// gow3 U2: the newest tick a DMA draw or dispatch was recorded at (NoteDmaUse).
    std::atomic<u64> dma_use_tick{0};

    u32 arena_memory_type_index{};
    u32 block_size{};
    u32 block_shift{};
    u32 blocks_per_arena_page{};
    u32 blocks_per_arena_page_shift{};
};

} // namespace VideoCore
