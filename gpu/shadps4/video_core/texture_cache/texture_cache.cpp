// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <format>
#include <limits>
#include <mutex>
#include <unordered_set>
#include <xxhash.h>
#include "video_core/amdgpu/liverpool.h"

#include "gow3_sections.h"
#include "gow3_toggles.h"
#include "common/assert.h"
#include "common/debug.h"
#include "common/div_ceil.h"
#include "common/hash.h"
#include "common/scope_exit.h"
#include "core/emulator_settings.h"
#include "core/memory.h"
#include "video_core/buffer_cache/buffer_cache.h"
#include "video_core/page_manager.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_runtime.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_wait_diagnostics.h"
#include "video_core/texture_cache/aliasing.h"
#include "video_core/texture_cache/host_compatibility.h"
#include "video_core/texture_cache/texture_cache.h"
#include <unordered_map>
#include "video_core/texture_cache/tile_manager.h"

namespace VideoCore {

namespace {
// gow3 (plan-linux-deck.md "Mausoleum lighting"): a color target or storage image of at most 64
// texels (and 64 KiB) is copied back to guest memory whatever its tiling (the rule before:
// linear or at most 8 wide). UE4's sky light lighting is a 9x1 RGBA16F tiled target the CPU reads
// back once at a level's load: never copied back, the CPU took the bytes left there (zeros on the
// PC by luck, leftovers on the Deck: the Hilltop Mausoleum lit in wrong colours).
bool IsTinyTarget(const ImageInfo& info) {
    return !info.props.is_depth && info.guest_address != 0 && info.guest_size <= 64_KB &&
           u64{info.size.width} * info.size.height * info.size.depth * info.resources.layers <= 64;
}

float HalfToFloat(u16 h) {
    const u32 sign = (h >> 15) & 1, exponent = (h >> 10) & 31, mantissa = h & 1023;
    float value;
    if (exponent == 0) {
        value = std::ldexp(float(mantissa), -24);
    } else if (exponent == 31) {
        value = mantissa ? std::numeric_limits<float>::quiet_NaN()
                         : std::numeric_limits<float>::infinity();
    } else {
        value = std::ldexp(float(mantissa | 1024), int(exponent) - 25);
    }
    return sign ? -value : value;
}

/// GOW3_READBACK_LOG: up to `max` 8-byte groups of `data` that are not zero, as offset and four
/// half floats (where a tiny RGBA16F target's texels landed in its guest copy).
std::string NonZeroTexels(const u8* data, u64 size, u32 max) {
    std::string out;
    u32 found = 0;
    for (u64 offset = 0; offset + 8 <= size && found < max; offset += 8) {
        u16 h[4];
        std::memcpy(h, data + offset, 8);
        if ((h[0] | h[1] | h[2] | h[3]) == 0) {
            continue;
        }
        out += std::format(" +{}:({:.5g} {:.5g} {:.5g} {:.5g})", offset, HalfToFloat(h[0]),
                           HalfToFloat(h[1]), HalfToFloat(h[2]), HalfToFloat(h[3]));
        ++found;
    }
    return out.empty() ? std::string(" all zero") : out;
}
} // namespace

namespace {
// gow3: GOW3_TEXCACHE_STATS=1 (diagnostics): what the cache does per 5 s, printed after a
// submission's collector run. MediEvil lost FPS the longer a level was played and a restart
// restored it: these show which part grows. Off: one load per counter.
struct CacheStats {
    std::atomic<u64> fast_lookups, slow_lookups, slow_candidates, overlaps, expands, registered,
        unregistered, refreshes, refresh_bytes, alias_copies, alias_syncs, volume_views, unmap_frees,
        unmap_free_bytes, recreated, recreated_bytes, eop_waits, eop_skips, undefined_targets,
        undefined_target_bytes, cleared_targets, cleared_target_bytes, gpu_freed_reused,
        gpu_freed_reused_bytes, parked, parked_reused, parked_evicted;
};
CacheStats cache_stats;
bool CacheStatsEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("GOW3_TEXCACHE_STATS");
        return value && value[0] == '1';
    }();
    return enabled;
}
#define CACHE_STAT(name, n)                                                                       \
    do {                                                                                           \
        if (CacheStatsEnabled()) cache_stats.name.fetch_add((n), std::memory_order_relaxed);      \
    } while (0)

// gow3: GOW3_TEXCACHE_STATS: a render or depth target at the start of a mip of an image with
// mips, once per kind: whether it resolved as a view of that mip. A rejected one renders into
// an image of its own and the parent's mip keeps what it held (UE4's HZB is built mip by mip
// as render targets; small mips are micro tiled, MipOf compared array modes: t-hzb1).
// merge: an image of its own met by its grown parent, merged into the mip or kept apart.
void NoteMipTarget(const ImageInfo& target, const ImageInfo& parent, s32 mip, s32 slice,
                   bool merge) {
    if (!CacheStatsEnabled() || target.resources.levels != 1) {
        return;
    }
    s32 at = -1;
    for (u32 m = 1; m < parent.resources.levels; ++m) {
        if (target.guest_address == parent.guest_address + parent.mips_layout[m].offset) {
            at = s32(m);
            break;
        }
    }
    // The next mip, before the image grows to it (a mip's render target comes before its view).
    const bool next = at < 0 && !merge &&
                      target.guest_address == parent.guest_address + parent.guest_size;
    if (at < 0 && !next) {
        return;
    }
    static std::mutex mutex;
    static std::unordered_set<u64> seen;
    const u64 key = u64(parent.pixel_format) << 40 ^ u64(parent.size.width) << 26 ^
                    u64(parent.size.height) << 12 ^ u64(next ? parent.resources.levels : at) << 6 ^
                    u64(mip + 1) << 3 ^ u64(slice + 1) ^ u64(merge) << 63 ^ u64(next) << 62;
    std::scoped_lock lock{mutex};
    if (seen.size() >= 256 || !seen.insert(key).second) {
        return;
    }
    const bool resolved = mip >= 0 && slice >= 0;
    std::printf("Mip target%s: %ux%u %s L%u at 0x%llx, mip %d%s: target %ux%u pitch %u tile mode %u "
                "array mode %u, parent tile mode %u array mode %u: %s\n",
                merge ? " (merge)" : "", parent.size.width, parent.size.height,
                vk::to_string(parent.pixel_format).c_str(), parent.resources.levels,
                static_cast<unsigned long long>(parent.guest_address),
                next ? s32(parent.resources.levels) : at, next ? " (past the image's levels)" : "",
                target.size.width, target.size.height, target.pitch, u32(target.tile_mode),
                u32(target.array_mode), u32(parent.tile_mode), u32(parent.array_mode),
                merge ? (resolved ? "merged into the mip" : "kept apart")
                      : (resolved ? "view of the mip" : "rejected, own image"));
    std::fflush(stdout);
}

// gow3: a volume the guest renders slice by slice and samples as 3D: a 2D render target with
// one layer per depth slice at the address of a 3D image of the same size. 3D images are
// created 2D-array compatible, so it renders through a 2D array view of the volume.
// gow3: GOW3_TEXCACHE_STATS: identifies an image across a free and a later re-creation.
u64 RecreateKey(const ImageInfo& info) {
    u64 key = info.guest_address ^ u64(info.guest_size) << 20 ^ u64(info.pixel_format) << 44 ^
              u64(info.size.width) << 8 ^ u64(info.size.height) << 30;
    key = (key ^ key >> 33) * 0xFF51AFD7ED558CCDull;
    return key ^ key >> 33;
}

bool IsVolumeSliceTarget(const ImageInfo& info, TextureCache::BindingType binding,
                         const ImageInfo& volume) {
    return binding == TextureCache::BindingType::RenderTarget &&
           volume.type == AmdGpu::ImageType::Color3D && info.type != AmdGpu::ImageType::Color3D &&
           info.size.width == volume.size.width && info.size.height == volume.size.height &&
           info.resources.layers == volume.size.depth && info.resources.levels == 1 &&
           volume.resources.levels == 1 &&
           IsVulkanFormatCompatible(volume.pixel_format, info.pixel_format) &&
           !Gow3Toggle::Disabled(Gow3Toggle::VolumeSliceViews);
}
} // namespace

static constexpr u64 PageShift = 12;
static constexpr u64 NumFramesBeforeRemoval = 32;

TextureCache::TextureCache(const Vulkan::Instance& instance_, Vulkan::Scheduler& scheduler_,
                           Vulkan::Runtime& runtime_, AmdGpu::Liverpool* liverpool_,
                           BufferCache& buffer_cache_, PageManager& tracker_)
    : instance{instance_}, scheduler{scheduler_}, runtime{runtime_}, liverpool{liverpool_},
      buffer_cache{buffer_cache_}, tracker{tracker_}, blit_helper{instance, scheduler},
      tile_manager{instance, scheduler, runtime, buffer_cache.GetStreamBuffer()},
      readback_linear_images{EmulatorSettings.IsReadbackLinearImagesEnabled()} {

    std::printf("GPU: deferred image readback %s (64 copies, 64 MiB pending limit)%s\n",
                deferred_readbacks ? "enabled" : "disabled",
                stale_readbacks ? "; CPU reads use the previous frame's copy" : "");

    u32 max_samplers = instance.GetMaxSamplerAllocationCount();
    trigger_gc_samplers = max_samplers * 3 / 4;
    pressure_gc_samplers = max_samplers * 7 / 8;
    critical_gc_samplers = max_samplers * 15 / 16;

    // Set up garbage collection parameters.
    if (!instance.CanReportMemoryUsage()) {
        trigger_gc_memory = 0;
        pressure_gc_memory = DEFAULT_PRESSURE_GC_MEMORY;
        critical_gc_memory = DEFAULT_CRITICAL_GC_MEMORY;
        return;
    }

    // GOW3_GC_TOTAL_MB=N: tests the startup rule as on a GPU with N MiB (e.g. 8192: a laptop).
    const char* total_env = std::getenv("GOW3_GC_TOTAL_MB");
    const s64 device_local_memory = total_env ? static_cast<s64>(std::strtoll(total_env, nullptr, 10)) << 20
                                              : static_cast<s64>(instance.GetTotalMemoryBudget());
    const s64 min_spacing_expected = device_local_memory - 1_GB;
    const s64 min_spacing_critical = device_local_memory - 512_MB;
    const s64 mem_threshold = std::min<s64>(device_local_memory, TARGET_GC_THRESHOLD);
    const s64 min_vacancy_expected = (6 * mem_threshold) / 10;
    const s64 min_vacancy_critical = (2 * mem_threshold) / 10;
    pressure_gc_memory = static_cast<u64>(
        std::max<u64>(std::min(device_local_memory - min_vacancy_expected, min_spacing_expected),
                      DEFAULT_PRESSURE_GC_MEMORY));
    critical_gc_memory = static_cast<u64>(
        std::max<u64>(std::min(device_local_memory - min_vacancy_critical, min_spacing_critical),
                      DEFAULT_CRITICAL_GC_MEMORY));
    trigger_gc_memory = static_cast<u64>((device_local_memory - mem_threshold) / 2);
}

TextureCache::~TextureCache() {
    if (deferred_readbacks) {
        scheduler.Finish();
        std::scoped_lock lock{readback_mutex};
        RetireReadbacks();
    }
}

template <bool track>
void TextureCache::WatchReadback(VAddr address, u64 size) {
    if (stale_readbacks) {
        tracker.UpdatePageWatchers<track, false>(address, size);
    } else {
        tracker.UpdatePageWatchers<track, true>(address, size);
    }
}

template <typename Entry>
void TextureCache::UnwatchReadback(const Entry& entry) {
    if (entry.reads) {
        tracker.UpdatePageWatchers<false, true>(entry.address, entry.size);
    } else {
        WatchReadback<false>(entry.address, entry.size);
    }
}

void TextureCache::PublishReadbackSpans() {
    std::array<std::pair<u64, u64>, ReadbackSpanCount> spans{};
    const std::size_t count = pending_readbacks.Spans(spans);
    auto& p = readback_spans;
    const u32 seq = p.seq.load(std::memory_order_relaxed);
    p.seq.store(seq + 1, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_release);
    for (std::size_t i = 0; i < count; ++i) {
        p.begin[i].store(spans[i].first, std::memory_order_relaxed);
        p.end[i].store(spans[i].second, std::memory_order_relaxed);
    }
    p.count.store(u32(count), std::memory_order_relaxed);
    p.seq.store(seq + 2, std::memory_order_release);
}

bool TextureCache::ReadbackSpansMiss(VAddr address, u64 size) const {
    // A seqlock read: a writer in progress or a changed seq sends the caller to the lock.
    const auto& p = readback_spans;
    const u32 seq = p.seq.load(std::memory_order_acquire);
    if (seq & 1) {
        return false;
    }
    const u32 count = std::min<u32>(p.count.load(std::memory_order_relaxed), ReadbackSpanCount);
    bool overlap = false;
    for (u32 i = 0; i < count; ++i) {
        overlap |= address < p.end[i].load(std::memory_order_relaxed) &&
                   p.begin[i].load(std::memory_order_relaxed) < address + size;
    }
    std::atomic_thread_fence(std::memory_order_acquire);
    return !overlap && p.seq.load(std::memory_order_relaxed) == seq;
}

void TextureCache::RetireReadbacks(bool polled) {
    const std::size_t queued = pending_readbacks.Size();
    // gow3 (study-stage-b-per-draw.md C1): `polled` (the end-of-pipe events): a copy the last
    // known GPU tick does not cover yet retires at a later event (Scheduler::IsFreePolled).
    pending_readbacks.Retire(
        [this, polled](u64 tick) { return polled ? scheduler.IsFreePolled(tick) : scheduler.IsFree(tick); },
        [this](auto& entry) {
            if (entry.valid) {
                entry.payload.Invalidate();
                Core::Memory::Instance()->TryWriteBacking(std::bit_cast<u8*>(entry.address),
                    entry.payload.mapped, entry.size);
                if (entry.watched) UnwatchReadback(entry);
                if (entry.seq) NoteTrackedPublish(entry.address, entry.size, entry.seq);
                if (stale_readbacks && readback_published.insert(entry.address).second &&
                    readback_log) {
                    std::printf("Readback first copy published: %#llx (%llu bytes); non-zero 8-byte "
                                "groups:%s\n",
                                static_cast<unsigned long long>(entry.address),
                                static_cast<unsigned long long>(entry.size),
                                NonZeroTexels(entry.payload.mapped, entry.size, 16).c_str());
                }
                ++readbacks_completed;
            }
            runtime.GetStagingPool().FreeDeferred(entry.payload);
        });
    if (pending_readbacks.Size() != queued) {
        PublishReadbackSpans(); // fewer pages: tighter spans (optional, never required)
    }
    has_pending_readbacks.store(pending_readbacks.Bytes() != 0, std::memory_order_release);
}

bool TextureCache::ResolveReadbacks(VAddr address, u64 size, bool assume_locks,
                                    ReadbackSource source) {
    if (!deferred_readbacks || !has_pending_readbacks.load(std::memory_order_acquire)) return false;
    // gow3 C5 (impl-stage-b-bundle-2.md; study-stage-b-per-draw.md): the buffer cache asks 3-4
    // times a draw, and with copies always queued each ask took readback_mutex and scanned the
    // queue. A range on no page of a pending copy needs no lock: the answer would be "nothing
    // to wait for". Only the buffer cache's calls (the thread that owns the caches, the same
    // one that queues the copies, or one ordered after it by the draw-pipe drain) skip the
    // lock; the CPU fault paths keep it: a page is watched before its copy is queued, so a
    // fault may come before the spans include the copy (the lock waits for that).
    // Second toggle word 128: always the lock.
    if (source == ReadbackSource::BufferObtain || source == ReadbackSource::BufferForImage ||
        source == ReadbackSource::BufferUpload) {
        if (!Gow3Toggle2::Disabled(Gow3Toggle2::ReadbackPrecheck) &&
            ReadbackSpansMiss(address, size)) {
            if (Gow3Stats::enabled) {
                Gow3Stats::readback_prechecks[0].fetch_add(1, std::memory_order_relaxed);
            }
            return false;
        }
        if (Gow3Stats::enabled) {
            Gow3Stats::readback_prechecks[1].fetch_add(1, std::memory_order_relaxed);
        }
    }
    {
        std::scoped_lock lock{readback_mutex};
        // Stale mode, byte rule: God of War III writes every frame to the page that holds a
        // 1x1 exposure copy, but not to its bytes. That write need not wait for the GPU: the
        // copy only stops protecting its page and is still published when the GPU is done.
        if (stale_readbacks && source == ReadbackSource::CpuWrite) {
            readbacks_released += pending_readbacks.ReleaseOutside(address, size, 64_KB,
                [this](auto& entry) { UnwatchReadback(entry); });
        }
        if (!pending_readbacks.RequiredTick(address, size)) return false;
    }
    const auto resolve = [this, address, size, source] {
        const bool diag = Vulkan::WaitDiagnostics::Enabled();
        std::scoped_lock lock{readback_mutex};
        bool waited = false;
        u64 wait_ns = 0;
        if (diag && (source == ReadbackSource::CpuRead || source == ReadbackSource::CpuWrite)) {
            // Which copied image the CPU touched while its copy was pending.
            for (auto& image : readback_images) {
                if (image.copies && address < image.address + image.Bytes() && image.address < address + size) {
                    ++image.cpu_accesses;
                }
            }
        }
        Gow3Stats::Timer readback_timer{Gow3Stats::t_readback_wait};
        if (const u64 tick = pending_readbacks.RequiredTick(address, size)) {
            ++readbacks_waited;
            waited = !scheduler.IsFree(tick);
            if (readback_log && source == ReadbackSource::CpuRead) {
                // gow3: in stale mode only a first copy's pages watch reads (GOW3_READBACK_FIRST_WAIT).
                std::printf("Readback first copy read by the CPU: %#llx (%llu bytes), GPU %s\n",
                            static_cast<unsigned long long>(address),
                            static_cast<unsigned long long>(size),
                            waited ? "not done yet: waiting" : "already done");
            }
            const auto start = diag ? std::chrono::steady_clock::now()
                                    : std::chrono::steady_clock::time_point{};
            scheduler.Wait(tick);
            if (diag) {
                wait_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                              std::chrono::steady_clock::now() - start).count();
            }
            RetireReadbacks();
        }
        if (diag) readback_sources.Record(source, waited, wait_ns, size);
    };
    if (assume_locks) resolve();
    else liverpool->SendCommand<true>(resolve);
    return true;
}

u64 TextureCache::NoteTrackedPush(VAddr address, u64 size) {
    for (auto& t : tracked_readbacks) {
        if (t.size && address < t.address + t.size && t.address < address + size) return ++t.pushed;
    }
    return 0;
}

void TextureCache::NoteTrackedPublish(VAddr address, u64 size, u64 seq) {
    for (auto& t : tracked_readbacks) {
        if (t.size && address < t.address + t.size && t.address < address + size) {
            t.published = std::max(t.published, seq);
            return;
        }
    }
}

TextureCache::ReadbackSnapshot TextureCache::SnapshotReadback(VAddr address, u64 size, void* dst,
                                                              u64 copy_size, bool restart) {
    if (!deferred_readbacks || !size) return {};
    std::scoped_lock lock{readback_mutex};
    auto it = std::ranges::find_if(tracked_readbacks, [&](const TrackedReadback& t) {
        return t.address == address && t.size == size;
    });
    if (it == tracked_readbacks.end()) {
        // Copies queued before this call carry no number: their bytes count as copy 0.
        it = std::ranges::find(tracked_readbacks, u64{0}, &TrackedReadback::size);
        if (it == tracked_readbacks.end()) return {};
        *it = {address, size, 0, 0};
    }
    RetireReadbacks();
    if (restart) {
        *it = {address, size, 0, 0};
        pending_readbacks.ClearSeq(address, size);
    }
    if (copy_size && !Core::Memory::Instance()->ReadBacking(address, dst, std::min(copy_size, size))) {
        return {};
    }
    return {it->pushed, it->published, true};
}

void TextureCache::CancelReadbacks(VAddr address, u64 size) {
    if (!deferred_readbacks) return;
    std::scoped_lock lock{readback_mutex};
    const u64 canceled_before = readbacks_canceled;
    pending_readbacks.Cancel(address, size, [this](auto& entry) {
        if (entry.watched) UnwatchReadback(entry);
        ++readbacks_canceled;
    });
    if (readbacks_canceled != canceled_before) {
        PublishReadbackSpans(); // C5: tighter spans (optional)
    }
}

// gow3: every copy is recorded first and the GPU is waited for once. A wait per image (as
// before) left the GPU mostly idle in God of War III's gameplay, at about 2 FPS.
void TextureCache::ProcessDownloadImages() {
    GOW3_SECTION(EndOfPipe);
    if (deferred_readbacks) {
        // gow3: most ends of pipe (MediEvil: ~3 a frame, ~5 tiny readbacks a frame) have no image
        // to download and no copy ready to publish; the host copy wait below then protects
        // nothing, yet it was ~20% of the draw recording thread in heavy views (P2a). Those skip
        // it; every end of pipe with readback work still waits (skipping those broke the torches,
        // R1 / V2). Opt-in (GOW3_IDLE_EOP_SKIP=1): +2% in heavy views, but the copies the wait
        // spread over the frame then land at once and the spikes when a torch comes into view
        // got deeper (T-P2: worst frames 29-42 ms vs <= 19 ms). Toggle bit 60 turns it off live.
        static const bool idle_eop_skip = [] {
            const char* env = std::getenv("GOW3_IDLE_EOP_SKIP");
            return env && env[0] == '1';
        }();
        if (idle_eop_skip && !Gow3Toggle::Disabled(Gow3Toggle::IdleEopSkip)) {
            bool idle;
            {
                std::scoped_lock readback_lock{readback_mutex};
                std::unique_lock lock{download_images_mutex};
                idle = download_images.empty() &&
                       !pending_readbacks.FrontReady([this](u64 tick) { return scheduler.IsFreePolled(tick); });
            }
            if (idle) {
                CACHE_STAT(eop_skips, 1);
                return;
            }
        }
        CACHE_STAT(eop_waits, 1);
        // Upload workers must finish reading RAM before these pages become inaccessible.
        // gow3 W2 (on by default since T-E1; GOW3_EOP_TARGETED=0 or toggle bit 60 live: off): only the copies that
        // read the memory this end of pipe writes or protects: the queued readbacks (retired
        // into guest memory) and the images about to be downloaded (watched from now on).
        // Usually none: the full wait was ~21% of the draw recording thread (T-P3). The ranges
        // are collected under the locks and waited for without them: a copy thread faulting on
        // a watched page resolves readbacks under readback_mutex.
        static const bool targeted = [] {
            const char* env = std::getenv("GOW3_EOP_TARGETED");
            return !env || env[0] != '0';
        }();
        if (targeted && !Gow3Toggle::Disabled(Gow3Toggle::IdleEopSkip)) {
            boost::container::small_vector<std::pair<VAddr, u64>, 16> ranges;
            {
                std::scoped_lock readback_lock{readback_mutex};
                std::unique_lock lock{download_images_mutex};
                pending_readbacks.ForEach(
                    [&](u64 address, u64 size) { ranges.emplace_back(address, size); });
                for (const ImageId image_id : download_images) {
                    const auto& info = slot_images[image_id].info;
                    ranges.emplace_back(info.guest_address, info.guest_size);
                }
            }
            for (const auto& [address, size] : ranges) {
                scheduler.WaitHostCopiesFor(address, size);
            }
        } else {
            scheduler.WaitHostCopies();
        }
        std::scoped_lock readback_lock{readback_mutex};
        RetireReadbacks(true); // the end-of-pipe poll (C1)
        std::unique_lock lock{download_images_mutex};
        for (const ImageId image_id : download_images) {
            const auto& image = slot_images[image_id];
            if (False(image.flags & ImageFlagBits::GpuModified)) continue;
            const u64 size = u64{image.info.pitch} * image.info.size.height *
                image.info.size.depth * image.info.resources.layers * (image.info.num_bits / 8);
            if (!pending_readbacks.CanFit(size)) {
                ++readbacks_waited;
                scheduler.Finish();
                RetireReadbacks();
            }
            if (auto copy = RecordImageDownload(image_id)) {
                if (!pending_readbacks.CanFit(copy->size)) {
                    scheduler.Finish();
                    copy->staging.Invalidate();
                    Core::Memory::Instance()->TryWriteBacking(
                        std::bit_cast<u8*>(copy->guest_address), copy->staging.mapped, copy->size);
                    if (const u64 seq = NoteTrackedPush(copy->guest_address, copy->size)) {
                        NoteTrackedPublish(copy->guest_address, copy->size, seq);
                    }
                    readback_published.insert(copy->guest_address);
                    runtime.GetStagingPool().FreeDeferred(copy->staging);
                } else {
                    // gow3: a first copy (nothing published at this range yet) in stale mode:
                    // logged, and with GOW3_READBACK_FIRST_WAIT its pages watch CPU reads too.
                    const bool first = stale_readbacks &&
                                       !readback_published.contains(copy->guest_address);
                    const bool reads = first && readback_first_wait;
                    if (reads) {
                        tracker.UpdatePageWatchers<true, true>(copy->guest_address, copy->size);
                    } else {
                        WatchReadback<true>(copy->guest_address, copy->size);
                    }
                    if (first && readback_log) {
                        std::printf("Readback first copy queued: %#llx (%llu bytes), image %ux%u "
                                    "%u bits %s, tick %llu; CPU reads %s\n",
                                    static_cast<unsigned long long>(copy->guest_address),
                                    static_cast<unsigned long long>(copy->size),
                                    image.info.size.width, image.info.size.height,
                                    image.info.num_bits,
                                    vk::to_string(image.info.pixel_format).c_str(),
                                    static_cast<unsigned long long>(scheduler.CurrentTick()),
                                    reads ? "wait for it" : "see the old bytes (stale mode)");
                    }
                    if (Vulkan::WaitDiagnostics::Enabled()) {
                        // Which images go back to RAM each frame (the report lists the busiest).
                        auto* slot = std::ranges::find(readback_images, image.info.guest_address,
                                                       &ReadbackImageStat::address);
                        if (slot == readback_images.end()) {
                            slot = std::ranges::find(readback_images, VAddr{0}, &ReadbackImageStat::address);
                        }
                        if (slot != readback_images.end()) {
                            *slot = {image.info.guest_address, image.info.size.width,
                                     image.info.size.height, image.info.num_bits, slot->copies + 1,
                                     slot->cpu_accesses, image.info.pixel_format};
                        }
                    }
                    pending_readbacks.Push(copy->guest_address, copy->size,
                        scheduler.CurrentTick(), copy->staging,
                        NoteTrackedPush(copy->guest_address, copy->size), reads);
                    PublishReadbackSpans(); // C5: before any pre-check may skip the lock
                    has_pending_readbacks.store(true, std::memory_order_release);
                    ++readbacks_queued;
                    readback_submit_wanted = !stale_readbacks;
                    readbacks_peak_bytes = std::max(readbacks_peak_bytes, pending_readbacks.Bytes());
                }
            }
        }
        download_images.clear();
        if (Vulkan::WaitDiagnostics::Enabled()) {
            const auto now = std::chrono::steady_clock::now();
            if (now - readback_report >= std::chrono::seconds(5)) {
                std::printf("Readback profile: queued %llu, completed %llu, demand/budget waits %llu, canceled %llu, pending %.2f MiB, peak %.2f MiB\n",
                    static_cast<unsigned long long>(readbacks_queued),
                    static_cast<unsigned long long>(readbacks_completed),
                    static_cast<unsigned long long>(readbacks_waited),
                    static_cast<unsigned long long>(readbacks_canceled),
                    pending_readbacks.Bytes() / 1048576.0, readbacks_peak_bytes / 1048576.0);
                readbacks_queued = readbacks_completed = readbacks_waited = readbacks_canceled = 0;
                readbacks_peak_bytes = pending_readbacks.Bytes();
                readback_sources.Report(stdout);
                if (stale_readbacks) {
                    std::printf("Readback byte rule: %llu CPU writes went on without waiting\n",
                                static_cast<unsigned long long>(readbacks_released));
                    readbacks_released = 0;
                }
                std::ranges::sort(readback_images, std::greater{}, &ReadbackImageStat::copies);
                for (u32 i = 0; i < 4 && readback_images[i].copies; ++i) {
                    const auto& r = readback_images[i];
                    std::printf("Readback image %#llx: %ux%u, %u bits, %s, %u copies, %u CPU accesses "
                                "while pending\n",
                                static_cast<unsigned long long>(r.address), r.width, r.height, r.bits,
                                vk::to_string(r.format).c_str(), r.copies, r.cpu_accesses);
                }
                readback_images = {};
                readback_report = now;
            }
        }
        return;
    }
    std::unique_lock lk{download_images_mutex};
    boost::container::small_vector<PendingDownload, 16> pending;
    for (const ImageId image_id : download_images) {
        if (auto download = RecordImageDownload(image_id)) {
            pending.push_back(*download);
        }
    }
    download_images.clear();
    if (pending.empty()) {
        return;
    }
    scheduler.Finish();
    for (const PendingDownload& download : pending) {
        download.staging.Invalidate();
        Core::Memory::Instance()->TryWriteBacking(std::bit_cast<u8*>(download.guest_address),
                                                  download.staging.mapped, download.size);
        runtime.GetStagingPool().FreeDeferred(download.staging);
    }
}

std::optional<TextureCache::PendingDownload> TextureCache::RecordImageDownload(ImageId image_id) {
    Image& image = slot_images[image_id];
    if (False(image.flags & ImageFlagBits::GpuModified)) {
        return std::nullopt;
    }
    if (image.info.props.is_tiled && tiny_target_readback && IsTinyTarget(image.info)) {
        // gow3: in the guest's tiled layout (the tiling shader writes the whole guest range,
        // every mip), so a CPU read of the target finds each texel where the game looks for it.
        boost::container::small_vector<vk::BufferImageCopy, 16> copies;
        for (u32 mip = 0; mip < image.info.resources.levels; ++mip) {
            const auto& mip_info = image.info.mips_layout[mip];
            copies.push_back(vk::BufferImageCopy{
                .bufferOffset = mip_info.offset,
                .bufferRowLength = mip_info.pitch,
                .bufferImageHeight = mip_info.height,
                .imageSubresource{
                    .aspectMask = vk::ImageAspectFlagBits::eColor,
                    .mipLevel = mip,
                    .baseArrayLayer = 0,
                    .layerCount = image.info.resources.layers,
                },
                .imageOffset = {0, 0, 0},
                .imageExtent = {std::max(image.info.size.width >> mip, 1u),
                                std::max(image.info.size.height >> mip, 1u),
                                std::max(image.info.size.depth >> mip, 1u)},
            });
        }
        const auto download = runtime.GetStagingPool().Request(
            image.info.guest_size, MemoryType::HostCached, 256, true);
        tile_manager.TileImage(image, copies, download.buffer, download.offset);
        if (readback_log && !readback_published.contains(image.info.guest_address)) {
            // gow3 GOW3_READBACK_LOG: the first copy's texels as the GPU holds them (linear, mip
            // 0, layer 0), to compare with where the tiled copy puts them.
            const u64 linear_size =
                u64{image.info.size.width} * image.info.size.height * (image.info.num_bits / 8);
            const auto linear =
                runtime.GetStagingPool().Request(linear_size, MemoryType::HostCached, 16, true);
            vk::BufferImageCopy copy = copies[0];
            copy.bufferOffset = linear.offset;
            copy.bufferRowLength = image.info.size.width;
            copy.bufferImageHeight = image.info.size.height;
            copy.imageSubresource.layerCount = 1;
            copy.imageExtent.depth = 1;
            runtime.DownloadImage(&image, linear.buffer, std::span{&copy, 1});
            scheduler.DeferPriorityOperation([this, linear, linear_size,
                                              address = image.info.guest_address,
                                              width = image.info.size.width,
                                              height = image.info.size.height] {
                linear.Invalidate();
                std::printf("Readback first copy GPU texels (linear): %#llx %ux%u:%s\n",
                            static_cast<unsigned long long>(address), width, height,
                            NonZeroTexels(linear.mapped, linear_size, 16).c_str());
                runtime.GetStagingPool().FreeDeferred(linear);
            });
        }
        return PendingDownload{image.info.guest_address, image.info.guest_size, download};
    }
    const u32 download_size = image.info.pitch * image.info.size.height * image.info.size.depth *
                              image.info.resources.layers * (image.info.num_bits / 8);
    ASSERT(download_size <= image.info.guest_size);
    const auto download =
        runtime.GetStagingPool().Request(download_size, MemoryType::HostCached, 16, true);
    const vk::BufferImageCopy image_download = {
        .bufferOffset = download.offset,
        .bufferRowLength = image.info.pitch,
        .bufferImageHeight = image.info.size.height,
        .imageSubresource =
            {
                .aspectMask = image.info.props.is_depth ? vk::ImageAspectFlagBits::eDepth
                                                        : vk::ImageAspectFlagBits::eColor,
                .mipLevel = 0,
                .baseArrayLayer = 0,
                .layerCount = image.info.resources.layers,
            },
        .imageOffset = {0, 0, 0},
        .imageExtent = {image.info.size.width, image.info.size.height, image.info.size.depth},
    };
    runtime.DownloadImage(&image, download.buffer, std::span{&image_download, 1});
    return PendingDownload{image.info.guest_address, download_size, download};
}

void TextureCache::LogTexelStats(Image& image) {
    const auto& info = image.info;
    const auto format = info.pixel_format;
    const bool half = format == vk::Format::eR16G16B16A16Sfloat ||
                      format == vk::Format::eR16G16Sfloat || format == vk::Format::eR16Sfloat;
    const bool single = format == vk::Format::eR32G32B32A32Sfloat ||
                        format == vk::Format::eR32G32Sfloat || format == vk::Format::eR32Sfloat;
    if ((!half && !single) || info.props.is_depth || info.size.width > 256 ||
        info.size.height > 256 || info.size.depth != 1) {
        return;
    }
    static std::unordered_set<u64> seen; // GPU thread only
    if (!seen.insert(info.guest_address ^ (u64(format) << 56) ^ (u64{info.size.width} << 40))
             .second) {
        return;
    }
    const u32 texel_bytes = info.num_bits / 8;
    const u32 layers = info.resources.layers, levels = info.resources.levels;
    boost::container::small_vector<vk::BufferImageCopy, 16> copies;
    std::array<u64, 16> offsets{};
    u64 total = 0;
    for (u32 mip = 0; mip < levels && mip < 16; ++mip) {
        const u32 w = std::max(info.size.width >> mip, 1u), h = std::max(info.size.height >> mip, 1u);
        offsets[mip] = total;
        copies.push_back(vk::BufferImageCopy{
            .bufferOffset = total,
            .bufferRowLength = w,
            .bufferImageHeight = h,
            .imageSubresource{.aspectMask = vk::ImageAspectFlagBits::eColor,
                              .mipLevel = mip,
                              .baseArrayLayer = 0,
                              .layerCount = layers},
            .imageOffset = {0, 0, 0},
            .imageExtent = {w, h, 1},
        });
        total += u64{w} * h * layers * texel_bytes;
    }
    const auto staging = runtime.GetStagingPool().Request(total, MemoryType::HostCached, 16, true);
    for (auto& copy : copies) {
        copy.bufferOffset += staging.offset;
    }
    runtime.DownloadImage(&image, staging.buffer, copies);
    scheduler.DeferPriorityOperation([this, staging, offsets, half, layers, texel_bytes,
                                      levels = u32(copies.size()), address = info.guest_address,
                                      width = info.size.width, height = info.size.height,
                                      name = vk::to_string(format)] {
        staging.Invalidate();
        const u32 components = texel_bytes / (half ? 2 : 4);
        for (u32 mip = 0; mip < levels; ++mip) {
            const u32 w = std::max(width >> mip, 1u), h = std::max(height >> mip, 1u);
            const u64 count = u64{w} * h * layers * components;
            u64 nans = 0, infs = 0, nonzero = 0;
            float lo = std::numeric_limits<float>::infinity(), hi = -lo;
            const u8* base = staging.mapped + offsets[mip];
            for (u64 i = 0; i < count; ++i) {
                float v;
                if (half) {
                    u16 h16;
                    std::memcpy(&h16, base + i * 2, 2);
                    v = HalfToFloat(h16);
                } else {
                    std::memcpy(&v, base + i * 4, 4);
                }
                if (std::isnan(v)) {
                    ++nans;
                } else if (std::isinf(v)) {
                    ++infs;
                } else {
                    lo = std::min(lo, v);
                    hi = std::max(hi, v);
                    nonzero += v != 0.0f;
                }
            }
            std::printf("Texel stats %#llx %ux%u x%u %s mip %u: %llu values, NaN %llu, inf %llu, "
                        "non-zero %llu, finite range %g .. %g\n",
                        static_cast<unsigned long long>(address), width, height, layers,
                        name.c_str(), mip, static_cast<unsigned long long>(count),
                        static_cast<unsigned long long>(nans), static_cast<unsigned long long>(infs),
                        static_cast<unsigned long long>(nonzero), lo, hi);
        }
        runtime.GetStagingPool().FreeDeferred(staging);
    });
}

void TextureCache::DownloadImageMemory(ImageId image_id, bool sync) {
    Image& image = slot_images[image_id];
    ResolveReadbacks(image.info.guest_address, image.info.guest_size, true,
                     ReadbackSource::ImageDownload);
    if (False(image.flags & ImageFlagBits::GpuModified)) {
        return;
    }
    const u32 download_size = image.info.pitch * image.info.size.height * image.info.size.depth *
                              image.info.resources.layers * (image.info.num_bits / 8);
    ASSERT(download_size <= image.info.guest_size);
    const auto download =
        runtime.GetStagingPool().Request(download_size, MemoryType::HostCached, 16, !sync);
    const vk::BufferImageCopy image_download = {
        .bufferOffset = download.offset,
        .bufferRowLength = image.info.pitch,
        .bufferImageHeight = image.info.size.height,
        .imageSubresource =
            {
                .aspectMask = image.info.props.is_depth ? vk::ImageAspectFlagBits::eDepth
                                                        : vk::ImageAspectFlagBits::eColor,
                .mipLevel = 0,
                .baseArrayLayer = 0,
                .layerCount = image.info.resources.layers,
            },
        .imageOffset = {0, 0, 0},
        .imageExtent = {image.info.size.width, image.info.size.height, image.info.size.depth},
    };
    runtime.DownloadImage(&image, download.buffer, std::span{&image_download, 1});
    if (sync) {
        scheduler.Finish();
        download.Invalidate();
        Core::Memory::Instance()->TryWriteBacking(std::bit_cast<u8*>(image.info.guest_address),
                                                  download.mapped, download_size);
    } else {
        scheduler.DeferPriorityOperation(
            [this, device_addr = image.info.guest_address, download, download_size] {
                download.Invalidate();
                Core::Memory::Instance()->TryWriteBacking(std::bit_cast<u8*>(device_addr),
                                                          download.mapped, download_size);
                runtime.GetStagingPool().FreeDeferred(download);
            });
    }
}

// gow3: images sharing guest memory (aliases), ported from cuesta4/shadPS4 (eltutz) commit
// 189ca3b6. A write to one is copied into the others before they are read, as on the PS4,
// where they are the same memory.
static bool Covers(const Extent3D& extent, const ImageInfo& info) {
    return extent.width == info.size.width && extent.height == info.size.height &&
           extent.depth == info.size.depth;
}

static bool CanAlias(const Image& lhs, const Image& rhs) {
    return GetAliasCopyExtent(lhs.info, rhs.info) || GetAliasCopyExtent(rhs.info, lhs.info);
}

template <typename Func>
void TextureCache::ForEachAlias(const Image& image, Func&& func) {
    const auto page = page_table.find(image.info.guest_address >> Traits::PageBits);
    if (page == nullptr) {
        return;
    }
    for (const ImageId candidate_id : *page) {
        Image& candidate = slot_images[candidate_id];
        if (&candidate != &image && candidate.info.guest_address == image.info.guest_address &&
            CanAlias(candidate, image)) {
            func(candidate_id, candidate);
        }
    }
}

void TextureCache::SynchronizeAlias(ImageId image_id) {
    CACHE_STAT(alias_syncs, 1);
    Image& dst = slot_images[image_id];
    if (False(dst.flags & ImageFlagBits::Aliased)) {
        return;
    }
    const auto state_it = alias_states.find(dst.info.guest_address);
    if (state_it == alias_states.end()) {
        return;
    }
    AliasState& state = state_it.value();
    if (!IsLiveImage(state.backing, state.backing_uid)) {
        if (!IsLiveImage(state.writer, state.writer_uid)) {
            state.ResetAuthority();
            return;
        }
        state.backing = state.writer, state.backing_uid = state.writer_uid;
        state.writer = {}, state.writer_uid = 0;
    }
    if (state.writer == image_id && state.writer_uid == dst.image_uid) {
        return;
    }
    if (!CommitAliasWriter(state)) {
        state.ResetAuthority();
        return;
    }
    Image& current = slot_images[state.backing];
    const std::optional extent = GetAliasCopyExtent(current.info, dst.info);
    if (state.backing == image_id || current.alias_generation <= dst.alias_generation || !extent) {
        return;
    }
    CopyAlias(state.backing, image_id, *extent);
    dst.alias_generation = current.alias_generation;
    const std::optional reverse = GetAliasCopyExtent(dst.info, current.info);
    if (reverse && Covers(*reverse, current.info) && !Covers(*extent, dst.info)) {
        state.backing = image_id, state.backing_uid = dst.image_uid;
    }
}

bool TextureCache::CommitAliasWriter(AliasState& state) {
    if (!state.writer) {
        return true;
    }
    if (!IsLiveImage(state.writer, state.writer_uid) ||
        !IsLiveImage(state.backing, state.backing_uid)) {
        return false;
    }
    Image& writer = slot_images[state.writer];
    Image& backing = slot_images[state.backing];
    if (state.writer != state.backing && writer.alias_generation > backing.alias_generation) {
        const std::optional extent = GetAliasCopyExtent(writer.info, backing.info);
        if (!extent) {
            return false;
        }
        CopyAlias(state.writer, state.backing, *extent);
        backing.alias_generation = writer.alias_generation;
    }
    state.writer = {}, state.writer_uid = 0;
    return true;
}

void TextureCache::CopyAlias(ImageId src_id, ImageId dst_id, const Extent3D& extent) {
    CACHE_STAT(alias_copies, 1);
    Image& dst = slot_images[dst_id];
    runtime.CopyAliasImage(&slot_images[src_id], &dst, {extent.width, extent.height, extent.depth});
    dst.flags |= ImageFlagBits::GpuModified;
    dst.flags &= ~ImageFlagBits::Dirty;
}

void TextureCache::PublishAliasWrite(ImageId image_id) {
    Image& image = slot_images[image_id];
    image.alias_generation = ++alias_generation;
    if (False(image.flags & ImageFlagBits::Aliased)) {
        return;
    }
    std::scoped_lock lock{mutex};
    AliasState& state = alias_states[image.info.guest_address];
    if (IsLiveImage(state.backing, state.backing_uid) &&
        !CanAlias(image, slot_images[state.backing])) {
        state.ResetAuthority();
    }
    state.members = std::max(state.members, 2u);
    if (state.writer == image_id && state.writer_uid == image.image_uid) {
        return;
    }
    if (IsLiveImage(state.backing, state.backing_uid)) {
        const ImageInfo& backing = slot_images[state.backing].info;
        const std::optional extent = GetAliasCopyExtent(image.info, backing);
        if (extent && Covers(*extent, backing)) {
            state.backing = image_id, state.backing_uid = image.image_uid;
        } else {
            state.writer = image_id, state.writer_uid = image.image_uid;
        }
    } else {
        state.backing = image_id, state.backing_uid = image.image_uid;
    }
}

void TextureCache::InvalidateAlias(Image& image) {
    image.alias_generation = 0;
    if (True(image.flags & ImageFlagBits::Aliased)) {
        if (const auto it = alias_states.find(image.info.guest_address); it != alias_states.end()) {
            it.value().ResetAuthority();
        }
    }
}

namespace {
struct ImageWatchSpec {
    u32 width = 0, height = 0, min_layers = 0;
    std::atomic<u32> lines{0};
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
};
ImageWatchSpec& ImageWatch() {
    static ImageWatchSpec spec;
    static const bool parsed = [] {
        const char* env = std::getenv("GOW3_IMAGE_WATCH");
        if (env && std::sscanf(env, "%ux%u:%u", &spec.width, &spec.height, &spec.min_layers) >= 2) {
            const char* lines = std::getenv("GOW3_IMAGE_WATCH_LINES");
            spec.lines = lines && *lines ? u32(std::strtoul(lines, nullptr, 10)) : 4000;
        }
        return true;
    }();
    (void)parsed;
    return spec;
}
} // namespace

bool TextureCache::Watched(const ImageInfo& info) const {
    const auto& spec = ImageWatch();
    return spec.width != 0 && info.size.width == spec.width && info.size.height == spec.height &&
           info.resources.layers >= spec.min_layers;
}

void TextureCache::WatchNote(const Image& image, const char* event, const std::string& detail) {
    auto& spec = ImageWatch();
    u32 left = spec.lines.load(std::memory_order_relaxed);
    while (left && !spec.lines.compare_exchange_weak(left, left - 1, std::memory_order_relaxed)) {
    }
    if (!left) {
        return;
    }
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - spec.start).count();
    const auto flags = image.flags;
    std::printf("Image watch %.3f s: %#llx %ux%u L%u/%u %s%s [%s%s%s%s%s] tick %llu\n", seconds,
                static_cast<unsigned long long>(image.info.guest_address), image.info.size.width,
                image.info.size.height, image.info.resources.levels, image.info.resources.layers,
                event, detail.c_str(), True(flags & ImageFlagBits::Undefined) ? " undefined" : "",
                True(flags & ImageFlagBits::CpuDirty) ? " cpu-dirty" : "",
                True(flags & ImageFlagBits::MaybeCpuDirty) ? " maybe-cpu-dirty" : "",
                True(flags & ImageFlagBits::GpuDirty) ? " gpu-dirty" : "",
                True(flags & ImageFlagBits::GpuModified) ? " gpu-modified" : "",
                static_cast<unsigned long long>(scheduler.CurrentTick()));
}

void TextureCache::MarkAsMaybeDirty(ImageId image_id, Image& image) {
    if (image.hash == 0) {
        // Initialize hash
        const u8* addr = std::bit_cast<u8*>(image.info.guest_address);
        image.hash = XXH3_64bits(addr, image.info.guest_size);
    }
    image.flags |= ImageFlagBits::MaybeCpuDirty;
    UntrackImage(image_id);
}

void TextureCache::InvalidateMemory(VAddr addr, size_t size) {
    std::scoped_lock lock{mutex};
    const auto pages_start = PageManager::GetPageAddr(addr);
    const auto pages_end = PageManager::GetNextPageAddr(addr + size - 1);
    ForEachImageInRegion(pages_start, pages_end - pages_start, [&](ImageId image_id, Image& image) {
        const auto image_begin = image.info.guest_address;
        const auto image_end = image.info.guest_address + image.info.guest_size;
        if (image.Overlaps(addr, size)) {
            // Modified region overlaps image, so the image was definitely accessed by this fault.
            // Untrack the image, so that the range is unprotected and the guest can write freely.
            InvalidateAlias(image);
            image.flags |= ImageFlagBits::CpuDirty;
            UntrackImage(image_id);
            if (Watched(image.info)) {
                WatchNote(image, "CPU dirty",
                          std::format(" by a write at +{:#x} size {:#x}", addr - image_begin, size));
            }
        } else if (pages_end < image_end) {
            // This page access may or may not modify the image.
            // We should not mark it as dirty now. If it really was modified
            // it will receive more invalidations on its other pages.
            // Remove tracking from this page only.
            UntrackImageHead(image_id);
        } else if (image_begin < pages_start) {
            // This page access does not modify the image but the page should be untracked.
            // We should not mark this image as dirty now. If it really was modified
            // it will receive more invalidations on its other pages.
            UntrackImageTail(image_id);
        } else {
            // Image begins and ends on this page so it can not receive any more invalidations.
            // We will check it's hash later to see if it really was modified.
            MarkAsMaybeDirty(image_id, image);
            if (Watched(image.info)) {
                WatchNote(image, "maybe CPU dirty");
            }
        }
    });
}

void TextureCache::InvalidateMemoryFromGPU(VAddr address, size_t max_size) {
    // Partial buffer writes still need the image's untouched bytes in guest memory.
    ResolveReadbacks(address, max_size, true, ReadbackSource::GpuPartialWrite);
    std::scoped_lock lock{mutex};
    // gow3 GOW3_GPU_WRITE_SUBRESOURCES=1 (plan-linux-deck.md "Mausoleum lighting"): a GPU write
    // anywhere inside an image marks it too, not only one at its base address. UE4 copies each
    // reflection capture into its slice of the capture array: those writes start inside the
    // array, so the cached array kept what guest memory held when it was first sampled.
    static const bool subresources = [] {
        const char* env = std::getenv("GOW3_GPU_WRITE_SUBRESOURCES");
        return env && env[0] == '1';
    }();
    ForEachImageInRegion(address, max_size, [&](ImageId image_id, Image& image) {
        // Only consider images that match base address.
        // TODO: Maybe also consider subresources
        if (image.info.guest_address != address) {
            const bool inside = image.Overlaps(address, max_size);
            if (inside && Watched(image.info)) {
                WatchNote(image, subresources ? "GPU write inside" : "GPU write inside, ignored",
                          std::format(" at +{:#x} size {:#x}", address - image.info.guest_address,
                                      max_size));
            }
            if (!subresources || !inside) {
                return;
            }
        }
        // Ensure image is reuploaded when accessed again.
        InvalidateAlias(image);
        image.flags |= ImageFlagBits::GpuDirty;
        if (Watched(image.info)) {
            WatchNote(image, "GPU dirty", std::format(" by a GPU buffer write of {:#x}", max_size));
        }
    });
}

void TextureCache::UnmapMemory(VAddr cpu_addr, size_t size) {
    CancelReadbacks(cpu_addr, size);
    std::scoped_lock lk{mutex};

    ImageIds deleted_images;
    ForEachImageInRegion(cpu_addr, size, [&](ImageId id, Image&) { deleted_images.push_back(id); });
    for (const ImageId id : deleted_images) {
        if (unmapped_addresses.size() > 16384) {
            unmapped_addresses.clear();
        }
        unmapped_addresses.insert(slot_images[id].info.guest_address);
        if (CacheStatsEnabled()) {
            const ImageInfo& info = slot_images[id].info;
            CACHE_STAT(unmap_frees, 1);
            CACHE_STAT(unmap_free_bytes, info.guest_size);
            unmapped_recently[RecreateKey(info)] = stats_reports;
        }
        if (Watched(slot_images[id].info)) {
            WatchNote(slot_images[id], "unmapped",
                      std::format(" (guest unmap at {:#x} size {:#x})", cpu_addr, size));
        }
        if (ParkImage(id)) {
            continue;
        }
        // TODO: Download image data back to host.
        FreeImage(id);
    }
}

namespace {
bool ParkUnmappedEnabled() {
    static const bool enabled = [] {
        const char* env = std::getenv("GOW3_PARK_UNMAPPED");
        return !(env && env[0] == '0');
    }();
    return enabled;
}
// Several 4K pairs and many shadow maps; a parked image is dropped after ParkedMaxAge.
constexpr u64 ParkedBudget = 768_MB;
constexpr u64 ParkedMaxImage = 128_MB;
constexpr auto ParkedMaxAge = std::chrono::seconds(10);
} // namespace

bool TextureCache::ParkImage(ImageId image_id) {
    Image& image = slot_images[image_id];
    if (!ParkUnmappedEnabled() || !(image.usage.render_target || image.usage.depth_target) ||
        image.usage.vo_surface || image.scene_proxy || True(image.flags & ImageFlagBits::Aliased) ||
        image.info.guest_size > ParkedMaxImage) {
        return false;
    }
    // As FreeImage and DeleteImage, without releasing the image and its views.
    ResolveReadbacks(image.info.guest_address, image.info.guest_size, true,
                     ReadbackSource::ImageFree);
    CancelReadbacks(image.info.guest_address, image.info.guest_size);
    UntrackImage(image_id);
    UnregisterImage(image_id);
    const auto& meta_info = image.info.meta_info;
    if (meta_info.cmask_addr) {
        surface_metas.erase(meta_info.cmask_addr);
    }
    if (meta_info.fmask_addr) {
        surface_metas.erase(meta_info.fmask_addr);
    }
    if (meta_info.htile_addr) {
        surface_metas.erase(meta_info.htile_addr);
    }
    {
        std::unique_lock lk{download_images_mutex};
        download_images.erase(image_id);
    }
    parked_images.push_back({image_id, std::chrono::steady_clock::now()});
    parked_bytes += image.info.guest_size;
    parked_count.store(u32(parked_images.size()), std::memory_order_relaxed);
    CACHE_STAT(parked, 1);
    EvictParkedImages(true);
    return true;
}

ImageId TextureCache::TakeParkedImage(const ImageInfo& info) {
    for (auto it = parked_images.begin(); it != parked_images.end(); ++it) {
        Image& image = slot_images[it->id];
        const ImageInfo& parked = image.info;
        if (parked.guest_address != info.guest_address || parked.guest_size != info.guest_size ||
            parked.size != info.size || parked.pixel_format != info.pixel_format ||
            parked.type != info.type || parked.tile_mode != info.tile_mode ||
            parked.pitch != info.pitch || parked.resources != info.resources ||
            parked.num_samples != info.num_samples || parked.num_bits != info.num_bits ||
            parked.props.is_tiled != info.props.is_tiled ||
            parked.props.is_depth != info.props.is_depth ||
            parked.props.has_stencil != info.props.has_stencil ||
            parked.props.is_block != info.props.is_block ||
            parked.props.is_pow2 != info.props.is_pow2 ||
            parked.props.is_volume != info.props.is_volume || parked.alt_tile != info.alt_tile) {
            continue;
        }
        const ImageId image_id = it->id;
        parked_bytes -= parked.guest_size;
        parked_images.erase(it);
        parked_count.store(u32(parked_images.size()), std::memory_order_relaxed);
        // Back as a new image would start: dirty, not bound, no usage yet (the caller registers
        // it and flags it Undefined as the unmap's address). Its layout state stays: it is the
        // Vulkan image's.
        image.flags = ImageFlagBits::Dirty;
        image.binding = {};
        image.usage = {};
        image.depth_id = {};
        image.depth_uid = 0;
        image.info.meta_info = info.meta_info;
        image.info.stencil_addr = info.stencil_addr;
        image.info.stencil_size = info.stencil_size;
        CACHE_STAT(parked_reused, 1);
        return image_id;
    }
    return {};
}

void TextureCache::EvictParkedImages(bool over_budget_only) {
    const auto now = std::chrono::steady_clock::now();
    while (!parked_images.empty()) {
        const ParkedImage& oldest = parked_images.front();
        if (parked_bytes <= ParkedBudget &&
            (over_budget_only || now - oldest.since < ParkedMaxAge)) {
            break;
        }
        const ImageId image_id = oldest.id;
        parked_bytes -= slot_images[image_id].info.guest_size;
        parked_images.erase(parked_images.begin());
        DeleteImage(image_id);
        CACHE_STAT(parked_evicted, 1);
    }
    parked_count.store(u32(parked_images.size()), std::memory_order_relaxed);
}

ImageId TextureCache::ResolveDepthOverlap(const ImageInfo& requested_info, BindingType binding,
                                          ImageId cache_image_id) {
    auto& cache_image = slot_images[cache_image_id];

    if (!cache_image.info.props.is_depth && !requested_info.props.is_depth) {
        return {};
    }

    const bool stencil_match =
        requested_info.props.has_stencil == cache_image.info.props.has_stencil;
    const bool bpp_match = requested_info.num_bits == cache_image.info.num_bits;

    // If an image in the cache has less slices we need to expand it
    bool recreate = cache_image.info.resources < requested_info.resources;

    switch (binding) {
    case BindingType::Texture:
        // The guest requires a depth sampled texture, but cache can offer only Rxf. Need to
        // recreate the image.
        recreate |= requested_info.props.is_depth && !cache_image.info.props.is_depth;
        break;
    case BindingType::Storage:
        // If the guest is going to use previously created depth as storage, the image needs to be
        // recreated. (TODO: Probably a case with linear rgba8 aliasing is legit)
        recreate |= cache_image.info.props.is_depth;
        break;
    case BindingType::RenderTarget:
        // Render target can have only Rxf format. If the cache contains only Dx[S8] we need to
        // re-create the image.
        ASSERT(!requested_info.props.is_depth);
        recreate |= cache_image.info.props.is_depth;
        break;
    case BindingType::DepthTarget:
        // The guest has requested previously allocated texture to be bound as a depth target.
        // In this case we need to convert Rx float to a Dx[S8] as requested
        recreate |= !cache_image.info.props.is_depth;

        // The guest is trying to bind a depth target and cache has it. Need to be sure that aspects
        // and bpp match
        recreate |= cache_image.info.props.is_depth && !(stencil_match && bpp_match);
        break;
    default:
        break;
    }

    if (recreate) {
        auto new_info = requested_info;
        new_info.resources = std::max(requested_info.resources, cache_image.info.resources);
        new_info.UpdateSize();
        const auto new_image_id = slot_images.insert(instance, runtime, slot_image_views, new_info);
        RegisterImage(new_image_id);
        // gow3 (from the Uncharted port): the insert moves every image when the slot vector
        // grows, so `cache_image` may point into freed memory now: fetched again.
        auto& old_image = slot_images[cache_image_id];

        // Inherit image usage
        auto& new_image = slot_images[new_image_id];
        new_image.usage = old_image.usage;
        new_image.flags &= ~ImageFlagBits::Dirty;
        // When creating a depth buffer through overlap resolution don't clear it on first use.
        new_image.info.meta_info.htile_clear_mask = 0;
        runtime.CopyColorAndDepth(&old_image, &new_image);

        // Free the cache image.
        FreeImage(cache_image_id);
        return new_image_id;
    }

    // Will be handled by view
    return cache_image_id;
}

std::tuple<ImageId, int, int> TextureCache::ResolveOverlap(const ImageInfo& image_info,
                                                           BindingType binding,
                                                           ImageId cache_image_id,
                                                           ImageId merged_image_id) {
    CACHE_STAT(overlaps, 1);
    auto& cache_image = slot_images[cache_image_id];
    const bool safe_to_delete =
        scheduler.CurrentTick() - cache_image.tick_accessed_last > NumFramesBeforeRemoval;

    // Equal address
    if (image_info.guest_address == cache_image.info.guest_address) {
        const u32 lhs_block_size = image_info.num_bits * image_info.num_samples;
        const u32 rhs_block_size = cache_image.info.num_bits * cache_image.info.num_samples;
        if (image_info.BlockDim() != cache_image.info.BlockDim() ||
            lhs_block_size != rhs_block_size) {
            // Very likely this kind of overlap is caused by allocation from a pool.
            if (safe_to_delete) {
                FreeImage(cache_image_id);
            }
            return {merged_image_id, -1, -1};
        }

        if (const auto depth_image_id = ResolveDepthOverlap(image_info, binding, cache_image_id)) {
            return {depth_image_id, -1, -1};
        }

        // Compressed view of uncompressed image with same block size.
        if (image_info.props.is_block && !cache_image.info.props.is_block) {
            return {ExpandImage(image_info, cache_image_id), -1, -1};
        }

        // gow3: MediEvil's colour grading LUT and translucency lighting volumes switched
        // between a 2D array (render target) and a 3D image (texture) several times a frame,
        // each switch a full re-upload from guest memory: about 2 GB/s over PCIe (T-C1). The
        // 3D image serves the slices through a 2D array view (V3; upstream 1.3.4 has the same
        // idea; its GOW3_VOLUME_VIEWS=0 also turns this off, as does toggle bit 58).
        static const bool volume_views = [] {
            const char* env = std::getenv("GOW3_VOLUME_VIEWS");
            return !env || env[0] != '0';
        }();
        if (volume_views && IsVolumeSliceTarget(image_info, binding, cache_image.info)) {
            CACHE_STAT(volume_views, 1);
            return {cache_image_id, -1, -1};
        }

        if (image_info.guest_size == cache_image.info.guest_size &&
            (image_info.type == AmdGpu::ImageType::Color3D ||
             cache_image.info.type == AmdGpu::ImageType::Color3D)) {
            return {ExpandImage(image_info, cache_image_id), -1, -1};
        }

        const bool pow2_padding_only =
            image_info.props.is_pow2 != cache_image.info.props.is_pow2 &&
            image_info.tile_mode == cache_image.info.tile_mode &&
            image_info.size == cache_image.info.size &&
            image_info.pitch == cache_image.info.pitch && image_info.resources.levels == 1 &&
            cache_image.info.resources.levels == 1 && image_info.resources.layers == 1 &&
            cache_image.info.resources.layers == 1;

        // Size and resources are less than or equal, use image view.
        if (image_info.pixel_format != cache_image.info.pixel_format ||
            image_info.guest_size <= cache_image.info.guest_size || pow2_padding_only) {
            auto result_id = merged_image_id ? merged_image_id : cache_image_id;
            const auto& result_image = slot_images[result_id];
            const bool is_compatible =
                IsVulkanFormatCompatible(result_image.info.pixel_format, image_info.pixel_format);
            return {is_compatible ? result_id : ImageId{}, -1, -1};
        }

        // Size and resources are greater, expand the image.
        if (image_info.type == cache_image.info.type &&
            image_info.resources > cache_image.info.resources) {
            return {ExpandImage(image_info, cache_image_id), -1, -1};
        }

        // Size is greater but resources are not, because the tiling mode is different.
        // Likely the address is reused for a image with a different tiling mode.
        if (image_info.tile_mode != cache_image.info.tile_mode) {
            if (safe_to_delete) {
                FreeImage(cache_image_id);
            }
            return {merged_image_id, -1, -1};
        }

        // Enhanced debug logging for unreachable case
        // Calculate expected size based on format and dimensions
        u64 expected_size =
            (static_cast<u64>(image_info.size.width) * static_cast<u64>(image_info.size.height) *
             static_cast<u64>(image_info.size.depth) * static_cast<u64>(image_info.num_bits) / 8);
        LOG_ERROR(Render_Vulkan,
                  "Unresolvable image overlap with equal memory address:\n"
                  "=== OLD IMAGE (cached) ===\n"
                  "  Address:        {:#x}\n"
                  "  Size:           {:#x} bytes\n"
                  "  Format:         {}\n"
                  "  Type:           {}\n"
                  "  Width:          {}\n"
                  "  Height:         {}\n"
                  "  Depth:          {}\n"
                  "  Pitch:          {}\n"
                  "  Mip levels:     {}\n"
                  "  Array layers:   {}\n"
                  "  Samples:        {}\n"
                  "  Tile mode:      {:#x}\n"
                  "  Block size:     {} bits\n"
                  "  Is block-comp:  {}\n"
                  "  Guest size:     {:#x}\n"
                  "  Last accessed:  tick {}\n"
                  "  Safe to delete: {}\n"
                  "  isPow2:         {}\n"
                  "  Alt tile:       {}\n"
                  "\n"
                  "=== NEW IMAGE (requested) ===\n"
                  "  Address:        {:#x}\n"
                  "  Size:           {:#x} bytes\n"
                  "  Format:         {}\n"
                  "  Type:           {}\n"
                  "  Width:          {}\n"
                  "  Height:         {}\n"
                  "  Depth:          {}\n"
                  "  Pitch:          {}\n"
                  "  Mip levels:     {}\n"
                  "  Array layers:   {}\n"
                  "  Samples:        {}\n"
                  "  Tile mode:      {:#x}\n"
                  "  Block size:     {} bits\n"
                  "  Is block-comp:  {}\n"
                  "  Guest size:     {:#x}\n"
                  "  isPow2:         {}\n"
                  "  Alt tile:       {}\n"
                  "\n"
                  "=== COMPARISON ===\n"
                  "  Same format:           {}\n"
                  "  Same type:             {}\n"
                  "  Same tile mode:        {}\n"
                  "  Same block size:       {}\n"
                  "  Same BlockDim:         {}\n"
                  "  Same pitch:            {}\n"
                  "  Same pow2:             {}\n"
                  "  Same alt tile:         {}\n"
                  "  Old resources <= new:  {} (old: {}, new: {})\n"
                  "  Old size <= new size:  {}\n"
                  "  Expected size (calc):  {} bytes\n"
                  "  Size ratio (new/expected): {:.2f}x\n"
                  "  Size ratio (new/old):  {:.2f}x\n"
                  "  Old vs expected diff:  {} bytes ({:+.2f}%)\n"
                  "  New vs expected diff:  {} bytes ({:+.2f}%)\n"
                  "  Merged image ID:       {}\n"
                  "  Binding type:          {}\n"
                  "  Current tick:          {}\n"
                  "  Age (ticks since last access): {}",

                  // Old image details
                  cache_image.info.guest_address, cache_image.info.guest_size,
                  vk::to_string(cache_image.info.pixel_format),
                  static_cast<int>(cache_image.info.type), cache_image.info.size.width,
                  cache_image.info.size.height, cache_image.info.size.depth, cache_image.info.pitch,
                  cache_image.info.resources.levels, cache_image.info.resources.layers,
                  cache_image.info.num_samples, static_cast<u32>(cache_image.info.tile_mode),
                  cache_image.info.num_bits, +cache_image.info.props.is_block,
                  cache_image.info.guest_size, cache_image.tick_accessed_last, safe_to_delete,
                  bool(cache_image.info.props.is_pow2), cache_image.info.alt_tile,

                  // New image details
                  image_info.guest_address, image_info.guest_size,
                  vk::to_string(image_info.pixel_format), static_cast<int>(image_info.type),
                  image_info.size.width, image_info.size.height, image_info.size.depth,
                  image_info.pitch, image_info.resources.levels, image_info.resources.layers,
                  image_info.num_samples, static_cast<u32>(image_info.tile_mode),
                  image_info.num_bits, image_info.props.is_block, image_info.guest_size,
                  bool(image_info.props.is_pow2), image_info.alt_tile,

                  // Comparison
                  (image_info.pixel_format == cache_image.info.pixel_format),
                  (image_info.type == cache_image.info.type),
                  (image_info.tile_mode == cache_image.info.tile_mode),
                  (image_info.num_bits == cache_image.info.num_bits),
                  (image_info.BlockDim() == cache_image.info.BlockDim()),
                  (image_info.pitch == cache_image.info.pitch),
                  (image_info.props.is_pow2 == cache_image.info.props.is_pow2),
                  (image_info.alt_tile == cache_image.info.alt_tile),
                  (cache_image.info.resources <= image_info.resources),
                  cache_image.info.resources.levels, image_info.resources.levels,
                  (cache_image.info.guest_size <= image_info.guest_size), expected_size,

                  // Size ratios
                  static_cast<double>(image_info.guest_size) / expected_size,
                  static_cast<double>(image_info.guest_size) / cache_image.info.guest_size,

                  // Difference between actual and expected sizes with percentages
                  static_cast<s64>(cache_image.info.guest_size) - static_cast<s64>(expected_size),
                  (static_cast<double>(cache_image.info.guest_size) / expected_size - 1.0) * 100.0,

                  static_cast<s64>(image_info.guest_size) - static_cast<s64>(expected_size),
                  (static_cast<double>(image_info.guest_size) / expected_size - 1.0) * 100.0,

                  merged_image_id.index, static_cast<int>(binding), scheduler.CurrentTick(),
                  scheduler.CurrentTick() - cache_image.tick_accessed_last);

        UNREACHABLE_MSG("Encountered unresolvable image overlap with equal memory address.");
    }

    // Right overlap, the image requested is a possible subresource of the image from cache.
    if (image_info.guest_address > cache_image.info.guest_address) {
        const auto mip = image_info.MipOf(cache_image.info);
        const auto slice = mip >= 0 ? image_info.SliceOf(cache_image.info, mip) : -1;
        if (binding == BindingType::RenderTarget || binding == BindingType::DepthTarget) {
            NoteMipTarget(image_info, cache_image.info, mip, slice, false);
        }
        if (mip >= 0 && slice >= 0) {
            return {cache_image_id, mip, slice};
        }

        // Image isn't a subresource but a chance overlap.
        if (safe_to_delete) {
            FreeImage(cache_image_id);
        }

        return {{}, -1, -1};
    } else {
        // Left overlap, the image from cache is a possible subresource of the image requested
        const auto merge_mip = cache_image.info.MipOf(image_info);
        if (cache_image.binding.is_target) {
            NoteMipTarget(cache_image.info, image_info, merge_mip,
                          merge_mip >= 0 ? cache_image.info.SliceOf(image_info, merge_mip) : -1,
                          true);
        }
        if (auto mip = merge_mip; mip >= 0) {
            if (auto slice = cache_image.info.SliceOf(image_info, mip); slice >= 0) {
                // We have a larger image created and a separate one, representing a subres of it
                // bound as render target. In this case we need to rebind render target.
                if (cache_image.binding.is_target) {
                    cache_image.binding.needs_rebind = 1u;
                    if (merged_image_id) {
                        GetImage(merged_image_id).binding.is_target = 1u;
                    }

                    FreeImage(cache_image_id);
                    return {merged_image_id, -1, -1};
                }

                // We need to have a larger, already allocated image to copy this one into
                if (merged_image_id) {
                    auto& merged_image = slot_images[merged_image_id];
                    runtime.CopyMip(&cache_image, &merged_image, mip, slice);
                    if (Watched(merged_image.info)) {
                        WatchNote(merged_image, "mip merged in",
                                  std::format(" (mip {} slice {} from a separate image)", mip, slice));
                    }
                    FreeImage(cache_image_id);
                }
            }
        }
    }

    return {merged_image_id, -1, -1};
}

ImageId TextureCache::ExpandImage(const ImageInfo& info, ImageId image_id) {
    CACHE_STAT(expands, 1);
    if (CacheStatsEnabled()) {
        // Which images are expanded, from what to what (the busiest are printed with the stats).
        const ImageInfo& old = slot_images[image_id].info;
        const char* why = info.props.is_block && !old.props.is_block ? "block view"
                          : (info.type == AmdGpu::ImageType::Color3D) !=
                                    (old.type == AmdGpu::ImageType::Color3D)
                              ? "2D/3D"
                              : "bigger";
        char key[192];
        std::snprintf(key, sizeof(key), "%#llx %s: %ux%ux%u t%d %s L%u/%u -> %ux%ux%u t%d %s L%u/%u, %llu KiB",
                      (unsigned long long)info.guest_address, why, old.size.width, old.size.height,
                      old.size.depth, int(old.type), vk::to_string(old.pixel_format).c_str(),
                      old.resources.levels, old.resources.layers, info.size.width, info.size.height,
                      info.size.depth, int(info.type), vk::to_string(info.pixel_format).c_str(),
                      info.resources.levels, info.resources.layers,
                      (unsigned long long)(info.guest_size >> 10));
        ++expand_sites[key];
    }
    if (image_stats) {
        static int logged = 0;
        if (logged < 12) {
            ++logged;
            const auto& old = slot_images[image_id].info;
            std::printf("Image expand: %#llx %s type %u %ux%ux%u mips %u layers %u tile %u -> type %u "
                        "%ux%ux%u mips %u layers %u tile %u %s\n",
                        (unsigned long long)info.guest_address, vk::to_string(info.pixel_format).c_str(),
                        u32(old.type), old.size.width, old.size.height, old.size.depth,
                        old.resources.levels, old.resources.layers, u32(old.tile_mode), u32(info.type),
                        info.size.width, info.size.height, info.size.depth, info.resources.levels,
                        info.resources.layers, u32(info.tile_mode), vk::to_string(old.pixel_format).c_str());
        }
    }
    const auto new_image_id = slot_images.insert(instance, runtime, slot_image_views, info);
    RegisterImage(new_image_id);

    auto& src_image = slot_images[image_id];
    auto& new_image = slot_images[new_image_id];

    RefreshImage(new_image);
    runtime.CopyImage(&src_image, &new_image);
    if (Watched(new_image.info)) {
        WatchNote(new_image, "expanded",
                  std::format(" (contents copied from the old image L{}/{})",
                              src_image.info.resources.levels, src_image.info.resources.layers));
    }

    if (src_image.binding.is_bound || src_image.binding.is_target) {
        src_image.binding.needs_rebind = 1u;
    }

    FreeImage(image_id);
    TrackImage(new_image_id);
    return new_image_id;
}

ImageId TextureCache::FindImage(ImageDesc& desc, bool exact_fmt) {
    const auto& info = desc.info;
    ASSERT(info.guest_address != 0);
    GOW3_SECTION(FindImage);

    scheduler.SyncWithDrawRecorder(); // before the lock: the draw recorder takes it too
    std::unique_lock lock{mutex, std::defer_lock};
    {
        GOW3_SECTION(CacheLock);
        lock.lock();
    }

    u64 key_hash = info.guest_address ^ info.guest_size << 17 ^ u64(info.pixel_format) << 40 ^
                   u64(info.size.width) << 24 ^ info.size.height ^ u64(desc.type) << 58 ^
                   u64(exact_fmt) << 63;
    key_hash = (key_hash ^ key_hash >> 33) * 0xFF51AFD7ED558CCDull;
    key_hash = (key_hash ^ key_hash >> 33) * 0xC4CEB9FE1A85EC53ull;
    key_hash ^= key_hash >> 33;
    auto& cached = find_image_cache[key_hash % find_image_cache.size()];
    // An exact match stays valid while that image is registered with the same description;
    // resolved overlaps (views into other images) only until any image registration changes.
    const auto still_exact = [&] {
        if (cached.view_mip >= 0 || cached.view_slice >= 0 || !cached.image_id) {
            return false;
        }
        const Image& image = slot_images[cached.image_id];
        return True(image.flags & ImageFlagBits::Registered) &&
               image.info.guest_address == info.guest_address &&
               image.info.guest_size == info.guest_size && image.info.size == info.size &&
               image.info.pixel_format == info.pixel_format;
    };
    if ((cached.generation == registry_generation.load(std::memory_order_relaxed) || still_exact()) &&
        cached.address == info.guest_address &&
        cached.size == info.guest_size && cached.extent == info.size &&
        cached.format == info.pixel_format && cached.type == info.type &&
        cached.exact_fmt == exact_fmt && cached.binding == desc.type &&
        cached.levels == info.resources.levels && cached.layers == info.resources.layers &&
        !Gow3Toggle::Disabled(Gow3Toggle::FindImageCache)) {
        CACHE_STAT(fast_lookups, 1);
        Image& image = slot_images[cached.image_id];
        image.tick_accessed_last = scheduler.CurrentTick();
        TouchImage(image);
        if (cached.view_mip > 0) {
            desc.view_info.range.base.level = cached.view_mip;
        }
        if (cached.view_slice > 0) {
            desc.view_info.range.base.layer = cached.view_slice;
        }
        return cached.image_id;
    }

    // Study 2026-10-29 item 10: a miss only because another image's registration changed the
    // generation (the same key resolves to the same image again), counted below.
    const bool same_key = Gow3Stats::enabled && cached.image_id && cached.address == info.guest_address &&
                          cached.size == info.guest_size && cached.extent == info.size &&
                          cached.format == info.pixel_format && cached.type == info.type &&
                          cached.exact_fmt == exact_fmt && cached.binding == desc.type &&
                          cached.levels == info.resources.levels &&
                          cached.layers == info.resources.layers;
    const FindImageCacheEntry previous = cached;

    ImageIds image_ids;
    ForEachImageInRegion(info.guest_address, info.guest_size,
                         [&](ImageId image_id, Image& image) { image_ids.push_back(image_id); });
    CACHE_STAT(slow_lookups, 1);
    if (Gow3Stats::enabled) Gow3Stats::image_slow_lookups.fetch_add(1, std::memory_order_relaxed);
    CACHE_STAT(slow_candidates, image_ids.size());

    ImageId image_id{};

    // Check for a perfect match first
    for (const auto& cache_id : image_ids) {
        auto& cache_image = slot_images[cache_id];
        if (cache_image.info.guest_address != info.guest_address) {
            continue;
        }
        if (cache_image.info.guest_size != info.guest_size) {
            continue;
        }
        if (cache_image.info.size != info.size) {
            continue;
        }
        if (!IsVulkanFormatCompatible(cache_image.info.pixel_format, info.pixel_format) ||
            (cache_image.info.type != info.type && info.size != Extent3D{1, 1, 1})) {
            continue;
        }
        if (exact_fmt && info.pixel_format != cache_image.info.pixel_format) {
            continue;
        }
        image_id = cache_id;
    }

    // Try to resolve overlaps (if any)
    int view_mip{-1};
    int view_slice{-1};
    if (!image_id) {
        for (const auto& cache_id : image_ids) {
            view_mip = -1;
            view_slice = -1;

            const auto& merged_info = image_id ? slot_images[image_id].info : info;
            auto [overlap_image_id, overlap_view_mip, overlap_view_slice] =
                ResolveOverlap(merged_info, desc.type, cache_id, image_id);
            if (overlap_image_id) {
                image_id = overlap_image_id;
                view_mip = overlap_view_mip;
                view_slice = overlap_view_slice;
            }
        }
    }

    if (image_id) {
        Image& image_resolved = slot_images[image_id];
        if (exact_fmt && info.pixel_format != image_resolved.info.pixel_format) {
            // Cannot reuse this image as we need the exact requested format.
            image_id = {};
        } else if (image_resolved.info.resources < info.resources &&
                   !IsVolumeSliceTarget(info, desc.type, image_resolved.info)) {
            // The image was clearly picked up wrong.
            FreeImage(image_id);
            image_id = {};
            LOG_WARNING(Render_Vulkan, "Image overlap resolve failed");
        }
    }
    // Create and register a new image
    if (!image_id) {
        if (CacheStatsEnabled()) {
            // Which images are created, and how many were freed by a guest unmap shortly before
            // (MediEvil: hundreds per frame while panning, T-M1b).
            if (const auto it = unmapped_recently.find(RecreateKey(info)); it != unmapped_recently.end()) {
                CACHE_STAT(recreated, 1);
                CACHE_STAT(recreated_bytes, info.guest_size);
                unmapped_recently.erase(it);
            }
            static constexpr const char* bindings[] = {"texture", "storage", "render target",
                                                       "depth target", "video out"};
            char key[160];
            std::snprintf(key, sizeof(key), "%ux%ux%u t%d %s L%u/%u %s %s, %llu KiB", info.size.width,
                          info.size.height, info.size.depth, int(info.type),
                          vk::to_string(info.pixel_format).c_str(), info.resources.levels,
                          info.resources.layers, info.props.is_tiled ? "tiled" : "linear",
                          bindings[u32(desc.type)], (unsigned long long)(info.guest_size >> 10));
            ++create_sites[key];
        }
        if (parked_count.load(std::memory_order_relaxed)) {
            image_id = TakeParkedImage(info);
        }
        if (!image_id) {
            const auto start = std::chrono::steady_clock::now();
            image_id = slot_images.insert(instance, runtime, slot_image_views, info);
            if (CacheStatsEnabled()) {
                const u64 ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                   std::chrono::steady_clock::now() - start).count();
                create_ns += ns;
                create_max_ns = std::max(create_max_ns, ns);
            }
        }
        RegisterImage(image_id);
        if (Watched(slot_images[image_id].info)) {
            static constexpr const char* bindings[] = {"texture", "storage", "render target",
                                                       "depth target", "video out"};
            const u32 b = u32(desc.type);
            WatchNote(slot_images[image_id], "created",
                      std::format(" for use as {}", b < 5 ? bindings[b] : "?"));
        }
        // gow3: UE4 takes render targets from pool memory it decommits while a pass is unused
        // and commits again when it is needed (MediEvil: the first torch in view); the guest
        // memory there holds nothing the GPU needs to load (T-R1: 64 + 32 MiB uploaded per torch
        // coming into view, a 20 ms frame).
        // gow3 GOW3_NEW_IMAGES_UNDEFINED (default on; 0: only the unmap / GPU-freed cases): every
        // new image counts as undefined until its first use. A first use as a render / depth
        // target then skips the upload (the GPU renders into it; nothing the CPU wrote there is
        // read first); any other first use uploads as before (RefreshImage drops the flag), so
        // textures the CPU loads are unchanged. t-up4: 4K targets in fresh pool memory (no image
        // had been there) uploaded 32-64 MB each right before being rendered into.
        static const bool new_undefined = [] {
            const char* env = std::getenv("GOW3_NEW_IMAGES_UNDEFINED");
            return !(env && env[0] == '0');
        }();
        // Also where a GPU-written image was freed (FreeImage: its memory never got the GPU's data).
        const bool unmapped = unmapped_addresses.erase(info.guest_address) != 0;
        const bool gpu_freed = gpu_freed_addresses.erase(info.guest_address) != 0;
        if ((unmapped || gpu_freed || new_undefined) &&
            !Gow3Toggle::Disabled(Gow3Toggle::UndefinedTargets)) {
            slot_images[image_id].flags |= ImageFlagBits::Undefined;
            if (!unmapped) {
                CACHE_STAT(gpu_freed_reused, 1);
                CACHE_STAT(gpu_freed_reused_bytes, info.guest_size);
            }
        }
    }

    Image& image = slot_images[image_id];
    image.tick_accessed_last = scheduler.CurrentTick();
    TouchImage(image);

    // If the image requested is a subresource of the image from cache record its location.
    if (view_mip > 0) {
        desc.view_info.range.base.level = view_mip;
    }
    if (view_slice > 0) {
        desc.view_info.range.base.layer = view_slice;
    }

    if (same_key && previous.image_id == image_id && previous.view_mip == view_mip &&
        previous.view_slice == view_slice) {
        Gow3Stats::image_generation_misses.fetch_add(1, std::memory_order_relaxed);
    }
    cached = FindImageCacheEntry{
        .address = info.guest_address,
        .size = info.guest_size,
        .extent = info.size,
        .format = info.pixel_format,
        .type = info.type,
        .exact_fmt = exact_fmt,
        .binding = desc.type,
        .levels = info.resources.levels,
        .layers = info.resources.layers,
        .generation = registry_generation.load(std::memory_order_relaxed),
        .image_id = image_id,
        .view_mip = view_mip,
        .view_slice = view_slice,
    };
    return image_id;
}

ImageId TextureCache::FindImageFromRange(VAddr address, size_t size, bool ensure_valid) {
    ImageIds image_ids;
    ForEachImageInRegion(address, size, [&](ImageId image_id, Image& image) {
        if (image.info.guest_address != address) {
            return;
        }
        if (ensure_valid && !image.SafeToDownload()) {
            return;
        }
        image_ids.push_back(image_id);
    });
    if (image_ids.size() == 1) {
        // Sometimes image size might not exactly match with requested buffer size
        // If we only found 1 candidate image use it without too many questions.
        return image_ids.back();
    }
    if (!image_ids.empty()) {
        for (s32 i = 0; i < image_ids.size(); ++i) {
            Image& image = slot_images[image_ids[i]];
            if (image.info.guest_size == size) {
                return image_ids[i];
            }
        }
        LOG_WARNING(Render_Vulkan,
                    "Failed to find exact image match for copy addr={:#x}, size={:#x}", address,
                    size);
    }
    return {};
}

namespace {
// gow3 (plan-linux-deck.md "Mausoleum lighting"): GOW3_SKIPPED_TARGET_LOG=1 logs, once per image,
// each small render target or storage image (at most 64 KiB) the GPU writes but this cache never
// copies back to guest memory (tiled and wider than 8 pixels), with the first bytes the CPU would
// read there instead, as half floats: a CPU read of such a target sees those bytes.
bool SkippedTargetLog() {
    static const bool on = [] {
        const char* env = std::getenv("GOW3_SKIPPED_TARGET_LOG");
        return env && env[0] == '1';
    }();
    return on;
}

void LogSkippedTarget(const ImageInfo& info, const char* kind) {
    if (info.guest_address == 0 || info.guest_size > 64_KB) {
        return;
    }
    static std::mutex mutex;
    static std::unordered_set<u64> seen;
    const u64 key = info.guest_address ^ (u64{info.size.width} << 48) ^ (u64{info.size.height} << 32) ^
                    u64(info.pixel_format);
    {
        std::scoped_lock lk{mutex};
        if (!seen.insert(key).second) {
            return;
        }
    }
    std::array<u16, 16> words{};
    const bool read = Core::Memory::Instance()->ReadBacking(info.guest_address, words.data(),
                                                            sizeof(words));
    std::string halves;
    for (const u16 w : words) {
        halves += std::format(" {:.6g}", HalfToFloat(w));
    }
    std::printf("Skipped target (%s, never copied back): %#llx %u bytes, %ux%u x%u, %s, %s; guest "
                "bytes as halves:%s\n",
                kind, static_cast<unsigned long long>(info.guest_address), info.guest_size,
                info.size.width, info.size.height, info.resources.layers,
                vk::to_string(info.pixel_format).c_str(),
                std::string(AmdGpu::NameOf(info.tile_mode)).c_str(),
                read ? halves.c_str() : " (unreadable)");
}
} // namespace

ImageView& TextureCache::FindTexture(ImageId image_id, const ImageDesc& desc, ViewMemo* memo,
                                     bool refresh) {
    Image& image = slot_images[image_id];
    if (desc.type == BindingType::Storage) {
        image.flags |= ImageFlagBits::GpuModified;
        if (Watched(image.info)) {
            const auto& range = desc.view_info.range;
            WatchNote(image, "storage write", std::format(" mips {}+{} layers {}+{}", range.base.level,
                                                          range.extent.levels, range.base.layer,
                                                          range.extent.layers));
        }
        if (readback_linear_images && (!image.info.props.is_tiled || image.info.size.width <= 8 ||
                                       (tiny_target_readback && IsTinyTarget(image.info))) &&
            image.info.guest_address != 0) {
            std::unique_lock lk{download_images_mutex};
            download_images.emplace(image_id);
        } else if (SkippedTargetLog()) {
            LogSkippedTarget(image.info, "storage");
        }
    }
    if (refresh) {
        UpdateImage(image_id);
    }
    if (desc.type == BindingType::Storage) {
        PublishAliasWrite(image_id);
    }
    if (memo && !Gow3Toggle::Disabled(Gow3Toggle::TextureViewMemo)) {
        if (memo->image_id == image_id && memo->backing == image.backing && memo->view_id) {
            return slot_image_views[memo->view_id];
        }
        ImageView& view = image.FindView(desc.view_info);
        const auto& ids = image.backing->image_view_ids;
        const u32 last = image.backing->last_view;
        memo->image_id = image_id;
        memo->backing = image.backing;
        memo->view_id = last < ids.size() && &slot_image_views[ids[last]] == &view ? ids[last]
                                                                                    : ids.back();
        return view;
    }
    return image.FindView(desc.view_info);
}

ImageView& TextureCache::FindRenderTarget(ImageId image_id, const ImageDesc& desc) {
    if (Watched(slot_images[image_id].info)) {
        const auto& range = desc.view_info.range;
        WatchNote(slot_images[image_id], "render target", std::format(
            " mips {}+{} layers {}+{}", range.base.level, range.extent.levels, range.base.layer,
            range.extent.layers));
    }
    DiscardUndefinedContents(image_id);
    Image& image = slot_images[image_id];
    image.flags |= ImageFlagBits::GpuModified;
    if (readback_linear_images && (!image.info.props.is_tiled || image.info.size.width <= 8 ||
                                   (tiny_target_readback && IsTinyTarget(image.info)))) {
        std::unique_lock lk{download_images_mutex};
        download_images.emplace(image_id);
    } else if (SkippedTargetLog()) {
        LogSkippedTarget(image.info, "render target");
    }
    image.usage.render_target = 1u;
    UpdateImage(image_id);
    PublishAliasWrite(image_id);

    // Register meta data for this color buffer
    if (desc.info.meta_info.cmask_addr) {
        surface_metas.emplace(desc.info.meta_info.cmask_addr,
                              MetaDataInfo{.type = MetaType::CMask});
        image.info.meta_info.cmask_addr = desc.info.meta_info.cmask_addr;
    }

    if (desc.info.meta_info.fmask_addr) {
        surface_metas.emplace(desc.info.meta_info.fmask_addr,
                              MetaDataInfo{.type = MetaType::FMask});
        image.info.meta_info.fmask_addr = desc.info.meta_info.fmask_addr;
    }

    return image.FindView(desc.view_info, false);
}

ImageView& TextureCache::FindDepthTarget(ImageId image_id, const ImageDesc& desc) {
    DiscardUndefinedContents(image_id);
    Image& image = slot_images[image_id];
    image.flags |= ImageFlagBits::GpuModified;
    image.usage.depth_target = 1u;
    UpdateImage(image_id);

    // Register meta data for this depth buffer
    if (desc.info.meta_info.htile_addr) {
        surface_metas.emplace(desc.info.meta_info.htile_addr,
                              MetaDataInfo{.type = MetaType::HTile,
                                           .clear_mask = image.info.meta_info.htile_clear_mask});
        image.info.meta_info.htile_addr = desc.info.meta_info.htile_addr;
    }

    // If there is a stencil attachment, link depth and stencil.
    if (desc.info.stencil_addr != 0) {
        // Only an 8-bit image (or the format-less placeholder created below) stands for the
        // stencil plane: a color texture that later starts at this address must not be
        // redirected to the depth-stencil image (Uncharted port: GPU page fault).
        const ImageId depth_id = image_id;
        ImageId stencil_id{};
        ForEachImageInRegion(desc.info.stencil_addr, desc.info.stencil_size,
                             [&](ImageId candidate_id, Image& candidate) {
                                 if (candidate_id != depth_id &&
                                     candidate.info.guest_address == desc.info.stencil_addr &&
                                     IsStencilProxyCandidate(candidate)) {
                                     stencil_id = candidate_id;
                                 }
                             });
        if (!stencil_id) {
            ImageInfo info{};
            info.guest_address = desc.info.stencil_addr;
            info.guest_size = desc.info.stencil_size;
            info.size = desc.info.size;
            stencil_id = slot_images.insert(instance, runtime, slot_image_views, info);
            RegisterImage(stencil_id);
        }
        // The insert above moves every image when the slot vector grows: `image` fetched again.
        Image& depth_image = slot_images[image_id];
        Image& stencil_image = slot_images[stencil_id];
        TouchImage(stencil_image);
        stencil_image.AssociateDepth(image_id, depth_image.image_uid);
        return depth_image.FindView(desc.view_info, false);
    }

    return image.FindView(desc.view_info, false);
}

void TextureCache::DiscardUndefinedContents(ImageId image_id) {
    // Without `mutex`: some callers of FindRenderTarget / FindDepthTarget hold it (std::mutex).
    // The flags are changed atomically, as UpdateImage's fast path reads them.
    Image& image = slot_images[image_id];
    std::atomic_ref<u32> flags{reinterpret_cast<u32&>(image.flags)};
    constexpr u32 Undefined = static_cast<u32>(ImageFlagBits::Undefined);
    constexpr u32 Dirty = static_cast<u32>(ImageFlagBits::Dirty);
    if (flags.load(std::memory_order_acquire) & Undefined &&
        flags.fetch_and(~(Undefined | Dirty), std::memory_order_acq_rel) & Undefined) {
        CACHE_STAT(undefined_targets, 1);
        CACHE_STAT(undefined_target_bytes, image.info.guest_size);
    }
}

void TextureCache::DiscardClearedContents(ImageId image_id) {
    // Same locking as DiscardUndefinedContents.
    Image& image = slot_images[image_id];
    std::atomic_ref<u32> flags{reinterpret_cast<u32&>(image.flags)};
    constexpr u32 Dirty = static_cast<u32>(ImageFlagBits::Dirty);
    if (flags.load(std::memory_order_acquire) & Dirty &&
        flags.fetch_and(~Dirty, std::memory_order_acq_rel) & Dirty) {
        CACHE_STAT(cleared_targets, 1);
        CACHE_STAT(cleared_target_bytes, image.info.guest_size);
    }
}

void TextureCache::RefreshImage(Image& image, std::source_location caller) {
    // A first use other than as a render / depth target: the contents count from here.
    const bool was_undefined = True(image.flags & ImageFlagBits::Undefined);
    image.flags &= ~ImageFlagBits::Undefined;
    if (False(image.flags & ImageFlagBits::Dirty) || image.info.num_samples > 1) {
        return;
    }
    if (Watched(image.info)) {
        std::string_view file{caller.file_name()};
        file = file.substr(file.find_last_of("/\\") + 1);
        WatchNote(image, "uploaded from guest memory",
                  std::format(" ({:#x} bytes{}, from {}:{})", image.info.guest_size,
                              was_undefined ? ", was undefined" : "", file, caller.line()));
    }
    CACHE_STAT(refreshes, 1);
    CACHE_STAT(refresh_bytes, image.info.guest_size);
    if (CacheStatsEnabled()) {
        // Which images are uploaded again and why (MediEvil: ~100 MB in one frame when a torch
        // comes into view, for 3 new images).
        const ImageInfo& info = image.info;
        const char* why = True(image.flags & ImageFlagBits::CpuDirty)        ? "CPU dirty"
                          : True(image.flags & ImageFlagBits::GpuDirty)      ? "GPU dirty (buffer)"
                          : True(image.flags & ImageFlagBits::MaybeCpuDirty) ? "maybe CPU dirty"
                                                                             : "?";
        // Also: an image made where an unmap freed images (C3: its memory is undefined), and the
        // caller of this first use (sampled, storage, copy, ...).
        std::string_view file{caller.file_name()};
        file = file.substr(file.find_last_of("/\\") + 1);
        char key[300];
        std::snprintf(key, sizeof(key), "%#llx %ux%ux%u %s L%u/%u%s%s%s, %s%s%s, from %.*s:%u",
                      (unsigned long long)info.guest_address, info.size.width, info.size.height,
                      info.size.depth, vk::to_string(info.pixel_format).c_str(), info.resources.levels,
                      info.resources.layers, image.usage.render_target ? " RT" : "",
                      image.usage.depth_target ? " depth" : "", image.usage.texture ? " tex" : "", why,
                      True(image.flags & ImageFlagBits::GpuModified) ? ", GPU modified" : "",
                      was_undefined ? ", undefined (unmap-freed)" : "", int(file.size()), file.data(),
                      caller.line());
        std::scoped_lock lock{refresh_sites_mutex};
        auto& site = refresh_sites[key];
        ++site.first;
        site.second += info.guest_size;
    }
    ResolveReadbacks(image.info.guest_address, image.info.guest_size, true,
                     ReadbackSource::ImageRefresh);
    Gow3Stats::Timer timer{Gow3Stats::t_refresh};

    RENDERER_TRACE;
    TRACE_HINT(fmt::format("{:x}:{:x}", image.info.guest_address, image.info.guest_size));

    if (True(image.flags & ImageFlagBits::MaybeCpuDirty) &&
        False(image.flags & ImageFlagBits::CpuDirty)) {
        // The image size should be less than page size to be considered MaybeCpuDirty
        // So this calculation should be very uncommon and reasonably fast
        // For now we'll just check up to 64 first pixels
        const auto addr = std::bit_cast<u8*>(image.info.guest_address);
        const u32 w = std::min(image.info.size.width, u32(8));
        const u32 h = std::min(image.info.size.height, u32(8));

        const u32 s_w = image.info.props.is_block ? Common::DivCeil(w, 4u) : w;
        const u32 s_h = image.info.props.is_block ? Common::DivCeil(h, 4u) : h;
        const u32 size = s_w * s_h * (image.info.num_bits / 8);
        const u64 hash = XXH3_64bits(addr, size);
        if (image.hash == hash) {
            image.flags &= ~ImageFlagBits::MaybeCpuDirty;
            return;
        }
        image.hash = hash;
    }

    const u32 num_layers = image.info.resources.layers;
    const u32 num_mips = image.info.resources.levels;
    const bool is_gpu_modified = True(image.flags & ImageFlagBits::GpuModified);
    const bool is_gpu_dirty = True(image.flags & ImageFlagBits::GpuDirty);

    Gow3Stats::image_upload_bytes.fetch_add(image.info.guest_size, std::memory_order_relaxed);
    boost::container::small_vector<vk::BufferImageCopy, 14> image_copies;
    for (u32 m = 0; m < num_mips; m++) {
        const u32 width = std::max(image.info.size.width >> m, 1u);
        const u32 height = std::max(image.info.size.height >> m, 1u);
        const u32 depth =
            image.info.props.is_volume ? std::max(image.info.size.depth >> m, 1u) : 1u;
        const auto [mip_size, mip_pitch, mip_height, mip_offset] = image.info.mips_layout[m];

        // Protect GPU modified resources from accidental CPU reuploads.
        if (is_gpu_modified && !is_gpu_dirty) {
            const u8* addr = std::bit_cast<u8*>(image.info.guest_address);
            const u64 hash = XXH3_64bits(addr + mip_offset, mip_size);
            if (image.mip_hashes[m] == hash) {
                continue;
            }
            image.mip_hashes[m] = hash;
        }

        const u32 extent_width = mip_pitch ? std::min(mip_pitch, width) : width;
        const u32 extent_height = mip_height ? std::min(mip_height, height) : height;
        image_copies.push_back({
            .bufferOffset = mip_offset,
            .bufferRowLength = mip_pitch,
            .bufferImageHeight = mip_height,
            .imageSubresource{
                .aspectMask = image.aspect_mask & ~vk::ImageAspectFlagBits::eStencil,
                .mipLevel = m,
                .baseArrayLayer = 0,
                .layerCount = num_layers,
            },
            .imageOffset = {0, 0, 0},
            .imageExtent = {extent_width, extent_height, depth},
        });
    }

    if (image_copies.empty()) {
        image.flags &= ~ImageFlagBits::Dirty;
        return;
    }

    scheduler.EndRendering();

    const auto [in_buffer, in_offset] =
        buffer_cache.ObtainBufferForImage(image.info.guest_address, image.info.guest_size);
    const auto [buffer, offset] = tile_manager.DetileImage(in_buffer, in_offset, image.info);
    for (auto& copy : image_copies) {
        copy.bufferOffset += offset;
    }

    runtime.UploadImage(&image, buffer, image_copies);
}

vk::Sampler TextureCache::GetSampler(const AmdGpu::Sampler& sampler,
                                     AmdGpu::BorderColorBuffer border_color_base,
                                     const bool is_depth, float extra_lod_bias) {
    // Compare and plain uses of one S# need separate samplers; so do extra LOD biases.
    const u64 hash = HashCombine(HashCombine(XXH3_64bits(&sampler, sizeof(sampler)), is_depth),
                                 u64(std::bit_cast<u32>(extra_lod_bias)));

    std::scoped_lock lock{samplers_mutex};
    const auto [it, new_sampler] = samplers.try_emplace(hash, instance, sampler, border_color_base,
                                                        is_depth, extra_lod_bias);
    if (new_sampler) {
        samplers.at(hash).lru_id = sampler_lru_cache.Insert(hash, gc_tick);
    } else {
        sampler_lru_cache.Touch(it->second.lru_id, gc_tick);
    }

    return it->second.Handle();
}

void TextureCache::NoteImageChurn(bool registered, u32 line, const ImageInfo& info) {
    struct Churn {
        std::unordered_map<u32, u64> frees_by_line;
        std::unordered_map<u64, u64> registrations; // size << 32 | format
        u64 registered = 0, freed = 0;
        std::chrono::steady_clock::time_point report = std::chrono::steady_clock::now();
    };
    static Churn churn;
    static std::mutex churn_mutex;
    std::scoped_lock lk{churn_mutex};
    if (registered) {
        ++churn.registered;
        ++churn.registrations[u64(info.guest_size) << 32 | u32(info.pixel_format)];
    } else {
        ++churn.freed;
        ++churn.frees_by_line[line];
    }
    const auto now = std::chrono::steady_clock::now();
    if (now - churn.report < std::chrono::seconds(5)) {
        return;
    }
    std::printf("Image churn (5 s): %llu registered, %llu freed; frees by texture_cache.cpp line:",
                (unsigned long long)churn.registered, (unsigned long long)churn.freed);
    for (const auto& [l, n] : churn.frees_by_line) {
        std::printf(" %u=%llu", l, (unsigned long long)n);
    }
    std::vector<std::pair<u64, u64>> top(churn.registrations.begin(), churn.registrations.end());
    std::sort(top.begin(), top.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
    std::printf("\n  most registered (size, format):");
    for (size_t i = 0; i < std::min<size_t>(top.size(), 8); ++i) {
        std::printf(" %llu KB %s x%llu;", (unsigned long long)((top[i].first >> 32) >> 10),
                    vk::to_string(vk::Format(u32(top[i].first))).c_str(),
                    (unsigned long long)top[i].second);
    }
    std::printf("\n");
    std::fflush(stdout);
    churn = Churn{};
    churn.report = now;
}

void TextureCache::RegisterImage(ImageId image_id) {
    Gow3Stats::images_registered.fetch_add(1, std::memory_order_relaxed);
    CACHE_STAT(registered, 1);
    Image& image = slot_images[image_id];
    if (image_stats) {
        NoteImageChurn(true, 0, image.info);
    }
    ASSERT_MSG(False(image.flags & ImageFlagBits::Registered),
               "Trying to register an already registered image");
    image.flags |= ImageFlagBits::Registered;
    // Region stamps first, then the generation: a memo lookup (BindTexturesFromSetIn) reads the
    // generation, then the stamps, and confirms a set at the generation it read. Bumped the other
    // way round, a lookup between the two could confirm a set at the new generation with the old
    // stamps, and later lookups at that generation skip the stamp check.
    BumpRegions(image.info.guest_address, image.info.guest_size);
    ++registry_generation;
    total_used_memory += Common::AlignUp(image.info.guest_size, 1024);
    image.lru_id = lru_cache.Insert(image_id, gc_tick);
    image.lru_touched_tick = gc_tick;

    AliasState* state{};
    u32 aliases{};
    bool inserted{};
    ForEachAlias(image, [&](ImageId candidate_id, Image& candidate) {
        if (!state) {
            auto result = alias_states.try_emplace(image.info.guest_address);
            state = std::addressof(result.first.value());
            inserted = result.second;
        }
        ++aliases;
        candidate.flags |= ImageFlagBits::Aliased;
        if (!IsLiveImage(state->backing, state->backing_uid) ||
            slot_images[state->backing].alias_generation < candidate.alias_generation) {
            state->backing = candidate_id, state->backing_uid = candidate.image_uid;
        }
    });
    if (state) {
        image.flags |= ImageFlagBits::Aliased;
        state->members = inserted ? aliases + 1 : state->members + 1;
    }

    ForEachPage(image.info.guest_address, image.info.guest_size,
                [this, image_id](u64 page) { page_table[page].push_back(image_id); });
}

void TextureCache::UnregisterImage(ImageId image_id) {
    CACHE_STAT(unregistered, 1);
    Image& image = slot_images[image_id];
    ASSERT_MSG(True(image.flags & ImageFlagBits::Registered),
               "Trying to unregister an already unregistered image");
    image.flags &= ~ImageFlagBits::Registered;
    BumpRegions(image.info.guest_address, image.info.guest_size); // before the generation (RegisterImage)
    ++registry_generation;
    lru_cache.Free(image.lru_id);
    total_used_memory -= Common::AlignUp(image.info.guest_size, 1024);
    ForEachPage(image.info.guest_address, image.info.guest_size, [this, image_id](u64 page) {
        const auto page_it = page_table.find(page);
        if (page_it == nullptr) {
            UNREACHABLE_MSG("Unregistering unregistered page=0x{:x}", page << PageShift);
            return;
        }
        auto& image_ids = *page_it;
        const auto vector_it = std::ranges::find(image_ids, image_id);
        if (vector_it == image_ids.end()) {
            ASSERT_MSG(false, "Unregistering unregistered image in page=0x{:x}", page << PageShift);
            return;
        }
        image_ids.erase(vector_it);
    });
}

void TextureCache::TrackImage(ImageId image_id) {
    auto& image = slot_images[image_id];
    if (!(image.flags & ImageFlagBits::Registered)) {
        return;
    }
    const auto image_begin = image.info.guest_address;
    const auto image_end = image.info.guest_address + image.info.guest_size;
    if (image_begin == image.track_addr && image_end == image.track_addr_end) {
        return;
    }

    if (!image.IsTracked()) {
        // Re-track the whole image
        image.track_addr = image_begin;
        image.track_addr_end = image_end;
        tracker.UpdatePageWatchers<1>(image_begin, image.info.guest_size);
    } else {
        if (image_begin < image.track_addr) {
            TrackImageHead(image_id);
        }
        if (image.track_addr_end < image_end) {
            TrackImageTail(image_id);
        }
    }
}

void TextureCache::TrackImageHead(ImageId image_id) {
    auto& image = slot_images[image_id];
    if (!(image.flags & ImageFlagBits::Registered)) {
        return;
    }
    const auto image_begin = image.info.guest_address;
    if (image_begin == image.track_addr) {
        return;
    }
    ASSERT(image.track_addr != 0 && image_begin < image.track_addr);
    const auto size = image.track_addr - image_begin;
    image.track_addr = image_begin;
    tracker.UpdatePageWatchers<1>(image_begin, size);
}

void TextureCache::TrackImageTail(ImageId image_id) {
    auto& image = slot_images[image_id];
    if (!(image.flags & ImageFlagBits::Registered)) {
        return;
    }
    const auto image_end = image.info.guest_address + image.info.guest_size;
    if (image_end == image.track_addr_end) {
        return;
    }
    ASSERT(image.track_addr_end != 0 && image.track_addr_end < image_end);
    const auto addr = image.track_addr_end;
    const auto size = image_end - image.track_addr_end;
    image.track_addr_end = image_end;
    tracker.UpdatePageWatchers<1>(addr, size);
}

void TextureCache::UntrackImage(ImageId image_id) {
    auto& image = slot_images[image_id];
    if (!image.IsTracked()) {
        return;
    }
    const auto addr = image.track_addr;
    const auto size = image.track_addr_end - image.track_addr;
    image.track_addr = 0;
    image.track_addr_end = 0;
    if (size != 0) {
        tracker.UpdatePageWatchers<false>(addr, size);
    }
}

void TextureCache::UntrackImageHead(ImageId image_id) {
    auto& image = slot_images[image_id];
    const auto image_begin = image.info.guest_address;
    if (!image.IsTracked() || image_begin < image.track_addr) {
        return;
    }
    const auto addr = tracker.GetNextPageAddr(image_begin);
    const auto size = addr - image_begin;
    image.track_addr = addr;
    if (image.track_addr == image.track_addr_end) {
        // This image spans only 2 pages and both are modified,
        // but the image itself was not directly affected.
        // Cehck its hash later.
        MarkAsMaybeDirty(image_id, image);
    }
    tracker.UpdatePageWatchers<false>(image_begin, size);
}

void TextureCache::UntrackImageTail(ImageId image_id) {
    auto& image = slot_images[image_id];
    const auto image_end = image.info.guest_address + image.info.guest_size;
    if (!image.IsTracked() || image.track_addr_end < image_end) {
        return;
    }
    ASSERT(image.track_addr_end != 0);
    const auto addr = tracker.GetPageAddr(image_end);
    const auto size = image_end - addr;
    image.track_addr_end = addr;
    if (image.track_addr == image.track_addr_end) {
        // This image spans only 2 pages and both are modified,
        // but the image itself was not directly affected.
        // Cehck its hash later.
        MarkAsMaybeDirty(image_id, image);
    }
    tracker.UpdatePageWatchers<false>(addr, size);
}

void TextureCache::GarbageCollectImages() {
    Gow3Stats::Timer gc_timer{Gow3Stats::t_gc};
    if (instance.CanReportMemoryUsage()) {
        total_used_memory = instance.GetDeviceMemoryUsage();
        // gow3: on integrated GPUs (Steam Deck) the usage covers system-memory heaps holding
        // much more than images (buffers backing guest memory), and the startup budget left
        // ~1 GB after its 8 GB system reserve: usage stayed above the critical mark, so the
        // collector evicted images used two or three frames ago on every submission and wrote
        // GPU-written ones back. Compare with the driver's current budget instead.
        // GOW3_GC_BUDGET_MB=N: this rule with a fixed budget on any GPU (tests on a desktop).
        // Discrete GPUs use it too (from the Uncharted port, RTX 5060 Laptop, 8 GB): the startup
        // rule put the trigger at 0 and the pressure mark at ~40% of an 8 GB card, and the usage
        // counts the buffer arenas in device memory, so images still in use were evicted and
        // registered again with all their uploads: 40-75 ms frames while moving through a
        // level (stutter on RTX 3060/4060/5060). GOW3_GC_STARTUP_RULE=1: the old startup rule.
        static const u64 forced_budget = [] {
            const char* env = std::getenv("GOW3_GC_BUDGET_MB");
            return env ? std::strtoull(env, nullptr, 10) << 20 : 0;
        }();
        static const bool startup_rule = [] {
            const char* env = std::getenv("GOW3_GC_STARTUP_RULE");
            return env && env[0] == '1';
        }();
        if (instance.IsIntegrated() || forced_budget || !startup_rule) {
            const u64 budget = forced_budget ? forced_budget : instance.GetDeviceMemoryBudgetNow();
            if (budget != 0) {
                trigger_gc_memory = budget / 10 * 7;
                pressure_gc_memory = budget / 100 * 85;
                critical_gc_memory = budget / 100 * 95;
            }
        }
    }
    if (total_used_memory < trigger_gc_memory) {
        return;
    }
    std::scoped_lock lock{mutex};
    bool pressured = false;
    bool aggresive = false;
    u64 ticks_to_destroy = 0;
    size_t num_deletions = 0;

    const auto configure = [&](bool allow_aggressive) {
        pressured = total_used_memory >= pressure_gc_memory;
        aggresive = allow_aggressive && total_used_memory >= critical_gc_memory;
        ticks_to_destroy = aggresive ? 160 : pressured ? 80 : 16;
        ticks_to_destroy = std::min(ticks_to_destroy, gc_tick);
        num_deletions = aggresive ? 40 : pressured ? 20 : 10;
    };
    const auto clean_up = [&](ImageId image_id) {
        if (num_deletions == 0) {
            return true;
        }
        --num_deletions;
        auto& image = slot_images[image_id];
        ResolveReadbacks(image.info.guest_address, image.info.guest_size, true,
                         ReadbackSource::CacheCleanup);
        const bool download = image.SafeToDownload();
        const bool tiled = image.info.IsTiled();
        if (tiled && download) {
            // This is a workaround for now. We can't handle non-linear image downloads.
            return false;
        }
        if (download && !pressured) {
            return false;
        }
        if (download) {
            // gow3: synchronously, while the image still protects its pages. A deferred
            // write-back landed after FreeImage had unprotected them, over whatever the game
            // had meanwhile stored there (e.g. its heap after unloading an area).
            DownloadImageMemory(image_id, true);
            ++gc_downloads;
        }
        ++gc_evictions;
        FreeImage(image_id);
        if (total_used_memory < critical_gc_memory) {
            if (aggresive) {
                num_deletions >>= 2;
                aggresive = false;
                return false;
            }
            if (pressured && total_used_memory < pressure_gc_memory) {
                num_deletions >>= 1;
                pressured = false;
            }
        }
        return false;
    };

    // Try to remove anything old enough and not high priority.
    configure(false);
    lru_cache.ForEachItemBelow(gc_tick - ticks_to_destroy, clean_up);

    if (total_used_memory >= critical_gc_memory) {
        // If we are still over the critical limit, run an aggressive GC
        configure(true);
        lru_cache.ForEachItemBelow(gc_tick - ticks_to_destroy, clean_up);
    }
    // gow3: evictions under memory pressure, at most every 5 s (GOW3_FRAME_STATS or not).
    static const bool frame_stats = std::getenv("GOW3_FRAME_STATS") != nullptr;
    if (pressured || gc_downloads != 0 || (frame_stats && gc_evictions != 0)) {
        const auto now = std::chrono::steady_clock::now();
        if (now - gc_report_time >= std::chrono::seconds(5)) {
            std::printf("Texture cache: %s, %llu MiB used (trigger %llu, pressure %llu, critical %llu "
                        "MiB): %llu images evicted, %llu written back since the last report\n",
                        pressured ? "memory pressure" : "collecting",
                        (unsigned long long)(total_used_memory >> 20),
                        (unsigned long long)(trigger_gc_memory >> 20),
                        (unsigned long long)(pressure_gc_memory >> 20),
                        (unsigned long long)(critical_gc_memory >> 20),
                        (unsigned long long)gc_evictions, (unsigned long long)gc_downloads);
            gc_report_time = now;
            gc_evictions = gc_downloads = 0;
        }
    }
}

void TextureCache::GarbageCollectSamplers() {
    total_used_samplers = samplers.size();
    if (total_used_samplers < trigger_gc_samplers) {
        return;
    }
    std::scoped_lock lock{samplers_mutex};
    bool pressured = false;
    bool aggresive = false;
    u64 ticks_to_destroy = 0;
    size_t num_deletions = 0;

    const auto configure = [&](bool allow_aggressive) {
        pressured = total_used_samplers >= pressure_gc_samplers;
        aggresive = allow_aggressive && total_used_samplers >= critical_gc_samplers;
        ticks_to_destroy = aggresive ? 160 : pressured ? 80 : 16;
        ticks_to_destroy = std::min(ticks_to_destroy, gc_tick);
        num_deletions = aggresive ? 40 : pressured ? 20 : 10;
    };
    const auto clean_up = [&](u64 hash) {
        if (num_deletions == 0) {
            return true;
        }
        --num_deletions;
        auto& sampler = samplers.at(hash);
        const size_t lru_id = sampler.lru_id;
        // Submitted commands may still use the sampler: destroyed once the GPU passed them.
        auto doomed = std::make_shared<Sampler>(std::move(sampler));
        scheduler.DeferOperation([doomed] {});
        samplers.erase(hash);
        sampler_lru_cache.Free(lru_id);
        return false;
    };

    // Try to remove anything old enough and not high priority.
    configure(false);
    sampler_lru_cache.ForEachItemBelow(gc_tick - ticks_to_destroy, clean_up);

    if (total_used_samplers >= critical_gc_samplers) {
        // If we are still over the critical limit, run an aggressive GC
        configure(true);
        sampler_lru_cache.ForEachItemBelow(gc_tick - ticks_to_destroy, clean_up);
    }
}

void TextureCache::RunGarbageCollector() {
    SCOPE_EXIT {
        ++gc_tick;
    };

    if (parked_count.load(std::memory_order_relaxed)) {
        std::scoped_lock lock{mutex};
        EvictParkedImages(false);
    }
    GarbageCollectImages();
    GarbageCollectSamplers();
    if (CacheStatsEnabled()) {
        static auto last = std::chrono::steady_clock::now();
        static u64 live_registered = 0;
        const auto now = std::chrono::steady_clock::now();
        if (now - last >= std::chrono::seconds(5)) {
            last = now;
            const auto take = [](std::atomic<u64>& v) { return v.exchange(0, std::memory_order_relaxed); };
            const u64 reg = take(cache_stats.registered), unreg = take(cache_stats.unregistered);
            live_registered += reg - unreg;
            const u64 slow = take(cache_stats.slow_lookups), candidates = take(cache_stats.slow_candidates);
            std::printf("Texture cache stats (5 s): %llu images registered (+%llu -%llu), %zu slots, "
                        "%zu alias groups; lookups fast %llu slow %llu (avg %.1f candidates); "
                        "overlaps %llu, expands %llu; uploads %llu (%.1f MiB); alias syncs %llu, "
                        "copies %llu, volume views %llu; GPU memory %llu MiB, collector trigger %llu / pressure %llu MiB, "
                        "tick %llu\n",
                        (unsigned long long)live_registered, (unsigned long long)reg,
                        (unsigned long long)unreg, slot_images.size(), alias_states.size(),
                        (unsigned long long)take(cache_stats.fast_lookups), (unsigned long long)slow,
                        slow ? double(candidates) / slow : 0.0,
                        (unsigned long long)take(cache_stats.overlaps),
                        (unsigned long long)take(cache_stats.expands),
                        (unsigned long long)take(cache_stats.refreshes),
                        take(cache_stats.refresh_bytes) / 1048576.0,
                        (unsigned long long)take(cache_stats.alias_syncs),
                        (unsigned long long)take(cache_stats.alias_copies),
                        (unsigned long long)take(cache_stats.volume_views),
                        (unsigned long long)(total_used_memory >> 20),
                        (unsigned long long)(trigger_gc_memory >> 20),
                        (unsigned long long)(pressure_gc_memory >> 20), (unsigned long long)gc_tick);
            std::scoped_lock lock{mutex};
            std::vector<std::pair<u64, std::string>> top;
            for (const auto& [key, count] : expand_sites) top.emplace_back(count, key);
            std::sort(top.begin(), top.end(), std::greater<>{});
            for (size_t i = 0; i < top.size() && i < 8; ++i) {
                std::printf("  expand %6llu x  %s\n", (unsigned long long)top[i].first, top[i].second.c_str());
            }
            expand_sites.clear();
            const u64 frees = take(cache_stats.unmap_frees), free_bytes = take(cache_stats.unmap_free_bytes),
                      recreated = take(cache_stats.recreated), recreated_bytes = take(cache_stats.recreated_bytes);
            std::printf("  freed by guest unmaps %llu (%.1f MiB); created again within ~30 s %llu (%.1f MiB)\n",
                        (unsigned long long)frees, free_bytes / 1048576.0, (unsigned long long)recreated,
                        recreated_bytes / 1048576.0);
            const u64 undefined = take(cache_stats.undefined_targets),
                      undefined_bytes = take(cache_stats.undefined_target_bytes);
            std::printf("  targets made where an unmap freed images, not uploaded first: %llu (%.1f MiB)\n",
                        (unsigned long long)undefined, undefined_bytes / 1048576.0);
            const u64 cleared = take(cache_stats.cleared_targets),
                      cleared_bytes = take(cache_stats.cleared_target_bytes);
            std::printf("  dirty targets cleared by their pass, not uploaded first: %llu (%.1f MiB)\n",
                        (unsigned long long)cleared, cleared_bytes / 1048576.0);
            const u64 reused = take(cache_stats.gpu_freed_reused),
                      reused_bytes = take(cache_stats.gpu_freed_reused_bytes);
            std::printf("  other new images flagged undefined (GPU-freed or new): %llu (%.1f MiB)\n",
                        (unsigned long long)reused, reused_bytes / 1048576.0);
            // (under mutex, taken above)
            std::printf("  unmapped targets parked %llu, taken back %llu, evicted %llu (now %zu, "
                        "%.1f MiB); new images made in %.2f ms, max %.3f ms\n",
                        (unsigned long long)take(cache_stats.parked),
                        (unsigned long long)take(cache_stats.parked_reused),
                        (unsigned long long)take(cache_stats.parked_evicted), parked_images.size(),
                        parked_bytes / 1048576.0, create_ns / 1e6, create_max_ns / 1e6);
            create_ns = create_max_ns = 0;
            std::printf("  end of pipe with readback work (host copy wait) %llu, without (skipped) %llu\n",
                        (unsigned long long)take(cache_stats.eop_waits),
                        (unsigned long long)take(cache_stats.eop_skips));
            top.clear();
            for (const auto& [key, count] : create_sites) top.emplace_back(count, key);
            std::sort(top.begin(), top.end(), std::greater<>{});
            for (size_t i = 0; i < top.size() && i < 8; ++i) {
                std::printf("  create %6llu x  %s\n", (unsigned long long)top[i].first, top[i].second.c_str());
            }
            create_sites.clear();
            {
                std::scoped_lock refresh_lock{refresh_sites_mutex};
                std::vector<std::tuple<u64, u64, std::string>> refreshed;
                for (const auto& [key, site] : refresh_sites) refreshed.emplace_back(site.second, site.first, key);
                std::sort(refreshed.begin(), refreshed.end(), std::greater<>{});
                for (size_t i = 0; i < refreshed.size() && i < 8; ++i) {
                    std::printf("  refresh %5llu x %8.1f MiB  %s\n", (unsigned long long)std::get<1>(refreshed[i]),
                                std::get<0>(refreshed[i]) / 1048576.0, std::get<2>(refreshed[i]).c_str());
                }
                refresh_sites.clear();
            }
            ++stats_reports;
            std::erase_if(unmapped_recently, [&](const auto& entry) { return stats_reports - entry.second > 6; });
        }
    }
}

void TextureCache::TouchImage(const Image& image) {
    if (image.lru_touched_tick == gc_tick) {
        return;
    }
    image.lru_touched_tick = gc_tick;
    lru_cache.Touch(image.lru_id, gc_tick);
}

void TextureCache::DeleteImage(ImageId image_id) {
    Image& image = slot_images[image_id];
    CancelReadbacks(image.info.guest_address, image.info.guest_size);
    ASSERT_MSG(!image.IsTracked(), "Image was not untracked");
    ASSERT_MSG(False(image.flags & ImageFlagBits::Registered), "Image was not unregistered");

    const auto alias_it = True(image.flags & ImageFlagBits::Aliased)
                              ? alias_states.find(image.info.guest_address)
                              : alias_states.end();
    if (alias_it != alias_states.end()) {
        AliasState& state = alias_it.value();
        // ponytail: no copy here (DeleteImage also runs on guest threads through UnmapMemory,
        // which must not record commands); the aliases then re-read guest memory.
        if ((state.writer == image_id && state.writer_uid == image.image_uid) ||
            (state.backing == image_id && state.backing_uid == image.image_uid)) {
            state.ResetAuthority();
        }
        if (state.members <= 2) {
            ForEachAlias(image, [](ImageId, Image& candidate) {
                candidate.flags &= ~ImageFlagBits::Aliased;
            });
            alias_states.erase(alias_it);
        } else {
            --state.members;
        }
    }

    // Remove any registered meta areas.
    const auto& meta_info = image.info.meta_info;
    if (meta_info.cmask_addr) {
        surface_metas.erase(meta_info.cmask_addr);
    }
    if (meta_info.fmask_addr) {
        surface_metas.erase(meta_info.fmask_addr);
    }
    if (meta_info.htile_addr) {
        surface_metas.erase(meta_info.htile_addr);
    }

    {
        std::unique_lock lk{download_images_mutex};
        if (download_images.contains(image_id)) {
            download_images.erase(image_id);
        }
    }

    // Reclaim image and any image views it references.
    scheduler.DeferOperation([this, image_id] {
        Image& image = slot_images[image_id];
        for (auto& backing : image.backing_images) {
            for (const ImageViewId image_view_id : backing.image_view_ids) {
                slot_image_views.erase(image_view_id);
            }
        }
        slot_images.erase(image_id);
    });
}

} // namespace VideoCore
