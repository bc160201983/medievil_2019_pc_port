// SPDX-License-Identifier: GPL-2.0-or-later
// gow3: small read-only guest buffers (constants) copied by the GPU command thread (stage A of
// the draw pipeline, vk_draw_pipe.h) into this ring, so the draw recording thread only binds
// them. Positions are monotonic byte counts; a region is reused once the submission that
// recorded its last draw has completed on the GPU. Stage B stamps each packet it records with
// the submission tick (Stamp), stage A retires stamps whose tick the GPU has passed.

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <string>

#include "gow3_async_log.h"
#include "gow3_toggles.h"
#include "common/assert.h"
#include "common/types.h"
#include "video_core/buffer_cache/buffer.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"

namespace Vulkan {

class ConstantRing {
public:
    /// gow3: 96 MiB (was 32): ring reuse (GOW3_RING_REUSE) holds every copy of a guest command
    /// buffer until the GPU has completed it; with stats off stage A runs further ahead and the
    /// smaller ring ran full (t-nostats bisect: bumps 210 <-> 160 FPS with reuse on, flat off).
    /// VRAM through the BAR: 64 MiB more of a 24 GB card.
    /// GOW3_CONSTANT_RING_MB: another size for A/B runs (t-pan5: 9-10 ms GPU-starved frames after
    /// the change from 32 MiB).
    static inline const u64 Capacity = [] {
        const char* env = std::getenv("GOW3_CONSTANT_RING_MB");
        const u64 mb = env ? std::strtoull(env, nullptr, 10) : 96;
        return std::clamp<u64>(mb, 8, 512) << 20;
    }();

    ConstantRing(const Instance& instance, Scheduler& scheduler_)
        : scheduler{scheduler_},
          buffer{instance, 0, Capacity, VideoCore::MemoryType::Stream, "gow3 constant ring"} {
        ASSERT(!buffer.mapped_data.empty());
    }

    [[nodiscard]] vk::Buffer Handle() const noexcept {
        return buffer.Handle();
    }

    /// Stage A: `size` bytes at `alignment`; returns the ring offset, or nullopt when the ring is
    /// still in use by the GPU (the caller leaves the binding to stage B).
    std::optional<u64> Allocate(u64 size, u64 alignment, u64* absolute = nullptr) {
        u64 at = (position + alignment - 1) & ~(alignment - 1);
        if (at % Capacity + size > Capacity) {
            at += Capacity - at % Capacity; // wrap to the ring start
        }
        const auto in_use = [&] { return at + size > Capacity && at + size - Capacity > retired; };
        if (in_use()) {
            ++retire_scans;
            Retire();
            if (in_use()) {
                ++full_allocations;
                ReportPressure();
                return std::nullopt;
            }
        }
        position = at + size;
        Gow3Stats::constant_ring_bytes.fetch_add(size, std::memory_order_relaxed);
        if (absolute) {
            *absolute = at;
        }
        return at % Capacity;
    }

    /// Stage A (GOW3_RING_REUSE): the region just allocated at this absolute position may be
    /// bound again by later packets. The ring frees nothing from there on while the reuse epoch
    /// is open, nor until the GPU has completed every packet recorded before it closed: a region
    /// is otherwise freed once the packet that first wrote it completes. Held from the copy,
    /// not from the reuse: by then that first packet may have completed.
    void Hold(u64 absolute) {
        reuse_floor = std::min(reuse_floor, absolute);
    }

    /// Stage A: the copies made so far are no longer reused (a new guest command buffer).
    void CloseReuseEpoch() {
        if (reuse_floor == NoFloor) {
            return;
        }
        // One entry per closed epoch, released in order: merging them into one floor with a
        // moving release point never released it, and the ring stayed full (t-rr1).
        if (held_count == held.size()) {
            held[(held_head + held_count - 1) % held.size()].until = position; // full: extend the last
        } else {
            held[(held_head + held_count) % held.size()] = {reuse_floor, position};
            ++held_count;
        }
        reuse_floor = NoFloor;
    }

    [[nodiscard]] u8* Data(u64 offset) noexcept {
        return buffer.mapped_data.data() + offset;
    }

    /// Stage A, after writing: makes the bytes visible to the GPU.
    void Flush(u64 offset, u64 size) {
        buffer.Flush(offset, size);
    }

    /// Stage A: the position the next packet's allocations end at.
    [[nodiscard]] u64 Position() const noexcept {
        return position;
    }

    /// Stage B, after recording a packet whose allocations end at `end`. Stamps of one
    /// submission tick are merged: only the last end per tick is published.
    void Stamp(u64 end) {
        const u64 tick = scheduler.CurrentTick();
        if (end == pending_end) {
            // The same end on a later tick: its region is complete, publish it. Otherwise a ring
            // that ran full (the position no longer moves) never published its last stamp, never
            // released anything and stayed full (t-gpu1-tonight-off: ~1.7 M failed allocations
            // a second, each a retirement scan, all constants left to the stream buffer).
            if (tick != pending_tick && pending_end != last_published) {
                Publish(pending_end, pending_tick);
            }
            return;
        }
        if (tick != pending_tick && pending_end != last_published) {
            Publish(pending_end, pending_tick);
        }
        pending_end = end;
        pending_tick = tick;
    }

private:
    void Publish(u64 end, u64 tick) {
        const u64 at = stamps_head.load(std::memory_order_relaxed);
        if (at - stamps_tail.load(std::memory_order_acquire) >= stamps.size()) {
            return; // full: this region retires with a later stamp
        }
        stamps[at % stamps.size()] = {end, tick};
        stamps_head.store(at + 1, std::memory_order_release);
        last_published = end;
    }

    void Retire() {
        auto* semaphore = scheduler.GetWorkSemaphore();
        semaphore->Refresh();
        u64 at = stamps_tail.load(std::memory_order_relaxed);
        const u64 head = stamps_head.load(std::memory_order_acquire);
        while (at < head && semaphore->IsFree(stamps[at % stamps.size()].tick)) {
            completed = stamps[at % stamps.size()].end;
            ++at;
        }
        stamps_tail.store(at, std::memory_order_release);
        // A closed epoch's regions are free once every packet recorded before it closed is done.
        while (held_count && completed >= held[held_head].until) {
            held_head = (held_head + 1) % held.size();
            --held_count;
        }
        const u64 held_floor = held_count ? held[held_head].floor : NoFloor;
        retired = std::min({completed, reuse_floor, held_floor});
    }

    static constexpr u64 NoFloor = ~0ull;

    /// Stage A: at most every 5 s, and only when the ring was full (no stats switch needed: the
    /// t-nostats bisect pointed at ring reuse with stats off; the report goes to the background
    /// writer, off this thread).
    void ReportPressure() {
        const auto now = std::chrono::steady_clock::now();
        if (now - report_time < std::chrono::seconds(5)) {
            return;
        }
        std::string text;
        Gow3AsyncLog::Append(text,
                             "Constant ring (5 s): %llu allocations found it full (left to stage "
                             "B), %llu retirement scans; held epochs %u, open epoch holds from "
                             "%llu, position %llu, completed %llu\n",
                             (unsigned long long)full_allocations, (unsigned long long)retire_scans,
                             held_count,
                             (unsigned long long)(reuse_floor == NoFloor ? 0 : position - reuse_floor),
                             (unsigned long long)position, (unsigned long long)completed);
        Gow3AsyncLog::Write(std::move(text));
        full_allocations = retire_scans = 0;
        report_time = now;
    }
    u64 full_allocations = 0, retire_scans = 0;            ///< stage A
    std::chrono::steady_clock::time_point report_time{};  ///< stage A

    struct StampEntry {
        u64 end;
        u64 tick;
    };
    Scheduler& scheduler;
    VideoCore::Buffer buffer;
    u64 position = 0;     ///< stage A
    u64 retired = 0;      ///< stage A: free to overwrite everything before this
    u64 completed = 0;    ///< stage A: the GPU is done with packets ending before this
    u64 reuse_floor = NoFloor;          ///< stage A: oldest region the open reuse epoch holds
    struct Held {
        u64 floor; ///< oldest region the epoch holds
        u64 until; ///< position when it closed: free once packets up to here are done
    };
    std::array<Held, 1024> held{}; ///< stage A: closed epochs, oldest first (floors ascend)
    u32 held_head = 0, held_count = 0;
    u64 pending_end = 0, pending_tick = 0, last_published = 0; ///< stage B
    std::array<StampEntry, 1024> stamps{};
    alignas(64) std::atomic<u64> stamps_head{0}; ///< stage B writes
    alignas(64) std::atomic<u64> stamps_tail{0}; ///< stage A reads
};

} // namespace Vulkan
