// SPDX-License-Identifier: GPL-2.0-or-later
// gow3: gate 2 of the Deck's U5 plan (research/docs/study-fence-waits.md section 5), with
// GOW3_FRAME_STATS=1 only: what the guest waits for (event queues, the submission lock) and the
// cache actions its command streams carry. Printed every 5 s after "Frame stats"
// (videoout/driver.cpp); every counter is reset by the report.
#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#ifndef _WIN32
#include <pthread.h>
#endif
#include "gow3_toggles.h"

namespace Gow3FenceStats {

inline std::uint64_t NowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

/// The calling thread's name (guest threads carry the game's names, up to 15 characters).
inline void ThreadName(char (&name)[16]) {
    name[0] = 0;
#ifndef _WIN32
    pthread_getname_np(pthread_self(), name, sizeof(name));
#endif
    if (!name[0]) {
        std::snprintf(name, sizeof(name), "tid%zx",
                      std::hash<std::thread::id>{}(std::this_thread::get_id()) & 0xffff);
    }
}

// ---- Event queue waits (sceKernelWaitEqueue): by event filter / ident and waiting thread ----
struct EqueueSlot {
    std::int16_t filter = 0;
    std::uint64_t ident = 0;
    char thread[16]{};
    std::uint64_t calls = 0, waited = 0, wait_ns = 0, wait_max_ns = 0, timeouts = 0;
    std::uint64_t immediate = 0, slack_ns = 0; ///< returned at once: time since the trigger
};
inline std::mutex equeue_mutex;
inline std::array<EqueueSlot, 24> equeue_slots{};
/// Set by the event queue when it hands out an event: when that event was triggered (ns).
inline thread_local std::uint64_t last_trigger_ns = 0;

inline void NoteEqueueWait(std::int16_t filter, std::uint64_t ident, std::uint64_t wait_ns,
                           bool timeout) {
    char name[16];
    ThreadName(name);
    const std::uint64_t now = NowNs();
    std::scoped_lock lock{equeue_mutex};
    EqueueSlot* slot = nullptr;
    for (auto& s : equeue_slots) {
        if (s.calls && s.filter == filter && s.ident == ident && !std::strcmp(s.thread, name)) {
            slot = &s;
            break;
        }
        if (!s.calls && !slot) {
            slot = &s;
        }
    }
    if (!slot) {
        return; // table full this window
    }
    if (!slot->calls) {
        slot->filter = filter;
        slot->ident = ident;
        std::memcpy(slot->thread, name, sizeof(name));
    }
    ++slot->calls;
    if (timeout) {
        ++slot->timeouts;
    } else if (wait_ns > 20'000) {
        ++slot->waited;
        slot->wait_ns += wait_ns;
        slot->wait_max_ns = std::max(slot->wait_max_ns, wait_ns);
    } else {
        ++slot->immediate;
        if (last_trigger_ns && now > last_trigger_ns) {
            slot->slack_ns += now - last_trigger_ns;
        }
    }
}

// ---- Invalidations (Rasterizer::InvalidateMemory: runtime hooks, faults): by thread, with the
// draw-pipe drain each one starts with ----
struct InvalidateSlot {
    char thread[16]{};
    std::uint64_t calls = 0, bytes = 0, drain_ns = 0, drain_max_ns = 0, drained = 0;
};
inline std::mutex invalidate_mutex;
inline std::array<InvalidateSlot, 16> invalidate_slots{};

inline void NoteInvalidate(std::uint64_t bytes, std::uint64_t drain_ns) {
    char name[16];
    ThreadName(name);
    std::scoped_lock lock{invalidate_mutex};
    InvalidateSlot* slot = nullptr;
    for (auto& s : invalidate_slots) {
        if (s.calls && !std::strcmp(s.thread, name)) {
            slot = &s;
            break;
        }
        if (!s.calls && !slot) {
            slot = &s;
        }
    }
    if (!slot) {
        return;
    }
    if (!slot->calls) {
        std::memcpy(slot->thread, name, sizeof(name));
    }
    ++slot->calls;
    slot->bytes += bytes;
    slot->drain_ns += drain_ns;
    if (drain_ns > 20'000) {
        ++slot->drained;
    }
    slot->drain_max_ns = std::max(slot->drain_max_ns, drain_ns);
}

// ---- Fence lag (section 5.1, level 1): for each end-of-pipe fence, when the port signals it today
// and when the GPU has finished the work recorded before it (the command buffer's timeline tick
// is free). Lag = GPU done - our signal: what GPU-ordered fences would add to a guest waiting on
// it. The tick is taken where the fence sits in the recording stream (Scheduler::
// SignalAfterHostCopies); a probe thread waits on the timeline for it. ----
enum FenceKind : int { FenceEopIrq, FenceEop, FenceEos, FenceKindCount };
struct FenceEntry {
    std::atomic<std::uint64_t> tick{0}, t_sig{0};
    std::atomic<int> kind{-1};
};
inline std::array<FenceEntry, 4096> fence_ring{};
inline std::atomic<std::uint64_t> fence_next{0};
/// Set by the fence's packet handler just before it hands the signal to the scheduler.
inline thread_local int pending_fence_kind = -1;
inline void (*wait_gpu_tick)(void* context, std::uint64_t tick) = nullptr;
inline void* wait_gpu_context = nullptr;
inline std::mutex fence_samples_mutex;
inline std::array<std::vector<double>, FenceKindCount> fence_lag_ms{};
inline std::array<std::uint64_t, FenceKindCount> fence_gpu_first{};

inline void FenceProbeLoop() {
    std::uint64_t at = 0;
    for (;;) {
        const std::uint64_t next = fence_next.load(std::memory_order_acquire);
        if (at >= next) {
            std::this_thread::sleep_for(std::chrono::microseconds(500));
            continue;
        }
        if (next - at > fence_ring.size()) {
            at = next - fence_ring.size() / 2; // fell behind: skip ahead
        }
        FenceEntry& e = fence_ring[at % fence_ring.size()];
        std::uint64_t tick = 0;
        for (int tries = 0; !(tick = e.tick.load(std::memory_order_acquire)) && tries < 200; ++tries) {
            std::this_thread::sleep_for(std::chrono::microseconds(500)); // not recorded yet
        }
        const int kind = e.kind.load(std::memory_order_relaxed);
        if (!tick || kind < 0 || !wait_gpu_tick) {
            ++at;
            continue;
        }
        wait_gpu_tick(wait_gpu_context, tick);
        const std::uint64_t t_gpu = NowNs();
        std::uint64_t t_sig = 0;
        for (int tries = 0; !(t_sig = e.t_sig.load(std::memory_order_acquire)) && tries < 200; ++tries) {
            std::this_thread::sleep_for(std::chrono::microseconds(500));
        }
        if (t_sig) {
            std::scoped_lock lock{fence_samples_mutex};
            fence_lag_ms[kind].push_back((double(t_gpu) - double(t_sig)) / 1e6);
            if (t_gpu < t_sig) {
                ++fence_gpu_first[kind];
            }
        }
        ++at;
    }
}

/// The scheduler registers how to wait for a timeline tick (any thread).
inline void SetGpuWaiter(void* context, void (*wait)(void*, std::uint64_t)) {
    if (!wait_gpu_tick) {
        wait_gpu_context = context;
        wait_gpu_tick = wait;
        std::thread(FenceProbeLoop).detach();
    }
}

/// A fence handed to the scheduler: returns its entry (or ~0 when none is pending).
inline std::uint64_t BeginFence() {
    const int kind = std::exchange(pending_fence_kind, -1);
    if (kind < 0 || !Gow3Stats::enabled) {
        return ~0ull;
    }
    const std::uint64_t index = fence_next.load(std::memory_order_relaxed);
    FenceEntry& e = fence_ring[index % fence_ring.size()];
    e.tick.store(0, std::memory_order_relaxed);
    e.t_sig.store(0, std::memory_order_relaxed);
    e.kind.store(kind, std::memory_order_relaxed);
    fence_next.store(index + 1, std::memory_order_release);
    return index;
}
inline void FenceRecorded(std::uint64_t index, std::uint64_t tick) {
    if (index != ~0ull) {
        fence_ring[index % fence_ring.size()].tick.store(tick, std::memory_order_release);
    }
}
inline void FenceSignaled(std::uint64_t index) {
    if (index != ~0ull) {
        fence_ring[index % fence_ring.size()].t_sig.store(NowNs(), std::memory_order_release);
    }
}

// ---- GOW3_EARLY_FRAME_EOP: the per-frame EOP (interrupt + 64-bit frame label) signalled at
// decode; the lead is frames signalled at decode minus frames the draw recording reached ----
inline std::atomic<std::uint64_t> early_eops{0}, early_fallbacks{0}, early_max_lead{0};

// ---- Submission lock (sceGnmSubmitDone arms it; submits wait for GPU idle) ----
inline std::atomic<std::uint64_t> lock_done_calls{0}, lock_armed{0}, lock_waits{0},
    lock_wait_ns{0}, lock_wait_max_ns{0}, lock_held_ns{0}, lock_held_max_ns{0}, lock_releases{0};
inline std::atomic<std::uint64_t> lock_armed_at{0};

inline void AtomicMax(std::atomic<std::uint64_t>& m, std::uint64_t v) {
    for (std::uint64_t cur = m.load(std::memory_order_relaxed);
         v > cur && !m.compare_exchange_weak(cur, v, std::memory_order_relaxed);) {
    }
}

// ---- Cache actions in the guest's command streams ----
inline std::array<std::atomic<std::uint64_t>, 64> eop_event{}, eos_event{}, event_write{},
    release_event{};
inline std::atomic<std::uint64_t> eop_int_irq{0}, eop_data_none{0}, release_count{0},
    acquire_count{0}, acquire_coher_or{0}, acquire_bytes{0},
    acquire_full{0}, event_write_inv_l2{0};
/// RELEASE_MEM cache bits, counted per bit (tcl1_vol, tc_vol, tc_wb, tcl1, tc).
inline std::array<std::atomic<std::uint64_t>, 5> release_bit_counts{};

inline void NoteAcquireMem(std::uint32_t coher_cntl, std::uint64_t size_256) {
    acquire_count.fetch_add(1, std::memory_order_relaxed);
    acquire_coher_or.fetch_or(coher_cntl, std::memory_order_relaxed);
    // CP_COHER_SIZE is in 256-byte units; all ones (or zero) means the whole address space.
    if (size_256 == 0 || size_256 >= 0xffffffffull) {
        acquire_full.fetch_add(1, std::memory_order_relaxed);
    } else {
        acquire_bytes.fetch_add(size_256 * 256, std::memory_order_relaxed);
    }
}

inline std::string EventHistogram(std::array<std::atomic<std::uint64_t>, 64>& counts) {
    std::string out;
    for (unsigned i = 0; i < counts.size(); ++i) {
        if (const auto n = counts[i].exchange(0, std::memory_order_relaxed)) {
            char item[32];
            std::snprintf(item, sizeof(item), "%s%u:%llu", out.empty() ? "" : " ", i,
                          static_cast<unsigned long long>(n));
            out += item;
        }
    }
    return out.empty() ? "-" : out;
}

/// The three report lines (each ends with a newline); resets the counters.
inline std::string Report(double window_s) {
    std::string text;
    char line[512];
    {
        std::scoped_lock lock{equeue_mutex};
        text += "Guest event waits (5 s):";
        bool any = false;
        for (auto& s : equeue_slots) {
            if (!s.calls) {
                continue;
            }
            any = true;
            std::snprintf(line, sizeof(line),
                          " [filter %d ident %llu %s: %llu calls, %llu waited %.2f ms avg %.2f max, "
                          "%llu immediate slack %.2f ms avg, %llu timeouts]",
                          s.filter, static_cast<unsigned long long>(s.ident), s.thread,
                          static_cast<unsigned long long>(s.calls),
                          static_cast<unsigned long long>(s.waited),
                          s.waited ? s.wait_ns / 1e6 / s.waited : 0.0, s.wait_max_ns / 1e6,
                          static_cast<unsigned long long>(s.immediate),
                          s.immediate ? s.slack_ns / 1e6 / s.immediate : 0.0,
                          static_cast<unsigned long long>(s.timeouts));
            text += line;
            s = {};
        }
        text += any ? "\n" : " none\n";
    }
    {
        std::scoped_lock lock{invalidate_mutex};
        text += "Invalidations (5 s, by thread: calls, MB, draw-pipe drain):";
        bool any = false;
        for (auto& s : invalidate_slots) {
            if (!s.calls) {
                continue;
            }
            any = true;
            std::snprintf(line, sizeof(line),
                          " [%s: %llu, %.1f MB, drain %.1f ms/s, %llu over 20 us, max %.2f ms]",
                          s.thread, static_cast<unsigned long long>(s.calls), s.bytes / 1e6,
                          window_s > 0 ? s.drain_ns / 1e6 / window_s : 0.0,
                          static_cast<unsigned long long>(s.drained), s.drain_max_ns / 1e6);
            text += line;
            s = {};
        }
        text += any ? "\n" : " none\n";
    }
    {
        static const char* const kind_names[] = {"EOP irq", "EOP", "EOS"};
        std::scoped_lock lock{fence_samples_mutex};
        text += "Fence lag (5 s, our signal -> GPU done, ms p50/p90/max (n, GPU done first)):";
        for (int k = 0; k < FenceKindCount; ++k) {
            auto& v = fence_lag_ms[k];
            if (v.empty()) {
                continue;
            }
            std::sort(v.begin(), v.end());
            std::snprintf(line, sizeof(line), " %s %.2f/%.2f/%.2f (%zu, %llu);", kind_names[k],
                          v[v.size() / 2], v[v.size() * 9 / 10], v.back(), v.size(),
                          static_cast<unsigned long long>(fence_gpu_first[k]));
            text += line;
            v.clear();
            fence_gpu_first[k] = 0;
        }
        text += "\n";
    }
    if (const auto early = early_eops.exchange(0), fallback = early_fallbacks.exchange(0);
        early || fallback) {
        std::snprintf(line, sizeof(line),
                      "Early frame EOP (5 s): %llu signalled at decode, %llu at stage B (lead cap), "
                      "max lead %llu frames\n",
                      static_cast<unsigned long long>(early),
                      static_cast<unsigned long long>(fallback),
                      static_cast<unsigned long long>(early_max_lead.exchange(0)));
        text += line;
    }
    const auto done = lock_done_calls.exchange(0), armed = lock_armed.exchange(0),
               waits = lock_waits.exchange(0), wait_ns = lock_wait_ns.exchange(0),
               wait_max = lock_wait_max_ns.exchange(0), held = lock_held_ns.exchange(0),
               held_max = lock_held_max_ns.exchange(0), releases = lock_releases.exchange(0);
    std::snprintf(line, sizeof(line),
                  "Submission lock (5 s): %llu SubmitDone, %llu armed, held %.2f ms avg %.2f max "
                  "(%llu releases); submits waited %llu times, %.2f ms avg %.2f max, %.1f ms/s\n",
                  static_cast<unsigned long long>(done), static_cast<unsigned long long>(armed),
                  releases ? held / 1e6 / releases : 0.0, held_max / 1e6,
                  static_cast<unsigned long long>(releases), static_cast<unsigned long long>(waits),
                  waits ? wait_ns / 1e6 / waits : 0.0, wait_max / 1e6,
                  window_s > 0 ? wait_ns / 1e6 / window_s : 0.0);
    text += line;
    std::string release_bits;
    static const char* const bit_names[] = {"tcl1_vol", "tc_vol", "tc_wb", "tcl1", "tc"};
    for (unsigned i = 0; i < release_bit_counts.size(); ++i) {
        if (const auto n = release_bit_counts[i].exchange(0)) {
            std::snprintf(line, sizeof(line), " %s:%llu", bit_names[i],
                          static_cast<unsigned long long>(n));
            release_bits += line;
        }
    }
    const std::string eop = EventHistogram(eop_event), eos = EventHistogram(eos_event),
                      ew = EventHistogram(event_write), rel = EventHistogram(release_event);
    std::snprintf(line, sizeof(line),
                  "Cache actions (5 s, event type:count): EOP %s (irq %llu, no data %llu); EOS %s; "
                  "EVENT_WRITE %s (inv_l2 %llu); RELEASE_MEM %llu %s bits%s; ACQUIRE_MEM %llu "
                  "coher 0x%08llx, %llu whole-space, %.1f MB ranged\n",
                  eop.c_str(), static_cast<unsigned long long>(eop_int_irq.exchange(0)),
                  static_cast<unsigned long long>(eop_data_none.exchange(0)), eos.c_str(), ew.c_str(),
                  static_cast<unsigned long long>(event_write_inv_l2.exchange(0)),
                  static_cast<unsigned long long>(release_count.exchange(0)), rel.c_str(),
                  release_bits.empty() ? " -" : release_bits.c_str(),
                  static_cast<unsigned long long>(acquire_count.exchange(0)),
                  static_cast<unsigned long long>(acquire_coher_or.exchange(0)),
                  static_cast<unsigned long long>(acquire_full.exchange(0)),
                  acquire_bytes.exchange(0) / 1e6);
    text += line;
    return text;
}

} // namespace Gow3FenceStats
