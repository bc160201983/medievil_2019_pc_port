// SPDX-License-Identifier: GPL-2.0-or-later
// gow3 (research/docs/plan-linux-deck.md "Signal waits"): the waits of the recording thread for
// the recorder (Scheduler::WaitDeferredSignals before a guest-visible write, WaitHostCopies) as a
// bounded spin and then a sleep. Today both call sched_yield until the recorder has run the
// chunks queued before the wait: on the Steam Deck stage B spent ~7% of its time there ("blocked:
// host copies", ~0.9 ms a frame at 80 FPS, two or three waits a frame), on a CPU whose power the
// GPU shares under the 15 W cap. At a frame cap stage B has slack, so the wake latency of a
// sleep (~70 us on the Deck, stage A's drain line) costs no frame time there.
// With GOW3_SIGNAL_SLEEP=1 such a wait spins with pauses only (no system call) for up to
// GOW3_SIGNAL_SPIN_US (default 30), then sleeps on a futex (Windows: WaitOnAddress). The recorder
// and the copy threads raise their done counters as before, then (one fence and one load) wake
// the sleepers only when there is one. Opt-in until measured; second toggle word 16777216
// switches it off live (today's wait again).
//
// With GOW3_FRAME_STATS=1 the waits on either path are counted ("Recorder waits per frame" line,
// videoout/driver.cpp).
#pragma once

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>

#include "gow3_drain_wait.h"
#include "gow3_toggles.h"

namespace Gow3SignalWait {

/// GOW3_SIGNAL_SLEEP=1 at start.
inline bool Wanted() {
    static const bool wanted = [] {
        const char* env = std::getenv("GOW3_SIGNAL_SLEEP");
        return env && env[0] == '1';
    }();
    return wanted;
}

/// Wanted and not switched off (second toggle word 16777216).
inline bool Enabled() {
    return Wanted() && !Gow3Toggle2::Disabled(Gow3Toggle2::SignalSleepOff);
}

/// The pause spin before a sleep (GOW3_SIGNAL_SPIN_US, default 30, at most 2000).
inline std::uint64_t SpinNs() {
    static const std::uint64_t ns = [] {
        const char* env = std::getenv("GOW3_SIGNAL_SPIN_US");
        const long us = env && *env ? std::strtol(env, nullptr, 10) : 30;
        return std::uint64_t(std::clamp<long>(us, 0, 2000)) * 1000;
    }();
    return ns;
}

/// The word sleepers wait on (raised for each wake) and how many sleep or are about to.
inline std::atomic<std::uint32_t> generation{0};
inline std::atomic<std::uint32_t> sleepers{0};

/// Counters (GOW3_FRAME_STATS), reset by the report: waits that found work outstanding on either
/// path and their time; on the hybrid one, those ended by the spin, those that slept, the sleeps
/// and their time, and the wakes made.
inline std::atomic<std::uint64_t> waits{0}, wait_ns{0}, spin_ends{0}, slept{0}, sleeps{0},
    sleep_ns{0}, wakes{0};

/// After a done counter the waits read was raised (release): wakes the sleepers, if any.
inline void Notify() {
    // Pairs with the sleeper's registration: either this load sees it, or its check of the
    // counter sees the new value.
    std::atomic_thread_fence(std::memory_order_seq_cst);
    if (sleepers.load(std::memory_order_relaxed) == 0) {
        return;
    }
    generation.fetch_add(1, std::memory_order_seq_cst);
    wakes.fetch_add(1, std::memory_order_relaxed);
    Gow3DrainWait::FutexWakeAll(generation);
}

/// Waits until done() holds: pauses for up to SpinNs, then futex sleeps. done() reads its counter
/// with acquire ordering; `stats` counts the wait.
template <typename Done>
void Wait(Done&& done, bool stats) {
    const std::uint64_t start = stats ? Gow3DrainWait::NowNs() : 0;
    const std::uint64_t spin_until = Gow3DrainWait::NowNs() + SpinNs();
    bool spun_out = false;
    for (std::uint32_t spins = 1; !done(); ++spins) {
        __builtin_ia32_pause();
        if (!(spins & 63) && Gow3DrainWait::NowNs() >= spin_until) {
            spun_out = true;
            break;
        }
    }
    if (!spun_out) {
        if (stats) {
            spin_ends.fetch_add(1, std::memory_order_relaxed);
        }
    } else {
        const std::uint64_t sleep_start = stats ? Gow3DrainWait::NowNs() : 0;
        std::uint64_t rounds = 0;
        while (true) {
            sleepers.fetch_add(1, std::memory_order_seq_cst);
            const std::uint32_t seen = generation.load(std::memory_order_seq_cst);
            std::atomic_thread_fence(std::memory_order_seq_cst);
            if (done()) {
                sleepers.fetch_sub(1, std::memory_order_relaxed);
                break;
            }
            Gow3DrainWait::FutexWait(generation, seen);
            sleepers.fetch_sub(1, std::memory_order_relaxed);
            ++rounds;
        }
        if (stats) {
            slept.fetch_add(1, std::memory_order_relaxed);
            sleeps.fetch_add(rounds, std::memory_order_relaxed);
            sleep_ns.fetch_add(Gow3DrainWait::NowNs() - sleep_start, std::memory_order_relaxed);
        }
    }
    if (stats) {
        waits.fetch_add(1, std::memory_order_relaxed);
        wait_ns.fetch_add(Gow3DrainWait::NowNs() - start, std::memory_order_relaxed);
    }
}

} // namespace Gow3SignalWait
