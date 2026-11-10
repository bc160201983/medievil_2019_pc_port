// SPDX-License-Identifier: GPL-2.0-or-later
// gow3 (research/docs/impl-cpu-waste.md): stage A's wait for stage B (DrawPipe::Drain) as a
// bounded spin and then a sleep. Today's wait spins 4096 pauses and then calls sched_yield until
// stage B has run every packet: on the Steam Deck stage A (shadPS4:GpuComm) used ~1.1 CPU while
// it waited ~33-60% of the time (drains of ~0.5-2 ms, ~300-1000 a second), ~0.3 CPU of it in
// the kernel, and the APU is at its power cap: that CPU power is GPU clock.
// With GOW3_DRAIN_HYBRID=1 a drain that has to wait spins with pauses only (no system call) for up
// to GOW3_DRAIN_SPIN_US (default 30), then sleeps on a futex (Windows: WaitOnAddress). Stage B
// wakes it once, when it reaches the position stage A asked for: GOW3_DRAIN_WAKE_AHEAD packets
// (default 2, 0-15) before the last one, so stage A is running again when the last packet ends
// and spins the rest; if that takes longer than the spin it sleeps again until the end. Stage B
// checks one word per packet and makes a system call only for a registered sleeper (the opt-in
// GOW3_DRAIN_SLEEP woke stage A at every packet it ran, a futex call each on stage B).
// Opt-in until measured; second toggle word 4096 switches it off live (today's wait again).
//
// With GOW3_FRAME_STATS=1 every drain that waited is counted, on either path ("Stage A drain
// waits per frame" line, videoout/driver.cpp).
#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>

#include "gow3_toggles.h"

namespace Gow3DrainWait {

/// GOW3_DRAIN_HYBRID=1 at start.
inline bool Wanted() {
    static const bool wanted = [] {
        const char* env = std::getenv("GOW3_DRAIN_HYBRID");
        return env && env[0] == '1';
    }();
    return wanted;
}

/// Wanted and not switched off (second toggle word 4096).
inline bool Enabled() {
    return Wanted() && !Gow3Toggle2::Disabled(Gow3Toggle2::DrainHybridOff);
}

/// The pause spin before a sleep (GOW3_DRAIN_SPIN_US, default 30, at most 2000).
inline std::uint64_t SpinNs() {
    static const std::uint64_t ns = [] {
        const char* env = std::getenv("GOW3_DRAIN_SPIN_US");
        const long us = env && *env ? std::strtol(env, nullptr, 10) : 30;
        return std::uint64_t(std::clamp<long>(us, 0, 2000)) * 1000;
    }();
    return ns;
}

/// Packets before the last one at which stage B wakes a sleeping stage A (GOW3_DRAIN_WAKE_AHEAD,
/// default 2, 0-15; the commit positions are kept for the last 16 packets). Second toggle word
/// 8388608: the default 2 again, live (a same-launch A/B of the start's count).
inline unsigned WakeAhead() {
    static const unsigned ahead = [] {
        const char* env = std::getenv("GOW3_DRAIN_WAKE_AHEAD");
        const long n = env && *env ? std::strtol(env, nullptr, 10) : 2;
        return unsigned(std::clamp<long>(n, 0, 15));
    }();
    return Gow3Toggle2::Disabled(Gow3Toggle2::DrainWakeAheadDefault) ? 2u : ahead;
}

inline std::uint64_t NowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

/// Sleeps while `word` holds `seen` (or returns at once if it does not; spurious returns allowed).
void FutexWait(std::atomic<std::uint32_t>& word, std::uint32_t seen);
/// Wakes one thread sleeping in FutexWait on `word`.
void FutexWake(std::atomic<std::uint32_t>& word);
/// Wakes every thread sleeping in FutexWait on `word` (gow3_signal_wait.h).
void FutexWakeAll(std::atomic<std::uint32_t>& word);

/// Counters (GOW3_FRAME_STATS), reset by the report. Both paths: drains that waited and their
/// time. Today's path: drains that ended within the 4096 pauses, sched_yield calls (or
/// GOW3_DRAIN_SLEEP rounds). Hybrid: drains ended by the first spin; drains that slept, futex
/// sleeps, the ones woken ahead of the end and of those the ones the spin then finished; spin and
/// sleep time; wake latency (stage B's wake to stage A running) and its count; stage B's wake
/// calls and their time.
inline std::atomic<std::uint64_t> waits{0}, wait_ns{0}, pause_only{0}, yields{0};
inline std::atomic<std::uint64_t> hybrid_waits{0}, spin_ends{0}, slept{0}, sleeps{0},
    woken_ahead{0}, ahead_caught{0}, spin_ns{0}, sleep_ns{0}, wake_latency_ns{0},
    wake_latency_n{0}, wake_latency_max_ns{0}, wakes{0}, wake_call_ns{0};
/// When stage B last woke stage A (NowNs, stats only).
inline std::atomic<std::uint64_t> last_wake_ns{0};

inline void AtomicMax(std::atomic<std::uint64_t>& value, std::uint64_t candidate) {
    std::uint64_t seen = value.load(std::memory_order_relaxed);
    while (candidate > seen &&
           !value.compare_exchange_weak(seen, candidate, std::memory_order_relaxed)) {
    }
}

} // namespace Gow3DrainWait
