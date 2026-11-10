// SPDX-License-Identifier: GPL-2.0-or-later
// gow3 #4 (research/docs/impl-stage-b-bundle-2.md; study-deck-apu-deep-dive.md 2.1, 2.5): on the
// Steam Deck (amd-pstate active, default EPP) any block longer than ~0.2 ms puts a busy core back
// on a sticky ~2.4 GHz plateau (of 3.5). Stage B (gow3:DrawRec) is ~91% busy there, but it blocks
// every frame: for the next packet, for the recorder at submissions and the flip, for the GPU
// (frames ahead, readbacks) and for a free presentation frame. With GOW3_STAGE_B_SPIN=1 those
// waits spin up to GOW3_STAGE_B_SPIN_US (default 200, at most 2000) with pauses before they block,
// on stage B only: a short wait never sleeps. It costs power (the spin keeps the core in C0),
// so it is opt-in; second toggle word 1024 switches it off live.
//
// With GOW3_FRAME_STATS=1 stage B's blocking waits are timed by site and the ones over 0.2 ms
// counted ("Stage B sleeps per frame" line, videoout/driver.cpp), spin on or off.
#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <x86intrin.h>

#include "gow3_sections.h"
#include "gow3_toggles.h"

namespace Gow3StageBWait {

/// Where stage B blocks.
enum Site : unsigned {
    PacketRing, ///< no packet from stage A (DrawPipe::Run)
    Recorder,   ///< the Vulkan recorder catching up (Scheduler::SyncRecording)
    GpuTick,    ///< a GPU timeline tick (Semaphore::Wait: frames ahead, readbacks, Finish)
    Present,    ///< a free presentation frame or its present fence (Presenter::GetRenderFrame)
    Sites
};

/// Blocking waits that took over 20 us (slept), over 0.2 ms, and their time; spins, their time,
/// and the spins that ended the wait. Reset by the report.
inline std::atomic<std::uint64_t> sleeps[Sites]{}, long_sleeps[Sites]{}, sleep_ns[Sites]{};
inline std::atomic<std::uint64_t> spins{0}, spin_ns{0}, spin_catches{0};

inline std::uint64_t NowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

/// GOW3_STAGE_B_SPIN=1 at start.
inline bool Wanted() {
    static const bool wanted = [] {
        const char* env = std::getenv("GOW3_STAGE_B_SPIN");
        return env && env[0] == '1';
    }();
    return wanted;
}

/// Wanted and not switched off (second toggle word 1024).
inline bool Enabled() {
    return Wanted() && !Gow3Toggle2::Disabled(Gow3Toggle2::StageBSpin);
}

inline std::uint64_t SpinNs() {
    static const std::uint64_t ns = [] {
        const char* env = std::getenv("GOW3_STAGE_B_SPIN_US");
        const long us = env && *env ? std::strtol(env, nullptr, 10) : 200;
        return std::uint64_t(std::clamp<long>(us, 0, 2000)) * 1000;
    }();
    return ns;
}

inline bool OnStageB() {
    return Gow3Sections::recording_thread;
}

/// On stage B with the spin on: calls `done` (at most every `poll_ns`, for checks that cost a
/// system call) with pauses between, until it returns true or SpinNs() has passed. True when the
/// wait is over. Off stage B or with the spin off: false at once, nothing called.
template <typename Done>
bool Spin(Done&& done, std::uint64_t poll_ns = 0) {
    if (!OnStageB() || !Enabled()) {
        return false;
    }
    const std::uint64_t start = NowNs();
    const std::uint64_t deadline = start + SpinNs();
    std::uint64_t next_poll = start;
    bool caught = false;
    for (std::uint64_t now = start;; now = NowNs()) {
        if (now >= next_poll) {
            if (done()) {
                caught = true;
                break;
            }
            next_poll = now + poll_ns;
        }
        if (now >= deadline) {
            break;
        }
        for (int i = 0; i < 8; ++i) {
            _mm_pause();
        }
    }
    if (Gow3Stats::enabled) {
        spins.fetch_add(1, std::memory_order_relaxed);
        spin_ns.fetch_add(NowNs() - start, std::memory_order_relaxed);
        if (caught) {
            spin_catches.fetch_add(1, std::memory_order_relaxed);
        }
    }
    return caught;
}

/// Times a blocking wait on stage B (GOW3_FRAME_STATS): one over 20 us counts as a sleep.
class SleepTimer {
public:
    explicit SleepTimer(Site site_) : site{site_}, start{Gow3Stats::enabled && OnStageB() ? NowNs() : 0} {}
    ~SleepTimer() {
        if (!start) {
            return;
        }
        const std::uint64_t ns = NowNs() - start;
        if (ns < 20'000) {
            return;
        }
        sleeps[site].fetch_add(1, std::memory_order_relaxed);
        sleep_ns[site].fetch_add(ns, std::memory_order_relaxed);
        if (ns > 200'000) {
            long_sleeps[site].fetch_add(1, std::memory_order_relaxed);
        }
    }
    SleepTimer(const SleepTimer&) = delete;
    SleepTimer& operator=(const SleepTimer&) = delete;

private:
    Site site;
    std::uint64_t start;
};

} // namespace Gow3StageBWait
