// SPDX-License-Identifier: GPL-2.0-or-later
// gow3 U5-2 (research/docs/impl-gpu-ordered-fences.md): GOW3_GPU_FENCES=1, Bloodborne 0.7's
// GPU-ordered fence model (bloodborne_pc 0fc2921: gpu/render/raster_signals.cpp,
// gpu/cp/command_processor.cpp, gpu/platform/gnm/gnm_driver.cpp) under today's copy model.
//
// End-of-pipe and end-of-shader labels, compute RELEASE_MEM labels and their interrupts reach
// the guest only once the GPU has executed the work recorded before them (a signal thread on the
// scheduler's timeline, Rasterizer::SignalAfterGpu), in order. GPU idle and "frame retired" go
// the same way. The submission lock becomes 0.7's `decode` pacing: a frame's first
// GOW3_SUBMIT_AHEAD submissions go straight through, later ones and sceGnmSubmitDone wait until
// the GPU has finished the previous frame (a frame: one sceGnmSubmitDone to the next). Off by
// default: today's behaviour, where fences signal when the port's CPU pipeline reaches them.
#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <string>

namespace Gow3GpuFences {

/// GOW3_GPU_FENCES=1 (read once).
inline bool Enabled() {
    static const bool on = [] {
        const char* env = std::getenv("GOW3_GPU_FENCES");
        return env && env[0] == '1';
    }();
    return on;
}

/// GOW3_SUBMIT_LOCK=gpu (with GOW3_GPU_FENCES=1, a diagnostic): the console's lock instead of the
/// `decode` pacing: submissions after sceGnmSubmitDone wait until the GPU is idle (the GPU idle
/// interrupt, now GPU-ordered). Bloodborne 0.7: BB_SUBMIT_LOCK=gpu, 101 vs 154 FPS.
inline bool LockOnGpu() {
    static const bool on = [] {
        const char* env = std::getenv("GOW3_SUBMIT_LOCK");
        return env && std::string(env) == "gpu";
    }();
    return Enabled() && on;
}

/// The `decode` pacing below is in use (GOW3_GPU_FENCES=1 and not GOW3_SUBMIT_LOCK=gpu).
inline bool PacingOnDecode() {
    return Enabled() && !LockOnGpu();
}

/// GOW3_SUBMIT_AHEAD (default 2, 0 = no overlap): a frame's submissions that do not wait for the
/// previous frame. 0 while the leak watch has turned the overlap off (GOW3_LEAK_WATCH=stop).
unsigned SubmitAhead();

// ---- Pacing (gnmdriver.cpp; the guest's submitting threads, under its submission lock) ----

/// sceGnmSubmitDone: waits until the GPU has finished every frame before this one, then returns
/// the new frame's number (1, 2, ...), for Liverpool::SubmitDone.
std::uint64_t BeginFrame();
/// A submission (graphics or compute) was handed to the command processor: the frame's
/// submissions after the first SubmitAhead() wait until the GPU has finished the previous frame.
void AfterSubmit();
/// The signal thread: the GPU has finished everything of frames 1..frame.
void NoteFramesRetired(std::uint64_t frame);

// ---- Diagnostics ----

/// What the signal thread and the GPU are at (set by the rasterizer; for the watchdogs).
inline std::string (*describe_signals)(void* context) = nullptr;
inline void* describe_context = nullptr;
/// The pacing state, one line (for the watchdogs).
std::string DescribePacing();

inline std::uint64_t NowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}
inline void AtomicMax(std::atomic<std::uint64_t>& m, std::uint64_t v) {
    for (std::uint64_t cur = m.load(std::memory_order_relaxed);
         v > cur && !m.compare_exchange_weak(cur, v, std::memory_order_relaxed);) {
    }
}

/// Counters for the 5 s report (GOW3_FRAME_STATS=1); each is reset by Report().
struct Counters {
    // Signals handed to the signal thread, and run by it after the GPU.
    std::atomic<std::uint64_t> published{0}, run{0}, queue_max{0};
    // The signal thread's waits on the timeline (count, ns).
    std::atomic<std::uint64_t> waits{0}, wait_ns{0};
    // Submissions for signals: asked by the signal thread (tick unsubmitted), the GPU command
    // thread idle with signals unsubmitted, WAIT_REG_MEM, a frame's end, a guest submission's
    // end (the GPU had run dry / was on its last submission / BB_HONEST_FLUSH_US-like interval).
    std::atomic<std::uint64_t> flush_signal_thread{0}, flush_decoder_idle{0},
        flush_wait_reg_mem{0}, flush_frame_end{0}, flush_submit_idle{0}, flush_submit_drain{0},
        flush_submit_interval{0};
    // Vulkan submissions of the draw scheduler (with the switch off too).
    std::atomic<std::uint64_t> submits{0};
    // Pacing: sceGnmSubmitDone and later submissions that waited for the previous frame.
    std::atomic<std::uint64_t> done_calls{0}, done_waits{0}, done_wait_ns{0}, done_wait_max_ns{0};
    std::atomic<std::uint64_t> guest_submits{0}, submit_waits{0}, submit_wait_ns{0},
        submit_wait_max_ns{0};
    // WAIT_REG_MEM under the switch: met by a pending signal, waited (count, ns, max), flushes
    // for unsubmitted signals, yields.
    std::atomic<std::uint64_t> wrm_pending_met{0}, wrm_waits{0}, wrm_wait_ns{0},
        wrm_wait_max_ns{0}, wrm_flushes{0}, wrm_yields{0};
    // Guest-visible CPU writes made while GPU-ordered labels were still pending (they land first:
    // the overtaking risk), and labels dropped because their memory was no longer mapped.
    std::atomic<std::uint64_t> overtake_write_data{0}, overtake_flip{0}, labels_unmapped{0};
};
inline Counters counters;

/// The report lines for the 5 s block (videoout/driver.cpp); each ends with a newline. Resets
/// the counters. `frames`: frames presented in the window.
std::string Report(double window_s, unsigned frames);

} // namespace Gow3GpuFences
