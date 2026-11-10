// SPDX-License-Identifier: GPL-2.0-or-later
// gow3 U5-2, GOW3_GPU_FENCES=1 (research/docs/impl-gpu-ordered-fences.md): fences, labels and
// interrupts the guest observes, written once the GPU has executed the work recorded before them,
// and the GPU watchdog. Adapted from Bloodborne 0.7 (bloodborne_pc 0fc2921,
// gpu/render/raster_signals.cpp: SignalAfterGpu, HasUnsubmittedSignals, SubmitForSignalsNow,
// PendingSignalValue, GpuSignalLoop, GpuWatchdog). Differences: signals go straight to the
// shared queue (0.7 holds stage B's per submission, PublishSignals), and only urgent ones (the
// guest may wait for them) ask for a submission.

#include <chrono>
#include <cstdio>
#include <cstdlib>

#include "common/assert.h"
#include "common/thread.h"
#include "gow3_fence_stats.h"
#include "gow3_gpu_fences.h"
#include "gow3_sections.h"
#include "gow3_toggles.h"
#include "video_core/amdgpu/liverpool.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"

namespace Vulkan {

namespace {
u64 EnvU64(const char* name, u64 fallback) {
    const char* env = std::getenv(name);
    return env && *env ? std::strtoull(env, nullptr, 10) : fallback;
}
bool EnvOff(const char* name) {
    const char* env = std::getenv(name);
    return env && env[0] == '0';
}
s64 SteadyNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}
} // namespace

void Rasterizer::SignalAfterGpu(std::function<void()> signal, VAddr label, u64 value,
                                bool urgent) {
    // The tick below must cover everything recorded before: stage B's draws included.
    if (OnStageA()) {
        DrainDrawPipe(DrawPipe::ReasonRasterizer);
    }
    // gow3 gate 2 (GOW3_FRAME_STATS): the fence lag probe, now our signal is at GPU time (its
    // lag should be ~0, "GPU done first" for all).
    const u64 fence = Gow3FenceStats::BeginFence();
    if (fence != ~0ull) {
        Gow3FenceStats::SetGpuWaiter(&scheduler, [](void* s, u64 tick) {
            static_cast<Scheduler*>(s)->GetWorkSemaphore()->Wait(tick);
        });
    }
    // The command buffer being recorded: work recorded before this call is in it or an earlier
    // one (a submission started meanwhile took it along, or numbered it lower).
    const u64 tick = scheduler.CurrentTick();
    Gow3FenceStats::FenceRecorded(fence, tick);
    bool was_empty;
    size_t depth;
    {
        std::scoped_lock lk{gpu_signals_mutex};
        if (!gpu_signal_thread.joinable()) {
            Gow3GpuFences::describe_context = this;
            Gow3GpuFences::describe_signals = [](void* self) {
                return static_cast<Rasterizer*>(self)->DescribeSignals();
            };
            gpu_signal_thread =
                std::jthread([this](std::stop_token token) { GpuSignalLoop(token); });
        }
        was_empty = gpu_signals.empty();
        gpu_signals.push_back({tick, std::move(signal), label, value, fence});
        depth = gpu_signals.size();
        pending_signals.fetch_add(1, std::memory_order_release);
    }
    if (urgent) {
        // One recording thread at a time (stage A records only with stage B drained).
        if (newest_urgent_tick.load(std::memory_order_relaxed) < tick) {
            newest_urgent_tick.store(tick, std::memory_order_release);
        }
    }
    if (Gow3Stats::enabled) {
        Gow3GpuFences::counters.published.fetch_add(1, std::memory_order_relaxed);
        Gow3GpuFences::AtomicMax(Gow3GpuFences::counters.queue_max, depth);
    }
    // The signal thread waits on the condition only with nothing queued (else it waits for the
    // GPU and takes the rest when it gets back): no notify per fence (~180 a frame).
    if (was_empty) {
        gpu_signals_cv.notify_one();
    }
}

bool Rasterizer::PendingSignalValue(VAddr address, u64& value) {
    std::scoped_lock lk{gpu_signals_mutex};
    for (auto it = gpu_signals.rbegin(); it != gpu_signals.rend(); ++it) {
        if (it->label == address) {
            value = it->value;
            return true;
        }
    }
    return false;
}

void Rasterizer::SubmitForSignals() {
    // Decided where this submission's draws and fences are recorded (stage B, in order): on
    // stage A they may not even be recorded yet. Bloodborne 0.7: SubmitForSignals with
    // BB_PIPE_SIGNALS.
    static constexpr u8 none = 0;
    RunInOrder(
        [](Rasterizer& self, const u8*) {
            GOW3_SECTION(TaskSignals);
            self.SubmitForSignalsNow();
        },
        &none, 0);
}

void Rasterizer::SubmitForSignalsNow() {
    if (!HasUnsubmittedSignals()) {
        return;
    }
    // Batched (one submission per GOW3_SIGNAL_FLUSH_US, 0.7's BB_HONEST_FLUSH_US) while the GPU
    // still has work; at once when it has finished everything submitted (holding the work back
    // then only leaves it idle), unless draws are queued for stage B (GOW3_SIGNAL_IDLE_BACKLOG
    // bytes, 64 KiB): they are then what the GPU waits for. Also at once when the GPU is on the
    // last submission (GOW3_SIGNAL_DRAIN_FLUSH): it would run dry before the next batch.
    // Bloodborne 0.7: "a submission costs ~65 us (heavy scenes: 13 a frame; 95 -> 100 FPS with
    // half as many)".
    static const s64 interval_ns = s64(EnvU64("GOW3_SIGNAL_FLUSH_US", 2000)) * 1000;
    static const u64 idle_backlog = EnvU64("GOW3_SIGNAL_IDLE_BACKLOG", 64 * 1024);
    static const bool idle_flush = !EnvOff("GOW3_SIGNAL_IDLE_FLUSH");
    static const bool drain_flush = !EnvOff("GOW3_SIGNAL_DRAIN_FLUSH");
    const u64 tick = scheduler.CurrentTick();
    const bool gpu_idle =
        idle_flush &&
        (!draw_pipe || !DrawPipe::OnStageB() || draw_pipe->Backlog() < idle_backlog) &&
        scheduler.IsFree(tick - 1);
    const bool gpu_draining = drain_flush && !gpu_idle && tick >= 2 && scheduler.IsFree(tick - 2);
    const s64 last_submit = scheduler.SubmitTime(tick - 1);
    const bool due = last_submit == 0 || SteadyNs() - last_submit >= interval_ns;
    if (!gpu_idle && !gpu_draining && !due) {
        return;
    }
    auto& counter = gpu_idle       ? Gow3GpuFences::counters.flush_submit_idle
                    : gpu_draining ? Gow3GpuFences::counters.flush_submit_drain
                                   : Gow3GpuFences::counters.flush_submit_interval;
    counter.fetch_add(1, std::memory_order_relaxed);
    Flush();
}

void Rasterizer::GpuSignalLoop(std::stop_token token) {
    Common::SetCurrentThreadName("gow3:GpuSignals");
    auto* semaphore = scheduler.GetWorkSemaphore();
    const vk::Semaphore handle = semaphore->Handle();
    while (!token.stop_requested()) {
        u64 tick = 0;
        {
            std::unique_lock lk{gpu_signals_mutex};
            if (!gpu_signals_cv.wait(lk, token, [this] { return !gpu_signals.empty(); })) {
                return;
            }
            tick = gpu_signals.front().tick;
        }
        // Bounded waits: the tick may not be submitted yet (the GPU command thread submits it
        // when it runs out of work), and a stop request must be seen. Our first attempt
        // (f97479f) waited here without a bound for a tick nobody submitted: black screen at
        // boot.
        const vk::SemaphoreWaitInfo wait_info = {
            .semaphoreCount = 1,
            .pSemaphores = &handle,
            .pValues = &tick,
        };
        const auto wait_start = std::chrono::steady_clock::now();
        bool watched = false;
        auto asked_at = wait_start - std::chrono::seconds(1);
        bool done = semaphore->IsFree(tick);
        while (!done && !token.stop_requested()) {
            const vk::Result result = instance.GetDevice().waitSemaphores(&wait_info, 2'000'000);
            if (result == vk::Result::eSuccess) {
                break;
            }
            // The tick is still being recorded and the GPU command thread may be idle already:
            // ask it to submit, every 50 ms while this lasts.
            const auto now = std::chrono::steady_clock::now();
            if (result == vk::Result::eTimeout && tick >= scheduler.CurrentTick() &&
                now - asked_at >= std::chrono::milliseconds(50)) {
                asked_at = now;
                Gow3GpuFences::counters.flush_signal_thread.fetch_add(1, std::memory_order_relaxed);
                liverpool->SendCommand([this] { Flush(); });
            }
            if (result == vk::Result::eErrorDeviceLost) {
                std::printf("GPU fences: device lost while the signal thread waited for tick %llu\n",
                            static_cast<unsigned long long>(tick));
                std::fflush(stdout);
                ASSERT_MSG(false, "Device lost while waiting for the GPU");
            }
            // 10 s without the GPU finishing this tick: say what is stuck (once; it keeps waiting).
            if (!watched && now - wait_start > std::chrono::seconds(10)) {
                watched = true;
                GpuWatchdog(tick);
            }
        }
        if (!done && Gow3Stats::enabled) {
            Gow3GpuFences::counters.waits.fetch_add(1, std::memory_order_relaxed);
            Gow3GpuFences::counters.wait_ns.fetch_add(
                u64(std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - wait_start)
                        .count()),
                std::memory_order_relaxed);
        }
        semaphore->Refresh();
        // Every signal the GPU is past, in order. Each stays queued while it runs, so that
        // PendingSignalValue still sees its label.
        while (true) {
            std::function<void()> signal;
            u64 fence;
            {
                std::scoped_lock lk{gpu_signals_mutex};
                if (gpu_signals.empty() || !semaphore->IsFree(gpu_signals.front().tick)) {
                    break;
                }
                signal = std::move(gpu_signals.front().signal);
                fence = gpu_signals.front().fence;
            }
            signal();
            Gow3FenceStats::FenceSignaled(fence);
            {
                std::scoped_lock lk{gpu_signals_mutex};
                gpu_signals.pop_front();
            }
            pending_signals.fetch_sub(1, std::memory_order_release);
            if (Gow3Stats::enabled) {
                Gow3GpuFences::counters.run.fetch_add(1, std::memory_order_relaxed);
            }
        }
    }
}

std::string Rasterizer::DescribeSignals() {
    auto* semaphore = scheduler.GetWorkSemaphore();
    semaphore->Refresh();
    const u64 recording = scheduler.CurrentTick();
    char line[512];
    std::snprintf(line, sizeof(line),
                  "  GPU finished tick %llu; recording tick %llu (tick %llu %s); draw pipe %s "
                  "(backlog %llu bytes); newest urgent signal at tick %llu\n",
                  static_cast<unsigned long long>(semaphore->KnownGpuTick()),
                  static_cast<unsigned long long>(recording),
                  static_cast<unsigned long long>(recording - 1),
                  scheduler.SubmitTime(recording - 1) ? "handed to the queue"
                                                      : "not handed to the queue yet",
                  DrawPipeIdle() ? "idle" : "busy",
                  static_cast<unsigned long long>(draw_pipe ? draw_pipe->Backlog() : 0),
                  static_cast<unsigned long long>(newest_urgent_tick.load()));
    std::string out = line;
    std::scoped_lock lk{gpu_signals_mutex};
    std::snprintf(line, sizeof(line), "  %zu signals wait for the GPU", gpu_signals.size());
    out += line;
    if (!gpu_signals.empty()) {
        const auto& front = gpu_signals.front();
        const auto& back = gpu_signals.back();
        std::snprintf(line, sizeof(line),
                      ": first at tick %llu (label %#llx = %#llx), last at tick %llu (label %#llx "
                      "= %#llx)",
                      static_cast<unsigned long long>(front.tick),
                      static_cast<unsigned long long>(front.label),
                      static_cast<unsigned long long>(front.value),
                      static_cast<unsigned long long>(back.tick),
                      static_cast<unsigned long long>(back.label),
                      static_cast<unsigned long long>(back.value));
        out += line;
    }
    out += '\n';
    return out;
}

void Rasterizer::GpuWatchdog(u64 tick) {
    // Logs only: the Deck froze with its sound playing before 0.7 had this (bloodborne_pc
    // 0fc2921 GpuWatchdog). Says whether the GPU is stuck or the port's threads wait for each
    // other, and what the guest's pacing waits for.
    auto* semaphore = scheduler.GetWorkSemaphore();
    semaphore->Refresh();
    const u64 done = semaphore->KnownGpuTick(), recording = scheduler.CurrentTick();
    const auto ull = [](u64 v) { return static_cast<unsigned long long>(v); };
    if (tick < recording && !scheduler.SubmitTime(tick)) {
        std::printf("GPU fences watchdog: tick %llu recorded but not handed to the queue after 10 s "
                    "(the GPU finished %llu, recording %llu): its submission waits (recording "
                    "thread, host copies): the port's threads are waiting for each other\n",
                    ull(tick), ull(done), ull(recording));
    } else if (tick < recording) {
        std::printf("GPU fences watchdog: tick %llu submitted and not finished after 10 s (the GPU "
                    "finished %llu, recording %llu): the GPU is stuck\n",
                    ull(tick), ull(done), ull(recording));
    } else {
        std::printf("GPU fences watchdog: waited 10 s for tick %llu, not submitted yet (the GPU "
                    "finished %llu, draw pipe backlog %llu bytes; flushes are asked every 50 ms): "
                    "the GPU command thread is not getting to it\n",
                    ull(tick), ull(done), ull(draw_pipe ? draw_pipe->Backlog() : 0));
    }
    const std::string out = DescribeSignals() + "  " + Gow3GpuFences::DescribePacing() + "\n";
    std::fputs(out.c_str(), stdout);
    std::fflush(stdout);
}

} // namespace Vulkan
