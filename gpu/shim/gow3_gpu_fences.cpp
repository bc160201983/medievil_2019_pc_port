// SPDX-License-Identifier: GPL-2.0-or-later
// gow3 U5-2: GOW3_GPU_FENCES' frame pacing and report (gow3_gpu_fences.h). The pacing is
// adapted from Bloodborne 0.7 (bloodborne_pc 0fc2921, gpu/platform/gnm/gnm_driver.cpp,
// BB_SUBMIT_LOCK=decode, BB_SUBMIT_AHEAD, WaitPreviousFrame, WatchOverlapLeak).

#include "gow3_gpu_fences.h"

#include <algorithm>
#include <condition_variable>
#include <cstdio>
#include <mutex>

#include "gow3_toggles.h"

extern "C" void runtime_memory_usage(std::uint64_t* direct_live, std::uint64_t* flexible,
                                     std::uint64_t* pool_committed);

namespace Gow3GpuFences {

namespace {

unsigned SubmitAheadOption() {
    static const unsigned ahead = [] {
        const char* env = std::getenv("GOW3_SUBMIT_AHEAD");
        return env && *env ? unsigned(std::max(0, std::atoi(env))) : 2u;
    }();
    return ahead;
}

/// GOW3_LEAK_WATCH: `log` (default) reports guest memory growth over the low seen after the
/// warm-up; `stop` also turns the overlap off for the session past GOW3_LEAK_LIMIT_MB (as 0.7's
/// leak watchdog does); `0` neither.
enum class LeakWatch { Off, Log, Stop };
LeakWatch LeakWatchMode() {
    static const LeakWatch mode = [] {
        const char* env = std::getenv("GOW3_LEAK_WATCH");
        const std::string value = env ? env : "";
        return value == "0" ? LeakWatch::Off : value == "stop" ? LeakWatch::Stop : LeakWatch::Log;
    }();
    return mode;
}

std::uint64_t LeakLimitBytes() {
    static const std::uint64_t limit = [] {
        const char* env = std::getenv("GOW3_LEAK_LIMIT_MB");
        return (env && *env ? std::strtoull(env, nullptr, 10) : 1024ull) << 20;
    }();
    return limit;
}

struct Pacing {
    std::mutex mutex;
    std::condition_variable cv;
    std::uint64_t frames_submitted = 0; ///< sceGnmSubmitDone calls
    std::uint64_t frames_retired = 0;   ///< frames the GPU has finished
    std::atomic<unsigned> submits_this_frame{0};
    std::atomic<bool> overlap_off{false}; ///< the leak watch stopped the overlap
    std::atomic<unsigned> waiting{0};     ///< guest threads in WaitPreviousFrame
};
Pacing pacing;

/// Guest memory the runtime sees in use: direct memory allocated (the pool's expansion is one
/// such allocation), flexible memory mapped, pool pages committed (the game's heaps grow through
/// these). The sum counts the pool twice; only its changes matter here.
struct GuestMemory {
    std::uint64_t direct = 0, flexible = 0, pool = 0;
    std::uint64_t Total() const {
        return direct + flexible + pool;
    }
};
GuestMemory ReadGuestMemory() {
    GuestMemory m;
    runtime_memory_usage(&m.direct, &m.flexible, &m.pool);
    return m;
}
struct LeakState {
    std::uint64_t frames = 0, lowest = 0, highest_growth = 0;
    bool reported = false;
};
LeakState leak;
std::mutex leak_mutex;

/// Once a frame (sceGnmSubmitDone). Bloodborne 0.7 counts its game's live heap allocations
/// (wrappers on Bloodborne's malloc); MediEvil's heaps sit on the runtime's direct, flexible and
/// pool memory, so this watches their sum: after a warm-up of 600 frames, growth over the lowest
/// value seen. Level loads move it too, so by default it only reports (Report()).
void WatchOverlapLeak() {
    if (LeakWatchMode() == LeakWatch::Off) {
        return;
    }
    const std::uint64_t used = ReadGuestMemory().Total();
    std::scoped_lock lock{leak_mutex};
    if (++leak.frames < 600 || used < leak.lowest) {
        leak.lowest = used; // warming up (boot, the first load)
        return;
    }
    const std::uint64_t growth = used - leak.lowest;
    leak.highest_growth = std::max(leak.highest_growth, growth);
    if (growth <= LeakLimitBytes() || leak.reported) {
        return;
    }
    leak.reported = true;
    const bool stop = LeakWatchMode() == LeakWatch::Stop && SubmitAheadOption() != 0;
    if (stop) {
        pacing.overlap_off = true;
    }
    std::printf("GPU fences: guest memory in use grew %.0f MiB over its low of %.0f MiB%s\n",
                growth / 1048576.0, leak.lowest / 1048576.0,
                stop ? ": frames no longer overlap (GOW3_SUBMIT_AHEAD 0 for the session)"
                     : " (GOW3_LEAK_WATCH=stop would turn the overlap off)");
    std::fflush(stdout);
}

/// Until the GPU has finished every frame submitted so far. Logs (once per wait) when that
/// takes 10 s, with what the signal thread waits for.
void WaitPreviousFrame(bool submit_done) {
    std::unique_lock lock{pacing.mutex};
    const auto done = [] { return pacing.frames_retired >= pacing.frames_submitted; };
    if (done()) {
        return;
    }
    const std::uint64_t start = NowNs();
    ++pacing.waiting;
    bool told = false;
    while (!pacing.cv.wait_for(lock, std::chrono::seconds(10), done)) {
        if (told) {
            continue;
        }
        told = true;
        const std::uint64_t submitted = pacing.frames_submitted, retired = pacing.frames_retired;
        lock.unlock();
        std::string out = "GPU fences watchdog: a guest thread waited 10 s in ";
        out += submit_done ? "sceGnmSubmitDone" : "a submission";
        out += " for frame " + std::to_string(submitted) + " to finish on the GPU (retired " +
               std::to_string(retired) + ")\n  " + DescribePacing() + "\n";
        if (describe_signals) {
            out += describe_signals(describe_context);
        }
        std::fputs(out.c_str(), stdout);
        std::fflush(stdout);
        lock.lock();
    }
    --pacing.waiting;
    if (Gow3Stats::enabled) {
        const std::uint64_t waited = NowNs() - start;
        auto& count = submit_done ? counters.done_waits : counters.submit_waits;
        auto& total = submit_done ? counters.done_wait_ns : counters.submit_wait_ns;
        auto& max = submit_done ? counters.done_wait_max_ns : counters.submit_wait_max_ns;
        count.fetch_add(1, std::memory_order_relaxed);
        total.fetch_add(waited, std::memory_order_relaxed);
        AtomicMax(max, waited);
    }
}

} // namespace

unsigned SubmitAhead() {
    return pacing.overlap_off.load(std::memory_order_relaxed) ? 0u : SubmitAheadOption();
}

std::uint64_t BeginFrame() {
    if (Gow3Stats::enabled) {
        counters.done_calls.fetch_add(1, std::memory_order_relaxed);
    }
    // A frame without submissions waits here, as the console's lock makes sceGnmSubmitDone wait.
    WaitPreviousFrame(true);
    pacing.submits_this_frame = 0;
    WatchOverlapLeak();
    std::scoped_lock lock{pacing.mutex};
    return ++pacing.frames_submitted;
}

void AfterSubmit() {
    if (Gow3Stats::enabled) {
        counters.guest_submits.fetch_add(1, std::memory_order_relaxed);
    }
    if (++pacing.submits_this_frame <= SubmitAhead()) {
        return;
    }
    WaitPreviousFrame(false);
}

void NoteFramesRetired(std::uint64_t frame) {
    std::scoped_lock lock{pacing.mutex};
    pacing.frames_retired = std::max(pacing.frames_retired, frame);
    pacing.cv.notify_all();
}

std::string DescribePacing() {
    std::uint64_t submitted, retired;
    {
        std::scoped_lock lock{pacing.mutex};
        submitted = pacing.frames_submitted;
        retired = pacing.frames_retired;
    }
    char line[256];
    std::snprintf(line, sizeof(line),
                  "pacing: %s, ahead %u; frames submitted %llu, retired by the GPU %llu; %u "
                  "submissions in this frame; %u guest threads waiting for a frame",
                  LockOnGpu() ? "console lock (GOW3_SUBMIT_LOCK=gpu)" : "decode", SubmitAhead(),
                  static_cast<unsigned long long>(submitted),
                  static_cast<unsigned long long>(retired), pacing.submits_this_frame.load(),
                  pacing.waiting.load());
    return line;
}

std::string Report(double window_s, unsigned frames) {
    std::string text;
    char line[1024];
    const auto take = [](std::atomic<std::uint64_t>& a) {
        return static_cast<unsigned long long>(a.exchange(0, std::memory_order_relaxed));
    };
    const double per_frame = frames ? 1.0 / frames : 0.0;
    const unsigned long long submits = take(counters.submits);
    if (!Enabled()) {
        std::snprintf(line, sizeof(line), "GPU fences (5 s): off (GOW3_GPU_FENCES); submits %.1f/frame\n",
                      submits * per_frame);
        text += line;
    } else {
        const auto ms = [](unsigned long long ns) { return ns / 1e6; };
        const unsigned long long published = take(counters.published), run = take(counters.run),
                                 queue_max = take(counters.queue_max), waits = take(counters.waits),
                                 wait_ns = take(counters.wait_ns);
        const unsigned long long f_thread = take(counters.flush_signal_thread),
                                 f_idle = take(counters.flush_decoder_idle),
                                 f_wrm = take(counters.flush_wait_reg_mem),
                                 f_frame = take(counters.flush_frame_end),
                                 f_sub_idle = take(counters.flush_submit_idle),
                                 f_sub_drain = take(counters.flush_submit_drain),
                                 f_sub_interval = take(counters.flush_submit_interval);
        std::snprintf(line, sizeof(line),
                      "GPU fences (5 s, GOW3_GPU_FENCES=1): signals %llu published, %llu run "
                      "(queue max %llu); signal thread waited %.1f ms/s (%llu waits); flushes for "
                      "signals: signal thread %llu, command thread idle %llu, WAIT_REG_MEM %llu, "
                      "frame end %llu, submission end %llu (GPU idle %llu, last submission %llu, "
                      "interval %llu); submits %.1f/frame; CPU writes over pending labels: "
                      "WRITE_DATA %llu, flip %llu; labels to unmapped memory %llu\n",
                      published, run, queue_max, window_s > 0 ? ms(wait_ns) / window_s : 0.0,
                      waits, f_thread, f_idle, f_wrm, f_frame,
                      f_sub_idle + f_sub_drain + f_sub_interval, f_sub_idle, f_sub_drain,
                      f_sub_interval, submits * per_frame, take(counters.overtake_write_data),
                      take(counters.overtake_flip), take(counters.labels_unmapped));
        text += line;
        const unsigned long long done_calls = take(counters.done_calls),
                                 done_waits = take(counters.done_waits),
                                 done_ns = take(counters.done_wait_ns),
                                 done_max = take(counters.done_wait_max_ns),
                                 guest_submits = take(counters.guest_submits),
                                 sub_waits = take(counters.submit_waits),
                                 sub_ns = take(counters.submit_wait_ns),
                                 sub_max = take(counters.submit_wait_max_ns);
        const unsigned long long wrm_met = take(counters.wrm_pending_met),
                                 wrm_waits = take(counters.wrm_waits),
                                 wrm_ns = take(counters.wrm_wait_ns),
                                 wrm_max = take(counters.wrm_wait_max_ns),
                                 wrm_flushes = take(counters.wrm_flushes),
                                 wrm_yields = take(counters.wrm_yields);
        std::snprintf(line, sizeof(line),
                      "Gnm pacing (5 s, %s, ahead %u): SubmitDone %llu, waited %llu (%.2f ms avg "
                      "%.2f max, %.1f ms/s); submissions %llu (%.1f/frame), waited %llu (%.2f ms "
                      "avg %.2f max, %.1f ms/s); WAIT_REG_MEM met by pending signals %llu, waited "
                      "%llu (%.2f ms avg %.2f max), %llu flushes, %llu yields\n",
                      LockOnGpu() ? "console lock" : "decode", SubmitAhead(), done_calls,
                      done_waits, done_waits ? ms(done_ns) / done_waits : 0.0, ms(done_max),
                      window_s > 0 ? ms(done_ns) / window_s : 0.0, guest_submits,
                      guest_submits * per_frame, sub_waits, sub_waits ? ms(sub_ns) / sub_waits : 0.0,
                      ms(sub_max), window_s > 0 ? ms(sub_ns) / window_s : 0.0, wrm_met, wrm_waits,
                      wrm_waits ? ms(wrm_ns) / wrm_waits : 0.0, ms(wrm_max), wrm_flushes, wrm_yields);
        text += line;
    }
    // The leak view (both ways, for the A/B): the runtime's guest memory, its change over the
    // window, and its growth over the low after the warm-up.
    static std::uint64_t previous = 0;
    const GuestMemory now = ReadGuestMemory();
    const double mib = 1048576.0;
    std::uint64_t lowest, highest_growth, watched;
    {
        std::scoped_lock lock{leak_mutex};
        lowest = leak.lowest;
        highest_growth = leak.highest_growth;
        watched = leak.frames;
    }
    std::snprintf(line, sizeof(line),
                  "Guest memory (5 s): direct %.0f MiB allocated (the pool's included), pool %.0f "
                  "MiB committed, flexible %.0f MiB; sum %+.0f MiB this window%s",
                  now.direct / mib, now.pool / mib, now.flexible / mib,
                  previous ? (double(now.Total()) - double(previous)) / mib : 0.0,
                  Enabled() && watched >= 600 ? "" : "\n");
    text += line;
    if (Enabled() && watched >= 600) {
        std::snprintf(line, sizeof(line), ", %+.0f MiB over the low since the warm-up (most %.0f); "
                      "frames %s\n",
                      (double(now.Total()) - double(lowest)) / mib, highest_growth / mib,
                      SubmitAhead() ? "overlap" : "do not overlap");
        text += line;
    }
    previous = now.Total();
    return text;
}

} // namespace Gow3GpuFences
