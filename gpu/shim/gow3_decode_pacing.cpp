// SPDX-License-Identifier: GPL-2.0-or-later
// gow3 GOW3_DECODE_PACING: the copy model's submission pacing and its report
// (gow3_decode_pacing.h, research/docs/impl-decode-pacing.md). The pacing is adapted from
// Bloodborne 0.7 (bloodborne_pc 0fc2921, gpu/platform/gnm/gnm_driver.cpp, BB_SUBMIT_LOCK=decode,
// BB_SUBMIT_AHEAD, WaitPreviousFrame) by way of gow3_gpu_fences.cpp, with a frame lead cap.

#include "gow3_decode_pacing.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <mutex>

#include "gow3_gpu_fences.h"
#include "gow3_toggles.h"

namespace Gow3DecodePacing {

namespace {

unsigned LeadOption() {
    static const unsigned lead = [] {
        const char* env = std::getenv("GOW3_DECODE_PACING_LEAD");
        const long value = env && *env ? std::atol(env) : 1;
        // Above 2 the guest's frame-counted recycling is no longer covered (the 3-frame label
        // free list, render-thread data in UE4's 4-slot transient pool: impl-decode-pacing.md).
        return unsigned(std::clamp(value, 1l, 2l));
    }();
    return lead;
}

unsigned SubmitAheadOption() {
    static const unsigned ahead = [] {
        const char* env = std::getenv("GOW3_SUBMIT_AHEAD");
        return env && *env ? unsigned(std::max(0, std::atoi(env))) : 2u;
    }();
    return ahead;
}

struct Pacing {
    std::mutex mutex;
    std::condition_variable cv;
    std::uint64_t frames_ended = 0;   ///< sceGnmSubmitDone calls
    std::uint64_t frames_retired = 0; ///< frames stage B and their label writes have passed
    std::atomic<unsigned> submits_this_frame{0};
    std::atomic<unsigned> waiting{0}; ///< guest threads in WaitForLead
};
Pacing pacing;

enum class Site { SubmitDone, Submit, Flip };

/// Counters for the 5 s report (GOW3_FRAME_STATS=1); each is reset by Report().
struct Counters {
    std::atomic<std::uint64_t> done_calls{0}, done_waits{0}, done_wait_ns{0}, done_wait_max_ns{0};
    std::atomic<std::uint64_t> submits{0}, submit_waits{0}, submit_wait_ns{0},
        submit_wait_max_ns{0}, flip_waits{0};
    std::atomic<std::uint64_t> lead_sum{0}, lead_max{0};
    std::atomic<std::uint64_t> retired{0}, retire_ns{0}, retire_max_ns{0};
    std::atomic<std::uint64_t> idle_deferred{0}, idle_raised{0}, idle_skipped{0}, idle_lag_ns{0},
        idle_lag_max_ns{0};
    std::atomic<std::uint64_t> onsubmit_runs{0}, onsubmit_lag_ns{0}, onsubmit_lag_max_ns{0},
        onsubmit_work_ns{0}, onsubmit_work_max_ns{0};
    std::atomic<std::uint64_t> vo_met{0}, vo_stale{0}, vo_drained{0};
};
Counters counters;

void AtomicMax(std::atomic<std::uint64_t>& m, std::uint64_t v) {
    for (std::uint64_t cur = m.load(std::memory_order_relaxed);
         v > cur && !m.compare_exchange_weak(cur, v, std::memory_order_relaxed);) {
    }
}

void AddWait(std::atomic<std::uint64_t>& count, std::atomic<std::uint64_t>& total,
             std::atomic<std::uint64_t>& max, std::uint64_t ns) {
    count.fetch_add(1, std::memory_order_relaxed);
    total.fetch_add(ns, std::memory_order_relaxed);
    AtomicMax(max, ns);
}

std::string DescribePacing() {
    char line[256];
    std::snprintf(line, sizeof(line),
                  "decode pacing: lead %u, ahead %u; frames ended %llu, retired %llu; %u "
                  "submissions in this frame; %u guest threads waiting",
                  LeadOption(), SubmitAheadOption(),
                  static_cast<unsigned long long>(pacing.frames_ended),
                  static_cast<unsigned long long>(pacing.frames_retired),
                  pacing.submits_this_frame.load(), pacing.waiting.load());
    return line;
}

/// Until the game is fewer than Lead() frames ahead of the newest retired frame. Logs once per
/// stall when no frame retired for 10 s (never gives up: passing the wait could let a late label
/// land in reused guest memory).
void WaitForLead(std::unique_lock<std::mutex>& lock, Site site) {
    const auto ready = [] { return pacing.frames_ended - pacing.frames_retired < LeadOption(); };
    if (ready()) {
        return;
    }
    const std::uint64_t start = NowNs();
    ++pacing.waiting;
    std::uint64_t seen = pacing.frames_retired;
    bool told = false;
    while (!pacing.cv.wait_for(lock, std::chrono::seconds(10), ready)) {
        if (pacing.frames_retired != seen) {
            seen = pacing.frames_retired; // progress: a slow stretch, not a stall
            told = false;
            continue;
        }
        if (told) {
            continue;
        }
        told = true;
        std::string out = "Decode pacing watchdog: a guest thread waited 10 s in ";
        out += site == Site::SubmitDone ? "sceGnmSubmitDone"
               : site == Site::Flip    ? "a flip submission"
                                       : "a submission";
        out += " with no frame retired (" + DescribePacing() + ")\n";
        lock.unlock();
        if (describe_gpu) {
            out += "  " + describe_gpu(describe_context) + "\n";
        }
        std::fputs(out.c_str(), stdout);
        std::fflush(stdout);
        lock.lock();
    }
    --pacing.waiting;
    if (Gow3Stats::enabled) {
        const std::uint64_t waited = NowNs() - start;
        if (site == Site::SubmitDone) {
            AddWait(counters.done_waits, counters.done_wait_ns, counters.done_wait_max_ns, waited);
        } else {
            AddWait(counters.submit_waits, counters.submit_wait_ns, counters.submit_wait_max_ns,
                    waited);
            if (site == Site::Flip) {
                counters.flip_waits.fetch_add(1, std::memory_order_relaxed);
            }
        }
    }
}

} // namespace

bool Enabled() {
    static const bool on = [] {
        const char* env = std::getenv("GOW3_DECODE_PACING");
        if (!(env && env[0] == '1')) {
            return false;
        }
        if (Gow3GpuFences::Enabled()) {
            std::printf("Decode pacing: GOW3_DECODE_PACING=1 ignored (GOW3_GPU_FENCES=1 paces on "
                        "the GPU)\n");
            std::fflush(stdout);
            return false;
        }
        const char* lead = std::getenv("GOW3_DECODE_PACING_LEAD");
        std::printf("Decode pacing: on (GOW3_DECODE_PACING=1): lead %u frame(s)%s, %u submissions "
                    "of a frame ahead (GOW3_SUBMIT_AHEAD), GPU idle in stream order\n",
                    LeadOption(),
                    lead && *lead && std::atol(lead) != long(LeadOption())
                        ? " (GOW3_DECODE_PACING_LEAD clamped to 1..2)"
                        : "",
                    SubmitAheadOption());
        std::fflush(stdout);
        return true;
    }();
    return on;
}

unsigned Lead() {
    return LeadOption();
}

unsigned SubmitAhead() {
    return SubmitAheadOption();
}

std::uint64_t NowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

std::uint64_t BeginFrame() {
    if (Gow3Stats::enabled) {
        counters.done_calls.fetch_add(1, std::memory_order_relaxed);
    }
    std::unique_lock lock{pacing.mutex};
    // A frame without submissions waits here too, as the console's lock makes it wait.
    WaitForLead(lock, Site::SubmitDone);
    pacing.submits_this_frame = 0;
    const std::uint64_t frame = ++pacing.frames_ended;
    if (Gow3Stats::enabled) {
        const std::uint64_t lead = frame - pacing.frames_retired;
        counters.lead_sum.fetch_add(lead, std::memory_order_relaxed);
        AtomicMax(counters.lead_max, lead);
    }
    return frame;
}

void AfterSubmit(bool flip) {
    if (Gow3Stats::enabled) {
        counters.submits.fetch_add(1, std::memory_order_relaxed);
    }
    if (++pacing.submits_this_frame <= SubmitAheadOption() && !flip) {
        return;
    }
    std::unique_lock lock{pacing.mutex};
    WaitForLead(lock, flip ? Site::Flip : Site::Submit);
}

void NoteFramesRetired(std::uint64_t frame, std::uint64_t decoded_ns) {
    {
        std::scoped_lock lock{pacing.mutex};
        pacing.frames_retired = std::max(pacing.frames_retired, frame);
        pacing.cv.notify_all();
    }
    if (Gow3Stats::enabled) {
        const std::uint64_t ns = NowNs() - decoded_ns;
        counters.retired.fetch_add(1, std::memory_order_relaxed);
        counters.retire_ns.fetch_add(ns, std::memory_order_relaxed);
        AtomicMax(counters.retire_max_ns, ns);
    }
}

void NoteIdleDeferred() {
    if (Gow3Stats::enabled) {
        counters.idle_deferred.fetch_add(1, std::memory_order_relaxed);
    }
}

void NoteIdleRun(bool raised, std::uint64_t handed_ns) {
    if (!Gow3Stats::enabled) {
        return;
    }
    (raised ? counters.idle_raised : counters.idle_skipped).fetch_add(1, std::memory_order_relaxed);
    const std::uint64_t ns = NowNs() - handed_ns;
    counters.idle_lag_ns.fetch_add(ns, std::memory_order_relaxed);
    AtomicMax(counters.idle_lag_max_ns, ns);
}

bool OnSubmitOnStageB() {
    static const bool on = [] {
        const char* env = std::getenv("GOW3_ONSUBMIT_STAGE_B");
        const bool wanted = env && *env ? env[0] == '1' : Enabled();
        if (!wanted) {
            return false;
        }
        if (Gow3GpuFences::Enabled()) {
            std::printf(
                "OnSubmit on stage B: GOW3_ONSUBMIT_STAGE_B ignored (GOW3_GPU_FENCES=1)\n");
            std::fflush(stdout);
            return false;
        }
        std::printf("OnSubmit on stage B: on (GOW3_ONSUBMIT_STAGE_B%s): fault buffer, image "
                    "downloads, collectors, staging trim and the frame's Flush in stream order\n",
                    env && *env ? "=1" : " unset, with GOW3_DECODE_PACING");
        std::fflush(stdout);
        return true;
    }();
    return on;
}

void NoteOnSubmitRun(std::uint64_t handed_ns, std::uint64_t start_ns) {
    if (!Gow3Stats::enabled) {
        return;
    }
    const std::uint64_t now = NowNs();
    counters.onsubmit_runs.fetch_add(1, std::memory_order_relaxed);
    counters.onsubmit_lag_ns.fetch_add(start_ns - handed_ns, std::memory_order_relaxed);
    AtomicMax(counters.onsubmit_lag_max_ns, start_ns - handed_ns);
    counters.onsubmit_work_ns.fetch_add(now - start_ns, std::memory_order_relaxed);
    AtomicMax(counters.onsubmit_work_max_ns, now - start_ns);
}

void NoteVoLabelWait(VoWait kind) {
    if (!Gow3Stats::enabled) {
        return;
    }
    (kind == VoWait::MetNoDrain     ? counters.vo_met
     : kind == VoWait::StalePending ? counters.vo_stale
                                    : counters.vo_drained)
        .fetch_add(1, std::memory_order_relaxed);
}

std::string Report(double window_s, unsigned frames) {
    std::string text;
    char line[768];
    const auto take = [](std::atomic<std::uint64_t>& a) {
        return a.exchange(0, std::memory_order_relaxed);
    };
    const auto ms = [](std::uint64_t ns) { return ns / 1e6; };
    const auto avg = [&](std::uint64_t ns, std::uint64_t n) { return n ? ms(ns) / n : 0.0; };
    const auto per_s = [&](std::uint64_t ns) { return window_s > 0 ? ms(ns) / window_s : 0.0; };
    const auto ull = [](std::uint64_t v) { return static_cast<unsigned long long>(v); };

    if (OnSubmitOnStageB()) {
        const std::uint64_t runs = take(counters.onsubmit_runs),
                            lag = take(counters.onsubmit_lag_ns),
                            lag_max = take(counters.onsubmit_lag_max_ns),
                            work = take(counters.onsubmit_work_ns),
                            work_max = take(counters.onsubmit_work_max_ns);
        std::snprintf(line, sizeof(line),
                      "OnSubmit on stage B (5 s): %llu frame-end steps handed over (drains "
                      "avoided): stage B reached them %.2f ms avg %.2f max after the hand-off (%.1f "
                      "ms/s stage A no longer waits); work and Flush %.2f ms avg %.2f max on stage "
                      "B\n",
                      ull(runs), avg(lag, runs), ms(lag_max), per_s(lag), avg(work, runs),
                      ms(work_max));
        text += line;
    }
    if (const std::uint64_t met = take(counters.vo_met), stale = take(counters.vo_stale),
        drained = take(counters.vo_drained);
        met || stale || drained) {
        std::snprintf(line, sizeof(line),
                      "Display buffer label waits (5 s): %llu met without a drain, %llu met in "
                      "memory but a flip's write pending (drained), %llu drained (not met%s)\n",
                      ull(met), ull(stale), ull(drained),
                      [] {
                          const char* env = std::getenv("GOW3_VO_LABEL_DRAIN");
                          return env && env[0] == '1' ? ", or GOW3_VO_LABEL_DRAIN=1" : "";
                      }());
        text += line;
    }
    if (!Enabled()) {
        return text;
    }
    const std::uint64_t done_calls = take(counters.done_calls), done_waits = take(counters.done_waits),
                        done_ns = take(counters.done_wait_ns),
                        done_max = take(counters.done_wait_max_ns), submits = take(counters.submits),
                        sub_waits = take(counters.submit_waits), sub_ns = take(counters.submit_wait_ns),
                        sub_max = take(counters.submit_wait_max_ns),
                        flip_waits = take(counters.flip_waits), lead_sum = take(counters.lead_sum),
                        lead_max = take(counters.lead_max), retired = take(counters.retired),
                        retire_ns = take(counters.retire_ns), retire_max = take(counters.retire_max_ns);
    const std::uint64_t deferred = take(counters.idle_deferred), raised = take(counters.idle_raised),
                        skipped = take(counters.idle_skipped), lag_ns = take(counters.idle_lag_ns),
                        lag_max = take(counters.idle_lag_max_ns);
    std::snprintf(line, sizeof(line),
                  "Decode pacing (5 s, lead %u, ahead %u): SubmitDone %llu, waited %llu (%.2f ms avg "
                  "%.2f max, %.1f ms/s); later submissions: %llu submissions (%.1f/frame), waited "
                  "%llu (%.2f ms avg %.2f max, %.1f ms/s; %llu of them flips); frame lead at "
                  "SubmitDone %.2f avg, %llu max; frames retired %llu, %.2f ms avg %.2f max after "
                  "their last submission was decoded\n",
                  LeadOption(), SubmitAheadOption(), ull(done_calls), ull(done_waits),
                  avg(done_ns, done_waits), ms(done_max), per_s(done_ns), ull(submits),
                  frames ? double(submits) / frames : 0.0, ull(sub_waits), avg(sub_ns, sub_waits),
                  ms(sub_max), per_s(sub_ns), ull(flip_waits),
                  done_calls ? double(lead_sum) / done_calls : 0.0, ull(lead_max), ull(retired),
                  avg(retire_ns, retired), ms(retire_max));
    text += line;
    std::snprintf(line, sizeof(line),
                  "Decode pacing idle (5 s): %llu GPU idle steps deferred to stage B (%llu raised, "
                  "%llu skipped by the generation check); stage A drains avoided %llu: stage B and "
                  "the labels reached them %.2f ms avg %.2f max after the hand-off (%.1f ms/s stage "
                  "A no longer waits)\n",
                  ull(deferred), ull(raised), ull(skipped), ull(deferred), avg(lag_ns, raised + skipped),
                  ms(lag_max), per_s(lag_ns));
    text += line;
    return text;
}

} // namespace Gow3DecodePacing
