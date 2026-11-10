// SPDX-License-Identifier: GPL-2.0-or-later
// gow3: the game's occlusion queries (ZPASS_DONE events) answered with real Vulkan occlusion
// queries (O2). The design follows the Bloodborne port's vk_occlusion.h (deadinside28, 0.5): a
// chain of query segments, one active while a render pass is open, ending at every event and at
// every render pass boundary, an event remembering how many segments ended before it. The
// results reach the game differently: Bloodborne's GPU writes them into the game's memory (its
// guest memory is GPU memory); here the GPU works on a copy, so a completion thread waits for
// each submission, reads the segments' results on the host, adds them up in order and writes the
// running totals into the game's memory.
//
// The PS4 (UE4's PS4 RHI, plan-frame-pacing.md "What the render thread waits for"): every depth
// block keeps a running count of the samples that pass the depth test; ZPASS_DONE has each block
// write its count, 16 bytes apart, with the valid bit (63). A query is one event before its
// draws (counters at A) and one after (at A + 8); the game sums end - begin over the blocks once
// every valid bit is set. Here the whole count goes into the first block's counter and the
// others write 0, so the differences add up the same.
//
// Until now the port wrote "visible" at once (shadPS4's synthetic counters), which culls
// nothing: the game drew every primitive in view. GOW3_OCCLUSION=1 turns this on (opt-in while
// it is tested; the game also needs r.AllowOcclusionQueries=1, run-game.ps1 turns it off).
#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>

#include "common/types.h"
#include "video_core/renderer_vulkan/vk_common.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"

namespace Vulkan {

class Instance;

class OcclusionQueries final : public Scheduler::RenderPassHook {
public:
    OcclusionQueries(const Instance& instance, Scheduler& scheduler);
    ~OcclusionQueries();

    /// GOW3_OCCLUSION=1.
    static bool Enabled();

    /// A ZPASS_DONE event writing the counters of `pairs` depth blocks at `address`: the
    /// recording thread, in order with the draws.
    void Event(VAddr address, u32 pairs);

    /// O3: true once when a render pass ends and the oldest event not submitted yet waited at
    /// least GOW3_OCCLUSION_SUBMIT_US (default 1000 us; 0: off): the recording thread submits
    /// before its next command, so queries reach the GPU without waiting for the game's next
    /// submission (T-O2d: 2.6-3.0 ms). Not at every pass with an event: MediEvil draws each query
    /// box in its own render pass, and a submission per pass (~128 a frame) cost more than it
    /// saved (T-O3).
    bool TakeSubmitWanted() {
        if (!submit_wanted) {
            return false;
        }
        submit_wanted = false;
        unsubmitted = false;
        return true;
    }

    /// Scheduler: a render pass begins / is about to end (the recording thread).
    void OnBeginRendering() override;
    void OnEndRendering() override;

private:
    static constexpr u32 NumSlots = 16384;
    /// Added to the count in place of a segment that could not begin (no free slot): the query
    /// spanning it reads as visible rather than hidden.
    static constexpr u64 MissedSamples = 1ull << 32;

    struct Segment {
        u32 slot;  ///< query slot; ~0u: not begun (MissedSamples)
        u64 tick;  ///< the submission that holds it
    };
    struct Pending {
        VAddr address;
        u32 pairs;
        u64 upto; ///< segments ended before the event (absolute)
        std::chrono::steady_clock::time_point at;
        u64 tick = 0; ///< submission of the last counted segment before it (0: none)
    };

    void BeginSegment();
    void EndSegment();
    void CompletionLoop(std::stop_token stop);
    void WriteCounters(const Pending& event, u64 total, std::chrono::steady_clock::time_point done);
    void Report();

    const Instance& instance;
    Scheduler& scheduler;
    vk::UniqueQueryPool pool;
    bool precise = false;

    // Recording thread state (and `mutex` for what the completion thread reads).
    bool started = false;
    bool active = false;
    bool unsubmitted = false; ///< an event since the last early submission
    std::chrono::steady_clock::time_point unsubmitted_since;
    bool submit_wanted = false;

    std::mutex mutex;
    std::condition_variable cv;
    std::vector<u32> free_slots;
    std::deque<Segment> segments; ///< ended or active, not yet added up
    u64 segments_base = 0;        ///< absolute index of segments.front()
    u64 ended = 0;                ///< segments ended (absolute)
    std::deque<Pending> pending;  ///< events whose counters are not written yet

    // Completion thread state.
    u64 total = 0;
    std::vector<std::pair<VAddr, u64>> begins; ///< last begin total per counter address (stats)

    // Statistics (GOW3_FRAME_STATS, every 5 s).
    std::atomic<u64> events{0}, missed{0}, queries_hidden{0}, queries_visible{0}, written{0};
    std::atomic<u64> latency_us{0}, latency_max_us{0};
    /// The latency in parts: event -> its submission, submission -> seen done, done -> written.
    std::atomic<u64> to_submit_us{0}, to_done_us{0}, to_written_us{0}, parts{0};
    std::chrono::steady_clock::time_point last_report = std::chrono::steady_clock::now();

    std::jthread completion;
};

} // namespace Vulkan
