// SPDX-License-Identifier: GPL-2.0-or-later
#include "video_core/renderer_vulkan/vk_occlusion.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>

#include "common/thread.h"
#include "core/memory.h"
#include "video_core/renderer_vulkan/vk_instance.h"

namespace Vulkan {

namespace {
constexpr u64 ValidBit = 1ull << 63;
constexpr u32 NoSlot = ~0u;

u64 Microseconds(std::chrono::steady_clock::duration d) {
    return u64(std::chrono::duration_cast<std::chrono::microseconds>(d).count());
}
} // namespace

bool OcclusionQueries::Enabled() {
    static const bool enabled = [] {
        const char* env = std::getenv("GOW3_OCCLUSION");
        return env && env[0] == '1';
    }();
    return enabled;
}

OcclusionQueries::OcclusionQueries(const Instance& instance_, Scheduler& scheduler_)
    : instance{instance_}, scheduler{scheduler_} {
    const auto device = instance.GetDevice();
    pool = Check(device.createQueryPoolUnique({
        .queryType = vk::QueryType::eOcclusion,
        .queryCount = NumSlots,
    }));
    // Slots are reset on the host once their results are read (hostQueryReset): a reset never
    // has to be recorded, so a segment never waits for one outside a render pass.
    device.resetQueryPool(*pool, 0, NumSlots);
    free_slots.reserve(NumSlots);
    for (u32 slot = NumSlots; slot-- > 0;) {
        free_slots.push_back(slot); // popped from the back: 0, 1, 2, ...
    }
    precise = instance.IsOcclusionQueryPrecise();
    scheduler.SetRenderPassHook(this);
    completion = std::jthread([this](std::stop_token stop) { CompletionLoop(stop); });
    std::printf("GPU: occlusion queries counted by Vulkan occlusion queries (%s counts)\n",
                precise ? "exact" : "approximate");
}

OcclusionQueries::~OcclusionQueries() {
    scheduler.SetRenderPassHook(nullptr);
    completion.request_stop();
    cv.notify_all();
}

void OcclusionQueries::BeginSegment() {
    if (free_slots.empty()) {
        // Every slot waits for its results: the samples until the next boundary go uncounted.
        // The query spanning them must not read as hidden: count them as many.
        segments.push_back({NoSlot, scheduler.CurrentTick()});
        ++ended;
        missed.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    const u32 slot = free_slots.back();
    free_slots.pop_back();
    scheduler.Record([pool = *pool, slot, precise = precise](vk::CommandBuffer cmdbuf) {
        cmdbuf.beginQuery(pool, slot,
                          precise ? vk::QueryControlFlagBits::ePrecise : vk::QueryControlFlags{});
    });
    segments.push_back({slot, scheduler.CurrentTick()});
    active = true;
}

void OcclusionQueries::EndSegment() {
    const u32 slot = segments.back().slot;
    scheduler.Record([pool = *pool, slot](vk::CommandBuffer cmdbuf) { cmdbuf.endQuery(pool, slot); });
    ++ended;
    active = false;
}

void OcclusionQueries::Event(VAddr address, u32 pairs) {
    {
        std::scoped_lock lock{mutex};
        started = true;
        if (active) {
            EndSegment();
        }
        u64 tick = 0;
        for (auto it = segments.rbegin(); it != segments.rend(); ++it) {
            if (it->slot != NoSlot && u64(segments.rend() - it) - 1 + segments_base < ended) {
                tick = it->tick; // the last ended, counted segment
                break;
            }
        }
        pending.push_back({address, pairs, ended, std::chrono::steady_clock::now(), tick});
        if (!unsubmitted) {
            unsubmitted = true;
            unsubmitted_since = std::chrono::steady_clock::now();
        }
        if (scheduler.IsRendering()) {
            BeginSegment();
        }
    }
    // A query's begin counters (16-byte aligned; its end counters are 8 bytes after them): both
    // cleared here, in order with the draws, so that what the counters held from an earlier use
    // never reads as valid before this query's results are written.
    if (address % 16 == 0) {
        static constexpr u8 zeros[16 * 16]{};
        Core::Memory::Instance()->TryWriteBacking(reinterpret_cast<void*>(address), zeros,
                                                  std::min<u64>(u64(pairs) * 16, sizeof(zeros)));
    }
    events.fetch_add(1, std::memory_order_relaxed);
    cv.notify_one();
}

void OcclusionQueries::OnBeginRendering() {
    std::scoped_lock lock{mutex};
    if (started && !active) {
        BeginSegment();
    }
}

void OcclusionQueries::OnEndRendering() {
    static const auto submit_after = [] {
        const char* env = std::getenv("GOW3_OCCLUSION_SUBMIT_US");
        return std::chrono::microseconds(env ? std::strtoll(env, nullptr, 10) : 1000);
    }();
    std::scoped_lock lock{mutex};
    if (active) {
        EndSegment();
    }
    if (unsubmitted && submit_after.count() > 0 &&
        std::chrono::steady_clock::now() - unsubmitted_since >= submit_after) {
        submit_wanted = true;
    }
}

void OcclusionQueries::WriteCounters(const Pending& event, u64 count,
                                     std::chrono::steady_clock::time_point done) {
    auto* memory = Core::Memory::Instance();
    for (u32 i = 0; i < event.pairs; ++i) {
        const u64 value = (i == 0 ? count : 0) | ValidBit;
        memory->TryWriteBacking(reinterpret_cast<void*>(event.address + u64(i) * 16), &value,
                                sizeof(value));
    }
    // Statistics: a query is an event at A (begin) and one at A + 8 (end).
    if (event.address % 16 == 0) {
        begins.emplace_back(event.address, count);
        if (begins.size() > 4096) {
            begins.erase(begins.begin(), begins.begin() + 2048);
        }
    } else {
        for (auto it = begins.rbegin(); it != begins.rend(); ++it) {
            if (it->first == event.address - 8) {
                ++(count == it->second ? queries_hidden : queries_visible);
                break;
            }
        }
    }
    const auto now = std::chrono::steady_clock::now();
    if (event.tick != 0) {
        if (const s64 submitted = scheduler.SubmitTime(event.tick); submitted != 0) {
            const auto submit_at =
                std::chrono::steady_clock::time_point{std::chrono::nanoseconds{submitted}};
            to_submit_us.fetch_add(Microseconds(submit_at - event.at), std::memory_order_relaxed);
            to_done_us.fetch_add(done > submit_at ? Microseconds(done - submit_at) : 0,
                                 std::memory_order_relaxed);
            to_written_us.fetch_add(Microseconds(now - done), std::memory_order_relaxed);
            parts.fetch_add(1, std::memory_order_relaxed);
        }
    }
    const u64 us = Microseconds(now - event.at);
    latency_us.fetch_add(us, std::memory_order_relaxed);
    if (us > latency_max_us.load(std::memory_order_relaxed)) {
        latency_max_us.store(us, std::memory_order_relaxed);
    }
    written.fetch_add(1, std::memory_order_relaxed);
}

void OcclusionQueries::CompletionLoop(std::stop_token stop) {
    Common::SetCurrentThreadName("gow3:Occlusion");
    const auto device = instance.GetDevice();
    const vk::Semaphore timeline = scheduler.GetWorkSemaphore()->Handle();
    std::vector<Segment> batch;
    std::vector<Pending> ready;
    while (!stop.stop_requested()) {
        u64 upto = 0;
        {
            std::unique_lock lock{mutex};
            cv.wait_for(lock, std::chrono::milliseconds(5),
                        [&] { return stop.stop_requested() || !pending.empty(); });
            if (pending.empty()) {
                continue;
            }
            // The segments ended before the oldest event, in order.
            upto = pending.front().upto;
            batch.assign(segments.begin(), segments.begin() + (upto - segments_base));
        }
        u64 need = 0;
        for (const auto& segment : batch) {
            if (segment.slot != NoSlot) {
                need = std::max(need, segment.tick);
            }
        }
        // The submission holding them done (it may not be submitted yet: wait in steps). `done`:
        // when this thread saw it (earlier when it was done before the wait began).
        auto done_at = std::chrono::steady_clock::now();
        if (need != 0 && !scheduler.GetWorkSemaphore()->IsFree(need)) {
            bool finished = false;
            while (!finished && !stop.stop_requested()) {
                const vk::SemaphoreWaitInfo wait{
                    .semaphoreCount = 1, .pSemaphores = &timeline, .pValues = &need};
                finished = device.waitSemaphores(wait, 50'000'000) == vk::Result::eSuccess;
            }
            if (!finished) {
                return;
            }
            done_at = std::chrono::steady_clock::now();
        }
        for (const auto& segment : batch) {
            if (segment.slot == NoSlot) {
                total += MissedSamples;
                continue;
            }
            u64 samples = 0;
            const auto result = device.getQueryPoolResults(*pool, segment.slot, 1, sizeof(samples),
                                                           &samples, sizeof(samples),
                                                           vk::QueryResultFlagBits::e64);
            total += result == vk::Result::eSuccess ? samples : MissedSamples;
            device.resetQueryPool(*pool, segment.slot, 1);
        }
        {
            std::scoped_lock lock{mutex};
            for (const auto& segment : batch) {
                if (segment.slot != NoSlot) {
                    free_slots.push_back(segment.slot);
                }
            }
            segments.erase(segments.begin(), segments.begin() + batch.size());
            segments_base = upto;
            ready.clear();
            while (!pending.empty() && pending.front().upto <= segments_base) {
                ready.push_back(pending.front());
                pending.pop_front();
            }
        }
        for (const auto& event : ready) {
            WriteCounters(event, total, done_at);
        }
        Report();
    }
}

void OcclusionQueries::Report() {
    static const bool stats = std::getenv("GOW3_FRAME_STATS") != nullptr;
    const auto now = std::chrono::steady_clock::now();
    if (!stats || now - last_report < std::chrono::seconds(5)) {
        return;
    }
    const double seconds = std::chrono::duration<double>(now - last_report).count();
    last_report = now;
    const u64 n = written.exchange(0);
    const u64 hidden = queries_hidden.exchange(0), visible = queries_visible.exchange(0);
    const u64 p = parts.exchange(0);
    std::printf("Occlusion queries: %.0f events/s, %.0f queries/s of which %.1f%% hidden; results "
                "written %.2f ms after the event on average (max %.2f ms; event -> submitted %.2f, "
                "-> GPU done %.2f, -> written %.2f); %llu segments missed; %.0f early submissions/s\n",
                events.exchange(0) / seconds, (hidden + visible) / seconds,
                hidden + visible ? 100.0 * hidden / (hidden + visible) : 0.0,
                n ? latency_us.exchange(0) / 1000.0 / n : 0.0, latency_max_us.exchange(0) / 1000.0,
                p ? to_submit_us.exchange(0) / 1000.0 / p : 0.0,
                p ? to_done_us.exchange(0) / 1000.0 / p : 0.0,
                p ? to_written_us.exchange(0) / 1000.0 / p : 0.0,
                static_cast<unsigned long long>(missed.exchange(0)),
                Gow3Stats::occlusion_submits.exchange(0) / seconds);
    std::fflush(stdout);
}

} // namespace Vulkan
