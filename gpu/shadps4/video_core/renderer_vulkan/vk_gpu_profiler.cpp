// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <cstdio>
#include <cstdlib>

#include "video_core/renderer_vulkan/vk_gpu_profiler.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"

namespace Vulkan {

void GpuProfiler::Init(const Instance& instance, Scheduler& scheduler) {
    const char* env = std::getenv("GOW3_GPU_PROFILE");
    if (!env || env[0] != '1' || instance_ptr) {
        return;
    }
    instance_ptr = new GpuProfiler(instance, scheduler);
}

GpuProfiler::GpuProfiler(const Instance& instance, Scheduler& scheduler_)
    : device{instance.GetDevice()}, scheduler{scheduler_} {
    const vk::QueryPoolCreateInfo info = {
        .queryType = vk::QueryType::eTimestamp,
        .queryCount = NumSlices * SliceQueries,
    };
    pool = Check<"create timestamp query pool">(device.createQueryPoolUnique(info));
    device.resetQueryPool(*pool, 0, NumSlices * SliceQueries);
    period_ns = instance.GetPhysicalDevice().getProperties().limits.timestampPeriod;
    for (auto& k : keys) {
        k.reserve(SliceQueries);
    }
    std::printf("GPU profile: on (timestamp period %.2f ns)\n", period_ns);
}

void GpuProfiler::WriteTimestamp(u64 key) {
    if (skip_frame || used[slice] + 1 >= SliceQueries) {
        return; // the frame's slice is full: the rest of the frame goes to the last label
    }
    // Outside render passes: radv_CmdWriteTimestamp2 crashed inside some. Marks sit where a
    // pass, dispatch or submission ends anyway.
    scheduler.EndRendering();
    const u32 query = slice * SliceQueries + used[slice]++;
    keys[slice].push_back(key);
    current = key;
    scheduler.Record([pool = *pool, query](vk::CommandBuffer cmdbuf) {
        cmdbuf.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, pool, query);
    });
}

void GpuProfiler::BeginFrame() {
    // Close the frame: one more timestamp without a label.
    if (!skip_frame && used[slice] > 0 && used[slice] < SliceQueries) {
        scheduler.EndRendering();
        const u32 query = slice * SliceQueries + used[slice]++;
        scheduler.Record([pool = *pool, query](vk::CommandBuffer cmdbuf) {
            cmdbuf.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, pool, query);
        });
        pending[slice] = true;
    }
    slice = (slice + 1) % NumSlices;
    // The oldest slice: its frame was recorded four frames ago. Never wait for it: the draw
    // recording thread can be behind this one, and a blocking wait here stalled the game at boot
    // (T-P960 profile runs). Not written yet: this frame is left out, the slice is tried again.
    if (pending[slice] && !Collect(slice)) {
        skip_frame = true;
        Print();
        return;
    }
    skip_frame = false;
    if (used[slice]) {
        device.resetQueryPool(*pool, slice * SliceQueries, used[slice]);
    }
    used[slice] = 0;
    keys[slice].clear();
    pending[slice] = false;
    Print();
}

bool GpuProfiler::Collect(u32 which) {
    const u32 count = used[which];
    std::vector<u64> stamps(count);
    // Four frames on this is complete unless the GPU lags that far: then wait for it.
    const auto result = device.getQueryPoolResults(
        *pool, which * SliceQueries, count, count * sizeof(u64), stamps.data(), sizeof(u64),
        vk::QueryResultFlagBits::e64);
    if (result != vk::Result::eSuccess) {
        return false; // eNotReady: the GPU has not written them all yet
    }
    double frame_ms = 0;
    frame_segments.clear();
    for (u32 i = 0; i + 1 < count; ++i) {
        const double ms = double(stamps[i + 1] - stamps[i]) * period_ns / 1e6;
        auto& total = totals[keys[which][i]];
        total.ms += ms;
        ++total.segments;
        frame_ms += ms;
        frame_segments.emplace_back(keys[which][i], ms);
    }
    ++frames;
    if (frame_ms >= 1.0) {
        double median = 0;
        if (recent_count >= 16) {
            std::array<double, 128> sorted_recent = recent_scene;
            const auto middle = sorted_recent.begin() + recent_count / 2;
            std::nth_element(sorted_recent.begin(), middle, sorted_recent.begin() + recent_count);
            median = *middle;
        }
        const bool slow = median > 0 && frame_ms >= 1.35 * median;
        auto& bucket = slow ? slow_totals : scene_totals;
        for (const auto& [key, ms] : frame_segments) {
            auto& total = bucket[key];
            total.ms += ms;
            ++total.segments;
        }
        ++(slow ? slow_frames : scene_frames);
        recent_scene[recent_next] = frame_ms;
        recent_next = (recent_next + 1) % recent_scene.size();
        recent_count = std::min<u32>(recent_count + 1, recent_scene.size());
    }
    return true;
}

void GpuProfiler::Print() {
    const auto now = std::chrono::steady_clock::now();
    if (now - window < std::chrono::seconds(5) || frames == 0) {
        return;
    }
    window = now;
    std::vector<std::pair<u64, Total>> sorted(totals.begin(), totals.end());
    std::ranges::sort(sorted, [](const auto& a, const auto& b) { return a.second.ms > b.second.ms; });
    double sum = 0;
    for (const auto& [key, total] : sorted) {
        sum += total.ms;
    }
    std::printf("GPU profile: %.2f ms/frame over %llu frames, %zu labels\n", sum / frames,
                static_cast<unsigned long long>(frames), sorted.size());
    for (size_t i = 0; i < std::min<size_t>(sorted.size(), 30); ++i) {
        const auto& [key, total] = sorted[i];
        std::printf("  %6.3f ms/frame %5.1f/frame  %s\n", total.ms / frames,
                    double(total.segments) / frames, described[key].c_str());
    }
    if (slow_frames) {
        // Labels by growth in slow scene parts over the other scene parts of the window.
        double slow_sum = 0, scene_sum = 0;
        for (const auto& [key, total] : slow_totals) {
            slow_sum += total.ms;
        }
        for (const auto& [key, total] : scene_totals) {
            scene_sum += total.ms;
        }
        const double other = scene_frames ? double(scene_frames) : 1.0;
        std::printf("GPU profile, slow scene parts (>= 1.35x median): %llu of %llu, %.2f ms each "
                    "vs %.2f; by growth (slow / other ms, segments):\n",
                    static_cast<unsigned long long>(slow_frames),
                    static_cast<unsigned long long>(slow_frames + scene_frames),
                    slow_sum / slow_frames, scene_sum / other);
        std::vector<std::pair<u64, double>> growth;
        for (const auto& [key, total] : slow_totals) {
            const auto it = scene_totals.find(key);
            const double base = it == scene_totals.end() ? 0.0 : it->second.ms / other;
            growth.emplace_back(key, total.ms / slow_frames - base);
        }
        std::ranges::sort(growth, [](const auto& a, const auto& b) { return a.second > b.second; });
        for (size_t i = 0; i < std::min<size_t>(growth.size(), 20); ++i) {
            const u64 key = growth[i].first;
            const auto& slow_total = slow_totals[key];
            const auto it = scene_totals.find(key);
            const double base_ms = it == scene_totals.end() ? 0.0 : it->second.ms / other;
            const double base_segments =
                it == scene_totals.end() ? 0.0 : double(it->second.segments) / other;
            std::printf("  %+6.3f ms: %6.3f / %6.3f ms, %5.1f / %5.1f  %s\n", growth[i].second,
                        slow_total.ms / slow_frames, base_ms,
                        double(slow_total.segments) / slow_frames, base_segments,
                        described[key].c_str());
        }
    }
    slow_totals.clear();
    scene_totals.clear();
    slow_frames = scene_frames = 0;
    totals.clear();
    frames = 0;
    std::fflush(stdout); // the log keeps it even if the game exits without flushing
}

} // namespace Vulkan
