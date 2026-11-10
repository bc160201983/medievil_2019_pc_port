// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <chrono>
#include <cstdlib>
#include <deque>

#include <condition_variable>

#include "core/libraries/videoout/buffer.h"
#include "video_core/renderer_vulkan/host_passes/fsr_pass.h"
#include "video_core/renderer_vulkan/host_passes/pp_pass.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_runtime.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_swapchain.h"
#include "video_core/texture_cache/texture_cache.h"

namespace Frontend {
class WindowSDL;
}

namespace AmdGpu {
struct Liverpool;
}

namespace Vulkan {

struct Frame {
    u32 width;
    u32 height;
    VmaAllocation allocation;
    vk::Image image;
    vk::ImageView image_view;
    vk::Fence present_done;
    vk::Semaphore ready_semaphore;
    u64 ready_tick;
    bool is_hdr{false};
    u8 id{};

};

enum SchedulerType {
    Draw,
    Present,
    CpuFlip,
};

class Rasterizer;

class Presenter {
public:
    Presenter(Frontend::WindowSDL& window, AmdGpu::Liverpool* liverpool);
    ~Presenter();

    HostPasses::PostProcessingPass::Settings& GetPPSettingsRef() {
        return pp_settings;
    }

    HostPasses::FsrPass::Settings& GetFsrSettingsRef() {
        return fsr_settings;
    }

    Frontend::WindowSDL& GetWindow() const {
        return window;
    }

    Rasterizer& GetRasterizer() const {
        return *rasterizer.get();
    }

    bool IsHDRSupported() const {
        return swapchain.HasHDR();
    }

    void SetHDR(bool enable) {
        if (!IsHDRSupported()) {
            return;
        }
        swapchain.SetHDR(enable);
        pp_settings.hdr = enable ? 1 : 0;
    }

    VideoCore::Image& RegisterVideoOutSurface(
        const Libraries::VideoOut::BufferAttributeGroup& attribute, VAddr cpu_address) {
        vo_buffers_addr.emplace_back(cpu_address);
        auto desc = VideoCore::TextureCache::ImageDesc{attribute, cpu_address};
        const auto image_id = texture_cache.FindImage(desc);
        auto& image = texture_cache.GetImage(image_id);
        image.usage.vo_surface = 1u;
        return image;
    }

    bool IsVideoOutSurface(const AmdGpu::ColorBuffer& color_buffer) const;

    Frame* PrepareFrame(const Libraries::VideoOut::BufferAttributeGroup& attribute,
                        VAddr cpu_address);

    Frame* PrepareBlankFrame(bool present_thread);

    void Present(Frame* frame, bool is_reusing_frame = false, bool is_game_frame = true);
    /// gow3: a black frame with the loading screen (Gow3Overlay::SetLoading) only.
    void PresentLoadingFrame();
    Frame* PrepareLastFrame();

    /// gow3 GOW3_LIMIT_IN_PRESENT: the next Present waits until `deadline` just before
    /// vkQueuePresentKHR, after its acquire and blit (where the RTSS limiter waits); {}: none.
    void SetPresentDeadline(std::chrono::steady_clock::time_point deadline) {
        present_deadline = deadline;
    }

    /// gow3 Reflex (GOW3_REFLEX): the driver's minimum frame interval; its sleep after a present.
    void SetLatencyInterval(u32 interval_us) {
        swapchain.SetLatencyInterval(interval_us);
    }
    bool LatencySleep() {
        return swapchain.LatencySleep();
    }
    /// gow3: the driver has Reflex (VK_NV_low_latency2) enabled for this device.
    bool ReflexEnabled() const {
        return instance.IsLowLatency2Enabled();
    }
    /// gow3 GOW3_REFLEX_SLEEP_BEFORE_PRESENT=1: the Reflex sleep runs inside Present, right before
    /// vkQueuePresentKHR (where the RTSS Reflex limiter sleeps), not after it.
    static bool ReflexSleepBeforePresent() {
        static const bool enabled = [] {
            const char* env = std::getenv("GOW3_REFLEX_SLEEP_BEFORE_PRESENT");
            return env && env[0] == '1';
        }();
        return enabled;
    }

    /// gow3 flip trace: the guest flip argument of the next Present (display times).
    void SetPresentTag(s64 tag) {
        swapchain.SetPresentTag(tag);
    }

    /// gow3 newest-frame mode (GOW3_NEWEST_FRAME): waits until the frame's image is finished on
    /// the GPU or `until` passes; true when finished. Frames finish in the order prepared.
    bool WaitFrameReady(Frame* frame, std::chrono::steady_clock::time_point until);
    /// gow3 newest-frame mode: returns a prepared frame to the pool without showing it.
    void DiscardFrame(Frame* frame);

    /// gow3 flip trace (GOW3_FLIP_TRACE): notes "gpu_ready" once the GPU has finished the frame's
    /// work (the game's rendering, then the copy out of its buffer), on a thread of its own.
    void TraceGpuReady(const Frame* frame, s64 flip_arg);

private:
    Frame* GetRenderFrame();

    void RecreateFrame(Frame* frame, u32 width, u32 height);

    void SetExpectedGameSize(s32 width, s32 height);

private:
    float expected_ratio{1920.0 / 1080.0f};
    u32 expected_frame_width{1920};
    u32 expected_frame_height{1080};

    Frontend::WindowSDL& window;
    Instance instance;
    HostPasses::FsrPass fsr_pass;
    HostPasses::FsrPass::Settings fsr_settings{};
    HostPasses::PostProcessingPass::Settings pp_settings{};
    HostPasses::PostProcessingPass pp_pass;
    AmdGpu::Liverpool* liverpool;
    Scheduler draw_scheduler;
    std::deque<u64> recent_frame_ticks; ///< gow3: GOW3_FRAMES_AHEAD bound (PrepareFrame)
    Scheduler present_scheduler;
    Scheduler flip_scheduler;
    Swapchain swapchain;
    Runtime runtime;
    std::unique_ptr<Rasterizer> rasterizer;
    VideoCore::TextureCache& texture_cache;
    vk::UniqueCommandPool command_pool;
    std::vector<Frame> present_frames;
    std::queue<Frame*> free_queue;
    Frame* last_submit_frame;
    std::mutex free_mutex;
    std::condition_variable free_cv;
    std::condition_variable_any frame_cv;
    std::vector<VAddr> vo_buffers_addr;

    /// Waits until `tick` of `semaphore` (one of the schedulers' work semaphores) or `until`.
    bool WaitReady(vk::Semaphore semaphore, u64 tick, std::chrono::steady_clock::time_point until);
    void GpuReadyThread(std::stop_token token);
    struct GpuReadyEntry {
        vk::Semaphore semaphore;
        u64 tick;
        s64 flip_arg;
    };
    std::mutex gpu_ready_mutex;
    std::condition_variable_any gpu_ready_cv;
    std::deque<GpuReadyEntry> gpu_ready_queue;
    std::chrono::steady_clock::time_point present_deadline{}; ///< SetPresentDeadline
    std::jthread gpu_ready_thread; ///< last: stopped first
};

} // namespace Vulkan
