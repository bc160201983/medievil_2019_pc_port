// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>
#include "common/types.h"
#include "video_core/renderer_vulkan/vk_common.h"

namespace Frontend {
class WindowSDL;
}

namespace Vulkan {

class Instance;
class Scheduler;

class Swapchain {
public:
    explicit Swapchain(const Instance& instance, const Frontend::WindowSDL& window);
    ~Swapchain();

    /// Creates (or recreates) the swapchain with a given size.
    void Create(u32 width, u32 height);

    /// Recreates the swapchain with a given size and current surface.
    void Recreate(u32 width, u32 height);

    /// Acquires the next image in the swapchain.
    bool AcquireNextImage();

    /// Presents the current image and move to the next one
    bool Present();

    /// gow3 flip trace: the guest flip argument of the next present (-1: not a game flip).
    void SetPresentTag(s64 tag) {
        present_tag = tag;
    }
    [[nodiscard]] s64 PresentTag() const {
        return present_tag;
    }

    /// gow3 Reflex (VK_NV_low_latency2): the driver's minimum frame interval (0: no limit),
    /// applied now and to every recreated swapchain.
    void SetLatencyInterval(u32 interval_us);
    /// gow3 Reflex: blocks until the driver says the next frame should start, then marks its
    /// simulation start. False without Reflex.
    bool LatencySleep();

    vk::SurfaceKHR GetSurface() const {
        return surface;
    }

    vk::Image Image() const {
        return images[image_index];
    }

    vk::ImageView ImageView() const {
        return images_view[image_index];
    }

    vk::SurfaceFormatKHR GetSurfaceFormat() const {
        return surface_format;
    }

    vk::SwapchainKHR GetHandle() const {
        return swapchain;
    }

    u32 GetWidth() const {
        return width;
    }

    u32 GetHeight() const {
        return height;
    }

    u32 GetImageCount() const {
        return image_count;
    }

    u32 GetFrameIndex() const {
        return frame_index;
    }

    vk::Extent2D GetExtent() const {
        return extent;
    }

    [[nodiscard]] vk::Semaphore GetImageAcquiredSemaphore() const {
        return image_acquired[frame_index];
    }

    [[nodiscard]] vk::Semaphore GetPresentReadySemaphore() const {
        return present_ready[image_index];
    }

    bool HasHDR() const {
        return supports_hdr;
    }

    void SetHDR(bool hdr);

    /// gow3: VSync from the in-game menu (Gow3Settings::vsync); recreates the swapchain when it
    /// changed. Present thread only.
    void UpdateVsync();

    bool GetHDR() const {
        return needs_hdr;
    }

private:
    /// Selects the best available swapchain image format
    void FindPresentFormat();

    /// Selects the best available present mode
    void FindPresentMode();

    /// Sets the surface properties according to device capabilities
    void SetSurfaceProperties();

    /// Destroys current swapchain resources
    void Destroy();

    /// Performs creation of image views and framebuffers from the swapchain images
    void SetupImages();

    /// Creates the image acquired and present ready semaphores
    void RefreshSemaphores();

private:
    const Instance& instance;
    const Frontend::WindowSDL& window;
    vk::SwapchainKHR swapchain{};
    vk::SurfaceKHR surface{};
    vk::SurfaceFormatKHR surface_format;
    vk::Format view_format;
    vk::PresentModeKHR present_mode;
    bool vsync = true; ///< the setting present_mode was chosen for
    vk::Extent2D extent;
    vk::SurfaceTransformFlagBitsKHR transform;
    vk::CompositeAlphaFlagBitsKHR composite_alpha;
    std::vector<vk::Image> images;
    std::vector<vk::ImageView> images_view;
    std::vector<vk::Semaphore> image_acquired;
    std::vector<vk::Semaphore> present_ready;
    u32 width = 0;
    u32 height = 0;
    u32 image_count = 0;
    u32 image_index = 0;
    u32 frame_index = 0;
    bool needs_recreation = true;
    bool needs_hdr = false;    // The game requested HDR swapchain
    bool supports_hdr = false; // SC supports HDR output

    // gow3 flip trace display times (Instance::IsPresentWaitEnabled): presents waited for in
    // order on their own thread; Destroy holds wait_mutex so no wait runs on a dying swapchain.
    void DisplayWaitThread(std::stop_token token);
    struct PendingPresent {
        vk::SwapchainKHR swapchain;
        u64 id;
        s64 tag;
    };
    s64 present_tag = -1;
    u64 present_id = 0;
    void ApplyLatencyMode();
    void LatencyMarker(vk::LatencyMarkerNV marker, u64 id);
    u32 latency_interval_us = 0;
    vk::Semaphore latency_semaphore{};
    u64 latency_value = 0;
    std::mutex wait_mutex;
    std::mutex pending_mutex;
    std::condition_variable_any pending_cv;
    std::deque<PendingPresent> pending;
    std::jthread display_wait_thread;
};

} // namespace Vulkan
