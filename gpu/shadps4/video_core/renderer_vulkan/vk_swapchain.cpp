// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "gow3_flip_trace.h"
#include <algorithm>
#include <string_view>
#include <cstdio>
#include <limits>
#include "common/assert.h"
#include "common/logging/log.h"
#include "core/emulator_settings.h"
#include "gow3_settings.h"
#include "imgui/renderer/imgui_core.h"
#include "sdl_window.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_swapchain.h"

namespace Vulkan {

static constexpr vk::SurfaceFormatKHR SURFACE_FORMAT_HDR = {
    .format = vk::Format::eA2B10G10R10UnormPack32,
    .colorSpace = vk::ColorSpaceKHR::eHdr10St2084EXT,
};

Swapchain::Swapchain(const Instance& instance_, const Frontend::WindowSDL& window_)
    : instance{instance_}, window{window_}, surface{CreateSurface(instance.GetInstance(), window)} {
    FindPresentFormat();
    FindPresentMode();

    Create(window.GetWidth(), window.GetHeight());
    ImGui::Core::Initialize(instance, window, image_count, surface_format.format);
}

void Swapchain::DisplayWaitThread(std::stop_token token) {
    while (!token.stop_requested()) {
        PendingPresent next{};
        {
            std::unique_lock lock{pending_mutex};
            if (!pending_cv.wait(lock, token, [&] { return !pending.empty(); })) {
                return;
            }
            next = pending.front();
            pending.pop_front();
        }
        for (int tries = 0; tries < 100 && !token.stop_requested(); ++tries) {
            std::scoped_lock lock{wait_mutex};
            if (swapchain != next.swapchain) {
                break; // recreated: its presents are gone
            }
            const auto result = instance.GetDevice().waitForPresentKHR(swapchain, next.id, 2000000);
            if (result == vk::Result::eSuccess) {
                Gow3FlipTrace::Note("displayed", next.tag);
                break;
            }
            if (result != vk::Result::eTimeout) {
                break;
            }
        }
    }
}

Swapchain::~Swapchain() {
    display_wait_thread = {};
    Destroy();
    instance.GetInstance().destroySurfaceKHR(surface);
}

void Swapchain::Create(u32 width_, u32 height_) {
    width = width_;
    height = height_;
    needs_recreation = false;

    Destroy();

    SetSurfaceProperties();

    const std::array queue_family_indices = {
        instance.GetGraphicsQueueFamilyIndex(),
        instance.GetPresentQueueFamilyIndex(),
    };

    const bool exclusive = queue_family_indices[0] == queue_family_indices[1];
    const u32 queue_family_indices_count = exclusive ? 1u : 2u;
    const vk::SharingMode sharing_mode =
        exclusive ? vk::SharingMode::eExclusive : vk::SharingMode::eConcurrent;
    const auto format = needs_hdr ? SURFACE_FORMAT_HDR : surface_format;
    const vk::SwapchainLatencyCreateInfoNV latency_info = {.latencyModeEnable = true};
    const vk::SwapchainCreateInfoKHR swapchain_info = {
        .pNext = instance.IsLowLatency2Enabled() ? &latency_info : nullptr,
        .surface = surface,
        .minImageCount = image_count,
        .imageFormat = format.format,
        .imageColorSpace = format.colorSpace,
        .imageExtent = extent,
        .imageArrayLayers = 1,
        .imageUsage = vk::ImageUsageFlagBits::eColorAttachment |
                      vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst,
        .imageSharingMode = sharing_mode,
        .queueFamilyIndexCount = queue_family_indices_count,
        .pQueueFamilyIndices = queue_family_indices.data(),
        .preTransform = transform,
        .compositeAlpha = composite_alpha,
        .presentMode = present_mode,
        .clipped = true,
        .oldSwapchain = nullptr,
    };

    auto [swapchain_result, chain] = instance.GetDevice().createSwapchainKHR(swapchain_info);
    ASSERT_MSG(swapchain_result == vk::Result::eSuccess, "Failed to create swapchain: {}",
               vk::to_string(swapchain_result));
    swapchain = chain;

    SetupImages();
    RefreshSemaphores();
    ApplyLatencyMode();
}

void Swapchain::SetLatencyInterval(u32 interval_us) {
    latency_interval_us = interval_us;
    ApplyLatencyMode();
}

void Swapchain::ApplyLatencyMode() {
    if (!instance.IsLowLatency2Enabled() || !swapchain) {
        return;
    }
    const vk::LatencySleepModeInfoNV mode = {
        .lowLatencyMode = true,
        .lowLatencyBoost = false,
        .minimumIntervalUs = latency_interval_us,
    };
    const auto result = instance.GetDevice().setLatencySleepModeNV(swapchain, &mode);
    std::printf("Swapchain: Reflex sleep mode, minimum interval %u us (%s)\n", latency_interval_us,
                vk::to_string(result).c_str());
}

void Swapchain::LatencyMarker(vk::LatencyMarkerNV marker, u64 id) {
    if (!instance.IsLowLatency2Enabled() || !swapchain) {
        return;
    }
    const vk::SetLatencyMarkerInfoNV info = {.presentID = id, .marker = marker};
    instance.GetDevice().setLatencyMarkerNV(swapchain, &info);
}

bool Swapchain::LatencySleep() {
    if (!instance.IsLowLatency2Enabled() || !swapchain) {
        return false;
    }
    const vk::Device device = instance.GetDevice();
    if (!latency_semaphore) {
        const vk::SemaphoreTypeCreateInfo type = {.semaphoreType = vk::SemaphoreType::eTimeline,
                                                  .initialValue = 0};
        const vk::SemaphoreCreateInfo info = {.pNext = &type};
        auto [result, semaphore] = device.createSemaphore(info);
        if (result != vk::Result::eSuccess) {
            return false;
        }
        latency_semaphore = semaphore;
    }
    const vk::LatencySleepInfoNV sleep = {.signalSemaphore = latency_semaphore, .value = ++latency_value};
    if (device.latencySleepNV(swapchain, &sleep) == vk::Result::eSuccess) {
        const vk::SemaphoreWaitInfo wait = {
            .semaphoreCount = 1, .pSemaphores = &latency_semaphore, .pValues = &latency_value};
        (void)device.waitSemaphores(&wait, 100'000'000); // at most 100 ms
    }
    LatencyMarker(vk::LatencyMarkerNV::eSimulationStart, present_id + 1);
    return true;
}

void Swapchain::Recreate(u32 width_, u32 height_) {
    LOG_DEBUG(Render_Vulkan, "Recreate the swapchain: width={} height={} HDR={}", width_, height_,
              needs_hdr);
    Create(width_, height_);
}

void Swapchain::SetHDR(bool hdr) {
    if (needs_hdr == hdr) {
        return;
    }

    auto result = instance.GetDevice().waitIdle();
    if (result != vk::Result::eSuccess) {
        LOG_WARNING(ImGui, "Failed to wait for Vulkan device idle on mode change: {}",
                    vk::to_string(result));
    }

    needs_hdr = hdr;
    Recreate(width, height);
    ImGui::Core::OnSurfaceFormatChange(needs_hdr ? SURFACE_FORMAT_HDR.format
                                                 : surface_format.format);
}

bool Swapchain::AcquireNextImage() {
    vk::Device device = instance.GetDevice();
    vk::Result result =
        device.acquireNextImageKHR(swapchain, std::numeric_limits<u64>::max(),
                                   image_acquired[frame_index], VK_NULL_HANDLE, &image_index);

    switch (result) {
    case vk::Result::eSuccess:
        break;
    case vk::Result::eSuboptimalKHR:
    case vk::Result::eErrorSurfaceLostKHR:
    case vk::Result::eErrorOutOfDateKHR:
    case vk::Result::eErrorUnknown:
        needs_recreation = true;
        break;
    default:
        LOG_CRITICAL(Render_Vulkan, "Swapchain acquire returned unknown result {}",
                     vk::to_string(result));
        UNREACHABLE();
        break;
    }

    return !needs_recreation;
}

bool Swapchain::Present() {
    vk::PresentInfoKHR present_info = {
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &present_ready[image_index],
        .swapchainCount = 1,
        .pSwapchains = &swapchain,
        .pImageIndices = &image_index,
    };
    const u64 id = ++present_id;
    const vk::PresentIdKHR present_id_info = {
        .swapchainCount = 1,
        .pPresentIds = &id,
    };
    const bool track = instance.IsPresentWaitEnabled();
    if (track || instance.IsPresentIdEnabled()) {
        present_info.pNext = &present_id_info;
    }
    // gow3 Reflex markers of this frame (its simulation started after the last LatencySleep).
    LatencyMarker(vk::LatencyMarkerNV::eSimulationEnd, id);
    LatencyMarker(vk::LatencyMarkerNV::eRendersubmitStart, id);
    LatencyMarker(vk::LatencyMarkerNV::eRendersubmitEnd, id);
    LatencyMarker(vk::LatencyMarkerNV::ePresentStart, id);
    if (track) {
        if (!display_wait_thread.joinable()) {
            display_wait_thread = std::jthread([this](std::stop_token token) { DisplayWaitThread(token); });
        }
    }

    auto result = instance.GetPresentQueue().presentKHR(present_info);
    LatencyMarker(vk::LatencyMarkerNV::ePresentEnd, id);
    if (track && present_tag >= 0 && result == vk::Result::eSuccess) {
        {
            std::scoped_lock lock{pending_mutex};
            pending.push_back({swapchain, id, present_tag});
        }
        pending_cv.notify_one();
    }
    present_tag = -1;
    if (result == vk::Result::eErrorOutOfDateKHR || result == vk::Result::eSuboptimalKHR) {
        needs_recreation = true;
    } else {
        ASSERT_MSG(result == vk::Result::eSuccess, "Swapchain presentation failed: {}",
                   vk::to_string(result));
    }

    frame_index = (frame_index + 1) % image_count;

    return !needs_recreation;
}

void Swapchain::FindPresentFormat() {
    const auto [formats_result, formats] =
        instance.GetPhysicalDevice().getSurfaceFormatsKHR(surface);
    ASSERT_MSG(formats_result == vk::Result::eSuccess, "Failed to query surface formats: {}",
               vk::to_string(formats_result));

    // Check if the device supports HDR formats. Here we care of Rec.2020 PQ only as it is expected
    // game output. Other variants as e.g. linear Rec.2020 will require additional color space
    // rotation
    supports_hdr =
        std::find_if(formats.begin(), formats.end(), [](const vk::SurfaceFormatKHR& format) {
            return format == SURFACE_FORMAT_HDR;
        }) != formats.end();
    // Also make sure that user allowed us to use HDR
    supports_hdr &= EmulatorSettings.IsHdrAllowed();

    // If there is a single undefined surface format, the device doesn't care, so we'll just use
    // RGBA sRGB.
    if (formats[0].format == vk::Format::eUndefined) {
        surface_format.format = vk::Format::eR8G8B8A8Unorm;
        surface_format.colorSpace = vk::ColorSpaceKHR::eSrgbNonlinear;
        return;
    }

    // Try to find a suitable format.
    for (const vk::SurfaceFormatKHR& sformat : formats) {
        vk::Format format = sformat.format;
        if (format != vk::Format::eR8G8B8A8Unorm && format != vk::Format::eB8G8R8A8Unorm) {
            continue;
        }

        surface_format.format = format;
        surface_format.colorSpace = sformat.colorSpace;
        return;
    }

    UNREACHABLE_MSG("Unable to find required swapchain format!");
}

void Swapchain::FindPresentMode() {
    const auto [modes_result, modes] =
        instance.GetPhysicalDevice().getSurfacePresentModesKHR(surface);
    if (modes_result != vk::Result::eSuccess) {
        LOG_ERROR(Render, "Failed to query available present modes, falling back to Fifo as "
                          "guaranteed supported option.");
        present_mode = vk::PresentModeKHR::eFifo;
        return;
    }

    // gow3: VSync on is Fifo (no tearing, waits for the display); off is Mailbox when the
    // driver has it, else Immediate (tearing allowed), else Fifo.
    vsync = Gow3Settings::Get().vsync;
    const auto has = [&](vk::PresentModeKHR mode) { return std::ranges::find(modes, mode) != modes.cend(); };
    // gow3 GOW3_PRESENT_MODE (launcher > Presentation > Present mode): a fixed Vulkan present
    // mode instead of following VSync; unsupported modes fall back to Fifo.
    if (const char* forced = std::getenv("GOW3_PRESENT_MODE"); forced && forced[0]) {
        const std::string_view name{forced};
        const auto mode = name == "fifo_relaxed" ? vk::PresentModeKHR::eFifoRelaxed
                          : name == "mailbox"    ? vk::PresentModeKHR::eMailbox
                          : name == "immediate"  ? vk::PresentModeKHR::eImmediate
                                                 : vk::PresentModeKHR::eFifo;
        // Immediate asks for no wait on the display: without it (gamescope offers no Immediate),
        // Mailbox keeps that, where Fifo would cap the frame rate at the refresh.
        present_mode = has(mode) ? mode
                       : mode == vk::PresentModeKHR::eImmediate && has(vk::PresentModeKHR::eMailbox)
                           ? vk::PresentModeKHR::eMailbox
                           : vk::PresentModeKHR::eFifo;
        std::printf("Swapchain: present mode %s (GOW3_PRESENT_MODE=%s%s)\n", vk::to_string(present_mode).c_str(),
                    forced, present_mode == mode ? "" : ", not offered by the surface");
        return;
    }
    present_mode = vsync ? vk::PresentModeKHR::eFifo
                   : has(vk::PresentModeKHR::eMailbox) ? vk::PresentModeKHR::eMailbox
                   : has(vk::PresentModeKHR::eImmediate) ? vk::PresentModeKHR::eImmediate
                                                         : vk::PresentModeKHR::eFifo;
    std::printf("Swapchain: VSync %s (%s)\n", vsync ? "on" : "off", vk::to_string(present_mode).c_str());
}

void Swapchain::UpdateVsync() {
    if (Gow3Settings::Get().vsync == vsync) {
        return;
    }
    if (const auto result = instance.GetDevice().waitIdle(); result != vk::Result::eSuccess) {
        LOG_WARNING(Render_Vulkan, "Failed to wait for the device before a VSync change: {}",
                    vk::to_string(result));
    }
    FindPresentMode();
    Recreate(width, height);
}

void Swapchain::SetSurfaceProperties() {
    const auto [capabilities_result, capabilities] =
        instance.GetPhysicalDevice().getSurfaceCapabilitiesKHR(surface);
    ASSERT_MSG(capabilities_result == vk::Result::eSuccess,
               "Failed to query surface capabilities: {}", vk::to_string(capabilities_result));

    extent = capabilities.currentExtent;
    if (capabilities.currentExtent.width == std::numeric_limits<u32>::max()) {
        extent.width = std::max(capabilities.minImageExtent.width,
                                std::min(capabilities.maxImageExtent.width, width));
        extent.height = std::max(capabilities.minImageExtent.height,
                                 std::min(capabilities.maxImageExtent.height, height));
    }

    // Select number of images in swap chain, we prefer one buffer in the background to work on
    image_count = capabilities.minImageCount + 1;
    if (capabilities.maxImageCount > 0) {
        image_count = std::min(image_count, capabilities.maxImageCount);
    }

    // Prefer identity transform if possible
    transform = vk::SurfaceTransformFlagBitsKHR::eIdentity;
    if (!(capabilities.supportedTransforms & transform)) {
        transform = capabilities.currentTransform;
    }

    // Opaque is not supported everywhere.
    composite_alpha = vk::CompositeAlphaFlagBitsKHR::eOpaque;
    if (!(capabilities.supportedCompositeAlpha & vk::CompositeAlphaFlagBitsKHR::eOpaque)) {
        composite_alpha = vk::CompositeAlphaFlagBitsKHR::eInherit;
    }
}

void Swapchain::Destroy() {
    std::scoped_lock wait_lock{wait_mutex};
    vk::Device device = instance.GetDevice();
    const auto wait_result = device.waitIdle();
    if (wait_result != vk::Result::eSuccess) {
        LOG_WARNING(Render_Vulkan, "Failed to wait for device to become idle: {}",
                    vk::to_string(wait_result));
    }

    for (auto& image_view : images_view) {
        device.destroyImageView(image_view);
    }
    images_view.clear();

    if (swapchain) {
        device.destroySwapchainKHR(swapchain);
    }

    for (const auto& sem : image_acquired) {
        device.destroySemaphore(sem);
    }
    for (const auto& sem : present_ready) {
        device.destroySemaphore(sem);
    }

    image_acquired.clear();
    present_ready.clear();
}

void Swapchain::RefreshSemaphores() {
    const vk::Device device = instance.GetDevice();
    image_acquired.resize(image_count);
    present_ready.resize(image_count);

    for (vk::Semaphore& semaphore : image_acquired) {
        auto [semaphore_result, sem] = device.createSemaphore({});
        ASSERT_MSG(semaphore_result == vk::Result::eSuccess,
                   "Failed to create image acquired semaphore: {}",
                   vk::to_string(semaphore_result));
        semaphore = sem;
    }
    for (vk::Semaphore& semaphore : present_ready) {
        auto [semaphore_result, sem] = device.createSemaphore({});
        ASSERT_MSG(semaphore_result == vk::Result::eSuccess,
                   "Failed to create present ready semaphore: {}", vk::to_string(semaphore_result));
        semaphore = sem;
    }

    for (u32 i = 0; i < image_count; ++i) {
        SetObjectName(device, image_acquired[i], "Swapchain Semaphore: image_acquired {}", i);
        SetObjectName(device, present_ready[i], "Swapchain Semaphore: present_ready {}", i);
    }
}

void Swapchain::SetupImages() {
    vk::Device device = instance.GetDevice();
    auto [images_result, imgs] = device.getSwapchainImagesKHR(swapchain);
    ASSERT_MSG(images_result == vk::Result::eSuccess, "Failed to create swapchain images: {}",
               vk::to_string(images_result));
    images = std::move(imgs);
    image_count = static_cast<u32>(images.size());
    images_view.resize(image_count);
    for (u32 i = 0; i < image_count; ++i) {
        if (images_view[i]) {
            device.destroyImageView(images_view[i]);
        }
        auto [im_view_result, im_view] = device.createImageView(vk::ImageViewCreateInfo{
            .image = images[i],
            .viewType = vk::ImageViewType::e2D,
            .format = needs_hdr ? SURFACE_FORMAT_HDR.format : surface_format.format,
            .subresourceRange =
                {
                    .aspectMask = vk::ImageAspectFlagBits::eColor,
                    .levelCount = 1,
                    .layerCount = 1,
                },
        });
        ASSERT_MSG(im_view_result == vk::Result::eSuccess, "Failed to create image view: {}",
                   vk::to_string(im_view_result));
        images_view[i] = im_view;
    }

    for (u32 i = 0; i < image_count; ++i) {
        SetObjectName(device, images[i], "Swapchain Image {}", i);
        SetObjectName(device, images_view[i], "Swapchain ImageView {}", i);
    }
}

} // namespace Vulkan
