// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/debug.h"
#include "common/elf_info.h"
#include "common/io_file.h"
#include "common/path_util.h"
#include "common/singleton.h"
#include "core/debug_state.h"
#include "core/emulator_settings.h"
#include "sdl_window.h"
#include "video_core/buffer_cache/buffer.h"
#include "video_core/renderdoc.h"
#include "video_core/renderer_vulkan/vk_platform.h"
#include "gow3_overlay.h"
#include "gow3_settings.h"
#include "gow3_graphics.h"
#include "video_core/renderer_vulkan/vk_temporal_upscaler.h"
#include "video_core/renderer_vulkan/vk_presenter.h"
#include "video_core/renderer_vulkan/vk_warmup_inbox.h"
#include "video_core/renderer_vulkan/vk_frame_gen.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"
#include "video_core/texture_cache/image.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <csetjmp>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <span>
#include <sstream>
#include <system_error>
#include <vector>
#include <thread>
#include <vk_mem_alloc.h>
#include <miniz.h>
#ifdef MemoryBarrier
#undef MemoryBarrier // gow3: winnt.h macro (through fmt), clashes with vk::MemoryBarrier
#endif

namespace Vulkan {

bool CanBlitToSwapchain(const vk::PhysicalDevice physical_device, vk::Format format) {
    const vk::FormatProperties props{physical_device.getFormatProperties(format)};
    return static_cast<bool>(props.optimalTilingFeatures & vk::FormatFeatureFlagBits::eBlitDst);
}

[[nodiscard]] vk::ImageSubresourceLayers MakeImageSubresourceLayers() {
    return vk::ImageSubresourceLayers{
        .aspectMask = vk::ImageAspectFlagBits::eColor,
        .mipLevel = 0,
        .baseArrayLayer = 0,
        .layerCount = 1,
    };
}

[[nodiscard]] vk::ImageBlit MakeImageBlit(s32 frame_width, s32 frame_height, s32 dst_width,
                                          s32 dst_height, s32 offset_x, s32 offset_y) {
    return vk::ImageBlit{
        .srcSubresource = MakeImageSubresourceLayers(),
        .srcOffsets =
            std::array{
                vk::Offset3D{
                    .x = 0,
                    .y = 0,
                    .z = 0,
                },
                vk::Offset3D{
                    .x = frame_width,
                    .y = frame_height,
                    .z = 1,
                },
            },
        .dstSubresource = MakeImageSubresourceLayers(),
        .dstOffsets =
            std::array{
                vk::Offset3D{
                    .x = offset_x,
                    .y = offset_y,
                    .z = 0,
                },
                vk::Offset3D{
                    .x = offset_x + dst_width,
                    .y = offset_y + dst_height,
                    .z = 1,
                },
            },
    };
}

[[nodiscard]] vk::ImageBlit MakeImageBlitStretch(s32 frame_width, s32 frame_height,
                                                 s32 swapchain_width, s32 swapchain_height) {
    return MakeImageBlit(frame_width, frame_height, swapchain_width, swapchain_height, 0, 0);
}

static vk::Rect2D FitImage(s32 frame_width, s32 frame_height, s32 swapchain_width,
                           s32 swapchain_height) {
    float frame_aspect = static_cast<float>(frame_width) / frame_height;
    float swapchain_aspect = static_cast<float>(swapchain_width) / swapchain_height;

    u32 dst_width = swapchain_width;
    u32 dst_height = swapchain_height;

    if (frame_aspect > swapchain_aspect) {
        dst_height = static_cast<s32>(swapchain_width / frame_aspect);
    } else {
        dst_width = static_cast<s32>(swapchain_height * frame_aspect);
    }

    const s32 offset_x = (swapchain_width - dst_width) / 2;
    const s32 offset_y = (swapchain_height - dst_height) / 2;

    return vk::Rect2D{{offset_x, offset_y}, {dst_width, dst_height}};
}

[[nodiscard]] vk::ImageBlit MakeImageBlitFit(s32 frame_width, s32 frame_height, s32 swapchain_width,
                                             s32 swapchain_height) {
    const auto& dst_rect = FitImage(frame_width, frame_height, swapchain_width, swapchain_height);

    return MakeImageBlit(frame_width, frame_height, dst_rect.extent.width, dst_rect.extent.height,
                         dst_rect.offset.x, dst_rect.offset.y);
}

// gow3: screenshot capture and ImGui overlays removed; the port presents directly.

Presenter::Presenter(Frontend::WindowSDL& window_, AmdGpu::Liverpool* liverpool_)
    : window{window_}, liverpool{liverpool_},
      instance{window, EmulatorSettings.GetGpuId(), EmulatorSettings.IsVkValidationEnabled(),
               EmulatorSettings.IsVkCrashDiagnosticEnabled()},
      draw_scheduler{instance, true}, present_scheduler{instance}, flip_scheduler{instance},
      swapchain{instance, window}, runtime{instance, draw_scheduler},
      rasterizer{std::make_unique<Rasterizer>(instance, draw_scheduler, runtime, liverpool)},
      texture_cache{rasterizer->GetTextureCache()} {
    frame_gen = std::make_unique<FrameGen>(instance);
    const u32 num_images = swapchain.GetImageCount();
    const vk::Device device = instance.GetDevice();

    // Create presentation frames.
    present_frames.resize(num_images);
    for (u32 i = 0; i < num_images; i++) {
        Frame& frame = present_frames[i];
        frame.id = i;
        auto fence = Check<"create present done fence">(
            device.createFence({.flags = vk::FenceCreateFlagBits::eSignaled}));
        frame.present_done = fence;
        free_queue.push(&frame);
    }

    // Frame generation runs the passes twice a frame (its HUD-less copy too).
    fsr_pass.Create(device, instance.GetAllocator(),
                    frame_gen->Enabled() ? num_images * 2 : num_images);
    pp_pass.Create(device, swapchain.GetSurfaceFormat().format);
    Gow3Overlay::Init(instance, swapchain.GetSurfaceFormat().format, num_images);

    const bool binary_cache = EmulatorSettingsImpl::Flag("GOW3_PIPELINE_BINARIES", true) &&
                              EmulatorSettings.IsPipelineCacheEnabled();
    const bool capture_only = EmulatorSettingsImpl::Flag("GOW3_PIPELINE_BINARY_ISOLATED", false) &&
                              EmulatorSettingsImpl::Flag("GOW3_PIPELINE_CAPTURE_ONLY", false);
    if (UseOnDemandPipelines(binary_cache, instance.GetPipelineBinaryGlobalKey().keySize != 0,
                            capture_only)) {
        rasterizer->GetPipelineCache().StartOnDemand();
        return;
    }

    // gow3: by default the cached pipelines load while the game runs (a corner note shows the
    // progress). Otherwise rebuilding them takes up to minutes before the first game frame:
    // show what is happening instead of a black window.
    const char* background_warmup = std::getenv("GOW3_BACKGROUND_WARMUP");
    if ((!background_warmup || *background_warmup != '0') &&
        rasterizer->GetPipelineCache().StartBackgroundWarmUp()) {
        return;
    }
    Gow3Overlay::SetLoading(true, 0, 0);
    PresentLoadingFrame();
    auto last = std::chrono::steady_clock::now();
    rasterizer->GetPipelineCache().WarmUp([&](u32 done, u32 total) {
        const auto now = std::chrono::steady_clock::now();
        if (now - last < std::chrono::milliseconds(250)) {
            return;
        }
        last = now;
        Gow3Overlay::SetLoading(true, done, total);
        PresentLoadingFrame();
    });
    Gow3Overlay::SetLoading(false, 0, 0);

}

Presenter::~Presenter() {

    draw_scheduler.Finish();
    present_scheduler.Finish();
    flip_scheduler.Finish();
    Check(draw_scheduler.CommandBuffer().reset());
    Check(present_scheduler.CommandBuffer().reset());
    Check(flip_scheduler.CommandBuffer().reset());

    const vk::Device device = instance.GetDevice();
    for (auto& frame : present_frames) {
        vmaDestroyImage(instance.GetAllocator(), frame.image, frame.allocation);
        if (frame.gen_image) {
            vmaDestroyImage(instance.GetAllocator(), frame.gen_image, frame.gen_allocation);
        }
        if (frame.hud_image) {
            device.destroyImageView(frame.hud_view);
            vmaDestroyImage(instance.GetAllocator(), frame.hud_image, frame.hud_allocation);
        }
        device.destroyImageView(frame.image_view);
        device.destroyFence(frame.present_done);
    }
}

bool Presenter::IsVideoOutSurface(const AmdGpu::ColorBuffer& color_buffer) const {
    return std::ranges::find(vo_buffers_addr, color_buffer.Address()) != vo_buffers_addr.cend();
}

void Presenter::RecreateFrame(Frame* frame, u32 width, u32 height) {
    const vk::Device device = instance.GetDevice();
    if (frame->image_view) {
        device.destroyImageView(frame->image_view);
    }
    if (frame->image) {
        vmaDestroyImage(instance.GetAllocator(), frame->image, frame->allocation);
    }
    if (frame->gen_image) {
        vmaDestroyImage(instance.GetAllocator(), frame->gen_image, frame->gen_allocation);
        frame->gen_image = vk::Image{};
    }
    if (frame->hud_image) {
        device.destroyImageView(frame->hud_view);
        vmaDestroyImage(instance.GetAllocator(), frame->hud_image, frame->hud_allocation);
        frame->hud_image = vk::Image{};
        frame->hud_view = vk::ImageView{};
    }
    frame->generated = false;

    const vk::Format format = swapchain.GetSurfaceFormat().format;
    const vk::ImageCreateInfo image_info = {
        .flags = vk::ImageCreateFlagBits::eMutableFormat,
        .imageType = vk::ImageType::e2D,
        .format = format,
        .extent = {width, height, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = vk::SampleCountFlagBits::e1,
        .usage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferDst |
                 vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eSampled,
    };

    const VmaAllocationCreateInfo alloc_info = {
        .flags = VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT,
        .usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        .requiredFlags = 0,
        .preferredFlags = 0,
        .pool = VK_NULL_HANDLE,
        .pUserData = nullptr,
    };

    VkImage unsafe_image{};
    VkImageCreateInfo unsafe_image_info = static_cast<VkImageCreateInfo>(image_info);

    VkResult result = vmaCreateImage(instance.GetAllocator(), &unsafe_image_info, &alloc_info,
                                     &unsafe_image, &frame->allocation, nullptr);
    if (result != VK_SUCCESS) [[unlikely]] {
        LOG_CRITICAL(Render_Vulkan, "Failed allocating texture with error {}",
                     vk::to_string(vk::Result{result}));
        UNREACHABLE();
    }
    frame->image = vk::Image{unsafe_image};
    SetObjectName(device, frame->image, "Frame image #{}", frame->id);
    if (frame_gen && frame_gen->Enabled()) {
        VkImage gen_image{};
        if (vmaCreateImage(instance.GetAllocator(), &unsafe_image_info, &alloc_info, &gen_image,
                           &frame->gen_allocation, nullptr) == VK_SUCCESS) {
            frame->gen_image = vk::Image{gen_image};
            SetObjectName(device, frame->gen_image, "Generated frame #{}", frame->id);
        }
        VkImage hud_image{};
        if (vmaCreateImage(instance.GetAllocator(), &unsafe_image_info, &alloc_info, &hud_image,
                           &frame->hud_allocation, nullptr) == VK_SUCCESS) {
            frame->hud_image = vk::Image{hud_image};
            SetObjectName(device, frame->hud_image, "HUD-less frame #{}", frame->id);
            frame->hud_view = Check<"create HUD-less frame view">(device.createImageView({
                .image = frame->hud_image,
                .viewType = vk::ImageViewType::e2D,
                .format = format,
                .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
            }));
        }
    }

    const vk::ImageViewCreateInfo view_info = {
        .image = frame->image,
        .viewType = vk::ImageViewType::e2D,
        .format = format,
        .subresourceRange{
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
    };
    auto view = Check<"create frame image view">(device.createImageView(view_info));
    frame->image_view = view;
    frame->width = width;
    frame->height = height;

    frame->is_hdr = swapchain.GetHDR();
}

Frame* Presenter::PrepareLastFrame() {
    if (last_submit_frame == nullptr) {
        return nullptr;
    }

    Frame* frame = last_submit_frame;

    while (true) {
        vk::Result result = instance.GetDevice().waitForFences(frame->present_done, false,
                                                               std::numeric_limits<u64>::max());
        if (result == vk::Result::eSuccess) {
            break;
        }
        if (result == vk::Result::eTimeout) {
            continue;
        }
        ASSERT_MSG(result != vk::Result::eErrorDeviceLost,
                   "Device lost during waiting for a frame");
    }

    auto& scheduler = flip_scheduler;
    scheduler.EndRendering();
    const auto cmdbuf = scheduler.CommandBuffer();

    const auto frame_subresources = vk::ImageSubresourceRange{
        .aspectMask = vk::ImageAspectFlagBits::eColor,
        .baseMipLevel = 0,
        .levelCount = 1,
        .baseArrayLayer = 0,
        .layerCount = VK_REMAINING_ARRAY_LAYERS,
    };

    const auto pre_barrier =
        vk::ImageMemoryBarrier2{.srcStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                                .srcAccessMask = vk::AccessFlagBits2::eColorAttachmentRead,
                                .dstStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                                .dstAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
                                .oldLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
                                .newLayout = vk::ImageLayout::eGeneral,
                                .image = frame->image,
                                .subresourceRange{frame_subresources}};

    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers = &pre_barrier,
    });

    // Flush frame creation commands.
    frame->ready_semaphore = scheduler.GetWorkSemaphore()->Handle();
    frame->ready_tick = scheduler.CurrentTick();
    SubmitInfo info{};
    scheduler.Flush(info);
    return frame;
}

static vk::Format GetFrameViewFormat(const Libraries::VideoOut::PixelFormat format) {
    switch (format) {
    case Libraries::VideoOut::PixelFormat::A8B8G8R8Srgb:
        return vk::Format::eR8G8B8A8Srgb;
    case Libraries::VideoOut::PixelFormat::A8R8G8B8Srgb:
        return vk::Format::eB8G8R8A8Srgb;
    case Libraries::VideoOut::PixelFormat::A2R10G10B10:
    case Libraries::VideoOut::PixelFormat::A2R10G10B10Srgb:
    case Libraries::VideoOut::PixelFormat::A2R10G10B10Bt2020Pq:
        return vk::Format::eA2R10G10B10UnormPack32;
    default:
        break;
    }
    UNREACHABLE_MSG("Unknown format={}", static_cast<u32>(format));
    return {};
}

// MediEvil: F12 screenshot of the game's picture as a PNG (GOW3_SCREENSHOT_DIR), encoded on a
// thread of its own once the GPU is done (a 4K PNG takes a moment).
static void SaveScreenshot(const Instance& instance, Scheduler& scheduler, vk::CommandBuffer cmdbuf,
                           vk::Image image, vk::ImageLayout layout, u32 w, u32 h, vk::Format format) {
    const bool bgra = format == vk::Format::eB8G8R8A8Unorm || format == vk::Format::eB8G8R8A8Srgb;
    const bool rgba = format == vk::Format::eR8G8B8A8Unorm || format == vk::Format::eR8G8B8A8Srgb;
    const bool a2r10 = format == vk::Format::eA2R10G10B10UnormPack32;
    const bool a2b10 = format == vk::Format::eA2B10G10R10UnormPack32;
    if (!bgra && !rgba && !a2r10 && !a2b10) {
        std::printf("Screenshot: picture format %s is not supported\n", vk::to_string(format).c_str());
        return;
    }
    const VkBufferCreateInfo buffer_ci{.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                                       .size = VkDeviceSize(w) * h * 4,
                                       .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT};
    const VmaAllocationCreateInfo alloc_ci{
        .flags = VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT,
        .usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST};
    VkBuffer buffer{};
    VmaAllocation allocation{};
    VmaAllocationInfo info{};
    if (vmaCreateBuffer(instance.GetAllocator(), &buffer_ci, &alloc_ci, &buffer, &allocation,
                        &info) != VK_SUCCESS) {
        return;
    }
    cmdbuf.copyImageToBuffer(image, layout, buffer,
                             vk::BufferImageCopy{.imageSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
                                                 .imageExtent = {w, h, 1}});
    const char* env = std::getenv("GOW3_SCREENSHOT_DIR");
    const std::filesystem::path dir = env && *env ? env : "screenshots";
    const std::time_t now = std::time(nullptr);
    std::tm local{};
#ifdef _WIN32
    localtime_s(&local, &now);
#else
    localtime_r(&now, &local);
#endif
    char stamp[64];
    std::strftime(stamp, sizeof(stamp), "MediEvil_%Y%m%d_%H%M%S", &local);
    scheduler.DeferPriorityOperation([allocator = instance.GetAllocator(), buffer, allocation, info, w, h,
                                      bgra, a2r10, a2b10, dir, name = std::string{stamp}] {
        vmaInvalidateAllocation(allocator, allocation, 0, VK_WHOLE_SIZE);
        std::vector<u8> rgb(size_t(w) * h * 3);
        const auto* src = static_cast<const u32*>(info.pMappedData);
        for (size_t i = 0, n = size_t(w) * h; i < n; ++i) {
            const u32 p = src[i];
            u8 r, g, b;
            if (a2r10 || a2b10) {
                const u8 hi = u8(((p >> 20) & 0x3ff) >> 2), mid = u8(((p >> 10) & 0x3ff) >> 2),
                         lo = u8((p & 0x3ff) >> 2);
                r = a2r10 ? hi : lo;
                g = mid;
                b = a2r10 ? lo : hi;
            } else {
                const u8 c0 = u8(p), c1 = u8(p >> 8), c2 = u8(p >> 16);
                r = bgra ? c2 : c0;
                g = c1;
                b = bgra ? c0 : c2;
            }
            rgb[i * 3] = r;
            rgb[i * 3 + 1] = g;
            rgb[i * 3 + 2] = b;
        }
        vmaDestroyBuffer(allocator, buffer, allocation);
        std::thread([rgb = std::move(rgb), w, h, dir, name] {
            size_t size = 0;
            void* png = tdefl_write_image_to_png_file_in_memory_ex(rgb.data(), int(w), int(h), 3, &size,
                                                                   MZ_DEFAULT_LEVEL, MZ_FALSE);
            if (!png) {
                return;
            }
            std::error_code ec;
            std::filesystem::create_directories(dir, ec);
            std::filesystem::path path = dir / (name + ".png");
            for (int n = 2; std::filesystem::exists(path); ++n) {
                path = dir / (name + "_" + std::to_string(n) + ".png");
            }
#ifdef _WIN32
            std::FILE* f = _wfopen(path.wstring().c_str(), L"wb");
#else
            std::FILE* f = std::fopen(path.c_str(), "wb");
#endif
            if (f) {
                std::fwrite(png, 1, size, f);
                std::fclose(f);
                std::printf("Screenshot: %s (%ux%u)\n", path.string().c_str(), w, h);
                std::fflush(stdout);
            }
            mz_free(png);
        }).detach();
    });
}

// gow3: raw copy of an image for GOW3_FRAME_DUMP_TRIGGER, written once the GPU is done.
static void DumpRaw(const Instance& instance, Scheduler& scheduler, vk::CommandBuffer cmdbuf,
                    vk::Image image, vk::ImageLayout layout, u32 w, u32 h, const char* name,
                    int index) {
    const VkBufferCreateInfo buffer_ci{.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                                       .size = VkDeviceSize(w) * h * 4,
                                       .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT};
    const VmaAllocationCreateInfo alloc_ci{
        .flags = VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT,
        .usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST};
    VkBuffer buffer{};
    VmaAllocation allocation{};
    VmaAllocationInfo info{};
    if (vmaCreateBuffer(instance.GetAllocator(), &buffer_ci, &alloc_ci, &buffer, &allocation,
                        &info) != VK_SUCCESS) {
        return;
    }
    cmdbuf.copyImageToBuffer(image, layout, buffer,
                             vk::BufferImageCopy{.imageSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
                                                 .imageExtent = {w, h, 1}});
    const char* dir = std::getenv("GOW3_DUMP_DIR");
    char path[512];
    std::snprintf(path, sizeof(path), "%s/%s_%03d_%ux%u.raw", dir && *dir ? dir : "out/dump", name,
                  index, w, h);
    scheduler.DeferPriorityOperation([allocator = instance.GetAllocator(), buffer, allocation, info,
                                      size = buffer_ci.size, file = std::string{path}] {
        vmaInvalidateAllocation(allocator, allocation, 0, VK_WHOLE_SIZE);
        std::error_code ec;
        std::filesystem::create_directories(std::filesystem::path{file}.parent_path(), ec);
        if (FILE* f = std::fopen(file.c_str(), "wb")) {
            std::fwrite(info.pMappedData, 1, size, f);
            std::fclose(f);
        }
        vmaDestroyBuffer(allocator, buffer, allocation);
    });
}

// gow3: GOW3_FG_STATS=1 also times the game's wait for a free frame and the swap thread's phases.
static std::atomic<u64> fg_frame_wait_ns{0};
static std::atomic<u64> fg_record_ns{0};

Frame* Presenter::PrepareFrame(const Libraries::VideoOut::BufferAttributeGroup& attribute,
                               VAddr cpu_address) {
    // gow3: GOW3_FRAME_DUMP_TRIGGER=<file>: when the file exists it is removed and this frame's
    // guest display buffer and presented image are written to GOW3_DUMP_DIR (default out/dump)
    // as raw 32-bit pixels (menus and movies included; the upscaler dumps scene frames only).
    static const char* frame_dump_trigger = std::getenv("GOW3_FRAME_DUMP_TRIGGER");
    static int frame_dump_index = 0;
    // gow3: GOW3_FRAME_DUMP_COUNT=N dumps N consecutive frames per trigger (flicker reports).
    static const int frame_dump_count = [] {
        const char* env = std::getenv("GOW3_FRAME_DUMP_COUNT");
        return env ? std::clamp(std::atoi(env), 1, 64) : 1;
    }();
    static int frame_dump_remaining = 0;
    // The trigger file is looked for 4 times a second (a remove() per frame cost 1% of the
    // draw recording thread).
    static auto next_trigger_check = std::chrono::steady_clock::now();
    const auto trigger_now = std::chrono::steady_clock::now();
    const bool check_trigger = frame_dump_trigger && frame_dump_remaining == 0 && trigger_now >= next_trigger_check;
    if (check_trigger) {
        next_trigger_check = trigger_now + std::chrono::milliseconds(250);
    }
    if (check_trigger && std::remove(frame_dump_trigger) == 0) {
        frame_dump_remaining = frame_dump_count;
    }
    const bool frame_dump = frame_dump_remaining > 0;
    if (frame_dump) {
        --frame_dump_remaining;
    }
    // gow3: scaled upscaler presets: the output-size display buffer drawn by the port.
    TemporalUpscaler::Display display{};
    const bool upscaled = rasterizer->GetUpscaler().DisplayOverride(cpu_address, display);
    VideoCore::ImageId image_id{};
    if (!upscaled) {
        auto desc = VideoCore::TextureCache::ImageDesc{attribute, cpu_address};
        image_id = texture_cache.FindImage(desc);
        texture_cache.UpdateImage(image_id);
    }

    Frame* frame = GetRenderFrame();

    const auto frame_subresources = vk::ImageSubresourceRange{
        .aspectMask = vk::ImageAspectFlagBits::eColor,
        .baseMipLevel = 0,
        .levelCount = 1,
        .baseArrayLayer = 0,
        .layerCount = VK_REMAINING_ARRAY_LAYERS,
    };

    const auto pre_barrier = vk::ImageMemoryBarrier2{
        .srcStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        .srcAccessMask = vk::AccessFlagBits2::eColorAttachmentRead,
        .dstStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        .dstAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
        .oldLayout = vk::ImageLayout::eUndefined,
        .newLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .image = frame->image,
        .subresourceRange{frame_subresources},
    };

    draw_scheduler.EndRendering();
    const auto cmdbuf = draw_scheduler.CommandBuffer();
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers = &pre_barrier,
    });

    VideoCore::ImageViewInfo view_info{};
    view_info.format = GetFrameViewFormat(attribute.attrib.pixel_format);
    // Exclude alpha from output frame to avoid blending with UI.
    view_info.mapping.a = vk::ComponentSwizzle::eOne;

    vk::ImageView image_view{};
    vk::Extent2D image_size{};
    if (upscaled) {
        const auto device = instance.GetDevice();
        image_view = Check(device.createImageView({
            .image = display.image,
            .viewType = vk::ImageViewType::e2D,
            .format = view_info.format,
            .components = {vk::ComponentSwizzle::eIdentity, vk::ComponentSwizzle::eIdentity,
                           vk::ComponentSwizzle::eIdentity, vk::ComponentSwizzle::eOne},
            .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
        }));
        draw_scheduler.DeferOperation([device, image_view] { device.destroyImageView(image_view); });
        image_size = vk::Extent2D{display.width, display.height};
        const vk::ImageMemoryBarrier2 to_read{
            .srcStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
            .srcAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
            .oldLayout = vk::ImageLayout::eGeneral,
            .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
            .image = display.image,
            .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
        };
        cmdbuf.pipelineBarrier2({.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &to_read});
        if (Gow3Overlay::TakeScreenshotRequest()) {
            auto to_copy = to_read;
            to_copy.srcStageMask = vk::PipelineStageFlagBits2::eAllCommands;
            to_copy.srcAccessMask = vk::AccessFlagBits2::eShaderRead;
            to_copy.dstStageMask = vk::PipelineStageFlagBits2::eTransfer;
            to_copy.dstAccessMask = vk::AccessFlagBits2::eTransferRead;
            to_copy.oldLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
            to_copy.newLayout = vk::ImageLayout::eTransferSrcOptimal;
            cmdbuf.pipelineBarrier2({.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &to_copy});
            SaveScreenshot(instance, draw_scheduler, cmdbuf, display.image,
                           vk::ImageLayout::eTransferSrcOptimal, display.width, display.height,
                           view_info.format);
            auto back = to_copy;
            back.srcStageMask = vk::PipelineStageFlagBits2::eTransfer;
            back.srcAccessMask = vk::AccessFlagBits2::eTransferRead;
            back.dstStageMask = vk::PipelineStageFlagBits2::eAllCommands;
            back.dstAccessMask = vk::AccessFlagBits2::eShaderRead;
            back.oldLayout = vk::ImageLayout::eTransferSrcOptimal;
            back.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
            cmdbuf.pipelineBarrier2({.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &back});
        }
    } else {
        auto& image = texture_cache.GetImage(image_id);
        if (Gow3Overlay::TakeScreenshotRequest()) {
            runtime.Transit(&image, vk::ImageLayout::eTransferSrcOptimal,
                            vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferRead);
            runtime.FlushBarriers();
            SaveScreenshot(instance, draw_scheduler, cmdbuf, image.GetImage(),
                           vk::ImageLayout::eTransferSrcOptimal, image.info.size.width,
                           image.info.size.height, view_info.format);
        }
        if (frame_dump) {
            runtime.Transit(&image, vk::ImageLayout::eTransferSrcOptimal,
                            vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferRead);
            runtime.FlushBarriers();
            DumpRaw(instance, draw_scheduler, cmdbuf, image.GetImage(),
                    vk::ImageLayout::eTransferSrcOptimal, image.info.size.width,
                    image.info.size.height, "display", frame_dump_index);
        }
        image_view = *image.FindView(view_info).image_view;
        image_size = vk::Extent2D{image.info.size.width, image.info.size.height};
        runtime.Transit(&image, vk::ImageLayout::eShaderReadOnlyOptimal,
                        vk::PipelineStageFlagBits2::eFragmentShader,
                        vk::AccessFlagBits2::eShaderRead);
        runtime.FlushBarriers();
    }
    expected_ratio = static_cast<float>(image_size.width) / static_cast<float>(image_size.height);

    // gow3: RCAS and its strength come from the in-game menu, every frame; FSR 1 only from
    // medievil.ini (fsr1=1) or GOW3_FSR1, no longer offered in the menus.
    const auto& graphics = Gow3Settings::Get();
    fsr_settings.enable = graphics.fsr1;
    fsr_settings.use_rcas = graphics.rcas;
    fsr_settings.rcas_attenuation = Gow3Graphics::RcasAttenuation(graphics.rcas_strength);
    Gow3Settings::Get().image_width = int(image_size.width);
    Gow3Settings::Get().image_height = int(image_size.height);
    Gow3Settings::Get().window_width = int(frame->width);
    Gow3Settings::Get().window_height = int(frame->height);
    Gow3Settings::Get().rcas_applied =
        fsr_settings.use_rcas &&
        Gow3Graphics::FsrPassRuns(fsr_settings.enable, true, int(image_size.width), int(image_size.height),
                                  int(frame->width), int(frame->height));
    image_view = fsr_pass.Render(cmdbuf, image_view, image_size, {frame->width, frame->height},
                                 fsr_settings, frame->is_hdr);

    // Vulkan has no sRGB variant of the 10-bit format, so an A2R10G10B10Srgb buffer reaches
    // the post process pass still sRGB encoded and has to be decoded there instead.
    pp_settings.srgb_input =
        attribute.attrib.pixel_format == Libraries::VideoOut::PixelFormat::A2R10G10B10Srgb;
    pp_pass.Render(cmdbuf, image_view, image_size, *frame, pp_settings);
    frame->generated = false;
    if (frame->gen_image) {
        // The game frame is in General after the pass; the generated one is written whole.
        const vk::ImageMemoryBarrier2 gen_general{
            .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
            .oldLayout = vk::ImageLayout::eUndefined,
            .newLayout = vk::ImageLayout::eGeneral,
            .image = frame->gen_image,
            .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
        };
        cmdbuf.pipelineBarrier2({.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &gen_general});
        // The HUD-less copy of this frame's display buffer, through the same passes.
        const auto record_start = std::chrono::steady_clock::now();
        const auto& fg_inputs = rasterizer->GetNeuralRender().FrameGen();
        vk::Image hudless{};
        if (!upscaled && frame->hud_image && fg_inputs.hudless &&
            fg_inputs.hudless_serial == fg_inputs.serial &&
            fg_inputs.hudless_w == image_size.width && fg_inputs.hudless_h == image_size.height &&
            Gow3Settings::Get().fg_enabled) {
            const auto device = instance.GetDevice();
            const vk::ImageView hud_input = Check(device.createImageView({
                .image = fg_inputs.hudless,
                .viewType = vk::ImageViewType::e2D,
                .format = view_info.format,
                .components = {vk::ComponentSwizzle::eIdentity, vk::ComponentSwizzle::eIdentity,
                               vk::ComponentSwizzle::eIdentity, vk::ComponentSwizzle::eOne},
                .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
            }));
            draw_scheduler.DeferOperation([device, hud_input] { device.destroyImageView(hud_input); });
            auto hud_barrier = pre_barrier;
            hud_barrier.image = frame->hud_image;
            cmdbuf.pipelineBarrier2({.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &hud_barrier});
            Frame hud_frame = *frame;
            hud_frame.image = frame->hud_image;
            hud_frame.image_view = frame->hud_view;
            const vk::ImageView hud_view = fsr_pass.Render(cmdbuf, hud_input, image_size,
                                                           {frame->width, frame->height},
                                                           fsr_settings, frame->is_hdr);
            pp_pass.Render(cmdbuf, hud_view, image_size, hud_frame, pp_settings);
            hudless = frame->hud_image;
        }
        frame->generated = frame_gen->Record(draw_scheduler, frame->image, hudless, frame->gen_image,
                                             swapchain.GetSurfaceFormat().format, frame->width,
                                             frame->height,
                                             rasterizer->GetNeuralRender().FrameGen());
        fg_record_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - record_start)
                            .count();
        // GOW3_FRAMEGEN_SHOT=<file>: when it exists, the next generated frame is saved as a
        // screenshot (the screenshot key saves game frames only).
        static const char* fg_shot = std::getenv("GOW3_FRAMEGEN_SHOT");
        if (frame->generated && fg_shot && std::remove(fg_shot) == 0) {
            SaveScreenshot(instance, draw_scheduler, cmdbuf, frame->gen_image,
                           vk::ImageLayout::eGeneral, frame->width, frame->height,
                           swapchain.GetSurfaceFormat().format);
        }
    }
    if (frame_dump) {
        // The presented image as the swapchain blit reads it (General after the pass).
        const vk::MemoryBarrier2 done{.srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
                                      .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
                                      .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
                                      .dstAccessMask = vk::AccessFlagBits2::eTransferRead};
        cmdbuf.pipelineBarrier2({.memoryBarrierCount = 1, .pMemoryBarriers = &done});
        DumpRaw(instance, draw_scheduler, cmdbuf, frame->image, vk::ImageLayout::eGeneral,
                frame->width, frame->height, "output", frame_dump_index);
        ++frame_dump_index;
    }

    // Flush frame creation commands.
    frame->ready_semaphore = draw_scheduler.GetWorkSemaphore()->Handle();
    frame->ready_tick = draw_scheduler.CurrentTick();
    SubmitInfo info{};
    draw_scheduler.Flush(info);

    // gow3: the GPU command thread runs at most GOW3_FRAMES_AHEAD (default 1) guest frames
    // ahead of the GPU: it waits here for the frame that many flips back. When the GPU is the
    // bottleneck it finishes frames at an even rate; without this bound the command thread ran
    // ahead and then blocked wherever a resource ran out, so flips (and the guest's frame
    // timing) came in bursts: 12.5/25 ms alternation at 80 FPS. 0 turns it off.
    // The in-game menu changes it live (Gow3Settings::frames_queued; GOW3_FRAMES_AHEAD at start).
    const u32 frames_ahead = u32(std::max(0, Gow3Settings::Get().frames_queued.load()));
    if (!frames_ahead) {
        recent_frame_ticks.clear();
    } else {
        recent_frame_ticks.push_back(frame->ready_tick);
        while (recent_frame_ticks.size() > frames_ahead) {
            const u64 tick = recent_frame_ticks.front();
            recent_frame_ticks.pop_front();
            if (recent_frame_ticks.size() == frames_ahead) {
                draw_scheduler.Wait(tick);
            }
        }
    }
    return frame;
}

Frame* Presenter::PrepareBlankFrame(bool present_thread) {
    // Request a free presentation frame.
    Frame* frame = GetRenderFrame();

    auto& scheduler = present_thread ? present_scheduler : draw_scheduler;
    scheduler.EndRendering();

    const auto cmdbuf = scheduler.CommandBuffer();

    constexpr vk::ImageSubresourceRange simple_subresource = {
        .aspectMask = vk::ImageAspectFlagBits::eColor,
        .levelCount = 1,
        .layerCount = 1,
    };
    const auto pre_barrier = vk::ImageMemoryBarrier2{
        .srcStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        .srcAccessMask = vk::AccessFlagBits2::eColorAttachmentRead,
        .dstStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        .dstAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
        .oldLayout = vk::ImageLayout::eUndefined,
        .newLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .image = frame->image,
        .subresourceRange = simple_subresource,
    };

    const auto post_barrier = vk::ImageMemoryBarrier2{
        .srcStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        .srcAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eFragmentShader,
        .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
        .oldLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .newLayout = vk::ImageLayout::eGeneral,
        .image = frame->image,
        .subresourceRange = simple_subresource,
    };

    const vk::RenderingAttachmentInfo attachment = {
        .imageView = frame->image_view,
        .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eStore,
    };
    const vk::RenderingInfo rendering_info = {
        .renderArea =
            {
                .extent = {frame->width, frame->height},
            },
        .layerCount = 1,
        .colorAttachmentCount = 1u,
        .pColorAttachments = &attachment,
    };

    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers = &pre_barrier,
    });

    cmdbuf.beginRendering(rendering_info);
    cmdbuf.endRendering();

    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers = &post_barrier,
    });

    // Flush frame creation commands.
    frame->ready_semaphore = scheduler.GetWorkSemaphore()->Handle();
    frame->ready_tick = scheduler.CurrentTick();
    SubmitInfo info{};
    scheduler.Flush(info);
    return frame;
}

void Presenter::PresentLoadingFrame() {
    if (window.GetWidth() == 0 || window.GetHeight() == 0) {
        return;
    }
    swapchain.UpdateVsync();
    if (window.GetWidth() != swapchain.GetWidth() || window.GetHeight() != swapchain.GetHeight()) {
        swapchain.Recreate(window.GetWidth(), window.GetHeight());
    }
    if (!swapchain.AcquireNextImage()) {
        return;
    }
    const vk::Image image = swapchain.Image();
    const vk::Extent2D extent = swapchain.GetExtent();
    const auto cmdbuf = present_scheduler.CommandBuffer();
    const vk::ImageSubresourceRange range{
        .aspectMask = vk::ImageAspectFlagBits::eColor,
        .levelCount = 1,
        .layerCount = VK_REMAINING_ARRAY_LAYERS,
    };
    const auto barrier = [&](vk::ImageLayout from, vk::ImageLayout to, vk::AccessFlags src,
                             vk::AccessFlags dst) {
        cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
                               vk::PipelineStageFlagBits::eAllCommands, {}, {}, {},
                               vk::ImageMemoryBarrier{
                                   .srcAccessMask = src,
                                   .dstAccessMask = dst,
                                   .oldLayout = from,
                                   .newLayout = to,
                                   .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                   .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                   .image = image,
                                   .subresourceRange = range,
                               });
    };
    barrier(vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal, {},
            vk::AccessFlagBits::eTransferWrite);
    cmdbuf.clearColorImage(image, vk::ImageLayout::eTransferDstOptimal,
                           vk::ClearColorValue{std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f}},
                           range);
    barrier(vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eColorAttachmentOptimal,
            vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eColorAttachmentWrite);
    Gow3Overlay::Render(cmdbuf, swapchain.ImageView(), extent);
    barrier(vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::ePresentSrcKHR,
            vk::AccessFlagBits::eColorAttachmentWrite, {});
    SubmitInfo info{};
    info.AddWait(swapchain.GetImageAcquiredSemaphore());
    info.AddSignal(swapchain.GetPresentReadySemaphore());
    present_scheduler.Flush(info);
    {
        std::scoped_lock submit_lock{Scheduler::submit_mutex};
        swapchain.Present();
    }
    // A few frames a second during start-up: waiting keeps semaphores and the swapchain simple.
    present_scheduler.Finish();
}

void Presenter::Present(Frame* frame, bool is_reusing_frame, bool is_game_frame) {
    const auto arrival = std::chrono::steady_clock::now();
    // Free the frame for reuse
    const auto free_frame = [&] {
        if (!is_reusing_frame) {
            last_submit_frame = frame;
            std::scoped_lock fl{free_mutex};
            free_queue.push(frame);
            free_cv.notify_one();
        }
    };

    // gow3: a minimized window has no pixels on Windows (0x0); keep the swapchain and skip
    // presenting until the window is restored.
    if (window.GetWidth() == 0 || window.GetHeight() == 0) {
        free_frame();
        return;
    }

    // Recreate the swapchain if the window was resized.
    swapchain.UpdateVsync();
    if (window.GetWidth() != swapchain.GetWidth() || window.GetHeight() != swapchain.GetHeight()) {
        swapchain.Recreate(window.GetWidth(), window.GetHeight());
    }

    // gow3: frame generation (vk_frame_gen) presents the generated frame first, then the game's.
    const bool generated = frame->generated && !is_reusing_frame;
    frame->generated = false;
    const auto present_one = [&](vk::Image source, bool last) -> bool {
        const auto acquire_start = std::chrono::steady_clock::now();
        if (!swapchain.AcquireNextImage()) {
            swapchain.Recreate(window.GetWidth(), window.GetHeight());
            if (!swapchain.AcquireNextImage()) {
                // User resizes the window too fast and GPU can't keep up. Skip this frame.
                LOG_WARNING(Render_Vulkan, "Skipping frame!");
                return false;
            }
        }
        const float acquire_ms = std::chrono::duration<float, std::milli>(
                                     std::chrono::steady_clock::now() - acquire_start)
                                     .count();

        // Reset fence for queue submission. Do it here instead of GetRenderFrame() because we may
        // skip frame because of slow swapchain recreation. If a frame skip occurs, we skip signal
        // the frame's present fence and future GetRenderFrame() call will hang waiting for this frame.
        if (last) {
            const auto reset_result = instance.GetDevice().resetFences(frame->present_done);
            ASSERT_MSG(reset_result == vk::Result::eSuccess,
                       "Unexpected error resetting present done fence: {}", vk::to_string(reset_result));
        }

        // gow3: the game frame is blitted (letterboxed) straight into the swapchain image.
        const vk::Image swapchain_image = swapchain.Image();
        auto& scheduler = present_scheduler;
        const auto cmdbuf = scheduler.CommandBuffer();

        if (EmulatorSettings.IsVkHostMarkersEnabled()) {
            cmdbuf.beginDebugUtilsLabelEXT(vk::DebugUtilsLabelEXT{
                .pLabelName = "Present",
            });
        }

        {
            const vk::Extent2D extent = swapchain.GetExtent();
            SetExpectedGameSize(s32(extent.width), s32(extent.height));
            const vk::ImageSubresourceRange color_range{
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = VK_REMAINING_ARRAY_LAYERS,
            };
            const std::array pre_barriers{
                vk::ImageMemoryBarrier{
                    .srcAccessMask = vk::AccessFlagBits::eNone,
                    .dstAccessMask = vk::AccessFlagBits::eTransferWrite,
                    .oldLayout = vk::ImageLayout::eUndefined,
                    .newLayout = vk::ImageLayout::eTransferDstOptimal,
                    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .image = swapchain_image,
                    .subresourceRange = color_range,
                },
                vk::ImageMemoryBarrier{
                    .srcAccessMask = vk::AccessFlagBits::eColorAttachmentWrite,
                    .dstAccessMask = vk::AccessFlagBits::eTransferRead,
                    .oldLayout = vk::ImageLayout::eGeneral,
                    .newLayout = vk::ImageLayout::eTransferSrcOptimal,
                    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .image = source,
                    .subresourceRange = color_range,
                },
            };
            cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
                                   vk::PipelineStageFlagBits::eTransfer, vk::DependencyFlagBits::eByRegion,
                                   {}, {}, pre_barriers);
            const vk::ClearColorValue black{std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f}};
            cmdbuf.clearColorImage(swapchain_image, vk::ImageLayout::eTransferDstOptimal, black, color_range);
            const vk::MemoryBarrier clear_done{
                .srcAccessMask = vk::AccessFlagBits::eTransferWrite,
                .dstAccessMask = vk::AccessFlagBits::eTransferWrite,
            };
            cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eTransfer,
                                   vk::DependencyFlagBits::eByRegion, clear_done, {}, {});
            cmdbuf.blitImage(source, vk::ImageLayout::eTransferSrcOptimal, swapchain_image,
                             vk::ImageLayout::eTransferDstOptimal,
                             MakeImageBlitFit(frame->width, frame->height, extent.width, extent.height),
                             vk::Filter::eLinear);
            // gow3: the settings menu / FPS counter over the frame, at display resolution.
            const bool overlay = Gow3Overlay::Visible();
            const std::array post_barriers{
                vk::ImageMemoryBarrier{
                    .srcAccessMask = vk::AccessFlagBits::eTransferWrite,
                    .dstAccessMask = overlay ? vk::AccessFlagBits::eColorAttachmentRead |
                                                   vk::AccessFlagBits::eColorAttachmentWrite
                                             : vk::AccessFlagBits::eNone,
                    .oldLayout = vk::ImageLayout::eTransferDstOptimal,
                    .newLayout = overlay ? vk::ImageLayout::eColorAttachmentOptimal
                                         : vk::ImageLayout::ePresentSrcKHR,
                    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .image = swapchain_image,
                    .subresourceRange = color_range,
                },
                vk::ImageMemoryBarrier{
                    .srcAccessMask = vk::AccessFlagBits::eTransferRead,
                    .dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite,
                    .oldLayout = vk::ImageLayout::eTransferSrcOptimal,
                    .newLayout = vk::ImageLayout::eGeneral,
                    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .image = source,
                    .subresourceRange = color_range,
                },
            };
            cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                   vk::PipelineStageFlagBits::eAllCommands,
                                   vk::DependencyFlagBits::eByRegion, {}, {}, post_barriers);
            if (last) {
                Gow3Overlay::LogPresent();
            }
            if (overlay) {
                Gow3Overlay::Render(cmdbuf, swapchain.ImageView(), extent);
                const vk::ImageMemoryBarrier to_present{
                    .srcAccessMask = vk::AccessFlagBits::eColorAttachmentWrite,
                    .dstAccessMask = vk::AccessFlagBits::eNone,
                    .oldLayout = vk::ImageLayout::eColorAttachmentOptimal,
                    .newLayout = vk::ImageLayout::ePresentSrcKHR,
                    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .image = swapchain_image,
                    .subresourceRange = color_range,
                };
                cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eColorAttachmentOutput,
                                       vk::PipelineStageFlagBits::eBottomOfPipe,
                                       vk::DependencyFlagBits::eByRegion, {}, {}, to_present);
            }
        }
        if (EmulatorSettings.IsVkHostMarkersEnabled()) {
            cmdbuf.endDebugUtilsLabelEXT();
        }

        // Flush vulkan commands.

        SubmitInfo info{};
        info.AddWait(swapchain.GetImageAcquiredSemaphore());
        info.AddWait(frame->ready_semaphore, frame->ready_tick);
        info.AddSignal(swapchain.GetPresentReadySemaphore());
        if (last) {
            info.AddSignal(frame->present_done);
        }
        scheduler.Flush(info);

        // Present to swapchain.
        {
            std::scoped_lock submit_lock{Scheduler::submit_mutex};
            if (!swapchain.Present() && window.GetWidth() != 0 && window.GetHeight() != 0) {
                swapchain.Recreate(window.GetWidth(), window.GetHeight());
            }
        }
        if (frame_gen && frame_gen->Enabled()) {
            frame_gen->NotePresent(acquire_ms);
        }
        // gow3: GOW3_FG_STATS=1: every 5 s, the time from each present to the next, after a
        // generated frame and after a game frame (frame generation pacing).
        static const bool fg_stats = [] {
            const char* env = std::getenv("GOW3_FG_STATS");
            return env && env[0] == '1';
        }();
        if (fg_stats) {
            using Clock = std::chrono::steady_clock;
            static Clock::time_point last_present{}, window_start = Clock::now();
            static bool last_generated = false;
            static double sum[2]{}, sq[2]{}, worst[2]{};
            static u32 count[2]{};
            const auto now = Clock::now();
            if (last_present != Clock::time_point{}) {
                const double ms = std::chrono::duration<double, std::milli>(now - last_present).count();
                const int k = last_generated ? 1 : 0;
                sum[k] += ms;
                sq[k] += ms * ms;
                worst[k] = std::max(worst[k], ms);
                ++count[k];
            }
            last_present = now;
            last_generated = !last;
            if (now - window_start >= std::chrono::seconds(5)) {
                const double secs = std::chrono::duration<double>(now - window_start).count();
                for (int k = 0; k < 2; ++k) {
                    const double mean = count[k] ? sum[k] / count[k] : 0.0;
                    const double sd =
                        count[k] ? std::sqrt(std::max(0.0, sq[k] / count[k] - mean * mean)) : 0.0;
                    std::printf("FG stats: after %s frames: %u (%.1f/s), next present in %.2f ms "
                                "(sd %.2f, worst %.2f)\n",
                                k ? "generated" : "game", count[k], count[k] / secs, mean, sd,
                                worst[k]);
                    sum[k] = sq[k] = worst[k] = 0.0;
                    count[k] = 0;
                }
                window_start = now;
            }
        }
        return true;
    };
    // gow3: frame pacing: the generated frame is halfway between two game frames, so the game
    // frame is presented half the game's frame interval after the frame arrived. The interval is
    // measured between arrivals and the hold counts from the arrival: the game waits on this
    // thread, so a hold measured from the present would lengthen the interval it is based on.
    using PaceClock = std::chrono::steady_clock;
    static PaceClock::time_point last_game_arrival{};
    static double game_interval_ms = 0.0;
    if (generated && !present_one(frame->gen_image, false)) {
        free_frame();
        return;
    }
    const auto gen_done = PaceClock::now();
    if (generated && game_interval_ms > 0.0) {
        const std::chrono::duration<double, std::milli> hold{std::min(game_interval_ms * 0.5, 50.0)};
        std::this_thread::sleep_until(arrival +
                                      std::chrono::duration_cast<PaceClock::duration>(hold));
    }
    const auto held = PaceClock::now();
    if (!present_one(frame->image, true)) {
        free_frame();
        return;
    }
    if (generated && !is_reusing_frame) {
        static const bool phase_stats = EmulatorSettingsImpl::Flag("GOW3_FG_STATS", false);
        if (phase_stats) {
            static PaceClock::time_point window = PaceClock::now();
            static double gen_ms, hold_ms, real_ms;
            static u32 n;
            const auto ms = [](auto a, auto b) {
                return std::chrono::duration<double, std::milli>(b - a).count();
            };
            const auto done = PaceClock::now();
            gen_ms += ms(arrival, gen_done);
            hold_ms += ms(gen_done, held);
            real_ms += ms(held, done);
            ++n;
            if (done - window >= std::chrono::seconds(5)) {
                const double wait_ms = fg_frame_wait_ns.exchange(0) / 1e6;
                const double record_ms = fg_record_ns.exchange(0) / 1e6;
                std::printf("FG phases per game frame: draw thread records %.2f ms and waits %.2f ms "
                            "for a free frame; swap thread: generated present %.2f ms, hold %.2f "
                            "ms, game frame present %.2f ms\n",
                            record_ms / n, wait_ms / n, gen_ms / n, hold_ms / n, real_ms / n);
                gen_ms = hold_ms = real_ms = 0.0;
                n = 0;
                window = done;
            }
        }
    }
    if (!is_reusing_frame) {
        if (last_game_arrival != PaceClock::time_point{}) {
            const double ms =
                std::chrono::duration<double, std::milli>(arrival - last_game_arrival).count();
            // Long gaps (loading, menus without the TAAU) do not count.
            if (ms < 100.0) {
                game_interval_ms = game_interval_ms <= 0.0 ? ms : game_interval_ms * 0.9 + ms * 0.1;
            }
        }
        last_game_arrival = arrival;
    }

    free_frame();
    if (!is_reusing_frame && is_game_frame) {
        DebugState.IncFlipFrameNum();
    }
}

Frame* Presenter::GetRenderFrame() {
    const auto wait_start = std::chrono::steady_clock::now();
    struct WaitTimer {
        std::chrono::steady_clock::time_point start;
        ~WaitTimer() {
            fg_frame_wait_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
                                    std::chrono::steady_clock::now() - start)
                                    .count();
        }
    } wait_timer{wait_start};
    // Wait for free presentation frames
    Frame* frame;
    {
        std::unique_lock lock{free_mutex};
        free_cv.wait(lock, [this] { return !free_queue.empty(); });
        LOG_DEBUG(Render_Vulkan, "Got render frame, remaining {}", free_queue.size() - 1);

        // Take the frame from the queue
        frame = free_queue.front();
        free_queue.pop();
    }

    const vk::Device device = instance.GetDevice();
    vk::Result result{};

    const auto wait = [&]() {
        result = device.waitForFences(frame->present_done, false, std::numeric_limits<u64>::max());
        return result;
    };

    // Wait for the presentation to be finished so all frame resources are free
    while (wait() != vk::Result::eSuccess) {
        ASSERT_MSG(result != vk::Result::eErrorDeviceLost,
                   "Device lost during waiting for a frame");
        // Retry if the waiting times out
        if (result == vk::Result::eTimeout) {
            continue;
        }
    }

    if (frame->width != expected_frame_width || frame->height != expected_frame_height ||
        frame->is_hdr != swapchain.GetHDR()) {
        RecreateFrame(frame, expected_frame_width, expected_frame_height);
    }

    return frame;
}

void Presenter::SetExpectedGameSize(s32 width, s32 height) {
    if (width <= 0 || height <= 0) {
        return; // no surface (minimized window): keep the last frame size
    }
    const float ratio = (float)width / (float)height;

    expected_frame_height = height;
    expected_frame_width = width;
    if (ratio > expected_ratio) {
        expected_frame_width = static_cast<s32>(height * expected_ratio);
    } else {
        expected_frame_height = static_cast<s32>(width / expected_ratio);
    }
}

} // namespace Vulkan
