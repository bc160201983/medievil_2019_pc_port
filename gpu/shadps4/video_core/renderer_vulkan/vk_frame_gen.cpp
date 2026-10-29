// SPDX-License-Identifier: GPL-2.0-or-later
#include "video_core/renderer_vulkan/vk_frame_gen.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>

#include "ffx_vk_fsr3_3_1_5_bridge.h"
#include "core/emulator_settings.h"
#include "gow3_settings.h"
#include "video_core/host_shaders/fg_hud_compose_comp.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"

namespace Vulkan {

namespace {

constexpr vk::Format kColorFormat = vk::Format::eR8G8B8A8Unorm;
constexpr vk::ImageUsageFlags kColorUsage =
    vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage |
    vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst;
// UE4 units are centimetres; its projection is reversed and infinite, the SDK's is not.
constexpr float kFarPlane = 1.0e6f;
constexpr float kViewSpaceToMeters = 0.01f;
// A game frame's pixel this far (in 8-bit steps) from its HUD-less copy is UI.
constexpr float kHudThreshold = 2.0f / 255.0f;

struct ComposeConstants {
    std::array<s32, 2> size;
    float threshold;
};

FfxVkFsr3_3_1_6FrameGenerationImage Describe(vk::Image image, vk::Format format, u32 width,
                                             u32 height, vk::ImageUsageFlags usage) {
    return {
        .image = image,
        .format = static_cast<VkFormat>(format),
        .width = width,
        .height = height,
        .layout = VK_IMAGE_LAYOUT_GENERAL,
        .usage = static_cast<VkImageUsageFlags>(usage),
    };
}

void MemoryBarrier(vk::CommandBuffer cmdbuf) {
    const vk::MemoryBarrier2 memory{
        .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
    };
    cmdbuf.pipelineBarrier2({.memoryBarrierCount = 1, .pMemoryBarriers = &memory});
}

} // namespace

FrameGen::FrameGen(const Instance& instance_) : instance{instance_} {
    const char* env = std::getenv("GOW3_FRAMEGEN");
    enabled = env && env[0] == '1';
    if (!enabled) {
        return;
    }
    std::printf("GPU: FSR 3.1.6 frame generation on (UE4 TAAU frames, UI kept from the game frame)\n");
    const auto device = instance.GetDevice();
    std::array<vk::DescriptorSetLayoutBinding, 3> bindings{};
    for (u32 i = 0; i < 3; ++i) {
        bindings[i] = {.binding = i,
                       .descriptorType = vk::DescriptorType::eStorageImage,
                       .descriptorCount = 1,
                       .stageFlags = vk::ShaderStageFlagBits::eCompute};
    }
    compose_layout = Check(device.createDescriptorSetLayoutUnique({
        .flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR,
        .bindingCount = u32(bindings.size()),
        .pBindings = bindings.data(),
    }));
    const vk::PushConstantRange range{.stageFlags = vk::ShaderStageFlagBits::eCompute,
                                      .offset = 0,
                                      .size = sizeof(ComposeConstants)};
    compose_pipeline_layout = Check(device.createPipelineLayoutUnique({
        .setLayoutCount = 1,
        .pSetLayouts = &*compose_layout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &range,
    }));
    const auto module = CompileSPV(FG_HUD_COMPOSE_COMP, device);
    compose_pipeline = Check(device.createComputePipelineUnique(
        {}, vk::ComputePipelineCreateInfo{
                .stage = {.stage = vk::ShaderStageFlagBits::eCompute, .module = module, .pName = "main"},
                .layout = *compose_pipeline_layout,
            }));
    device.destroyShaderModule(module);
}

FrameGen::~FrameGen() {
    Destroy();
}

void FrameGen::Destroy() {
    if (context) {
        // The caller's frames are done (the presenter waits for the device before this).
        ffxVkFsr3_3_1_6FrameGenerationContextRetireFrame(context, frame_id);
        ffxVkFsr3_3_1_6FrameGenerationContextDestroy(context);
        context = nullptr;
    }
    in_flight.clear();
}

bool FrameGen::EnsureContext(u32 w, u32 h, u32 render_w, u32 render_h) {
    if (context && w == width && h == height && render_w <= max_render_w &&
        render_h <= max_render_h) {
        return true;
    }
    Destroy();
    const auto device = instance.GetDevice();
    const FfxVkFsr3_3_1_6FrameGenerationCreateInfo info{
        .physicalDevice = instance.GetPhysicalDevice(),
        .device = device,
        .maxRenderWidth = std::max(render_w, w),
        .maxRenderHeight = std::max(render_h, h),
        .displayWidth = w,
        .displayHeight = h,
        .colorFormat = static_cast<VkFormat>(kColorFormat),
    };
    if (ffxVkFsr3_3_1_6FrameGenerationContextCreate(&info, &context) !=
        FFX_VK_FSR3_3_1_6_FRAMEGEN_OK) {
        std::printf("GPU: FSR frame generation off: its context could not be created\n");
        context = nullptr;
        return false;
    }
    const auto make = [&](VideoCore::UniqueImage& image, vk::UniqueImageView& view) {
        view.reset();
        image = VideoCore::UniqueImage(device, instance.GetAllocator());
        image.Create(vk::ImageCreateInfo{
            .imageType = vk::ImageType::e2D,
            .format = kColorFormat,
            .extent = {w, h, 1},
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = vk::SampleCountFlagBits::e1,
            .tiling = vk::ImageTiling::eOptimal,
            .usage = kColorUsage,
            .initialLayout = vk::ImageLayout::eUndefined,
        });
        view = Check(device.createImageViewUnique({
            .image = vk::Image(image),
            .viewType = vk::ImageViewType::e2D,
            .format = kColorFormat,
            .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
        }));
    };
    make(color, color_view);
    make(output, output_view);
    make(real, real_view);
    width = w;
    height = h;
    max_render_w = info.maxRenderWidth;
    max_render_h = info.maxRenderHeight;
    last_serial = 0;
    std::printf("GPU: FSR frame generation %ux%u (render up to %ux%u)\n", w, h, max_render_w,
                max_render_h);
    return true;
}

void FrameGen::NotePresent(float acquire_ms) {
    const auto now = std::chrono::steady_clock::now();
    const float interval_ms =
        std::chrono::duration<float, std::milli>(now - last_present).count();
    last_present = now;
    // Only presents that waited for an image: the others are paced by the game.
    if (acquire_ms < 1.0f || interval_ms > 100.0f) {
        return;
    }
    paced_ms = paced_ms <= 0.0f ? interval_ms : paced_ms * 0.9f + interval_ms * 0.1f;
    paced_interval_ms.store(paced_ms, std::memory_order_relaxed);
    paced_at.store(now.time_since_epoch().count(), std::memory_order_relaxed);
}

float FrameGen::RefreshMs() const {
    const float display_ms = 1000.0f / float(Gow3DisplayRefreshHz());
    const std::chrono::steady_clock::time_point at{
        std::chrono::steady_clock::duration{paced_at.load(std::memory_order_relaxed)}};
    // Paced presents seen in the last two seconds.
    if (std::chrono::steady_clock::now() - at > std::chrono::seconds(2)) {
        return display_ms;
    }
    return std::max(display_ms, paced_interval_ms.load(std::memory_order_relaxed));
}

bool FrameGen::Record(Scheduler& scheduler, vk::Image frame, vk::Image hudless, vk::Image out,
                      vk::Format format, u32 w, u32 h, const NeuralRender::FrameGenInputs& inputs) {
    if (!enabled || failed) {
        return false;
    }
    // The views of frames the GPU has finished.
    while (!in_flight.empty() && scheduler.IsFree(in_flight.front().second)) {
        ffxVkFsr3_3_1_6FrameGenerationContextRetireFrame(context, in_flight.front().first);
        in_flight.pop_front();
    }
    const auto now = std::chrono::steady_clock::now();
    const float frame_ms =
        std::clamp(std::chrono::duration<float, std::milli>(now - last_time).count(), 1.0f, 100.0f);
    last_time = now;
    // Only frames drawn by the TAAU (not menus, videos, loading screens), 32-bit SDR ones.
    const bool new_scene = inputs.serial != 0 && inputs.serial != last_serial;
    const bool sdr8 = format == vk::Format::eB8G8R8A8Unorm || format == vk::Format::eB8G8R8A8Srgb ||
                      format == vk::Format::eR8G8B8A8Unorm || format == vk::Format::eR8G8B8A8Srgb;
    // A game at more than half the refresh rate has no room for generated frames: each one
    // would take a refresh from a game frame (hysteresis keeps it from flickering on and off).
    if (new_scene) {
        smoothed_ms = smoothed_ms <= 0.0f ? frame_ms : smoothed_ms * 0.9f + frame_ms * 0.1f;
        const float refresh_ms = RefreshMs();
        if (fast_game ? smoothed_ms > 2.2f * refresh_ms : smoothed_ms < 1.9f * refresh_ms) {
            fast_game = !fast_game;
            std::printf("GPU: FSR frame generation %s (game frames %.1f ms, refresh %.1f ms)\n",
                        fast_game ? "paused" : "resumed", smoothed_ms, refresh_ms);
        }
    }
    if (!new_scene || !sdr8 || !Gow3Settings::Get().fg_enabled || fast_game) {
        last_serial = 0;
        return false;
    }
    if (context && (w != width || h != height || inputs.view_w > max_render_w ||
                    inputs.view_h > max_render_h)) {
        scheduler.Finish(); // its images and views are replaced
    }
    if (!EnsureContext(w, h, inputs.view_w, inputs.view_h)) {
        failed = true;
        return false;
    }
    Gow3Settings::Get().fg_active = true;
    // A frame without the TAAU in between, or a new context: nothing to interpolate from.
    const bool reset = last_serial == 0 || inputs.serial != last_serial + 1;
    last_serial = inputs.serial;
    ++frame_id;

    const auto cmdbuf = scheduler.CommandBuffer();
    if (reset) {
        const std::array<vk::Image, 3> own = {vk::Image(color), vk::Image(output), vk::Image(real)};
        std::array<vk::ImageMemoryBarrier2, 3> to_general{};
        for (u32 i = 0; i < 3; ++i) {
            to_general[i] = vk::ImageMemoryBarrier2{
                .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
                .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
                .dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
                .oldLayout = vk::ImageLayout::eUndefined,
                .newLayout = vk::ImageLayout::eGeneral,
                .image = own[i],
                .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
            };
        }
        cmdbuf.pipelineBarrier2({.imageMemoryBarrierCount = 3, .pImageMemoryBarriers = to_general.data()});
    }
    const vk::ImageCopy region{
        .srcSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
        .dstSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
        .extent = {w, h, 1},
    };
    MemoryBarrier(cmdbuf);
    // Raw copy: a BGRA frame stays BGRA in the RGBA8 image; interpolation works per channel.
    cmdbuf.copyImage(hudless ? hudless : frame, vk::ImageLayout::eGeneral, vk::Image(color),
                     vk::ImageLayout::eGeneral, region);
    if (hudless) {
        cmdbuf.copyImage(frame, vk::ImageLayout::eGeneral, vk::Image(real), vk::ImageLayout::eGeneral,
                         region);
    }
    MemoryBarrier(cmdbuf);

    FfxVkFsr3_3_1_6FrameGenerationPrepareInfo prepare{};
    prepare.commandBuffer = cmdbuf;
    prepare.color = Describe(vk::Image(color), kColorFormat, w, h, kColorUsage);
    prepare.depth = Describe(inputs.depth, vk::Format::eR32Sfloat, inputs.width, inputs.height,
                             vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage);
    prepare.motionVectors =
        Describe(inputs.motion, vk::Format::eR16G16Sfloat, inputs.width, inputs.height,
                 vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage);
    prepare.renderWidth = inputs.view_w;
    prepare.renderHeight = inputs.view_h;
    prepare.jitterOffsetX = inputs.jitter[0];
    prepare.jitterOffsetY = inputs.jitter[1];
    prepare.motionVectorScaleX = 1.0f;
    prepare.motionVectorScaleY = 1.0f;
    prepare.frameTimeMilliseconds = frame_ms;
    prepare.minLuminance = 0.0f;
    prepare.maxLuminance = 1.0f;
    prepare.transferFunction = FFX_VK_FSR3_3_1_6_FRAMEGEN_TRANSFER_SRGB;
    prepare.cameraNear = inputs.near_plane;
    prepare.cameraFar = kFarPlane;
    prepare.viewSpaceToMeters = kViewSpaceToMeters;
    prepare.cameraVerticalFovRadians = inputs.fov_y;
    prepare.cameraUp[1] = 1.0f;
    prepare.cameraRight[0] = 1.0f;
    prepare.cameraForward[2] = 1.0f;
    prepare.frameId = frame_id;
    prepare.reset = reset;

    FfxVkFsr3_3_1_6FrameGenerationDispatchInfo dispatch{};
    dispatch.commandBuffer = cmdbuf;
    dispatch.color = prepare.color;
    dispatch.output = Describe(vk::Image(output), kColorFormat, w, h, kColorUsage);
    dispatch.displayWidth = w;
    dispatch.displayHeight = h;
    dispatch.frameTimeMilliseconds = frame_ms;
    dispatch.cameraNear = prepare.cameraNear;
    dispatch.cameraFar = prepare.cameraFar;
    dispatch.viewSpaceToMeters = prepare.viewSpaceToMeters;
    dispatch.cameraVerticalFovRadians = prepare.cameraVerticalFovRadians;
    dispatch.minLuminance = prepare.minLuminance;
    dispatch.maxLuminance = prepare.maxLuminance;
    dispatch.transferFunction = prepare.transferFunction;
    dispatch.frameId = frame_id;
    dispatch.reset = reset;

    const auto t0 = std::chrono::steady_clock::now();
    auto result = ffxVkFsr3_3_1_6FrameGenerationContextRecordPrepare(context, &prepare);
    const auto t1 = std::chrono::steady_clock::now();
    if (result == FFX_VK_FSR3_3_1_6_FRAMEGEN_OK) {
        result = ffxVkFsr3_3_1_6FrameGenerationContextRecordDispatch(context, &dispatch);
    }
    {
        // TEMP timing
        static double prep_ms, disp_ms, rec_ms; static u32 n;
        const auto t2 = std::chrono::steady_clock::now();
        prep_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
        disp_ms += std::chrono::duration<double, std::milli>(t2 - t1).count();
        rec_ms += std::chrono::duration<double, std::milli>(t0 - now).count();
        if (++n == 150) {
            std::printf("FG record TEMP: before ffx %.2f ms, prepare %.2f ms, dispatch %.2f ms\n",
                        rec_ms / n, prep_ms / n, disp_ms / n);
            prep_ms = disp_ms = rec_ms = 0; n = 0;
        }
    }
    in_flight.emplace_back(frame_id, scheduler.CurrentTick());
    if (result != FFX_VK_FSR3_3_1_6_FRAMEGEN_OK) {
        std::printf("GPU: FSR frame generation off: recording failed (%d)\n", int(result));
        failed = true;
        return false;
    }
    MemoryBarrier(cmdbuf);
    if (hudless) {
        // The game frame's UI over the interpolated scene.
        const std::array<vk::DescriptorImageInfo, 3> infos = {
            vk::DescriptorImageInfo{.imageView = *real_view, .imageLayout = vk::ImageLayout::eGeneral},
            vk::DescriptorImageInfo{.imageView = *color_view, .imageLayout = vk::ImageLayout::eGeneral},
            vk::DescriptorImageInfo{.imageView = *output_view, .imageLayout = vk::ImageLayout::eGeneral},
        };
        std::array<vk::WriteDescriptorSet, 3> writes{};
        for (u32 i = 0; i < 3; ++i) {
            writes[i] = {.dstBinding = i,
                         .descriptorCount = 1,
                         .descriptorType = vk::DescriptorType::eStorageImage,
                         .pImageInfo = &infos[i]};
        }
        const ComposeConstants constants{{s32(w), s32(h)}, kHudThreshold};
        cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, *compose_pipeline);
        cmdbuf.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, *compose_pipeline_layout, 0,
                                    writes);
        cmdbuf.pushConstants(*compose_pipeline_layout, vk::ShaderStageFlagBits::eCompute, 0,
                             sizeof(constants), &constants);
        cmdbuf.dispatch((w + 7) / 8, (h + 7) / 8, 1);
        MemoryBarrier(cmdbuf);
    }
    cmdbuf.copyImage(vk::Image(output), vk::ImageLayout::eGeneral, out, vk::ImageLayout::eGeneral,
                     region);
    MemoryBarrier(cmdbuf);
    static bool logged = false;
    if (!logged) {
        logged = true;
        std::printf("GPU: FSR frame generation running: %ux%u from a %ux%u view (%s)\n", w, h,
                    inputs.view_w, inputs.view_h, hudless ? "HUD-less" : "with the HUD");
    }
    return true;
}

} // namespace Vulkan
