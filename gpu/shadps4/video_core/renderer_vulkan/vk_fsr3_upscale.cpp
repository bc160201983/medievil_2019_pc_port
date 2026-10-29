// SPDX-License-Identifier: GPL-2.0-or-later

#include "video_core/renderer_vulkan/vk_fsr3_upscale.h"

#include <ffx_vk_fsr3_3_1_5_bridge.h>

#include "video_core/renderer_vulkan/vk_instance.h"

namespace Vulkan {

namespace {

// As frame generation (vk_frame_gen.cpp): UE4's reversed infinite projection, centimetres.
constexpr float kFarPlane = 1.0e6f;
constexpr float kViewSpaceToMeters = 0.01f;

constexpr vk::ImageUsageFlags kInputUsage =
    vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage;

} // namespace

Fsr3Upscaler::Fsr3Upscaler(const Instance& instance_) : instance{instance_} {}

Fsr3Upscaler::~Fsr3Upscaler() {
    Destroy();
}

void Fsr3Upscaler::Destroy() {
    if (context) {
        FfxVkFsr3_3_1_5Bridge* const bridge = ffxVkFsr3_3_1_5UpscalerContextGetBridge(context);
        for (void*& token : imported) {
            if (token) {
                ffxVkFsr3_3_1_5BridgeReleaseImportedImage(bridge, {token});
            }
            token = nullptr;
        }
        ffxVkFsr3_3_1_5UpscalerContextDestroy(context);
        context = nullptr;
    }
    dilated_depth = VideoCore::UniqueImage();
    dilated_motion = VideoCore::UniqueImage();
    prev_depth = VideoCore::UniqueImage();
    width = height = out_width = out_height = 0;
}

bool Fsr3Upscaler::Create(const Frame& frame) {
    const FfxVkFsr3_3_1_5UpscalerCreateInfo create{
        .physicalDevice = instance.GetPhysicalDevice(),
        .device = instance.GetDevice(),
        .maxRenderWidth = frame.width,
        .maxRenderHeight = frame.height,
        .maxUpscaleWidth = frame.out_width,
        .maxUpscaleHeight = frame.out_height,
        .hdrColorInput = VK_TRUE,
        .autoExposure = frame.auto_exposure ? VK_TRUE : VK_FALSE,
    };
    if (const auto result = ffxVkFsr3_3_1_5UpscalerContextCreate(&create, &context);
        result != FFX_VK_FSR3_3_1_5_OK) {
        context = nullptr;
        error = "context creation failed (" + std::to_string(int(result)) + ")";
        return false;
    }
    FfxVkFsr3_3_1_5SharedResourceDescriptions shared{};
    if (ffxVkFsr3_3_1_5UpscalerContextGetSharedResourceDescriptions(context, &shared) !=
        FFX_VK_FSR3_3_1_5_OK) {
        error = "no shared resource descriptions";
        return false;
    }
    FfxVkFsr3_3_1_5Bridge* const bridge = ffxVkFsr3_3_1_5UpscalerContextGetBridge(context);
    const auto import = [&](vk::Image image, vk::Format format, u32 w, u32 h, u32 state,
                            vk::ImageUsageFlags usage) {
        const FfxVkFsr3_3_1_5ImportedImageDescription description{
            .image = image,
            .format = static_cast<VkFormat>(format),
            .width = w,
            .height = h,
            .mipCount = 1,
            .arrayLayers = 1,
            .layout = VK_IMAGE_LAYOUT_GENERAL,
            .state = state,
            .usage = static_cast<VkImageUsageFlags>(usage),
        };
        return ffxVkFsr3_3_1_5BridgeImportImage(bridge, &description).resource;
    };
    const auto make = [&](VideoCore::UniqueImage& image, const auto& desc) {
        image = VideoCore::UniqueImage(instance.GetDevice(), instance.GetAllocator());
        image.Create(vk::ImageCreateInfo{
            .imageType = vk::ImageType::e2D,
            .format = static_cast<vk::Format>(desc.format),
            .extent = {desc.width, desc.height, 1},
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = vk::SampleCountFlagBits::e1,
            .tiling = vk::ImageTiling::eOptimal,
            .usage = kInputUsage,
            .initialLayout = vk::ImageLayout::eUndefined,
        });
        return import(vk::Image(image), static_cast<vk::Format>(desc.format), desc.width,
                      desc.height, FFX_VK_FSR3_3_1_5_RESOURCE_STATE_UNORDERED_ACCESS, kInputUsage);
    };
    const u32 read = FFX_VK_FSR3_3_1_5_RESOURCE_STATE_COMPUTE_READ;
    imported[0] = import(frame.color, vk::Format::eR16G16B16A16Sfloat, frame.width, frame.height,
                         read, kInputUsage);
    imported[1] = import(frame.depth, vk::Format::eR32Sfloat, frame.width, frame.height, read,
                         kInputUsage);
    imported[2] = import(frame.motion, vk::Format::eR16G16Sfloat, frame.width, frame.height, read,
                         kInputUsage);
    imported[3] = import(frame.output, vk::Format::eR16G16B16A16Sfloat, frame.out_width,
                         frame.out_height, FFX_VK_FSR3_3_1_5_RESOURCE_STATE_UNORDERED_ACCESS,
                         kInputUsage | vk::ImageUsageFlagBits::eTransferSrc);
    imported[4] = make(dilated_depth, shared.dilatedDepth);
    imported[5] = make(dilated_motion, shared.dilatedMotionVectors);
    imported[6] = make(prev_depth, shared.reconstructedPrevNearestDepth);
    for (const void* token : imported) {
        if (!token) {
            error = "image import failed";
            return false;
        }
    }
    // The shared images start in General, as imported.
    std::array<vk::ImageMemoryBarrier2, 3> barriers;
    const vk::Image own[3] = {vk::Image(dilated_depth), vk::Image(dilated_motion),
                              vk::Image(prev_depth)};
    for (u32 i = 0; i < 3; ++i) {
        barriers[i] = {
            .srcStageMask = vk::PipelineStageFlagBits2::eNone,
            .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
            .oldLayout = vk::ImageLayout::eUndefined,
            .newLayout = vk::ImageLayout::eGeneral,
            .image = own[i],
            .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
        };
    }
    frame.cmdbuf.pipelineBarrier2(
        {.imageMemoryBarrierCount = u32(barriers.size()), .pImageMemoryBarriers = barriers.data()});
    images[0] = frame.color;
    images[1] = frame.depth;
    images[2] = frame.motion;
    images[3] = frame.output;
    width = frame.width;
    height = frame.height;
    out_width = frame.out_width;
    out_height = frame.out_height;
    auto_exposure = frame.auto_exposure;
    return true;
}

bool Fsr3Upscaler::Record(const Frame& frame) {
    // The images, their sizes or the exposure mode changed: a new context (the caller waited).
    if (context && (width != frame.width || height != frame.height ||
                    out_width != frame.out_width || out_height != frame.out_height ||
                    auto_exposure != frame.auto_exposure || images[0] != frame.color ||
                    images[1] != frame.depth || images[2] != frame.motion ||
                    images[3] != frame.output)) {
        Destroy();
    }
    bool reset = frame.reset;
    if (!context) {
        if (!Create(frame)) {
            Destroy();
            return false;
        }
        reset = true;
    }
    if (frame.completed_id) {
        ffxVkFsr3_3_1_5UpscalerContextRetireFrame(context, frame.completed_id);
    }
    FfxVkFsr3_3_1_5UpscalerDispatchInfo dispatch{};
    dispatch.commandBuffer = frame.cmdbuf;
    dispatch.color = {imported[0]};
    dispatch.depth = {imported[1]};
    dispatch.motionVectors = {imported[2]};
    dispatch.output = {imported[3]};
    dispatch.dilatedDepth = {imported[4]};
    dispatch.dilatedMotionVectors = {imported[5]};
    dispatch.reconstructedPrevNearestDepth = {imported[6]};
    dispatch.jitterOffsetX = frame.jitter[0];
    dispatch.jitterOffsetY = frame.jitter[1];
    dispatch.motionVectorScaleX = 1.0f;
    dispatch.motionVectorScaleY = 1.0f;
    dispatch.renderWidth = frame.render_width;
    dispatch.renderHeight = frame.render_height;
    dispatch.upscaleWidth = frame.out_width;
    dispatch.upscaleHeight = frame.out_height;
    dispatch.frameTimeMilliseconds = frame.frame_ms > 0.0f ? frame.frame_ms : 16.7f;
    dispatch.preExposure = 1.0f;
    dispatch.enableSharpening = frame.sharpness > 0.0f ? VK_TRUE : VK_FALSE;
    dispatch.sharpness = frame.sharpness;
    dispatch.reset = reset ? VK_TRUE : VK_FALSE;
    dispatch.cameraNear = frame.near_plane;
    dispatch.cameraFar = kFarPlane;
    dispatch.cameraVerticalFovRadians = frame.fov_y;
    dispatch.viewSpaceToMeters = kViewSpaceToMeters;
    dispatch.frameId = frame.frame_id;
    if (const auto result = ffxVkFsr3_3_1_5UpscalerContextRecordDispatch(context, &dispatch);
        result != FFX_VK_FSR3_3_1_5_OK) {
        error = "dispatch failed (" + std::to_string(int(result)) + ")";
        return false;
    }
    return true;
}

} // namespace Vulkan
