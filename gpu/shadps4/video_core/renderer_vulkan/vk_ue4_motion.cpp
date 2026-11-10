// SPDX-License-Identifier: GPL-2.0-or-later
// gow3 U2: UE4 motion vectors and their debug view (vk_ue4_motion.h).

#include "video_core/renderer_vulkan/vk_ue4_motion.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>

#include "video_core/host_shaders/ue4_motion_comp.h"
#include "video_core/host_shaders/ue4_motion_view_comp.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_runtime.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"
#include "video_core/renderer_vulkan/vk_ue4_frame_map.h"
#include "video_core/texture_cache/texture_cache.h"

namespace Vulkan {

namespace {

// View uniform buffer offsets (vk_ue4_frame_map.cpp, study-taau-frame-map.md section 3).
constexpr u32 OffsetClipToPrevClip = 1824;
constexpr u32 OffsetViewSize = 1952;
constexpr u32 OffsetFrameNumber = 2140;

struct MotionPush {
    std::array<float, 16> clip_to_prev_clip;
    std::array<float, 2> view_size;
    std::array<float, 2> image_size;
    u32 has_velocity;
};

struct ViewPush {
    std::array<float, 2> display_size;
    std::array<float, 2> view_size;
    u32 mode;
    u32 history;
    u32 swap_red_blue;
};

vk::UniqueDescriptorSetLayout MakeLayout(vk::Device device,
                                         std::initializer_list<vk::DescriptorType> types) {
    std::vector<vk::DescriptorSetLayoutBinding> bindings;
    for (const auto type : types) {
        bindings.push_back({.binding = u32(bindings.size()),
                            .descriptorType = type,
                            .descriptorCount = 1,
                            .stageFlags = vk::ShaderStageFlagBits::eCompute});
    }
    return Check(device.createDescriptorSetLayoutUnique({
        .flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR,
        .bindingCount = u32(bindings.size()),
        .pBindings = bindings.data(),
    }));
}

void MakePipeline(vk::Device device, std::span<const u32> code,
                  const vk::DescriptorSetLayout& set_layout, u32 push_size,
                  vk::UniquePipelineLayout& layout, vk::UniquePipeline& pipeline) {
    const vk::PushConstantRange range{.stageFlags = vk::ShaderStageFlagBits::eCompute,
                                      .offset = 0,
                                      .size = push_size};
    layout = Check(device.createPipelineLayoutUnique({
        .setLayoutCount = 1,
        .pSetLayouts = &set_layout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &range,
    }));
    const auto module = CompileSPV(code, device);
    pipeline = Check(device.createComputePipelineUnique(
        {}, vk::ComputePipelineCreateInfo{
                .stage = {.stage = vk::ShaderStageFlagBits::eCompute,
                          .module = module,
                          .pName = "main"},
                .layout = *layout,
            }));
    device.destroyShaderModule(module);
}

/// The raw (UNORM) format of an 8-bit RGBA display buffer, or Undefined.
vk::Format RawFormat(vk::Format format) {
    switch (format) {
    case vk::Format::eR8G8B8A8Unorm:
    case vk::Format::eR8G8B8A8Srgb:
        return vk::Format::eR8G8B8A8Unorm;
    default:
        return vk::Format::eUndefined;
    }
}

} // namespace

std::atomic<int> Ue4Motion::view_mode{0};

bool Ue4Motion::Enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("GOW3_UE4_MOTION_VIEW");
        if (!value || !*value) {
            return false;
        }
        view_mode.store(std::clamp(std::atoi(value), 0, 4), std::memory_order_relaxed);
        return true;
    }();
    return enabled;
}

void Ue4Motion::CycleViewMode() {
    if (!Enabled()) {
        return;
    }
    const int mode = (view_mode.load(std::memory_order_relaxed) + 1) % 5;
    view_mode.store(mode, std::memory_order_relaxed);
    static constexpr const char* names[] = {"off", "colours over the picture", "colours only",
                                            "last frame moved", "difference x4"};
    std::printf("UE4 motion: view %d (%s)\n", mode, names[mode]);
    std::fflush(stdout);
}

Ue4Motion::Ue4Motion(const Instance& instance_, Scheduler& scheduler_,
                     VideoCore::TextureCache& texture_cache_, Runtime& runtime_)
    : instance{instance_}, scheduler{scheduler_}, texture_cache{texture_cache_}, runtime{runtime_} {
    const auto device = instance.GetDevice();
    using T = vk::DescriptorType;
    motion_desc_layout = MakeLayout(device, {T::eSampledImage, T::eSampledImage, T::eStorageImage});
    MakePipeline(device, UE4_MOTION_COMP, *motion_desc_layout, sizeof(MotionPush), motion_layout,
                 motion_pipeline);
    view_desc_layout = MakeLayout(device, {T::eSampledImage, T::eSampledImage, T::eStorageImage,
                                           T::eStorageImage, T::eStorageImage});
    MakePipeline(device, UE4_MOTION_VIEW_COMP, *view_desc_layout, sizeof(ViewPush), view_layout,
                 view_pipeline);
    std::printf("UE4 motion: on, view %d (F10 cycles)\n", view_mode.load());
}

Ue4Motion::~Ue4Motion() = default;

void Ue4Motion::Ensure(OwnImage& own, u32 width, u32 height, vk::Format format,
                       vk::Format present_format) {
    if (own.image.image && own.width == width && own.height == height && own.format == format) {
        return;
    }
    if (own.image.image) {
        // Recorded commands of earlier frames may still use the old image: destroyed once the
        // GPU is past them (no Finish here: the presenter holds an open command buffer).
        struct Old {
            VideoCore::UniqueImage image;
            vk::UniqueImageView view, present;
        };
        auto old = std::make_shared<Old>(
            Old{std::move(own.image), std::move(own.view), std::move(own.present)});
        scheduler.DeferOperation([old] {});
    }
    const auto device = instance.GetDevice();
    own.image = VideoCore::UniqueImage(device, instance.GetAllocator());
    const std::array<vk::Format, 2> formats{format, present_format};
    const vk::ImageFormatListCreateInfo format_list{.viewFormatCount = 2,
                                                    .pViewFormats = formats.data()};
    const bool mutable_format = present_format != vk::Format::eUndefined && present_format != format;
    own.image.Create({
        .pNext = mutable_format ? &format_list : nullptr,
        .flags = mutable_format ? vk::ImageCreateFlagBits::eMutableFormat : vk::ImageCreateFlags{},
        .imageType = vk::ImageType::e2D,
        .format = format,
        .extent = {width, height, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = vk::SampleCountFlagBits::e1,
        .tiling = vk::ImageTiling::eOptimal,
        .usage = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled,
    });
    const auto make_view = [&](vk::Format view_format, vk::ComponentMapping mapping) {
        return Check(device.createImageViewUnique({
            .image = vk::Image(own.image),
            .viewType = vk::ImageViewType::e2D,
            .format = view_format,
            .components = mapping,
            .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
        }));
    };
    own.view = make_view(format, {});
    if (present_format != vk::Format::eUndefined) {
        // As the presenter's view of a display buffer: alpha ignored.
        own.present = make_view(present_format,
                                {vk::ComponentSwizzle::eIdentity, vk::ComponentSwizzle::eIdentity,
                                 vk::ComponentSwizzle::eIdentity, vk::ComponentSwizzle::eOne});
    }
    own.width = width;
    own.height = height;
    own.format = format;
    own.layout = vk::ImageLayout::eUndefined;
}

Ue4Motion::Target Ue4Motion::Remember(VideoCore::ImageId id) {
    auto* image = id ? texture_cache.TryGetImage(id, 0) : nullptr;
    return image ? Target{id, image->image_uid} : Target{};
}

VideoCore::Image* Ue4Motion::Find(const Target& target) {
    return target.id ? texture_cache.TryGetImage(target.id, target.uid) : nullptr;
}

void Ue4Motion::OnVelocityDraw(VideoCore::ImageId velocity_id, VAddr address,
                               VideoCore::ImageId depth_id) {
    if (velocity_id != velocity.id || !Find(velocity)) {
        velocity = Remember(velocity_id);
    }
    velocity_address = address;
    if (depth_id != depth.id || !Find(depth)) {
        depth = Remember(depth_id);
    }
}

void Ue4Motion::OnSceneDraw(VideoCore::ImageId depth_id) {
    if (depth_id && (depth_id != scene_depth.id || !Find(scene_depth))) {
        scene_depth = Remember(depth_id);
    }
}

void Ue4Motion::OnTaa(const u8* view, std::span<const VAddr> reads, VideoCore::ImageId taa_depth) {
    if (!view) {
        return;
    }
    u32 frame_number;
    std::memcpy(&frame_number, view + OffsetFrameNumber, sizeof(frame_number));
    if (recorded && frame_number == recorded_frame) {
        return; // the second TAA draw of the frame
    }
    // The velocity target when this TAA pass samples it (in gameplay frames without velocity
    // draws too, U1; not in menus): else camera motion only. The depth of the velocity pass,
    // else TAA's own, else the G-buffer pass's.
    auto* velocity_image =
        velocity_address && std::ranges::find(reads, velocity_address) != reads.end()
            ? Find(velocity)
            : nullptr;
    auto* depth_image = Find(depth);
    if (!depth_image && taa_depth) {
        depth = Remember(taa_depth);
        depth_image = Find(depth);
    }
    if (!depth_image) {
        depth_image = Find(scene_depth);
    }
    static bool warned = false;
    const auto warn = [&](const char* why) {
        if (!warned) {
            warned = true;
            std::printf("UE4 motion: pass skipped (%s)\n", why);
        }
    };
    if (!depth_image) {
        warn("no depth");
        return;
    }
    const auto depth_format = depth_image->info.pixel_format;
    if (depth_format != vk::Format::eD32Sfloat && depth_format != vk::Format::eD32SfloatS8Uint) {
        warn("depth format");
        return;
    }
    if (velocity_image && (velocity_image->info.pixel_format != vk::Format::eR16G16Unorm ||
                           velocity_image->info.size.width != depth_image->info.size.width ||
                           velocity_image->info.size.height != depth_image->info.size.height)) {
        warn("velocity format or size");
        velocity_image = nullptr;
    }
    recorded = true;
    recorded_frame = frame_number;
    pass_depth = Remember(depth_image == Find(depth) ? depth.id : scene_depth.id);

    const u32 width = depth_image->info.size.width;
    const u32 height = depth_image->info.size.height;
    MotionPush push;
    std::memcpy(push.clip_to_prev_clip.data(), view + OffsetClipToPrevClip,
                sizeof(push.clip_to_prev_clip));
    std::memcpy(push.view_size.data(), view + OffsetViewSize, sizeof(push.view_size));
    push.view_size = {std::clamp(push.view_size[0], 1.0f, float(width)),
                      std::clamp(push.view_size[1], 1.0f, float(height))};
    push.image_size = {float(width), float(height)};
    push.has_velocity = velocity_image ? 1u : 0u;
    view_size = push.view_size;

    Ensure(motion, width, height, vk::Format::eR16G16B16A16Sfloat);
    scheduler.EndRendering();
    if (velocity_image) {
        runtime.Transit(velocity_image, vk::ImageLayout::eGeneral,
                        vk::PipelineStageFlagBits2::eComputeShader,
                        vk::AccessFlagBits2::eShaderRead);
    }
    runtime.Transit(depth_image, vk::ImageLayout::eGeneral,
                    vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderRead);
    runtime.FlushBarriers();
    const auto device = instance.GetDevice();
    const vk::ImageView depth_view = Check(device.createImageView({
        .image = depth_image->GetImage(),
        .viewType = vk::ImageViewType::e2D,
        .format = depth_format,
        .subresourceRange = {vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1},
    }));
    // Without velocity the depth view fills the binding; the shader does not read it.
    const vk::ImageView velocity_view =
        velocity_image ? Check(device.createImageView({
                             .image = velocity_image->GetImage(),
                             .viewType = vk::ImageViewType::e2D,
                             .format = vk::Format::eR16G16Unorm,
                             .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
                         }))
                       : depth_view;
    const vk::Image motion_image = vk::Image(motion.image);
    const vk::ImageView motion_view = *motion.view;
    const vk::ImageLayout old_layout = motion.layout;
    motion.layout = vk::ImageLayout::eGeneral;
    scheduler.Record([=, pipeline = *motion_pipeline, layout = *motion_layout](
                         vk::CommandBuffer cmdbuf) {
        // Last frame's debug view read the motion image.
        const vk::ImageMemoryBarrier2 to_write{
            .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
            .srcAccessMask = vk::AccessFlagBits2::eShaderRead,
            .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
            .dstAccessMask = vk::AccessFlagBits2::eShaderWrite,
            .oldLayout = old_layout,
            .newLayout = vk::ImageLayout::eGeneral,
            .image = motion_image,
            .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
        };
        cmdbuf.pipelineBarrier2({.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &to_write});
        const vk::DescriptorImageInfo infos[3] = {
            {.imageView = velocity_view, .imageLayout = vk::ImageLayout::eGeneral},
            {.imageView = depth_view, .imageLayout = vk::ImageLayout::eGeneral},
            {.imageView = motion_view, .imageLayout = vk::ImageLayout::eGeneral},
        };
        const vk::DescriptorType types[3] = {vk::DescriptorType::eSampledImage,
                                             vk::DescriptorType::eSampledImage,
                                             vk::DescriptorType::eStorageImage};
        std::array<vk::WriteDescriptorSet, 3> writes{};
        for (u32 i = 0; i < 3; ++i) {
            writes[i] = {.dstBinding = i,
                         .descriptorCount = 1,
                         .descriptorType = types[i],
                         .pImageInfo = &infos[i]};
        }
        cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline);
        cmdbuf.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, layout, 0, writes);
        cmdbuf.pushConstants(layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push), &push);
        cmdbuf.dispatch((width + 7) / 8, (height + 7) / 8, 1);
    });
    scheduler.DeferOperation([device, velocity_view, depth_view] {
        if (velocity_view != depth_view) {
            device.destroyImageView(velocity_view);
        }
        device.destroyImageView(depth_view);
    });
    fresh = true;
    static const bool quiet = [] {
        const char* value = std::getenv("GOW3_FPS_LOG");
        return value && std::strcmp(value, "0") == 0;
    }();
    if (++passes == 1 || (!quiet && passes % 3000 == 0)) {
        std::printf("UE4 motion: %llu passes, last %ux%u (view %gx%g)\n",
                    static_cast<unsigned long long>(passes), width, height, push.view_size[0],
                    push.view_size[1]);
    }
}

bool Ue4Motion::GetPass(u32 frame_number, LastPass& pass) {
    if (!recorded || recorded_frame != frame_number || !motion.image.image) {
        return false;
    }
    auto* depth_image = Find(pass_depth);
    if (!depth_image) {
        return false;
    }
    pass = {vk::Image(motion.image), *motion.view, motion.width, motion.height, depth_image};
    return true;
}

vk::ImageView Ue4Motion::Compose(vk::CommandBuffer cmdbuf, VideoCore::Image& display,
                                 vk::Format present_format) {
    const int mode = view_mode.load(std::memory_order_relaxed);
    if (mode == 0 || !fresh || !motion.image.image) {
        history_valid = false;
        return {};
    }
    fresh = false;
    const vk::Format raw = RawFormat(display.info.pixel_format);
    if (raw == vk::Format::eUndefined) {
        static bool warned = false;
        if (!warned) {
            warned = true;
            std::printf("UE4 motion: view skipped (display %s)\n",
                        vk::to_string(display.info.pixel_format).c_str());
        }
        return {};
    }
    const u32 width = display.info.size.width;
    const u32 height = display.info.size.height;
    Ensure(output, width, height, raw, present_format);
    for (auto& image : history) {
        if (image.width != width || image.height != height) {
            history_valid = false;
        }
        Ensure(image, width, height, raw);
    }
    const auto device = instance.GetDevice();
    const vk::ImageView display_view = Check(device.createImageView({
        .image = display.GetImage(),
        .viewType = vk::ImageViewType::e2D,
        .format = raw,
        .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
    }));
    scheduler.DeferOperation([device, display_view] { device.destroyImageView(display_view); });

    auto& prev = history[history_index ^ 1];
    auto& next = history[history_index];
    history_index ^= 1;
    std::array<vk::ImageMemoryBarrier2, 3> barriers;
    u32 count = 0;
    for (auto* own : {&output, &prev, &next}) {
        barriers[count++] = {
            .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
            .dstAccessMask = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
            .oldLayout = own->layout,
            .newLayout = vk::ImageLayout::eGeneral,
            .image = vk::Image(own->image),
            .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
        };
        own->layout = vk::ImageLayout::eGeneral;
    }
    // The motion vector pass wrote the motion image earlier in this command stream.
    const vk::MemoryBarrier2 motion_written{
        .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .srcAccessMask = vk::AccessFlagBits2::eShaderWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
    };
    cmdbuf.pipelineBarrier2({.memoryBarrierCount = 1,
                             .pMemoryBarriers = &motion_written,
                             .imageMemoryBarrierCount = count,
                             .pImageMemoryBarriers = barriers.data()});
    const vk::DescriptorImageInfo infos[5] = {
        {.imageView = display_view, .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal},
        {.imageView = *motion.view, .imageLayout = vk::ImageLayout::eGeneral},
        {.imageView = *output.view, .imageLayout = vk::ImageLayout::eGeneral},
        {.imageView = *prev.view, .imageLayout = vk::ImageLayout::eGeneral},
        {.imageView = *next.view, .imageLayout = vk::ImageLayout::eGeneral},
    };
    std::array<vk::WriteDescriptorSet, 5> writes{};
    for (u32 i = 0; i < 5; ++i) {
        writes[i] = {.dstBinding = i,
                     .descriptorCount = 1,
                     .descriptorType = i < 2 ? vk::DescriptorType::eSampledImage
                                             : vk::DescriptorType::eStorageImage,
                     .pImageInfo = &infos[i]};
    }
    const ViewPush push{
        .display_size = {float(width), float(height)},
        .view_size = view_size,
        .mode = u32(mode),
        .history = history_valid ? 1u : 0u,
        // The presenter views an RGBA display buffer as BGRA (A8R8G8B8 guest buffers).
        .swap_red_blue = (raw == vk::Format::eR8G8B8A8Unorm &&
                          (present_format == vk::Format::eB8G8R8A8Srgb ||
                           present_format == vk::Format::eB8G8R8A8Unorm))
                             ? 1u
                             : 0u,
    };
    history_valid = true;
    cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, *view_pipeline);
    cmdbuf.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, *view_layout, 0, writes);
    cmdbuf.pushConstants(*view_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push), &push);
    cmdbuf.dispatch((width + 7) / 8, (height + 7) / 8, 1);
    const vk::ImageMemoryBarrier2 to_present{
        .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .srcAccessMask = vk::AccessFlagBits2::eShaderWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
        .oldLayout = vk::ImageLayout::eGeneral,
        .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
        .image = vk::Image(output.image),
        .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
    };
    cmdbuf.pipelineBarrier2({.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &to_present});
    output.layout = vk::ImageLayout::eShaderReadOnlyOptimal;
    return *output.present;
}

} // namespace Vulkan
