// SPDX-License-Identifier: GPL-2.0-or-later
#ifdef GOW3_HAVE_DLSSNR
// First: its own Vulkan prototypes (vk_common.h sets VK_NO_PROTOTYPES), with vk_common.h's
// beta extensions so vulkan.hpp still finds them.
#define VK_ENABLE_BETA_EXTENSIONS
#include "nr_runtime.hpp"
#else
namespace nr {
class Runtime {}; // the network is not built in: never made
} // namespace nr
#endif

#include "video_core/renderer_vulkan/vk_neural_render.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <atomic>
#include <cstring>
#include <filesystem>
#include <string>
#include <unordered_set>

#include "gow3_settings.h"
#include "video_core/host_shaders/ue4_motion_comp.h"
#include "video_core/renderer_vulkan/fsr411/fsr411.h"
#include "video_core/renderer_vulkan/vk_fsr3_upscale.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_runtime.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"
#include "video_core/texture_cache/texture_cache.h"

namespace Vulkan {

namespace {

struct PushConstants {
    std::array<float, 16> clip_to_prev;
    std::array<s32, 2> rect_min;
    std::array<s32, 2> size;
    std::array<s32, 2> extent;
    u32 flags; ///< ue4_motion.comp: 1 motion view, 2 FSR 4 inputs
};

// UE 4.21 FViewUniformShaderParameters, in floats (docs: memory ue4-taau-frame-layout).
// ViewToClip: row-major from float 96; [1][1] = 1/tan(fov_y/2), [3][2] = near (reversed infinite).
constexpr u32 ViewToClip = 96;
constexpr u32 ClipToPrevClip = 456;
// TemporalAAJitter: current x, y then previous x, y, in clip space (y up; pixels = v * size / 2).
constexpr u32 TemporalAAJitter = 472;
constexpr u32 ViewRectMin = 484;
constexpr u32 ViewSizeAndInvSize = 488;

} // namespace

NeuralRender::NeuralRender(const Instance& instance_, Scheduler& scheduler_,
                           VideoCore::TextureCache& texture_cache_, Runtime& runtime_)
    : instance{instance_}, scheduler{scheduler_}, texture_cache{texture_cache_}, runtime{runtime_} {
    const char* env = std::getenv("GOW3_NR");
    const char* fsr4_env = std::getenv("GOW3_UE4_FSR4");
    fsr4_mode = fsr4_env && fsr4_env[0] == '1';
    const char* fsr3_env = std::getenv("GOW3_UE4_FSR3");
    fsr3_mode = fsr3_env && fsr3_env[0] == '1';
    nr_mode = env && env[0] == '1';
    const char* fg_env = std::getenv("GOW3_FRAMEGEN");
    fg_mode = fg_env && fg_env[0] == '1';
    if (fsr4_mode && !instance.IsFsr411Supported() && !instance.IsFsr412Supported()) {
        std::printf("GPU: FSR 4 off: the GPU lacks INT8 dot products or "
                    "VK_VALVE_shader_mixed_float_dot_product; FSR 3.1 instead\n");
        fsr4_mode = false;
        fsr3_mode = true;
    }
    if (fsr4_mode) {
        fsr3_mode = false;
    }
    if (!nr_mode && !fsr4_mode && !fsr3_mode && !fg_mode) {
        return;
    }
#ifndef GOW3_HAVE_DLSSNR
    if (nr_mode) {
        std::printf("GPU: DLSSNR-AMD is not part of this build\n");
        nr_mode = false;
    }
#endif
    if (nr_mode && !instance.IsNeuralRenderSupported()) {
        std::printf("GPU: DLSSNR-AMD off: the GPU lacks its features (RDNA4 needed)\n");
        nr_mode = false;
    }
    if (!nr_mode && !fsr4_mode && !fsr3_mode && !fg_mode) {
        return;
    }
    const char* debug_env = std::getenv("GOW3_NR_DEBUG");
    debug = debug_env && debug_env[0] == '1';
    const auto device = instance.GetDevice();
    const auto binding = [](u32 index, vk::DescriptorType type) {
        return vk::DescriptorSetLayoutBinding{.binding = index,
                                              .descriptorType = type,
                                              .descriptorCount = 1,
                                              .stageFlags = vk::ShaderStageFlagBits::eCompute};
    };
    const std::array<vk::DescriptorSetLayoutBinding, 6> bindings = {
        binding(0, vk::DescriptorType::eSampledImage), binding(1, vk::DescriptorType::eSampledImage),
        binding(2, vk::DescriptorType::eStorageImage), binding(3, vk::DescriptorType::eStorageImage),
        binding(4, vk::DescriptorType::eStorageImage), binding(5, vk::DescriptorType::eSampledImage)};
    desc_layout = Check(device.createDescriptorSetLayoutUnique({
        .flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR,
        .bindingCount = static_cast<u32>(bindings.size()),
        .pBindings = bindings.data(),
    }));
    const vk::PushConstantRange range{.stageFlags = vk::ShaderStageFlagBits::eCompute,
                                      .offset = 0,
                                      .size = sizeof(PushConstants)};
    pipeline_layout = Check(device.createPipelineLayoutUnique({
        .setLayoutCount = 1,
        .pSetLayouts = &*desc_layout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &range,
    }));
    const auto module = CompileSPV(UE4_MOTION_COMP, device);
    pipeline = Check(device.createComputePipelineUnique(
        {}, vk::ComputePipelineCreateInfo{
                .stage = {.stage = vk::ShaderStageFlagBits::eCompute,
                          .module = module,
                          .pName = "main"},
                .layout = *pipeline_layout,
            }));
    device.destroyShaderModule(module);
    enabled = true;
    std::printf("GPU: %s on the UE4 TAAU%s%s\n",
                nr_mode && fsr4_mode ? "DLSSNR-AMD neural rendering, then FSR 4.1.1"
                : fsr4_mode          ? "FSR 4.1.1"
                : nr_mode && fsr3_mode ? "DLSSNR-AMD neural rendering, then FSR 3.1.5"
                : fsr3_mode          ? "FSR 3.1.5"
                : nr_mode            ? "DLSSNR-AMD neural rendering"
                                     : "motion and depth",
                fg_mode ? " (inputs of FSR frame generation)" : "",
                debug ? " (motion vector view)" : "");
}

NeuralRender::~NeuralRender() = default;

void NeuralRender::OnBuffer(const void* data, u64 size) {
    if (capturing && size == ViewBufferSize) {
        std::memcpy(view.data(), data, ViewBufferSize);
        view_valid = true;
    }
}

void NeuralRender::EnsureTargets(u32 w, u32 h) {
    if (color && w <= width && h <= height) {
        return;
    }
    if (color) {
        // A bigger view than any before: the network is rebuilt for it too.
        scheduler.Finish();
        network.reset();
        w = std::max(w, width);
        h = std::max(h, height);
    }
    const auto device = instance.GetDevice();
    const auto make = [&](VideoCore::UniqueImage& image, vk::UniqueImageView& view_out,
                          vk::Format format, vk::ImageUsageFlags usage) {
        view_out.reset();
        image = VideoCore::UniqueImage(device, instance.GetAllocator());
        image.Create(vk::ImageCreateInfo{
            .imageType = vk::ImageType::e2D,
            .format = format,
            .extent = {w, h, 1},
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = vk::SampleCountFlagBits::e1,
            .tiling = vk::ImageTiling::eOptimal,
            .usage = usage | vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled,
            .initialLayout = vk::ImageLayout::eUndefined,
        });
        view_out = Check(device.createImageViewUnique({
            .image = vk::Image(image),
            .viewType = vk::ImageViewType::e2D,
            .format = format,
            .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
        }));
    };
    make(color, color_view, vk::Format::eR16G16B16A16Sfloat,
         vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst);
    make(motion, motion_view, vk::Format::eR16G16Sfloat, {});
    make(depth, depth_view, vk::Format::eR32Sfloat, {});
    width = w;
    height = h;
    fresh_targets = true;
    reset_history = true;
    fsr4_reset = true;
}

bool NeuralRender::EnsureRuntime(u32 w, u32 h) {
#ifdef GOW3_HAVE_DLSSNR
    if (network) {
        return true;
    }
    std::string root;
    if (const char* env = std::getenv("GOW3_NR_ROOT"); env && env[0]) {
        root = env;
    } else if (const char* home = std::getenv("HOME")) {
        root = std::string(home) + "/.local/share/dlssnr/root";
    }
    nr::HostDevice host{};
    host.instance = instance.GetInstance();
    host.physical = instance.GetPhysicalDevice();
    host.device = instance.GetDevice();
    host.queue = instance.GetGraphicsQueue();
    host.queue_family = instance.GetGraphicsQueueFamilyIndex();
    host.queue_lock = [] { Scheduler::submit_mutex.lock(); };
    host.queue_unlock = [] { Scheduler::submit_mutex.unlock(); };
    nr::RuntimeConfig config{};
    config.root = root;
    config.width = w;
    config.height = h;
    config.colour_format = VK_FORMAT_R16G16B16A16_SFLOAT;
    config.linear_input = true;
    config.max_passes = 3;
    nr::TemporalConfig temporal{};
    temporal.enable = true;
    std::printf("GPU: DLSSNR-AMD building the network for %ux%u (root %s)...\n", w, h,
                root.c_str());
    try {
        network = std::make_unique<nr::Runtime>(host, config, nr::ControlMaskConfig{}, temporal);
    } catch (const std::exception& e) {
        std::printf("GPU: DLSSNR-AMD off: %s\n", e.what());
        enabled = false;
        return false;
    }
    reset_history = true;
    std::printf("GPU: DLSSNR-AMD ready\n");
    Gow3Settings::Get().nr_active = true;
    return true;
#else
    return false;
#endif
}

void NeuralRender::Apply() {
    capturing = false;
    // GOW3_NR_HASHES: every compute shader's sampled images, once per shader (finds TAAU variants).
    if (!IsTaau(current_hash)) {
        static std::unordered_set<u64> logged;
        if (logged.insert(current_hash).second) {
            std::printf("GPU: DLSSNR-AMD compute 0x%016llx view %d:", static_cast<unsigned long long>(current_hash), view_valid);
            for (const auto id : images) {
                const auto& image = texture_cache.GetImage(id);
                std::printf(" %s %ux%u", vk::to_string(image.info.pixel_format).c_str(),
                            image.info.size.width, image.info.size.height);
            }
            std::printf("\n");
        }
        return;
    }
    static bool seen = false;
    if (!seen) {
        seen = true;
        std::printf("GPU: DLSSNR-AMD found the TAAU (%zu images, View buffer %d)\n", images.size(),
                    view_valid);
    }
    if (!enabled || !view_valid) {
        return;
    }
    // DLSSNR rewrites the TAAU's input colour; FSR 4 runs after the TAAU (AfterDispatch) on
    // the same inputs, so with both it upscales DLSSNR's result.
    if (nr_mode) {
        Prepare(debug ? 1u : 0u);
    }
    if (fsr4_mode || fsr3_mode || fg_mode) {
        post_pending = true;
    }
}

bool NeuralRender::Prepare(u32 flags) {
    VideoCore::Image* scene = nullptr;
    VideoCore::Image* scene_depth = nullptr;
    VideoCore::Image* velocity = nullptr;
    for (const auto id : images) {
        auto& image = texture_cache.GetImage(id);
        const auto format = image.info.pixel_format;
        // The depth buffer is sampled through an R32F copy of it.
        if (image.info.props.is_depth || format == vk::Format::eR32Sfloat) {
            scene_depth = scene_depth ? scene_depth : &image;
        } else if (format == vk::Format::eR16G16Unorm) {
            velocity = velocity ? velocity : &image;
        } else if (format == vk::Format::eR16G16B16A16Sfloat && !scene) {
            scene = &image;
        }
    }
    rect_x = s32(view[ViewRectMin]);
    rect_y = s32(view[ViewRectMin + 1]);
    const u32 w = u32(view[ViewSizeAndInvSize]), h = u32(view[ViewSizeAndInvSize + 1]);
    view_w = w;
    view_h = h;
    static bool warned = false;
    if (!scene || !scene_depth || !velocity || w < 64 || h < 64 || rect_x < 0 || rect_y < 0 ||
        rect_x + w > scene->info.size.width || rect_y + h > scene->info.size.height ||
        rect_x + w > velocity->info.size.width || rect_x + w > scene_depth->info.size.width) {
        if (!warned) {
            warned = true;
            for (const auto id : images) {
                const auto& image = texture_cache.GetImage(id);
                std::printf("  %s %ux%u depth %d\n", vk::to_string(image.info.pixel_format).c_str(),
                            image.info.size.width, image.info.size.height,
                            int(image.info.props.is_depth));
            }
            std::printf("GPU: UE4 TAAU hook skipped a TAAU (colour %d depth %d velocity %d, view "
                        "%dx%d+%d+%d)\n",
                        scene != nullptr, scene_depth != nullptr, velocity != nullptr, w, h,
                        rect_x, rect_y);
        }
        return false;
    }
    EnsureTargets(w, h);
    const bool fsr = (flags & 2u) != 0;
    if (!fsr && !debug && nr_mode && !EnsureRuntime(width, height)) {
        return false;
    }

    const auto device = instance.GetDevice();
    const auto scene_depth_view = Check(device.createImageView({
        .image = scene_depth->GetImage(),
        .viewType = vk::ImageViewType::e2D,
        .format = scene_depth->info.pixel_format,
        .subresourceRange = {scene_depth->info.props.is_depth ? vk::ImageAspectFlagBits::eDepth
                                                              : vk::ImageAspectFlagBits::eColor,
                             0, 1, 0, 1},
    }));
    const auto velocity_view = Check(device.createImageView({
        .image = velocity->GetImage(),
        .viewType = vk::ImageViewType::e2D,
        .format = vk::Format::eR16G16Unorm,
        .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
    }));
    const auto scene_view = Check(device.createImageView({
        .image = scene->GetImage(),
        .viewType = vk::ImageViewType::e2D,
        .format = vk::Format::eR16G16B16A16Sfloat,
        .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
    }));
    const vk::ImageLayout depth_layout = scene_depth->backing->state.layout;
    const vk::ImageLayout velocity_layout = velocity->backing->state.layout;
    const vk::ImageLayout scene_layout = scene->backing->state.layout;
    runtime.Transit(scene, vk::ImageLayout::eGeneral, vk::PipelineStageFlagBits2::eAllCommands,
                    vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eTransferWrite);
    runtime.FlushBarriers();

    PushConstants push{};
    std::memcpy(push.clip_to_prev.data(), &view[ClipToPrevClip], sizeof(push.clip_to_prev));
    push.rect_min = {rect_x, rect_y};
    push.size = {s32(w), s32(h)};
    // FSR 4 reads the view only (its render size): motion in the view's pixels.
    push.extent = fsr ? std::array<s32, 2>{s32(w), s32(h)} : std::array<s32, 2>{s32(width), s32(height)};
    const u32 ext_w = u32(push.extent[0]), ext_h = u32(push.extent[1]);
    push.flags = flags;
    // Clip space, y up -> render pixels, y down.
    jitter = {view[TemporalAAJitter] * 0.5f * float(w), -view[TemporalAAJitter + 1] * 0.5f * float(h)};

    const bool fresh = std::exchange(fresh_targets, false);
    const bool reset = fsr ? false : std::exchange(reset_history, false);
    const vk::Image scene_image = scene->GetImage();
    const vk::Image own_color = vk::Image(color), own_motion = vk::Image(motion),
                    own_depth = vk::Image(depth);
    const vk::ImageView own_color_view = *color_view, own_motion_view = *motion_view,
                        own_depth_view = *depth_view;
    const vk::Pipeline handle = *pipeline;
    const vk::PipelineLayout layout = *pipeline_layout;
    const bool log = ++frames % 600 == 1;
    // GOW3_NR_OFF_FILE: while that file exists the network is bypassed (A/B in one session).
    static const char* off_file = std::getenv("GOW3_NR_OFF_FILE");
    static bool bypass = false;
    if (off_file && frames % 30 == 0) {
        bypass = std::filesystem::exists(off_file);
    }
    const auto& settings = Gow3Settings::Get();
    const bool run_network = !fsr && nr_mode && !debug && !bypass && settings.nr_enabled;
#ifdef GOW3_HAVE_DLSSNR
    nr::Runtime* const net = network.get();
    nr::Controls controls{};
    controls.intensity = settings.nr_intensity / 100.0f;
    controls.detail_strength = settings.nr_detail / 100.0f;
    controls.colour_strength = settings.nr_colour / 100.0f;
    controls.max_ratio = settings.nr_highlight / 100.0f;
    controls.local_tone = settings.nr_local_tone / 100.0f;
    controls.local_structure = settings.nr_local_structure / 100.0f;
    controls.passes = std::clamp(settings.nr_passes.load(), 1, 3);
#endif
    // FSR 4 mode keeps the scene untouched (no copy back) unless the motion view is drawn.
    const bool copy_back = !fsr || debug;
    const s32 rx = rect_x, ry = rect_y;
    scheduler.Record([=](vk::CommandBuffer cmdbuf) {
        const auto barrier = [&](vk::PipelineStageFlags2 src, vk::PipelineStageFlags2 dst) {
            const vk::MemoryBarrier2 memory{
                .srcStageMask = src,
                .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
                .dstStageMask = dst,
                .dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
            };
            cmdbuf.pipelineBarrier2({.memoryBarrierCount = 1, .pMemoryBarriers = &memory});
        };
        if (fresh) {
            std::array<vk::ImageMemoryBarrier2, 3> to_general{};
            const std::array<vk::Image, 3> own = {own_color, own_motion, own_depth};
            for (u32 i = 0; i < 3; ++i) {
                to_general[i] = vk::ImageMemoryBarrier2{
                    .srcStageMask = vk::PipelineStageFlagBits2::eNone,
                    .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
                    .dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
                    .oldLayout = vk::ImageLayout::eUndefined,
                    .newLayout = vk::ImageLayout::eGeneral,
                    .image = own[i],
                    .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
                };
            }
            cmdbuf.pipelineBarrier2({.imageMemoryBarrierCount = 3,
                                     .pImageMemoryBarriers = to_general.data()});
        }
        // The previous frame's reads of our images are done before they are rewritten.
        barrier(vk::PipelineStageFlagBits2::eAllCommands, vk::PipelineStageFlagBits2::eAllCommands);
        const vk::DescriptorImageInfo infos[6] = {
            {.imageView = scene_depth_view, .imageLayout = depth_layout},
            {.imageView = velocity_view, .imageLayout = velocity_layout},
            {.imageView = own_motion_view, .imageLayout = vk::ImageLayout::eGeneral},
            {.imageView = own_depth_view, .imageLayout = vk::ImageLayout::eGeneral},
            {.imageView = own_color_view, .imageLayout = vk::ImageLayout::eGeneral},
            {.imageView = scene_view, .imageLayout = vk::ImageLayout::eGeneral},
        };
        std::array<vk::WriteDescriptorSet, 6> writes{};
        for (u32 i = 0; i < 6; ++i) {
            writes[i] = vk::WriteDescriptorSet{
                .dstBinding = i,
                .descriptorCount = 1,
                .descriptorType = i < 2 || i == 5 ? vk::DescriptorType::eSampledImage
                                                  : vk::DescriptorType::eStorageImage,
                .pImageInfo = &infos[i],
            };
        }
        cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, handle);
        cmdbuf.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, layout, 0, writes);
        cmdbuf.pushConstants(layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push), &push);
        cmdbuf.dispatch((ext_w + 7) / 8, (ext_h + 7) / 8, 1);
        barrier(vk::PipelineStageFlagBits2::eComputeShader, vk::PipelineStageFlagBits2::eAllCommands);

#ifdef GOW3_HAVE_DLSSNR
        if (run_network) {
            nr::EngineFrame frame{};
            const auto general = [&](vk::Image image, VkFormat format, VkImageUsageFlags usage) {
                nr::ColourFrame out{};
                out.image = image;
                out.format = format;
                out.width = ext_w;
                out.height = ext_h;
                out.usage = usage;
                return out;
            };
            frame.colour = general(own_color, VK_FORMAT_R16G16B16A16_SFLOAT,
                                   VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                                       VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                       VK_IMAGE_USAGE_TRANSFER_DST_BIT);
            frame.motion = general(own_motion, VK_FORMAT_R16G16_SFLOAT,
                                   VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
            frame.depth = general(own_depth, VK_FORMAT_R32_SFLOAT,
                                  VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
            frame.depth_inverted = true;
            frame.reset = reset;
            const auto result = net->record_engine(cmdbuf, frame, controls);
            if (log) {
                std::printf("GPU: DLSSNR-AMD %ux%u: %u dispatches, applied %d, history %d (%s)\n",
                            w, h, result.frame.network_dispatches, result.frame.applied,
                            result.history_consumed, result.motion_provider);
            }
        }
#endif
        if (copy_back) {
            barrier(vk::PipelineStageFlagBits2::eAllCommands, vk::PipelineStageFlagBits2::eTransfer);
            const vk::ImageCopy out_region{
                .srcSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
                .dstSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
                .dstOffset = {rx, ry, 0},
                .extent = {w, h, 1},
            };
            cmdbuf.copyImage(own_color, vk::ImageLayout::eGeneral, scene_image,
                             vk::ImageLayout::eGeneral, out_region);
        }
    });
    (void)log;
    (void)run_network;
    (void)reset;
    scheduler.DeferOperation([device, scene_depth_view, velocity_view, scene_view] {
        device.destroyImageView(scene_view);
        device.destroyImageView(scene_depth_view);
        device.destroyImageView(velocity_view);
    });
    // Back to the layout the TAAU's descriptors were written with.
    runtime.Transit(scene, scene_layout, vk::PipelineStageFlagBits2::eAllCommands,
                    vk::AccessFlagBits2::eShaderRead);
    runtime.FlushBarriers();
    if (fsr && !debug) {
        fg_inputs.motion = own_motion;
        fg_inputs.depth = own_depth;
        fg_inputs.width = width;
        fg_inputs.height = height;
        fg_inputs.view_w = w;
        fg_inputs.view_h = h;
        fg_inputs.jitter = jitter;
        const float m11 = view[ViewToClip + 5], near_plane = view[ViewToClip + 14];
        fg_inputs.fov_y = m11 > 0.0f ? 2.0f * std::atan(1.0f / m11) : 1.0f;
        fg_inputs.near_plane = near_plane > 0.0f ? near_plane : 10.0f;
        ++fg_inputs.serial;
        hudless_wanted = fg_mode;
    }
    return true;
}

void NeuralRender::CaptureHudless(VideoCore::ImageId id) {
    hudless_wanted = false;
    auto& image = texture_cache.GetImage(id);
    const u32 w = image.info.size.width, h = image.info.size.height;
    const vk::Format format = image.info.pixel_format;
    if (!hudless || w != fg_inputs.hudless_w || h != fg_inputs.hudless_h ||
        format != fg_inputs.hudless_format) {
        if (hudless) {
            scheduler.Finish(); // the presenter may still read the old one
        }
        hudless = VideoCore::UniqueImage(instance.GetDevice(), instance.GetAllocator());
        hudless.Create(vk::ImageCreateInfo{
            .flags = vk::ImageCreateFlagBits::eMutableFormat,
            .imageType = vk::ImageType::e2D,
            .format = format,
            .extent = {w, h, 1},
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = vk::SampleCountFlagBits::e1,
            .tiling = vk::ImageTiling::eOptimal,
            .usage = vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
            .initialLayout = vk::ImageLayout::eUndefined,
        });
        fg_inputs.hudless = vk::Image(hudless);
        fg_inputs.hudless_format = format;
        fg_inputs.hudless_w = w;
        fg_inputs.hudless_h = h;
    }
    scheduler.EndRendering();
    const vk::ImageLayout layout = image.backing->state.layout;
    runtime.Transit(&image, vk::ImageLayout::eTransferSrcOptimal,
                    vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferRead);
    runtime.FlushBarriers();
    const vk::Image src = image.GetImage(), dst = vk::Image(hudless);
    scheduler.Record([src, dst, w, h](vk::CommandBuffer cmdbuf) {
        vk::ImageMemoryBarrier2 barrier{
            .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask = vk::AccessFlagBits2::eShaderRead,
            .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
            .dstAccessMask = vk::AccessFlagBits2::eTransferWrite,
            .oldLayout = vk::ImageLayout::eUndefined,
            .newLayout = vk::ImageLayout::eTransferDstOptimal,
            .image = dst,
            .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
        };
        cmdbuf.pipelineBarrier2({.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &barrier});
        const vk::ImageCopy region{
            .srcSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
            .dstSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
            .extent = {w, h, 1},
        };
        cmdbuf.copyImage(src, vk::ImageLayout::eTransferSrcOptimal, dst,
                         vk::ImageLayout::eTransferDstOptimal, region);
        barrier.srcStageMask = vk::PipelineStageFlagBits2::eTransfer;
        barrier.srcAccessMask = vk::AccessFlagBits2::eTransferWrite;
        barrier.dstStageMask = vk::PipelineStageFlagBits2::eAllCommands;
        barrier.dstAccessMask = vk::AccessFlagBits2::eShaderRead;
        barrier.oldLayout = vk::ImageLayout::eTransferDstOptimal;
        barrier.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
        cmdbuf.pipelineBarrier2({.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &barrier});
    });
    // The UI draws next, in the layout it had.
    runtime.Transit(&image, layout, vk::PipelineStageFlagBits2::eAllCommands,
                    vk::AccessFlagBits2::eColorAttachmentWrite | vk::AccessFlagBits2::eColorAttachmentRead);
    runtime.FlushBarriers();
    fg_inputs.hudless_serial = fg_inputs.serial;
}

void NeuralRender::AfterDispatch() {
    post_pending = false;
    if (fsr4_broken.load()) {
        fsr4_failed = true;
    }
    if ((!fsr4_mode && !fsr3_mode) || fsr4_failed) {
        // Frame generation alone: its motion and depth.
        Prepare(2u);
        return;
    }
    // The TAAU's output: its storage image of the output size (the largest one it writes).
    VideoCore::Image* output = nullptr;
    for (const auto id : storage) {
        auto& image = texture_cache.GetImage(id);
        if (!output || image.info.size.width * image.info.size.height >
                           output->info.size.width * output->info.size.height) {
            output = &image;
        }
    }
    static bool described = false;
    if (!described) {
        described = true;
        std::printf("GPU: UE4 TAAU inputs:");
        for (const auto id : images) {
            const auto& image = texture_cache.GetImage(id);
            std::printf(" %s %ux%u", vk::to_string(image.info.pixel_format).c_str(),
                        image.info.size.width, image.info.size.height);
        }
        std::printf("; outputs:");
        for (const auto id : storage) {
            const auto& image = texture_cache.GetImage(id);
            std::printf(" %s %ux%u", vk::to_string(image.info.pixel_format).c_str(),
                        image.info.size.width, image.info.size.height);
        }
        std::printf("\n");
    }
    if (!output || output->info.pixel_format != vk::Format::eR16G16B16A16Sfloat) {
        static bool warned = false;
        if (!warned) {
            warned = true;
            std::printf("GPU: FSR skipped: the TAAU output is not RGBA16F\n");
        }
        return;
    }
    if (!Prepare(2u | (debug ? 1u : 0u))) {
        return;
    }
    if (debug) {
        return;
    }
    // Output size: the TAAU's target (UE4 sizes it to the output view).
    const u32 ow = output->info.size.width, oh = output->info.size.height;
    const auto device = instance.GetDevice();
    if (!fsr4_out || ow != out_w || oh != out_h) {
        if (fsr4_out) {
            scheduler.Finish();
        }
        fsr4_out_view.reset();
        fsr4_out = VideoCore::UniqueImage(device, instance.GetAllocator());
        fsr4_out.Create(vk::ImageCreateInfo{
            .imageType = vk::ImageType::e2D,
            .format = vk::Format::eR16G16B16A16Sfloat,
            .extent = {ow, oh, 1},
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = vk::SampleCountFlagBits::e1,
            .tiling = vk::ImageTiling::eOptimal,
            .usage = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled |
                     vk::ImageUsageFlagBits::eTransferSrc,
            .initialLayout = vk::ImageLayout::eUndefined,
        });
        fsr4_out_view = Check(device.createImageViewUnique({
            .image = vk::Image(fsr4_out),
            .viewType = vk::ImageViewType::e2D,
            .format = vk::Format::eR16G16B16A16Sfloat,
            .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
        }));
        const vk::Image image = vk::Image(fsr4_out);
        scheduler.Record([image](vk::CommandBuffer cmdbuf) {
            const vk::ImageMemoryBarrier2 to_general{
                .srcStageMask = vk::PipelineStageFlagBits2::eNone,
                .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
                .dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
                .oldLayout = vk::ImageLayout::eUndefined,
                .newLayout = vk::ImageLayout::eGeneral,
                .image = image,
                .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
            };
            cmdbuf.pipelineBarrier2({.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &to_general});
        });
        out_w = ow;
        out_h = oh;
        fsr4_reset = true;
    }
    if (fsr3_mode) {
        RecordFsr3(*output, ow, oh);
        return;
    }
    if (!fsr4) {
        // 4.1.2's RDNA 4 kernels (cooperative matrices) when the GPU and the assets have them;
        // GOW3_FSR4_KERNELS=411 keeps 4.1.1's where the GPU can run those (a driver without
        // VK_VALVE_shader_mixed_float_dot_product, RADV 25.3, crashes creating them).
        const char* dir411 = std::getenv("GOW3_FSR411_DIR");
        const char* dir412 = std::getenv("GOW3_FSR412_DIR");
        const char* kernels = std::getenv("GOW3_FSR4_KERNELS");
        const std::string set412 = dir412 && dir412[0] ? dir412 : "fsr4_412";
        const bool use412 = instance.IsFsr412Supported() &&
                            !(kernels && std::string_view{kernels} == "411" &&
                              instance.IsFsr411Supported()) &&
                            std::filesystem::is_directory(set412);
        if (!use412 && !instance.IsFsr411Supported()) {
            std::printf("GPU: FSR 4 off: the GPU runs only the 4.1.2 kernels, and %s is missing\n",
                        set412.c_str());
            fsr4_failed = true;
            return;
        }
        const std::string dir = use412 ? set412 : dir411 && dir411[0] ? dir411 : "fsr4_411";
        std::printf("GPU: FSR 4 kernels: %s (%s)\n", use412 ? "4.1.2 cooperative matrix" : "4.1.1 INT8",
                    dir.c_str());
        fsr4 = std::make_unique<Fsr411::Upscaler>(instance.GetPhysicalDevice(), device, dir);
    }
    // Its constant ring holds kFramesInFlight frames: the oldest must be done.
    while (fsr4_ticks.size() >= Fsr411::kFramesInFlight) {
        scheduler.Wait(fsr4_ticks.front());
        fsr4_ticks.erase(fsr4_ticks.begin());
    }
    if (view_w != last_view_w || view_h != last_view_h) {
        last_view_w = view_w;
        last_view_h = view_h;
        fsr4_reset = true;
    }
    const auto& settings = Gow3Settings::Get();
    static const float sharpness = [] {
        const char* env = std::getenv("GOW3_UE4_FSR4_SHARPNESS");
        return env && env[0] ? std::clamp(float(std::atof(env)), 0.0f, 1.0f) : 0.0f;
    }();
    const float sign = settings.fsr4_invert_jitter ? -1.0f : 1.0f;
    Fsr411::Frame frame;
    frame.color = {vk::Image(color), *color_view, width, height};
    frame.depth = {vk::Image(depth), *depth_view, width, height};
    frame.motion = {vk::Image(motion), *motion_view, width, height};
    frame.output = {vk::Image(fsr4_out), *fsr4_out_view, ow, oh};
    frame.render_width = view_w;
    frame.render_height = view_h;
    frame.jitter[0] = sign * jitter[0];
    frame.jitter[1] = sign * jitter[1];
    frame.sharpness = sharpness;
    frame.sharpen = sharpness > 0.0f;
    frame.reset = std::exchange(fsr4_reset, false);
    frame.auto_exposure = settings.fsr4_auto_exposure;
    const vk::Image out_image = output->GetImage(), own_out = vk::Image(fsr4_out);
    Fsr411::Upscaler* const upscaler = fsr4.get();
    std::atomic<bool>* const broken = &fsr4_broken;
    // Recorded on the recording thread, which alone uses the upscaler.
    scheduler.Record([frame, upscaler, out_image, own_out, ow, oh, broken](vk::CommandBuffer cmdbuf) mutable {
        frame.cmdbuf = cmdbuf;
        if (!upscaler->Record(frame)) {
            // The TAAU's own result stays; FSR 4 is off for the session.
            std::printf("GPU: FSR 4 unavailable: %s\n", upscaler->Error().c_str());
            broken->store(true);
            return;
        }
        const vk::MemoryBarrier2 memory{
            .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
        };
        cmdbuf.pipelineBarrier2({.memoryBarrierCount = 1, .pMemoryBarriers = &memory});
        const vk::ImageCopy region{
            .srcSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
            .dstSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
            .extent = {ow, oh, 1},
        };
        cmdbuf.copyImage(own_out, vk::ImageLayout::eGeneral, out_image, vk::ImageLayout::eGeneral,
                         region);
        cmdbuf.pipelineBarrier2({.memoryBarrierCount = 1, .pMemoryBarriers = &memory});
    });
    fsr4_ticks.push_back(scheduler.CurrentTick());
    static bool logged = false;
    if (!logged) {
        logged = true;
        std::printf("GPU: FSR 4.1.1 on the UE4 TAAU: %ux%u -> %ux%u\n", view_w, view_h, ow, oh);
    }
}

void NeuralRender::RecordFsr3(VideoCore::Image& output, u32 ow, u32 oh) {
    if (!fsr3) {
        fsr3 = std::make_unique<Fsr3Upscaler>(instance);
    }
    // New input or output images: the context goes, after the GPU is done with it.
    if (fsr3_w != width || fsr3_h != height || fsr3_w == 0) {
        if (fsr3_w != 0) {
            scheduler.Finish();
        }
        fsr3_w = width;
        fsr3_h = height;
    }
    while (!fsr3_in_flight.empty() && scheduler.IsFree(fsr3_in_flight.front().second)) {
        fsr3_completed = fsr3_in_flight.front().first;
        fsr3_in_flight.pop_front();
    }
    if (view_w != last_view_w || view_h != last_view_h) {
        last_view_w = view_w;
        last_view_h = view_h;
        fsr4_reset = true;
    }
    const auto now = std::chrono::steady_clock::now();
    const float frame_ms =
        fsr3_last.time_since_epoch().count() == 0
            ? 16.7f
            : std::clamp(std::chrono::duration<float, std::milli>(now - fsr3_last).count(), 1.0f,
                         100.0f);
    fsr3_last = now;
    const auto& settings = Gow3Settings::Get();
    static const float sharpness = [] {
        const char* env = std::getenv("GOW3_UE4_FSR4_SHARPNESS");
        return env && env[0] ? std::clamp(float(std::atof(env)), 0.0f, 1.0f) : 0.0f;
    }();
    const float sign = settings.fsr4_invert_jitter ? -1.0f : 1.0f;
    Fsr3Upscaler::Frame frame;
    frame.color = vk::Image(color);
    frame.depth = vk::Image(depth);
    frame.motion = vk::Image(motion);
    frame.output = vk::Image(fsr4_out);
    frame.width = width;
    frame.height = height;
    frame.out_width = ow;
    frame.out_height = oh;
    frame.render_width = view_w;
    frame.render_height = view_h;
    frame.jitter = {sign * jitter[0], sign * jitter[1]};
    frame.near_plane = fg_inputs.near_plane;
    frame.fov_y = fg_inputs.fov_y;
    frame.frame_ms = frame_ms;
    frame.sharpness = sharpness;
    frame.reset = std::exchange(fsr4_reset, false);
    frame.auto_exposure = settings.fsr4_auto_exposure;
    frame.frame_id = ++fsr3_frame_id;
    frame.completed_id = fsr3_completed;
    const vk::Image out_image = output.GetImage(), own_out = vk::Image(fsr4_out);
    Fsr3Upscaler* const upscaler = fsr3.get();
    std::atomic<bool>* const broken = &fsr4_broken;
    scheduler.Record([frame, upscaler, out_image, own_out, ow, oh, broken](vk::CommandBuffer cmdbuf) mutable {
        frame.cmdbuf = cmdbuf;
        if (!upscaler->Record(frame)) {
            std::printf("GPU: FSR 3.1 unavailable: %s\n", upscaler->Error().c_str());
            broken->store(true);
            return;
        }
        const vk::MemoryBarrier2 memory{
            .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
        };
        cmdbuf.pipelineBarrier2({.memoryBarrierCount = 1, .pMemoryBarriers = &memory});
        const vk::ImageCopy region{
            .srcSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
            .dstSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
            .extent = {ow, oh, 1},
        };
        cmdbuf.copyImage(own_out, vk::ImageLayout::eGeneral, out_image, vk::ImageLayout::eGeneral,
                         region);
        cmdbuf.pipelineBarrier2({.memoryBarrierCount = 1, .pMemoryBarriers = &memory});
    });
    fsr3_in_flight.emplace_back(fsr3_frame_id, scheduler.CurrentTick());
    static bool logged = false;
    if (!logged) {
        logged = true;
        std::printf("GPU: FSR 3.1.5 on the UE4 TAAU: %ux%u -> %ux%u\n", view_w, view_h, ow, oh);
    }
}

} // namespace Vulkan
