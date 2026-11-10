// SPDX-License-Identifier: GPL-2.0-or-later
// gow3 U3: DLAA in place of UE4's TAA (vk_ue4_dlss.h).

#include "video_core/renderer_vulkan/vk_ue4_dlss.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>

#include "gow3_settings.h"
#include "video_core/renderer_vulkan/vk_dlss.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_runtime.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_ue4_motion.h"
#include "video_core/texture_cache/texture_cache.h"

namespace Vulkan {

namespace {

// View uniform buffer offsets (study-taau-frame-map.md section 3).
constexpr u32 OffsetClipToPrevClip = 1824;
constexpr u32 OffsetJitter = 1888;
constexpr u32 OffsetViewSize = 1952;
constexpr u32 OffsetNearPlane = 2112;
constexpr u32 OffsetFrameNumber = 2140;

template <typename T>
T Load(const u8* data, u32 offset) {
    T value;
    std::memcpy(&value, data + offset, sizeof(T));
    return value;
}

/// The largest one-frame move, in pixels, of five screen points (the centre and four near the
/// corners) at infinity and at 10 m, through View.ClipToPrevClip (row vector times matrix).
/// Logged with each reset only: no reset rule uses it.
float CameraJumpPixels(const u8* view, std::array<float, 2> size) {
    std::array<std::array<float, 4>, 4> m;
    std::memcpy(m.data(), view + OffsetClipToPrevClip, sizeof(m));
    // Reversed Z, infinite far plane: device Z = near / view distance (UE units, cm).
    const float near_plane = Load<float>(view, OffsetNearPlane);
    const std::array<float, 2> depths{0.0f, near_plane > 0.0f ? near_plane / 1000.0f : 0.0f};
    static constexpr std::array<std::array<float, 2>, 5> points{
        {{0.0f, 0.0f}, {-0.8f, -0.8f}, {0.8f, -0.8f}, {-0.8f, 0.8f}, {0.8f, 0.8f}}};
    float worst = 0.0f;
    for (const float z : depths) {
        for (const auto& p : points) {
            std::array<float, 4> prev;
            for (u32 c = 0; c < 4; ++c) {
                prev[c] = p[0] * m[0][c] + p[1] * m[1][c] + z * m[2][c] + m[3][c];
            }
            if (!(std::fabs(prev[3]) > 1e-6f)) {
                return 1e30f;
            }
            const float dx = (p[0] - prev[0] / prev[3]) * size[0] * 0.5f;
            const float dy = (p[1] - prev[1] / prev[3]) * size[1] * 0.5f;
            worst = std::max(worst, std::sqrt(dx * dx + dy * dy));
        }
    }
    return std::isfinite(worst) ? worst : 1e30f;
}

} // namespace

std::atomic<bool> Ue4Dlss::evaluate_failed{false};
std::atomic<s64> Ue4Dlss::last_run_ns{0};

bool Ue4Dlss::Running() {
    const s64 now = std::chrono::steady_clock::now().time_since_epoch().count();
    const s64 last = last_run_ns.load(std::memory_order_relaxed);
    return last != 0 && now - last < 100'000'000;
}

bool Ue4Dlss::Enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("GOW3_UE4_DLSS");
        return (value && value[0] == '1') || Dlss::Get() != nullptr;
    }();
    return enabled;
}

bool Ue4Dlss::On() {
    return Enabled() && Gow3Settings::Get().ue4_dlss.load(std::memory_order_relaxed);
}

namespace {
constexpr const char* JitterSignNames[] = {"DLSS jitter +x +y", "DLSS jitter -x +y",
                                           "DLSS jitter +x -y", "DLSS jitter -x -y"};

s64 SteadyNs() {
    return std::chrono::steady_clock::now().time_since_epoch().count();
}
} // namespace

std::atomic<int> Ue4Dlss::jitter_sign{0};
std::atomic<bool> Ue4Dlss::sign_changed{false};
std::atomic<s64> Ue4Dlss::notice_until_ns{0};
std::atomic<const char*> Ue4Dlss::notice{nullptr};

void Ue4Dlss::Toggle() {
    if (!Enabled()) {
        return;
    }
    auto& setting = Gow3Settings::Get().ue4_dlss;
    const bool now = !setting.load(std::memory_order_relaxed);
    setting.store(now, std::memory_order_relaxed);
    std::printf("UE4 DLSS: %s (F9)\n", now ? "DLSS on" : "UE4 TAA");
    std::fflush(stdout);
    notice.store(now ? "DLSS on" : "UE4 TAA", std::memory_order_relaxed);
    notice_until_ns.store(SteadyNs() + 3'000'000'000LL, std::memory_order_relaxed);
}

void Ue4Dlss::CycleJitterSign() {
    if (!Enabled()) {
        return;
    }
    const int sign = (jitter_sign.load(std::memory_order_relaxed) + 1) % 4;
    jitter_sign.store(sign, std::memory_order_relaxed);
    sign_changed.store(true, std::memory_order_relaxed);
    std::printf("UE4 DLSS: %s (F8)\n", JitterSignNames[sign]);
    std::fflush(stdout);
    notice.store(JitterSignNames[sign], std::memory_order_relaxed);
    notice_until_ns.store(SteadyNs() + 3'000'000'000LL, std::memory_order_relaxed);
}

const char* Ue4Dlss::Notice() {
    return SteadyNs() < notice_until_ns.load(std::memory_order_relaxed)
               ? notice.load(std::memory_order_relaxed)
               : nullptr;
}

Ue4Dlss::Ue4Dlss(const Instance& instance_, Scheduler& scheduler_,
                 VideoCore::TextureCache& texture_cache_, Runtime& runtime_, Ue4Motion& motion_)
    : instance{instance_}, scheduler{scheduler_}, texture_cache{texture_cache_}, runtime{runtime_},
      motion{motion_} {
    const Dlss* dlss = Dlss::Get();
    const bool available = dlss && dlss->Available();
    Gow3Settings::Get().ue4_dlss_available.store(available, std::memory_order_relaxed);
    std::printf("UE4 DLSS: %s, %s (menu Image tab or F9)\n",
                available ? "available" : "not available (DLLs or GPU)",
                Gow3Settings::Get().ue4_dlss.load() ? "on" : "off");
}

Ue4Dlss::~Ue4Dlss() = default;

bool Ue4Dlss::Fail(const char* why) {
    static const char* last = nullptr;
    if (why != last) {
        last = why;
        std::printf("UE4 DLSS: UE4 TAA this frame (%s)\n", why);
    }
    was_on = false;
    return false;
}

void Ue4Dlss::OnColorTarget(VideoCore::ImageId id, VAddr address) {
    if (id != color.id || address != color.address) {
        auto* image = texture_cache.TryGetImage(id, 0);
        color = image ? Target{id, image->image_uid, address} : Target{};
    }
}

bool Ue4Dlss::OnTaaDraw(const u8* view, std::span<const VAddr> reads, VideoCore::ImageId output_id,
                        VAddr output_address, const std::function<void()>& mark_written) {
    return Run(view, reads, output_id, output_address, 0, 0, mark_written);
}

bool Ue4Dlss::OnTaauDispatch(const u8* view, std::span<const VAddr> reads,
                             VideoCore::ImageId output_id, VAddr output_address, u32 output_width,
                             u32 output_height, const std::function<void()>& mark_written) {
    if (output_width == 0 || output_height == 0) {
        return Fail("TAAU output size unknown");
    }
    return Run(view, reads, output_id, output_address, output_width, output_height, mark_written);
}

bool Ue4Dlss::Run(const u8* view, std::span<const VAddr> reads, VideoCore::ImageId output_id,
                  VAddr output_address, u32 output_width, u32 output_height,
                  const std::function<void()>& mark_written) {
    const bool upscale = output_width != 0;
    if (!view) {
        return Fail("no View buffer");
    }
    const u32 frame_number = Load<u32>(view, OffsetFrameNumber);
    if (!Gow3Settings::Get().ue4_dlss.load(std::memory_order_relaxed)) {
        was_on = false;
        return false;
    }
    if (was_on && have_frame && frame_number == last_frame) {
        return true; // the second TAA draw: DLSS already wrote the output
    }
    if (evaluate_failed.exchange(false, std::memory_order_relaxed)) {
        std::printf("UE4 DLSS: evaluation failed; UE4 TAA from now on\n");
        Gow3Settings::Get().ue4_dlss.store(false, std::memory_order_relaxed);
        return Fail("DLSS failed");
    }
    Dlss* dlss = Dlss::Get();
    if (!dlss || !dlss->Available()) {
        return Fail(dlss ? "DLSS not available" : "DLSS DLLs not found");
    }
    Ue4Motion::LastPass pass;
    if (!motion.GetPass(frame_number, pass)) {
        return Fail("no motion vectors");
    }
    auto* color_image = color.id ? texture_cache.TryGetImage(color.id, color.uid) : nullptr;
    if (!color_image || color.address == output_address ||
        std::ranges::find(reads, color.address) == reads.end() ||
        color_image->info.pixel_format != vk::Format::eR16G16B16A16Sfloat) {
        return Fail("TAA's colour input not found");
    }
    auto* output_image = output_id ? texture_cache.TryGetImage(output_id, 0) : nullptr;
    if (!output_image) {
        return Fail("TAA's output not found");
    }
    // Native: TAA's RGBA16F colour target, copied into. TAAU: the dispatch's storage image; an
    // other format is blitted into (converted).
    const vk::Format output_format = output_image->info.pixel_format;
    const bool same_format = output_format == vk::Format::eR16G16B16A16Sfloat;
    if (!upscale && !same_format) {
        return Fail("TAA's output not RGBA16F");
    }
    static vk::Format reported_format = vk::Format::eUndefined;
    if (upscale && output_format != reported_format) {
        reported_format = output_format;
        std::printf("UE4 DLSS: TAAU output %s %ux%u\n", vk::to_string(output_format).c_str(),
                    output_image->info.size.width, output_image->info.size.height);
    }
    auto* depth_image = pass.depth;
    // The rendered rectangle, from (0, 0); every input holds it.
    const std::array<float, 2> view_size{Load<float>(view, OffsetViewSize),
                                         Load<float>(view, OffsetViewSize + 4)};
    u32 width = std::min({u32(view_size[0]), color_image->info.size.width,
                          depth_image->info.size.width, pass.width});
    u32 height = std::min({u32(view_size[1]), color_image->info.size.height,
                           depth_image->info.size.height, pass.height});
    // DLSS's output rectangle, from (0, 0): native, the input's; TAAU, the dispatch's T# size.
    u32 out_width = upscale ? std::min(output_width, output_image->info.size.width)
                            : std::min(width, output_image->info.size.width);
    u32 out_height = upscale ? std::min(output_height, output_image->info.size.height)
                             : std::min(height, output_image->info.size.height);
    if (!upscale) {
        width = out_width;
        height = out_height;
    }
    if (width < 64 || height < 64 || out_width < width || out_height < height) {
        return Fail("view too small or output smaller than the input");
    }

    // Reset: the first frame after UE4's TAA ran, or a View frame number that is not the next
    // one. The game's camera cuts, skips and level loads all move it by more than one; pans, however
    // fast, never do (run cut-evidence). View.CameraCut stays 0 in this game; camera-motion rules
    // misfired (u3-dlaa: a ClipToPrevClip element above 1.0 reset ~6% of moving frames).
    const float camera_jump = CameraJumpPixels(view, view_size);
    const char* reset_reason = !was_on                          ? "first frame"
                               : frame_number != last_frame + 1 ? "frame number"
                                                                : nullptr;
    if (sign_changed.exchange(false, std::memory_order_relaxed)) {
        reset_reason = "jitter sign (F8)";
    }

    const auto device = instance.GetDevice();
    if (!output.image || own_width != out_width || own_height != out_height) {
        if (output.image) {
            struct Old {
                VideoCore::UniqueImage image;
                vk::UniqueImageView view;
            };
            auto old = std::make_shared<Old>(Old{std::move(output), std::move(output_view)});
            scheduler.DeferOperation([old] {});
        }
        output = VideoCore::UniqueImage(device, instance.GetAllocator());
        output.Create({
            .imageType = vk::ImageType::e2D,
            .format = vk::Format::eR16G16B16A16Sfloat,
            .extent = {out_width, out_height, 1},
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = vk::SampleCountFlagBits::e1,
            .tiling = vk::ImageTiling::eOptimal,
            .usage = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled |
                     vk::ImageUsageFlagBits::eTransferSrc,
        });
        output_view = Check(device.createImageViewUnique({
            .image = vk::Image(output),
            .viewType = vk::ImageViewType::e2D,
            .format = vk::Format::eR16G16B16A16Sfloat,
            .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
        }));
        own_width = out_width;
        own_height = out_height;
    }

    scheduler.EndRendering();
    // As the skipped draw's colour target: not Undefined, GPU-modified, its aliases told. Without
    // it a TAA output made at a level load stayed "new" and the tonemap's read refreshed it from
    // guest memory (an old frame) over DLSS's output: the picture snapped back again and again in
    // pans until UE4's TAA had drawn into it once (run u3-dlaa-2).
    mark_written();
    runtime.Transit(color_image, vk::ImageLayout::eGeneral, vk::PipelineStageFlagBits2::eAllCommands,
                    vk::AccessFlagBits2::eShaderRead);
    runtime.Transit(depth_image, vk::ImageLayout::eGeneral, vk::PipelineStageFlagBits2::eAllCommands,
                    vk::AccessFlagBits2::eShaderRead);
    runtime.Transit(output_image, vk::ImageLayout::eTransferDstOptimal,
                    vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite);
    runtime.FlushBarriers();
    const vk::ImageView color_view = Check(device.createImageView({
        .image = color_image->GetImage(),
        .viewType = vk::ImageViewType::e2D,
        .format = vk::Format::eR16G16B16A16Sfloat,
        .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
    }));
    const vk::ImageView depth_view = Check(device.createImageView({
        .image = depth_image->GetImage(),
        .viewType = vk::ImageViewType::e2D,
        .format = depth_image->info.pixel_format,
        .subresourceRange = {vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1},
    }));
    scheduler.DeferOperation([device, color_view, depth_view] {
        device.destroyImageView(color_view);
        device.destroyImageView(depth_view);
    });

    const vk::Image output_handle = vk::Image(output);
    const int quality = upscale ? Dlss::QualityForScale(float(out_width) / float(width)) : 0;
    const Dlss::FeatureDesc desc{width, height, out_width, out_height, quality, true, true};
    if (!dlss->HasFeature(desc)) {
        // Rare (the first frame, a size change): directly, once the recording thread has run
        // everything recorded (CommandBuffer) and the GPU is done with the old feature.
        const auto cmdbuf = scheduler.CommandBuffer();
        scheduler.WaitSubmitted();
        dlss->ReleaseFeature();
        const bool created = dlss->CreateFeature(cmdbuf, desc);
        std::printf("UE4 DLSS: feature %ux%u -> %ux%u, quality %d (%s)%s\n", width, height,
                    out_width, out_height, quality, upscale ? "TAAU" : "DLAA",
                    created ? "" : " failed");
        if (!created) {
            Gow3Settings::Get().ue4_dlss.store(false, std::memory_order_relaxed);
            return Fail("DLSS feature creation failed");
        }
        reset_reason = "feature created";
    }
    const auto now = std::chrono::steady_clock::now();
    float frame_ms = std::chrono::duration<float, std::milli>(now - last_time).count();
    if (frame_ms <= 0.0f || frame_ms > 200.0f) {
        frame_ms = 16.6f;
    }
    last_time = now;
    // UE4's TemporalAAJitter.xy is the projection's NDC offset: pixels = x * W / 2, -y * H / 2
    // (y down). F8 flips the signs to find the one NGX wants (sign 0: as computed).
    const int sign = jitter_sign.load(std::memory_order_relaxed);
    const float jitter_x =
        (sign & 1 ? -1.0f : 1.0f) * Load<float>(view, OffsetJitter) * view_size[0] * 0.5f;
    const float jitter_y =
        (sign & 2 ? 1.0f : -1.0f) * Load<float>(view, OffsetJitter + 4) * view_size[1] * 0.5f;
    const Dlss::Frame frame{
        .color = {color_image->GetImage(), color_view, vk::Format::eR16G16B16A16Sfloat,
                  vk::ImageAspectFlagBits::eColor, width, height},
        .depth = {depth_image->GetImage(), depth_view, depth_image->info.pixel_format,
                  vk::ImageAspectFlagBits::eDepth, width, height},
        .motion = {pass.image, pass.view, vk::Format::eR16G16B16A16Sfloat,
                   vk::ImageAspectFlagBits::eColor, width, height},
        .output = {output_handle, *output_view, vk::Format::eR16G16B16A16Sfloat,
                   vk::ImageAspectFlagBits::eColor, out_width, out_height},
        .jitter_x = jitter_x,
        .jitter_y = jitter_y,
        .reset = reset_reason != nullptr,
        .frame_ms = frame_ms,
        .sharpness = 0.0f,
    };
    const vk::Image taa_output = output_image->GetImage();
    const vk::Image motion_image = pass.image;
    // In order with the frame's other commands, on the recording thread: no CommandBuffer() here,
    // which made the GPU thread wait for that thread to catch up once a frame (U3 runs 1-3; the
    // user saw a small saw-tooth in the frame times once). NGX only records into `cmdbuf`.
    scheduler.Record([dlss, frame, output_handle, taa_output, motion_image, out_width, out_height,
                      same_format](vk::CommandBuffer cmdbuf) {
        const std::array<vk::ImageMemoryBarrier2, 2> before{{
            {
                // U2's motion pass wrote the vectors.
                .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
                .srcAccessMask = vk::AccessFlagBits2::eShaderWrite,
                .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
                .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
                .oldLayout = vk::ImageLayout::eGeneral,
                .newLayout = vk::ImageLayout::eGeneral,
                .image = motion_image,
                .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
            },
            {
                // Last frame's copy read the output; its contents are not needed.
                .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
                .srcAccessMask = vk::AccessFlagBits2::eTransferRead,
                .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
                .dstAccessMask =
                    vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
                .oldLayout = vk::ImageLayout::eUndefined,
                .newLayout = vk::ImageLayout::eGeneral,
                .image = output_handle,
                .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
            },
        }};
        cmdbuf.pipelineBarrier2({.imageMemoryBarrierCount = u32(before.size()),
                                 .pImageMemoryBarriers = before.data()});
        if (!dlss->Evaluate(cmdbuf, frame)) {
            // TAA's output keeps last frame's contents once; the GPU thread switches to UE4's
            // TAA at the next TAA draw.
            evaluate_failed.store(true, std::memory_order_relaxed);
            return;
        }
        // Into TAA's colour target: bloom, the tonemap, the SSR-like pass and (with UE4's TAA
        // back) the next TAA read it there.
        const vk::ImageMemoryBarrier2 to_copy{
            .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask = vk::AccessFlagBits2::eShaderWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
            .dstAccessMask = vk::AccessFlagBits2::eTransferRead,
            .oldLayout = vk::ImageLayout::eGeneral,
            .newLayout = vk::ImageLayout::eTransferSrcOptimal,
            .image = output_handle,
            .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
        };
        cmdbuf.pipelineBarrier2({.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &to_copy});
        if (same_format) {
            const vk::ImageCopy region{
                .srcSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
                .dstSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
                .extent = {out_width, out_height, 1},
            };
            cmdbuf.copyImage(output_handle, vk::ImageLayout::eTransferSrcOptimal, taa_output,
                             vk::ImageLayout::eTransferDstOptimal, region);
        } else {
            const std::array<vk::Offset3D, 2> rect{
                vk::Offset3D{0, 0, 0}, vk::Offset3D{s32(out_width), s32(out_height), 1}};
            const vk::ImageBlit region{
                .srcSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
                .srcOffsets = rect,
                .dstSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
                .dstOffsets = rect,
            };
            cmdbuf.blitImage(output_handle, vk::ImageLayout::eTransferSrcOptimal, taa_output,
                             vk::ImageLayout::eTransferDstOptimal, region, vk::Filter::eNearest);
        }
    });

    if (reset_reason) {
        ++resets;
        if (resets <= 50) {
            std::printf("UE4 DLSS: reset (%s, View frame %u, camera moved %.0f px)\n",
                        reset_reason, frame_number, camera_jump);
        }
    }
    // GOW3_FPS_LOG=0 (the user's quiet runs): no periodic lines.
    static const bool quiet = [] {
        const char* value = std::getenv("GOW3_FPS_LOG");
        return value && std::strcmp(value, "0") == 0;
    }();
    if (++evaluations == 1 || (!quiet && evaluations % 3000 == 0)) {
        std::printf("UE4 DLSS: %llu frames, %llu resets, %ux%u -> %ux%u, jitter %.3f %.3f px\n",
                    static_cast<unsigned long long>(evaluations),
                    static_cast<unsigned long long>(resets), width, height, out_width, out_height,
                    jitter_x, jitter_y);
    }
    have_frame = true;
    last_frame = frame_number;
    was_on = true;
    last_run_ns.store(std::chrono::steady_clock::now().time_since_epoch().count(),
                      std::memory_order_relaxed);
    return true;
}

} // namespace Vulkan
