// SPDX-License-Identifier: GPL-2.0-or-later
// gow3: FSR 3.1.6 frame interpolation (third_party/fsr-vulkan), GOW3_FRAMEGEN=1. Each game frame
// with a UE4 TAAU gets a generated frame halfway to the previous one, presented before it; the
// motion and depth are the ones vk_neural_render prepares for FSR 4. The frames interpolated are
// the ones before the UI (vk_neural_render's HUD-less copy); the game frame's UI is drawn over
// the result (fg_hud_compose.comp). Gow3Settings::fg_enabled turns it off and on live.

#pragma once

#include <atomic>
#include <chrono>
#include <deque>
#include <utility>

#include "common/types.h"
#include "video_core/renderer_vulkan/vk_common.h"
#include "video_core/renderer_vulkan/vk_neural_render.h"
#include "video_core/texture_cache/image.h"

struct FfxVkFsr3_3_1_6FrameGenerationContext;

namespace Vulkan {

class Instance;
class Scheduler;

class FrameGen {
public:
    explicit FrameGen(const Instance& instance);
    ~FrameGen();

    [[nodiscard]] bool Enabled() const noexcept {
        return enabled;
    }

    /// Records on `scheduler`'s command buffer: the frame between the previous game frame and
    /// `frame` (General layout, `format` with 32 bits per pixel) into `out` (same format,
    /// TransferDst-capable). `hudless`: the same frame without the UI (General, or null).
    /// False when none was generated (no TAAU this frame, HDR, turned off, failure).
    bool Record(Scheduler& scheduler, vk::Image frame, vk::Image hudless, vk::Image out,
                vk::Format format, u32 width, u32 height, const NeuralRender::FrameGenInputs& inputs);

    /// The swap thread, after each present: how long it waited for a swapchain image. Waits
    /// mean presents are paced by the compositor; their interval is the refresh frame
    /// generation gets, which can be below the display's (GNOME gave 60 presents a second on
    /// a 120 Hz display).
    void NotePresent(float acquire_ms);

private:
    /// The refresh interval presents get: the display's, or longer when presents are paced.
    float RefreshMs() const;

    bool EnsureContext(u32 width, u32 height, u32 render_w, u32 render_h);
    void Destroy();

    const Instance& instance;
    bool enabled = false;
    bool failed = false;
    FfxVkFsr3_3_1_6FrameGenerationContext* context = nullptr;
    u32 width = 0, height = 0, max_render_w = 0, max_render_h = 0;
    /// RGBA8 copies of the interpolated frame (HUD-less when there is one), of the generated one
    /// and of the game frame with its UI (the frame's own format is BGRA).
    VideoCore::UniqueImage color, output, real;
    vk::UniqueImageView color_view, output_view, real_view;
    vk::UniqueDescriptorSetLayout compose_layout;
    vk::UniquePipelineLayout compose_pipeline_layout;
    vk::UniquePipeline compose_pipeline;
    u64 frame_id = 0;
    u64 last_serial = 0;
    std::deque<std::pair<u64, u64>> in_flight; ///< (frame id, scheduler tick)
    std::chrono::steady_clock::time_point last_time{};
    float smoothed_ms = 0.0f;
    bool fast_game = false;
    // NotePresent (swap thread): the present-paced interval and when it was last seen.
    std::chrono::steady_clock::time_point last_present{};
    float paced_ms = 0.0f;
    std::atomic<float> paced_interval_ms{0.0f};
    std::atomic<std::chrono::steady_clock::rep> paced_at{0};
};

} // namespace Vulkan
