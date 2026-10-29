// SPDX-License-Identifier: GPL-2.0-or-later
// gow3: FSR 3.1.5 upscaling (third_party/fsr-vulkan) on the UE4 TAAU, GOW3_UE4_FSR3=1: the
// analytic upscaler for GPUs FSR 4 cannot run on (no INT8 dot products: RDNA 1, older ones,
// other vendors) and a lighter one for the Steam Deck. Same inputs as FSR 4 (vk_neural_render's
// Prepare()): RGBA16F color, R32F depth (1 - z), RG16F motion in render pixels, all General.

#pragma once

#include <array>
#include <string>

#include "common/types.h"
#include "video_core/renderer_vulkan/vk_common.h"
#include "video_core/texture_cache/image.h"

struct FfxVkFsr3_3_1_5UpscalerContext;

namespace Vulkan {

class Instance;

class Fsr3Upscaler {
public:
    struct Frame {
        vk::CommandBuffer cmdbuf;
        vk::Image color, depth, motion, output;
        u32 width = 0, height = 0;        ///< size of the input images
        u32 out_width = 0, out_height = 0;
        u32 render_width = 0, render_height = 0; ///< the view inside the inputs
        std::array<float, 2> jitter{};     ///< render pixels
        float near_plane = 10.0f;
        float fov_y = 1.0f;
        float frame_ms = 16.7f;
        float sharpness = 0.0f;
        bool reset = false;
        bool auto_exposure = true;
        u64 frame_id = 0;
        u64 completed_id = 0; ///< every frame up to it has finished on the GPU
    };

    explicit Fsr3Upscaler(const Instance& instance);
    ~Fsr3Upscaler();

    /// Recording thread only. False: FSR 3.1 failed (Error() says why), nothing recorded.
    bool Record(const Frame& frame);
    /// Recording thread only; the GPU must be idle.
    void Destroy();

    [[nodiscard]] const std::string& Error() const noexcept {
        return error;
    }

private:
    bool Create(const Frame& frame);

    const Instance& instance;
    FfxVkFsr3_3_1_5UpscalerContext* context = nullptr;
    VideoCore::UniqueImage dilated_depth, dilated_motion, prev_depth;
    std::array<void*, 7> imported{}; ///< bridge tokens: color, depth, motion, output, shared x3
    vk::Image images[4]{};
    u32 width = 0, height = 0, out_width = 0, out_height = 0;
    bool auto_exposure = true;
    std::string error;
};

} // namespace Vulkan
