// SPDX-License-Identifier: GPL-2.0-or-later
// gow3 U2 (plan-ue4-motion-vectors.md): UE4's motion vectors for the temporal upscalers. At the
// first TAA pass of a frame (draws or the TAAU dispatch) a compute pass turns the game's
// velocity target, where it wrote one, and otherwise the camera's motion (scene depth and the
// View buffer's ClipToPrevClip) into render pixels, previous minus current (ue4_motion.comp).
// Nothing in the game's frame changes. GOW3_UE4_MOTION_VIEW=1-4 shows them on the presented
// frame (ue4_motion_view.comp); F10 cycles 1-4 and off. Off by default.

#pragma once

#include <array>
#include <atomic>
#include <span>

#include "common/types.h"
#include "video_core/renderer_vulkan/vk_common.h"
#include "video_core/texture_cache/image.h"

namespace VideoCore {
class TextureCache;
}

namespace Vulkan {

class Instance;
class Scheduler;
class Runtime;

class Ue4Motion {
public:
    /// GOW3_UE4_MOTION_VIEW set (read once): the pass runs, the view starts in its mode.
    static bool Enabled();
    /// F10: the next view mode (1-4, then 0 = picture unchanged).
    static void CycleViewMode();

    Ue4Motion(const Instance& instance, Scheduler& scheduler,
              VideoCore::TextureCache& texture_cache, Runtime& runtime);
    ~Ue4Motion();

    /// A draw into the velocity target (one R16G16_UNORM target at `address`, with a depth buffer).
    void OnVelocityDraw(VideoCore::ImageId velocity, VAddr address, VideoCore::ImageId depth);
    /// A G-buffer draw (3+ colour targets): its depth buffer is the scene depth, for frames
    /// without a velocity pass (menus).
    void OnSceneDraw(VideoCore::ImageId depth);
    /// A TAA pass with the View buffer bound (`view`, ViewBytes, read now) sampling `reads`. The
    /// first one of a frame records the motion vector pass. `depth`: the pass's depth buffer
    /// (draws), if any.
    void OnTaa(const u8* view, std::span<const VAddr> reads, VideoCore::ImageId depth);
    /// The motion vector pass of View frame `frame_number` (U3), if it ran: its image (General
    /// layout, written by a compute pass recorded before) and the scene depth it read (General).
    struct LastPass {
        vk::Image image;
        vk::ImageView view;
        u32 width, height;
        VideoCore::Image* depth;
    };
    bool GetPass(u32 frame_number, LastPass& pass);
    /// The velocity target's guest address (its last draw), or 0.
    [[nodiscard]] VAddr VelocityAddress() const noexcept {
        return velocity_address;
    }
    /// Presenter, at a flip, on the draw scheduler's command buffer: after this frame's motion
    /// vectors, the debug view of `display` (in ShaderReadOnlyOptimal for compute) into an image
    /// of the port; its view in `present_format`, to present instead. Null: present as before.
    vk::ImageView Compose(vk::CommandBuffer cmdbuf, VideoCore::Image& display,
                          vk::Format present_format);

private:
    struct Target {
        VideoCore::ImageId id{};
        u64 uid = 0; ///< the image's, so a freed and reused slot is not taken for it
    };
    struct OwnImage {
        VideoCore::UniqueImage image;
        vk::UniqueImageView view;    ///< storage / sampled
        vk::UniqueImageView present; ///< Compose's output only
        u32 width = 0, height = 0;
        vk::Format format = vk::Format::eUndefined;
        vk::ImageLayout layout = vk::ImageLayout::eUndefined;
    };

    Target Remember(VideoCore::ImageId id);
    VideoCore::Image* Find(const Target& target);
    void Ensure(OwnImage& own, u32 width, u32 height, vk::Format format,
                vk::Format present_format = vk::Format::eUndefined);

    const Instance& instance;
    Scheduler& scheduler;
    VideoCore::TextureCache& texture_cache;
    Runtime& runtime;
    vk::UniqueDescriptorSetLayout motion_desc_layout;
    vk::UniquePipelineLayout motion_layout;
    vk::UniquePipeline motion_pipeline;
    vk::UniqueDescriptorSetLayout view_desc_layout;
    vk::UniquePipelineLayout view_layout;
    vk::UniquePipeline view_pipeline;

    Target velocity, depth, scene_depth;
    Target pass_depth; ///< the depth the last pass read
    VAddr velocity_address = 0;
    bool recorded = false;
    u32 recorded_frame = 0;
    bool fresh = false;                ///< motion vectors of a frame not yet presented
    std::array<float, 2> view_size{};  ///< of the last motion vector pass
    OwnImage motion;
    OwnImage output;
    std::array<OwnImage, 2> history;
    u32 history_index = 0;
    bool history_valid = false;
    u64 passes = 0;
    static std::atomic<int> view_mode;
};

} // namespace Vulkan
