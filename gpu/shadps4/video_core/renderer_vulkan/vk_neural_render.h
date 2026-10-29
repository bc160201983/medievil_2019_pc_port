// SPDX-License-Identifier: GPL-2.0-or-later
// gow3: DLSSNR-AMD neural rendering (third_party/dlssnr-amd) on UE4's TAAU input, with the
// game's own motion vectors. GOW3_NR=1 turns it on (needs MediEvil's ue4_upscale TAAU);
// GOW3_NR_DEBUG=1 draws the motion vectors instead; GOW3_NR_ROOT overrides the shader/model root.
// GOW3_UE4_FSR4=1: FSR 4.1.1 (fsr411/, assets in GOW3_FSR411_DIR or fsr4_411) replaces the TAAU's
// result: it runs on the same inputs right after the TAAU and overwrites the TAAU's output.
// GOW3_UE4_FSR3=1 (or FSR 4 on a GPU that cannot run it): FSR 3.1.5 (vk_fsr3_upscale) the same way.

#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <deque>
#include <memory>
#include <utility>
#include <vector>

#include <boost/container/small_vector.hpp>

#include "common/types.h"
#include "video_core/renderer_vulkan/vk_common.h"
#include "video_core/texture_cache/image.h"

namespace nr {
class Runtime;
}
namespace Fsr411 {
class Upscaler;
}

namespace VideoCore {
class TextureCache;
}

namespace Vulkan {

class Instance;
class Fsr3Upscaler;
class Scheduler;
class Runtime;

class NeuralRender {
public:
    /// UE4's TAAU compute shader (MediEvil, r.TemporalAA.Upsampling=1).
    static constexpr u64 TaauHash = 0x9c89e98d32a43b49ull;
    /// The same pass above ~71% screen percentage (UE4 picks another permutation there).
    static constexpr u64 TaauHighHash = 0xb373f39908ae979full;
    static constexpr bool IsTaau(u64 hash) {
        return hash == TaauHash || hash == TaauHighHash;
    }
    /// UE4's View uniform buffer.
    static constexpr u32 ViewBufferSize = 3296;

    NeuralRender(const Instance& instance, Scheduler& scheduler,
                 VideoCore::TextureCache& texture_cache, Runtime& runtime);
    ~NeuralRender();

    [[nodiscard]] bool Enabled() const noexcept {
        return enabled;
    }

    /// Before binding a dispatch: collects its images and View buffer when it is the TAAU.
    void BeginDispatch(u64 pgm_hash) noexcept {
        capturing = enabled && (IsTaau(pgm_hash) || (log_hashes && pgm_hash));
        current_hash = pgm_hash;
        images.clear();
        storage.clear();
        view_valid = false;
        post_pending = false;
    }
    u64 current_hash{};
    static inline const bool log_hashes = std::getenv("GOW3_NR_HASHES") != nullptr;
    [[nodiscard]] bool Capturing() const noexcept {
        return capturing;
    }
    void OnSampled(VideoCore::ImageId id, bool is_storage = false) {
        if (capturing) {
            (is_storage ? storage : images).push_back(id);
        }
    }
    void OnBuffer(const void* data, u64 size);

    /// After binding the TAAU, before its dispatch: runs the network on its input colour.
    void Apply();
    /// FSR 4 mode: after the TAAU's dispatch is recorded, overwrites its output.
    [[nodiscard]] bool PostPending() const noexcept {
        return post_pending;
    }
    void AfterDispatch();

    /// gow3: frame generation's inputs, from the last TAAU (motion in view pixels, depth 1-z).
    struct FrameGenInputs {
        vk::Image motion, depth;
        u32 width = 0, height = 0;     ///< the images' size
        u32 view_w = 0, view_h = 0;    ///< the valid view at their origin
        std::array<float, 2> jitter{}; ///< view pixels
        float near_plane = 10.0f;
        float fov_y = 1.0f; ///< radians
        u64 serial = 0;     ///< counts the TAAUs prepared
        /// The display buffer after UE4's tonemapper, before the UI (ShaderReadOnlyOptimal).
        vk::Image hudless;
        vk::Format hudless_format = vk::Format::eUndefined;
        u32 hudless_w = 0, hudless_h = 0;
        u64 hudless_serial = 0; ///< the serial it was captured after
    };
    [[nodiscard]] bool FrameGenMode() const noexcept {
        return fg_mode;
    }
    [[nodiscard]] const FrameGenInputs& FrameGen() const noexcept {
        return fg_inputs;
    }
    /// Frame generation: the first full-screen draw into a display buffer after the TAAU is the
    /// tonemapper; the rasterizer calls CaptureHudless right after recording it.
    [[nodiscard]] bool HudlessWanted() const noexcept {
        return hudless_wanted;
    }
    void CaptureHudless(VideoCore::ImageId id);

private:
    void EnsureTargets(u32 width, u32 height);
    bool EnsureRuntime(u32 width, u32 height);
    /// Records ue4_motion.comp: own colour, motion and depth from the TAAU inputs (false: skipped).
    bool Prepare(u32 flags);
    void RecordFsr4();
    void RecordFsr3(VideoCore::Image& output, u32 ow, u32 oh);

    const Instance& instance;
    Scheduler& scheduler;
    VideoCore::TextureCache& texture_cache;
    Runtime& runtime;
    bool enabled = false;
    bool debug = false;
    bool fsr4_mode = false;
    bool fsr3_mode = false;
    bool nr_mode = false;
    bool fg_mode = false;
    FrameGenInputs fg_inputs;
    bool hudless_wanted = false;
    VideoCore::UniqueImage hudless;
    bool post_pending = false;
    bool capturing = false;
    bool view_valid = false;
    bool fresh_targets = false;
    bool reset_history = true;
    boost::container::small_vector<VideoCore::ImageId, 8> images, storage;
    std::array<float, ViewBufferSize / 4> view{};

    u32 width = 0, height = 0;
    VideoCore::UniqueImage color, motion, depth;
    vk::UniqueImageView motion_view, depth_view, color_view;
    vk::UniqueDescriptorSetLayout desc_layout;
    vk::UniquePipelineLayout pipeline_layout;
    vk::UniquePipeline pipeline;
    /// Built for the largest view seen: a smaller one (the game's dynamic resolution) runs in it
    /// with its edge repeated, so a size change rebuilds nothing.
    std::unique_ptr<nr::Runtime> network;

    // FSR 4 mode: the view prepared by Prepare(), the upscaler and its output.
    s32 rect_x = 0, rect_y = 0;
    u32 view_w = 0, view_h = 0;
    std::array<float, 2> jitter{};
    std::unique_ptr<Fsr411::Upscaler> fsr4;
    std::vector<u64> fsr4_ticks; ///< scheduler ticks of its recent frames (constant ring)
    VideoCore::UniqueImage fsr4_out;
    vk::UniqueImageView fsr4_out_view;
    u32 out_w = 0, out_h = 0, last_view_w = 0, last_view_h = 0;
    bool fsr4_failed = false;
    bool fsr4_reset = true;
    std::atomic<bool> fsr4_broken{false}; ///< set by the recording thread
    // FSR 3.1 mode (GOW3_UE4_FSR3=1, or FSR 4 unsupported): into fsr4_out as FSR 4.
    std::unique_ptr<Fsr3Upscaler> fsr3;
    std::deque<std::pair<u64, u64>> fsr3_in_flight; ///< (frame id, scheduler tick)
    u64 fsr3_frame_id = 0, fsr3_completed = 0;
    std::chrono::steady_clock::time_point fsr3_last{};
    u32 fsr3_w = 0, fsr3_h = 0;
    u64 frames = 0;
};

} // namespace Vulkan
