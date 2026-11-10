// SPDX-License-Identifier: GPL-2.0-or-later
// gow3 U3 / U4 (plan-ue4-motion-vectors.md, "U3 design", "U4 design"): DLSS in place of UE4's TAA.
// At render scale 100 (U3), at the first of the two native TAA draws, DLSS as DLAA; below it (U4)
// at the TAAU compute dispatch, DLSS upscaling (quality from the output / render ratio). Linear
// HDR with auto exposure, reversed depth, on TAA's colour input, the scene depth and U2's motion
// vectors with the game's jitter; the result goes into TAA's output and UE4's TAA is skipped.
// GOW3_UE4_DLSS=1, off by default; F9 switches between DLSS and UE4's TAA live. Needs
// gow3_dlss.dll and nvngx_dlss.dll (Dlss::Get).

#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <functional>
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
class Ue4Motion;

class Ue4Dlss {
public:
    /// The DLSS DLLs are beside the exe (Dlss::Get), or GOW3_UE4_DLSS=1: the hooks exist; whether
    /// DLSS runs is the ue4_dlss setting (menu, F9, ini).
    static bool Enabled();
    /// F9: the ue4_dlss setting on / off (UE4's TAA). The caller saves the settings.
    static void Toggle();
    /// F8: the next jitter sign handed to DLSS (+x +y, -x +y, +x -y, -x -y), shown on screen and
    /// logged; DLSS's history restarts. NVIDIA's own Ctrl+Alt+F9 setting cannot be read back.
    static void CycleJitterSign();
    /// The on-screen notice of the last F8 / F9 press while it should show (3 s), else null.
    static const char* Notice();
    /// DLAA replaced TAA within the last 100 ms (the presenter keeps RCAS off then).
    static bool Running();

    Ue4Dlss(const Instance& instance, Scheduler& scheduler, VideoCore::TextureCache& texture_cache,
            Runtime& runtime, Ue4Motion& motion);
    ~Ue4Dlss();

    /// A draw into one RGBA16F colour target (not TAA): the last one before TAA is TAA's colour
    /// input (the separate translucency composite).
    void OnColorTarget(VideoCore::ImageId id, VAddr address);
    /// A native TAA draw with the View buffer bound, sampling `reads`, into `output`. True: DLSS
    /// covers it (recorded at the first draw of the frame), so the draw must be skipped.
    /// `mark_written` gives `output` a render target's texture cache bookkeeping (the skipped
    /// draw would have): called before DLSS writes it.
    bool OnTaaDraw(const u8* view, std::span<const VAddr> reads, VideoCore::ImageId output,
                   VAddr output_address, const std::function<void()>& mark_written);
    /// The TAAU compute dispatch (U4), writing `output` (`output_width` x `output_height` from its
    /// T#). True: DLSS upscaled into it, so the dispatch must be skipped. `mark_written` gives it
    /// a storage write's texture cache bookkeeping.
    bool OnTaauDispatch(const u8* view, std::span<const VAddr> reads, VideoCore::ImageId output,
                        VAddr output_address, u32 output_width, u32 output_height,
                        const std::function<void()>& mark_written);
    /// Enabled and switched on (F9): worth looking up the TAAU output image.
    static bool On();

private:
    struct Target {
        VideoCore::ImageId id{};
        u64 uid = 0;
        VAddr address = 0;
    };
    bool Fail(const char* why);
    /// Both hooks: `output_width` 0 = native (DLAA into the draws' colour target).
    bool Run(const u8* view, std::span<const VAddr> reads, VideoCore::ImageId output_id,
             VAddr output_address, u32 output_width, u32 output_height,
             const std::function<void()>& mark_written);

    const Instance& instance;
    Scheduler& scheduler;
    VideoCore::TextureCache& texture_cache;
    Runtime& runtime;
    Ue4Motion& motion;

    Target color;
    bool have_frame = false;
    u32 last_frame = 0;  ///< View frame number of the last frame DLSS ran
    bool was_on = false; ///< DLSS ran in the last frame with TAA draws: else the next one resets
    VideoCore::UniqueImage output;
    vk::UniqueImageView output_view;
    u32 own_width = 0, own_height = 0;
    std::chrono::steady_clock::time_point last_time{};
    u64 evaluations = 0, resets = 0;
    static std::atomic<int> jitter_sign;      ///< 0-3, see CycleJitterSign
    static std::atomic<bool> sign_changed;    ///< the next frame resets
    static std::atomic<s64> notice_until_ns;  ///< steady clock
    static std::atomic<const char*> notice;
    static std::atomic<bool> evaluate_failed; ///< set on the recording thread
    static std::atomic<s64> last_run_ns; ///< steady clock
};

} // namespace Vulkan
