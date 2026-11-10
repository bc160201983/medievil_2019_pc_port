// SPDX-License-Identifier: GPL-2.0-or-later
// gow3 U1 (plan-ue4-motion-vectors.md, study-taau-frame-map.md): finds UE4's temporal passes in
// each frame (the velocity pass, TAA as two draws or one compute dispatch, the View uniform
// buffer bound to TAA, the tonemap, the first UI draw) and logs what it found. Log only, nothing
// in the frame changes. GOW3_UE4_FRAME_MAP=1; off by default.
//
// A frame here runs from one frame's first draw into a display buffer to the next one's: the
// tonemap and UI of one frame, then the scene passes (velocity, TAA) of the next.

#pragma once

#include <array>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "common/types.h"

namespace Vulkan {

class Ue4FrameMap {
public:
    // Shaders known from the captures (study-taau-frame-map.md section 5). Other shaders in
    // the same roles are logged by role, not matched.
    static constexpr u64 TaaNormal = 0xe88c8fcbe08ea0b6ULL;     ///< fs, stencil bit 3 clear
    static constexpr u64 TaaResponsive = 0xe5588ed466d8f2f3ULL; ///< fs, stencil bit 3 set
    static constexpr u64 TaaCompute = 0xb373f39908ae979fULL;    ///< cs, TAAU (inferred)
    static constexpr u64 TaauOther = 0xa080f599034de78eULL;     ///< cs, same compile, role unknown
    static constexpr u64 Tonemap = 0x3632d4c4faaa85d2ULL;
    static constexpr u64 FirstUi = 0x4fe20af683865713ULL;
    static constexpr u32 ViewBytes = 3296;     ///< UE 4.21 View uniform buffer
    static constexpr u32 TaaParamBytes = 176;  ///< TAA's own parameters (binding 1 of its PS)
    static constexpr u32 EventLines = 40;      ///< cut evidence lines per summary at most
    static constexpr u32 SummaryFrames = 300;  ///< a summary every 300 frames (5 s at 60)
    static constexpr u32 ChangeLines = 10;     ///< change / cut lines per summary at most

    struct Image {
        VAddr address = 0;
        u32 width = 0;
        u32 height = 0;
    };
    /// What the rasterizer saw at one draw or dispatch (guest addresses).
    struct Pass {
        bool compute = false;
        const char* stage = "fs"; ///< stage of `hash`: fs, vs (draw without pixel shader), cs
        u64 hash = 0;
        std::array<u32, 3> groups{}; ///< dispatch size
        Image target{};              ///< first colour target (draws)
        u32 targets = 0;             ///< colour targets bound
        bool target_rg16 = false;    ///< R16G16_UNORM, UE4's velocity format
        bool target_rgba16f = false; ///< R16G16B16A16_SFLOAT (scene colour, TAA's output)
        bool display = false;        ///< the target is a display buffer
        VAddr depth = 0;             ///< bound depth buffer (draws)
        std::array<VAddr, 16> reads{}; ///< sampled images
        std::array<std::array<u32, 2>, 16> read_sizes{};
        u32 num_reads = 0;
        std::array<Image, 4> writes{}; ///< storage images
        u32 num_writes = 0;
        const u8* view = nullptr; ///< a bound ViewBytes buffer; read during OnPass only
        const u8* taa_params = nullptr; ///< a bound TaaParamBytes buffer; read during OnPass only
        std::array<u32, 8> buffer_sizes{};
        u32 num_buffers = 0;
    };

    using Sink = std::function<void(const std::string&)>;

    /// GOW3_UE4_FRAME_MAP=1 (read once).
    static bool Enabled();
    explicit Ue4FrameMap(Sink sink = {});

    void OnPass(const Pass& pass);

private:
    struct View {
        bool valid = false;
        std::array<float, 4> jitter{};  ///< TemporalAAJitter: this frame xy, last frame zw (NDC)
        bool jitter_in_projection = false; ///< ViewToClip[2][0..1] equals jitter.xy
        std::array<float, 2> view_size{};
        std::array<float, 2> buffer_size{};
        float pre_exposure = 0;
        float near_plane = 0;
        float mip_bias = 0;
        u32 frame_number = 0;
        u32 index = 0; ///< StateFrameIndexMod8
        u32 camera_cut = 0;
        float camera_delta = 0; ///< largest |ClipToPrevClip - identity| element
        float camera_pixels = 0; ///< largest one-frame move of 5 screen points at infinity / 10 m
    };
    struct Frame {
        u32 passes = 0;
        VAddr display = 0;
        Image velocity{};
        VAddr velocity_depth = 0;
        u32 velocity_draws = 0;
        u32 other_rg16_draws = 0;
        u32 taa_draws = 0;
        u32 taa_dispatches = 0;
        std::array<u64, 2> taa_hashes{};
        const char* taa_stage = "fs";
        std::array<u32, 3> taa_groups{};
        Image taa_output{};
        bool taa_reads_velocity = false;
        bool taa_reads_depth = false;
        bool taa_without_view = false;
        std::array<u32, 8> taa_buffer_sizes{};
        u32 taa_num_buffers = 0;
        View view{};
        u32 taau_other_dispatches = 0;
        std::array<u32, 3> taau_other_groups{};
        u32 display_draws = 0;
        std::array<u64, 4> display_shaders{}; ///< distinct, in draw order
        u32 num_display_shaders = 0;
        Image display_target{};
        u32 tonemap_draws = 0;
        u32 ui_draws = 0;
        u32 before_ui_other = 0; ///< display draws after the tonemap, before the first UI draw
        std::vector<std::pair<const char*, u64>> readers; ///< passes sampling the velocity
        // Camera cut evidence (U3): does the first TAA pass read last frame's TAA output (its
        // history)? 0 unknown (no TAA last frame), 1 yes, 2 no. Its other sampled images.
        int taa_history = 0;
        std::array<Image, 8> taa_reads{};
        u32 taa_num_reads = 0;
        bool have_params = false;
        std::array<float, TaaParamBytes / 4> params{};
    };

    void Close();
    void Print(const std::string& line) const;
    std::string Describe(const Frame& frame) const;
    std::string Signature(const Frame& frame) const;
    static View ReadView(const u8* data);

    Sink sink;
    Frame frame{};
    VAddr last_taa_output = 0; ///< the last frame's TAA output (this frame's TAA history)
    bool have_last_params = false;
    std::array<float, TaaParamBytes / 4> last_params{};
    VAddr last_velocity = 0; ///< last frame with a velocity pass: its target and depth
    VAddr last_depth = 0;
    std::string last_signature;
    std::vector<std::pair<const char*, u64>> seen_readers;
    std::vector<std::pair<float, float>> jitters; ///< distinct this summary
    bool have_frame_number = false;
    u32 last_frame_number = 0;
    u64 frames_total = 0;
    struct Counts {
        u32 frames = 0, velocity = 0, taa = 0, taa_draws = 0, taa_compute = 0, view = 0,
            tonemap = 0, ui = 0, cuts = 0, gaps = 0, repeats = 0, changes = 0, unshown = 0,
            history_drops = 0, param_changes = 0, events = 0;
        float max_camera_pixels = 0;
    } counts{};
};

} // namespace Vulkan
