// SPDX-License-Identifier: GPL-2.0-or-later
// gow3 U1: the UE4 frame map finds the passes of a native frame (TAA draws) and a TAAU frame
// (TAA compute) from synthetic passes, reads the View buffer and logs changes and summaries.
#include <cassert>
#include <cstdio>
#include <array>
#include <cstring>
#include <string>
#include <vector>
#include "video_core/renderer_vulkan/vk_ue4_frame_map.h"

using Vulkan::Ue4FrameMap;

namespace {

constexpr VAddr Display[2] = {0x1000000, 0x2000000};
constexpr VAddr Velocity = 0x3000000, Depth = 0x4000000, Colour = 0x5000000, History = 0x6000000;
constexpr VAddr TaaOut = 0x7000000;

std::vector<u8> MakeView(float width, float height, u32 frame_number, u32 index, u32 cut) {
    std::vector<u8> view(Ue4FrameMap::ViewBytes);
    const auto put = [&](u32 offset, auto value) { std::memcpy(view.data() + offset, &value, 4); };
    const float jitter_x = 0.0001f * (index + 1), jitter_y = -0.0002f * (index + 1);
    for (u32 i = 0; i < 16; ++i) {
        put(1824 + i * 4, i % 5 == 0 ? 1.0f : 0.0f); // ClipToPrevClip = identity
    }
    put(384 + 32, jitter_x);
    put(384 + 36, jitter_y);
    put(1888, jitter_x);
    put(1892, jitter_y);
    put(1952, width);
    put(1956, height);
    put(1968, width);
    put(1972, height);
    put(2004, 1.0f);
    put(2112, 10.0f);
    put(2128, width < 3840 ? -0.32f : 0.0f);
    put(2140, frame_number);
    put(2144, index);
    put(2148, cut);
    return view;
}

Ue4FrameMap::Pass Draw(u64 ps, VAddr target, u32 w, u32 h) {
    Ue4FrameMap::Pass pass;
    pass.hash = ps;
    pass.target = {target, w, h};
    pass.targets = 1;
    pass.display = target == Display[0] || target == Display[1];
    return pass;
}

// TAA alternates between two outputs; each frame's history is the last frame's output. On a
// cut (`drop`) TAA reads a placeholder instead and its current-frame weight [25] is 1.
VAddr TaaOutput(u32 n) {
    return TaaOut + (n % 2) * 0x100000;
}

// One frame: tonemap + UI into the display buffer, then the next frame's velocity and TAA.
void Frame(Ue4FrameMap& map, u32 n, bool taau, u32 cut = 0, bool drop = false) {
    const VAddr history = drop ? History : TaaOutput(n + 1);
    std::array<float, Ue4FrameMap::TaaParamBytes / 4> params{};
    params[25] = drop ? 1.0f : 0.04f;
    params[10] = 0.01f * float(n % 8); // jitter-dependent: ignored
    const u32 w = taau ? 1024 : 3840, h = taau ? 576 : 2160;
    const u32 out_w = taau ? 1280 : 3840, out_h = taau ? 720 : 2160;
    map.OnPass(Draw(Ue4FrameMap::Tonemap, Display[n % 2], out_w, out_h));
    for (u32 i = 0; i < 3; ++i) {
        map.OnPass(Draw(Ue4FrameMap::FirstUi, Display[n % 2], out_w, out_h));
    }
    for (u32 i = 0; i < 5; ++i) {
        auto pass = Draw(0x1111 + i, Velocity, w, h);
        pass.target_rg16 = true;
        pass.depth = Depth;
        map.OnPass(pass);
    }
    auto ssr = Draw(0xf8cfb0056f2c2e12ULL, 0x8000000, w, h);
    ssr.reads = {Velocity, History, Depth};
    ssr.num_reads = 3;
    map.OnPass(ssr);
    const auto view = MakeView(float(w), float(h), 100 + n, n % 8, cut);
    if (taau) {
        Ue4FrameMap::Pass cs;
        cs.compute = true;
        cs.stage = "cs";
        cs.hash = Ue4FrameMap::TaaCompute;
        cs.groups = {160, 90, 1};
        cs.reads = {Colour, Velocity, Depth, history};
        cs.num_reads = 4;
        cs.writes[0] = {TaaOutput(n), out_w, out_h};
        cs.taa_params = reinterpret_cast<const u8*>(params.data());
        cs.num_writes = 1;
        cs.view = view.data();
        cs.buffer_sizes = {Ue4FrameMap::ViewBytes, 176};
        cs.num_buffers = 2;
        map.OnPass(cs);
    } else {
        for (const u64 ps : {Ue4FrameMap::TaaNormal, Ue4FrameMap::TaaResponsive}) {
            auto pass = Draw(ps, TaaOutput(n), w, h);
            pass.reads = {Depth, Velocity, Colour, history};
            pass.num_reads = 4;
            pass.view = view.data();
            pass.taa_params = reinterpret_cast<const u8*>(params.data());
            map.OnPass(pass);
        }
    }
}

bool Has(const std::vector<std::string>& lines, const char* text) {
    for (const auto& line : lines) {
        if (line.find(text) != std::string::npos) {
            return true;
        }
    }
    return false;
}

} // namespace

int main() {
    std::vector<std::string> lines;
    Ue4FrameMap map([&](const std::string& line) { lines.push_back(line); });
    assert(lines.size() == 1 && Has(lines, "on (log only)"));

    // Native frames: one change line when the first full frame closes, then none.
    for (u32 n = 0; n < 4; ++n) {
        Frame(map, n, false);
    }
    assert(Has(lines, "velocity read by fs f8cfb0056f2c2e12"));
    assert(Has(lines, "velocity read by fs e88c8fcbe08ea0b6"));
    assert(Has(lines, "TAA draws x2 fs e88c8fcbe08ea0b6 fs e5588ed466d8f2f3 -> 3840x2160, reads "
                      "velocity yes depth yes"));
    assert(Has(lines, "View 3840x2160 in 3840x2160"));
    assert(Has(lines, "(tonemap x1) fs 4fe20af683865713 (UI x3)"));
    const size_t native_lines = lines.size();
    Frame(map, 4, false);
    assert(lines.size() == native_lines);

    // A camera cut prints at once; so does TAA dropping its history and its weight going to 1
    // (and back), with the View frame.
    assert(Has(lines, "UE4 frame map: TAA parameters: "));
    assert(!Has(lines, "TAA does not read last frame's output"));
    Frame(map, 5, false, 1, true);
    Frame(map, 6, false);
    Frame(map, 7, false);
    assert(Has(lines, "camera cut at View frame 105"));
    assert(Has(lines, "TAA does not read last frame's output 7000000 (View frame 105, camera moved 0 px)"));
    assert(Has(lines, "TAA parameters changed (View frame 105, camera moved 0 px): [25] 0.04 -> 1"));
    assert(Has(lines, "TAA parameters changed (View frame 106, camera moved 0 px): [25] 1 -> 0.04"));

    // TAAU: TAA becomes one dispatch; the change is logged with its groups and output size.
    for (u32 n = 8; n < 10; ++n) {
        Frame(map, n, true);
    }
    assert(Has(lines, "TAA compute x1 cs b373f39908ae979f (160x90x1 groups) -> 1280x720"));
    assert(Has(lines, "View 1024x576 in 1024x576"));
    assert(Has(lines, "velocity read by cs b373f39908ae979f"));
    assert(Has(lines, "mip bias -0.32"));

    // The summary after SummaryFrames frames.
    for (u32 n = 10; n < Ue4FrameMap::SummaryFrames + 2; ++n) {
        Frame(map, n, true);
    }
    assert(Has(lines, "UE4 frame map: 300 frames: velocity 300, TAA 300"));
    assert(Has(lines, "camera cuts 1, View frame gaps 0, repeats 0, jitter values 8; TAA history "
                      "drops 1, parameter changes 2, largest camera move 0 px"));
    assert(Has(lines, "UE4 frame map: last frame: velocity 1024x576 (5 draws)"));
    for (const auto& line : lines) {
        std::printf("%s\n", line.c_str());
    }
    std::printf("ue4-frame-map-test: ok\n");
    return 0;
}
