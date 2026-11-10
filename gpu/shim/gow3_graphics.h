// SPDX-License-Identifier: GPL-2.0-or-later
// gow3: graphics options of the in-game menu (Insert / R3+L2), kept in medievil.ini. The live ones
// apply on the next frame; the startup ones (patches, GPU readback modes) are read by run.py and
// apply through "Apply and restart". The launcher reads and writes the same keys.
#pragma once

#include <array>
#include <string_view>

namespace Gow3Graphics {

enum DisplayMode : int { Windowed = 0, Borderless = 1, Fullscreen = 2, DisplayModeCount };
inline constexpr std::array<std::string_view, DisplayModeCount> DisplayModeKeys = {
    "windowed", "borderless", "fullscreen"};
inline constexpr std::array<const char*, DisplayModeCount> DisplayModeNames = {
    "Windowed", "Borderless", "Fullscreen"};

/// Present rate limits; 0 = unlimited. Negative: from the display refresh (ResolveFpsLimit).
/// -1: half the display refresh (each frame shows for exactly two refreshes: even pacing).
/// -2: with VRR (G-Sync / FreeSync) below the refresh by NVIDIA Reflex's rule, refresh -
///     refresh^2 / 3600 (its automatic caps: 116 at 120 Hz, 138 at 144, 171 at 180, 225 at 240);
///     on a fixed-refresh display the refresh itself.
/// -3: with VRR one under half the refresh (119 at 240 Hz: at exactly 120 the VRR display
///     flickered, T-F19 / T-F27); on a fixed-refresh display exactly half (each frame two
///     refreshes). 119 / 141 / 237: fixed values for VRR on 240 and 144 Hz displays.
inline constexpr std::array<int, 11> FpsLimits = {-2, -3, 30, 60, 119, 120, 141, 237, 240, 0, -1};
/// The limit in FPS of a FpsLimits value (or any 20-1000) on a `refresh` Hz display, with or
/// without VRR (Gow3Settings::display_vrr); 0: none.
inline int ResolveFpsLimit(int value, unsigned refresh, bool vrr = true) {
    switch (value) {
    case -1: return int(refresh / 2);
    case -2: return int(refresh) - (vrr ? int(refresh * refresh / 3600) : 0);
    case -3: return int(refresh / 2) - (vrr ? 1 : 0);
    default: return value < 0 ? 0 : value;
    }
}

/// Game patch suffixes ("Resolution Patch - <name>"); "native" is no resolution patch.
/// Unreal Engine 4 (MediEvil): the scene scale; the *_native ones change the game's output size.
inline constexpr std::array<std::string_view, 10> Resolutions = {
    "native", "480p", "720p", "1440p", "1800p", "4K",
    "720p_native", "1440p_native", "1800p_native", "4K_native"};
/// Engine frame rate ceilings: 60 is the game's own, 120 and 240 are frame rate patches.
inline constexpr std::array<int, 3> EngineFps = {60, 120, 240};

template <typename T, std::size_t N>
constexpr int IndexOf(const std::array<T, N>& values, const T& value, int fallback) {
    for (std::size_t i = 0; i < N; ++i) {
        if (values[i] == value) return int(i);
    }
    return fallback;
}

inline int ParseDisplayMode(std::string_view text) {
    return IndexOf(DisplayModeKeys, text, Windowed);
}
/// A listed limit, or any limit from 20 to 1000 FPS (medievil.ini); else unlimited.
inline int ParseFpsLimit(int value) {
    return IndexOf(FpsLimits, value, -1) >= 0 || (value >= 20 && value <= 1000) ? value : 0;
}
/// Index into Resolutions; unknown text is native.
inline int ParseResolution(std::string_view text) {
    return IndexOf(Resolutions, text, 0);
}
/// A listed ceiling, else 120 (the default patch).
inline int ParseEngineFps(int value) {
    return IndexOf(EngineFps, value, -1) >= 0 ? value : 60;
}

/// Frames the GPU command thread may queue ahead of the GPU; 0 = unbounded.
inline constexpr std::array<int, 3> FramesQueued = {1, 2, 0};
inline int ParseFramesQueued(int value) {
    return IndexOf(FramesQueued, value, -1) >= 0 ? value : 1;
}
/// RCAS strength in percent; invalid values fall back to 75.
inline int ParseRcasStrength(int value) {
    return value >= 0 && value <= 100 ? value : 75;
}
/// FSR RCAS attenuation for a strength: 0 is the sharpest, 2 the softest.
inline float RcasAttenuation(int strength) {
    return 2.0f * (1.0f - float(ParseRcasStrength(strength)) / 100.0f);
}
/// Whether the FSR pass runs: FSR 1 when the window is larger than the game's image, RCAS
/// alone when they are the same size. A smaller window is left to the regular scaling.
inline bool FsrPassRuns(bool fsr1, bool rcas, int in_w, int in_h, int out_w, int out_h) {
    const bool upscale = in_w < out_w && in_h < out_h;
    const bool same = in_w == out_w && in_h == out_h;
    return (fsr1 && upscale) || (rcas && same);
}

/// What the present limit really allows: the engine ceiling caps it.
inline int EffectiveFps(int limit, int engine_fps) {
    return limit == 0 || limit > engine_fps ? engine_fps : limit;
}

} // namespace Gow3Graphics
