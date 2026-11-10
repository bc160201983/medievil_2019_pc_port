// SPDX-License-Identifier: GPL-2.0-or-later
// gow3 N0 (research/docs/study-native-render-paths.md, impl-draw-reduction.md), with
// GOW3_FRAME_STATS=1 only: the recorded draws per frame by class, told apart by their render
// targets (the shapes of research/docs/study-heavy-frame-map.md), and stage B's time per draw of
// each class (DrawRecord, inclusive). Printed every 5 s after "Frame stats" (videoout/driver.cpp)
// as "Draw classes per frame: ..."; every counter is reset by the report.
#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <string>
#include <x86intrin.h>

namespace Gow3DrawClasses {

enum Class : std::uint32_t {
    ParticleSim,    ///< 2 targets, the first RGBA32F, no depth: GPU particle simulation (and tile clears)
    ParticleInject, ///< 3+ targets, the first RGBA32F: new GPU particles written into the state
    DepthPrepass,   ///< no colour, 32-bit depth written: depth prepass
    ShadowDepth,    ///< no colour, 16-bit depth: per-object / light shadow maps
    DepthTestOnly,  ///< no colour, depth / stencil without depth writes: occlusion boxes, stencil
    NoAttachment,   ///< no colour, no depth (the 3-vertex draws before shadow maps)
    BasePass,       ///< 5+ colour targets with depth: the G-buffer
    Velocity,       ///< R16G16 with depth
    SceneColor,     ///< RGBA16F with depth: translucency, particles, light volumes, fog
    OtherDepth,     ///< any other colour target with depth (shadow projection, decals)
    Post,           ///< colour without depth: post-processing, HZB build, bloom, UI off-screen
    Display,        ///< into a display buffer: tonemap / UI / present
    Count,
};
inline constexpr const char* Names[Count] = {
    "particle sim",  "particle inject", "depth prepass", "shadow depth",
    "depth/stencil only", "no attachment", "base pass", "velocity",
    "scene colour",  "other with depth", "post", "display",
};

/// A draw's render targets, as bound for it (Rasterizer::PrepareRenderState).
struct Shape {
    std::uint32_t colors = 0; ///< colour attachments bound
    bool color0_rgba32f = false, color0_rgba16f = false, color0_rg16 = false, display = false;
    bool depth = false, depth16 = false, depth_write = false;
};

inline Class Classify(const Shape& s) {
    if (s.colors == 0) {
        if (!s.depth) {
            return NoAttachment;
        }
        if (s.depth16) {
            return ShadowDepth;
        }
        return s.depth_write ? DepthPrepass : DepthTestOnly;
    }
    if (s.display) {
        return Display;
    }
    if (s.color0_rgba32f && !s.depth) {
        return s.colors == 2 ? ParticleSim : s.colors >= 3 ? ParticleInject : Post;
    }
    if (!s.depth) {
        return Post;
    }
    if (s.colors >= 5) {
        return BasePass;
    }
    if (s.color0_rg16) {
        return Velocity;
    }
    return s.color0_rgba16f ? SceneColor : OtherDepth;
}

inline std::array<std::atomic<std::uint64_t>, Count> draws{}, cycles{};
/// Particle simulation runs (consecutive simulation draws), and the runs with an inject draw
/// since the previous run (for the dt=0 copy skip: research/docs/impl-draw-reduction.md).
inline std::atomic<std::uint64_t> sim_runs{0}, sim_runs_after_inject{0};
/// Runs whose first draw writes the same colour target 0 (guest address) as the previous run's
/// (research/docs/study-particle-fixed-step.md): UE4 alternates the state pair once per
/// simulation, so with the single pass (r.GPUParticle.FixDeltaSeconds 0) this stays 0 unless a
/// flip went missing or doubled (the run then reads a pair one simulation too old); with the
/// default fixed step the next frame's first run writes the previous frame's second run's pair:
/// about one a frame.
inline std::atomic<std::uint64_t> sim_runs_same_target{0};
inline std::uint64_t last_run_target = 0;
/// The previous recorded draw's class and whether an inject came since the last simulation run
/// (draws are recorded on one thread at a time: stage B, or stage A without the draw pipe).
inline Class last_class = Count;
inline bool inject_since_run = false;
/// The simulation class's pixel shaders (guest hash) and their draws: the simulation itself and
/// any other shader drawing into the same targets (tile clears). The first four seen are kept.
struct SimShader {
    std::atomic<std::uint64_t> hash{0}, draws{0};
};
inline std::array<SimShader, 4> sim_shaders{};

inline void Note(Class c, std::uint64_t draw_cycles, std::uint64_t ps, std::uint64_t target = 0) {
    draws[c].fetch_add(1, std::memory_order_relaxed);
    cycles[c].fetch_add(draw_cycles, std::memory_order_relaxed);
    if (c == ParticleSim) {
        for (auto& shader : sim_shaders) {
            std::uint64_t hash = shader.hash.load(std::memory_order_relaxed);
            if (!hash) {
                shader.hash.store(ps, std::memory_order_relaxed);
                hash = ps;
            }
            if (hash == ps) {
                shader.draws.fetch_add(1, std::memory_order_relaxed);
                break;
            }
        }
    }
    if (c == ParticleSim && last_class != ParticleSim) {
        sim_runs.fetch_add(1, std::memory_order_relaxed);
        if (inject_since_run) {
            sim_runs_after_inject.fetch_add(1, std::memory_order_relaxed);
        }
        inject_since_run = false;
        if (target && target == last_run_target) {
            sim_runs_same_target.fetch_add(1, std::memory_order_relaxed);
        }
        last_run_target = target;
    } else if (c == ParticleInject) {
        inject_since_run = true;
    }
    last_class = c;
}

/// One recorded draw: its time from construction to destruction, counted in the class set
/// after its targets are known (nothing without GOW3_FRAME_STATS: `enabled` false).
struct Scope {
    explicit Scope(bool enabled) : start{enabled ? __rdtsc() : 0} {}
    ~Scope() {
        if (start && cls != Count) {
            Note(cls, __rdtsc() - start, ps, target);
        }
    }
    std::uint64_t start;
    Class cls = Count;
    std::uint64_t ps = 0;     ///< the pixel shader's guest hash
    std::uint64_t target = 0; ///< colour target 0's guest address
};

/// The 5 s report line: draws per frame and us per draw of each class with any draws.
inline std::string Report(std::uint64_t frames) {
    // TSC cycles per microsecond over the window (the report's own clock pair).
    static std::uint64_t last_tsc = 0, last_ns = 0;
    const std::uint64_t tsc = __rdtsc();
    const std::uint64_t ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                 std::chrono::steady_clock::now().time_since_epoch())
                                 .count();
    const double cycles_per_us =
        last_ns && ns > last_ns ? double(tsc - last_tsc) * 1e3 / double(ns - last_ns) : 0.0;
    last_tsc = tsc;
    last_ns = ns;
    const double per_frame = frames ? double(frames) : 1.0;
    std::string out = "Draw classes per frame:";
    char item[96];
    std::uint64_t total = 0;
    for (std::uint32_t c = 0; c < Count; ++c) {
        const std::uint64_t n = draws[c].exchange(0, std::memory_order_relaxed);
        const std::uint64_t cyc = cycles[c].exchange(0, std::memory_order_relaxed);
        total += n;
        if (!n) {
            continue;
        }
        if (cycles_per_us > 0) {
            std::snprintf(item, sizeof(item), " %s %.1f (%.2f us),", Names[c], n / per_frame,
                          cyc / cycles_per_us / n);
        } else {
            std::snprintf(item, sizeof(item), " %s %.1f,", Names[c], n / per_frame);
        }
        out += item;
    }
    const std::uint64_t runs = sim_runs.exchange(0, std::memory_order_relaxed);
    const std::uint64_t after_inject = sim_runs_after_inject.exchange(0, std::memory_order_relaxed);
    const std::uint64_t same_target = sim_runs_same_target.exchange(0, std::memory_order_relaxed);
    std::snprintf(item, sizeof(item), " total %.1f; particle sim runs %.2f (%.2f after an inject draw",
                  total / per_frame, runs / per_frame, after_inject / per_frame);
    out += item;
    std::snprintf(item, sizeof(item), ", %.2f on the previous run's target)", same_target / per_frame);
    out += item;
    for (auto& shader : sim_shaders) {
        const std::uint64_t n = shader.draws.exchange(0, std::memory_order_relaxed);
        if (n) {
            std::snprintf(item, sizeof(item), ", sim ps %016llx %.1f",
                          static_cast<unsigned long long>(shader.hash.load(std::memory_order_relaxed)),
                          n / per_frame);
            out += item;
        }
    }
    out += "\n";
    return out;
}

} // namespace Gow3DrawClasses
