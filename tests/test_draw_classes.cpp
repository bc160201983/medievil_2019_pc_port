// N0 draw classes (gpu/shim/gow3_draw_classes.h): classification by render targets, the
// particle simulation runs and the report line.
#include <cassert>
#include <cstdio>
#include <string>
#include "gow3_draw_classes.h"

using namespace Gow3DrawClasses;

int main() {
    // The heavy frame's passes (research/docs/study-heavy-frame-map.md section 1).
    Shape sim{.colors = 2, .color0_rgba32f = true};
    assert(Classify(sim) == ParticleSim);
    Shape inject{.colors = 4, .color0_rgba32f = true};
    assert(Classify(inject) == ParticleInject);
    assert(Classify(Shape{.depth = true, .depth_write = true}) == DepthPrepass);
    assert(Classify(Shape{.depth = true, .depth16 = true, .depth_write = true}) == ShadowDepth);
    assert(Classify(Shape{.depth = true}) == DepthTestOnly);
    assert(Classify(Shape{}) == NoAttachment);
    assert(Classify(Shape{.colors = 6, .color0_rgba16f = true, .depth = true, .depth_write = true}) ==
           BasePass);
    assert(Classify(Shape{.colors = 1, .color0_rg16 = true, .depth = true}) == Velocity);
    assert(Classify(Shape{.colors = 1, .color0_rgba16f = true, .depth = true}) == SceneColor);
    assert(Classify(Shape{.colors = 1, .depth = true}) == OtherDepth);
    assert(Classify(Shape{.colors = 1, .color0_rgba16f = true}) == Post);
    assert(Classify(Shape{.colors = 1, .color0_rgba32f = true}) == Post);
    assert(Classify(Shape{.colors = 1, .display = true}) == Display);

    // Two runs of 3 simulation draws with an inject between them, then a depth draw.
    Note(ParticleSim, 10, 0xabc);
    Note(ParticleSim, 10, 0xabc);
    Note(ParticleSim, 10, 0xdef);
    Note(ParticleInject, 10, 0x123);
    Note(ParticleSim, 10, 0xabc);
    Note(ParticleSim, 10, 0xabc);
    Note(ParticleSim, 10, 0xabc);
    Note(DepthPrepass, 10, 0);
    assert(sim_runs == 2 && sim_runs_after_inject == 1);
    Report(1); // the first report has no clock pair yet

    // The state pair alternates per run (targets A, B, A): no run on its predecessor's target.
    // A run on the same target as the one before (a missing or doubled flip) is counted once.
    Note(ParticleSim, 10, 0xabc, 0xa000);
    Note(ParticleSim, 10, 0xabc, 0xa000);
    Note(DepthPrepass, 10, 0);
    Note(ParticleSim, 10, 0xabc, 0xb000);
    Note(DepthPrepass, 10, 0);
    Note(ParticleSim, 10, 0xabc, 0xa000);
    Note(DepthPrepass, 10, 0);
    assert(sim_runs_same_target == 0);
    Note(ParticleSim, 10, 0xabc, 0xa000);
    Note(DepthPrepass, 10, 0);
    assert(sim_runs == 4 && sim_runs_same_target == 1);
    Report(1);

    Note(ShadowDepth, 10, 0);
    const std::string line = Report(1);
    std::fputs(line.c_str(), stdout);
    assert(line.rfind("Draw classes per frame: shadow depth 1.0", 0) == 0);
    assert(line.find("total 1.0; particle sim runs 0.00 (0.00 after an inject draw, 0.00 on the previous run's "
                     "target)") != std::string::npos);
    assert(line.back() == '\n');
    std::puts("draw classes: ok");
    return 0;
}
