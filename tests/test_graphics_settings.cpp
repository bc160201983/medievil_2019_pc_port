// gow3.ini graphics keys (gow3_settings.cpp, gow3_graphics.h).
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include "gow3_graphics.h"
#include "gow3_settings.h"

namespace G = Gow3Graphics;

// src/runtime_extra.c: the live console variables the settings request, as "name=value;".
static std::string requested;
extern "C" void runtime_ue4_request_cvar(const char* name, const char* value) {
    requested += std::string(name) + "=" + value + ";";
}

static std::string Read(const char* path) {
    std::ifstream in(path);
    std::stringstream s;
    s << in.rdbuf();
    return s.str();
}

int main() {
    assert(G::ParseDisplayMode("borderless") == G::Borderless && G::ParseDisplayMode("bogus") == G::Windowed);
    assert(G::ParseFpsLimit(240) == 240 && G::ParseFpsLimit(75) == 75 && G::ParseFpsLimit(5) == 0);
    assert(G::ParseResolution("1440p") == 3 && G::ParseResolution("8K") == 0);
    assert(G::ParseEngineFps(240) == 240 && G::ParseEngineFps(90) == 60);
    assert(G::ParseFramesQueued(2) == 2 && G::ParseFramesQueued(0) == 0 && G::ParseFramesQueued(5) == 1);
    assert(G::ParseRcasStrength(100) == 100 && G::ParseRcasStrength(101) == 75 && G::ParseRcasStrength(-1) == 75);
    assert(G::RcasAttenuation(100) == 0.0f && G::RcasAttenuation(0) == 2.0f && G::RcasAttenuation(75) == 0.5f);
    assert(G::FsrPassRuns(true, false, 1280, 720, 2560, 1440) && !G::FsrPassRuns(false, false, 1280, 720, 2560, 1440));
    assert(G::FsrPassRuns(false, true, 1920, 1080, 1920, 1080) && !G::FsrPassRuns(true, false, 1920, 1080, 1920, 1080));
    assert(!G::FsrPassRuns(true, true, 2560, 1440, 1920, 1080));  // larger image: regular scaling
    assert(G::EffectiveFps(0, 120) == 120 && G::EffectiveFps(240, 120) == 120 && G::EffectiveFps(60, 240) == 60);

    const char* path = "graphics-settings-test.ini";
    {
        std::ofstream out(path);
        out << "display_mode=borderless\nvsync=0\nfps_limit=240\nasync_shaders=0\n"
               "render_resolution=1440p\nengine_fps=240\ndeferred_readback=0\nstale_readback=1\n"
               "fsr1=1\nrcas=0\nrcas_strength=40\nframes_queued=2\n";
    }
    _putenv_s("GOW3_CONFIG", path);
    _putenv_s("GOW3_ASYNC_SHADERS", "");
    auto& v = Gow3Settings::Get();
    Gow3Settings::Load();
    assert(v.display_mode == G::Borderless && !v.vsync && v.fps_limit == 240 && !v.async_shaders);
    assert(v.render_resolution == 3 && v.engine_fps == 240 && !v.deferred_readback && v.stale_readback);
    assert(v.fsr1 && !v.rcas && v.rcas_strength == 40 && v.frames_queued == 2);
    assert(!Gow3Settings::GraphicsNeedRestart());
    v.engine_fps = 120;
    assert(Gow3Settings::GraphicsNeedRestart());
    v.fps_limit = 60;  // live: no restart because of it
    v.engine_fps = 240;
    assert(!Gow3Settings::GraphicsNeedRestart());
    // The single particle pass is live (r.GPUParticle.FixDeltaSeconds, runtime_extra.c); the decal
    // prepass still needs a start.
    v.perf[Gow3Settings::PerfParticleSinglePass] = !v.perf[Gow3Settings::PerfParticleSinglePass];
    assert(!Gow3Settings::GraphicsNeedRestart());
    v.perf[Gow3Settings::PerfParticleSinglePass] = !v.perf[Gow3Settings::PerfParticleSinglePass];
    v.perf[Gow3Settings::PerfNoDecalPrepass] = !v.perf[Gow3Settings::PerfNoDecalPrepass];
    assert(Gow3Settings::GraphicsNeedRestart());
    v.perf[Gow3Settings::PerfNoDecalPrepass] = !v.perf[Gow3Settings::PerfNoDecalPrepass];

    Gow3Settings::Save();
    const std::string saved = Read(path);
    for (const char* line : {"display_mode=borderless\n", "vsync=0\n", "fps_limit=60\n", "async_shaders=0\n",
                             "render_resolution=1440p\n", "engine_fps=240\n", "deferred_readback=0\n",
                             "stale_readback=1\n", "fsr1=1\n", "rcas=0\n", "rcas_strength=40\n",
                             "frames_queued=2\n"}) {
        assert(saved.find(line) != std::string::npos);
    }

    {
        std::ofstream out(path);
        out << "display_mode=huge\nfps_limit=7\nrender_resolution=8K\nengine_fps=1\n"
               "rcas_strength=250\nframes_queued=9\n";
    }
    _putenv_s("GOW3_ASYNC_SHADERS", "1");  // the environment still wins at start
    v.async_shaders = false;
    Gow3Settings::Load();
    assert(v.display_mode == G::Windowed && v.fps_limit == 0 && v.render_resolution == 0 && v.engine_fps == 60);
    assert(v.rcas_strength == 75 && v.frames_queued == 1);
    assert(v.async_shaders);
    {
        std::ofstream out(path);
        out << "cheat_double_jumps=1\ncheat_very_hard=1\n";
    }
    Gow3Settings::Load();
    Gow3Settings::Save();
    const std::string cheats = Read(path);
    assert(cheats.find("cheat_double_jumps=1\n") != std::string::npos);
    assert(cheats.find("cheat_very_hard=1\n") != std::string::npos);

    // Only the settings changed in the menu are written: an environment override of a test
    // launch is not saved, the launcher's keys stay, and no other key is added.
    {
        std::ofstream out(path);
        out << "frames_queued=1\nfps_limit=240\nue4_quality=console\n";
    }
    _putenv_s("GOW3_FRAMES_AHEAD", "2");
    Gow3Settings::Load();
    assert(v.frames_queued == 2);
    v.fps_limit = 120;
    Gow3Settings::Save();
    const std::string partial = Read(path);
    assert(partial == "frames_queued=1\nfps_limit=120\nue4_quality=console\n");
    Gow3Settings::Save();  // nothing changed since: the file stays as it is
    assert(Read(path) == partial);
    _putenv_s("GOW3_FRAMES_AHEAD", "");

    // Render scale: any whole percent (graphics_options.parse_render_scale), live from 50% up to the
    // start value or 100%, with the upsampling switch when it crosses 100%; written back as a number.
    namespace S = Gow3Settings;
    assert(S::ParseRenderScale("72") == 72 && S::ParseRenderScale("66.7") == 66 && S::ParseRenderScale("25") == 25);
    assert(S::ParseRenderScale("24") == 100 && S::ParseRenderScale("201") == 100 && S::ParseRenderScale("x") == 100);
    assert(S::ParseRenderScale("67%") == 100 && S::ParseRenderScale("") == 100 && S::ParseRenderScale(" 58") == 58);
    {
        std::ofstream out(path);
        out << "render_scale=67\nue4_quality=console\n";
    }
    Gow3Settings::Load();
    assert(v.render_scale == 67 && v.startup_render_scale == 67 && v.live_render_scale == 67);
    assert(S::LiveRenderScaleMin() == 50 && S::LiveRenderScaleMax() == 100 && !Gow3Settings::GraphicsNeedRestart());
    requested.clear();
    v.render_scale = 72;
    assert(S::RequestRenderScale(v.render_scale) == 72 && requested == "r.ScreenPercentage=72;");
    assert(!Gow3Settings::GraphicsNeedRestart()); // live: no restart
    requested.clear();
    assert(S::RequestRenderScale(72) == 72 && requested.empty()); // unchanged: nothing requested
    v.render_scale = 100;
    assert(S::RequestRenderScale(100) == 100 && requested == "r.TemporalAA.Upsampling=0;r.ScreenPercentage=100;");
    requested.clear();
    v.render_scale = 150; // above what this start's memory was sized for: 100% now, 150% on restart
    assert(S::RequestRenderScale(150) == 100 && requested.empty() && Gow3Settings::GraphicsNeedRestart());
    v.render_scale = 51;
    assert(S::RequestRenderScale(51) == 51 && requested == "r.TemporalAA.Upsampling=1;r.ScreenPercentage=51;");
    assert(!Gow3Settings::GraphicsNeedRestart() && S::RequestRenderScale(20) == 50);
    v.render_scale = 51;
    Gow3Settings::Save();
    assert(Read(path) == "render_scale=51\nue4_quality=console\n");
    {
        std::ofstream out(path);
        out << "render_scale=133\nue4_aa=fxaa\n";
    }
    Gow3Settings::Load();
    assert(S::LiveRenderScaleMax() == 133 && !v.ue4_aa_temporal);
    v.render_scale = 120;
    assert(!Gow3Settings::GraphicsNeedRestart() && S::RequestRenderScale(120) == 120);
    v.render_scale = 90; // FXAA start: live without temporal AA, which a start below 100% would use
    assert(Gow3Settings::GraphicsNeedRestart());
    {
        std::ofstream out(path);
        out << "render_scale=bogus\n";
    }
    Gow3Settings::Load();
    assert(v.render_scale == 100 && v.startup_render_scale == 100);
    Gow3Settings::Save();
    assert(Read(path) == "render_scale=bogus\n"); // unchanged in the menu: kept as written
    std::remove(path);

    // Graphics quality and its settings (plan-graphics-settings.md): "preset" and missing keys follow
    // ue4_quality, old tokens are translated, the level shown is the one all settings match, else Custom.
    {
        namespace S = Gow3Settings;
        const auto index = [](const char* key) {
            for (int o = 0; o < S::UE4LiveCount; ++o) {
                if (std::strcmp(S::UE4LiveOptions[o].key, key) == 0) return o;
            }
            return -1;
        };
        const auto token = [&](const char* key) { return std::string(S::UE4LiveOptions[index(key)].tokens[v.ue4_live[index(key)]]); };
        {
            std::ofstream out(path);
            out << "ue4_quality=high\nshadow_resolution=preset\nshadow_distance=far\nlod_distance=high\n";
        }
        S::Load();
        assert(token("shadow_resolution") == "dynamic" && token("shadow_distance") == "ps4");
        assert(token("texture_filtering") == "16" && token("shadow_quality") == "4"); // missing: High's
        assert(S::UE4MatchedLevel(v, v.ue4_quality) == 1);                           // all of High's: High
        v.ue4_live[index("bloom")] = 1;                                                  // Low bloom by hand
        assert(S::UE4MatchedLevel(v, v.ue4_quality) == -1);                            // Custom
        S::UE4ApplyLevel(4);                                                             // Low picked
        assert(v.ue4_quality == 4 && S::UE4MatchedLevel(v, 4) == 4 && token("foliage_density") == "0.6");
        assert(S::RestartChanges() > 0); // the settings that are not live changed: a restart applies them
        for (int o = 0; o < S::UE4LiveCount; ++o) {
            v.ue4_live[o] = S::UE4LevelChoice(o, S::UE4DeckLevel);
        }
        assert(S::UE4MatchedLevel(v, 4) == S::UE4DeckLevel); // the launcher's Deck Optimized
        std::remove(path);
    }
    std::puts("PASS: graphics settings");
}
