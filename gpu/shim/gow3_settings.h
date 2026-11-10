// SPDX-License-Identifier: GPL-2.0-or-later
// gow3: user settings changed at run time from the in-game menu (gow3_overlay.h) and kept
// in medievil.ini (GOW3_CONFIG overrides the path). Environment variables override the file at
// start. Readers load the atomics every frame; writers are the menu and Load().

#pragma once

#include "../../src/cheat_hook.h"

#include <atomic>
#include <cstdint>
#include <iterator>
#include <string>

namespace Gow3Settings {

enum Upscaler : int { UpscalerOff = 0, UpscalerFsr3 = 1, UpscalerFsr4 = 2, UpscalerFsr411 = 3,
                      UpscalerTaa = 4, UpscalerDlss = 5, UpscalerCount };
/// FSR 4 v07 or FSR 4.1.1: the same inputs, settings and placement in the frame.
inline bool IsFsr4(int upscaler) {
    return upscaler == UpscalerFsr4 || upscaler == UpscalerFsr411;
}
enum Preset : int { NativeAA = 0, Quality, Balanced, Performance, UltraPerformance, PresetCount };
enum DebugView : int { DebugNone = 0, DebugReactive = 1, DebugMotion = 2, DebugViewCount };

/// Live output resolutions: the upscaler's output and the UI host targets.
inline constexpr int OutputWidths[] = {1280, 1920, 2560, 3840};
inline constexpr int OutputHeights[] = {720, 1080, 1440, 2160};
inline constexpr int OutputCount = 4;
inline constexpr int OutputDefault = 1; ///< 1920x1080, the game's own size

struct Values {
    std::atomic<int> upscaler{UpscalerFsr3};
    std::atomic<int> preset{NativeAA};
    std::atomic<bool> sharpen{true};
    std::atomic<float> sharpness{0.3f};
    std::atomic<bool> jitter{true};
    std::atomic<bool> reactive{false};
    std::atomic<bool> object_motion{true};
    std::atomic<float> reactive_scale{1.0f};
    std::atomic<float> reactive_threshold{0.2f};
    std::atomic<float> reactive_max{0.9f};
    std::atomic<int> debug_view{DebugNone};
    std::atomic<bool> show_fps{false};
    /// Performance overlay (gow3_perf_stats.h): items shown, font size and background opacity
    /// in percent, corner (0 top left, 1 top right, 2 bottom left, 3 bottom right), layout
    /// (0 one line, 1 one item per line). menu_tab: the menu's last tab.
    std::atomic<unsigned> overlay_items{7};
    std::atomic<int> overlay_scale{100}, overlay_corner{1}, overlay_opacity{50}, overlay_layout{0};
    std::atomic<int> menu_tab{0};
    std::atomic<float> red_orb_multiplier{1.0f};
    std::atomic<bool> red_orbs_supported{false};
    /// Damage and orb pickup multipliers (src/damage_hook.h, src/orb_hook.h), 0.1x to 100x.
    /// The game's code reads these floats directly.
    std::atomic<float> damage_dealt{1.0f}, damage_taken{1.0f};
    std::atomic<float> green_orb_multiplier{1.0f}, blue_orb_multiplier{1.0f}, gold_orb_multiplier{1.0f};
    std::atomic<bool> damage_supported{false}, orb_pickup_supported{false};
    std::atomic<bool> cheats[GOW3_CHEAT_COUNT]{};
    std::atomic<uint32_t> cheats_supported{0};
    /// Damage hook (src/actor_hook.h): enemy health bar, off by default.
    std::atomic<bool> enemy_health_bar{false};
    std::atomic<bool> actor_watch_supported{false};
    // Graphics (gow3_graphics.h). Live: applied on the next frame.
    std::atomic<int> display_mode{0};
    std::atomic<bool> vsync{true};
    std::atomic<int> fps_limit{-3}; ///< Gow3Graphics::FpsLimits: one under half the refresh
    /// The display uses G-Sync / FreeSync: the automatic frame limits stay below the refresh
    /// (fixed refresh: exactly the refresh or half of it).
    std::atomic<bool> display_vrr{true};
    std::atomic<bool> async_shaders{true};
    std::atomic<bool> fsr1{false};
    std::atomic<bool> rcas{true};
    std::atomic<int> rcas_strength{75};
    /// Set by the presenter: sharpening ran on the last frame (it needs FSR 1 when the game's
    /// image is smaller than the window).
    std::atomic<bool> rcas_applied{false};
    /// gow3 U3 / U4 (ini ue4_dlss, GOW3_UE4_DLSS): NVIDIA DLSS in place of UE4's TAA, live (menu,
    /// F9). Needs gow3_dlss.dll and nvngx_dlss.dll beside the exe and an RTX GPU.
    std::atomic<bool> ue4_dlss{false};
    /// Set at start: the DLSS DLLs are there and NGX runs on this GPU.
    std::atomic<bool> ue4_dlss_available{false};
    /// Set by the presenter: the game's image and the window it is shown in, in pixels.
    std::atomic<int> image_width{0}, image_height{0}, window_width{0}, window_height{0};
    std::atomic<int> frames_queued{1};
    // Startup: run.py turns them into patches and environment variables; "Apply and restart".
    std::atomic<int> render_resolution{0};
    std::atomic<int> engine_fps{60};
    std::atomic<bool> deferred_readback{true};
    std::atomic<bool> stale_readback{true};
    /// Set before an "Apply and restart"; cleared once the new launch shows the game. run.py
    /// restores the previous startup options when a launch never cleared it.
    std::atomic<bool> restart_unconfirmed{false};
    int startup_render_resolution = 0, startup_engine_fps = 60;
    bool startup_deferred_readback = true, startup_stale_readback = true;
    /// Performance patches (PerfKeys): on by default; live ones apply at once, the others with
    /// "Apply and restart" (run.py: graphics_options.PERF_SWITCHES).
    std::atomic<bool> perf[12] = {true, true, true, true, true, true, true, true, true, true, true, true};
    bool startup_perf[12] = {true, true, true, true, true, true, true, true, true, true, true, true};
    /// MediEvil graphics quality and output resolution (UE4QualityKeys / UE4OutputKeys; -1: a
    /// value the menu does not list, such as a custom WxH, kept as written). On restart (run.py:
    /// graphics_options, ue4_override).
    int ue4_quality = 2, output_resolution = 4;
    int startup_ue4_quality = 2, startup_output_resolution = 4;
    std::string ue4_quality_raw = "console", output_resolution_raw = "1920x1080";
    /// MediEvil render scale in percent of the output per side (medievil.ini render_scale,
    /// ParseRenderScale): the value chosen, the one this launch started at (run.py:
    /// ue4_override's r.ScreenPercentage) and the one the game draws at now. Live up to
    /// LiveRenderScaleMax (RequestRenderScale); a larger value applies at the next start.
    int render_scale = 100, startup_render_scale = 100, live_render_scale = 100;
    /// medievil.ini ue4_aa is "taa" (the default). Not written by the menu: only its notes and
    /// RestartChanges read it (FXAA / off: a start below 100% still uses temporal AA).
    bool ue4_aa_temporal = true;
    /// MediEvil effects resolution (UE4EffectsKeys): UE4's separate translucency (enemy death
    /// effects, smoke, fire) at 100 / 75 / 50% of the render size. Live (in-game menu,
    /// runtime_ue4_request_cvar) and at boot (run.py: ue4_override). The death effects of several
    /// enemies cost 2-3 ms of GPU a frame at 4K (N-19).
    int effects_resolution = 1, startup_effects_resolution = 1; // 75% (N-26)
    std::string effects_resolution_raw = "75";
    /// MediEvil: UE4 preshadows (r.Shadow.Preshadows; moving objects' shadows from fixed lights),
    /// switched live through the game's console variable (runtime_ue4_request_cvar). Off: the
    /// heavy Graveyard view stayed at the frame cap (N-25).
    std::atomic<bool> preshadows{true};
    /// MediEvil graphics settings (UE4LiveOptions): the chosen index (-1 while loading: a missing key or
    /// an old "preset", resolved from ue4_quality after the file is read); the ones this launch started
    /// with (RestartChanges: the settings that are not live).
    int ue4_live[18] = {};
    int startup_ue4_live[18] = {};
    // FSR 4 checks (menu): the provider's auto exposure, the jitter sign it is given.
    std::atomic<bool> fsr4_auto_exposure{true};
    std::atomic<bool> fsr4_invert_jitter{false};
    std::atomic<int> active_render_width{1920}, active_render_height{1080};
    std::atomic<int> output_res{OutputDefault}; ///< index into OutputWidths
    /// Live resolution and preset changes (run.sh): 0 off by default (startup patch, fastest
    /// on the Steam Deck and older GPUs), -1 auto (strong discrete GPUs), 1 on. On restart.
    std::atomic<int> live_resolution{0};
    /// Why FSR 4 cannot run (assets, device features), or null. Set by the renderer.
    std::atomic<const char*> fsr4_problem{nullptr};
    std::atomic<bool> fsr4_supported{false}, fsr411_supported{false};
    /// DLSS (gow3_dlss.dll, NVIDIA RTX) is ready, or why not (null before the device exists).
    std::atomic<bool> dlss_supported{false};
    std::atomic<const char*> dlss_problem{nullptr};

    /// Startup settings for the explicit GOW3_RENDER_RES compatibility patch only.
    int startup_preset = NativeAA;
    int startup_upscaler = UpscalerFsr3;
    bool startup_object_motion = true;
    int startup_output_res = OutputDefault;
    int startup_live_resolution = 0;
};

Values& Get();

/// Reads the file, then the environment overrides. Called once at start.
void Load();
/// Checks the loaded choice before the first frame; unsupported FSR 4 uses FSR 3.1.
void ConfigureUpscalerSupport(bool fsr4, bool fsr411);
/// After device creation: DLSS availability; a DLSS setting falls back to FSR 3.1 without it.
void ConfigureDlssSupport(bool available, const char* problem);
/// Startup-patched scene dimensions cannot change until run.sh prepares a new image.
bool FixedRenderSession();
int RenderPreset();
bool ResolutionNeedsRestart();
/// A startup graphics option differs from the one this launch started with.
bool GraphicsNeedRestart();
/// Performance patches (medievil.ini perf_*; launcher > Performance).
enum PerfPatch {
    PerfHzbOcclusion, PerfNoDecalPrepass, PerfParticleSinglePass, PerfSmallBuffers, PerfTextureMemo,
    PerfLockfreeMemory, PerfCheapFrees, PerfGpuDataNoWaits, Perf4kMemory, PerfLeanBarriers, PerfLeanBinding,
    PerfRingReuse, PerfCount
};
inline constexpr const char* PerfKeys[PerfCount] = {
    "perf_hzb_occlusion", "perf_no_decal_prepass", "perf_particle_single_pass", "perf_small_buffers",
    "perf_texture_memo", "perf_lockfree_memory", "perf_cheap_frees", "perf_gpu_data_no_waits",
    "perf_4k_memory", "perf_lean_barriers", "perf_lean_binding", "perf_ring_reuse"};
/// Switched while the game runs (the others need a restart). The single particle pass through its
/// console variable (src/runtime_extra.c: runtime_ue4_particle_single_pass, 01.00 and 01.02).
inline constexpr bool PerfLive[PerfCount] = {false, false, true, true, false, true, true, false, false, false, true, true};
/// Whether a live patch is on now (read by the draw thread for every draw).
inline bool PerfOn(PerfPatch patch) {
    return Get().perf[patch].load(std::memory_order_relaxed);
}
inline constexpr const char* UE4QualityKeys[] = {"ultra", "high", "console", "medium", "low"};
inline constexpr const char* UE4OutputKeys[] = {"display", "3840x2160", "3200x1800", "2560x1440", "1920x1080",
                                                "1280x720"};
/// MediEvil render scale (medievil.ini render_scale; launcher > Performance, in-game menu > Applied
/// now): any whole percent from RenderScaleMin to RenderScaleMax, the range of UE4 4.21's temporal
/// upsampling (resolution fraction 0.5-2). The ini still starts 25-49 (graphics_options.py
/// parse_render_scale); anything else is 100.
inline constexpr int RenderScaleMin = 50, RenderScaleMax = 200, RenderScaleIniMin = 25;
int ParseRenderScale(const std::string& value);
/// The highest render scale the game can switch to live: the one it started at, or 100% if that was
/// lower. Larger scenes need more memory than the start sized (framebuffer pool, GOW3_DMEM_MB, by the
/// output size; research/docs/impl-live-render-scale.md).
int LiveRenderScaleMax();
/// The lowest one: RenderScaleMin, or the start value if an ini value below it started.
int LiveRenderScaleMin();
/// The chosen render scale needs a new start: above the live range, or (ue4_aa FXAA / off) on the
/// other side of 100% than the start, whose anti-aliasing a start would change.
bool RenderScaleNeedsRestart();
/// Switches the scene to `percent` (clamped to the live range) through r.TemporalAA.Upsampling and
/// r.ScreenPercentage (src/runtime_extra.c), logs "Render scale: A% -> B% (live)"; returns the
/// scale the game draws at now.
int RequestRenderScale(int percent);
/// One side of the scene at `percent` of an output side (as graphics_options.scene_size).
inline int ScaledSize(int size, int percent) {
    return size * percent / 100;
}
inline constexpr const char* UE4EffectsKeys[] = {"100", "75", "50"};
/// MediEvil graphics settings (research/docs/plan-graphics-settings.md): the same table as
/// scripts/graphics_options.py GRAPHICS_SETTINGS (tests/test_launcher_controls.py checks they match).
/// Graphics quality sets every one of them (UE4ApplyLevel); the quality shows the level they all match,
/// else Custom (UE4MatchedLevel). Live ones switch through UE4 console variables at once (in-game menu,
/// src/runtime_extra.c ue4_cvars[]); the others at the next start (run.py writes every setting into the
/// override pak). "(PS4 default)" marks the PS4's own value. Values fit the runtime's 15 characters.
struct UE4LiveOption {
    const char* key;   ///< medievil.ini; its value is one of `tokens`
    const char* title;
    const char* hint;
    bool live;            ///< switched at once; else at the next start
    const char* cvars[2]; ///< set together (nullptr: unused)
    int count;            ///< choices
    const char* labels[5];
    const char* tokens[5];
    const char* values[5][2]; ///< per choice, per cvar
    const char* levels[6];    ///< the token of each quality level (UE4QualityKeys order), then Deck Optimized
};
inline constexpr int UE4DeckLevel = 5; ///< levels[] index of Deck Optimized (set by the launcher only)
inline constexpr UE4LiveOption UE4LiveOptions[] = {
    {"shadow_resolution", "Shadow resolution",
     "Shadow maps of torches and other lights. Dynamic: sized by how big the light is on screen, "
     "as on the PS4; a fixed size costs less.",
     true, {"r.Shadow.MinResolution", "r.Shadow.MaxResolution"}, 4,
     {"256", "512", "1024", "Dynamic, up to 2048 (PS4 default)"},
     {"256", "512", "1024", "dynamic"},
     {{"256", "256"}, {"512", "512"}, {"1024", "1024"}, {"32", "2048"}},
     {"dynamic", "dynamic", "dynamic", "512", "256", "256"}},
    {"small_object_shadows", "Small object shadows",
     "Objects smaller than this on screen cast no shadow (fewer shadow draws).",
     false, {"r.Shadow.RadiusThreshold", nullptr}, 4,
     {"Large objects only", "Fewer", "Most", "All (PS4 default)"},
     {"0.15", "0.12", "0.1", "0.01"},
     {{"0.15"}, {"0.12"}, {"0.1"}, {"0.01"}},
     {"0.01", "0.01", "0.01", "0.12", "0.15", "0.15"}},
    {"shadow_distance", "Shadow distance",
     "How far the sun and moon cast shadows and in how many cascades; each cascade draws the "
     "scene again (draw thread and GPU).",
     true, {"r.Shadow.DistanceScale", "r.Shadow.CSM.MaxCascades"}, 3,
     {"Short", "Medium", "Full, up to 10 cascades (PS4 default)"},
     {"short", "medium", "ps4"},
     {{"0.6", "2"}, {"0.8", "2"}, {"1.0", "10"}},
     {"ps4", "ps4", "ps4", "medium", "short", "short"}},
    {"sun_shadow_resolution", "Sun and moon shadow resolution",
     "Size of the sun and moon shadow maps: sharper distant shadows at a GPU cost.",
     false, {"r.Shadow.MaxCSMResolution", nullptr}, 4,
     {"1024", "1536", "2048 (PS4 default)", "4096"},
     {"1024", "1536", "2048", "4096"},
     {{"1024"}, {"1536"}, {"2048"}, {"4096"}},
     {"4096", "2048", "2048", "1536", "1024", "1024"}},
    {"shadow_quality", "Shadow filtering",
     "Softness of shadow edges (filter samples). Mostly GPU.",
     true, {"r.ShadowQuality", nullptr}, 5,
     {"Low", "Medium", "High (PS4 default)", "Very high", "Ultra"},
     {"1", "2", "3", "4", "5"},
     {{"1"}, {"2"}, {"3"}, {"4"}, {"5"}},
     {"5", "4", "3", "2", "1", "1"}},
    {"view_distance", "View distance",
     "How far away objects are still drawn. Lower: fewer draws (draw thread and GPU), objects "
     "disappear sooner.",
     true, {"r.ViewDistanceScale", nullptr}, 5,
     {"Near", "Medium", "Normal (PS4 default)", "Far", "Farthest"},
     {"0.7", "0.85", "1", "1.5", "2"},
     {{"0.7"}, {"0.85"}, {"1"}, {"1.5"}, {"2"}},
     {"2", "1.5", "1", "0.85", "0.7", "0.7"}},
    {"lod_distance", "Level of detail",
     "When models and foliage switch to their simpler versions. Higher: detailed models further "
     "away, more triangles.",
     true, {"r.StaticMeshLODDistanceScale", "foliage.LODDistanceScale"}, 4,
     {"Low", "Normal (PS4 default)", "High", "Ultra"},
     {"low", "ps4", "high", "ultra"},
     {{"1.5", "0.7"}, {"1", "1"}, {"0.75", "1.5"}, {"0.5", "2"}},
     {"ultra", "high", "ps4", "ps4", "ps4", "low"}},
    {"foliage_density", "Foliage and grass density",
     "How much grass and foliage is placed. Lower: fewer draws.",
     false, {"foliage.DensityScale", "grass.DensityScale"}, 2,
     {"60%", "100% (PS4 default)"},
     {"0.6", "1"},
     {{"0.6", "0.6"}, {"1", "1"}},
     {"1", "1", "1", "1", "0.6", "0.6"}},
    {"texture_filtering", "Texture filtering",
     "Anisotropic filtering: sharper textures at a slant.",
     false, {"r.MaxAnisotropy", nullptr}, 2,
     {"8x (PS4 default)", "16x"},
     {"8", "16"},
     {{"8"}, {"16"}},
     {"16", "16", "8", "8", "8", "8"}},
    {"ambient_occlusion", "Ambient occlusion",
     "Contact shadows in corners and creases (screen space). GPU only.",
     true, {"r.AmbientOcclusionLevels", nullptr}, 5,
     {"Off", "Low", "Medium", "High", "Auto (PS4 default)"},
     {"0", "1", "2", "3", "auto"},
     {{"0"}, {"1"}, {"2"}, {"3"}, {"-1"}},
     {"3", "3", "auto", "1", "0", "2"}},
    {"reflections", "Reflections",
     "Screen-space reflections on wet and shiny surfaces. GPU only.",
     true, {"r.SSR.Quality", nullptr}, 5,
     {"Off", "Low", "Medium", "High (PS4 default)", "Ultra"},
     {"0", "1", "2", "3", "4"},
     {{"0"}, {"1"}, {"2"}, {"3"}, {"4"}},
     {"4", "3", "3", "1", "0", "1"}},
    {"refraction", "Refraction",
     "Heat haze and the bending of light through glass and water.",
     false, {"r.RefractionQuality", nullptr}, 2,
     {"Off", "On (PS4 default)"},
     {"0", "2"},
     {{"0"}, {"2"}},
     {"2", "2", "2", "2", "0", "0"}},
    {"volumetric_fog", "Volumetric fog",
     "Light shafts and lit fog. GPU only; off changes the look of misty areas.",
     true, {"r.VolumetricFog", nullptr}, 2,
     {"Off", "On (PS4 default)"},
     {"0", "1"},
     {{"0"}, {"1"}},
     {"1", "1", "1", "1", "0", "1"}},
    {"motion_blur", "Motion blur",
     "Blur of fast camera and object motion.",
     true, {"r.MotionBlurQuality", nullptr}, 3,
     {"Off", "Low", "Full (PS4 default)"},
     {"0", "2", "4"},
     {{"0"}, {"2"}, {"4"}},
     {"4", "4", "4", "2", "0", "0"}},
    {"depth_of_field", "Depth of field",
     "Blur of out-of-focus distances (mostly in cutscenes).",
     true, {"r.DepthOfFieldQuality", nullptr}, 3,
     {"Off", "Low", "Full (PS4 default)"},
     {"0", "1", "2"},
     {{"0"}, {"1"}, {"2"}},
     {"2", "2", "2", "1", "0", "0"}},
    {"bloom", "Bloom",
     "Glow around bright lights and fire.",
     true, {"r.BloomQuality", nullptr}, 3,
     {"Off", "Low", "Full (PS4 default)"},
     {"0", "2", "5"},
     {{"0"}, {"2"}, {"5"}},
     {"5", "5", "5", "5", "2", "2"}},
    {"lens_flares", "Lens flares",
     "Flares from bright lights.",
     true, {"r.LensFlareQuality", nullptr}, 3,
     {"Off", "Low", "Full (PS4 default)"},
     {"0", "1", "2"},
     {{"0"}, {"1"}, {"2"}},
     {"2", "2", "2", "2", "0", "0"}},
    {"chromatic_aberration", "Chromatic aberration",
     "Colour fringes towards the screen edges.",
     true, {"r.SceneColorFringeQuality", nullptr}, 2,
     {"Off", "On (PS4 default)"},
     {"0", "1"},
     {{"0"}, {"1"}},
     {"1", "1", "1", "1", "0", "0"}},
};
inline constexpr int UE4LiveCount = int(std::size(UE4LiveOptions));
static_assert(UE4LiveCount == int(sizeof(Values::ue4_live) / sizeof(int)), "Values::ue4_live: one per UE4LiveOptions");
/// Requests option `o`'s cvars for choice `choice` (live options only; the others apply at the next start).
void RequestUE4Live(int o, int choice);
/// The choice index quality level `level` (UE4QualityKeys, or UE4DeckLevel) gives option `o`.
int UE4LevelChoice(int o, int level);
/// The level every option matches, `picked` first (0-4, UE4DeckLevel), or -1 (Custom).
int UE4MatchedLevel(const Values& v, int picked);
/// Graphics quality picked in the menu: every option set to the level's choice, the live ones at once.
void UE4ApplyLevel(int level);

/// Startup options changed since this launch (applied by "Apply and restart").
int RestartChanges();

/// Writes the settings changed since the last load or save into the file, in place.
void Save();
/// Records the current values as unchanged (Load calls it after the environment overrides).
void TakeBaseline();

/// Render resolution divisor of a preset (1.0 native, 1.5 quality, ...).
float PresetScale(int preset);
const char* PresetName(int preset);
const char* UpscalerName(int upscaler);

} // namespace Gow3Settings
