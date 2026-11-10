#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
"""Startup graphics options of medievil.ini (gpu/shim/gow3_graphics.h) for run.py.

The in-game menu and the launcher write them; run.py turns them into game patches and
environment variables at every start, so "Apply and restart" uses the new values. After a
restart the game confirms them (restart_unconfirmed=0) once it shows frames; when a launch
never did, the next start goes back to the last confirmed values.
"""
import json
from pathlib import Path

RESOLUTIONS = ('native', '480p', '720p', '1440p', '1800p', '4K',
               '720p_native', '1440p_native', '1800p_native', '4K_native')
ENGINE_FPS = (60, 120, 240)
# MediEvil (Unreal Engine 4): the game's output size and the 3D scene's scale of it (two settings
# since 2026-10-25; render_resolution and ue4_upscale before, see migrate_resolution).
# output_resolution: WxH (any size from 640x360 to 7680x4320) or 'display' (the desktop's size).
# The game boots at it (patches/MediEvil.xml "Boot display size", research: T-R1, T-R2).
OUTPUT_PRESETS = ('1280x720', '1920x1080', '2560x1440', '3200x1800', '3840x2160', 'display')
# render_scale: percent of the output per axis. Below 100 UE4's temporal upsampling rebuilds the
# output from its motion vectors; above 100 the scene is drawn larger and scaled down (supersampling).
# RENDER_SCALES: the presets; the launcher and the in-game menu take any whole percent in
# RENDER_SCALE_RANGE (UE4 4.21's temporal upsampling range; gpu/shim/gow3_settings.h RenderScaleMin /
# RenderScaleMax). The ini still starts 25-49 (parse_render_scale). In the game the scale switches live
# within live_render_scale_range (research/docs/impl-live-render-scale.md).
RENDER_SCALES = (50, 58, 67, 100, 133, 167, 200)
RENDER_SCALE_RANGE = (50, 200)
DEFAULTS = {'render_resolution': 'native', 'engine_fps': '60', 'deferred_readback': '1',
            'stale_readback': '1', 'async_shaders': '1', 'output_resolution': '1920x1080',
            'render_scale': '100'}
RESOLUTION_PREFIX = 'Resolution Patch'
FPS_PREFIX = 'Frame Rate Patch'
TEXTURE_FIX = 'Bug Fix - Texture Corruption Fix'


def read_ini(path):
    values = {}
    try:
        lines = Path(path).read_text(encoding='utf-8').splitlines()
    except OSError:
        return values
    for line in lines:
        if '=' in line and not line.lstrip().startswith('#'):
            key, value = line.split('=', 1)
            values[key.strip()] = value.strip()
    return values


def write_keys(path, updates):
    """Rewrites `updates` in place and appends missing keys, keeping every other line."""
    path = Path(path)
    try:
        lines = path.read_text(encoding='utf-8').splitlines()
    except OSError:
        lines = ['# MediEvil settings (in-game menu: Insert / R3+L2)']
    done, out = set(), []
    for line in lines:
        key = line.split('=', 1)[0].strip() if '=' in line and not line.lstrip().startswith('#') else None
        if key in updates:
            out.append(f'{key}={updates[key]}')
            done.add(key)
        else:
            out.append(line)
    out += [f'{k}={v}' for k, v in updates.items() if k not in done]
    path.write_text('\n'.join(out) + '\n', encoding='utf-8')


def parse_output(value):
    """'display' or a normalized 'WxH' (even sizes, 640x360 to 7680x4320); else 1920x1080."""
    value = str(value).strip().lower()
    if value == 'display':
        return value
    width, sep, height = value.partition('x')
    try:
        w, h = int(width), int(height)
    except ValueError:
        return '1920x1080'
    if not sep or not (640 <= w <= 7680 and 360 <= h <= 4320):
        return '1920x1080'
    return f'{w & ~1}x{h & ~1}'


def parse_render_scale(value):
    """A render scale in percent from 25 to 200; else 100."""
    try:
        scale = int(float(value))
    except (TypeError, ValueError):
        return 100
    return scale if 25 <= scale <= 200 else 100


def scene_size(width, height, scale):
    """(width, height) of the 3D scene of a WxH output at `scale` percent."""
    return width * scale // 100, height * scale // 100


def live_render_scale_range(startup):
    """(lowest, highest) render scale the game can switch to live after starting at `startup`: down to
    50% (or the start value if lower), up to the start value or 100% if that is lower. A larger scene
    needs more memory than the start set up (framebuffer pool and GOW3_DMEM_MB follow the output
    size, output_environment); it applies at the next start (gow3_settings.cpp RequestRenderScale)."""
    return min(startup, RENDER_SCALE_RANGE[0]), max(startup, 100)


def display_size():
    """(width, height) of the primary display's current mode in pixels; 1920x1080 when unknown."""
    try:
        import ctypes
        from ctypes import wintypes

        class DEVMODEW(ctypes.Structure):
            _fields_ = [('dmDeviceName', wintypes.WCHAR * 32), ('dmSpecVersion', wintypes.WORD),
                        ('dmDriverVersion', wintypes.WORD), ('dmSize', wintypes.WORD),
                        ('dmDriverExtra', wintypes.WORD), ('dmFields', wintypes.DWORD),
                        ('dmPosition', wintypes.LONG * 2), ('dmDisplayOrientation', wintypes.DWORD),
                        ('dmDisplayFixedOutput', wintypes.DWORD), ('dmColor', ctypes.c_short),
                        ('dmDuplex', ctypes.c_short), ('dmYResolution', ctypes.c_short),
                        ('dmTTOption', ctypes.c_short), ('dmCollate', ctypes.c_short),
                        ('dmFormName', wintypes.WCHAR * 32), ('dmLogPixels', wintypes.WORD),
                        ('dmBitsPerPel', wintypes.DWORD), ('dmPelsWidth', wintypes.DWORD),
                        ('dmPelsHeight', wintypes.DWORD), ('dmDisplayFlags', wintypes.DWORD),
                        ('dmDisplayFrequency', wintypes.DWORD)]
        mode = DEVMODEW()
        mode.dmSize = ctypes.sizeof(DEVMODEW)
        # ENUM_CURRENT_SETTINGS: physical pixels, whatever the Windows display scaling.
        if ctypes.windll.user32.EnumDisplaySettingsW(None, -1, ctypes.byref(mode)) and mode.dmPelsWidth:
            return int(mode.dmPelsWidth), int(mode.dmPelsHeight)
    except (AttributeError, OSError, ValueError):
        pass
    return 1920, 1080


def output_size(value):
    """(width, height) of an output_resolution value."""
    value = parse_output(value)
    if value == 'display':
        return tuple(int(v) for v in parse_output('%dx%d' % display_size()).split('x'))
    return tuple(int(v) for v in value.split('x'))


POOL_PATCH = 'Native 4K framebuffer pool (512 MB)'
BOOT_4K_PATCH = '4K display metrics at boot'
BOOT_SIZE_PATCH = 'Boot display size'


def output_environment(env, width, height):
    """Sets env (GOW3_PATCHES, GOW3_INTERNAL_WIDTH/HEIGHT, GOW3_DMEM_MB) for a WxH MediEvil output.

    The game boots at that size: a VideoOut of that size and the default rect of UE4's display
    metrics and work area (patches/MediEvil.xml "Boot display size", research: T-R1, T-R2,
    study-boot-resolution.md). A live r.SetRes instead leaves the menus' navigation at the 1080p
    area (I1); GOW3_UE4_BOOT_SIZE=0: that live change, as before."""
    output = f'{width}x{height}'
    selected = [n for n in env.get('GOW3_PATCHES', '').split(';') if n]
    if width * height > 1920 * 1080:
        # Above 1080p, whatever the launcher's Patches tab says: the 512 MB framebuffer pool (the
        # stock 128 MB ran out at 4K and crashed; its direct memory requirement follows from the
        # patch notes), and 2 GiB more direct memory, which the game takes whole into its memory
        # pool for GPU and CPU memory (T-S2, 2026-10-19; research/docs/fix-inventory.md, Caveats).
        # GOW3_PS4_MEMORY=1: the PS4's size.
        selected.append(POOL_PATCH)
        if env.get('GOW3_PS4_MEMORY') != '1':
            env.setdefault('GOW3_DMEM_MB', '7104')
    if output == '1920x1080':
        pass  # the game's own size
    elif env.get('GOW3_UE4_BOOT_SIZE', '1') != '1':
        env.setdefault('GOW3_UE4_SETRES', output)
    elif output == '3840x2160' and env.get('GOW3_UE4_BOOT_4K', '1') == '1':
        # 4K as a PS4 Pro on a 4K TV: a 3840x2160 VideoOut and the game's own 4K display-metrics
        # branch (verified first, ebf85f4). GOW3_UE4_BOOT_4K=0: the boot rect below instead.
        env.setdefault('GOW3_INTERNAL_WIDTH', '3840')
        env.setdefault('GOW3_INTERNAL_HEIGHT', '2160')
        selected.append(BOOT_4K_PATCH)
    else:
        env.setdefault('GOW3_INTERNAL_WIDTH', str(width))
        env.setdefault('GOW3_INTERNAL_HEIGHT', str(height))
        # Without "4K display metrics at boot": its 3840x2160 would win over the rect.
        selected = [n for n in selected if n != BOOT_4K_PATCH] + [f'{BOOT_SIZE_PATCH} {output}']
    env['GOW3_PATCHES'] = ';'.join(dict.fromkeys(selected))
    return env


HZB_FULL_SIZE_PATCH = 'Finer occlusion culling (low render resolutions, Steam Deck)'


# Scenes up to this size get the "Finer occlusion culling" patch by default: their stock HZB is at
# most 1024x512 and culls about half of what the full-size one does (Steam Deck, 1280x720 at 67%:
# twice the objects hidden, -7% draws, +4.6% FPS, no pop-in; plan-linux-deck.md). Larger scenes
# already get a fine HZB, and at 4K the patch would build a 4096x4096 pyramid for nothing.
HZB_FULL_SIZE_AUTO_MAX = (1280, 800)


def hzb_full_size_environment(env, width=0, height=0, render_scale=100):
    """GOW3_HZB_FULL_SIZE=1 adds the "Finer occlusion culling" patch (HZB at the scene's size,
    research/docs/study-ue4-draw-reduction.md section 4) to GOW3_PATCHES, =0 removes it; unset,
    it is added when the startup scene (output size x render scale) is at most
    HZB_FULL_SIZE_AUTO_MAX, else the launcher's Patches page decides. A binary patch: a render
    scale changed in the game keeps the startup choice until the next start."""
    switch = env.get('GOW3_HZB_FULL_SIZE')
    selected = [n for n in env.get('GOW3_PATCHES', '').split(';') if n and n != HZB_FULL_SIZE_PATCH]
    if switch not in ('0', '1'):
        scene = scene_size(width, height, render_scale)
        if not width or scene[0] > HZB_FULL_SIZE_AUTO_MAX[0] or scene[1] > HZB_FULL_SIZE_AUTO_MAX[1] or                 HZB_FULL_SIZE_PATCH in env.get('GOW3_PATCHES', '').split(';'):
            return env
        print(f'Finer occlusion culling: on (scene {scene[0]}x{scene[1]}, at most '
              f'{HZB_FULL_SIZE_AUTO_MAX[0]}x{HZB_FULL_SIZE_AUTO_MAX[1]}; GOW3_HZB_FULL_SIZE=0: off)', flush=True)
        switch = '1'
    if switch == '1':
        selected.append(HZB_FULL_SIZE_PATCH)
    env['GOW3_PATCHES'] = ';'.join(selected)
    return env


PARTICLE_SINGLE_FIXED_PATCH = 'GPU particle fixed step in the single pass'


def particle_single_fixed_environment(env):
    """GOW3_PARTICLE_SINGLE_FIXED=1 adds the "GPU particle fixed step in the single pass" patch
    (the single GPU particle pass keeps the 1/30 s step of the emitters that ask for it,
    research/docs/study-particle-fixed-step.md) to GOW3_PATCHES, =0 removes it; unset, the
    launcher's Patches page decides (off by default)."""
    switch = env.get('GOW3_PARTICLE_SINGLE_FIXED')
    if switch not in ('0', '1'):
        return env
    selected = [n for n in env.get('GOW3_PATCHES', '').split(';') if n and n != PARTICLE_SINGLE_FIXED_PATCH]
    if switch == '1':
        selected.append(PARTICLE_SINGLE_FIXED_PATCH)
    env['GOW3_PATCHES'] = ';'.join(selected)
    return env


NAV_PURGE_PATCH = 'Navigation path purge without fences (CPU)'


def nav_purge_environment(env):
    """GOW3_NAV_PURGE_NOFENCE=1 adds the "Navigation path purge without fences" patch (the game
    thread's per-frame ActivePaths purge without its two mfences per entry,
    research/docs/impl-cpu-waste.md) to GOW3_PATCHES, =0 removes it; unset, the launcher's
    Patches page decides (off by default)."""
    switch = env.get('GOW3_NAV_PURGE_NOFENCE')
    if switch not in ('0', '1'):
        return env
    selected = [n for n in env.get('GOW3_PATCHES', '').split(';') if n and n != NAV_PURGE_PATCH]
    if switch == '1':
        selected.append(NAV_PURGE_PATCH)
    env['GOW3_PATCHES'] = ';'.join(selected)
    return env


# The single render_resolution of earlier versions: output size, scene scale.
LEGACY_RESOLUTION = {'native': ('1920x1080', 100), '480p': ('1920x1080', 50), '720p': ('1920x1080', 67),
                     '1440p': ('1920x1080', 133), '1800p': ('1920x1080', 167), '4K': ('1920x1080', 200),
                     '720p_native': ('1280x720', 100), '1440p_native': ('2560x1440', 100),
                     '1800p_native': ('3200x1800', 100), '4K_native': ('3840x2160', 100)}
LEGACY_UPSCALE = {'quality': 67, 'balanced': 58, 'performance': 50}


def migrate_resolution(ini):
    """{output_resolution, render_scale} from the older render_resolution / ue4_upscale keys when
    the ini has no output_resolution yet; else {}."""
    if 'output_resolution' in ini or ('render_resolution' not in ini and 'ue4_upscale' not in ini):
        return {}
    output, scale = LEGACY_RESOLUTION.get(ini.get('render_resolution', 'native'), ('1920x1080', 100))
    if scale == 100:
        scale = LEGACY_UPSCALE.get(ini.get('ue4_upscale', 'off'), 100)
    return {'output_resolution': output, 'render_scale': str(scale)}


# Graphics quality and its settings (research/docs/plan-graphics-settings.md, the user's design 2026-11-08).
# Graphics quality is the only preset: picking a level sets every setting below to the level's choice;
# the quality shows the level all settings match, else Custom. Every difference between levels is a
# setting here, so nothing is hidden. "(PS4 default)" marks the value the PS4 itself runs (Console).
# The in-game menu has the same table (gpu/shim/gow3_settings.h UE4LiveOptions; tests check they
# match). (key, title, hint, live, cvars, [(token, label, values per cvar)], {level: token}); live ones
# switch in the game at once, the others at the next start (the override pak only).
QUALITY_LEVELS = ('ultra', 'high', 'console', 'medium', 'low')
DECK_LEVEL = 'deck'
GRAPHICS_SETTINGS = [
    ('shadow_resolution', 'Shadow resolution',
     'Shadow maps of torches and other lights. Dynamic: sized by how big the light is on screen, as on the PS4; '
     'a fixed size costs less.', True, ('r.Shadow.MinResolution', 'r.Shadow.MaxResolution'),
     [('256', '256', ('256', '256')), ('512', '512', ('512', '512')), ('1024', '1024', ('1024', '1024')),
      ('dynamic', 'Dynamic, up to 2048 (PS4 default)', ('32', '2048'))],
     {'ultra': 'dynamic', 'high': 'dynamic', 'console': 'dynamic', 'medium': '512', 'low': '256', 'deck': '256'}),
    ('small_object_shadows', 'Small object shadows',
     'Objects smaller than this on screen cast no shadow (fewer shadow draws).', False, ('r.Shadow.RadiusThreshold',),
     [('0.15', 'Large objects only', ('0.15',)), ('0.12', 'Fewer', ('0.12',)), ('0.1', 'Most', ('0.1',)),
      ('0.01', 'All (PS4 default)', ('0.01',))],
     {'ultra': '0.01', 'high': '0.01', 'console': '0.01', 'medium': '0.12', 'low': '0.15', 'deck': '0.15'}),
    ('shadow_distance', 'Shadow distance',
     'How far the sun and moon cast shadows and in how many cascades; each cascade draws the scene again '
     '(draw thread and GPU).', True, ('r.Shadow.DistanceScale', 'r.Shadow.CSM.MaxCascades'),
     [('short', 'Short', ('0.6', '2')), ('medium', 'Medium', ('0.8', '2')),
      ('ps4', 'Full, up to 10 cascades (PS4 default)', ('1.0', '10'))],
     {'ultra': 'ps4', 'high': 'ps4', 'console': 'ps4', 'medium': 'medium', 'low': 'short', 'deck': 'short'}),
    ('sun_shadow_resolution', 'Sun and moon shadow resolution',
     'Size of the sun and moon shadow maps: sharper distant shadows at a GPU cost.', False,
     ('r.Shadow.MaxCSMResolution',),
     [('1024', '1024', ('1024',)), ('1536', '1536', ('1536',)), ('2048', '2048 (PS4 default)', ('2048',)),
      ('4096', '4096', ('4096',))],
     {'ultra': '4096', 'high': '2048', 'console': '2048', 'medium': '1536', 'low': '1024', 'deck': '1024'}),
    ('shadow_quality', 'Shadow filtering', 'Softness of shadow edges (filter samples). Mostly GPU.', True,
     ('r.ShadowQuality',),
     [('1', 'Low', ('1',)), ('2', 'Medium', ('2',)), ('3', 'High (PS4 default)', ('3',)), ('4', 'Very high', ('4',)),
      ('5', 'Ultra', ('5',))],
     {'ultra': '5', 'high': '4', 'console': '3', 'medium': '2', 'low': '1', 'deck': '1'}),
    ('view_distance', 'View distance',
     'How far away objects are still drawn. Lower: fewer draws (draw thread and GPU), objects disappear sooner.',
     True, ('r.ViewDistanceScale',),
     [('0.7', 'Near', ('0.7',)), ('0.85', 'Medium', ('0.85',)), ('1', 'Normal (PS4 default)', ('1',)),
      ('1.5', 'Far', ('1.5',)), ('2', 'Farthest', ('2',))],
     {'ultra': '2', 'high': '1.5', 'console': '1', 'medium': '0.85', 'low': '0.7', 'deck': '0.7'}),
    ('lod_distance', 'Level of detail',
     'When models and foliage switch to their simpler versions. Higher: detailed models further away, more '
     'triangles.', True, ('r.StaticMeshLODDistanceScale', 'foliage.LODDistanceScale'),
     [('low', 'Low', ('1.5', '0.7')), ('ps4', 'Normal (PS4 default)', ('1', '1')), ('high', 'High', ('0.75', '1.5')),
      ('ultra', 'Ultra', ('0.5', '2'))],
     {'ultra': 'ultra', 'high': 'high', 'console': 'ps4', 'medium': 'ps4', 'low': 'ps4', 'deck': 'low'}),
    ('foliage_density', 'Foliage and grass density', 'How much grass and foliage is placed. Lower: fewer draws.',
     False, ('foliage.DensityScale', 'grass.DensityScale'),
     [('0.6', '60%', ('0.6', '0.6')), ('1', '100% (PS4 default)', ('1', '1'))],
     {'ultra': '1', 'high': '1', 'console': '1', 'medium': '1', 'low': '0.6', 'deck': '0.6'}),
    ('texture_filtering', 'Texture filtering', 'Anisotropic filtering: sharper textures at a slant.', False,
     ('r.MaxAnisotropy',),
     [('8', '8x (PS4 default)', ('8',)), ('16', '16x', ('16',))],
     {'ultra': '16', 'high': '16', 'console': '8', 'medium': '8', 'low': '8', 'deck': '8'}),
    ('ambient_occlusion', 'Ambient occlusion', 'Contact shadows in corners and creases (screen space). GPU only.',
     True, ('r.AmbientOcclusionLevels',),
     [('0', 'Off', ('0',)), ('1', 'Low', ('1',)), ('2', 'Medium', ('2',)), ('3', 'High', ('3',)),
      ('auto', 'Auto (PS4 default)', ('-1',))],
     {'ultra': '3', 'high': '3', 'console': 'auto', 'medium': '1', 'low': '0', 'deck': '2'}),
    ('reflections', 'Reflections', 'Screen-space reflections on wet and shiny surfaces. GPU only.', True,
     ('r.SSR.Quality',),
     [('0', 'Off', ('0',)), ('1', 'Low', ('1',)), ('2', 'Medium', ('2',)), ('3', 'High (PS4 default)', ('3',)),
      ('4', 'Ultra', ('4',))],
     {'ultra': '4', 'high': '3', 'console': '3', 'medium': '1', 'low': '0', 'deck': '1'}),
    ('refraction', 'Refraction', 'Heat haze and the bending of light through glass and water.', False,
     ('r.RefractionQuality',),
     [('0', 'Off', ('0',)), ('2', 'On (PS4 default)', ('2',))],
     {'ultra': '2', 'high': '2', 'console': '2', 'medium': '2', 'low': '0', 'deck': '0'}),
    ('volumetric_fog', 'Volumetric fog', 'Light shafts and lit fog. GPU only; off changes the look of misty areas.',
     True, ('r.VolumetricFog',),
     [('0', 'Off', ('0',)), ('1', 'On (PS4 default)', ('1',))],
     {'ultra': '1', 'high': '1', 'console': '1', 'medium': '1', 'low': '0', 'deck': '1'}),
    ('motion_blur', 'Motion blur', 'Blur of fast camera and object motion.', True, ('r.MotionBlurQuality',),
     [('0', 'Off', ('0',)), ('2', 'Low', ('2',)), ('4', 'Full (PS4 default)', ('4',))],
     {'ultra': '4', 'high': '4', 'console': '4', 'medium': '2', 'low': '0', 'deck': '0'}),
    ('depth_of_field', 'Depth of field', 'Blur of out-of-focus distances (mostly in cutscenes).', True,
     ('r.DepthOfFieldQuality',),
     [('0', 'Off', ('0',)), ('1', 'Low', ('1',)), ('2', 'Full (PS4 default)', ('2',))],
     {'ultra': '2', 'high': '2', 'console': '2', 'medium': '1', 'low': '0', 'deck': '0'}),
    ('bloom', 'Bloom', 'Glow around bright lights and fire.', True, ('r.BloomQuality',),
     [('0', 'Off', ('0',)), ('2', 'Low', ('2',)), ('5', 'Full (PS4 default)', ('5',))],
     {'ultra': '5', 'high': '5', 'console': '5', 'medium': '5', 'low': '2', 'deck': '2'}),
    ('lens_flares', 'Lens flares', 'Flares from bright lights.', True, ('r.LensFlareQuality',),
     [('0', 'Off', ('0',)), ('1', 'Low', ('1',)), ('2', 'Full (PS4 default)', ('2',))],
     {'ultra': '2', 'high': '2', 'console': '2', 'medium': '2', 'low': '0', 'deck': '0'}),
    ('chromatic_aberration', 'Chromatic aberration', 'Colour fringes towards the screen edges.', True,
     ('r.SceneColorFringeQuality',),
     [('0', 'Off', ('0',)), ('1', 'On (PS4 default)', ('1',))],
     {'ultra': '1', 'high': '1', 'console': '1', 'medium': '1', 'low': '0', 'deck': '0'}),
]
GRAPHICS_KEYS = tuple(key for key, *_rest in GRAPHICS_SETTINGS)
# Tokens of the settings before 2026-11-08 that mean another choice now ("preset" followed the quality).
LEGACY_TOKENS = {'shadow_resolution': {'2048': 'dynamic'},
                 'shadow_distance': {'console': 'medium', 'far': 'ps4', 'farthest': 'ps4'},
                 'lod_distance': {'console': 'ps4'}}
# Every quality level wrote shadow caching on (upstream c2ad62e); kept, it is no setting (the same at every level).
ALWAYS_CVARS = {'r.Shadow.CacheWholeSceneShadows': '1'}


def level_choices(level):
    """{key: token} of a quality level (or 'deck'); an unknown level (diagnostic presets) reads as console."""
    level = level if level in (*QUALITY_LEVELS, DECK_LEVEL) else 'console'
    return {key: levels[level] for key, *_rest, levels in GRAPHICS_SETTINGS}


def graphics_choices(ini):
    """{key: token} of every setting from medievil.ini: older tokens translated, "preset" and missing keys
    resolved from ue4_quality, unknown values too."""
    level = level_choices(ini.get('ue4_quality', 'console'))
    out = {}
    for key, _title, _hint, _live, _cvars, choices, _levels in GRAPHICS_SETTINGS:
        token = ini.get(key, 'preset')
        token = LEGACY_TOKENS.get(key, {}).get(token, token)
        out[key] = token if token in {t for t, _label, _values in choices} else level[key]
    return out


def graphics_migration(ini):
    """The ini keys to write so that every setting holds a token of today's table; {} when it does."""
    choices = graphics_choices(ini)
    return {key: token for key, token in choices.items() if ini.get(key) != token}


def matched_level(choices, picked='console'):
    """The quality level (or 'deck') whose choices all settings have, the picked one first; '' (Custom) if
    none."""
    for level in (picked, *QUALITY_LEVELS, DECK_LEVEL):
        if level in (*QUALITY_LEVELS, DECK_LEVEL) and level_choices(level) == choices:
            return level
    return ''


def graphics_cvars(choices):
    """The console variables of the settings' choices (the override pak writes them at start)."""
    out = dict(ALWAYS_CVARS)
    for key, _title, _hint, _live, cvars, options, _levels in GRAPHICS_SETTINGS:
        values = next(v for t, _label, v in options if t == choices[key])
        out.update(zip(cvars, values))
    return out


def startup_options(ini):
    """The startup options, normalized; unknown values fall back to the defaults."""
    options = {key: ini.get(key, value) for key, value in DEFAULTS.items()}
    options['output_resolution'] = parse_output(options['output_resolution'])
    options['render_scale'] = str(parse_render_scale(options['render_scale']))
    if options['render_resolution'] not in RESOLUTIONS:
        options['render_resolution'] = 'native'
    if options['engine_fps'] not in {str(f) for f in ENGINE_FPS}:
        options['engine_fps'] = '60'
    for key in ('deferred_readback', 'stale_readback', 'async_shaders'):
        options[key] = '0' if options[key] == '0' else '1'
    return options


def prepare(config, state_path):
    """(options, reverted): the startup options of this launch, restoring the last confirmed
    ones when the previous launch after a restart never confirmed its own."""
    ini = read_ini(config)
    if migrated := migrate_resolution(ini):
        write_keys(config, migrated)
        ini.update(migrated)
    if migrated := graphics_migration(ini):  # every graphics setting holds a real value (no "preset")
        write_keys(config, migrated)
        ini.update(migrated)
    state = {}
    try:
        state = json.loads(Path(state_path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        pass
    options = startup_options(ini)
    reverted = False
    if ini.get('restart_unconfirmed') == '1':
        if state.get('attempt') and state.get('last_good'):
            options = startup_options(state['last_good'])
            write_keys(config, {**options, 'restart_unconfirmed': '0'})
            state['attempt'] = False
            reverted = True
        else:
            state['attempt'] = True
    else:
        state = {'last_good': options, 'attempt': False}
    Path(state_path).write_text(json.dumps(state, indent=2), encoding='utf-8')
    return options, reverted


def patch_names(names, options, texture_fix_available=True):
    """The launcher's patch list with the resolution and frame rate patches of `options`."""
    names = [n for n in names if not n.startswith(RESOLUTION_PREFIX) and not n.startswith(FPS_PREFIX)]
    resolution = options['render_resolution']
    if resolution != 'native':
        names = [n for n in names if n != TEXTURE_FIX] + [f'{RESOLUTION_PREFIX} - {resolution}']
    elif texture_fix_available and TEXTURE_FIX not in names:
        names.insert(0, TEXTURE_FIX)
    if options['engine_fps'] != '60':
        names.append(f'{FPS_PREFIX} - {options["engine_fps"]} FPS')
    return names


# Performance patches (launcher > Performance, in-game menu > Performance): medievil.ini key ->
# the environment that switches the patch off (perf_*=0). Missing keys mean on (PERF_DEFAULT_OFF: off).
PERF_SWITCHES = {
    'perf_hzb_occlusion': {'GOW3_HZB_REMAP': '0'},
    'perf_no_decal_prepass': {'GOW3_DECAL_PREPASS': '1'},
    'perf_particle_single_pass': {'GOW3_PARTICLE_FIXED_STEP': '1'},
    'perf_small_buffers': {'GOW3_STREAM_CLEAN': '0'},
    'perf_texture_memo': {'GOW3_TEXSET_REGIONS': '0', 'GOW3_TEXSET_TWO_WAY': '0'},
    'perf_lockfree_memory': {'GOW3_OPTIMISTIC_CLAMP': '0', 'GOW3_REGION_BUCKETS': '0'},
    'perf_cheap_frees': {'GOW3_DECOMMIT_KEEP_ACCESS': '0'},
    'perf_gpu_data_no_waits': {'GOW3_READBACK_QUEUE': '0', 'GOW3_GPU_WRITE_TWINS': '0',
                               'GOW3_EOS_DEFERRED': '0', 'GOW3_EOP_TARGETED': '0'},
    'perf_4k_memory': {'GOW3_PS4_MEMORY': '1'},
    'perf_lean_barriers': {'GOW3_SKIP_READ_BARRIERS': '0', 'GOW3_DEPTH_SAMPLE_IN_PLACE': '0'},
    'perf_lean_binding': {'GOW3_LEAN_BINDING': '0'},
    'perf_ring_reuse': {'GOW3_RING_REUSE': '0'},
}
# Off where medievil.ini has no key: the single particle pass slows some torches to a crawl and
# leaves an ember / golden stream in Dan's crypt (research/docs/plan-0102.md, T-0102-4).
PERF_DEFAULT_OFF = {'perf_particle_single_pass'}


# Presentation settings (launcher > Performance > Presentation): medievil.ini key -> how its value
# becomes environment variables (run.py; a variable already set wins, for tests). All default off.
PRESENT_MODES = ('auto', 'fifo', 'fifo_relaxed', 'mailbox', 'immediate')
# Defaults: the tested best (plan-stage-b-split.md T-F27: Reflex, sleep before the present, tick 960,
# precise tick, presents on the vblank thread).
PRESENT_DEFAULTS = {'present_limiter': 'reflex', 'present_tick_hz': '960', 'present_precise_tick': '1',
                    'present_inline': '1', 'present_no_frame_caps': '0', 'present_mode': 'auto',
                    'present_frame_trace': '0'}
PRESENT_FLAGS = {
    'present_no_frame_caps': {'GOW3_NO_FRAME_CAP': '1'},
    'present_inline': {'GOW3_PRESENT_THREAD': '0'},
    'present_precise_tick': {'GOW3_PRECISE_TICK': '1'},
}
# Frame limiter (medievil.ini present_limiter). Reflex falls back to the precise limiter where the
# driver has no VK_NV_low_latency2 (AMD, Intel): its variables are set too.
PRESENT_LIMITERS = {
    'reflex': {'GOW3_REFLEX': '1', 'GOW3_REFLEX_SLEEP_BEFORE_PRESENT': '1', 'GOW3_PRECISE_LIMIT': '1',
               'GOW3_LIMIT_HOLD': '1', 'GOW3_PRESENT_PRIORITY': '1'},
    'precise': {'GOW3_PRECISE_LIMIT': '1', 'GOW3_LIMIT_HOLD': '1', 'GOW3_PRESENT_PRIORITY': '1'},
    'standard': {},
}


def present_environment(ini, trace_path=None):
    """The environment of the presentation settings in medievil.ini."""
    env = {}
    value = lambda key: ini.get(key, PRESENT_DEFAULTS[key])
    for key, on in PRESENT_FLAGS.items():
        if value(key) == '1':
            env.update(on)
    env.update(PRESENT_LIMITERS.get(value('present_limiter'), {}))
    try:
        tick = int(value('present_tick_hz'))
    except ValueError:
        tick = 0
    if 1 <= tick <= 2000:
        env['GOW3_VBLANK_HZ'] = str(tick)
    mode = value('present_mode')
    if mode in PRESENT_MODES and mode != 'auto':
        env['GOW3_PRESENT_MODE'] = mode
    if value('present_frame_trace') == '1' and trace_path:
        env['GOW3_FLIP_TRACE'] = str(trace_path)
    return env


def no_frame_caps(ini):
    """Built-in frame caps off: the engine's tick-rate bound and the present thread's limit."""
    return ini.get('present_no_frame_caps', '0') == '1'


def perf_environment(ini):
    """The switches of the performance patches turned off in medievil.ini."""
    env = {}
    for key, off in PERF_SWITCHES.items():
        if ini.get(key, '0' if key in PERF_DEFAULT_OFF else '1') == '0':
            env.update(off)
    return env


def environment(options):
    return {'GOW3_DEFERRED_READBACK': options['deferred_readback'],
            'GOW3_STALE_READBACK': options['stale_readback'],
            'GOW3_ASYNC_SHADERS': options['async_shaders']}
