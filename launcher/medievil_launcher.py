#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
"""MediEvil (PS4, CUSA11227) launcher for Windows (Tkinter).

Every setting of the port in one window: the game (a folder or the .pkg, unpacked once), saves,
medievil.ini (display, frame rate, shaders, GPU data), start-up options passed to run.py as
environment variables, mods, third-party patches and controls.
Launcher options live in %APPDATA%/medievil-launcher/settings.json, or with a portable.txt beside
the launcher in its launcher-data folder (portable mode).

Frozen with PyInstaller the same MediEvil.exe also runs the game without the window (`--play`),
run.py (`--run`) and the preparation scripts (`--script`), so a packaged port needs no Python.
"""
import ctypes
import json
import os
from pathlib import Path
import queue
import re
import runpy
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import webbrowser
import zipfile

FROZEN = getattr(sys, 'frozen', False)
PORT_DIR = Path(sys.executable).resolve().parent if FROZEN else Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PORT_DIR / 'scripts'))
import save_backup  # noqa: E402  (scripts/)
import graphics_options  # noqa: E402  (scripts/)
WINDOWS = sys.platform == 'win32'
DATA_DIR = Path(os.environ.get('GOW3_DATA_DIR', PORT_DIR))
# Portable mode (issue #19): with a file portable.txt beside the launcher, its settings and caches
# stay in the launcher folder (launcher-data) instead of %APPDATA% (~/.config on Linux); saves already
# live there. Inside the read-only AppImage there is no portable.txt, so it never applies there.
PORTABLE = (PORT_DIR / 'portable.txt').exists()
if PORTABLE:
    CONFIG_DIR = PORT_DIR / 'launcher-data'
elif WINDOWS:
    CONFIG_DIR = Path(os.environ.get('APPDATA') or os.environ.get('USERPROFILE') or PORT_DIR) / 'medievil-launcher'
else:
    CONFIG_DIR = Path(os.environ.get('XDG_CONFIG_HOME') or Path.home() / '.config') / 'medievil-launcher'
CONFIG_FILE = CONFIG_DIR / 'settings.json'
MOD_PRESETS = DATA_DIR / 'mods_presets.json'
APP_NAME = 'MediEvil'
MAX_LOG_LINES = 6000
NO_WINDOW = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
# This build. The launcher checks GitHub releases at start and offers newer versions.
VERSION = '1.4.0'
RELEASES_API = 'https://api.github.com/repos/bc160201983/medievil_2019_pc_port/releases/latest'
RELEASES_PAGE = 'https://github.com/bc160201983/medievil_2019_pc_port/releases'
# The port's pages (Home and Advanced), opened in the browser.
LINKS = (('GitHub', 'https://github.com/bc160201983/medievil_2019_pc_port'),
         ('YouTube', 'https://www.youtube.com/@BilalReloaded'),
         ('Patreon', 'https://www.patreon.com/c/bilalreloaded'))
WINDOW_TITLE = '{} (PC) — Port By BilalReloaded · v{}'
UPDATE_DIR = Path(tempfile.gettempdir()) / 'medievil-update'
# Never copied over an installation by an update (the package does not hold them either).
USER_FILES = ('user', 'out', 'mods', 'medievil.ini', 'mods.json', 'mods_presets.json', 'patches.json', 'last_run.log',
              'launcher-data', 'portable.txt')


# ---------------------------------------------------------------------------------------------
# Command line roles of the frozen executable.

def attach_stdio():
    """A windowed executable starts without sys.stdout; inherited pipes or a console still exist."""
    if not WINDOWS:
        for name in ('stdout', 'stderr'):
            if getattr(sys, name) is None:
                setattr(sys, name, open(os.devnull, 'w'))
        return
    import msvcrt
    for name, std in (('stdout', -11), ('stderr', -12)):
        if getattr(sys, name) is not None:
            continue
        stream = None
        handle = ctypes.windll.kernel32.GetStdHandle(std)
        if handle and handle != ctypes.c_void_p(-1).value:
            try:
                stream = open(msvcrt.open_osfhandle(handle, os.O_WRONLY), 'w', encoding='utf-8',
                              errors='replace', buffering=1)
            except OSError:
                stream = None
        setattr(sys, name, stream or open(os.devnull, 'w'))


def run_role(argv):
    """--run [args]: run.py; --script <file> [args]: a preparation script. Returns an exit code."""
    attach_stdio()
    for stream in (sys.stdout, sys.stderr):  # keep messages in order with the game's output
        try:
            stream.reconfigure(line_buffering=True)
        except (AttributeError, ValueError):
            pass
    if argv[0] == '--script':
        path, sys.argv = argv[1], argv[1:]
    else:
        path = str(PORT_DIR / 'run.py')
        sys.argv = [path, *argv[1:]]
    try:
        runpy.run_path(path, run_name='__main__')
    except SystemExit as stop:
        return stop.code if isinstance(stop.code, int) else (0 if stop.code is None else 1)
    return 0


def run_command():
    """The command that starts run.py: this executable when frozen, else Python."""
    return [sys.executable, '--run'] if FROZEN else [sys.executable, str(PORT_DIR / 'run.py')]


# ---------------------------------------------------------------------------------------------
# Languages: every text is written in English with the Russian next to it; the other
# languages are in lang.py, keyed by the English text.

sys.path.insert(0, str(Path(__file__).resolve().parent))
import button_icons  # noqa: E402
import theme  # noqa: E402
import lang  # noqa: E402

LANG = 'en'


WARN = '  ⚠'  # marks a risky choice; the text before it is translated as usual


def _(en, ru=None):
    if en.endswith(WARN):
        return _(en[:-len(WARN)], ru and ru.removesuffix(WARN)) + WARN
    if LANG == 'ru':
        return ru or en
    return lang.table(LANG).get(en) or en


def windows_language():
    if not WINDOWS:  # Linux: the locale (LANGUAGE, LC_ALL, LC_MESSAGES, LANG), e.g. pt_BR.UTF-8
        for key in ('LANGUAGE', 'LC_ALL', 'LC_MESSAGES', 'LANG'):
            code = (os.environ.get(key) or '').split(':')[0].split('_')[0].split('.')[0].lower()
            if code and code not in ('c', 'posix'):
                return code if code == 'en' or code in lang.WINDOWS_LANGUAGES.values() else 'en'
        return 'en'
    try:
        primary = ctypes.windll.kernel32.GetUserDefaultUILanguage() & 0x3ff
    except (AttributeError, OSError):
        return 'en'
    return lang.WINDOWS_LANGUAGES.get(primary, 'en')


# ---------------------------------------------------------------------------------------------
# Settings. medievil.ini keys (the game reads them, the in-game menu edits them) and the
# launcher's own settings.json (passed to run.py as environment variables).

INI_FLAGS = {'cheat_menu', 'cheat_infinite_ammo', 'cheat_double_jumps', 'cheat_very_hard', 'show_fps', 'vsync', 'async_shaders', 'deferred_readback',
             'stale_readback', 'fsr1', 'rcas', 'ue4_dlss'}
# The graphics keys are also in the in-game menu (Insert / R3+L2); run.py reads the startup ones.
# The God of War III port's temporal upscaler keys (upscaler, preset, output_res, sharpen, ...)
# are not set here: run.py turns that upscaler off for MediEvil (its page was removed).
INI_DEFAULTS = {'show_fps': '1', 'display_mode': 'windowed', 'vsync': '1', 'fps_limit': '0',
                'async_shaders': '1', 'deferred_readback': '1', 'stale_readback': '1',
                'output_resolution': '1920x1080', 'render_scale': '100', 'engine_fps': '60', 'fsr1': '0',
                'rcas': '1', 'rcas_strength': '75', 'frames_queued': '1', 'ue4_quality': 'console',
                'ue4_dlss': '0',
                'effects_resolution': '75',
                'cheat_very_hard': '0', 'cheat_double_jumps': '0', 'cheat_menu': '0', 'cheat_infinite_ammo': '0',
                'ue4_aa': 'taa', 'ue4_voice': ''}
DISPLAY_MODES = [('windowed', ('Windowed',)), ('borderless', ('Borderless',)), ('fullscreen', ('Fullscreen',))]
FPS_LIMITS = [('-2', ('Auto: refresh (VRR: below it, Reflex rule)',)), ('-3', ('Auto: half refresh (VRR: one below)',)),
              ('30', ('30',)), ('60', ('60',)), ('119', ('119 (VRR, 240 Hz)',)), ('120', ('120',)),
              ('141', ('141 (VRR, 144 Hz)',)), ('237', ('237 (VRR, 240 Hz)',)), ('240', ('240',)),
              ('0', ('Unlimited',)), ('-1', ('Half display refresh (fixed refresh)',))]
APP_DEFAULTS = {'ui_language': 'en', 'game_dir': os.environ.get('GOW3_GAME_DIR', str(PORT_DIR.parent / 'medieval' / 'CUSA11227') if WINDOWS else ''), 'user_dir': '',
                'mods_dir': '', 'mods_enabled': True, 'patches_dir': '', 'language': '1',
                'player_name': '', 'hdr': False, 'draw_pipe': '', 'readbacks': '', 'vulkan_compat': False,
                'background_warmup': True, 'parallel_warmup': True, 'pipeline_binary_cache': True, 'perf_diag': False, 'frame_capture_key': False, 'frame_stats': False, 'gpu_profile': False,
                'vk_validation': False, 'close_on_play': False, 'prep_workers': '', 'stage_b_core': False,
                'update_check': True, 'game_patches': {}, 'game_patches_seen': {}, 'controls': {}, 'pad_style': 'playstation', 'stick_range': '',
                'last_pkg': '', 'graphics_preset': '', 'deck_switches': False, 'deck_switches_off': '',
                'deck_offered': False}

# PS4 button, default key, default gamepad button (SDL names; src/runtime_pad.c). L2/R2 come
# from the analog triggers and are not remapped on the gamepad.
CONTROLS = [('cross', 'Space', 'a'), ('circle', 'Left Shift', 'b'), ('square', 'E', 'x'),
            ('triangle', 'Q', 'y'), ('l1', '1', 'leftshoulder'), ('r1', '3', 'rightshoulder'),
            ('l2', 'R', None), ('r2', 'F', None), ('l3', 'Z', 'leftstick'),
            ('r3', 'C', 'rightstick'), ('options', 'Return', 'start'), ('touchpad', 'Tab', 'back'),
            ('up', 'I', 'dpup'), ('down', 'K', 'dpdown'), ('left', 'J', 'dpleft'), ('right', 'L', 'dpright')]
PS_BUTTON_NAMES = {'cross': 'Cross', 'circle': 'Circle', 'square': 'Square', 'triangle': 'Triangle',
                   'l1': 'L1', 'r1': 'R1', 'l2': 'L2', 'r2': 'R2', 'l3': 'L3', 'r3': 'R3',
                   'options': 'Options', 'touchpad': 'Touchpad', 'up': 'D-pad up', 'down': 'D-pad down',
                   'left': 'D-pad left', 'right': 'D-pad right'}
# Gamepad buttons by SDL name, labelled as on a PlayStation or an Xbox controller.
PAD_STYLES = {
    'playstation': {'a': 'Cross', 'b': 'Circle', 'x': 'Square', 'y': 'Triangle', 'leftshoulder': 'L1',
                    'rightshoulder': 'R1', 'leftstick': 'L3', 'rightstick': 'R3', 'start': 'Options',
                    'back': 'Share / Create', 'touchpad': 'Touchpad click', 'guide': 'PS button',
                    'dpup': 'D-pad up', 'dpdown': 'D-pad down', 'dpleft': 'D-pad left',
                    'dpright': 'D-pad right', 'misc1': 'Mute button'},
    'xbox': {'a': 'A', 'b': 'B', 'x': 'X', 'y': 'Y', 'leftshoulder': 'LB', 'rightshoulder': 'RB',
             'leftstick': 'LS (left stick click)', 'rightstick': 'RS (right stick click)', 'start': 'Menu',
             'back': 'View', 'touchpad': 'Touchpad click', 'guide': 'Xbox button', 'dpup': 'D-pad up',
             'dpdown': 'D-pad down', 'dpleft': 'D-pad left', 'dpright': 'D-pad right', 'misc1': 'Share'},
}


def control_changes(values, kind):
    """The buttons whose chosen key or SDL gamepad name differs from the default."""
    column = 1 if kind == 'key' else 2
    return {row[0]: values[row[0]].strip() for row in CONTROLS
            if row[column] and values.get(row[0], '').strip()
            and values[row[0]].strip().casefold() != row[column].casefold()}


def control_map(controls, kind):
    """GOW3_KEY_MAP/GOW3_PAD_MAP value: 'cross=Q,circle=Left Shift' for the changed buttons."""
    chosen = controls.get(kind, {}) if isinstance(controls, dict) else {}
    return ','.join(f'{button}={chosen[button].strip()}' for button, _key, _pad in CONTROLS
                    if isinstance(chosen.get(button), str) and chosen[button].strip()
                    and not set(chosen[button]) & set(',='))

# PS4 system language codes (sceSystemServiceParamGetInt LANG) of the languages MediEvil ships.
LANGUAGES = [('1', ('English', 'Английский')), ('18', ('English (UK)',)), ('8', ('Russian', 'Русский')),
             ('0', ('Japanese', 'Японский')), ('2', ('French', 'Французский')), ('3', ('Spanish', 'Испанский')),
             ('20', ('Spanish (Latin America)',)), ('4', ('German', 'Немецкий')), ('5', ('Italian', 'Итальянский')),
             ('16', ('Polish', 'Польский')), ('6', ('Dutch',)), ('7', ('Portuguese',)),
             ('17', ('Portuguese (Brazil)',)), ('12', ('Finnish',)), ('13', ('Swedish',)), ('14', ('Danish',)),
             ('15', ('Norwegian',)), ('19', ('Turkish',)), ('21', ('Arabic',)), ('23', ('Czech',)),
             ('24', ('Hungarian',)), ('25', ('Greek',)), ('26', ('Romanian',))]
# Spoken language (medievil.ini ue4_voice): '' follows the game language (the game's own map).
VOICE_LANGUAGES = [('', ('Same as game language',)), ('English(US)', ('English',)), ('French', ('French',)),
                   ('German', ('German',)), ('Italian', ('Italian',)), ('Spanish', ('Spanish',)),
                   ('es-MX', ('Spanish (Latin America)',)), ('Portuguese', ('Portuguese',)),
                   ('pt-BR', ('Portuguese (Brazil)',)), ('Dutch', ('Dutch',)), ('Polish', ('Polish',)),
                   ('Russian', ('Russian',)), ('Czech', ('Czech',)), ('Greek', ('Greek',)), ('Arabic', ('Arabic',)),
                   ('Japanese', ('Japanese',))]
DRAW_PIPE = [('', ('Auto (8+ threads)', 'Авто (8+ потоков)')), ('1', ('On', 'Включён')),
             ('0', ('Off (more stable)', 'Выключен (стабильнее)'))]
# "Precise" (GOW3_READBACKS=2) left MediEvil on a black screen (issue #15): not offered, and a
# saved choice of it starts with the default.
READBACKS = [('', ('Relaxed (default)', 'Relaxed (по умолчанию)')), ('0', ('Off', 'Выключены'))]
# Draw preparation helper threads (GOW3_PREP_WORKERS); '' = the runtime's default (half the CPU threads).
PREP_WORKERS = [('', ('Auto (half the CPU threads)',)), ('0', ('0 (none)',)), ('1', ('1',)), ('2', ('2',)),
                ('3', ('3',)), ('4', ('4',)), ('6', ('6',)), ('8', ('8',))]
FRAMES_QUEUED = [('1', ('1 (lowest latency)',)), ('2', ('2',)), ('0', ('Unlimited',))]
# Performance patches (research/docs/plan-stage-b-split.md, plan-frame-pacing.md): each is on by
# default except graphics_options.PERF_DEFAULT_OFF; switched off, its environment variables restore the earlier behaviour (A/B tests,
# or a fallback if one misbehaves on another PC). medievil.ini keys, shared with the in-game menu;
# run.py turns them into the variables (graphics_options.PERF_SWITCHES).
# (key, title, description, variables).
PERF_PATCHES = [
    ('perf_hzb_occlusion', 'HZB occlusion culling',
     "Unreal Engine 4's own culling skips objects hidden behind others, served from the GPU's "
     'finished results without waiting for it. Heavy areas ~130 -> ~165 FPS (MediEvil 01.00 and 01.02; '
     'off: no occlusion culling, as before).',
     {'GOW3_HZB_REMAP': '0'}),
    ('perf_no_decal_prepass', 'Skip the decal depth prepass',
     'The PS4 build draws the whole scene twice (depth first) for a few decals that are almost '
     'invisible. About 600 fewer draws a frame in heavy areas: ~10% FPS and the best 1% lows.',
     {'GOW3_DECAL_PREPASS': '1'}),
    ('perf_particle_single_pass', 'Single GPU particle pass',
     'Particles are simulated once a frame with the real frame time instead of a second, empty '
     'fixed-step pass: up to ~190 fewer draws while particles are on screen. Off by default: some '
     'torches animate very slowly with it.',
     {'GOW3_PARTICLE_FIXED_STEP': '1'}),
    ('perf_small_buffers', 'Small buffers from VRAM',
     'Small vertex and constant buffers the game has not changed are read from video memory instead '
     'of being copied for every draw (~1 GB/s less copying): about 5% in heavy areas.',
     {'GOW3_STREAM_CLEAN': '0'}),
    ('perf_texture_memo', 'Texture set memo',
     'Textures bound by a draw are remembered and reused unless their own memory changes (not '
     'whenever any texture streams in): texture binding ~7% cheaper, steadier when panning.',
     {'GOW3_TEXSET_REGIONS': '0', 'GOW3_TEXSET_TWO_WAY': '0'}),
    ('perf_lockfree_memory', 'Lock-free memory map lookups',
     "The GPU threads read the game's memory map without waiting for the game to finish committing "
     'or freeing memory: half the per-draw work while new content streams in (panning spikes).',
     {'GOW3_OPTIMISTIC_CLAMP': '0', 'GOW3_REGION_BUCKETS': '0'}),
    ('perf_cheap_frees', 'Cheap memory frees',
     "Memory the game frees (about a thousand times a second) keeps its page access instead of being "
     'locked: frees ~7x cheaper, the memory map blocked a third as often (smoother 0.1% lows).',
     {'GOW3_DECOMMIT_KEEP_ACCESS': '0'}),
    ('perf_gpu_data_no_waits', 'GPU data without waits',
     'Data the GPU writes is read back on a separate queue, kept beside the game\'s copy, and fences '
     'are signalled without stopping the draw thread: removed the constant micro-stutter (frame time '
     'spread 0.56 -> 0.06 ms in light areas).',
     {'GOW3_READBACK_QUEUE': '0', 'GOW3_GPU_WRITE_TWINS': '0', 'GOW3_EOS_DEFERRED': '0',
      'GOW3_EOP_TARGETED': '0'}),
    ('perf_4k_memory', '4K memory sizes',
     "Above 1080p the game gets 2 GB more memory than a PS4 (7,104 MB instead of 5,056), which it "
     "pools for graphics and texture streaming (output sizes above 1080p only).",
     {'GOW3_PS4_MEMORY': '1'}),
    ('perf_lean_barriers', 'Fewer render pass breaks',
     'The depth buffer is no longer switched between layouts, with a barrier each time, for every '
     'effect draw that reads it (death effects made ~100 needless pass breaks a frame): in fights the '
     'draw thread stays below 50% busy instead of 95%, 0.1% lows ~100 -> ~117 FPS at a 120 cap, '
     'no GPU usage spikes.',
     {'GOW3_SKIP_READ_BARRIERS': '0', 'GOW3_DEPTH_SAMPLE_IN_PLACE': '0'}),
    ('perf_lean_binding', 'Cheaper draw binding',
     "Each draw's descriptors are handed to the recording thread in one piece instead of ~16 small "
     "copies, and sampler handles are reused instead of hashed and looked up under a lock every draw: "
     '~0.3 ms less draw-thread work in heavy frames (~1,500 draws), most useful on slower CPUs. '
     'Applies at once in the in-game menu.',
     {'GOW3_LEAN_BINDING': '0'}),
    ('perf_ring_reuse', 'Constant ring reuse',
     "Shader constants the game binds again within one of its command buffers reuse the copy already in "
     'video memory instead of being copied again (about 13 -> 4 MB a frame written over PCIe): heavy-area '
     '0.1% lows ~145 -> ~185 FPS. Applies at once in the in-game menu.',
     {'GOW3_RING_REUSE': '0'}),
]
INI_FLAGS.update(entry[0] for entry in PERF_PATCHES)
INI_DEFAULTS.update({entry[0]: '0' if entry[0] in graphics_options.PERF_DEFAULT_OFF else '1'
                     for entry in PERF_PATCHES})

# Presentation (research/docs/plan-stage-b-split.md, T-L2 to T-V120h): how finished frames reach the
# screen. All off by default (the tested defaults); run.py turns them into environment variables
# (graphics_options.present_environment). (key, title, description).
PRESENT_SWITCHES = [
    ('present_inline', 'Present on the vblank thread',
     'Shows frames from the vblank thread itself, as upstream shadPS4 does, instead of a separate swap '
     'thread. A present that waits for the display then also delays the next vblank.'),
    ('present_precise_tick', 'Precise vblank tick',
     'The vblank tick runs on a fixed grid with a high-resolution timer and a short spin instead of '
     "Windows' coarse sleep (about 8% of one CPU core at 120 Hz)."),
]
# Advanced > Diagnostics: one CSV line per flip stage (GOW3_FLIP_TRACE, graphics_options.present_environment);
# not the frame statistics' 5 s summary.
FRAME_TRACE = ('present_frame_trace', 'Frame timing trace',
               'Writes every frame\'s timing to user/logs/flip-trace.csv (overwritten at each start).')
# One VSync control (vsync_choice_var): On / Off set the vsync key the in-game menu also switches
# (live); the two special modes set a fixed present_mode, which overrides that switch.
VSYNC_CHOICES = [('on', ('On',)), ('off', ('Off',)), ('immediate', ('Off, allow tearing',)),
                 ('fifo_relaxed', ('Adaptive',))]
VSYNC_HINT = ('On: no tearing; with G-Sync / FreeSync and a frame rate limit below the refresh it adds no latency. '
              'Off: lower latency, no tearing. On and Off can also be switched live in the game\'s menu. '
              'Off, allow tearing: the lowest latency. Adaptive: on, but a late frame shows at once (may tear). '
              'These two fix the mode: the in-game switch then has no effect.')
PRESENT_LIMITERS = [('reflex', ('NVIDIA Reflex (recommended)',)), ('precise', ('Precise',)),
                    ('standard', ('Standard',))]
# The engine frame rate choice also covers "no built-in caps" (present_no_frame_caps).
ENGINE_CAP_CHOICES = [('60', ('60 (original)',)), ('120', ('120',)), ('240', ('240',)),
                      ('unlimited', ('Unlimited (no built-in caps)',))]
INI_FLAGS.update(entry[0] for entry in PRESENT_SWITCHES)
INI_FLAGS.update({FRAME_TRACE[0], 'present_no_frame_caps', 'display_vrr'})
INI_DEFAULTS.update({'present_inline': '1', 'present_precise_tick': '1', 'present_frame_trace': '0',
                     'present_no_frame_caps': '0', 'present_tick_hz': '960', 'present_mode': 'auto',
                     'present_limiter': 'reflex', 'fps_limit': '-3', 'display_vrr': '1'})
# GOW3_STICK_RANGE (issue #36): that much tilt is a full push, for sticks that never reach the edge.
STICK_RANGES = [('', ('Full tilt (default)',)), ('90', ('90%',)), ('85', ('85%',)), ('80', ('80%',)),
                ('70', ('70%',))]
UI_LANGUAGES = lang.LANGUAGE_NAMES

def load_json(path, default):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return default


def screenshots_dir(s):
    return Path(s.get('user_dir') or DATA_DIR / 'user') / 'screenshots'


def ini_path():
    old = DATA_DIR / 'gow3.ini'  # settings of builds before 1.1
    if not (DATA_DIR / 'medievil.ini').exists() and old.exists():
        old.rename(DATA_DIR / 'medievil.ini')
    return Path(os.environ.get('GOW3_CONFIG', DATA_DIR / 'medievil.ini'))


def load_ini():
    # Graphics settings missing from the file stay out: graphics_migration (run by both callers) fills them
    # from the file's ue4_quality. INI_DEFAULTS holds Console's, which a Low file must not get.
    values = {key: value for key, value in INI_DEFAULTS.items() if key not in graphics_options.GRAPHICS_KEYS}
    lines, found = [], {}
    try:
        lines = ini_path().read_text(encoding='utf-8').splitlines()
    except OSError:
        pass
    seen = set()
    for line in lines:
        if '=' in line and not line.lstrip().startswith('#'):
            key, value = line.split('=', 1)
            found[key.strip()] = value.strip()
    values.update(found)
    # The single render_resolution (and ue4_upscale) of earlier versions as the two settings.
    values.update(graphics_options.migrate_resolution(found))
    return values, lines


def save_ini(values, lines):
    """Rewrites the edited keys in place, appends missing ones, keeps comments and other keys."""
    written, out = set(), []
    for line in lines:
        if '=' in line and not line.lstrip().startswith('#'):
            key = line.split('=', 1)[0].strip()
            if key in values:
                out.append(f'{key}={values[key]}')
                written.add(key)
                continue
        out.append(line)
    if not lines:
        out.append('# MediEvil settings (in-game menu: Insert / R3+L2)')
    out += [f'{key}={value}' for key, value in values.items() if key not in written]
    ini_path().write_text('\n'.join(out) + '\n', encoding='utf-8')


def game_info(folder):
    """(title, app version) of a game folder, or None without eboot.bin."""
    folder = Path(folder or '.')
    if not (folder / 'eboot.bin').is_file():
        return None
    try:
        from prepare import sfo
        values = sfo((folder / 'sce_sys/param.sfo').read_bytes())
        return values.get('TITLE', APP_NAME).replace('™', '').replace('®', '').strip(), values.get('APP_VER', '?')
    except (OSError, ValueError, ImportError):
        return APP_NAME, '?'


def game_profile_of(folder):
    """(title id, (ids, built-in XML, app version)) of a game with a patch profile, else (id, None)."""
    from patches import game_app_version, game_profile, game_title_id
    title_id = game_title_id(folder) if folder else None
    version = game_app_version(folder) if folder else None
    return title_id, game_profile(title_id, version) or game_profile(title_id)


def chosen_by_run_py(name):
    """Patches run.py picks from medievil.ini (resolution, frame rate, output size): not on the Patches page."""
    from patches import EXCLUSIVE_PREFIX
    return name.startswith(EXCLUSIVE_PREFIX) or name.startswith('Frame Rate Patch') or \
        name in (graphics_options.BOOT_SIZE_PATCH, graphics_options.BOOT_4K_PATCH)


# Built-in patches added after the launcher started keeping game_patches_seen (2026-11-02): for a
# selection saved before, every other patch counts as seen.
PATCHES_AFTER_SEEN_LIST = {'GPU particle bounds'}


def game_patch_selection(s, title_id, profile):
    """[names] on the Patches page for this game's version, or None without a saved selection (the
    file's defaults). A default patch added after the selection was saved is on: game_patches_seen
    lists the patches the page showed then, so a default the user switched off stays off."""
    import xml.etree.ElementTree as ET
    chosen = s.get('game_patches', {}).get(title_id) if title_id else None
    if not profile or chosen is None:
        return None
    metas = [m for m in ET.parse(profile[1]).getroot().iter('Metadata')
             if m.get('AppVer') == profile[2] and not chosen_by_run_py(m.get('Name'))]
    default = {m.get('Name') for m in metas if m.get('isEnabled', 'false').lower() == 'true'}
    seen = s.get('game_patches_seen', {}).get(title_id)
    if seen is None:
        seen = {m.get('Name') for m in metas} - PATCHES_AFTER_SEEN_LIST
    return [m.get('Name') for m in metas if m.get('Name') in chosen or m.get('Name') in default - set(seen)]


def game_environment(s):
    env = dict(os.environ)
    env['GOW3_GAME_DIR'] = s['game_dir']
    # The launcher's patch selection is the whole list (run.py, patches.py).
    title_id, profile = game_profile_of(s['game_dir'])
    chosen = game_patch_selection(s, title_id, profile)
    if chosen is not None:
        env['GOW3_PATCHES'] = ';'.join(chosen)
        env['GOW3_PATCHES_ONLY'] = '1'
    if s['user_dir']:
        env['GOW3_USER_DIR'] = s['user_dir']
    env['GOW3_MODS_DIR'] = s['mods_dir'] or str(DATA_DIR / 'mods')
    env['GOW3_MODS_CONFIG'] = str(DATA_DIR / 'mods.json')
    env['GOW3_MODS_ENABLED'] = '1' if s['mods_enabled'] else '0'
    env['GOW3_PATCHES_DIR'] = s['patches_dir'] or str(DATA_DIR / 'patches')
    env['GOW3_PATCHES_CONFIG'] = str(DATA_DIR / 'patches.json')
    env['GOW3_LANGUAGE'] = s['language']
    if str(s['player_name']).strip():
        env['GOW3_USER_NAME'] = str(s['player_name']).strip()
    if s['hdr']:
        env['GOW3_HDR'] = '1'
    for key, name in (('draw_pipe', 'GOW3_DRAW_PIPE'), ('readbacks', 'GOW3_READBACKS')):
        if s[key] and not (key == 'readbacks' and s[key] == '2'):
            env[name] = s[key]
    # Safe GPU mode (issues #18/#22): only the Vulkan features of older AMD drivers (RX 400/500, Vega).
    if s.get('vulkan_compat', False):
        env['GOW3_VULKAN_COMPAT'] = '1'
    env['GOW3_BACKGROUND_WARMUP'] = '1' if s.get('background_warmup', True) else '0'
    env['GOW3_PARALLEL_WARMUP'] = '1' if s.get('parallel_warmup', True) else '0'
    if s.get('prep_workers', '') != '':
        env['GOW3_PREP_WORKERS'] = str(s['prep_workers'])
    if s.get('stage_b_core', False):
        env['GOW3_STAGE_B_CORE'] = '1'
    # Deck Optimized (DECK_SWITCH_ITEMS): the checked ones that work here; a variable already set wins.
    deck_env, deck_summary = deck_switch_env(s)
    for name, value in deck_env.items():
        env.setdefault(name, value)
    if deck_summary:
        env['GOW3_DECK_SWITCHES'] = deck_summary  # logged by run.py
    env['GOW3_PIPELINE_BINARIES'] = '1' if s.get('pipeline_binary_cache', True) else '0'
    env.setdefault('GOW3_PERF_DIAG', '1' if s.get('perf_diag', False) else '0')
    for key, name in (('frame_stats', 'GOW3_FRAME_STATS'), ('gpu_profile', 'GOW3_GPU_PROFILE'),
                      ('vk_validation', 'GOW3_VK_VALIDATION'), ('frame_capture_key', 'GOW3_FRAME_CAPTURE_KEY')):
        if s[key]:
            env[name] = '1'
    if s.get('stick_range'):
        env['GOW3_STICK_RANGE'] = s['stick_range']
    for kind, name in (('key', 'GOW3_KEY_MAP'), ('pad', 'GOW3_PAD_MAP')):
        if value := control_map(s.get('controls', {}), kind):
            env[name] = value
    env.setdefault('GOW3_FPS_LOG', '1')  # FPS and worst frame every 5 s in the log (diagnostics)
    # Frame statistics (draws, compiles, stalls every 5 s) only with Advanced > Frame statistics:
    # part of them is printed by the draw recording thread and costs frame time stability (T-S2b).
    # F12 in the game saves a PNG screenshot of the game's picture.
    env.setdefault('GOW3_SCREENSHOT_DIR', str(screenshots_dir(s)))
    # Shift+F12 (or creating the trigger file) saves raw frames for bug reports. Each session gets
    # its own folder, made on the first dump (frame numbers restart at every start); one press
    # saves 8 consecutive frames so flicker shows up between them.
    shots = Path(s['user_dir'] or DATA_DIR / 'user') / 'frame-dumps'
    session = shots / time.strftime('%Y%m%d-%H%M%S')
    try:
        shots.mkdir(parents=True, exist_ok=True)
        env.setdefault('GOW3_DUMP_DIR', str(session))
        env.setdefault('GOW3_FRAME_DUMP_TRIGGER', str(shots / 'trigger'))
        env.setdefault('GOW3_FRAME_DUMP_COUNT', '8')
        # Diagnostics: GPU optimizations switched off while playing, by writing a number here.
        env.setdefault('GOW3_TOGGLE_FILE', str(shots / 'toggles.txt'))
        # Frame analyzer: creating this file records how the next frame is drawn (render passes).
        env.setdefault('GOW3_CAPTURE_TRIGGER', str(shots / 'capture'))
        env.setdefault('GOW3_CAPTURE_DIR', str(session))
    except OSError:
        pass
    env['PYTHONUNBUFFERED'] = '1'
    env['PYTHONIOENCODING'] = 'utf-8'
    return env


# ---------------------------------------------------------------------------------------------
# The window.

# MediEvil's colors (theme.py): Gallowmere's night purple and stone, bone text, the logo's gold
# and the green of souls.
BG, PANEL, CARD, LINE = theme.NIGHT, theme.STONE, theme.CARD, theme.EDGE
TEXT, MUTED, ACCENT, ACCENT_HI, GOLD = theme.BONE, theme.MIST, theme.SOUL_DARK, theme.SOUL, theme.GOLD


class ImageButton:
    """A button drawn by theme.play_button (normal, hover, disabled), with ttk.Button's
    configure(text=, state=) and pack()."""

    def __init__(self, parent, tk, fonts, text, command, width, height, bg):
        self.tk, self.fonts, self.command = tk, fonts, command
        self.size, self.text, self.state, self.hover = (width, height), text, 'normal', False
        self.images = {}
        self.label = tk.Label(parent, bg=bg, bd=0, cursor='hand2', highlightthickness=0)
        self.label.bind('<Enter>', lambda _e: self._set(hover=True))
        self.label.bind('<Leave>', lambda _e: self._set(hover=False))
        self.label.bind('<Button-1>', lambda _e: self.state == 'normal' and self.command())
        self._draw()

    def _image(self, look):
        key = (self.text, look)
        if key not in self.images:
            from PIL import ImageTk
            self.images[key] = ImageTk.PhotoImage(theme.play_button(self.fonts, self.text, *self.size, look))
        return self.images[key]

    def _draw(self):
        look = 'disabled' if self.state == 'disabled' else 'hover' if self.hover else 'normal'
        self.label.configure(image=self._image(look), cursor='hand2' if self.state == 'normal' else '')

    def _set(self, hover):
        self.hover = hover
        self._draw()

    def configure(self, text=None, state=None):
        if text is not None:
            self.text = text
        if state is not None:
            self.state = str(state)
        self._draw()

    config = configure

    def pack(self, **kwargs):
        self.label.pack(**kwargs)
# Output: the game's own picture at that size (for that display). Scene in 1080p: a 1080p picture whose
# 3D scene is drawn at that size and scaled into it (supersampling for a 1080p display).
# medievil.ini output_resolution: any WxH works (shown as "Custom"); graphics_options.parse_output.
OUTPUT_CHOICES = [('display', ('Match display ({}x{})'.format(*graphics_options.display_size()),)),
                  ('3840x2160', ('4K (3840x2160)',)), ('3200x1800', ('1800p (3200x1800)',)),
                  ('2560x1440', ('1440p (2560x1440)',)), ('1920x1080', ('1080p (original)',)),
                  ('1280x720', ('720p (1280x720)',))]
OUTPUT_HINT = ("The size of the game's picture. The game starts at this size, as a PS4 starts at its TV's; "
               'match your display for a sharp 1:1 image (in windowed mode the window opens at this size). '
               'Above 1080p the port also turns on its larger memory patches. Any size can be set in '
               'medievil.ini (output_resolution=WxH). Applied at start.')
RENDER_SCALE_CHOICES = [('200', ('200%: supersampled (sharpest, slowest)',)), ('167', ('167%: supersampled',)),
                        ('133', ('133%: supersampled',)), ('100', ('100%: native',)),
                        ('67', ('67%: upscaled, Quality',)), ('58', ('58%: upscaled, Balanced',)),
                        ('50', ('50%: upscaled, Performance',))]
RENDER_SCALE_HINT = ('How large the 3D scene is drawn, per side, compared with the output. Above 100%: drawn '
                     'larger and scaled down (sharper edges, slower). Below 100%: drawn smaller and rebuilt at '
                     "full size by the game's own temporal upscaler from its motion vectors (faster). The menus "
                     'and HUD always draw at the output size. Pick a preset or type any value from 50 to 200%. '
                     'Applied at start; the in-game menu (Performance > Applied now) also switches it while '
                     'playing, from 50% up to the value the game started at (at least 100%).')


def render_scale_scene(output, scale):
    """The scene size a render scale gives at an output_resolution value, for the launcher's row."""
    width, height = graphics_options.output_size(output)
    return _('scene {}x{}').format(*graphics_options.scene_size(width, height,
                                                                graphics_options.parse_render_scale(scale)))
AA_CHOICES = [('taa', ('Temporal (default, smooth)',)), ('fxaa', ('FXAA (sharper)',)),
              ('off', ('Off (sharpest, jagged edges)',))]
QUALITY_CHOICES = [('ultra', ('Ultra',)), ('high', ('High',)), ('console', ('Console (PS4)',)),
                   ('medium', ('Medium',)), ('low', ('Low (integrated graphics)',))]
EFFECTS_CHOICES = [('100', ('100%',)), ('75', ('75% (default)',)), ('50', ('50% (PS4 detail at 4K)',))]
# The graphics settings (scripts/graphics_options.py GRAPHICS_SETTINGS, the same table as the in-game menu's
# gpu/shim/gow3_settings.h UE4LiveOptions; tests check they match): (key, title, hint, [(value, label)]).
# Graphics quality sets all of them; a fresh medievil.ini starts at Console's (the PS4's) values.
GRAPHICS_OPTIONS = [(key, title, hint, [(token, label) for token, label, _values in choices])
                    for key, title, hint, _live, _cvars, choices, _levels in graphics_options.GRAPHICS_SETTINGS]
INI_DEFAULTS.update(graphics_options.level_choices('console'))
ENGINE_FPS_CHOICES = [('60', ('60 (original)',)), ('120', ('120',)), ('240', ('240',))]

# Steam Deck (research/docs/plan-linux-deck.md, "Deck Optimized preset"): the user's flat 60 FPS in
# the heaviest graveyard view (Deck OLED, SteamOS, AppImage, GPU clock fixed at 1,200 MHz): the Low
# logging settings with an 80% render scale, volumetric fog and a 60 FPS limit. "Deck Optimized" in
# Graphics quality writes these to medievil.ini (ue4_quality stays low for the game) and turns on
# DECK_SWITCHES. The 60 cap: the engine is unbounded (present_no_frame_caps, engine_fps 240) and the
# vblank tick runs at 900 Hz, far above the cap, so fps_limit is the only clock: the present thread
# releases one flip per 1/60 s slot (Standard limiter: no Reflex, no second limiter at the same rate;
# VSync off = Mailbox under gamescope, which offers no Immediate).
DECK_PRESET = {
    'show_fps': '0', 'display_mode': 'borderless', 'vsync': '0', 'fps_limit': '60', 'async_shaders': '1',
    'deferred_readback': '1', 'stale_readback': '1', 'output_resolution': '1280x720', 'render_scale': '80',
    'engine_fps': '240', 'fsr1': '0', 'rcas': '0', 'rcas_strength': '29', 'frames_queued': '1',
    'ue4_quality': 'low', 'effects_resolution': '50', 'ue4_aa': 'taa',
    'perf_hzb_occlusion': '1', 'perf_no_decal_prepass': '1', 'perf_particle_single_pass': '1',
    'perf_small_buffers': '1', 'perf_texture_memo': '1', 'perf_lockfree_memory': '1', 'perf_cheap_frees': '1',
    'perf_gpu_data_no_waits': '1', 'perf_4k_memory': '0', 'perf_lean_barriers': '1', 'perf_lean_binding': '1',
    'perf_ring_reuse': '1',
    'present_inline': '1', 'present_precise_tick': '0', 'present_frame_trace': '0', 'present_no_frame_caps': '1',
    'present_tick_hz': '900', 'present_mode': 'auto', 'present_limiter': 'standard', 'display_vrr': '0',
    'preshadows': '0', **graphics_options.level_choices(graphics_options.DECK_LEVEL)}
# medievil.ini keys the launcher writes only when they are set (the in-game menu's own switches).
INI_OPTIONAL_KEYS = ('preshadows',)
# The Deck's measured wins that are still opt-in start switches (plan-linux-deck.md): early frame EOP,
# decode pacing with a 2-frame lead, U2 off (no gain on the Deck, its tracking costs stage B), stage A's
# hybrid drain wait (-1/3 of its CPU), the wide texture-set memo (T2), the region cache index (R1),
# stage A's per-packet cuts A1 / A2 / A3, the particle flip guard (the single pass's lost updates: the
# frozen torch, the crypt floor) and nice -10 for stage B and the vblank thread (better lows at a fixed
# GPU clock). Measured on the Deck only, so they come with the preset (launcher setting deck_switches),
# not as Linux defaults; an environment variable already set wins, and run.py applies extra-env.txt last.
# One by one (Advanced: the switches under "Steam Deck performance switches"): (key, title, hint, variables,
# why it does nothing on Windows or None). The Windows reasons are the code's: A1 and thread nice are Linux
# code, and U2 is off on Windows anyway (buffer_cache.cpp HostVisibleArenaType), so those are greyed out
# and never set there; the others run on both and are untested on Windows (research/patches/deck).
DECK_SWITCH_ITEMS = [
    ('early_frame_eop', 'Early frame end signals',
     "The frame's end signal at decode instead of after recording (stage A runs ahead).",
     {'GOW3_EARLY_FRAME_EOP': '1'}, None),
    ('decode_pacing', 'Decode pacing (2 frames ahead)', 'Stage A decodes up to 2 frames ahead of the GPU.',
     {'GOW3_DECODE_PACING': '1', 'GOW3_DECODE_PACING_LEAD': '2'}, None),
    ('drain_hybrid', "Stage A's hybrid wait", 'Stage A spins briefly, then sleeps, while it waits for stage B.',
     {'GOW3_DRAIN_HYBRID': '1'}, None),
    ('texset_wide', 'Wide texture-set memo (T2)', 'Texture sets of draws without a prepared draw are memoized too.',
     {'GOW3_TEXSET_WIDE': '1'}, None),
    ('region_index', 'Region cache index (R1)', 'Guest memory lookups answered from an index by page.',
     {'GOW3_REGION_INDEX': '1'}, None),
    ('pending_filter', 'Pending-write filter (A2)', "Stage A's pending-write checks answered from a page filter.",
     {'GOW3_PENDING_FILTER': '1'}, None),
    ('reg_hash_paired', 'Paired register checksum (A3)', 'The register checksum takes two words per step.',
     {'GOW3_REG_HASH_PAIRED': '1'}, None),
    ('particle_flip_guard', 'Particle flip guard', "One state flip per particle simulation (the single pass's "
     'frozen torches).', {'GOW3_PARTICLE_FLIP_GUARD': '1'}, None),
    ('backing_copy', 'Backing copies without the lock (A1)', 'GPU-side guest copies read the backing memory '
     'without the region table lock.', {'GOW3_BACKING_COPY': '1'}, 'Linux only'),
    ('thread_nice', 'Thread priorities', 'Stage B and the vblank thread at a higher priority (nice -10).',
     {'GOW3_THREAD_NICE': '1'}, 'Linux only'),
    ('arena_writes_off', 'Direct arena writes off (U2)', 'Buffer data through staging copies instead of direct '
     'writes (no gain on the Deck; their tracking costs stage B).', {'GOW3_ARENA_HOST_VISIBLE': '0'},
     'already off on Windows'),
]
DECK_SWITCHES = {name: value for *_rest, env, _why in DECK_SWITCH_ITEMS for name, value in env.items()}


def deck_switch_env(s, windows=WINDOWS):
    """The Deck switches to set at start: with deck_switches on, every item not listed in deck_switches_off
    (the ones unchecked) and not one that does nothing on this platform. Also the summary run.py logs."""
    if not s.get('deck_switches', False):
        return {}, ''
    off = {key for key in str(s.get('deck_switches_off', '')).split(',') if key}
    env, on, skipped = {}, [], []
    for key, _title, _hint, variables, windows_reason in DECK_SWITCH_ITEMS:
        if windows and windows_reason:
            skipped.append(f'{key} ({windows_reason})')
        elif key in off:
            skipped.append(f'{key} (unchecked)')
        else:
            env.update(variables)
            on.append(key)
    summary = f'{len(on)} on: {", ".join(on) or "none"}' + (f'; off: {", ".join(skipped)}' if skipped else '')
    if windows:
        summary += '; Windows: untested'
    return env, summary
DECK_QUALITY = 'deck'
QUALITY_PRESETS = [*QUALITY_CHOICES, (DECK_QUALITY, ('Deck Optimized',))]
DECK_HINT = ('Steam Deck: in the Quick Access Menu (Performance) set a manual GPU clock of about 1,200 MHz and '
             'keep the frame limit at 60 for a flat 60 FPS; an uncapped GPU takes power from the CPU and frame '
             'times swing.')
# Valve's boards in /sys/devices/virtual/dmi/id: Jupiter (Steam Deck LCD), Galileo (Steam Deck OLED).
DECK_DMI = Path('/sys/devices/virtual/dmi/id')
DECK_PRODUCTS = ('Jupiter', 'Galileo')


def steam_deck(env=None, dmi=DECK_DMI):
    """True on a Steam Deck: SteamDeck=1 (Steam sets it for what it starts) or the board's DMI
    (vendor Valve, product Jupiter or Galileo)."""
    env = os.environ if env is None else env
    if env.get('SteamDeck') == '1':
        return True

    def read(name):
        try:
            return (Path(dmi) / name).read_text(encoding='utf-8', errors='replace').strip()
        except OSError:
            return ''
    vendor = read('board_vendor') or read('sys_vendor')
    product = read('product_name') or read('board_name')
    return vendor == 'Valve' and product in DECK_PRODUCTS


def deck_preset_offer(app, first_run, on_deck):
    """What the launcher does about the Deck Optimized preset when it opens: 'select' on the first run
    on a Deck (no settings.json and no medievil.ini yet), 'ask' once for an existing setup on a Deck,
    else None (asked or chosen before, or not a Deck). Existing settings never change unasked."""
    if not on_deck or app.get('deck_offered') or app.get('graphics_preset') == DECK_QUALITY:
        return None
    return 'select' if first_run else 'ask'


class Launcher:
    def __init__(self, root, tk, ttk, filedialog, messagebox):
        self.tk, self.ttk, self.filedialog, self.messagebox = tk, ttk, filedialog, messagebox
        self.root = root
        self.first_run = not CONFIG_FILE.exists() and not ini_path().exists()
        self.app = {**APP_DEFAULTS, **load_json(CONFIG_FILE, {})}
        if self.app.get('readbacks') == '2':  # "Precise" is no longer offered (issue #15)
            self.app['readbacks'] = ''
        self.ini, self.ini_lines = load_ini()
        self.ini.update(graphics_options.graphics_migration(self.ini))  # older tokens, "preset", new keys
        self.vars = {}
        self.process = self.job = None
        self.output = queue.Queue()
        self.banner_source = self.banner_image = None
        self.ui_calls = queue.Queue()  # work for the Tk thread from helper threads
        self.mod_order, self.mod_vars, self.patch_vars = [], {}, {}
        root.title(WINDOW_TITLE.format(APP_NAME, VERSION))
        if sys.platform == 'win32':
            try:  # the taskbar shows this window's icon, not Python's
                import ctypes
                ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID('medievilpc.launcher')
            except (AttributeError, OSError):
                pass
        root.configure(bg=BG)
        self.dpi = root.winfo_fpixels('1i') / 96.0
        self.fonts = theme.load_fonts(self.app.get('game_dir'), CONFIG_DIR)
        self.images = []  # PhotoImages the widgets show (kept alive here)
        root.geometry(f'{self.px(1120)}x{self.px(740)}')
        root.minsize(self.px(980), self.px(660))
        self.set_icon()
        self.style()
        self.build()
        self.show('home')
        root.protocol('WM_DELETE_WINDOW', self.close)
        root.after(100, self.drain_output)
        root.after(400, self.offer_deck_preset)
        if self.app.get('update_check', True):
            root.after(1500, lambda: threading.Thread(target=self.check_update, daemon=True).start())

    def px(self, size):
        return int(size * self.dpi)

    def f(self, role, size, *style):
        """A Tk font: role 'logo', 'head' or 'body' (the game's fonts when available)."""
        return (self.fonts.family[role], size, *style)

    def photo(self, image):
        """A Pillow image as a PhotoImage that stays alive with the launcher."""
        from PIL import ImageTk
        photo = ImageTk.PhotoImage(image)
        self.images.append(photo)
        return photo

    def gold(self, text, size, role='logo', glow=None, max_width=None):
        """Gold title text as an image (Pillow), or None without Pillow; narrowed to max_width."""
        try:
            from PIL import Image
            image = theme.gold_text(self.fonts, text, self.px(size), role=role, glow=glow)
        except ImportError:
            return None
        if max_width and image.width > self.px(max_width):
            ratio = self.px(max_width) / image.width
            image = image.resize((self.px(max_width), max(1, round(image.height * ratio))), Image.LANCZOS)
        return self.photo(image)

    def set_icon(self):
        game_icon = Path(self.var('game_dir', 'app').get() or '.') / 'sce_sys' / 'icon0.png'
        try:
            from PIL import Image
            ico = CONFIG_DIR / 'game_icon.ico'
            CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            with Image.open(game_icon) as image:
                image.convert('RGBA').save(ico, sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (256, 256)])
            self.root.iconbitmap(default=str(ico))
            return
        except (ImportError, OSError, self.tk.TclError):
            pass
        for icon in (game_icon, PORT_DIR / 'launcher' / 'medievil.png'):
            try:
                self.icon = self.tk.PhotoImage(file=str(icon))
                if self.icon.width() > 128:
                    self.icon = self.icon.subsample(self.icon.width() // 64)
                self.root.iconphoto(True, self.icon)
                return
            except (self.tk.TclError, OSError):
                continue

    # ---- look --------------------------------------------------------------------------------
    def check_images(self):
        """16 px (DPI-scaled) check box images: gold frames, green tick (theme.check_box)."""
        tk = self.tk
        n = max(14, int(16 * self.dpi))
        try:
            self.check_off, self.check_on = (self.photo(theme.check_box(n, c)) for c in (False, True))
            return
        except ImportError:
            pass
        images = []
        for checked in (False, True):
            image = tk.PhotoImage(width=n + self.px(8), height=n)  # unset pixels stay transparent
            fill = ACCENT if checked else CARD
            image.put(ACCENT_HI if checked else '#453f52', to=(0, 0, n, n))
            image.put(fill, to=(1, 1, n - 1, n - 1))
            if checked:
                # Tick: down from (0.22n, 0.52n) to (0.42n, 0.72n), up to (0.78n, 0.30n).
                pts = []
                x0, y0, x1, y1, x2, y2 = .22 * n, .52 * n, .42 * n, .72 * n, .78 * n, .28 * n
                for t in range(40):
                    f = t / 39
                    pts.append((x0 + (x1 - x0) * f, y0 + (y1 - y0) * f))
                    pts.append((x1 + (x2 - x1) * f, y1 + (y2 - y1) * f))
                width = max(1, round(n / 9))
                for x, y in pts:
                    image.put('#f4ece0', to=(int(x), int(y), int(x) + width, int(y) + width))
            images.append(image)
        self.check_off, self.check_on = images

    def style(self):
        ttk = self.ttk
        s = ttk.Style(self.root)
        s.theme_use('clam')
        self.check_images()
        s.element_create('Bb.indicator', 'image', self.check_off, ('selected', self.check_on), sticky='')
        s.layout('TCheckbutton', [('Checkbutton.padding', {'sticky': 'nswe', 'children': [
            ('Bb.indicator', {'side': 'left', 'sticky': ''}),
            ('Checkbutton.focus', {'side': 'left', 'sticky': 'w', 'children': [
                ('Checkbutton.label', {'sticky': 'nswe'})]})]})])
        base = self.f('body', 10)
        self.root.option_add('*TCombobox*Listbox.background', CARD)
        self.root.option_add('*TCombobox*Listbox.foreground', TEXT)
        self.root.option_add('*TCombobox*Listbox.selectBackground', ACCENT)
        self.root.option_add('*TCombobox*Listbox.font', base)
        s.configure('.', background=PANEL, foreground=TEXT, fieldbackground=CARD, bordercolor=LINE,
                    lightcolor=LINE, darkcolor=LINE, troughcolor=CARD, focuscolor=ACCENT, font=base)
        s.configure('TFrame', background=PANEL)
        s.configure('TLabel', background=PANEL, foreground=TEXT)
        s.configure('Muted.TLabel', background=PANEL, foreground=MUTED, font=self.f('body', 9))
        s.configure('Restart.TLabel', background=PANEL, foreground=ACCENT_HI, font=self.f('body', 8))
        s.configure('Section.TLabel', background=PANEL, foreground=GOLD, font=self.f('head', 15))
        s.configure('TCheckbutton', background=PANEL, foreground=TEXT, padding=(0, 3))
        s.map('TCheckbutton', background=[('active', PANEL)], foreground=[('disabled', MUTED)])
        s.configure('TRadiobutton', background=PANEL, foreground=TEXT, indicatorcolor=CARD)
        s.map('TRadiobutton', background=[('active', PANEL)], indicatorcolor=[('selected', ACCENT)])
        s.configure('Warning.TLabel', background='#1e2614', foreground='#c8e0a0', padding=(10, 6))
        s.configure('TCombobox', arrowcolor=MUTED, foreground=TEXT, padding=4)
        s.map('TCombobox', fieldbackground=[('readonly', CARD)], foreground=[('readonly', TEXT)],
              selectbackground=[('readonly', CARD)], selectforeground=[('readonly', TEXT)])
        s.configure('TEntry', foreground=TEXT, insertcolor=TEXT, padding=4)
        s.configure('TSpinbox', foreground=TEXT, arrowcolor=MUTED, insertcolor=TEXT, padding=4)
        s.configure('TButton', background=CARD, foreground=TEXT, padding=(12, 6), borderwidth=1,
                    bordercolor=theme.BRONZE, lightcolor=CARD, darkcolor=CARD)
        s.map('TButton', background=[('active', '#2e2440'), ('disabled', PANEL)],
              foreground=[('active', GOLD), ('disabled', MUTED)], bordercolor=[('active', GOLD)])
        s.map('TCombobox', bordercolor=[('focus', GOLD), ('hover', theme.BRONZE)])
        s.map('TEntry', bordercolor=[('focus', GOLD)])
        s.configure('Play.TButton', background=ACCENT, foreground='#f4ece0', font=('Georgia' if WINDOWS else 'DejaVu Serif', 15, 'bold'),
                    padding=(36, 10), borderwidth=0)
        s.map('Play.TButton', background=[('active', ACCENT_HI), ('disabled', '#1f2a1a')],
              foreground=[('disabled', '#8a7a78')])
        s.configure('Horizontal.TProgressbar', background=ACCENT_HI, troughcolor=CARD, bordercolor=LINE,
                    lightcolor=ACCENT_HI, darkcolor=ACCENT)
        s.configure('Vertical.TScrollbar', background=CARD, arrowcolor=GOLD, troughcolor=PANEL, bordercolor=PANEL,
                    lightcolor=CARD, darkcolor=CARD)
        s.map('Vertical.TScrollbar', background=[('active', '#3a2f4a')])

    # ---- widget helpers ----------------------------------------------------------------------
    def var(self, key, store):
        """The Tk variable of a setting (one per setting, shared by every widget that shows it)."""
        tk = self.tk
        if key not in self.vars:
            if store == 'ini':
                value = self.ini.get(key, INI_DEFAULTS[key])
                if key in INI_FLAGS:
                    v = tk.BooleanVar(value=value == '1')
                else:
                    v = tk.StringVar(value=value)
            else:
                value = self.app.get(key, APP_DEFAULTS[key])
                if isinstance(APP_DEFAULTS[key], bool):
                    v = tk.BooleanVar(value=bool(value))
                else:
                    v = tk.StringVar(value=str(value))
            v.store = store
            self.vars[key] = v
        return self.vars[key]

    def vsync_choice_var(self):
        """VSync On / Off (the vsync key, also switched in the game) or a fixed present mode."""
        if 'vsync_choice' not in self.vars:
            vsync, mode = self.var('vsync', 'ini'), self.var('present_mode', 'ini')
            current = mode.get() if mode.get() in ('immediate', 'fifo_relaxed') else \
                'on' if mode.get() == 'fifo' else 'off' if mode.get() == 'mailbox' else \
                'on' if vsync.get() else 'off'
            choice = self.tk.StringVar(value=current)

            def write(*_args):
                value = choice.get()
                if value in ('on', 'off'):
                    mode.set('auto')
                    vsync.set(value == 'on')
                else:
                    mode.set(value)
            choice.trace_add('write', write)
            choice.store = 'derived'  # saved through vsync and present_mode
            self.vars['vsync_choice'] = choice
            if mode.get() in ('fifo', 'mailbox'):  # older fixed modes that only duplicated the switch
                write()
        return self.vars['vsync_choice']

    def engine_cap_var(self):
        """Engine frame rate with "Unlimited" (present_no_frame_caps), kept in sync with both keys."""
        if 'engine_cap' not in self.vars:
            engine, caps = self.var('engine_fps', 'ini'), self.var('present_no_frame_caps', 'ini')
            cap = self.tk.StringVar(value='unlimited' if caps.get() else engine.get())

            def write(*_args):
                unlimited = cap.get() == 'unlimited'
                caps.set(unlimited)
                if unlimited:
                    engine.set('240')
                else:
                    engine.set(cap.get())
            cap.trace_add('write', write)
            cap.store = 'derived'  # saved through engine_fps and present_no_frame_caps
            self.vars['engine_cap'] = cap
        return self.vars['engine_cap']

    def quality_choice_var(self):
        """Graphics quality: the level every graphics setting matches (graphics_options.matched_level; the
        picked level first, Deck Optimized while graphics_preset is deck), else 'custom', shown but not
        picked. Picking a level sets every setting to its choice; Deck Optimized writes DECK_PRESET and turns
        on the Deck switches; picking another level after it ends the preset and its switches."""
        if 'quality_choice' not in self.vars:
            quality, preset = self.var('ue4_quality', 'ini'), self.var('graphics_preset', 'app')
            settings = {key: self.var(key, 'ini') for key in graphics_options.GRAPHICS_KEYS}

            def shown():
                picked = DECK_QUALITY if preset.get() == DECK_QUALITY else quality.get()
                return graphics_options.matched_level({k: v.get() for k, v in settings.items()}, picked) or 'custom'
            choice = self.tk.StringVar(value=shown())
            busy, last = [False], [choice.get()]

            def follow(*_args):  # a setting changed: the level it now matches, or Custom
                if busy[0]:
                    return
                now = shown()
                if now != choice.get():
                    busy[0] = True
                    try:
                        choice.set(now)
                    finally:
                        busy[0] = False
                last[0] = now

            def pick(*_args):
                value, previous = choice.get(), last[0]
                last[0] = value
                if busy[0] or value == previous or value == 'custom':
                    return
                busy[0] = True
                try:
                    if value == DECK_QUALITY:
                        self.apply_deck_preset()
                    else:
                        if preset.get() == DECK_QUALITY:
                            preset.set('')
                            self.var('deck_switches', 'app').set(False)
                        quality.set(value)
                        for key, token in graphics_options.level_choices(value).items():
                            settings[key].set(token)
                finally:
                    busy[0] = False
                follow()
            choice.trace_add('write', pick)
            for var in settings.values():
                var.trace_add('write', follow)
            choice.store = 'derived'  # saved through ue4_quality, graphics_preset and the settings
            self.vars['quality_choice'] = choice
        return self.vars['quality_choice']

    def apply_deck_preset(self):
        """Deck Optimized: DECK_PRESET into the settings (saved to medievil.ini like any change) and the
        Deck switches on; the VSync and engine frame rate lists follow their keys."""
        for key, value in DECK_PRESET.items():
            if key not in INI_DEFAULTS:  # the in-game menu's own keys (INI_OPTIONAL_KEYS)
                self.ini[key] = value
            elif key in INI_FLAGS:
                self.var(key, 'ini').set(value == '1')
            else:
                self.var(key, 'ini').set(value)
        mode = DECK_PRESET['present_mode']
        self.vsync_choice_var().set(mode if mode in ('immediate', 'fifo_relaxed')
                                    else 'on' if DECK_PRESET['vsync'] == '1' else 'off')
        self.engine_cap_var().set('unlimited' if DECK_PRESET['present_no_frame_caps'] == '1'
                                  else DECK_PRESET['engine_fps'])
        self.var('graphics_preset', 'app').set(DECK_QUALITY)
        self.var('deck_switches', 'app').set(True)

    def offer_deck_preset(self):
        """On a Steam Deck, once (deck_offered, saved with the other settings): the first run selects Deck
        Optimized and says so, an existing setup is asked (deck_preset_offer)."""
        action = deck_preset_offer(self.app, self.first_run, steam_deck())
        if not action:
            return
        self.app['deck_offered'] = True
        if action == 'select':
            self.quality_choice_var().set(DECK_QUALITY)  # applies the preset (quality_choice_var)
            self.messagebox.showinfo(APP_NAME, _('Steam Deck detected: the Deck Optimized graphics preset is '
                                                 'selected (Graphics quality).') + '\n\n' + _(DECK_HINT))
        elif self.messagebox.askyesno(APP_NAME, _('Steam Deck detected. Use the Deck Optimized graphics preset? '
                                                  'It replaces your graphics settings; you can pick another '
                                                  'Graphics quality later.') + '\n\n' + _(DECK_HINT)):
            self.quality_choice_var().set(DECK_QUALITY)

    def deck_note(self, parent):
        """The Deck recommendation (DECK_HINT), shown while Deck Optimized is the graphics quality."""
        label = self.ttk.Label(parent, text=_(DECK_HINT), style='Warning.TLabel', wraplength=self.px(640),
                               justify='left')
        choice = self.quality_choice_var()

        def show(*_args):
            if choice.get() == DECK_QUALITY:
                label.grid()
            else:
                label.grid_remove()
        choice.trace_add('write', show)
        return label, show

    def choice(self, parent, key, store, options, width=30, custom=None):
        """A combobox over (value, (english, russian)) pairs, kept in sync with its variable (a key, or
        a variable itself). custom: a label for a value the list does not have (kept, not reset)."""
        var = key if store is None else self.var(key, store)
        if custom and var.get() not in [v for v, _t in options]:
            options = [*options, (var.get(), (custom(var.get()),))]
        values = [v for v, _t in options]
        box = self.ttk.Combobox(parent, values=[_(*text) for _v, text in options], state='readonly', width=width,
                                font=self.f('body', 10))
        if var.get() not in values:
            var.set(values[0])

        def show(*_args):
            current = var.get()
            if current in values:
                box.current(values.index(current))
            elif custom:  # set elsewhere (the render scale's percent box): shown, not reset
                box.set(custom(current))
            else:
                box.current(0)
        show()
        box.bind('<<ComboboxSelected>>', lambda _e: var.set(values[box.current()]))
        var.trace_add('write', show)
        return box

    def render_scale_control(self, parent):
        """Render scale: the presets, or any whole percent in RENDER_SCALE_RANGE typed or stepped in the
        box beside them (as the in-game menu's slider), and the scene size it gives at the output."""
        ttk = self.ttk
        holder = ttk.Frame(parent)
        var, output = self.var('render_scale', 'ini'), self.var('output_resolution', 'ini')
        self.choice(holder, 'render_scale', 'ini', RENDER_SCALE_CHOICES, width=26,
                    custom=self.custom_label('render_scale')).pack(side='left')
        low, high = graphics_options.RENDER_SCALE_RANGE
        text = self.tk.StringVar(value=var.get())
        ttk.Spinbox(holder, from_=low, to=high, increment=1, width=5, textvariable=text,
                    font=self.f('body', 10)).pack(side='left', padx=(8, 0))
        ttk.Label(holder, text='%').pack(side='left', padx=(2, 0))
        size = ttk.Label(holder, style='Muted.TLabel')
        size.pack(side='left', padx=(10, 0))

        def from_text(*_args):  # only whole percents in the range reach medievil.ini
            try:
                value = int(text.get().strip())
            except ValueError:
                return
            if low <= value <= high and str(value) != var.get():
                var.set(str(value))

        def from_setting(*_args):
            if text.get().strip() != var.get():
                text.set(var.get())
            size.configure(text=render_scale_scene(output.get(), var.get()))
        text.trace_add('write', from_text)
        var.trace_add('write', from_setting)
        output.trace_add('write', from_setting)
        from_setting()
        return holder

    @staticmethod
    def custom_label(key):
        """The label of a value set in medievil.ini that a list does not offer."""
        if key == 'output_resolution':
            return lambda value: f'Custom ({value})'
        if key == 'render_scale':
            return lambda value: f'Custom ({value}%)'
        return None

    def next_row(self, parent):
        return parent.grid_size()[1]

    def title_cell(self, parent, title, restart=False):
        """The name of a row; `restart` adds a mark: changing it needs a new start of the game."""
        cell = self.ttk.Frame(parent)
        self.ttk.Label(cell, text=title).pack(side='left')
        if restart:
            self.ttk.Label(cell, text=_('applies at next start'), style='Restart.TLabel').pack(side='left', padx=(8, 0))
        return cell

    def deck_switch_rows(self, parent):
        """Steam Deck performance switches: the checkbox turns them all on (every one checked again); the
        arrow shows them one by one (DECK_SWITCH_ITEMS, unchecked ones in deck_switches_off). On Windows
        the ones that do nothing there are unchecked and greyed out, with the reason."""
        ttk, top, off = self.ttk, self.var('deck_switches', 'app'), self.var('deck_switches_off', 'app')
        holder = ttk.Frame(parent)
        ttk.Checkbutton(holder, variable=top).pack(side='left')
        arrow = ttk.Button(holder, text=_('Show the switches ▸'))
        arrow.pack(side='left', padx=(12, 0))
        self.row(parent, _('Steam Deck performance switches'), holder,
                 _('Measured on the Steam Deck, applied at start; on with Deck Optimized. On Windows untested, '
                   'and the Linux-only ones are left out.'), restart=True)
        slot = ttk.Frame(parent)  # always in the grid, empty while the list is hidden
        slot.grid(row=self.next_row(parent), column=0, columnspan=2, sticky='w', padx=(24, 0))
        panel = ttk.Frame(slot)
        boxes = {}
        for i, (key, title, hint, _env, windows_reason) in enumerate(DECK_SWITCH_ITEMS):
            dead = WINDOWS and windows_reason
            item = self.tk.BooleanVar(value=not dead and key not in off.get().split(','))
            text = _(title) + (' — ' + _('does nothing on Windows ({})').format(_(windows_reason)) if dead else '')
            box = ttk.Checkbutton(panel, text=text, variable=item)
            box.grid(row=2 * i, column=0, sticky='w', pady=(4, 0))
            ttk.Label(panel, text=_(hint), style='Muted.TLabel', wraplength=self.px(600), justify='left').grid(
                row=2 * i + 1, column=0, sticky='w', padx=(22, 0))
            boxes[key] = (item, box, dead)

        def write_off(*_args):
            off.set(','.join(k for k, (v, _b, dead) in boxes.items() if not dead and not v.get()))

        def follow_top(*_args):
            if top.get():  # on: all of them again
                for v, _b, dead in boxes.values():
                    v.set(not dead)
            for v, b, dead in boxes.values():
                b.state(['disabled'] if dead or not top.get() else ['!disabled'])
        for v, _b, _dead in boxes.values():
            v.trace_add('write', write_off)
        top.trace_add('write', follow_top)
        for v, b, dead in boxes.values():
            b.state(['disabled'] if dead or not top.get() else ['!disabled'])

        def show_or_hide():
            if panel.winfo_ismapped():
                panel.pack_forget()
                arrow.configure(text=_('Show the switches ▸'))
            else:
                panel.pack(anchor='w')
                arrow.configure(text=_('Hide the switches ▾'))
        arrow.configure(command=show_or_hide)

    def row(self, parent, title, widget, hint=None, restart=False):
        """Name on the left, control on the right, an optional note below."""
        r = self.next_row(parent)
        self.title_cell(parent, title, restart).grid(row=r, column=0, sticky='w', padx=(0, 18), pady=(8, 0))
        widget.grid(row=r, column=1, sticky='w', pady=(5, 0))
        if hint:
            self.ttk.Label(parent, text=hint, style='Muted.TLabel', wraplength=self.px(640), justify='left').grid(
                row=r + 1, column=0, columnspan=2, sticky='w', pady=(1, 2))
        return widget

    def check(self, parent, key, store, title, hint=None, var=None, restart=False):
        box = self.ttk.Checkbutton(parent, variable=var if var is not None else self.var(key, store))
        return self.row(parent, title, box, hint, restart)

    def note(self, parent, text, top=12):
        self.ttk.Label(parent, text=text, style='Muted.TLabel', wraplength=self.px(680), justify='left').grid(
            row=self.next_row(parent), column=0, columnspan=2, sticky='w', pady=(top, 0))

    def section(self, parent, title, top=20, columns=2):
        r = self.next_row(parent)
        self.ttk.Label(parent, text=title, style='Section.TLabel').grid(
            row=r, column=0, columnspan=columns, sticky='w', pady=(top, 0))
        self.ornament(parent).grid(row=r + 1, column=0, columnspan=columns, sticky='w', pady=(0, 4))

    def ornament(self, parent):
        """A heading divider: a gold diamond and a fading line (theme.divider)."""
        try:
            image = self.photo(theme.divider(self.px(560), self.px(9)))
            return self.tk.Label(parent, image=image, bg=PANEL, bd=0)
        except ImportError:
            return self.tk.Frame(parent, bg=LINE, height=1, width=self.px(560))

    def folder(self, parent, key, title, prompt, hint=None, on_change=None):
        ttk = self.ttk
        var = self.var(key, 'app')
        holder = ttk.Frame(parent)
        ttk.Entry(holder, textvariable=var, width=46, font=self.f('body', 10)).pack(side='left')

        def browse():
            chosen = self.filedialog.askdirectory(title=prompt, initialdir=var.get() or str(PORT_DIR))
            if chosen:
                var.set(str(Path(chosen)))
        ttk.Button(holder, text=_('Browse…', 'Обзор…'), command=browse).pack(side='left', padx=(6, 0))
        ttk.Button(holder, text=_('Open', 'Открыть'), command=lambda: self.open_path(var.get(), key)).pack(
            side='left', padx=(6, 0))
        if on_change:
            var.trace_add('write', lambda *_a: on_change())
        return self.row(parent, title, holder, hint)

    def scrolled_page(self, name, title, subtitle):
        """A settings page: a heading and content that scrolls vertically."""
        tk, ttk = self.tk, self.ttk
        outer = ttk.Frame(self.content)
        head = ttk.Frame(outer, padding=(28, 22, 28, 6))
        head.pack(fill='x')
        image = self.gold(title, 30, role='head')
        if image:
            tk.Label(head, image=image, bg=PANEL, bd=0).pack(anchor='w', padx=(0, 0))
        else:
            ttk.Label(head, text=title.upper(), background=PANEL, foreground=GOLD, font=self.f('logo', 22)).pack(
                anchor='w')
        ttk.Label(head, text=subtitle, style='Muted.TLabel').pack(anchor='w')
        canvas = tk.Canvas(outer, bg=PANEL, highlightthickness=0, bd=0)
        bar = ttk.Scrollbar(outer, orient='vertical', command=canvas.yview)
        inner = ttk.Frame(canvas, padding=(28, 0, 28, 24))
        inner.columnconfigure(1, weight=1)
        window = canvas.create_window(0, 0, window=inner, anchor='nw')
        inner.bind('<Configure>', lambda _e: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.bind('<Configure>', lambda e: canvas.itemconfigure(window, width=e.width))
        canvas.configure(yscrollcommand=bar.set)
        bar.pack(side='right', fill='y')
        canvas.pack(side='left', fill='both', expand=True)
        outer.scroll = lambda units: canvas.yview_scroll(units, 'units') if inner.winfo_height() > canvas.winfo_height() else None
        self.pages[name] = outer
        return inner

    # ---- layout ------------------------------------------------------------------------------
    def build(self):
        tk, ttk = self.tk, self.ttk
        side = tk.Frame(self.root, bg=BG, width=self.px(220))
        side.pack(side='left', fill='y')
        side.pack_propagate(False)
        logo = self.gold(APP_NAME, 54, glow=ACCENT_HI, max_width=205)
        self.side_title = tk.Label(side, text='' if logo else APP_NAME.upper(), image=logo or '', bg=BG,
                                   fg=GOLD, font=self.f('logo', 20), bd=0)
        self.side_title.pack(anchor='center', pady=(22, 4))
        tk.Label(side, text=_('PC edition'), bg=BG, fg=ACCENT_HI, font=self.f('head', 12), bd=0).pack(
            anchor='center', pady=(0, 20))
        self.side = side
        self.nav, self.current_page = {}, None
        for name, title in (('home', _('Home')), ('display', _('Display')), ('performance', _('Performance')),
                            ('game', _('Game', 'Игра')), ('patches', _('Patches', 'Патчи')),
                            ('mods', _('Mods', 'Моды')), ('controls', _('Controls', 'Управление')),
                            ('advanced', _('Advanced', 'Дополнительно')), ('log', _('Log', 'Журнал'))):
            item = tk.Label(side, text='    ' + title, bg=BG, fg=TEXT, anchor='w', font=self.f('head', 13),
                            pady=8, cursor='hand2', bd=0)
            item.looks = self.nav_images(title)
            if item.looks:
                item.configure(text='', image=item.looks['normal'], pady=0)
            item.pack(fill='x')
            item.bind('<Button-1>', lambda _e, n=name: self.show(n))
            item.bind('<Enter>', lambda _e, n=name: n != self.current_page and self.nav_look(n, 'hover'))
            item.bind('<Leave>', lambda _e, n=name: n != self.current_page and self.nav_look(n, 'normal'))
            self.nav[name] = item
        tk.Frame(side, bg=BG).pack(fill='both', expand=True)
        self.update_box = None
        self.side_note = tk.Label(side, text=_('In the game: Insert or R3+L2\nopens the port\'s menu.',
                                               'В игре: Insert или R3+L2\nоткрывает меню порта.'),
                                  bg=BG, fg=MUTED, font=self.f('body', 9), justify='left', wraplength=self.px(210))
        self.side_note.pack(anchor='w', padx=22, pady=(0, 18))

        right = tk.Frame(self.root, bg=PANEL)
        right.pack(side='left', fill='both', expand=True)
        bar = tk.Frame(right, bg=BG, height=self.px(80))
        bar.pack(fill='x', side='bottom')
        bar.pack_propagate(False)
        tk.Frame(right, bg=theme.BRONZE, height=1).pack(fill='x', side='bottom')
        self.status = tk.Label(bar, text='', bg=BG, fg=MUTED, font=self.f('body', 10), anchor='w', justify='left')
        self.status.pack(side='left', padx=24)
        try:
            self.play_button = ImageButton(bar, tk, self.fonts, _('PLAY', 'ИГРАТЬ'), self.play,
                                           self.px(230), self.px(66), BG)
            self.play_button.pack(side='right', padx=(10, 18), pady=self.px(6))
        except ImportError:
            self.play_button = ttk.Button(bar, text=_('PLAY', 'ИГРАТЬ'), style='Play.TButton', command=self.play)
            self.play_button.pack(side='right', padx=(10, 24), pady=11)
        self.content = tk.Frame(right, bg=PANEL)
        self.content.pack(fill='both', expand=True)

        self.pages = {}
        self.build_home()
        self.build_display()
        self.build_performance()
        self.build_game()
        self.build_patches()
        self.build_mods()
        self.build_controls()
        self.build_advanced()
        self.build_log()
        self.root.bind_all('<MouseWheel>', self.wheel)
        self.game_changed()

    def nav_images(self, title):
        """A menu item drawn in the game's menu font: normal, hover and selected (soul glow)."""
        try:
            from PIL import Image
        except ImportError:
            return None
        w, h = self.px(220), self.px(42)
        looks = {}
        for look, color in (('normal', TEXT), ('hover', GOLD), ('selected', GOLD)):
            img = Image.new('RGBA', (w, h), theme._hex(BG))
            if look == 'selected':
                img.alpha_composite(theme.sidebar_glow(w, h))
            text = theme.gold_text(self.fonts, title, self.px(19), role='head', glow=None, outline=1) \
                if look != 'normal' else self.plain_text(title, self.px(19), color)
            img.alpha_composite(text, (self.px(18), (h - text.height) // 2))
            looks[look] = self.photo(img)
        return looks

    def plain_text(self, text, size, color):
        """Text in the game's menu font, one color, with a soft shadow (Pillow)."""
        from PIL import Image, ImageDraw, ImageFilter
        font = theme._font(self.fonts, 'head', size)
        box = ImageDraw.Draw(Image.new('L', (1, 1))).textbbox((0, 0), text, font=font)
        pad = max(4, size // 5)
        w, h = box[2] - box[0] + 2 * pad, box[3] - box[1] + 2 * pad
        out = Image.new('RGBA', (w, h), (0, 0, 0, 0))
        shadow = Image.new('L', (w, h), 0)
        ImageDraw.Draw(shadow).text((pad - box[0] + 1, pad - box[1] + 2), text, font=font, fill=200)
        out.paste((0, 0, 0, 255), (0, 0), shadow.filter(ImageFilter.GaussianBlur(1.5)))
        ImageDraw.Draw(out).text((pad - box[0], pad - box[1]), text, font=font, fill=theme._hex(color))
        return out

    def nav_look(self, name, look):
        item = self.nav[name]
        if item.looks:
            item.configure(image=item.looks[look])
        else:
            item.configure(bg=PANEL if look == 'selected' else '#1c1628' if look == 'hover' else BG,
                           fg=GOLD if look != 'normal' else TEXT)

    def wheel(self, event):
        page = self.pages.get(self.current_page)
        if hasattr(page, 'scroll') and not isinstance(event.widget, (self.tk.Text, self.ttk.Combobox)) \
                and 'popdown' not in str(event.widget):
            page.scroll(int(-event.delta / 120))

    def show(self, name):
        for page in self.pages.values():
            page.pack_forget()
        self.pages[name].pack(fill='both', expand=True)
        self.current_page = name
        for n in self.nav:
            self.nav_look(n, 'selected' if n == name else 'normal')
        if name in ('mods', 'patches'):
            self.refresh_lists()
        if name == 'patches':
            self.refresh_game_patches()
        elif name == 'home':
            self.refresh_status()

    def build_home(self):
        tk, ttk = self.tk, self.ttk
        page = tk.Frame(self.content, bg=PANEL)
        self.pages['home'] = page
        self.banner = tk.Canvas(page, height=self.px(390), bg=BG, highlightthickness=0, bd=0)
        self.banner.pack(fill='x')
        self.banner.bind('<Configure>', lambda _e: self.draw_banner())
        self.banner_button = ttk.Button(self.banner, text=_('Choose the game or its .pkg'),
                                        command=lambda: self.show('game'))
        quick = ttk.Frame(page, padding=(28, 14, 28, 8))
        quick.pack(fill='x')
        self.section(quick, _('Quick settings', 'Основное'), top=0, columns=4)
        items = ((_('Display mode'), 'display_mode', 'ini', DISPLAY_MODES),
                 (_('Frame rate limit'), 'fps_limit', 'ini', FPS_LIMITS),
                 (_('Output resolution'), 'output_resolution', 'ini', OUTPUT_CHOICES),
                 (_('Render scale'), 'render_scale', 'ini', RENDER_SCALE_CHOICES),
                 (_('Graphics quality'), self.quality_choice_var(), None, QUALITY_PRESETS),
                 (_('VSync'), self.vsync_choice_var(), None, VSYNC_CHOICES))
        for index, (title, key, store, options) in enumerate(items):
            r, c = 2 + index // 2, (index % 2) * 2
            ttk.Label(quick, text=title).grid(row=r, column=c, sticky='w', pady=6, padx=(0, 12))
            quick.grid_rowconfigure(r, minsize=self.px(34))
            # Graphics quality shows Custom when the settings match no level (shown, never reset to a level).
            custom = (lambda _value: _('Custom')) if key is self.vars.get('quality_choice') else self.custom_label(key)
            self.choice(quick, key, store, options, width=26, custom=custom).grid(
                row=r, column=c + 1, sticky='w', padx=(0, 40))
        flags = ttk.Frame(quick)
        flags.grid(row=2 + (len(items) + 1) // 2, column=0, columnspan=4, sticky='w', pady=(6, 0))
        # VSync is the present mode above (its first choice follows the in-game VSync switch).
        for key, title in (('show_fps', _('Show performance overlay')),):
            ttk.Checkbutton(flags, text=title, variable=self.var(key, 'ini')).pack(side='left', padx=(0, 28))
        self.link_buttons(quick).grid(row=5, column=0, columnspan=4, sticky='w', pady=(14, 0))
        note, show = self.deck_note(quick)
        note.grid(row=6, column=0, columnspan=4, sticky='w', pady=(12, 0))
        show()

    def link_buttons(self, parent):
        """GitHub, YouTube and Patreon buttons that open the port's pages in the browser."""
        ttk = self.ttk
        holder = ttk.Frame(parent)
        for index, (name, url) in enumerate(LINKS):
            ttk.Button(holder, text=name, command=lambda u=url: webbrowser.open(u)).pack(
                side='left', padx=(0 if index == 0 else 8, 0))
        ttk.Label(holder, text=_('Support the port on Patreon', 'Поддержите порт на Patreon'),
                  style='Muted.TLabel').pack(side='left', padx=12)
        return holder

    def draw_banner(self):
        """The cover art of the selected dump (sce_sys/pic1.png) under the title."""
        tk, c = self.tk, self.banner
        c.delete('all')
        w, h = max(c.winfo_width(), 400), int(c['height'])
        folder = self.var('game_dir', 'app').get()
        info = game_info(folder)
        art = Path(folder or '.') / 'sce_sys' / 'pic1.png'
        if self.banner_source != art:
            self.banner_source, self.banner_image, self.banner_size = art, None, None
            try:
                self.banner_image = tk.PhotoImage(file=str(art))
            except (tk.TclError, OSError):
                pass
        if self.banner_image:
            if self.banner_size != (w, h):
                self.banner_size, self.banner_scaled = (w, h), self.scaled_art(art, w, h)
            c.create_image(0, 0, image=self.banner_scaled, anchor='nw')
        else:
            logo = self.gold(APP_NAME, 56, glow=ACCENT_HI)
            if logo:
                c.create_image(w // 2, h // 2 - self.px(30), image=logo)
            else:
                c.create_text(30, h - 74, text=APP_NAME.upper(), anchor='w', fill=GOLD, font=self.f('logo', 32))
        if info:
            title_id = game_profile_of(folder)[0]
            sub = _('{} · version {}', '{} · версия {}').format(title_id or '?', info[1])
            c.create_text(34, h - 26, text=sub, anchor='w', fill='#000000', font=self.f('head', 14))
            c.create_text(33, h - 28, text=sub, anchor='w', fill=ACCENT_HI, font=self.f('head', 14))
        else:
            c.create_text(w // 2, h // 2 + self.px(20), text=_('Choose your game folder to play.'),
                          fill=TEXT, font=self.f('body', 12))
            c.create_window(w // 2, h // 2 + self.px(60), window=self.banner_button)

    def scaled_art(self, path, width, height):
        """The cover art filling width x height and fading into the page (theme.banner_fade), or
        Tk's integer subsampling without Pillow."""
        try:
            from PIL import Image, ImageTk
            with Image.open(path) as image:
                return ImageTk.PhotoImage(theme.banner_fade(image, width, height, PANEL))
        except (ImportError, OSError):
            factor = max(1, round(self.banner_image.width() / max(width, 1)))
            return self.banner_image.subsample(factor) if factor > 1 else self.banner_image

    def build_display(self):
        tk, ttk = self.tk, self.ttk
        f = self.scrolled_page('display', _('Display'), _('How the game appears on your screen, and its frame rate.'))
        self.section(f, _('Window', 'Окно'), top=4)
        self.row(f, _('Display mode'), self.choice(f, 'display_mode', 'ini', DISPLAY_MODES),
                 _('Borderless covers the screen without changing the display mode.'))
        self.row(f, _('VSync'), self.choice(f, self.vsync_choice_var(), None, VSYNC_CHOICES), _(VSYNC_HINT),
                 restart=True)
        self.check(f, 'hdr', 'app', _('Allow HDR output', 'Разрешить HDR'),
                   _('When HDR is on in Windows and the display supports it.',
                     'Если HDR включён в Windows и монитор его поддерживает.'))
        self.section(f, _('Frame rate', 'Частота кадров'))
        self.row(f, _('Engine frame rate'), self.choice(f, self.engine_cap_var(), None, ENGINE_CAP_CHOICES),
                 _('The highest frame rate MediEvil runs at. The PS4 game stops at 60; 120 and 240 lift '
                   'that limit; Unlimited also removes the port\'s own caps (only Frame rate limit applies). '
                   'Applied at start.'), restart=True)
        self.row(f, _('Frame rate limit'), self.choice(f, 'fps_limit', 'ini', FPS_LIMITS),
                 _('The Auto choices follow your display: with G-Sync / FreeSync they stay just below the refresh '
                   '("Auto: half refresh" = 119 FPS on 240 Hz; "Auto: refresh" = NVIDIA Reflex\'s caps, 138 on '
                   '144 Hz, 225 on 240 Hz; a cap exactly at half the refresh flickered on a 240 Hz OLED); on a '
                   'fixed-refresh display they are exactly the refresh or half of it (use VSync on).'))
        self.check(f, 'display_vrr', 'ini', _('G-Sync / FreeSync display (VRR)'),
                   _('On: the automatic frame rate limits stay just below the refresh. Off: a fixed-refresh '
                     'display, where they match the refresh exactly.'))
        self.row(f, _('Frame limiter'), self.choice(f, 'present_limiter', 'ini', PRESENT_LIMITERS),
                 _('NVIDIA Reflex: the graphics driver paces the frames (as RTSS\'s Reflex limiter does): the '
                   'most even frame times. Falls back to Precise on AMD and Intel. Precise: the port\'s own '
                   'high-resolution limiter. Standard: the original limiter.'), restart=True)
        self.row(f, _('Frames queued'), self.choice(f, 'frames_queued', 'ini', FRAMES_QUEUED),
                 _('1 is the lowest input latency; more can raise FPS when the graphics card is the limit.'))
        self.section(f, _('Advanced presentation'))
        self.note(f, _('How finished frames reach the screen. The defaults are the tested best; change them '
                       'to compare. Applied at the next start.'), top=2)
        tick = self.ttk.Spinbox(f, from_=0, to=2000, increment=1, width=8, font=self.f('body', 10),
                                textvariable=self.var('present_tick_hz', 'ini'))
        self.row(f, _('Vblank tick rate (Hz)'), tick,
                 _('How often the game\'s virtual vblank fires (default 960): the PS4 game waits for it before '
                   'each frame. 0 = automatic (480 Hz above 60 FPS, else 60). A rate at or below the frame rate '
                   'limit paces the game by itself (120 = 120 FPS) and rounds slow frames up to whole ticks.'),
                 restart=True)
        for key, title, hint in PRESENT_SWITCHES:
            self.check(f, key, 'ini', _(title), _(hint), restart=True)
        self.section(f, _('Scaling and sharpening'))
        self.check(f, 'fsr1', 'ini', _('Upscale with FSR 1'),
                   _('Enlarges a smaller game image to the window. With Render resolution 720p (Performance) '
                     'it raises the FPS; at the size of the window it does nothing.'))
        self.check(f, 'rcas', 'ini', _('Final image sharpening (RCAS)'))
        self.check(f, 'ue4_dlss', 'ini', _('NVIDIA DLSS (replaces the game\'s anti-aliasing)'),
                   _('DLAA at Render scale 100, DLSS upscaling below it (Quality / Balanced / Performance from '
                     'the scale). Sharpening is off while it runs. Needs an NVIDIA RTX GPU and gow3_dlss.dll + '
                     'nvngx_dlss.dll next to medievil-runtime.exe (the bin folder; nvngx_dlss.dll is not included: see '
                     'README-Windows.txt). F9 switches it in the game.'))
        holder = ttk.Frame(f)
        strength = self.var('rcas_strength', 'ini')
        tk.Scale(holder, from_=0, to=100, orient='horizontal', variable=strength, showvalue=False, length=self.px(260),
                 width=self.px(12), sliderlength=self.px(18), bg=ACCENT, troughcolor=CARD, activebackground=ACCENT_HI,
                 highlightthickness=0, bd=0, sliderrelief='flat').pack(side='left')
        value = ttk.Label(holder, width=5)
        value.pack(side='left', padx=10)
        show = lambda *_a: value.configure(text=f'{strength.get()}%')
        strength.trace_add('write', show)
        show()
        self.row(f, _('Sharpening strength'), holder)
        self.section(f, _('Overlay'))
        self.check(f, 'show_fps', 'ini', _('Show performance overlay'),
                   _('Items, size and corner: in-game menu (Insert / R3+L2) > Overlay.'))
        self.section(f, _('Screenshots'))
        holder = ttk.Frame(f)
        holder.grid(row=self.next_row(f), column=0, columnspan=2, sticky='w', pady=(6, 0))
        ttk.Button(holder, text=_('Open screenshots folder'),
                   command=lambda: self.open_path(screenshots_dir(self.app))).pack(side='left')
        ttk.Label(holder, text=_('Press F12 in the game: a PNG of the game picture at its full resolution.'),
                  style='Muted.TLabel').pack(side='left', padx=10)

    def build_performance(self):
        f = self.scrolled_page('performance', _('Performance'), _('Graphics and speed options.'))
        self.section(f, _('Graphics settings'), top=4)
        self.note(f, _('Graphics quality (below) sets all of these; changing one makes it Custom. "PS4 default" '
                       'marks the value the PS4 itself uses. The in-game menu switches most of them while playing.'),
                  top=2)
        for key, title, hint, choices in GRAPHICS_OPTIONS:
            self.row(f, _(title), self.choice(f, key, 'ini', [(value, (label,)) for value, label in choices]),
                     _(hint), restart=True)
        self.section(f, _('Performance patches'))
        self.note(f, _('On by default except the single particle pass. Switch one off to compare, or if '
                       'something looks wrong; applied at the next start.'), top=2)
        for key, title, hint, _off in PERF_PATCHES:
            self.check(f, key, 'ini', _(title), _(hint), restart=True)
        self.section(f, _('Resolution and quality'))
        self.row(f, _('Graphics quality'), self.choice(f, self.quality_choice_var(), None, QUALITY_PRESETS,
                                                       custom=lambda _value: _('Custom')),
                 _('Sets every graphics setting above. Console: exactly the PS4. High and Ultra go beyond it: '
                   'sharper filtering, farther view and detail, better reflections, for strong graphics cards. Medium '
                   'and Low lower shadows, effects and distances: for integrated graphics (Radeon 780M, Intel Arc) '
                   'and older cards. Custom: some settings changed by hand.') + ' ' +
                 _('Deck Optimized: the Steam Deck settings of a flat 60 FPS (720p, 80% render scale, Low with '
                   'volumetric fog, a 60 FPS limit, VSync off) and the Steam Deck performance switches below.'),
                 restart=True)
        note, show = self.deck_note(f)
        note.grid(row=self.next_row(f), column=0, columnspan=2, sticky='w', pady=(4, 2))
        show()
        self.deck_switch_rows(f)
        self.row(f, _('Effects resolution'), self.choice(f, 'effects_resolution', 'ini', EFFECTS_CHOICES),
                 _('Smoke, fire and enemy death effects drawn at a lower resolution and blended in. At 4K '
                   'several enemy deaths at once cost 2-3 ms a frame at 100%; 50% matches the effect detail '
                   'of the PS4 at 1080p. The cost grows with the render resolution. Applied at start; the in-game menu '
                   'also switches it while playing.'), restart=True)
        self.row(f, _('Output resolution'), self.choice(f, 'output_resolution', 'ini', OUTPUT_CHOICES,
                                                        custom=self.custom_label('output_resolution')),
                 _(OUTPUT_HINT), restart=True)
        self.row(f, _('Render scale'), self.render_scale_control(f), _(RENDER_SCALE_HINT), restart=True)
        self.row(f, _('Anti-aliasing'), self.choice(f, 'ue4_aa', 'ini', AA_CHOICES),
                 _("The game's temporal AA smooths edges but blurs a little in motion. FXAA is sharper, Off "
                   'shows raw pixels (best with a render scale above 100%). A render scale below 100% always '
                   'uses temporal AA. Applied at start.'), restart=True)
        self.section(f, _('Shaders'))
        self.check(f, 'pipeline_binary_cache', 'app', _('Use pipeline binary cache (A/B test)'),
                   _('Reuse saved Vulkan pipelines. Disable to compare FPS with normal compilation. '
                     'Takes effect on the next launch; saved caches are preserved.'))
        self.check(f, 'background_warmup', 'app', _('Load shader cache in the background'),
                   _('The game opens at once and the saved shaders load while you play; a small note in the '
                     'corner shows the progress. Switch it off to wait on the loading screen instead.'))
        self.check(f, 'async_shaders', 'ini', _('Compile new shaders in the background'),
                   _('Fewer stutters in new areas: the game keeps running while new shaders compile. An object '
                     'or effect may appear a moment later. Switch it off if something stays missing.'))
        self.check(f, 'parallel_warmup', 'app', _('Fast shader cache loading (parallel)'),
                   _('Loading screen only: builds the cached pipelines on several CPU threads.'))
        self.section(f, _('GPU data'))
        self.check(f, 'stale_readback', 'ini', _('Game reads GPU data without waiting (big FPS gain)'),
                   _('About 40 to 66 FPS in a heavy scene. Needs deferred GPU readbacks. Switch it off if lighting '
                     'or objects look wrong.'), restart=True)
        self.check(f, 'deferred_readback', 'ini', _('Deferred GPU readbacks'),
                   _('Fewer waits for the GPU. Switch it off if textures or shadows look wrong.'), restart=True)
        self.row(f, _('Two-stage GPU pipeline', 'Двухстадийный конвейер GPU'), self.choice(f, 'draw_pipe', 'app', DRAW_PIPE),
                 _('20–30% faster; switch it off if the game is unstable.', 'Быстрее на 20–30%; при нестабильности выключите.'))
        self.row(f, _('Draw preparation helpers'), self.choice(f, 'prep_workers', 'app', PREP_WORKERS),
                 _('Threads that prepare draws ahead of the GPU thread. Fewer can help on CPUs with few cores '
                   '(Steam Deck: 4). Applied at start.'))
        self.check(f, 'stage_b_core', 'app', _('Dedicated core for the GPU thread'),
                   _('Gives the thread that records draws a CPU core of its own; every other thread uses the '
                     'remaining cores. Experimental. Applied at start.'))
        self.row(f, _('GPU readbacks', 'Чтение данных GPU'), self.choice(f, 'readbacks', 'app', READBACKS),
                 _('How exactly data the GPU writes is copied back for the game.',
                   'Насколько точно данные, записанные GPU, возвращаются игре.'))
        self.check(f, 'vulkan_compat', 'app', _('Safe GPU mode (older AMD GPUs)',
                                                'Безопасный режим GPU (старые видеокарты AMD)'),
                   _('For Radeon RX 400/500 and Vega GPUs that crash when the main menu loads: runs without newer '
                     'Vulkan features. Can be slower; leave it off on other GPUs.',
                     'Для Radeon RX 400/500 и Vega, на которых игра вылетает при загрузке главного меню: работает '
                     'без новых функций Vulkan. Может быть медленнее; на других видеокартах не включайте.'),
                   restart=True)

    def build_game(self):
        f = self.scrolled_page('game', _('Game', 'Игра'), _('Your game (folder or .pkg), saves and language.'))
        self.section(f, _('Install from PKG'), top=4)
        self.build_pkg_install(f)
        self.section(f, _('Folders'))
        self.folder(f, 'game_dir', _('Game folder', 'Папка игры'),
                    _('Choose the folder with eboot.bin', 'Выберите папку с eboot.bin'),
                    _('Your own copy of MediEvil (CUSA11227): the folder with eboot.bin, sce_module and sce_sys. '
                      'Installing a .pkg (above) fills this in.'), on_change=self.game_changed)
        self.folder(f, 'user_dir', _('Saves folder', 'Папка сохранений'),
                    _('Choose the saves folder', 'Выберите папку сохранений'),
                    _('Empty: {} (shader caches are kept there too).',
                      'Пусто: {} (там же кэш шейдеров).').format(DATA_DIR / 'user'), on_change=self.refresh_status)
        self.section(f, _('Saves'))
        self.build_save_backups(f)
        self.section(f, _('Language'))
        self.row(f, _('Game language', 'Язык игры'), self.choice(f, 'language', 'app', LANGUAGES))
        self.row(f, _('Voice language'), self.choice(f, 'ue4_voice', 'ini', VOICE_LANGUAGES),
                 _('The spoken language: 15 are in the game. "Same as game language" follows the game '
                   'language (English voices for languages without a dub). Applied at start.'), restart=True)
        self.section(f, _('Cheats'))
        self.check(f, 'cheat_menu', 'ini', _('Developer cheat menu (R3 in a level)'),
                   _("MediEvil's own debug menu, left in the game by its developers: level select, complete "
                     'level, weapons and items for the level, runes, gold, life bottles and more. Press R3 '
                     '(click the right stick) in a level; on 01.02 hold L1 + R1, then press R3. '
                     'Applied at start.'))
        self.check(f, 'cheat_infinite_ammo', 'ini', _('Unlimited ammo'),
                   _('Daggers, crossbow and longbow bolts, lightning and the other weapons never run out, and '
                     'weapons and shields do not wear out. Applied at start.'))

    def user_dir(self):
        return Path(self.var('user_dir', 'app').get() or DATA_DIR / 'user')

    def build_pkg_install(self, f):
        """A fake PKG (fPKG) is unpacked once into <its folder>/<TITLE_ID>; that becomes the game folder."""
        ttk = self.ttk
        holder = ttk.Frame(f)
        self.pkg_button = ttk.Button(holder, text=_('Choose .pkg…'), command=self.choose_pkg)
        self.pkg_button.pack(side='left')
        self.pkg_progress = ttk.Progressbar(holder, length=self.px(260), maximum=1000)
        self.pkg_progress.pack(side='left', padx=12)
        self.row(f, _('Game package'), holder,
                 _('Unpacks the game once (about 32 GB for MediEvil) next to the .pkg, then plays from that folder. '
                   'Fake PKGs only; an interrupted unpack continues where it stopped.'))
        self.pkg_label = ttk.Label(f, text='', style='Muted.TLabel', wraplength=self.px(640), justify='left')
        self.pkg_label.grid(row=self.next_row(f), column=0, columnspan=2, sticky='w')
        self.pkg_busy = False

    def choose_pkg(self):
        if self.pkg_busy:
            return
        last = self.app.get('last_pkg') or ''
        chosen = self.filedialog.askopenfilename(
            title=_('Choose the game .pkg'), filetypes=[('PS4 package', '*.pkg'), ('All files', '*.*')],
            initialdir=str(Path(last).parent) if last else str(PORT_DIR.parent))
        if chosen:
            self.install_pkg(Path(chosen))

    def install_pkg(self, pkg):
        import pkg_extract
        try:
            target = Path(pkg_extract.default_target(pkg))
        except (OSError, ValueError) as error:
            self.messagebox.showerror(APP_NAME, _('This is not a PKG that can be unpacked: {}').format(error))
            return
        self.app['last_pkg'] = str(pkg)
        if pkg_extract.is_extracted(pkg, target):
            self.var('game_dir', 'app').set(str(target))
            self.pkg_progress.configure(value=1000)
            self.pkg_label.configure(text=_('Already unpacked in {}.').format(target))
            return
        free = shutil.disk_usage(target.parent).free
        if not self.messagebox.askyesno(APP_NAME, _(
                'Unpack {} into {}?\n\nFree space on that drive: {} GB. MediEvil needs about 32 GB.').format(
                    pkg.name, target, free >> 30)):
            return
        self.pkg_busy = True
        self.pkg_button.configure(state='disabled')
        self.play_button.configure(state='disabled')
        self.pkg_label.configure(text=_('Reading {}…').format(pkg.name))

        def report(done, total, rate):
            def show():
                self.pkg_progress.configure(value=done * 1000 // max(total, 1))
                self.pkg_label.configure(text=_('Unpacking: {:.1f} of {:.1f} GB ({:.0f} MB/s)').format(
                    done / 2**30, total / 2**30, rate))
            self.ui_calls.put(show)

        def work():
            error = None
            try:
                pkg_extract.extract(pkg, target, report)
            except (OSError, ValueError) as failure:
                error = failure

            def done():
                self.pkg_busy = False
                self.pkg_button.configure(state='normal')
                if error:
                    self.pkg_label.configure(text=_('Unpacking stopped: {}. Choose the .pkg again to continue.').format(error))
                    self.messagebox.showerror(APP_NAME, _('Could not unpack: {}').format(error))
                else:
                    self.pkg_progress.configure(value=1000)
                    self.pkg_label.configure(text=_('Installed in {}.').format(target))
                    self.var('game_dir', 'app').set(str(target))
                self.refresh_status()
            self.ui_calls.put(done)
        threading.Thread(target=work, daemon=True).start()

    def build_save_backups(self, f):
        """run.py backs the saves up before every start; one of them can be restored here."""
        ttk = self.ttk
        holder = ttk.Frame(f)
        self.backup_choice = ttk.Combobox(holder, state='readonly', width=30)
        self.backup_choice.pack(side='left')
        ttk.Button(holder, text=_('Restore', 'Восстановить'), command=self.restore_backup).pack(side='left', padx=(6, 0))
        ttk.Button(holder, text=_('Refresh', 'Обновить'), command=self.refresh_backups).pack(side='left', padx=(6, 0))
        self.row(f, _('Save backups', 'Резервные копии'), holder,
                 _('Made before every start, newest first; the last {} are kept. Restoring backs up the '
                   'current saves first.').format(save_backup.KEEP))
        self.refresh_backups()

    def refresh_backups(self):
        names = save_backup.list_backups(self.user_dir())
        self.backup_choice.configure(values=names)
        self.backup_choice.set(names[0] if names else '')

    def restore_backup(self):
        name = self.backup_choice.get()
        if not name:
            return
        if self.process:
            self.messagebox.showwarning(APP_NAME, _('Close the game before restoring saves.'))
            return
        if not self.messagebox.askyesno(APP_NAME, _('Replace the current saves with the backup {}? '
                                                    'The current saves are backed up first.').format(name)):
            return
        try:
            safety = save_backup.restore(self.user_dir(), name)
        except (OSError, ValueError) as error:
            self.messagebox.showerror(APP_NAME, _('Could not restore: {}').format(error))
            return
        self.refresh_backups()
        self.refresh_status()
        self.messagebox.showinfo(APP_NAME, _('Saves restored from {}.').format(name) +
                                 (_(' Previous saves: {}.').format(safety.name) if safety else ''))

    def build_patches(self):
        """Community patches of the selected game's built-in XML and third-party patch files.
        Resolution and frame rate patches follow Performance (medievil.ini; run.py adds them)."""
        f = self.scrolled_page('patches', _('Patches', 'Патчи'), _('Changes applied to the game code at start.'))
        self.section(f, _('Game patches', 'Патчи игры'), top=4)
        self.game_patch_frame = self.ttk.Frame(f)
        self.game_patch_frame.grid(row=self.next_row(f), column=0, columnspan=2, sticky='we')
        self.game_patch_title, self.game_patch_vars = None, {}
        self.section(f, _('Third-party patches', 'Сторонние патчи'))
        self.folder(f, 'patches_dir', _('Patches folder', 'Папка патчей'), _('Choose the patches folder', 'Выберите папку патчей'),
                    _('shadPS4/GoldHEN XML patch files for this game version. Empty: {}',
                      'XML-патчи shadPS4/GoldHEN для этой версии игры. Пусто: {}').format(DATA_DIR / 'patches'),
                    on_change=self.refresh_lists)
        self.patches_frame = self.ttk.Frame(f)
        self.patches_frame.grid(row=self.next_row(f), column=0, columnspan=2, sticky='we', pady=(8, 0))
        self.ttk.Button(f, text=_('Refresh', 'Обновить'), command=self.refresh_lists).grid(
            row=self.next_row(f), column=0, sticky='w', pady=(14, 0))

    def refresh_game_patches(self):
        import xml.etree.ElementTree as ET
        tk, f = self.tk, self.game_patch_frame
        self.store_game_patches()
        for widget in f.winfo_children():
            widget.destroy()
        f.columnconfigure(1, weight=1)
        folder = self.var('game_dir', 'app').get()
        title_id, profile = game_profile_of(folder)
        self.game_patch_title, self.game_patch_vars = None, {}
        if not profile:
            self.note(f, _('No built-in patches for this game.', 'Для этой игры нет встроенных патчей.'), top=0)
            return
        _ids, xml, version = profile[:3]
        metas = [m for m in ET.parse(xml).getroot().iter('Metadata') if m.get('AppVer') == version]
        chosen = game_patch_selection(self.app, title_id, profile)
        on = set(chosen) if chosen is not None else {m.get('Name') for m in metas
                                                         if m.get('isEnabled', 'false').lower() == 'true'}
        self.note(f, _('{} · version {} · {}. Resolution and frame rate patches follow Performance.',
                       '{} · версия {} · {}.').format(title_id, version, xml.name), top=0)
        for meta in metas:
            name = meta.get('Name')
            # Chosen by run.py: the resolution and frame rate patches, and the output size ones
            # (Performance > Output resolution).
            if chosen_by_run_py(name):
                continue
            var = tk.BooleanVar(value=name in on)
            self.game_patch_vars[name] = var
            author = meta.get('Author')
            self.check(f, None, None, name + (f'  ({author})' if author else ''), meta.get('Note'), var=var)
        self.game_patch_title = title_id

    def store_game_patches(self):
        """The page's selection into settings (the whole list for this game, without the resolution
        and frame rate patches: run.py adds those from medievil.ini)."""
        if not self.game_patch_title:
            return
        self.app.setdefault('game_patches', {})[self.game_patch_title] = \
            [n for n, v in self.game_patch_vars.items() if v.get()]
        # What the page showed: a default patch added later starts on (game_patch_selection).
        seen = self.app.setdefault('game_patches_seen', {})
        seen[self.game_patch_title] = sorted(set(seen.get(self.game_patch_title, [])) | set(self.game_patch_vars))

    def build_mods(self):
        f = self.scrolled_page('mods', _('Mods', 'Моды'),
                               _('Files layered over the game at start; the game files are never changed.'))
        self.check(f, 'mods_enabled', 'app', _('Load mods', 'Загружать моды'),
                   _('Loose-file mods, each in its own folder with dvdroot_ps4.',
                     'Моды из файлов, каждый в своей папке с dvdroot_ps4.'))
        self.folder(f, 'mods_dir', _('Mods folder', 'Папка модов'), _('Choose the mods folder', 'Выберите папку модов'),
                    _('Empty: {}', 'Пусто: {}').format(DATA_DIR / 'mods'), on_change=self.refresh_lists)
        self.section(f, _('Mods', 'Моды'))
        self.mods_frame = self.ttk.Frame(f)
        self.mods_frame.grid(row=self.next_row(f), column=0, columnspan=2, sticky='we', pady=(8, 0))
        self.ttk.Button(f, text=_('Refresh', 'Обновить'), command=self.refresh_lists).grid(
            row=self.next_row(f), column=0, sticky='w', pady=(14, 0))

    def build_controls(self):
        """Remapping of the PS4 buttons (run.py passes GOW3_KEY_MAP and GOW3_PAD_MAP to the game)."""
        tk, ttk = self.tk, self.ttk
        f = self.scrolled_page('controls', _('Controls', 'Управление'),
                               _('Which key or gamepad button presses each button of the game.'))
        saved = self.app.get('controls') if isinstance(self.app.get('controls'), dict) else {}
        style = self.app.get('pad_style') if self.app.get('pad_style') in PAD_STYLES else 'playstation'
        self.pad_style = tk.StringVar(value=style)
        names = ttk.Frame(f)
        for value, label in (('playstation', 'PlayStation'), ('xbox', 'Xbox')):
            ttk.Radiobutton(names, text=label, value=value, variable=self.pad_style,
                            command=self.relabel_pad_controls).pack(side='left', padx=(0, 14))
        self.row(f, _('Gamepad names'), names)
        self.row(f, _('Stick range'), self.choice(f, 'stick_range', 'app', STICK_RANGES),
                 _('How far the sticks must tilt for a full push. Lower it if your character moves slowly '
                   'with the stick pushed all the way (some hall-effect sticks never reach the edge).'))
        table = ttk.Frame(f)
        table.grid(row=self.next_row(f), column=0, columnspan=2, sticky='w', pady=(16, 0))
        for column, span, title in ((0, 2, _('GAME BUTTON')), (2, 1, _('KEYBOARD')), (3, 2, _('GAMEPAD'))):
            ttk.Label(table, text=title, style='Section.TLabel').grid(row=0, column=column, columnspan=span, sticky='w')
        self.control_vars, self.pad_boxes, self.pad_icons = {'key': {}, 'pad': {}}, {}, {}
        size = self.px(28)
        labels = PAD_STYLES[style]

        def icon_canvas(parent):
            return tk.Canvas(parent, width=size, height=size, bg=PANEL, highlightthickness=0, bd=0)
        for r, (button, key, pad) in enumerate(CONTROLS, start=1):
            game_icon = icon_canvas(table)
            game_icon.grid(row=r, column=0, sticky='w', pady=3)
            button_icons.draw(game_icon, button_icons.GAME_BUTTONS[button], 'playstation', size)
            ttk.Label(table, text=PS_BUTTON_NAMES[button], width=14).grid(row=r, column=1, sticky='w', padx=(0, 12))
            key_var = tk.StringVar(value=saved.get('key', {}).get(button) or key)
            self.control_vars['key'][button] = key_var
            ttk.Entry(table, textvariable=key_var, width=16, font=self.f('body', 10)).grid(row=r, column=2, sticky='w', padx=(0, 18))
            pad_name = saved.get('pad', {}).get(button) or pad
            pad_var = tk.StringVar(value=pad_name or '')
            self.control_vars['pad'][button] = pad_var
            pad_icon = icon_canvas(table)
            pad_icon.grid(row=r, column=3, sticky='w', padx=(0, 6))
            self.pad_icons[button] = pad_icon
            if pad:
                box = ttk.Combobox(table, state='readonly', width=22, values=list(labels.values()))
                box.set(labels.get(pad_name, pad_name))

                def chosen(_e, b=box, v=pad_var, n=button):
                    v.set(self.pad_name(b.get()))
                    self.draw_pad_icon(n)
                box.bind('<<ComboboxSelected>>', chosen)
                self.pad_boxes[button] = box
            else:
                box = ttk.Label(table, text=_('{} trigger').format(PS_BUTTON_NAMES[button]), style='Muted.TLabel')
            box.grid(row=r, column=4, sticky='w')
            self.draw_pad_icon(button)
        self.note(f, _('Keys use SDL names: letters, digits, Space, Return, Tab, Left Shift, Left Ctrl, Left Alt, '
                       'Backspace, Up, Down, Left, Right, F1 to F12. A name the game does not know keeps the default '
                       '(the log says which). L2/R2 on the gamepad stay on the triggers. Insert and R3+L2 always open '
                       'the port\'s menu. Applied when the game starts.'))
        self.ttk.Button(f, text=_('Reset to defaults'), command=self.reset_controls).grid(
            row=self.next_row(f), column=0, sticky='w', pady=(14, 0))

    def draw_pad_icon(self, button):
        """The gamepad column's icon of a game button, in the chosen style (L2/R2: the triggers)."""
        name = self.control_vars['pad'][button].get() or button_icons.GAME_BUTTONS[button]
        button_icons.draw(self.pad_icons[button], name, self.pad_style.get(), self.px(28))

    def pad_name(self, label):
        """SDL gamepad button name of a label in the current naming style."""
        return next((n for n, l in PAD_STYLES[self.pad_style.get()].items() if l == label), label)

    def relabel_pad_controls(self):
        labels = PAD_STYLES[self.pad_style.get()]
        for button, box in self.pad_boxes.items():
            box.configure(values=list(labels.values()))
            name = self.control_vars['pad'][button].get()
            box.set(labels.get(name, name))
        for button in self.pad_icons:
            self.draw_pad_icon(button)

    def reset_controls(self):
        for button, key, pad in CONTROLS:
            self.control_vars['key'][button].set(key)
            self.control_vars['pad'][button].set(pad or '')
        self.relabel_pad_controls()

    def build_advanced(self):
        tk, ttk = self.tk, self.ttk
        f = self.scrolled_page('advanced', _('Advanced', 'Дополнительно'),
                               _('Launcher options, files and diagnostics.'))
        self.section(f, _('Launcher', 'Лаунчер'), top=4)
        self.row(f, _('Launcher language', 'Язык лаунчера'), self.choice(f, 'ui_language', 'app', UI_LANGUAGES),
                 _('Applies when the launcher opens again.', 'Применится при следующем открытии лаунчера.'))
        self.check(f, 'close_on_play', 'app', _('Close the launcher when the game starts', 'Закрывать лаунчер при запуске игры'))
        holder = ttk.Frame(f)
        holder.grid(row=self.next_row(f), column=0, columnspan=2, sticky='w', pady=(10, 0))
        ttk.Button(holder, text=_('Desktop shortcut', 'Ярлык на рабочем столе'), command=self.shortcut).pack(side='left')
        if WINDOWS:  # on Linux the desktop shortcut also adds the application menu entry
            ttk.Button(holder, text=_('Start menu shortcut', 'Ярлык в меню «Пуск»'),
                       command=lambda: self.shortcut('Programs')).pack(side='left', padx=(8, 0))
        ttk.Button(holder, text=_('Check for updates', 'Проверить обновления'),
                   command=lambda: threading.Thread(target=self.check_update, args=(True,), daemon=True).start()).pack(
            side='left', padx=(8, 0))
        ttk.Label(holder, text=f'v{VERSION}', style='Muted.TLabel').pack(side='left', padx=10)
        self.check(f, 'update_check', 'app', _('Check for updates when the launcher opens'),
                   _('Asks GitHub for a newer release; when there is one, the sidebar offers to download and '
                     'install it (saves, settings and mods are kept).'))
        self.section(f, _('Links', 'Ссылки'))
        self.link_buttons(f).grid(row=self.next_row(f), column=0, columnspan=2, sticky='w', pady=(6, 0))
        self.section(f, _('Files'))
        holder = ttk.Frame(f)
        holder.grid(row=self.next_row(f), column=0, columnspan=2, sticky='w', pady=(6, 0))
        ttk.Button(holder, text=_('Port folder', 'Папка порта'), command=lambda: self.open_path(DATA_DIR)).pack(side='left')
        ttk.Button(holder, text='medievil.ini', command=lambda: self.open_path(ini_path())).pack(side='left', padx=6)
        holder = ttk.Frame(f)
        holder.grid(row=self.next_row(f), column=0, columnspan=2, sticky='w', pady=(10, 0))
        ttk.Button(holder, text=_('Clear shader cache', 'Очистить кэш шейдеров'), command=self.clear_cache).pack(side='left')
        ttk.Label(holder, text=_('If the game only shows a black screen, this usually helps.',
                                 'Если игра показывает только чёрный экран, обычно это помогает.'),
                  style='Muted.TLabel').pack(side='left', padx=10)
        # Diagnostics: for development, folded away until asked for.
        toggle = ttk.Label(f, text='', style='Section.TLabel', cursor='hand2')
        toggle.grid(row=self.next_row(f), column=0, columnspan=2, sticky='w', pady=(22, 2))
        body = ttk.Frame(f)
        body.columnconfigure(1, weight=1)
        body_row = self.next_row(f)
        self.check(body, 'frame_stats', 'app', _('Frame statistics in the log (every 5 s)', 'Статистика кадров в журнале (раз в 5 с)'))
        self.check(body, 'gpu_profile', 'app', _('GPU time per pass in the log', 'Профиль GPU в журнале'))
        self.check(body, 'perf_diag', 'app', _('GPU wait and readback report in the log (every 5 s)'),
                   _('Shows where the frame waits for the GPU and which reads cause it. For performance tests.'))
        self.check(body, 'frame_capture_key', 'app', _('F11 captures a frame for analysis'),
                   _('Records how the next frame is drawn into the captures folder.'))
        key, title, hint = FRAME_TRACE
        self.check(body, key, 'ini', _(title), _(hint), restart=True)
        self.check(body, 'vk_validation', 'app', _('Vulkan validation layers (needs the Vulkan SDK; much slower)',
                                                   'Слои валидации Vulkan (нужен Vulkan SDK; сильно замедляет)'))
        shown = tk.BooleanVar(value=False)

        def flip(_e=None):
            shown.set(not shown.get())
            toggle.configure(text=('▾ ' if shown.get() else '▸ ') + _('Diagnostics (for development)').upper())
            if shown.get():
                body.grid(row=body_row, column=0, columnspan=2, sticky='we')
            else:
                body.grid_forget()
        toggle.bind('<Button-1>', flip)
        shown.set(True)
        flip()

    def build_log(self):
        tk, ttk = self.tk, self.ttk
        page = ttk.Frame(self.content, padding=(20, 14, 20, 8))
        self.pages['log'] = page
        top = ttk.Frame(page)
        top.pack(fill='x', pady=(0, 8))
        image = self.gold(_('Log', 'Журнал'), 30, role='head')
        if image:
            tk.Label(top, image=image, bg=PANEL, bd=0).pack(side='left')
        else:
            ttk.Label(top, text=_('Log', 'Журнал').upper(), foreground=GOLD, font=self.f('logo', 22)).pack(side='left')
        ttk.Button(top, text=_('Copy', 'Копировать'), command=self.copy_log).pack(side='right')
        ttk.Button(top, text=_('Clear', 'Очистить'), command=lambda: self.set_log('')).pack(side='right', padx=6)
        self.log = tk.Text(page, wrap='none', bg='#0a0810', fg='#d8e8c8', insertbackground=TEXT, relief='flat',
                           font=('Consolas', 9) if WINDOWS else 'TkFixedFont', padx=8, pady=6, state='disabled', highlightthickness=0)
        bar = ttk.Scrollbar(page, command=self.log.yview)
        self.log.configure(yscrollcommand=bar.set)
        bar.pack(side='right', fill='y')
        self.log.pack(fill='both', expand=True)

    # ---- state -------------------------------------------------------------------------------
    def game_changed(self):
        folder = self.var('game_dir', 'app').get()
        info = game_info(folder)
        title = info[0] if info else APP_NAME
        self.root.title(WINDOW_TITLE.format(title, VERSION))
        self.side_title.configure(text=title.upper())
        if self.current_page == 'patches':
            self.refresh_game_patches()
        self.banner_source = None
        self.draw_banner()
        self.set_icon()
        self.refresh_status()

    def refresh_status(self):
        info = game_info(self.var('game_dir', 'app').get())
        if not self.process:
            self.play_button.configure(text=_('PLAY', 'ИГРАТЬ'), state='normal' if info else 'disabled')
            self.status.configure(text=self.summary() if info else _('Choose the game folder first.',
                                                                     'Сначала выберите папку игры.'), fg=MUTED)

    def summary(self):
        self.store_game_patches()
        mode = dict((v, t[0]) for v, t in DISPLAY_MODES).get(self.var('display_mode', 'ini').get(), '?')
        limit = self.var('fps_limit', 'ini').get()
        engine = self.var('engine_fps', 'ini').get()
        output = self.var('output_resolution', 'ini').get()
        resolution = dict((v, t[0]) for v, t in OUTPUT_CHOICES).get(output, output)
        if (scale := self.var('render_scale', 'ini').get()) != '100':
            resolution = f'{_(resolution)}, {scale}% scale'
        return _('{} · {} · {} · VSync {}').format(
            _(mode), _(resolution), _('half display refresh') if limit == '-1'
            else _('auto (refresh)') if limit == '-2' else _('auto (half refresh)') if limit == '-3'
            else _('up to {} FPS').format(engine) if limit == '0' or int(limit) >= int(engine)
            else _('{} FPS').format(limit),
            _('on') if self.var('vsync', 'ini').get() else _('off'))

    def refresh_lists(self):
        if not hasattr(self, 'patches_frame'):
            return
        tk, ttk = self.tk, self.ttk
        from mods import discover
        from patches import external_patches
        for holder in (self.mods_frame, self.patches_frame):
            for widget in holder.winfo_children():
                widget.destroy()
        root = Path(self.var('mods_dir', 'app').get() or DATA_DIR / 'mods')
        profile = load_json(DATA_DIR / 'mods.json', {})
        available = discover(root)
        if set(self.mod_order) != set(available):
            known = [n for n in profile.get('order', []) if n in available]
            self.mod_order = known + [n for n in available if n not in known]
        old = {n: v.get() for n, v in self.mod_vars.items()}
        self.mod_vars = {}
        if not available:
            ttk.Label(self.mods_frame, text=_('No mods in {} yet.', 'В {} пока нет модов.').format(root),
                      style='Muted.TLabel').pack(anchor='w')
        for index, name in enumerate(self.mod_order):
            line = ttk.Frame(self.mods_frame)
            line.pack(fill='x', pady=1)
            var = tk.BooleanVar(value=old.get(name, name not in profile.get('disabled', [])))
            self.mod_vars[name] = var
            ttk.Button(line, text='▲', width=3, command=lambda i=index: self.move_mod(i, -1)).pack(side='left')
            ttk.Button(line, text='▼', width=3, command=lambda i=index: self.move_mod(i, 1)).pack(side='left', padx=(2, 10))
            ttk.Checkbutton(line, text=f'{index + 1}.  {name}', variable=var,
                            command=self.refresh_conflicts).pack(side='left')
        if len(self.mod_order) > 1:
            ttk.Label(self.mods_frame, text=_('Lower in the list loads later and wins conflicts.',
                                              'Ниже в списке — загружается позже и перекрывает.'),
                      style='Muted.TLabel').pack(anchor='w', pady=(4, 0))
        self.conflict_label = ttk.Label(self.mods_frame, text='', justify='left', foreground='#e08080',
                                        wraplength=self.px(680))
        self.conflict_label.pack(anchor='w', pady=(4, 0))
        self.refresh_conflicts()
        if available:
            presets = ttk.Frame(self.mods_frame)
            presets.pack(anchor='w', pady=(8, 0))
            ttk.Label(presets, text=_('Preset', 'Пресет')).pack(side='left', padx=(0, 6))
            self.preset_choice = ttk.Combobox(presets, values=sorted(load_json(MOD_PRESETS, {})), width=24)
            self.preset_choice.pack(side='left')
            ttk.Button(presets, text=_('Load', 'Загрузить'), command=self.load_mod_preset).pack(side='left', padx=(6, 0))
            ttk.Button(presets, text=_('Save', 'Сохранить'), command=self.save_mod_preset).pack(side='left', padx=(6, 0))
            ttk.Button(presets, text=_('Delete', 'Удалить'), command=self.delete_mod_preset).pack(side='left', padx=(6, 0))
        chosen = load_json(DATA_DIR / 'patches.json', {})
        _title_id, game = game_profile_of(self.var('game_dir', 'app').get())
        found = external_patches(Path(self.var('patches_dir', 'app').get() or DATA_DIR / 'patches'),
                                 game[2], game[1], game[0]) if game else []
        old = {n: v.get() for n, v in self.patch_vars.items()}
        self.patch_vars = {}
        if not found:
            ttk.Label(self.patches_frame, text=_('No patch files for this game version yet.', 'Пока нет патчей для этой версии игры.'),
                      style='Muted.TLabel').pack(anchor='w')
        for key, _path, meta in found:
            on = key in chosen.get('enabled', []) or (key not in chosen.get('disabled', []) and
                                                      meta.get('isEnabled', 'false').lower() == 'true')
            var = tk.BooleanVar(value=old.get(key, on))
            self.patch_vars[key] = var
            author = meta.get('Author')
            ttk.Checkbutton(self.patches_frame, text=key + (f'  —  {author}' if author else ''), variable=var).pack(anchor='w')

    def refresh_conflicts(self):
        from mods import conflicts
        if not getattr(self, 'conflict_label', None) or not self.conflict_label.winfo_exists():
            return
        root = Path(self.var('mods_dir', 'app').get() or DATA_DIR / 'mods')
        found = conflicts(root, [n for n in self.mod_order if self.mod_vars[n].get()])
        lines = [_('⚠ {} file(s) replaced by more than one mod (the last one wins):').format(len(found))] if found else []
        lines += [f'   {path}: {" > ".join(owners)}' for path, owners in found[:12]]
        if len(found) > 12:
            lines.append(_('   … and {} more').format(len(found) - 12))
        self.conflict_label.configure(text='\n'.join(lines))

    def save_mod_preset(self):
        name = self.preset_choice.get().strip()
        if not name:
            self.messagebox.showinfo(APP_NAME, _('Type a name for the preset first.'))
            return
        presets = load_json(MOD_PRESETS, {})
        presets[name] = {'order': self.mod_order, 'disabled': [n for n, v in self.mod_vars.items() if not v.get()]}
        MOD_PRESETS.write_text(json.dumps(presets, indent=2), encoding='utf-8')
        self.preset_choice.configure(values=sorted(presets))

    def load_mod_preset(self):
        from mods import apply_preset
        preset = load_json(MOD_PRESETS, {}).get(self.preset_choice.get().strip())
        if not isinstance(preset, dict):
            return
        order, disabled = apply_preset(preset, list(self.mod_order))
        self.mod_order = order
        for name, var in self.mod_vars.items():
            var.set(name not in disabled)
        chosen = self.preset_choice.get()
        self.refresh_lists()
        self.preset_choice.set(chosen)

    def delete_mod_preset(self):
        presets = load_json(MOD_PRESETS, {})
        if presets.pop(self.preset_choice.get().strip(), None) is not None:
            MOD_PRESETS.write_text(json.dumps(presets, indent=2), encoding='utf-8')
        self.preset_choice.configure(values=sorted(presets))
        self.preset_choice.set('')

    def move_mod(self, index, step):
        other = index + step
        if 0 <= other < len(self.mod_order):
            self.mod_order[index], self.mod_order[other] = self.mod_order[other], self.mod_order[index]
            self.refresh_lists()

    def collect(self):
        """Writes settings.json, medievil.ini, mods.json and patches.json."""
        self.store_game_patches()
        for key, var in self.vars.items():
            if var.store == 'derived':
                continue
            try:
                value = var.get()
            except self.tk.TclError:  # an unfinished number in a spinbox
                value = APP_DEFAULTS.get(key, INI_DEFAULTS.get(key))
            if var.store == 'ini':
                self.ini[key] = ('1' if value else '0') if key in INI_FLAGS else str(value)
            else:
                self.app[key] = value
        self.app['controls'] = {kind: control_changes({b: v.get() for b, v in vars_.items()}, kind)
                                for kind, vars_ in self.control_vars.items()}
        self.app['pad_style'] = self.pad_style.get()
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        CONFIG_FILE.write_text(json.dumps(self.app, indent=2, ensure_ascii=False), encoding='utf-8')
        save_ini({key: self.ini[key] for key in (*INI_DEFAULTS, *INI_OPTIONAL_KEYS) if key in self.ini},
                 self.ini_lines)
        self.ini, self.ini_lines = load_ini()
        self.ini.update(graphics_options.graphics_migration(self.ini))  # older tokens, "preset", new keys
        if self.mod_order:
            (DATA_DIR / 'mods.json').write_text(json.dumps(
                {'order': self.mod_order, 'disabled': [n for n, v in self.mod_vars.items() if not v.get()]},
                indent=2), encoding='utf-8')
        if self.patch_vars:
            chosen = load_json(DATA_DIR / 'patches.json', {})
            shown = set(self.patch_vars)
            enabled = {k for k in chosen.get('enabled', []) if k not in shown} | {k for k, v in self.patch_vars.items() if v.get()}
            disabled = {k for k in chosen.get('disabled', []) if k not in shown} | {k for k, v in self.patch_vars.items() if not v.get()}
            (DATA_DIR / 'patches.json').write_text(json.dumps(
                {'enabled': sorted(enabled), 'disabled': sorted(disabled)}, indent=2), encoding='utf-8')

    # ---- game process ------------------------------------------------------------------------
    def play(self):
        if self.process:
            return
        self.collect()
        if not game_info(self.app['game_dir']):
            self.messagebox.showerror(APP_NAME, _('Choose the game folder with eboot.bin (CUSA11227), or install the .pkg.'))
            self.show('game')
            return
        self.set_log('')
        try:
            self.process = subprocess.Popen(run_command(), cwd=PORT_DIR, env=game_environment(self.app),
                                            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                            stderr=subprocess.STDOUT, creationflags=NO_WINDOW,
                                            start_new_session=not WINDOWS)
        except OSError as error:
            self.append(_('Could not start: {}', 'Не удалось запустить: {}').format(error) + '\n')
            self.process = None
            return
        self.job = GameJob(self.process)
        threading.Thread(target=self.read_output, args=(self.process,), daemon=True).start()
        self.play_button.configure(text=_('RUNNING'), state='disabled')
        self.status.configure(text=_('Preparing the game; it opens in its own window…',
                                     'Подготовка игры; она откроется в своём окне…'), fg=TEXT)
        if self.app.get('close_on_play'):
            self.root.after(5000, self.root.destroy)  # the game keeps running

    def read_output(self, process):
        """The game's output for the Log page, also kept in <saves folder>/last_run.log."""
        log = None
        try:
            self.user_dir().mkdir(parents=True, exist_ok=True)
            log = open(self.user_dir() / 'last_run.log', 'w', encoding='utf-8', buffering=1)
        except OSError:
            pass
        for raw in iter(process.stdout.readline, b''):
            text = raw.decode('utf-8', errors='replace')
            if log:
                log.write(text)
            self.output.put(text)
        code = process.wait()
        if log:
            log.write(f'\nThe game exited (code {code}).\n')
            log.close()
        self.output.put((code,))

    def drain_output(self):
        while not self.ui_calls.empty():
            self.ui_calls.get_nowait()()
        try:
            for _i in range(500):
                item = self.output.get_nowait()
                if isinstance(item, tuple):
                    self.append(_('\nThe game exited (code {}).\n', '\nИгра завершилась (код {}).\n').format(item[0]))
                    self.process = None
                    if self.job:
                        self.job.close()
                    self.refresh_status()
                else:
                    if 'Entering original x86-64 code' in item:
                        self.status.configure(text=_('The game is running.', 'Игра запущена.'), fg=TEXT)
                    elif 'restarting through run.py' in item:
                        self.status.configure(text=_('Restarting with the new settings…', 'Перезапуск с новыми настройками…'), fg=TEXT)
                    self.append(item)
        except queue.Empty:
            pass
        self.root.after(100, self.drain_output)

    def append(self, text):
        self.log.configure(state='normal')
        self.log.insert('end', text)
        lines = int(self.log.index('end-1c').split('.')[0])
        if lines > MAX_LOG_LINES:
            self.log.delete('1.0', f'{lines - MAX_LOG_LINES}.0')
        self.log.see('end')
        self.log.configure(state='disabled')

    def set_log(self, text):
        self.log.configure(state='normal')
        self.log.delete('1.0', 'end')
        self.log.insert('end', text)
        self.log.configure(state='disabled')

    def copy_log(self):
        self.root.clipboard_clear()
        self.root.clipboard_append(self.log.get('1.0', 'end'))

    def open_path(self, path, key=None):
        if key == 'user_dir' and not path:
            path = DATA_DIR / 'user'
        elif key == 'mods_dir' and not path:
            path = DATA_DIR / 'mods'
        elif key == 'patches_dir' and not path:
            path = DATA_DIR / 'patches'
        path = Path(path or DATA_DIR)
        if path.suffix == '.ini':
            self.collect()
        elif not path.exists():
            path.mkdir(parents=True, exist_ok=True)
        if WINDOWS:
            os.startfile(str(path))
            return
        try:  # the desktop's file manager or editor
            subprocess.Popen(['xdg-open', str(path)], stdin=subprocess.DEVNULL,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        except OSError as error:
            self.messagebox.showerror(APP_NAME, f'{path}\n\n{error}')

    # ---- updates -------------------------------------------------------------------------------
    def check_update(self, manual=False):
        """Helper thread: asks GitHub for the newest release and offers it when it is newer."""
        try:
            version, url, page = latest_release()
        except (OSError, ValueError, KeyError) as failure:
            if manual:
                self.ui_calls.put(lambda error=failure: self.messagebox.showerror(
                    APP_NAME, _('Could not check for updates: {}', 'Не удалось проверить обновления: {}').format(error)))
            return

        def show():
            if version_tuple(version) > version_tuple(VERSION):
                self.offer_update(version, url, page)
            elif manual:
                self.messagebox.showinfo(APP_NAME, _('You have the latest version ({}).',
                                                         'У вас последняя версия ({}).').format(VERSION))
        self.ui_calls.put(show)

    def offer_update(self, version, url, page):
        if self.update_box:
            return
        tk, ttk = self.tk, self.ttk
        text = _('Version {} is available.', 'Доступна версия {}.').format(version)
        box = tk.Frame(self.side, bg=CARD, highlightthickness=1, highlightbackground=GOLD)
        self.update_label = tk.Label(box, text=text, bg=CARD, fg=GOLD, font=self.f('head', 12),
                                     wraplength=self.px(160), justify='left')
        self.update_label.pack(anchor='w', padx=10, pady=(8, 6))
        buttons = tk.Frame(box, bg=CARD)
        buttons.pack(fill='x', padx=10, pady=(0, 10))
        self.update_button = ttk.Button(buttons, text=_('Update', 'Обновить'),
                                        command=lambda: self.install_update(version, url, page))
        self.update_button.pack(fill='x')
        ttk.Button(buttons, text=_("What's new", 'Что нового'), command=lambda: webbrowser.open(page)).pack(
            fill='x', pady=(4, 0))
        box.pack(fill='x', padx=(22, 18), pady=(0, 14), before=self.side_note)
        self.update_box = box
        if not self.process:
            self.status.configure(text=text, fg=TEXT)

    def install_update(self, version, url, page):
        if self.process:
            self.messagebox.showinfo(APP_NAME, _('Close the game before updating.', 'Закройте игру перед обновлением.'))
            return
        if not FROZEN or not url:  # a source tree updates with git
            webbrowser.open(page)
            return
        if not self.messagebox.askyesno(APP_NAME, _(
                'Install version {} now? The launcher closes, installs it and opens again. Saves and settings are kept.',
                'Установить версию {} сейчас? Лаунчер закроется, установит её и откроется снова. Сохранения и '
                'настройки останутся.').format(version)):
            return
        self.update_button.configure(state='disabled')

        def progress(percent):
            self.update_label.configure(text=_('Downloading version {}… {}%', 'Загрузка версии {}… {}%').format(
                version, percent))

        def work():
            try:
                shutil.rmtree(UPDATE_DIR, ignore_errors=True)
                UPDATE_DIR.mkdir(parents=True, exist_ok=True)
                archive = UPDATE_DIR / 'update.zip'
                request = urllib.request.Request(url, headers={'User-Agent': 'medievil-launcher'})
                with urllib.request.urlopen(request, timeout=60) as response, open(archive, 'wb') as out:
                    total, done, shown = int(response.headers.get('Content-Length') or 0), 0, -1
                    while chunk := response.read(1 << 20):
                        out.write(chunk)
                        done += len(chunk)
                        percent = done * 100 // total if total else 0
                        if percent != shown:
                            shown = percent
                            self.ui_calls.put(lambda p=percent: progress(p))
                with zipfile.ZipFile(archive) as package:
                    package.extractall(UPDATE_DIR / 'new')
                archive.unlink()
                new = next((p.parent for p in (UPDATE_DIR / 'new').rglob('MediEvil.exe')), None)
                if not new:
                    raise OSError('MediEvil.exe is missing from the download')
                # The new launcher copies itself over this installation once this one has closed.
                subprocess.Popen([str(new / 'MediEvil.exe'), '--install-update', str(PORT_DIR), str(os.getpid())],
                                 cwd=str(new), stdin=subprocess.DEVNULL, creationflags=NO_WINDOW)
                self.ui_calls.put(self.root.destroy)
            except (OSError, zipfile.BadZipFile) as failure:
                def failed(error=failure):
                    self.update_button.configure(state='normal')
                    self.update_label.configure(text=_('Version {} is available.', 'Доступна версия {}.').format(version))
                    self.messagebox.showerror(APP_NAME, _('Update failed: {}', 'Не удалось обновить: {}').format(error))
                self.ui_calls.put(failed)
        threading.Thread(target=work, daemon=True).start()

    def clear_cache(self):
        """Shader and pipeline caches; they are rebuilt while playing."""
        if self.process:
            return
        import shutil
        cache = Path(self.app['user_dir'] or DATA_DIR / 'user') / 'cache'
        shutil.rmtree(cache, ignore_errors=True)
        self.messagebox.showinfo(APP_NAME, _('Shader cache cleared. The next start stutters for a few minutes while '
                                                 'it is rebuilt.', 'Кэш шейдеров очищен. Следующий запуск несколько '
                                                 'минут будет подтормаживать, пока кэш собирается заново.'))

    def linux_shortcut(self):
        """MediEvil.desktop in the application menu (and on the desktop when there is one)."""
        appimage = os.environ.get('APPIMAGE')
        if appimage:
            command = f'"{appimage}"'
        else:
            command = f'"{sys.executable}" "{Path(__file__).resolve()}"'
        icon = CONFIG_DIR / 'medievil.png'
        try:
            CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(PORT_DIR / 'launcher' / 'medievil.png', icon)
            entry = ('[Desktop Entry]\nType=Application\nName=MediEvil\nComment=MediEvil (PS4) for PC\n'
                     f'Exec={command}\nIcon={icon}\nTerminal=false\nCategories=Game;\n')
            data_home = Path(os.environ.get('XDG_DATA_HOME') or Path.home() / '.local' / 'share')
            targets = [data_home / 'applications' / 'medievil-pc.desktop']
            desktop = Path.home() / 'Desktop'
            if desktop.is_dir():
                targets.append(desktop / 'MediEvil.desktop')
            for target in targets:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(entry, encoding='utf-8')
                target.chmod(0o755)
        except OSError as error:
            self.messagebox.showerror(APP_NAME, str(error))
            return
        self.messagebox.showinfo(APP_NAME, _('Shortcut created on the desktop.', 'Ярлык создан на рабочем столе.'))

    def shortcut(self, folder='Desktop'):
        """MediEvil.lnk on the desktop, or in the Start menu (folder 'Programs': the user's
        Start Menu\\Programs, where Windows search finds it), with the game icon."""
        if not WINDOWS:
            return self.linux_shortcut()
        if FROZEN:
            target, arguments = sys.executable, ''
        else:  # a source tree: the launcher script with the windowless Python
            pythonw = Path(sys.executable).with_name('pythonw.exe')
            target = str(pythonw if pythonw.exists() else sys.executable)
            arguments = Path(__file__).resolve()
        icon = PORT_DIR / 'launcher' / 'medievil.ico'
        result = subprocess.run(['powershell', '-NoProfile', '-Command', shortcut_script(folder, target, arguments, icon)],
                                capture_output=True, creationflags=NO_WINDOW)
        if result.returncode == 0:
            self.messagebox.showinfo(APP_NAME, _('Shortcut created in the Start menu.', 'Ярлык создан в меню «Пуск».')
                                     if folder == 'Programs' else
                                     _('Shortcut created on the desktop.', 'Ярлык создан на рабочем столе.'))
        else:
            self.messagebox.showerror(APP_NAME, result.stderr.decode(errors='replace')[:400])

    def close(self):
        try:
            self.collect()
        except Exception:  # never keep the window open over a settings problem
            pass
        self.root.destroy()


class GameJob:
    """A Windows job holding run.py and everything it starts: medievil-runtime.exe and the launches made
    by the in-game restart (no longer descendants of the first process)."""

    def __init__(self, process):
        self.process = process
        if not WINDOWS:
            # Linux: run.py starts in its own session (play()), so its process group holds
            # medievil-runtime and the launches of the in-game restart.
            self.handle = None
            return
        kernel32 = ctypes.windll.kernel32
        kernel32.CreateJobObjectW.restype = ctypes.c_void_p
        self.handle = kernel32.CreateJobObjectW(None, None)
        if self.handle:
            kernel32.AssignProcessToJobObject(ctypes.c_void_p(self.handle), ctypes.c_void_p(int(process._handle)))

    def terminate(self):
        if not WINDOWS:
            try:
                os.killpg(self.process.pid, 15)
            except OSError:
                pass
            return
        if self.handle:
            ctypes.windll.kernel32.TerminateJobObject(ctypes.c_void_p(self.handle), 1)

    def close(self):
        if self.handle:
            ctypes.windll.kernel32.CloseHandle(ctypes.c_void_p(self.handle))
            self.handle = None


def play_without_window(settings):
    """--play: the game with the saved settings (shortcuts, Steam). Output goes to the console
    when there is one and to <saves folder>/last_run.log."""
    attach_stdio()
    log_dir = Path(settings.get('user_dir') or DATA_DIR / 'user')
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / 'last_run.log', 'w', encoding='utf-8', buffering=1) as log:
        process = subprocess.Popen(run_command(), cwd=PORT_DIR, env=game_environment(settings),
                                   stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, creationflags=NO_WINDOW)
        for raw in iter(process.stdout.readline, b''):
            text = raw.decode('utf-8', errors='replace')
            log.write(text)
            try:
                sys.stdout.write(text)
            except (OSError, ValueError):
                pass
        return process.wait()


def game_argument(args):
    """--game <path>: the game folder or .pkg for this start only (a front end such as RecompLauncher
    passes it). Empty without the option; the saved game folder is not changed."""
    if '--game' not in args:
        return ''
    position = args.index('--game') + 1
    return args[position] if position < len(args) else ''


def shortcut_script(folder, target, arguments, icon):
    """PowerShell that saves MediEvil.lnk in a special folder ('Desktop', 'Programs' = Start menu)."""
    return ('$s=(New-Object -ComObject WScript.Shell).CreateShortcut('
            f'[Environment]::GetFolderPath("{folder}")+"\\MediEvil.lnk");'
            f'$s.TargetPath="{target}";$s.WorkingDirectory="{PORT_DIR}";$s.IconLocation="{icon}";'
            '$s.Description="MediEvil (PS4) for PC";'
            + (f"$s.Arguments='\"{arguments}\"';" if arguments else '') + '$s.Save()')


def version_tuple(text):
    return tuple(int(number) for number in re.findall(r'\d+', text or ''))


def latest_release():
    """(version, zip URL, page URL) of the newest GitHub release."""
    request = urllib.request.Request(RELEASES_API, headers={'Accept': 'application/vnd.github+json',
                                                            'User-Agent': 'medievil-launcher'})
    with urllib.request.urlopen(request, timeout=15) as response:
        release = json.load(response)
    version = '.'.join(re.findall(r'\d+', release['tag_name']))
    url = next((asset['browser_download_url'] for asset in release.get('assets', [])
                if asset.get('name', '').lower().endswith('.zip')), None)
    return version, url, release.get('html_url') or RELEASES_PAGE


def install_update(target, wait_pid):
    """--install-update TARGET PID, run by the downloaded version from its temporary folder:
    waits for the old launcher to close, copies this version over TARGET (never the saves,
    settings or mods) and starts it."""
    target = Path(target)
    kernel = ctypes.windll.kernel32
    handle = kernel.OpenProcess(0x00100000, False, int(wait_pid))  # SYNCHRONIZE
    if handle:
        kernel.WaitForSingleObject(handle, 60000)
        kernel.CloseHandle(handle)
    ignore = shutil.ignore_patterns(*USER_FILES)
    for attempt in range(30):
        try:
            shutil.copytree(PORT_DIR, target, dirs_exist_ok=True, ignore=ignore)
            break
        except OSError:  # a file still in use: the old launcher is closing
            time.sleep(1)
    else:
        ctypes.windll.user32.MessageBoxW(None, _(
            'Could not install the update. Download it from the releases page.',
            'Не удалось установить обновление. Скачайте его со страницы релизов.'), APP_NAME, 0x10)
        webbrowser.open(RELEASES_PAGE)
        return 1
    subprocess.Popen([str(target / 'MediEvil.exe')], cwd=str(target))
    return 0


def main():
    global LANG
    args = sys.argv[1:]
    if args and args[0] in ('--run', '--script'):
        sys.exit(run_role(args))
    settings = {**APP_DEFAULTS, **load_json(CONFIG_FILE, {})}
    LANG = settings.get('ui_language') or windows_language()
    if WINDOWS and args[:1] == ['--install-update'] and len(args) == 3:
        sys.exit(install_update(args[1], args[2]))
    if FROZEN and UPDATE_DIR not in PORT_DIR.parents:
        shutil.rmtree(UPDATE_DIR, ignore_errors=True)  # what a finished update left behind
    # Without a usable game folder there is nothing to play yet: open the launcher instead.
    # --game names the copy itself: run.py checks it and says what is wrong with it.
    game = game_argument(args) if '--play' in args else ''
    if game:
        settings['game_dir'] = game
    if '--play' in args and (game or (Path(settings['game_dir'] or '.') / 'eboot.bin').is_file()):
        sys.exit(play_without_window(settings))
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)  # sharp text on scaled displays
    except (AttributeError, OSError):
        pass
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    root = tk.Tk()
    launcher = Launcher(root, tk, ttk, filedialog, messagebox)
    root.mainloop()
    # Linux: the window is gone but a running game is still this process's child. An AppImage is
    # mounted only while its first process lives, so stay until the game ends (its output keeps
    # going to last_run.log through the reader thread).
    if not WINDOWS and launcher.process and launcher.process.poll() is None:
        try:
            launcher.process.wait()
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()
