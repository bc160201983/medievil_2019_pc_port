#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
"""Windows counterpart of run.sh: prepares the game image and starts medievil-runtime.exe.

Same environment variables as run.sh (GOW3_GAME_DIR, GOW3_DATA_DIR, GOW3_PATCHES, GOW3_MODS_*,
GOW3_USER_DIR, ...). Extra arguments go to medievil-runtime.
The in-game "Apply and restart" runs this script again through GOW3_RESTART_COMMAND;
`--after PID` waits for the previous game process to end first (its GPU device and memory).
"""
import ctypes
import os
from pathlib import Path
import shutil
import subprocess
import sys

PORT = Path(__file__).resolve().parent


def fail(message):
    print(message, file=sys.stderr)
    sys.exit(1)


WINDOWS = sys.platform == 'win32'
RUNTIME = 'medievil-runtime.exe' if WINDOWS else 'medievil-runtime'


def wait_for(pid):
    if not WINDOWS:
        import time
        for _ in range(300):  # 30 s, as on Windows
            try:
                os.kill(pid, 0)
            except OSError:
                return
            time.sleep(0.1)
        return
    kernel32 = ctypes.windll.kernel32
    SYNCHRONIZE = 0x00100000
    handle = kernel32.OpenProcess(SYNCHRONIZE, False, pid)
    if handle:
        kernel32.WaitForSingleObject(handle, 30000)
        kernel32.CloseHandle(handle)


def no_console():
    """Started without a console (the launcher): console programs must not open their own."""
    if not WINDOWS:
        return 0
    return 0 if ctypes.windll.kernel32.GetConsoleWindow() else subprocess.CREATE_NO_WINDOW


def run_script(name, *args, capture=False):
    # Inside the packaged launcher executable (PyInstaller) the executable runs scripts itself.
    script = ['--script'] if getattr(sys, 'frozen', False) else []
    command = [sys.executable, *script, str(PORT / 'scripts' / name), *map(str, args)]
    if capture:
        result = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, text=True,
                                creationflags=no_console())
        if result.returncode:
            sys.exit(result.returncode)
        return result.stdout
    # stdin given: the output handles are passed explicitly (a windowed launcher child would
    # get none otherwise).
    result = subprocess.run(command, stdin=subprocess.DEVNULL, creationflags=no_console())
    if result.returncode:
        sys.exit(result.returncode)
    return None


# Prefixes extra-env.txt may set: the port's switches and the graphics stack's (driver, loader, SDL).
EXTRA_ENV_PREFIXES = ('GOW3_', 'RADV_', 'MESA_', 'AMD_', 'VK_', 'SDL_')


def load_extra_env(env, path):
    """Test settings from <data>/extra-env.txt, one KEY=VALUE a line (# comments), applied on every
    start however the game is launched (Steam, the launcher, --play); they override the environment.
    Diagnostics without editing launch options; the log lists what was applied."""
    try:
        lines = path.read_text(encoding='utf-8').splitlines()
    except OSError:
        return
    applied = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        key, sep, value = line.partition('=')
        key = key.strip()
        if not sep or not key.startswith(EXTRA_ENV_PREFIXES) or not key.replace('_', '').isalnum():
            print(f'extra-env.txt: ignored "{line}" (KEY=VALUE, key starting with {", ".join(EXTRA_ENV_PREFIXES)})')
            continue
        env[key] = value.strip()
        applied.append(f'{key}={env[key]}')
    if applied:
        print(f'Extra environment ({path}): {" ".join(applied)}')


def find_executable(name):
    """bin/ (packaged), else out/ (built with build.sh)."""
    for candidate in (PORT / 'bin' / name, PORT / 'out' / name):
        if candidate.is_file():
            return candidate
    return None


def main():
    args = sys.argv[1:]
    if len(args) >= 2 and args[0] == '--after':
        wait_for(int(args[1]))
        args = args[2:]
    # As run.sh: relative paths (GOW3_GAME_DIR, GOW3_DATA_DIR, fsr4_shaders/) are from the port.
    os.chdir(PORT)
    env = os.environ
    # Values this script derived on the previous start ("Apply and restart" inherits them) are
    # derived again from the new options. Dropped before extra-env.txt is read, so a test setting
    # there is never dropped with them.
    for key in env.pop('GOW3_DERIVED', '').split(','):
        env.pop(key, None)
    data = Path(env.get('GOW3_DATA_DIR', PORT))
    load_extra_env(env, data / 'extra-env.txt')
    if env.get('GOW3_DECK_SWITCHES'):  # the launcher's Steam Deck performance switches (deck_switch_env)
        print(f"Deck switches: {env['GOW3_DECK_SWITCHES']}", flush=True)
    out = data / 'out'
    out.mkdir(parents=True, exist_ok=True)
    if not (data / 'medievil.ini').exists() and (data / 'gow3.ini').exists():
        (data / 'gow3.ini').rename(data / 'medievil.ini')  # settings of builds before 1.1
    env.setdefault('GOW3_CONFIG', str(data / 'medievil.ini'))
    # Everything set from here on is derived from the options (GOW3_DERIVED below), including the
    # setdefault()s: the performance patches' off switches, the HZB / stream switches, the engine
    # frame-rate and memory sizes. Before, only a listed few were, so after "Apply and restart" a
    # patch turned back on in the menu stayed off (LANDMINES item 12).
    given = set(env)
    derived = []
    sys.path.insert(0, str(PORT / 'scripts'))
    import graphics_options
    graphics, reverted = graphics_options.prepare(env['GOW3_CONFIG'], out / 'graphics_state.json')
    if reverted:
        print('Graphics: the last restart never reached the game; previous options restored', flush=True)
    env.update(graphics_options.environment(graphics))
    ini_values = graphics_options.read_ini(env['GOW3_CONFIG'])
    for key, value in graphics_options.perf_environment(ini_values).items():
        env.setdefault(key, value)
    # Presentation settings (launcher > Performance > Presentation); derived, so an in-game
    # restart takes the new values.
    trace = PORT / 'user' / 'logs' / 'flip-trace.csv'
    for key, value in graphics_options.present_environment(ini_values, trace).items():
        if key not in env:
            env[key] = value
            derived.append(key)
    if 'GOW3_FLIP_TRACE' in derived:
        trace.parent.mkdir(parents=True, exist_ok=True)
    if not env.get('GOW3_FSR411_DIR') and not (PORT / 'fsr4_411').is_dir() and (data / 'fsr4_411').is_dir():
        env['GOW3_FSR411_DIR'] = str(data / 'fsr4_411')

    probe = Path(env['GOW3_PROBE']) if env.get('GOW3_PROBE') else find_executable(RUNTIME)
    if not probe or not probe.is_file():
        fail(f'{RUNTIME} not found: build it with build.sh or use a packaged build.')
    # Development builds run from the MSYS2 tree: its CLANG64 DLLs (SDL3, FFmpeg, ...).
    if WINDOWS and not (probe.parent / 'SDL3.dll').is_file():
        clang64 = Path(env.get('MSYS2_ROOT', r'C:\msys64')) / 'clang64' / 'bin'
        if clang64.is_dir():
            env['PATH'] = f'{clang64}{os.pathsep}{env.get("PATH", "")}'

    game = Path(env.get('GOW3_GAME_DIR', PORT.parent / 'CUSA11227'))
    if game.suffix.lower() == '.pkg' and game.is_file():
        # A PKG is unpacked once beside it (<folder>/<TITLE_ID>); later starts reuse the folder.
        import pkg_extract
        target = Path(env.get('GOW3_PKG_TARGET') or pkg_extract.default_target(game))
        if not pkg_extract.is_extracted(game, target):
            print(f'Unpacking {game.name} into {target} (first start only)…', flush=True)
            last = [-1]

            def report(done, total, rate):
                percent = done * 100 // max(total, 1)
                if percent != last[0]:
                    last[0] = percent
                    print(f'PKG: {percent}% ({done >> 20} of {total >> 20} MB, {rate:.0f} MB/s)', flush=True)
            try:
                pkg_extract.extract(game, target, report)
            except (OSError, ValueError) as error:
                fail(f'Could not unpack {game}: {error}')
        game = target
    if not (game / 'eboot.bin').is_file():
        fail(f'No eboot.bin in {game} (set GOW3_GAME_DIR to the game folder or its .pkg).')
    import game_profiles
    profile = game_profiles.profile(game)
    if profile['name']:
        print(f'Game profile: {profile["name"]} ({", ".join(profile["modules"])})', flush=True)
    engine_fps = int(graphics['engine_fps'])
    if graphics_options.no_frame_caps(ini_values):
        engine_fps = 1000  # no engine bound (UE4 SmoothedFrameRateRange); the present limit is off too
    if profile.get('ue4_render_scale') and engine_fps > 60:
        # Unreal Engine 4 flips at every vblank (rhi.SyncInterval 1). A 480 Hz vblank with a frame
        # limit shows each frame when it is ready: a 120 Hz vblank turned 9 ms frames into 17 ms.
        env.setdefault('GOW3_VBLANK_HZ', '0')
        env.setdefault('GOW3_FPS_CAP', str(engine_fps))
    if profile.get('ue4_render_scale'):
        import ue4_override
        # The game's output size (medievil.ini output_resolution); the 3D scene's render_scale is in
        # ue4_override.apply.
        width, height = graphics_options.output_size(graphics['output_resolution'])
        output = f'{width}x{height}'
        print(f'Output resolution: {output}' + (' (the display)' if graphics['output_resolution'] == 'display'
                                                 else ''), flush=True)
        graphics_options.output_environment(env, width, height)
        graphics_options.hzb_full_size_environment(env, width, height, int(graphics['render_scale']))
        graphics_options.nav_purge_environment(env)
        graphics_options.particle_single_fixed_environment(env)
        # HZB occlusion through the runtime's remap (research: T-HZ4, study-hzb-double-buffer.md):
        # its hooks fit the analysed eboot only. Small read-only buffers the CPU did not write are
        # bound from VRAM instead of copied every draw (S2, T-S2b). GOW3_HZB_REMAP=0 /
        # GOW3_STREAM_CLEAN=0: off.
        from patches import (eboot_matches as hzb_eboot_matches, game_app_version as hzb_version,
                             game_profile as hzb_profile, game_title_id as hzb_title)
        known = hzb_profile(hzb_title(game), hzb_version(game))  # 01.00 and 01.02: src/hzb_hook.h
        if known and known[3] and hzb_eboot_matches(game, known):
            env.setdefault('GOW3_HZB_REMAP', '1')
        env.setdefault('GOW3_STREAM_CLEAN', '1')
    for key, value in profile['env'].items():
        env.setdefault(key, value)
    profile_ue4 = bool(profile.get('ue4_render_scale'))
    if profile_ue4:
        import ue4_override
        try:
            ini = graphics_options.read_ini(env['GOW3_CONFIG'])
            # The graphics settings (graphics_options.GRAPHICS_SETTINGS) decide the console variables,
            # not the quality level: every difference between levels is a setting.
            choices = graphics_options.graphics_choices(ini)
            level = graphics_options.matched_level(choices, ini.get('ue4_quality', 'console'))
            print(f"Graphics: quality {level or 'custom'}; "
                  + ', '.join(f'{k}={v}' for k, v in choices.items()), flush=True)
            print('Render resolution:', ue4_override.apply(game, output, int(graphics['render_scale']),
                                                           ini.get('ue4_quality', 'console'), engine_fps,
                                                           ini.get('cheat_menu') == '1',
                                                           ini.get('cheat_infinite_ammo') == '1',
                                                           env.get('GOW3_HZB_REMAP') == '1',
                                                           ini.get('effects_resolution', '75'),
                                                           ini.get('ue4_aa', 'taa'),
                                                           ini.get('ue4_voice', ''),
                                                           graphics_options.graphics_cvars(choices)), flush=True)
        except (OSError, ValueError) as error:
            print(f'Render resolution: unchanged ({error})', flush=True)
    original_game = game.resolve()
    game = Path(run_script('mods.py', game, '--out', out,
                           '--mods-dir', env.get('GOW3_MODS_DIR', data / 'mods'),
                           '--config', env.get('GOW3_MODS_CONFIG', data / 'mods.json'),
                           '--enabled', env.get('GOW3_MODS_ENABLED', '1'), capture=True).strip())
    mod_view = game if game.resolve() != original_game else None
    try:
        run_script('prepare.py', game, '--out', out)
        if profile_ue4:
            # Where this eboot keeps r.SetRes, for native output and the in-game resolution
            # switches (runtime_extra.c).
            import ue4_override
            slot = ue4_override.setres_slot(out / 'eboot.elf')
            if slot:
                env['GOW3_UE4_SETRES_CVAR'] = hex(slot)
            elif env.get('GOW3_UE4_SETRES'):
                print('Native output: r.SetRes not found in this game build; output stays 1080p', flush=True)
                for key in ('GOW3_UE4_SETRES', 'GOW3_INTERNAL_WIDTH', 'GOW3_INTERNAL_HEIGHT'):
                    env.pop(key, None)
        run_script('link_libc.py', game, '--out', out)
        run_script('link_modules.py', game, '--out', out, '--modules', *profile['modules'])
        run_script('content_profile.py', game, '--out', out, '--sku', env.get('GOW3_CONTENT_SKU', 'full'))

        from patches import (eboot_matches, game_app_version, game_profile, game_title_id,
                             patch_requirements, selected_patches)
        profile = game_profile(game_title_id(game), game_app_version(game)) or game_profile(game_title_id(game))
        patched = bool(profile) and ((game_app_version(game) == profile[2] and eboot_matches(game, profile))
                                     or bool(env.get('GOW3_FORCE_PATCHES')))
        # The window shows the game's icon (BMP: the format SDL loads without extra libraries).
        try:
            from PIL import Image
            with Image.open(game / 'sce_sys' / 'icon0.png') as icon:
                icon.convert('RGBA').resize((64, 64), Image.LANCZOS).save(out / 'window_icon.bmp')
            env.setdefault('GOW3_WINDOW_ICON', str(out / 'window_icon.bmp'))
        except (ImportError, OSError):
            pass
        # The temporal upscalers and motion vectors are not calibrated for this game yet; linear
        # image readbacks fix its corrupted textures.
        env.setdefault('GOW3_UPSCALER', 'none')
        env.setdefault('GOW3_READBACK_LINEAR', '1')
        if patched:
            # The patch notes give the direct memory and VBlank rate the selected patches need.
            names = selected_patches(profile[1], profile[2], env.get('GOW3_PATCHES', ''), env.get('GOW3_PATCHES_ONLY') == '1')
            # medievil.ini decides the resolution and frame rate patches (in-game menu, launcher).
            if not profile_ue4:  # God of War III's resolution / frame rate patch names
                names = graphics_options.patch_names(names, graphics)
            env['GOW3_PATCHES'] = ';'.join(names)
            env['GOW3_PATCHES_ONLY'] = '1'
            dmem, vblank = patch_requirements(profile[1], names, profile[2])
            for key, value in (('GOW3_DMEM_MB', dmem), ('GOW3_VBLANK_HZ', vblank)):
                if value and key not in env:
                    env[key] = str(value)
                    derived.append(key)
        run_script('patches.py', '--out', out, '--extra', env.get('GOW3_PATCHES', ''), '--game-dir', game,
                   '--patches-dir', env.get('GOW3_PATCHES_DIR', data / 'patches'),
                   '--patches-config', env.get('GOW3_PATCHES_CONFIG', data / 'patches.json'))
        if 'GOW3_VBLANK_HZ' not in env:
            env['GOW3_VBLANK_HZ'] = '60'
            derived.append('GOW3_VBLANK_HZ')
        env['GOW3_DERIVED'] = ','.join(sorted(set(derived) | (set(env) - given)))

        # The in-game restart starts this script again once this process is gone.
        # GPU caches (shaders, pipelines) beside the saves; read before main(), so set here.
        user_dir = Path(env.get('GOW3_USER_DIR', data / 'user')).resolve()
        user_dir.mkdir(parents=True, exist_ok=True)
        env.setdefault('GOW3_GPU_USER_DIR', str(user_dir))
        from save_backup import backup
        try:
            if made := backup(user_dir):
                print(f'Saves backed up to {made}', flush=True)
        except OSError as error:
            print(f'Save backup failed, starting anyway: {error}', flush=True)
        this = ['--run'] if getattr(sys, 'frozen', False) else [str(Path(__file__).resolve())]
        if WINDOWS:
            restart = [sys.executable, *this, '--after', str(os.getpid()), *args]
            env['GOW3_RESTART_COMMAND'] = subprocess.list2cmdline(restart)
        else:
            # Linux: the runtime replaces itself with this command (exec), so the new launch
            # runs as this script's child and must not wait for this process.
            import shlex
            env['GOW3_RESTART_COMMAND'] = shlex.join([sys.executable, *this, *args])
        command = [str(probe), str(out / 'boot-linked.bin'), '--content-profile', str(out / 'content.bin'),
                   '--patches', str(out / 'patches.bin'), '--app0', str(game),
                   '--user', str(user_dir),
                   '--timeout', env.get('GOW3_TIMEOUT', '0'), *args]
        if capture := env.get('GOW3_RENDERDOC'):
            # Diagnostics: run under RenderDoc (its capture key, F12 / Print Screen, saves a frame
            # to <capture>_frameN.rdc); -w waits for the game, so the log still follows it.
            renderdoc = env.get('GOW3_RENDERDOC_CMD', r'C:\Program Files\RenderDoc\renderdoccmd.exe')
            command = [renderdoc, 'capture', '-w', '-c', capture, *command]
        elif trace_dir := env.get('GOW3_NGFX'):
            # Diagnostics: Nsight Graphics GPU Trace into <trace_dir>; F11 in the game starts a trace
            # (GOW3_NGFX_ARGS: other ngfx options). ngfx splits --args on spaces and keeps only
            # backslash-escaped quotes, which list2cmdline twice gives (once here, once for ngfx's own
            # command line). The game inherits this environment (checked with ngfx-test.ps1).
            ngfx = env.get('GOW3_NGFX_CMD', r'C:\Program Files\NVIDIA Corporation\Nsight Graphics 2026.3.1'
                           r'\host\windows-desktop-nomad-x64\ngfx.exe')
            # --real-time-shader-profiler: samples per shader (time per shader in the Shader Pipelines
            # view); it replaces some detailed SM / L1TEX counters, not the frame-level ones.
            options = env.get('GOW3_NGFX_ARGS', '--start-after-hotkey --limit-to-frames 5 --max-duration-ms 2000 '
                              '--architecture Ada --metric-set-id 2 --real-time-shader-profiler '
                              '--auto-export --keep-going').split()
            # --no-timeout: ngfx otherwise gave up while a trace was collected ("Operation timeout"), and
            # ending its session closed the game (ngfx-narrow-1, 2026-10-28).
            command = [ngfx, '--no-timeout', '--activity', 'GPU Trace Profiler', '--output-dir', trace_dir,
                       '--exe', command[0], '--dir', os.getcwd(), '--args', subprocess.list2cmdline(command[1:]),
                       *options]
        # No stdin: an inherited pipe (shells such as Git Bash) cost the game its console output.
        return subprocess.run(command, stdin=subprocess.DEVNULL, creationflags=no_console()).returncode
    finally:
        if mod_view:
            shutil.rmtree(mod_view, ignore_errors=True)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
