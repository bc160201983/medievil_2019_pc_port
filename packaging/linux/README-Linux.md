# MediEvil (PS4) for PC — Linux AppImage

A single-file Linux build of the port: `MediEvil-x86_64.AppImage`. The same native runtime and
shadPS4-based Vulkan renderer as the Windows build, with the same launcher, built for Linux.
Releases ship only the Windows zip for now, so the AppImage is built from the repository (see
[Building it yourself](#building-it-yourself)).

> **No game files are included.** You need your own copy of MediEvil (CUSA11227, or the European
> CUSA12982; version 01.00 or 01.02), as an unpacked folder (with `eboot.bin`) or a fake package
> (`.pkg`).

## Run it

```sh
chmod +x MediEvil-x86_64.AppImage
./MediEvil-x86_64.AppImage
```

The launcher opens. On **Game**, choose your `.pkg` (unpacked once, beside it) or the folder
with `eboot.bin`, then press **PLAY**.

| Command | What it does |
| --- | --- |
| `./MediEvil-x86_64.AppImage` | The launcher |
| `./MediEvil-x86_64.AppImage --play` | The game with the launcher's saved settings, no launcher window (Steam, shortcuts, Game Mode) |
| `GOW3_GAME_DIR=/path/to/CUSA11227 ./MediEvil-x86_64.AppImage --run` | The game straight from the command line, without the launcher |
| `./MediEvil-x86_64.AppImage --vulkan-info` | Which Vulkan drivers and GPU the game sees |
| `./MediEvil-x86_64.AppImage --help` | These options |

**Where things go**

- Saves, shader cache, generated files, `medievil.ini`, mods: `~/.local/share/medievil-pc`
  (set `GOW3_DATA_DIR` to move it).
- Launcher settings: `~/.config/medievil-launcher`.
- The render-resolution override pak is written into your game folder's `Paks`, as on Windows,
  so the game folder must be writable.
- *Create shortcut* in the launcher adds MediEvil to your application menu (and desktop).

## Steam Deck and Steam

1. Copy the AppImage to the Deck (e.g. `~/Games`), make it executable (Properties → Permissions).
2. In Desktop Mode, run it once and pick your game folder or `.pkg` in the launcher.
3. Steam → *Add a Non-Steam Game* → the AppImage. In its Properties, set the launch options to
   `--play` so Game Mode starts the game directly, without the launcher window.
4. Leave *Compatibility* off: it is a native Linux program, not a Windows one.

If Steam reports that the AppImage cannot be mounted (no FUSE, e.g. NixOS), set the launch options
to `APPIMAGE_EXTRACT_AND_RUN=1 %command% --play`.

## Requirements

- x86-64 Linux with **glibc 2.38 or newer**: SteamOS 3.6+, Ubuntu 24.04+, Linux Mint 22+,
  Pop!_OS 24.04+, Fedora 39+, Debian 13+, Arch/Manjaro/EndeavourOS, openSUSE Tumbleweed.
  On an older system the AppImage says so and exits.
- A Vulkan 1.3 GPU driver: Mesa (RADV for AMD, ANV for Intel) or NVIDIA's driver. The AppImage
  uses your system's driver.
- An X11 or Wayland desktop (on Wayland the game window runs through XWayland), or Steam Deck
  Game Mode. Audio through PipeWire (most desktops and SteamOS) or ALSA. Controllers through
  SDL3 (DualSense, DualShock 4, Xbox and others), keyboard too.
- FUSE 2 or 3 to mount the AppImage (most desktops; otherwise `APPIMAGE_EXTRACT_AND_RUN=1`).

## Building it yourself

From the repository, on a distribution with the requirement above (the AppImage needs the glibc
of the system it was built on, so build on the oldest one you want to support):

1. Dependencies: Clang 18+ (the GPU library uses C++23 features GCC 13 lacks, and GCC 14 stops
   at the `_Static_assert` in `src/actor_hook.h`, which GCC does not accept in C++), CMake, Ninja,
   pkg-config, Python 3, glslang, and the development files of the Vulkan loader and headers
   (1.4.363 builds it; 1.4.328 is too old: it has neither `VK_DRIVER_ID_MESA_KOSMICKRISP` nor
   `VK_VALVE_shader_mixed_float_dot_product`), SDL3, FFmpeg (libavcodec, libavformat,
   libavutil, libswscale, libswresample), Boost (headers), fmt, tsl-robin-map, xxHash, Zydis,
   VulkanMemoryAllocator and libX11. `shell.nix` lists them for Nix.
2. Build: `CC=clang CXX=clang++ bash build.sh`
3. Package: `bash packaging/linux/appimage.sh` → `dist/MediEvil-x86_64.AppImage`. It downloads
   appimagetool 1.9.1, the type2 AppImage runtime 20251108, Python 3.13.16 (python-build-standalone
   20261013, with Tk) and the launcher's Python packages, all at fixed versions; each download is
   checked against a SHA-256 in the script (`PINS`) and the Tcl/Tk sources against fixed commits.
   `--python`, `--appimagetool` and `--runtime` take local copies instead.

## What differs from the Windows build

- `run.py` and the launcher run on Linux as well (they were Windows-only): the in-game *Apply and
  restart* restarts through `run.py` on both systems, *Open folder* uses `xdg-open`, *Create
  shortcut* writes a `.desktop` file, the launcher follows the system language, and the game's
  own fonts are used for the launcher through fontconfig.
- The automatic updater is Windows-only (it is off in both builds; new versions are on the
  releases page).
