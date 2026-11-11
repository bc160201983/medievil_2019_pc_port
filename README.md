# MediEvil (PS4) for PC

<a href="https://recomplauncher.com/game/medievil-2019-pc-port"><picture><source media="(prefers-color-scheme: dark)" srcset="https://recomplauncher.com/assets/press/launcher/badge-light.svg"><img alt="Get it on RecompLauncher" src="https://recomplauncher.com/assets/press/launcher/badge-dark.svg" height="48"></picture></a>

A PC port of **MediEvil** (PlayStation 4, 2019, CUSA11227; the European CUSA12982 works too) for
Windows 10 and 11 and the Steam Deck (Linux), built on the
[God of War III PC port](https://github.com/FelipePrado0/gow3_pc).

The CPU side is native: the game's original x86-64 executable runs directly on the PC's processor
and a runtime written for this port replaces the PS4 system libraries, so there is no CPU emulator.
The graphics run on the [shadPS4](https://github.com/shadps4-emu/shadPS4) video core, which
emulates the PS4 GPU at a high level: it recompiles the game's shaders and translates its GPU
commands to Vulkan.

## What's new in 1.4.0

Most of this release is **[BevelTipDrip](https://github.com/BevelTipDrip)'s work** from their fork
([BevelTipDrip/medievil_2019_pc_port](https://github.com/BevelTipDrip/medievil_2019_pc_port),
releases 2.0.0 to 2.1.1), merged here with thanks. Their fork documents every fix with its
measurements in `research/docs/fix-inventory.md`.

- **Fights**: kill effects nearly doubled the render passes in a frame; most of those breaks are
  gone. A fight at a 120 FPS cap holds 1% / 0.1% lows of 119.7 / 117.4 FPS.
- **Culling**: Unreal's HZB occlusion culling works (its small mip levels were never rendered),
  and the hidden decal depth prepass (about 600 draws a frame) is skipped.
- **Memory sync**: no GPU downloads when the game writes memory the GPU wrote, no 2-5 s stalls
  when Windows trims the game's memory, cheaper lock and memory-pool lookups.
- **Draw translation**: less CPU work per draw (texture-set memo, register scans, deferred fences,
  bulk descriptor copies, sampler memo).
- **Fixes**: shadows drawn as black boxes; a crash and freeze in the draw pipe; GPU particles that
  vanished with the camera angle ("GPU particle bounds" patch); walls flooded with light after a
  level load (teal Hall of Heroes, white walls, purple reflections); the torch and crypt-floor
  particle glitch; the in-game menu no longer adds frames.
- **Developer cheat menu on 01.02**: **L1 + R1 + R3** in a level (01.00: R3).
- **NVIDIA DLSS / DLAA** (RTX cards, off by default): launcher > Graphics > NVIDIA DLSS, the
  in-game menu, or F9. NVIDIA's `nvngx_dlss.dll` is not included: copy it into `bin`
  ([README-Windows.txt](packaging/windows/README-Windows.txt)).
- **Graphics quality levels** with 18 individual settings (Console = the PS4's exact values), and
  live render scale from 50% to 200%.
- **Frame pacing above 60 FPS**: NVIDIA Reflex limiter, an automatic frame limit, presents on a
  precise vblank tick.
- **Steam Deck / Linux**: the native AppImage works again, with a Deck Optimized preset.
- Every optimisation is a switch (launcher > Performance > Performance patches, or a `GOW3_*`
  variable).

> **No game files are included.** You need your own copy of MediEvil (CUSA11227), as an unpacked
> folder or a fake package (.pkg). This project is not affiliated with Sony Interactive
> Entertainment or Other Ocean. MediEvil is a trademark of Sony Interactive Entertainment.

## Screenshots

| | |
|---|---|
| ![The Graveyard](docs/screenshots/graveyard.jpg) | ![Combat](docs/screenshots/combat.jpg) |
| ![Spirit flames in The Graveyard](docs/screenshots/spirits.jpg) | ![World map](docs/screenshots/world-map.jpg) |

Captured on Windows 11 (RTX 5080) with this port.

## What works

- Boot, intro movies, menus and gameplay (tested on an RTX 5080, a 1080p 200 Hz monitor and a 4K TV).
- **Intel and AMD CPUs**: the game's few AMD-only SSE4a instructions (`INSERTQ`) are replaced with
  SSE2 code on CPUs without them (all Intel CPUs), and any other one is emulated.
- **60 FPS by default; above 60 is experimental**: Engine frame rate 120 or 240 (launcher >
  Performance, or the in-game menu) lifts the PS4 game's 60 FPS cap, but can stutter and needs a
  fast CPU; settings from before 1.3.4 go back to 60 once. The cap is Unreal Engine 4's console frame-rate ceiling
  (`SmoothedFrameRateRange`, applied with VSync on), raised by the config override pak; game speed
  stays normal. Frames are shown as soon as they are ready (a 480 Hz virtual vblank with a frame
  limit), so pacing stays even. An RTX 5080 runs menus at 120 and levels at about 90 to 120 FPS.
- **Smooth frame pacing on any monitor**: Frame rate limit "Half display refresh (smoothest)" shows
  every frame for exactly two refreshes (72 FPS on 144 Hz, 100 FPS on 200 Hz; with Engine frame
  rate 120). 60 FPS on a 144/165 Hz screen cannot be shown evenly and looks like stutter. With
  VSync on, a limit at the display's own refresh (60 on a 60 Hz TV) is left to VSync, which paces it.
- Audio (Wwise through AudioOut, ATRAC9 and AAC). On stereo speakers and headphones the port mixes
  the game's 7.1 output down itself (SDL's downmix played it about 13 dB too quietly).
- Controller (DualSense, Xbox and others, through SDL3) and keyboard, both remappable. Every
  connected controller is read, also while another window has the focus, and the log names each
  one. Controls > Gamepad names (PlayStation / Xbox) only relabels the launcher; the game keeps its
  PS4 button prompts. Controls > Stick range (90% to 70%) counts less tilt as a full push, for
  sticks that never reach the edge (some hall-effect pads).
- Saving and loading; save backups before every start.
- **All the game's languages** (launcher > Game > Language): every PS4 system language can be
  picked, Polish included, whatever region your copy is from (a language the game has no text
  for shows English).
- **Voice language** (launcher > Game > Voice language): the spoken language can differ from the
  text. The game has 15 dubs (English, French, German, Italian, Spanish, Latin American Spanish,
  Portuguese, Brazilian Portuguese, Dutch, Polish, Russian, Czech, Greek, Arabic, Japanese), for
  example Japanese voices with English text. "Same as game language" keeps the game's own choice.
- **Screenshots**: F12 in the game saves a PNG of the game's picture at its full resolution
  (3840 × 2160 in native 4K) in `user/screenshots` (launcher > Display > Open screenshots folder).
  Shift+F12 saves raw frames for bug reports.
- **Update notice**: the launcher asks GitHub for a newer release when it opens and offers to
  download and install it (Advanced > Check for updates; can be switched off).
- **Install from PKG**: the launcher (or `run.py`) unpacks a fake PKG once, resumably.
- **Any output resolution** (launcher > Output resolution: 720p, 1080p, 1440p, 1800p, 4K, your
  display's size, or any `output_resolution=WxH` in `medievil.ini`): the game starts at that size,
  as a PS4 starts at its TV's (built-in patch "Boot display size", `patches/MediEvil.xml`), so the
  menus and controller navigation work at every size. In windowed mode the window opens at it.
  Above 1080p the port also uses a larger PS4 framebuffer pool ("Native 4K framebuffer pool
  (512 MB)") and more game memory, by itself.
- **Render scale** (launcher > Render scale, 50% to 200%): above 100% the 3D scene is drawn larger
  and scaled down to the output (supersampling, sharpest); below 100% the game's own UE4 temporal
  upsampling draws it smaller and rebuilds the full picture from its motion vectors (faster). The
  menus and HUD stay at the output size (a UE4 config override pak, `scripts/ue4_override.py`).
  The in-game menu (Performance > Game resolution now) can still switch between 1080p, 1440p,
  1800p and 4K live, for the developer debug menu.
- **Anti-aliasing** (launcher > Performance > Anti-aliasing): the game's temporal AA (default,
  smooth), FXAA (sharper) or Off (raw pixels, best with a high render resolution). A render scale
  below 100% always uses temporal AA.
- **Graphics quality**: Low (integrated graphics such as the Radeon 780M), Medium, Console (PS4),
  High and Ultra (sharper, longer-range shadows, better reflections and more distant detail).
  Every preset keeps point/spot light shadows at one fixed size, which stops the shadow flicker
  while moving, and limits shadow draws so busy outdoor scenes stay near 60 FPS.
- **Launcher** in MediEvil's style: Gallowmere's colors and the game's own fonts (read from your
  copy of the game, never shipped), with home and cover art, display, performance, game folder /
  PKG install / save backups, cheats, patches, mods, controls, diagnostics and log.
- **In-game menu** (Insert, or R3+L2): image (display mode, VSync, FSR 1, sharpening),
  performance and the performance overlay.
- Online features (PSN) answer as a console that is not signed in.

### Cheats (launcher > Game > Cheats)

- **Developer cheat menu**: MediEvil's own debug menu, left in the game by its developers and
  switched on by the override pak. Press **R3** in a level (01.02: hold **L1 + R1**, then press
  **R3**): unlock all levels, level select,
  complete level, unlock all weapons, invincibility, all runes and quest items, life bottles,
  gold, chalices, fly mode and more. Its 60 FPS lock is removed. Enemies pause while the menu is
  open (the developers' design) and carry on when it closes, however it is closed (Complete
  Level and other level changes included).
- **Unlimited ammo**: daggers, crossbow and longbow bolts, lightning and the other weapons never
  run out (weapons and shields also stop wearing out).

### Performance work in this port

- **Fewer hitches (from the Uncharted port)**: a 1 ms Windows timer, no power throttling of the
  game's threads, a spare GPU memory block per memory type (new textures no longer wait for the
  driver), pageable video memory where the driver offers it, merged buffer barrier ranges and
  lighter thread wake-ups. On 4 CPU cores the worst gameplay frame went from 31-32 ms to 18 ms and
  a busy scene from 33 to 54 FPS (measured twice). Each can be switched off for tests
  (`GOW3_TIMER_RES=0`, `GOW3_POWER_THROTTLE=1`, `GOW3_SPARE_IMAGE_BLOCK=0`,
  `GOW3_PAGEABLE_MEMORY=0`, `GOW3_TRACKER_MERGE_GAP=0`, `GOW3_FUTEX_COND=0`).
- GPU barriers are recorded at most 128 images per call (very large batches are what older AMD
  drivers crash on).
- **No more texture churn**: the game draws its lighting volumes and colour grading LUT (3D
  textures) as 2D slices; each switch rebuilt and copied the texture, about 14 times per frame.
  The 3D texture now serves the slices through a 2D array view (~4,700 texture rebuilds per 5 s
  down to ~90).
- **8 GB graphics cards (laptops)**: the texture garbage collector's limits now follow the
  driver's memory budget (from the Uncharted port). The old rule evicted textures still in use
  on 8 GB cards and loaded them again (746 evictions in a test route; now none).

- Texture cache images share VMA memory blocks instead of one driver allocation each (Unreal
  Engine 4 reuses render target memory every frame): 22% of the draw recording thread was spent
  creating and freeing GPU memory.
- A sampling profiler for the GPU threads: `GOW3_SAMPLE_FILE=<prefix>` and
  `scripts/symbolize_samples.py`.

### Known limits

- Above 60 FPS, the busiest levels are limited by the port's GPU command translation (one thread).
- "Game resolution now" (in-game menu) works with Native and the native output modes; with the
  1440p / 1800p / 4K render resolutions (supersampling) it stays at 1080p, as a larger output on top
  runs out of the game's PS4 memory.
- With a native output above 1080p (1440p / 1800p / 4K native output) the gamepad cannot reach
  menu items outside the top-left 1920×1080 area (pause menu "Quit to Map", the debug menu's right
  column): the game's UI window stays at 1080p while the picture switches. "4K (sharpest)" renders
  the 3D scene at 4K with working menus; or switch to 1080p (in-game menu > Performance) and back.
- Both MediEvil versions are supported: 01.00 (the base game) and 01.02 (the update), for
  CUSA11227 and CUSA12982. The built-in patches have addresses for each version and are chosen by
  the game's APP_VER and the eboot.bin hash (such as the 4K framebuffer pool); cheat edits that do
  not match the installed version are skipped (the log says which). Native output (1440p / 1800p /
  4K native output) finds the game's resolution setting in each version's own code (tested on 01.00;
  the 01.02 address was found by @Porticus in #29). If a game build doesn't have that code, native
  output stays at 1080p and the log says so. On the European 01.02 (CUSA12982) the 4K pool patch,
  unlimited ammo, the FPS counter edit and the voice option work; the developer cheat menu opens
  with L1 + R1 + R3 there as on the US 01.02 (not tested on the EU build). The 01.02 eboot does not export
  a few `libSceNpToolkit2` C++ functions the game links; those imports are logged no-ops returning
  0, since the port does not implement that module.
- **Radeon RX 400/500 and Vega** (older AMD drivers): some crash in `amdvlk64.dll` when the main
  menu loads. Try launcher > Performance > Safe GPU mode (older AMD GPUs) (`GOW3_VULKAN_COMPAT=1`),
  which runs without newer Vulkan features. The log names the GPU and driver (a `GPU: …` line), and
  a crash report lists the GPU barriers being recorded: please attach `user/last_run.log`.
- Experimental PS4 Pro mode (`GOW3_NEO=1`) stops in the texture cache.

## Download and play

1. Download `medievil-windows.zip` from the
   [Releases page](../../releases/latest) and extract it (not inside the zip).
2. Open **`MediEvil.exe`** (Windows may warn that it is not signed: More info, Run anyway).
3. On **Game**, either **Choose .pkg…** to install from your package, or **Browse…** to the
   folder with `eboot.bin`.
4. Choose an **Output resolution** (your display's size for a sharp picture) and a **Render scale** on
   **Home** and press **PLAY**.
5. Optional, NVIDIA RTX: copy NVIDIA's `nvngx_dlss.dll` into the `bin` folder and turn on **NVIDIA DLSS**
   (launcher > Graphics). DLAA at render scale 100, DLSS upscaling below it.

`Play MediEvil.exe` starts the game with the saved settings, for shortcuts and Steam.
**Portable mode**: with an empty file `portable.txt` next to `MediEvil.exe`, the launcher keeps its
settings in a `launcher-data` folder beside it instead of `%APPDATA%\medievil-launcher` (saves and
`medievil.ini` are in the port folder already); updates keep both.
More in [packaging/windows/README-Windows.txt](packaging/windows/README-Windows.txt).

## Building from source

1. Install [MSYS2](https://www.msys2.org) and, in the **MSYS2 CLANG64** shell, the packages listed
   in [packaging/windows/README.md](packaging/windows/README.md). Install
   [Python 3](https://www.python.org) with `pip install pillow cryptography`.
2. Build (newer CLANG64 toolchains need the standard headers forced for some third-party code):

   ```bash
   git clone --recursive https://github.com/bc160201983/medievil_2019_pc_port
   cd medievil_2019_pc_port
   CXXFLAGS="-include cstdlib -include cstdio -include cwchar -include cmath -include new" bash build.sh
   ```

3. Start the launcher with `MediEvil Launcher.cmd` (or `python launcher/medievil_launcher.py`),
   or the game directly:

   ```powershell
   $env:GOW3_GAME_DIR = 'D:\path\CUSA11227'   # or the .pkg
   python run.py
   ```

4. Package: `bash packaging/windows/package.sh` builds `dist/medievil-windows.zip`.

Linux (community build, not official yet): see [packaging/linux/README-Linux.md](packaging/linux/README-Linux.md)
for building from source and the AppImage script (#13, by @nydal91).

## How it works

| Part | Where | What it does |
|---|---|---|
| PKG install | `scripts/pkg_extract.py` | Decrypts and unpacks fake PKGs (PFS, PFSC), as shadPS4 does |
| Game profiles | `scripts/game_profiles.py` | Modules to link and GPU defaults per title |
| Preparation | `scripts/` | Reads `eboot.bin`, links the game's `libc`, `libSceFios2` and `libSceNpToolkit2` |
| Loader | `src/probe.c` | Loads the image and starts the game's original code |
| Runtime | `src/runtime_*.c` | PS4 system functions; `runtime_extra.c` holds the ones added for MediEvil |
| GPU | `gpu/shadps4/` | shadPS4 video core (Vulkan) with the speed work of the port it is based on |
| Override pak | `scripts/ue4_override.py` | Resolution, quality, frame rate cap, debug menu and unlimited ammo (config and Blueprint edits) |
| Game patches | `patches/MediEvil.xml` | Code patches for 01.00 and 01.02 (native 4K framebuffer pool) |
| Launcher | `launcher/medievil_launcher.py` | Settings, PKG install, mods, controls and the play button |

To port another PS4 game the same way: the loader stops at the first system function the runtime
does not implement and names it; `scripts/gen_import_names.py` adds a game's imports to the name
table.

## Credits

- **BevelTipDrip**: the performance, correctness, DLSS and Steam Deck work of 1.4.0, from
  [their fork](https://github.com/BevelTipDrip/medievil_2019_pc_port).
- **Porticus**: 01.02 support, native output on 01.02, the frame analyzer fix and more pull requests.
- **FelipePrado0**: [gow3_pc](https://github.com/FelipePrado0/gow3_pc), the God of War III port
  this project is built on, and its credits: **Supermedo** and **deadinside28** (bloodborne_pc),
  **cuesta4** (shadPS4 GPU fixes).
- **shadPS4 team**: the video core, and the PKG format and fake-PKG keysets used by
  `scripts/pkg_extract.py` (`src/core/file_format/pkg.cpp`, `src/core/crypto`) and the NID name
  list (`tools/aerolib.inl`).
- Also used: LibAtrac9, FFmpeg, SDL3, Dear ImGui, sirit, magic_enum, miniz, xbyak, Vulkan Memory
  Allocator, FSR-Vulkan, MSYS2, LLVM, Pillow and cryptography.

## License

GNU GPL v2 or later ([LICENSE](LICENSE)). Third-party components keep their own licenses.
