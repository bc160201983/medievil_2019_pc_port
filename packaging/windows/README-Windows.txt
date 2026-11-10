MediEvil (PS4) for PC (medievil_pc)
===================================

No game files are included. You need your own copy of MediEvil (PS4, 2019): CUSA11227 (or the
European CUSA12982), either as
an unpacked folder (eboot.bin, sce_module, sce_sys) or as a fake package (.pkg).

You need
- Windows 10 (1903 or newer) or Windows 11, 64-bit.
- A graphics card with Vulkan 1.3 and a current driver.
- About 32 GB free if you install from a .pkg.

Start
1. Extract this folder anywhere, for example C:\Games\medievil-windows (not inside the zip).
2. Open "MediEvil.exe". If Windows shows "Windows protected your PC", click More info, then
   Run anyway (the program is not signed).
3. Game page:
   - Install from PKG: "Choose .pkg..." unpacks the game once next to the .pkg (progress bar;
     an interrupted unpack continues where it stopped), or
   - Game folder: Browse... to the folder with eboot.bin.
4. Home shows the cover and "CUSA11227 · version 01.00". Choose Render resolution (up to 4K)
   and display mode, then press PLAY.

While playing
- Insert on the keyboard, or R3 + L2 on the controller, opens the port's menu (image,
  performance, overlay).
- F12 saves a screenshot (PNG, full resolution) in user\screenshots; the launcher's Display page
  opens that folder.
- Controller (DualSense, Xbox and others) and keyboard work; remap them on the Controls page.

Render resolution
- 4K draws the 3D scene at 3840 x 2160 and scales it to the game's 1080p picture (much sharper).
  It writes a small override file into the game folder:
  overbite\content\paks\pakchunk99-ps4_P.pak (removed again when you choose Native).
- 60 FPS is the default and the recommended setting. Engine frame rate 120 or 240 (Performance
  page) is experimental: it can stutter and needs a fast CPU.

NVIDIA DLSS (RTX graphics cards)
- DLSS replaces the game's anti-aliasing: DLAA at Render scale 100, DLSS upscaling below it
  (Quality / Balanced / Performance follow the render scale). Turn it on in the launcher
  (Graphics > NVIDIA DLSS), in the game's menu (Image tab) or with F9 while playing.
- NVIDIA's DLSS library is not included: copy nvngx_dlss.dll (from NVIDIA's DLSS SDK,
  github.com/NVIDIA/DLSS, lib\Windows_x86_64\rel, or from another game) into the bin folder next to
  medievil-runtime.exe. gow3_dlss.dll, the port's bridge to it, is already there.

Next time
- "Play MediEvil.exe" starts the game straight away with the saved settings (shortcut, Steam).
- Saves, shader cache and the log: user\ next to MediEvil.exe (keep it when updating).
- Settings: medievil.ini next to MediEvil.exe; launcher options in %APPDATA%\medievil-launcher.
- Portable mode (USB stick, several copies): create an empty file portable.txt next to
  MediEvil.exe and the launcher keeps its options in launcher-data\ there instead. Updates keep
  both.

If something goes wrong
- Black screen at start: Advanced > Clear shader cache, then start again.
- The game does not open: check the Log page, or send user\last_run.log with your report.
- Radeon RX 400/500 or Vega and the game closes when the main menu loads: Performance > Safe GPU
  mode (older AMD GPUs), then start again.

Online features (PSN) are not available: the game runs as a console that is not signed in.

This project is not affiliated with Sony Interactive Entertainment or Other Ocean.
MediEvil is a trademark of Sony Interactive Entertainment. Based on gow3_pc (GPL-2.0-or-later).
