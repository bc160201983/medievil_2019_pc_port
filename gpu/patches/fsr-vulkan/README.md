# Local changes to gpu/third_party/fsr-vulkan

The submodule points at upstream FireBurn/FSR-Vulkan (`c64f093`). `build.sh` applies the
patches here to its working tree when they are not applied yet:

- `0001-...`: `GOW3_FSR4_PROFILE` (GPU time per FSR 4 pass) and `GOW3_FSR4_STATS` (driver
  statistics of each pass) in the FSR 4 v07 provider.
- `0002-...`: `#include <mutex>` in the FFX Vulkan backend; libc++ (the Windows MSYS2 CLANG64
  build) does not include it transitively.
- `0003-...`: FSR 3.1.5 bridge and 3.1.6 frame generation: retire bridge frames and reuse
  persistently mapped constant buffers instead of creating one per dispatch.
- `0004-...`: FSR 3.1.6 frame interpolation updated to the SDK 2.4.0 sources: separate colour
  rect in the constants, motion vector fields cleared on the host (the optical flow ones at
  display / 8), and the regenerated SPIR-V (`tools/generate_fsr3_3_1_6_framegen_spirv.sh`,
  `FSR3_316_ALLOW_UNPINNED_DXC=1` with DXC 2026-07-29; the optical flow modules are unchanged).
