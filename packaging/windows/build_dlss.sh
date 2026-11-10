#!/usr/bin/env bash
# Builds gow3_dlss.dll (gpu/dlss_bridge, the only code that uses the NVIDIA DLSS SDK) with
# MSVC and puts it with NVIDIA's nvngx_dlss.dll next to out/medievil-runtime.exe; package.sh ships both.
# Needs Visual Studio 2022 (Build Tools) and a checkout of https://github.com/NVIDIA/DLSS:
#   DLSS_SDK_ROOT=F:/sdk/DLSS bash packaging/windows/build_dlss.sh
set -euo pipefail
cd -- "$(dirname -- "$0")/../.."
sdk=${DLSS_SDK_ROOT:?Set DLSS_SDK_ROOT to a checkout of github.com/NVIDIA/DLSS}
cmake=${CMAKE_EXE:-cmake}
# Only the Vulkan headers: MSYS2's C headers must not reach MSVC.
rm -rf out/vkinclude && mkdir -p out/vkinclude
inc="${MINGW_PREFIX:-/clang64}/include"  # the CLANG64 headers in any MSYS2 install (CI too)
cp -r "$inc/vulkan" "$inc/vk_video" out/vkinclude/
# The installed Visual Studio's generator (2022 here, 2026 on GitHub's current runners).
generator=${DLSS_VS_GENERATOR:-}
if [[ -z $generator ]]; then
    vswhere="$(cygpath -u "${ProgramFiles:-C:/Program Files} (x86)")/Microsoft Visual Studio/Installer/vswhere.exe"
    [[ -x $vswhere ]] || vswhere="/c/Program Files (x86)/Microsoft Visual Studio/Installer/vswhere.exe"
    version=$("$vswhere" -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 \
        -property installationVersion | tr -d '\r')
    case ${version%%.*} in
    17) generator="Visual Studio 17 2022" ;;
    18) generator="Visual Studio 18 2026" ;;
    *) echo "No Visual Studio 2022/2026 with the C++ tools found (vswhere: '$version')" >&2; exit 1 ;;
    esac
fi
echo "DLSS bridge: $generator"
rm -rf out/dlss-bridge
"$cmake" -S gpu/dlss_bridge -B out/dlss-bridge -G "$generator" -A x64 \
    -DDLSS_SDK_ROOT="$(cygpath -m "$sdk")" -DVULKAN_INCLUDE="$(cygpath -m "$PWD/out/vkinclude")"
"$cmake" --build out/dlss-bridge --config Release
cp out/dlss-bridge/Release/gow3_dlss.dll "$sdk/lib/Windows_x86_64/rel/nvngx_dlss.dll" out/
cp "$sdk/LICENSE.txt" out/NVIDIA-DLSS-LICENSE.txt
echo "DLSS bridge ready: out/gow3_dlss.dll, out/nvngx_dlss.dll"
