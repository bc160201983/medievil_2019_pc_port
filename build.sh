#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
mkdir -p out
# Windows: MSYS2 CLANG64 (packages in packaging/windows/README.md). The GPU library is static
# there and CMake links medievil-runtime.exe as well; no PGO (GCC-only flags). ThinLTO on by default
# (GOW3_LTO=OFF disables it): stage B per draw equal or slightly better, t-lto-a1/b, 2026-10-30.
case $(uname -s) in
MINGW*|MSYS*)
    # shellcheck source=msys2-env.sh
    source ./msys2-env.sh
    for tool in clang cmake ninja pkg-config; do
        command -v "$tool" >/dev/null || { echo "Need $tool from MSYS2 CLANG64 (packaging/windows/README.md)." >&2; exit 1; }
    done
    if [[ ! -f gpu/third_party/fsr-vulkan/CMakeLists.txt || ! -f gpu/third_party/imgui/imgui.h ||
          ! -f gpu/third_party/xbyak/xbyak/xbyak.h ]]; then
        git submodule update --init --recursive
    fi
    for patch in gpu/patches/fsr-vulkan/*.patch; do
        if ! git -C gpu/third_party/fsr-vulkan apply --reverse --check "$PWD/$patch" 2>/dev/null; then
            git -C gpu/third_party/fsr-vulkan apply "$PWD/$patch"
        fi
    done
    cmake -S gpu -B out/gpu -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DGOW3_PGO=off \
        -DGOW3_LTO="${GOW3_LTO:-ON}" -DGOW3_CACHELINE_PADDING="${GOW3_CACHELINE_PADDING:-ON}" \
        -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ >/dev/null
    echo "GPU library and loader: LTO ${GOW3_LTO:-ON}"
    if ! ninja -C out/gpu medievil-runtime medievil-gpu-check medievil-play > out/gpu-build.log 2>&1; then
        grep -v '^\[' out/gpu-build.log | tail -40 >&2
        echo 'Build failed (full log: out/gpu-build.log)' >&2; exit 1
    fi
    cp out/gpu/medievil-runtime.exe out/gpu/medievil-gpu-check.exe out/gpu/medievil-play.exe out/
    echo "Built $PWD/out/medievil-runtime.exe"
    exit 0
    ;;
esac
if [[ -z ${CC:-} ]]; then
    CC=$(command -v cc || command -v gcc || true)
    if [[ -z $CC ]]; then
        for candidate in /nix/store/*-gcc-wrapper-*/bin/gcc; do
            if [[ -x $candidate ]]; then CC=$candidate; break; fi
        done
    fi
fi
if [[ -z ${CC:-} ]]; then echo 'Install GCC/Clang or set CC.' >&2; exit 1; fi
# Dependencies come from pkg-config (Vulkan loader/headers, SDL3). On NixOS the
# environment is provided by shell.nix; re-enter it automatically if needed.
if ! { command -v pkg-config >/dev/null && pkg-config --exists vulkan sdl3 && command -v cmake >/dev/null && command -v ninja >/dev/null; }; then
    if [[ -z ${GOW3_IN_NIX_SHELL:-} ]] && command -v nix-shell >/dev/null; then
        exec env GOW3_IN_NIX_SHELL=1 nix-shell shell.nix --run "bash build.sh $*"
    fi
    echo 'Need pkg-config with vulkan and sdl3, cmake and ninja (see shell.nix).' >&2; exit 1
fi
read -r -a includes <<< "$(pkg-config --cflags vulkan sdl3 libavcodec libavutil libswscale)"
read -r -a libraries <<< "$(pkg-config --libs vulkan sdl3 libavcodec libavutil libswscale)"
# GPU library (shadPS4 video core + drivers), built by CMake.
# GOW3_PGO: generate (instrumented build that writes pgo/ while the game runs), use, off.
# Default: use the profile in pgo/ when there is one. GOW3_LTO=OFF disables link-time optimization.
# GOW3_STATIC_GPU (default ON): CMake links the library into medievil-runtime as on Windows (direct
# calls and TLS, one ThinLTO unit with the runtime C). OFF: out/gpu/libgow3gpu.so, linked below.
static=ON
case ${GOW3_STATIC_GPU:-ON} in OFF|off|0|NO|no|FALSE|false) static=OFF ;; esac
# GOW3_CPU_LEVEL=v3 (study-deck-build-optimization.md S1; impl-deck-build-sched-pool.md): host code
# for x86-64-v3 CPUs (AVX2, BMI2, FMA), tuned for Zen 2 (Steam Deck), linked with lld; static GPU
# library only, built in out/gpu-v3 so the default tree (out/gpu) stays incremental. out/cpu-level
# names the flavour of out/medievil-runtime for appimage.sh (absent: the default, any x86-64 CPU).
cpu_level=${GOW3_CPU_LEVEL:-}
case $cpu_level in ''|v3) ;; *) echo "GOW3_CPU_LEVEL: empty or v3, not '$cpu_level'" >&2; exit 1 ;; esac
build=out/gpu
if [[ $cpu_level == v3 ]]; then
    [[ $static == ON ]] || { echo 'GOW3_CPU_LEVEL=v3 needs the static GPU library (GOW3_STATIC_GPU=ON).' >&2; exit 1; }
    command -v ld.lld >/dev/null || { echo 'GOW3_CPU_LEVEL=v3 links with lld: install lld.' >&2; exit 1; }
    build=out/gpu-v3
fi
pgo=${GOW3_PGO:-}
if [[ -z $pgo ]]; then
    if [[ -n $(find pgo -name '*.gcda' -print -quit 2>/dev/null) ]]; then pgo=use; else pgo=off; fi
fi
mkdir -p pgo
# Submodules (git clone --recursive, or: git submodule update --init) and this port's changes
# to FSR-Vulkan (gpu/patches/fsr-vulkan), applied to its working tree once.
if [[ ! -f gpu/third_party/fsr-vulkan/CMakeLists.txt || ! -f gpu/third_party/imgui/imgui.h ]]; then
    git submodule update --init --recursive
fi
for patch in gpu/patches/fsr-vulkan/*.patch; do
    if ! git -C gpu/third_party/fsr-vulkan apply --reverse --check "$PWD/$patch" 2>/dev/null; then
        git -C gpu/third_party/fsr-vulkan apply "$PWD/$patch"
    fi
done
cmake -S gpu -B "$build" -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DGOW3_PGO="$pgo" \
    -DGOW3_LTO="${GOW3_LTO:-ON}" -DGOW3_STATIC_GPU="$static" -DGOW3_PGO_DIR="$PWD/pgo" \
    -DGOW3_CACHELINE_PADDING="${GOW3_CACHELINE_PADDING:-ON}" -DGOW3_CPU_LEVEL="$cpu_level" >/dev/null
echo "GPU library: PGO $pgo, LTO ${GOW3_LTO:-ON}, static $static, CPU level ${cpu_level:-baseline}"
targets=(gow3gpu)
if [[ $static == ON ]]; then
    targets=(medievil-runtime)
    if [[ ${1:-} == --test ]]; then targets+=(runtime-test sema-test); fi
fi
# A failed GPU build must stop here: an older library or runtime would otherwise be used silently.
if ! ninja -C "$build" "${targets[@]}" > out/gpu-build.log 2>&1; then
    grep -v '^\[' out/gpu-build.log | tail -40 >&2
    echo 'GPU library build failed (full log: out/gpu-build.log)' >&2; exit 1
fi
# $ORIGIN/gpu: packaged copies keep the library next to the binary without patching it.
gpu=(-Lout/gpu -lgow3gpu -Wl,-rpath,'$ORIGIN/gpu' -Wl,-rpath,"$PWD/out/gpu" -rdynamic)
runtime=(src/runtime*.c)
# Third-party decoders: compiled once, without this project's -Werror policy.
atrac9=(third_party/LibAtrac9/C/src/*.c)
if [[ ! -f out/libatrac9.a || -n $(find third_party/LibAtrac9/C/src -newer out/libatrac9.a -name '*.c') ]]; then
    rm -rf out/atrac9 && mkdir -p out/atrac9
    for source in "${atrac9[@]}"; do "$CC" -std=c99 -O2 -g -w -c "$source" -o "out/atrac9/$(basename "${source%.c}").o"; done
    ar rcs out/libatrac9.a out/atrac9/*.o
fi
if [[ $static == ON ]]; then
    cp "$build/medievil-runtime" out/
    # A shared library left by an earlier GOW3_STATIC_GPU=OFF build is not used; nothing may package it.
    rm -f out/gpu/libgow3gpu.so
else
    "$CC" -std=c11 -O2 -g -Wall -Wextra -Werror -pthread -no-pie "${includes[@]}" -I. -Isrc src/probe.c "${runtime[@]}" src/vulkan_smoke.c out/libatrac9.a -lm "${gpu[@]}" "${libraries[@]}" -o out/medievil-runtime
fi
if [[ -n $cpu_level ]]; then echo "x86-64-$cpu_level" > out/cpu-level; else rm -f out/cpu-level; fi
echo "Built $PWD/out/medievil-runtime"
# GPU check for run.sh (live_resolution=auto): links only the Vulkan loader.
"$CC" -std=c11 -O2 -Wall -Wextra -Werror tools/gpu_capabilities.c "${libraries[@]}" -o out/medievil-gpu-check
if [[ ${1:-} == --test ]]; then
    "$CC" -std=c11 -O2 -g -Wall -Wextra -Werror -pthread "${includes[@]}" -I. -Isrc tests/test_pad.c "${libraries[@]}" -o out/pad-test
    out/pad-test
    if [[ $static == ON ]]; then
        cp "$build/runtime-test" "$build/sema-test" out/
    else
        "$CC" -std=c11 -O2 -g -Wall -Wextra -Werror -pthread -I. -Isrc tests/test_runtime.c "${runtime[@]}" out/libatrac9.a -lm "${gpu[@]}" "${libraries[@]}" -o out/runtime-test
        "$CC" -std=c11 -O2 -g -Wall -Wextra -Werror -pthread -I. -Isrc tests/test_sema.c "${runtime[@]}" out/libatrac9.a -lm "${gpu[@]}" "${libraries[@]}" -o out/sema-test
    fi
    out/runtime-test
    "$CC" -std=c11 -O2 -g -Wall -Wextra -Werror -pthread -Isrc tests/test_file_mods.c -o out/file-mods-test
    out/file-mods-test
    out/sema-test
    "$CC" -std=c11 -D_GNU_SOURCE -O2 -g -Wall -Wextra -Werror -I. -Isrc tests/test_content.c src/runtime_content.c -o out/content-test
    out/content-test
    "$CC" -std=c11 -O2 -g -Wall -Wextra -Werror -pthread "${includes[@]}" -I. -Isrc tests/test_mp3.c src/runtime_mp3.c out/libatrac9.a "${libraries[@]}" -o out/mp3-test
    out/mp3-test tests/data/sine.mp3
    "$CC" -std=c11 -O2 -g -Wall -Wextra -Werror -pthread "${includes[@]}" -I. -Isrc tests/test_videodec.c "${libraries[@]}" -o out/videodec-test
    out/videodec-test tests/data/tiny.h264
    "$CC" -std=c11 -O2 -g -Wall -Wextra -Werror -pthread -I. -Isrc tests/test_services.c -o out/services-test
    out/services-test
fi
