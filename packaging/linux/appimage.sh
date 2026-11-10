#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-2.0-or-later
# Builds dist/MediEvil-x86_64.AppImage from the Linux build (run build.sh first);
# dist/MediEvil-x86_64-v3.AppImage from a GOW3_CPU_LEVEL=v3 build (its AppRun checks the CPU).
#
# What is inside: medievil-runtime (the GPU library linked in; or libgow3gpu.so beside it from a
# GOW3_STATIC_GPU=OFF build), the libraries a desktop may not have
# (SDL3, FFmpeg, xxHash), a standalone Python 3.13 with Tk, Pillow and cryptography for the
# launcher and the preparation scripts, and the port's scripts and launcher. The host provides
# glibc, the Vulkan loader and the GPU's Vulkan driver (Mesa, NVIDIA), as for a native game.
# libstdc++ is bundled as a fallback, used only when the host's is older than the build's.
#
# Usage: bash packaging/linux/appimage.sh [--python DIR] [--appimagetool FILE] [--runtime FILE]
#   --python DIR       an unpacked python-build-standalone "install_only" tree (python/ inside)
#                      with Pillow and cryptography installed; downloaded and set up when missing
#   --appimagetool     appimagetool-x86_64.AppImage (downloaded when missing)
#   --runtime          the type2 runtime (downloaded when missing)
# Downloads are pinned (PINS below): each is checked against its SHA-256 before use, and a cached
# copy that does not match is downloaded again. Local files given with the options above are the
# builder's choice; a hash different from the pin is reported, not refused.
# Libraries are looked up with ldd, so build.sh's PKG_CONFIG_PATH/LD_LIBRARY_PATH must also be
# set here when the dependencies live in a private prefix (MEDIEVIL_DEPS=/opt/mdeps does that).
set -euo pipefail
cd -- "$(dirname -- "$0")/../.."
root=$PWD
python_dir='' tool='' runtime=''
while (( $# )); do
    case $1 in
        --python) python_dir=$2; shift 2 ;;
        --appimagetool) tool=$2; shift 2 ;;
        --runtime) runtime=$2; shift 2 ;;
        *) echo "Unknown option $1" >&2; exit 1 ;;
    esac
done
[[ -f out/medievil-runtime ]] || { echo 'Build first: bash build.sh' >&2; exit 1; }
# A GOW3_STATIC_GPU=OFF build loads libgow3gpu.so (DT_NEEDED); the default build has it linked in.
shared_gpu=0
if objdump -p out/medievil-runtime | grep 'NEEDED *libgow3gpu\.so' >/dev/null; then
    shared_gpu=1
    [[ -f out/gpu/libgow3gpu.so ]] || { echo 'out/gpu/libgow3gpu.so missing: bash build.sh' >&2; exit 1; }
fi
# The CPU level build.sh built for (GOW3_CPU_LEVEL; out/cpu-level, absent for the default build).
cpu_level=$(cat out/cpu-level 2>/dev/null || true)
case $cpu_level in ''|x86-64-v3) ;; *) echo "Unknown CPU level in out/cpu-level: $cpu_level" >&2; exit 1 ;; esac
if [[ -n ${MEDIEVIL_DEPS:-} ]]; then
    export LD_LIBRARY_PATH=$MEDIEVIL_DEPS/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
fi
# ---- Pinned downloads (version, URL, SHA-256) ------------------------------------------------
# To update one: change the version and URL, download the file, put its sha256sum here.
# python-build-standalone publishes SHA256SUMS with each release (the hash below matches it).
# appimagetool and type2-runtime publish no checksum files; their hashes were taken from the
# tagged release assets (downloaded twice, identical) and can be compared with the SHA-256 that
# GitHub shows for each asset on the release page.
APPIMAGETOOL_VERSION=1.9.1
APPIMAGETOOL_URL=https://github.com/AppImage/appimagetool/releases/download/$APPIMAGETOOL_VERSION/appimagetool-x86_64.AppImage
APPIMAGETOOL_SHA256=ed4ce84f0d9caff66f50bcca6ff6f35aae54ce8135408b3fa33abfc3cb384eb0
RUNTIME_VERSION=20251108
RUNTIME_URL=https://github.com/AppImage/type2-runtime/releases/download/$RUNTIME_VERSION/runtime-x86_64
RUNTIME_SHA256=2fca8b443c92510f1483a883f60061ad09b46b978b2631c807cd873a47ec260d
# Python 3.13.16 with Tk 9.0.4 (build-tk-xft.sh pins the matching Tcl/Tk sources).
PYTHON_RELEASE=20261013
PYTHON_VERSION=3.13.16
PYTHON_URL=https://github.com/astral-sh/python-build-standalone/releases/download/$PYTHON_RELEASE/cpython-$PYTHON_VERSION+$PYTHON_RELEASE-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz
PYTHON_SHA256=d0080766b6b74aeb81e1949ef0d9a20e6c17f7b160fb295a69b936ccb17660be
# The launcher's Python packages (binary wheels only), with their dependencies. Pillow 13.0.0:
# 12.x has known image-decoder vulnerabilities (OSV: 4 critical, fixed in 13.0.0), and the
# launcher opens images from the player's game files.
PYTHON_PACKAGES=(pillow==13.0.0 cryptography==50.0.1 cffi==2.1.1 pycparser==3.0)

tools=out/appimage-tools
mkdir -p "$tools"
sha256_of() { sha256sum "$1" | cut -d' ' -f1; }
# fetch URL SHA256 FILE: FILE when it already has that hash, else downloaded and checked.
fetch() {
    local url=$1 want=$2 file=$3
    if [[ -s $file && $(sha256_of "$file") == "$want" ]]; then return 0; fi
    [[ -e $file ]] && echo "Cached $(basename "$file") does not match its pin; downloading it again."
    rm -f "$file" "$file.part"
    curl -fsSL -o "$file.part" "$url" || { echo "Download failed: $url" >&2; rm -f "$file.part"; exit 1; }
    local got
    got=$(sha256_of "$file.part")
    if [[ $got != "$want" ]]; then
        rm -f "$file.part"
        echo "SHA-256 mismatch for $url" >&2
        echo "  expected $want" >&2
        echo "  got      $got" >&2
        exit 1
    fi
    mv "$file.part" "$file"
}
# check_local FILE SHA256 NAME: a file given on the command line; reported when it differs.
check_local() {
    local got
    got=$(sha256_of "$1")
    [[ $got == "$2" ]] || echo "Note: $3 $1 is not the pinned build (sha256 $got)."
}
if [[ -z $tool ]]; then
    tool=$tools/appimagetool-$APPIMAGETOOL_VERSION
    fetch "$APPIMAGETOOL_URL" "$APPIMAGETOOL_SHA256" "$tool"
    chmod +x "$tool"
else
    check_local "$tool" "$APPIMAGETOOL_SHA256" appimagetool
fi
if [[ -z $runtime ]]; then
    runtime=$tools/runtime-x86_64-$RUNTIME_VERSION
    fetch "$RUNTIME_URL" "$RUNTIME_SHA256" "$runtime"
else
    check_local "$runtime" "$RUNTIME_SHA256" runtime
fi
if [[ -z $python_dir ]]; then
    python_dir=$tools/python-$PYTHON_VERSION-$PYTHON_RELEASE
    archive=$tools/python-$PYTHON_VERSION-$PYTHON_RELEASE.tar.gz
    fetch "$PYTHON_URL" "$PYTHON_SHA256" "$archive"
    # Set up again when the pinned packages changed since this tree was made.
    if [[ ! -x $python_dir/python/bin/python3 ||
          $(cat "$python_dir/.packages-installed" 2>/dev/null) != "$(printf '%s\n' "${PYTHON_PACKAGES[@]}")" ]]; then
        rm -rf "$python_dir" && mkdir -p "$python_dir" && tar -C "$python_dir" -xzf "$archive"
        "$python_dir/python/bin/python3" -m pip install --quiet --no-cache-dir --only-binary=:all: \
            "${PYTHON_PACKAGES[@]}"
        printf '%s\n' "${PYTHON_PACKAGES[@]}" > "$python_dir/.packages-installed"
    fi
fi
py=$python_dir/python
# The Python calls the launcher and the scripts make, so a package update that drops one fails
# here and not in the player's launcher. Installed versions are compared with the pins.
"$py/bin/python3" -I - "${PYTHON_PACKAGES[@]}" <<'EOF' || { echo "$py: unusable for the launcher (above)" >&2; exit 1; }
import importlib.metadata, sys, tkinter
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageTk  # noqa: F401 (ImageTk: launcher)
from cryptography.hazmat.primitives.ciphers import Cipher  # noqa: F401 (pkg_extract.py)
image = Image.new('RGBA', (8, 8)).resize((4, 4), Image.LANCZOS)
image.filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.GaussianBlur(1))
ImageDraw.Draw(image).textbbox((0, 0), 'MediEvil', font=ImageFont.load_default(12))
ImageFont.truetype('gpu/third_party/fonts/DejaVuSans.ttf', 12)
for pin in sys.argv[1:]:
    name, want = pin.split('==')
    if (got := importlib.metadata.version(name)) != want:
        print(f'Note: {name} {got} is installed, the pin is {want}')
EOF

app=out/MediEvil.AppDir
rm -rf "$app"
share=$app/usr/share/medievil   # the port tree (PORT_DIR / PORT in the launcher and run.py)
lib=$app/usr/lib
mkdir -p "$share/bin" "$lib/compat" "$app/usr/share/fonts" "$app/usr/share/applications" \
    "$app/usr/share/icons/hicolor/256x256/apps" "$app/usr/share/metainfo"

# The port: scripts, launcher, patches, docs (no build tree, no Windows packaging).
cp -r run.py scripts launcher patches LICENSE README.md "$share/"
[[ -d docs ]] && cp -r docs "$share/"
mkdir -p "$share/gpu/third_party" && cp -r gpu/third_party/fonts "$share/gpu/third_party/"
mkdir -p "$share/packaging/linux" && cp packaging/linux/README-Linux.md "$share/packaging/linux/" 2>/dev/null || true
find "$share" -name '__pycache__' -prune -exec rm -rf {} +
# Native programs: run.py finds bin/medievil-runtime; with a shared GPU library its RUNPATH has
# $ORIGIN/gpu.
install -m755 out/medievil-runtime "$share/bin/"
install -m755 out/medievil-gpu-check "$share/bin/" 2>/dev/null || true
patchelf=${PATCHELF:-$(command -v patchelf || echo "${MEDIEVIL_DEPS:-/opt/mdeps}/bin/patchelf")}
elves=("$share/bin/medievil-runtime")
if (( shared_gpu )); then
    mkdir -p "$share/bin/gpu"
    install -m755 out/gpu/libgow3gpu.so "$share/bin/gpu/"
    elves+=("$share/bin/gpu/libgow3gpu.so")
    "$patchelf" --set-rpath '$ORIGIN/gpu:$ORIGIN/../../../lib' "$share/bin/medievil-runtime"
    "$patchelf" --set-rpath '$ORIGIN:$ORIGIN/../../../../lib' "$share/bin/gpu/libgow3gpu.so"
else
    "$patchelf" --set-rpath '$ORIGIN/../../../lib' "$share/bin/medievil-runtime"
fi
[[ -f $share/bin/medievil-gpu-check ]] && "$patchelf" --set-rpath '$ORIGIN/../../../lib' "$share/bin/medievil-gpu-check"

# Bundled libraries: everything the programs need that is not part of a normal desktop.
# Host-provided: glibc, libstdc++/libgcc (fallback copy below), X11/xcb, zlib; GPU drivers.
host_lib() {
    case $1 in
        ld-linux*|libc.so*|libm.so*|libdl.so*|libpthread.so*|librt.so*|libresolv.so*|libutil.so*) return 0 ;;
        libstdc++.so*|libgcc_s.so*) return 0 ;;
        libX11.so*|libX11-xcb.so*|libxcb*.so*|libXau.so*|libXdmcp.so*|libXext.so*|libz.so*) return 0 ;;
        libbsd.so*|libmd.so*) return 0 ;;   # X11's own dependencies
        libgow3gpu.so) return 0 ;;          # installed beside the runtime (bin/gpu)
        libGL*|libEGL*|libdrm*|libnvidia*|libvulkan*|libgbm*) return 0 ;;  # GPU stack: the host's (one Vulkan loader)
        # The host's Mesa Vulkan drivers link the host's libwayland-client and need its newer
        # symbols (wl_display_dispatch_queue_timeout); a bundled older copy loaded first by SDL3
        # made every driver fail to load on SteamOS ("doesn't implement VK_KHR_surface").
        # From spacepepperoni/medievil_ps4_Linux 22d6f92.
        libwayland-*.so*) return 0 ;;
    esac
    return 1
}
collect() {
    local elf=$1 name path
    while read -r name _ path _; do
        [[ $path == /* ]] || continue
        host_lib "$name" && continue
        if [[ ! -e $lib/$name ]]; then
            cp -L "$path" "$lib/$name"
            chmod 644 "$lib/$name"
            "$patchelf" --set-rpath '$ORIGIN' "$lib/$name" 2>/dev/null || true
            echo "  bundled $name"
        fi
    done < <(ldd "$elf" | grep '=>')
}
echo 'Libraries:'
for elf in "${elves[@]}"; do collect "$elf"; done
# The Vulkan loader is the host's: a second, bundled loader next to it crashed (interposed symbols).
for compat in libstdc++.so.6 libgcc_s.so.1; do
    path=$("${CXX:-g++}" -print-file-name="$compat")
    [[ -f $path ]] && cp -L "$path" "$lib/compat/$compat"
done
# The newest GLIBCXX symbol version the bundle needs: AppRun uses the bundled libstdc++ when the
# host's lacks it.
newest() { # PREFIX FILE...: the newest symbol version PREFIX_x.y the files import
    local prefix=$1 f; shift
    for f in "$@"; do
        [[ -f $f ]] && file -b "$f" | grep -q ELF && objdump -T "$f" 2>/dev/null || true
    done | grep -o "${prefix}_[0-9.]*" | sort -V | tail -1 || true
}
native=("${elves[@]}" "$lib"/*.so*)
need=$(newest GLIBCXX "${native[@]}")
echo "${need:-GLIBCXX_3.4}" > "$lib/compat/needs"
# AppRun refuses to start a GOW3_CPU_LEVEL=v3 build on a CPU without AVX2 / BMI2 / FMA.
[[ -n $cpu_level ]] && echo "$cpu_level" > "$lib/compat/cpu-level"

# Python (launcher, run.py and the preparation scripts).
cp -a "$py" "$app/usr/python"
rm -rf "$app/usr/python/lib/python3"*/test "$app/usr/python/lib/python3"*/idlelib/idle_test \
    "$app/usr/python/lib/python3"*/ensurepip "$app/usr/python/lib/python3"*/lib2to3 2>/dev/null || true
# Tk with Xft (anti-aliased fonts) instead of the standalone build's X11 core fonts.
if ! ldd "$app"/usr/python/lib/libtcl*tk*.so 2>/dev/null | grep -q libXft; then
    PATCHELF=$patchelf bash packaging/linux/build-tk-xft.sh "$app/usr/python" "$root/$tools/tk-xft"
fi
rm -f "$app"/usr/python/lib/libtcl*tk*.so.orig
# Bytecode inside the image: a read-only AppImage cannot write it at run time.
"$app/usr/python/bin/python3" -I -m compileall -q -j0 "$app/usr/python/lib" "$share/scripts" "$share/launcher" "$share/run.py" >/dev/null 2>&1 || true
mapfile -t python_elves < <(find "$app/usr/python" -type f \( -name '*.so*' -o -path '*/bin/python3*' \))
glibc=$(newest GLIBC "${native[@]}" "${python_elves[@]}")
echo "Needs glibc ${glibc#GLIBC_}+ from the host; libstdc++ ${need} (bundled fallback)"
echo "${glibc#GLIBC_}" > "$lib/compat/glibc"

# Fonts for the launcher without the game's own (DejaVu: the GPU library's overlay font too).
cp gpu/third_party/fonts/*.ttf gpu/third_party/fonts/LICENSE-DejaVu.txt "$app/usr/share/fonts/"

# Desktop integration.
install -m755 packaging/linux/AppRun "$app/AppRun"
cp launcher/medievil.png "$app/usr/share/icons/hicolor/256x256/apps/medievil-pc.png"
cp launcher/medievil.png "$app/medievil-pc.png"
cat > "$app/usr/share/applications/medievil-pc.desktop" <<'EOF'
[Desktop Entry]
Type=Application
Name=MediEvil (PC)
Comment=MediEvil (PS4, 2019) running natively on Linux from your own copy of the game
Exec=AppRun
Icon=medievil-pc
Terminal=false
Categories=Game;ActionGame;
Keywords=MediEvil;PS4;Sir Dan;
EOF
cp "$app/usr/share/applications/medievil-pc.desktop" "$app/"
cat > "$app/usr/share/metainfo/medievil-pc.appdata.xml" <<'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<component type="desktop-application">
  <id>medievil-pc</id>
  <metadata_license>CC0-1.0</metadata_license>
  <project_license>GPL-2.0-or-later</project_license>
  <name>MediEvil (PC)</name>
  <summary>Native PC port runtime for MediEvil (PS4); bring your own game files</summary>
  <description><p>Runs the PlayStation 4 MediEvil executable from your own copy of the game. No game files included.</p></description>
  <launchable type="desktop-id">medievil-pc.desktop</launchable>
  <url type="homepage">https://github.com/bc160201983/medievil_2019_pc_port</url>
  <content_rating type="oars-1.1"/>
</component>
EOF

mkdir -p dist
out=dist/MediEvil-x86_64${cpu_level:+-${cpu_level#x86-64-}}.AppImage
rm -f "$out"
# No FUSE needed to build: the tool runs from its extracted copy.
ARCH=x86_64 APPIMAGE_EXTRACT_AND_RUN=1 "$tool" --no-appstream --runtime-file "$runtime" "$app" "$out" >out/appimagetool.log 2>&1 \
    || { tail -20 out/appimagetool.log >&2; exit 1; }
ls -lh "$out"
