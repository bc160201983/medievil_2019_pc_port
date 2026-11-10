"""Unreal Engine 4 render resolution override for MediEvil (up to 4K).

UE4 games read their config only from their paks. A small pak whose name ends in _P mounts with
priority over them, so a changed copy of Engine/Config/PS4/PS4Engine.ini (taken from the game's
own pak) wins. r.ScreenPercentage scales the 3D scene of the game's output (medievil.ini
render_scale): 200% of a 1080p output renders it at 3840x2160 and scales it down (supersampling:
much sharper edges and detail); below 100% UE4's temporal upsampling rebuilds the output from the
game's own motion vectors. The HUD stays at the output size. The output size itself is booted by
run.py (output_resolution). The pak is written into the game's Paks folder at every start and
removed when it changes nothing.

Usage: python scripts/ue4_override.py <game dir> [--output 2560x1440] [--scale 67]
"""
import argparse
import hashlib
import os
import re
import struct
import zlib
from pathlib import Path

PAK_MAGIC = 0x5A6F12E1
# The game's pak loader opens the patch pak in lowercase (`pakchunk99-ps4_p.pak`); on a
# case-sensitive filesystem the `_P` spelling alone is never mounted. Windows ignores case.
PAK_NAME = 'pakchunk99-ps4_p.pak'
OLD_PAK_NAME = 'pakchunk99-ps4_P.pak'
CONFIG = 'Engine/Config/PS4/PS4Engine.ini'
# Render scales below 100% use UE4's temporal upsampling: its temporal AA rebuilds the full
# output from the game's own motion vectors (as FSR 2 / DLSS do, without the AI).


# r.SetRes (native output, runtime_extra.c): UE4's FSystemResolution::RequestResolutionChange loads
# the variable's IConsoleVariable* with `mov rbx, [rip+rel32]` right before this code. The slot
# differs between eboots (US 01.00: 0x5c28328; the 01.02 dump of PR #29: 0x5c28368), so run.py
# finds it in each eboot.elf instead of using one address.
SETRES_CODE = re.compile(re.escape(bytes.fromhex('488b1d')) + b'(....)' + re.escape(bytes.fromhex(
    '837db80074044c8b75b0488b034889dfff5020488b0b25000000ff')), re.S)


def setres_slot(elf_path):
    """Image offset of r.SetRes's IConsoleVariable* in eboot.elf; None unless its code is found once."""
    elf = Path(elf_path).read_bytes()
    hits = list(SETRES_CODE.finditer(elf))
    if len(hits) != 1:
        return None
    offset = hits[0].start()
    phoff, = struct.unpack_from('<Q', elf, 0x20)
    entry_size, count = struct.unpack_from('<HH', elf, 0x36)
    for i in range(count):
        kind, _, file_offset, vaddr, _, file_size, _, _ = struct.unpack_from('<IIQQQQQQ', elf, phoff + i * entry_size)
        if kind == 1 and file_offset <= offset < file_offset + file_size:
            rel, = struct.unpack('<i', hits[0].group(1))
            return offset - file_offset + vaddr + 7 + rel
    return None


def paks_dir(game):
    return Path(game) / 'overbite' / 'content' / 'paks'


def read_index(path):
    """{name: (offset, size, compression, zlib blocks)} of a version 7 pak (unencrypted index)."""
    with open(path, 'rb') as f:
        f.seek(os.path.getsize(path) - 44)
        magic, version, index_offset, index_size = struct.unpack('<IIQQ', f.read(24))
        if magic != PAK_MAGIC or version != 7:
            raise ValueError(f'{path.name}: unsupported pak (magic {magic:#x}, version {version})')
        f.seek(index_offset)
        index = f.read(index_size)
    pos = 0

    def fstring():
        nonlocal pos
        n, = struct.unpack_from('<i', index, pos)
        pos += 4
        if n < 0:
            text = index[pos:pos - 2 * n].decode('utf-16le')[:-1]
            pos += -2 * n
        else:
            text = index[pos:pos + n].decode('latin1')[:-1]
            pos += n
        return text
    fstring()  # mount point
    count, = struct.unpack_from('<i', index, pos)
    pos += 4
    entries = {}
    for _ in range(count):
        name = fstring()
        offset, size, _uncompressed, compression = struct.unpack_from('<qqqi', index, pos)
        pos += 48
        blocks = []
        if compression:
            count_blocks, = struct.unpack_from('<i', index, pos)
            pos += 4
            for _ in range(count_blocks):
                blocks.append(struct.unpack_from('<qq', index, pos))  # [start, end) from the entry
                pos += 16
        pos += 5
        entries[name] = (offset, size, compression, blocks)
    return entries


def read_file(game, name):
    """A file of the game's own paks (stored or zlib-compressed)."""
    for pak in sorted(paks_dir(game).glob('pakchunk*-ps4.pak')):
        entries = read_index(pak)
        if name in entries:
            offset, size, compression, blocks = entries[name]
            with open(pak, 'rb') as f:
                if not compression:
                    f.seek(offset + 53)  # the entry header in front of the data
                    return f.read(size)
                data = bytearray()
                for start, end in blocks:
                    f.seek(offset + start)
                    data += zlib.decompress(f.read(end - start))
                return bytes(data)
    raise FileNotFoundError(name)


def fstring(text):
    data = text.encode('latin1') + b'\0'
    return struct.pack('<i', len(data)) + data


def entry(offset, data):
    """FPakEntry, version 7, uncompressed and unencrypted (as UnrealPak writes it)."""
    return struct.pack('<qqqi', offset, len(data), len(data), 0) + hashlib.sha1(data).digest() + b'\0' + \
        struct.pack('<I', 0)


def write_pak(path, files, mount='../../../'):
    body, records = b'', []
    for name, data in files.items():
        records.append((name, len(body)))
        body += entry(0, data) + data  # the copy in front of the data has offset 0
    index = fstring(mount) + struct.pack('<i', len(files))
    for (name, offset), data in zip(records, files.values()):
        index += fstring(name) + entry(offset, data)
    footer = bytes(16) + b'\0' + struct.pack('<IIQQ', PAK_MAGIC, 7, len(body), len(index)) + \
        hashlib.sha1(index).digest()
    tmp = path.with_suffix('.tmp')
    tmp.write_bytes(body + index + footer)
    tmp.replace(path)


def set_section(text, section, values):
    """Sets keys in the first [section] of an ini text, replacing existing lines there."""
    head, sep, rest = text.partition(f'[{section}]\n')
    if not sep:
        head, sep, rest = text.rstrip('\n') + '\n\n', f'[{section}]\n', ''
    lines = rest.split('\n')
    section_end = next((i for i, l in enumerate(lines) if l.startswith('[')), len(lines))
    body = [l for l in lines[:section_end] if l.strip() and l.split('=', 1)[0].strip() not in values]
    body += [f'{k}={v}' for k, v in values.items()] + ['']
    return head + sep + '\n'.join(body + lines[section_end:])


def set_values(ini, values, engine=None):
    """Sets keys in [SystemSettings] (console variables) and [/Script/Engine.Engine]."""
    text = ini.decode('utf-8-sig').replace('\r\n', '\n')
    if values:
        text = set_section(text, 'SystemSettings', values)
    if engine:
        text = set_section(text, '/Script/Engine.Engine', engine)
    return text.replace('\n', '\r\n').encode('utf-8')


def frame_rate_range(fps):
    """UGameEngine::GetMaxTickRate on consoles: with VSync on, the game ticks at most at the upper
    bound of SmoothedFrameRateRange (BaseEngine.ini: below 62, so 60 FPS), even with
    bSmoothFrameRate off. MediEvil keeps that default; a higher bound lifts the 60 FPS cap."""
    return {'SmoothedFrameRateRange': '(LowerBound=(Type="ERangeBoundTypes::Inclusive",Value=22),'
                                      f'UpperBound=(Type="ERangeBoundTypes::Exclusive",Value={fps + 2}))'}


# Graphics quality (medievil.ini ue4_quality): console keeps the PS4 settings; medium and low trade
# effects for speed on integrated and older graphics cards.
QUALITY = {
    'console': {},
    # Diagnostics (not in the launcher): console quality with one effect off.
    'diag_ao': {'r.AmbientOcclusionLevels': '0'},
    'diag_ssr': {'r.SSR.Quality': '0'},
    'diag_shadow': {'r.ShadowQuality': '2', 'r.Shadow.MaxCSMResolution': '1536'},
    'diag_mb': {'r.MotionBlurQuality': '0'},
    'diag_fxaa': {'r.PostProcessAAQuality': '2'},  # FXAA instead of temporal AA
    'diag_localres': {'r.Shadow.MaxResolution': '512'},  # point/spot light shadow maps
    'diag_nocube': {'r.AllowPointLightCubemapShadows': '0'},
    'diag_fixedres': {'r.Shadow.MinResolution': '1024', 'r.Shadow.MaxResolution': '1024'},
    # Fewer shadow draws (the GPU thread tops out near 2000 draws a frame at 60 FPS).
    'diag_draws': {'r.Shadow.MinResolution': '512', 'r.Shadow.MaxResolution': '512',
                   'r.Shadow.RadiusThreshold': '0.1', 'r.Shadow.CacheWholeSceneShadows': '1',
                   'r.Shadow.CSM.MaxCascades': '2', 'r.Shadow.DistanceScale': '0.8'},
    'diag_nopoint': {'r.Shadow.MinResolution': '512', 'r.Shadow.MaxResolution': '512',
                     'r.Shadow.CacheWholeSceneShadows': '1', 'r.AllowPointLightCubemapShadows': '0'},
    'diag_fixed512': {'r.Shadow.MinResolution': '512', 'r.Shadow.MaxResolution': '512',
                      'r.Shadow.RadiusThreshold': '0.05', 'r.Shadow.CacheWholeSceneShadows': '1'},
    'diag_lod': {'r.ViewDistanceScale': '0.7', 'foliage.DensityScale': '0.6', 'grass.DensityScale': '0.6',
                 'r.Shadow.CSM.MaxCascades': '2'},
    'diag_post': {'r.VolumetricFog': '0', 'r.DepthOfFieldQuality': '0', 'r.LensFlareQuality': '0',
                  'r.SceneColorFringeQuality': '0', 'r.BloomQuality': '2', 'r.RefractionQuality': '0'},
    # Above the PS4: GPU cost, little memory (the game keeps its PS4 memory budget, so texture
    # streaming pools stay as shipped).
    # Local light shadow sizes and the sun's distance / cascades: LOCAL_SHADOWS and SUN_SHADOWS.
    'high': {
        'r.ShadowQuality': '4', 'r.Shadow.MaxCSMResolution': '2048', 'r.AmbientOcclusionLevels': '3',
        'r.SSR.Quality': '3',
        'r.MaxAnisotropy': '16', 'r.ViewDistanceScale': '1.5', 'r.StaticMeshLODDistanceScale': '0.75',
        'foliage.LODDistanceScale': '1.5',
    },
    'ultra': {
        'r.ShadowQuality': '5', 'r.Shadow.MaxCSMResolution': '4096', 'r.AmbientOcclusionLevels': '3',
        'r.SSR.Quality': '4', 'r.MaxAnisotropy': '16', 'r.ViewDistanceScale': '2',
        'r.StaticMeshLODDistanceScale': '0.5', 'foliage.LODDistanceScale': '2', 'r.DetailMode': '2',
    },
    'medium': {
        'r.ShadowQuality': '2', 'r.Shadow.MaxCSMResolution': '1536', 'r.AmbientOcclusionLevels': '1', 'r.SSR.Quality': '1', 'r.MotionBlurQuality': '2',
        'r.DepthOfFieldQuality': '1', 'r.ViewDistanceScale': '0.85',
    },
    'low': {
        'r.ShadowQuality': '1', 'r.Shadow.MaxCSMResolution': '1024', 'r.VolumetricFog': '0', 'r.AmbientOcclusionLevels': '0',
        'r.SSR.Quality': '0', 'r.MotionBlurQuality': '0', 'r.DepthOfFieldQuality': '0',
        'r.LensFlareQuality': '0', 'r.SceneColorFringeQuality': '0', 'r.BloomQuality': '2',
        'r.RefractionQuality': '0', 'r.ViewDistanceScale': '0.7', 'foliage.DensityScale': '0.6',
        'grass.DensityScale': '0.6',
    },
}


# Point and spot light shadows at one fixed size per preset, with small lights unshadowed. UE4
# sizes them by their size on screen, so walking re-creates them constantly; those re-created
# maps flickered on the port (shadows blinking while moving, MediEvil's outdoor levels).
# The second number is r.Shadow.RadiusThreshold: shadow casters (objects) smaller than that on
# screen are left out of whole-scene (point, spot, cascade) shadow maps; it does not cull lights.
# Every shadowed point light redraws its surroundings six times, and the port's GPU thread keeps
# 60 FPS up to about 2000 draws a frame; busy outdoor scenes reached 4000 (30 FPS) before.
LOCAL_SHADOWS = {'ultra': (1024, 0.06), 'high': (1024, 0.08), 'console': (512, 0.1),
                 'medium': (512, 0.12), 'low': (256, 0.15)}
# Sun/moon shadow cascades and distance per preset (each cascade redraws the scene once).
SUN_SHADOWS = {'ultra': ('3', '1.5'), 'high': ('3', '1.2'), 'console': ('2', '0.8'),
               'medium': ('2', '0.75'), 'low': ('2', '0.6')}


# The developers' debug menu (R3 in a level: level select, complete level, gear and items for
# the level, runes, gold, life bottles, invulnerability). OverbiteGameSingleton reads the switch.
# Dan's Blueprint opens it on R3; on 01.02 only while L1 and R1 are held (its R3 event checks
# RightShoulderDown and LeftShoulderDown, set by its own shoulder key events). The main menu has
# no debug menu (research/patches/ue4/debug-menu-0102.md).
GAME_CONFIG = 'OVERBITE/Config/PS4/PS4Game.ini'


def debug_menu_config(game, enabled):
    text = read_file(game, GAME_CONFIG).decode('utf-8-sig').replace('\r\n', '\n')
    text = '\n'.join(l for l in text.split('\n') if not l.strip().startswith('AllowDebugMenu')).replace(' \n', '\n')
    text = set_section(text, '/Script/OVERBITE.OverbiteGameSingleton',
                       {'AllowDebugMenu': 'true' if enabled else 'false'})
    return text.replace('\n', '\r\n').encode('utf-8')


# The debug menu's widget Blueprint: when it is created it calls SetFrameRateLimit(60) and
# ApplySettings ("Set FPS Max to 30" off), which capped the game at 60 FPS again; 0 is no limit.
# Its "Cinematic Mode (1440p)" button (r.SetRes) becomes 4K and a profiling toggle 1440p. Same-length edits only (the Blueprint
# bytecode and the package's export sizes stay valid); each must match exactly once. An edit given as a
# list holds one variant per game version (object and name indices differ: US 01.00, then EU 01.02
# CUSA12982); the variant whose bytes are found is used.
DEBUG_HUD = 'OVERBITE/Content/Blueprints/Debug/DebugHUD.uexp'
DEBUG_HUD_EDITS = [
    # SetFrameRateLimit(60.0): EX_FloatConst 60.0 after the call to that import.
    [(bytes.fromhex('1c1dffffff1e00007042'), bytes.fromhex('1c1dffffff1e00000000'), 1),
     (bytes.fromhex('1c19ffffff1e00007042'), bytes.fromhex('1c19ffffff1e00000000'), 1)],
    # 4K needs the "Native 4K framebuffer pool" patch (patches/MediEvil.xml): the PS4 pool of
    # 128 MB ran out of memory.
    (b'r.SetRes 2560x1440\0', b'r.SetRes 3840x2160\0', 1),
    (b'Cinematic Mode (1440p)\0', b'Cinematic Mode (2160p)\0', 2),
    # The "Show Stat SceneRendering" toggle (a profiling overlay) switches to 1440p instead.
    (b'\x1fstat SceneRendering\0', b'\x1fr.SetRes 2560x1440 \0', 1),
    (b'Show Stat SceneRendering\0', b'Render 2560x1440 (1440p)\0', 2),
    # The menu pauses the enemies when it opens (Construct: SetAIPaused(true)) and only its own
    # back/Exit path (CloseDebug) resumed them: closing it with R3 again (the player's toggle) left
    # every enemy standing still, for the rest of the session (the flag is on the game instance).
    # Destruct runs however the menu closes: its gamma reset ("retain gamma" box) becomes
    # SetAIPaused(false, false, self), copied from CloseDebug, plus two EX_Nothing.
    [(bytes.fromhex('072d2b000000930000004d1920dafeffff13000000000000001c00ffffff171f67616d6d61002a16'),
      bytes.fromhex('072d2b000000930000000b1920c6feffff11000000000000001b5004000000000000282817160b0b'), 1),
     (bytes.fromhex('07d82b000000990000004d1920d5feffff13000000000000001cfbfeffff171f67616d6d61002a16'),
      bytes.fromhex('07d82b000000990000000b1920c0feffff11000000000000001b6704000000000000282817160b0b'), 1)],
]


def apply_edits(name, data, edits):
    """Same-length byte edits; a list entry is one variant per game version (the one found)."""
    for edit in edits:
        variants = edit if isinstance(edit, list) else [edit]
        for old, new, count in variants:
            if data.count(old) == count:
                data = data.replace(old, new)
                break
        else:
            raise ValueError(f'{name}: unexpected contents (game version?)')
    return data


# The debug menu's first resolution button follows the output resolution: the debug menu's
# navigation needs 1080p, and this button switches back. Command and label keep their lengths
# (the game's own 18 and 22 characters, padded with spaces).
def native_button(output):
    command, label = f'r.SetRes {output}'.encode(), f'Cinematic Mode ({output.split("x")[1]}p)'.encode()
    if len(command) > 18 or len(label) > 22:
        return None
    return command.ljust(18), label.ljust(22)


def debug_hud(game, output='1920x1080', supersampled=False):
    data = apply_edits(DEBUG_HUD, read_file(game, DEBUG_HUD), DEBUG_HUD_EDITS)
    if output != '1920x1080' and (button := native_button(output)):
        command, label = button
        data = data.replace(b'r.SetRes 3840x2160\0', command + b'\0', 1)
        data = data.replace(b'Cinematic Mode (2160p)\0', label + b'\0')
    elif supersampled:
        # A larger output on top of a supersampled scene runs out of the game's PS4 memory (the
        # crash of "Cinematic Mode" at 4K): both buttons keep 1080p, as the in-game menu's do.
        data = data.replace(b'r.SetRes 3840x2160\0', b'r.SetRes 1920x1080\0', 1)
        data = data.replace(b'Cinematic Mode (2160p)\0', b'Cinematic Mode (1080p)\0')
        data = data.replace(b'\x1fr.SetRes 2560x1440 \0', b'\x1fr.SetRes 1920x1080 \0', 1)
        data = data.replace(b'Render 2560x1440 (1440p)\0', b'Render 1920x1080 (1080p)\0')
    return data


# Unlimited ammo: every weapon uses up ammo (and wear) through the Blueprint function
# RemoveInventoryItemHealth(Item, Health), which adds Subtract_IntInt(0, Health) to the item. Health
# becomes the function's own result variable, still 0 then (UE4 zeroes a script function's locals):
# nothing is taken and the item's real count is still returned (no "out of ammo"). The operand must
# keep its size on disk and in memory (an object reference is 8 bytes loaded): EX_IntConst instead
# broke the loaded bytecode and crashed at start. Weapons stop wearing out, the Store may not
# charge for what uses the same function.
INFINITE_AMMO = 'OVERBITE/Content/Blueprints/GameInstance/OverbiteGameInstanceFunctions.uexp'
INFINITE_AMMO_EDITS = [
    [(bytes.fromhex('6816ffffff1d00000000007d030000161b53'),
      bytes.fromhex('6816ffffff1d00000000007c030000161b53'), 1),
     # EU 01.02: Subtract_IntInt is import -244, Health local 0x393, the result local 0x392.
     (bytes.fromhex('680cffffff1d000000000093030000161b58'),
      bytes.fromhex('680cffffff1d000000000092030000161b58'), 1)],
]


# Voice language (medievil.ini ue4_voice): the game takes its spoken language from its text
# language (GetWWiseLanguageFromLanguageID: LanguageIDToWWiseLanguageMap, else 'English(US)').
# Both versions ship 15 voice languages. With a voice language chosen, that function skips the map
# (its JumpIfNot becomes a Jump to the fallback, the condition left as a dead expression) and the
# fallback becomes the chosen Wwise language, padded with EX_Nothing: the same size on disk and in
# memory, so text and voices can differ (Japanese voices with English text, for example).
GAME_INSTANCE = 'OVERBITE/Content/Blueprints/GameInstance/OverbiteGameInstance.uexp'
VOICE_LANGUAGES = ('English(US)', 'French', 'German', 'Italian', 'Spanish', 'es-MX', 'Portuguese',
                   'pt-BR', 'Dutch', 'Polish', 'Russian', 'Czech', 'Greek', 'Arabic', 'Japanese')
VOICE_FUNCTION_END = b'\x1fEnglish(US)\x00\x04\x0b\x53'
VOICE_FUNCTION_SIZE = 138  # disk bytes of GetWWiseLanguageFromLanguageID in 01.00 and 01.02


def voice_language(game, voice):
    if voice not in VOICE_LANGUAGES:
        raise ValueError(f'unknown voice language {voice!r}')
    data = read_file(game, GAME_INSTANCE)
    if data.count(VOICE_FUNCTION_END) != 1:
        raise ValueError(f'{GAME_INSTANCE}: unexpected contents (game version?)')
    end = data.find(VOICE_FUNCTION_END) + len(VOICE_FUNCTION_END)
    start = end - VOICE_FUNCTION_SIZE
    script = bytearray(data[start:end])
    # The map's miss branch: EX_JumpIfNot 0xb8 (EX_LocalVariable condition) after EndFunctionParms.
    jump = script.find(b'\x16\x07\xb8\x00\x00\x00\x00')
    if jump < 0 or script.count(b'\x16\x07\xb8\x00\x00\x00\x00') != 1:
        raise ValueError(f'{GAME_INSTANCE}: unexpected contents (game version?)')
    script[jump + 1] = 0x06  # EX_Jump: always the fallback
    text = b'English(US)\x00'
    at = script.find(b'\x1f' + text) + 1
    script[at:at + len(text)] = voice.encode() + b'\x00' + b'\x0b' * (len(text) - len(voice) - 1)
    return data[:start] + bytes(script) + data[end:]


# 01.02 (not 01.00): the game instance runs "t.MaxFPS 30" or "t.MaxFPS 60" (ExecuteConsoleCommand),
# which capped the game at 60 FPS. Above 60 FPS both become "t.MaxFPS 00" (0: no limit; same
# length), so the engine range and the port's limiter decide as in 01.00.
MAX_FPS_COMMANDS = (b'\x1ft.MaxFPS 30\x00', b'\x1ft.MaxFPS 60\x00')


def without_max_fps(data):
    for command in MAX_FPS_COMMANDS:
        data = data.replace(command, command[:-3] + b'00\x00')
    return data


def edited_asset(game, name, edits):
    return apply_edits(name, read_file(game, name), edits)


# Anti-aliasing (medievil.ini ue4_aa, issue #27): the PS4 game uses temporal AA, soft in motion.
# r.PostProcessAAQuality 0 is off, 1-2 FXAA, 3+ temporal. Temporal upsampling needs temporal AA,
# so with upscaling active the setting stays temporal.
ANTI_ALIASING = {'taa': None, 'fxaa': '2', 'off': '0'}


def preset_values(quality):
    """The console variables a quality preset writes into the override pak."""
    values = dict(QUALITY.get(quality, {}))
    if quality in LOCAL_SHADOWS:
        size, threshold = LOCAL_SHADOWS[quality]
        values.update({'r.Shadow.MinResolution': str(size), 'r.Shadow.MaxResolution': str(size),
                       'r.Shadow.RadiusThreshold': str(threshold), 'r.Shadow.CacheWholeSceneShadows': '1'})
        cascades, distance = SUN_SHADOWS[quality]
        values.update({'r.Shadow.CSM.MaxCascades': cascades, 'r.Shadow.DistanceScale': distance})
    return values


# The cvars of the in-game menu's live graphics options (gpu/shim/gow3_settings.h UE4LiveOptions),
# and the engine's own defaults of those neither PS4Engine.ini nor BaseScalability.ini sets.
LIVE_CVARS = ('r.Shadow.MinResolution', 'r.Shadow.MaxResolution', 'r.Shadow.DistanceScale',
              'r.Shadow.CSM.MaxCascades', 'r.ShadowQuality', 'r.ViewDistanceScale',
              'r.StaticMeshLODDistanceScale', 'foliage.LODDistanceScale', 'r.AmbientOcclusionLevels',
              'r.SSR.Quality', 'r.VolumetricFog', 'r.MotionBlurQuality', 'r.DepthOfFieldQuality',
              'r.BloomQuality', 'r.LensFlareQuality', 'r.SceneColorFringeQuality')
LIVE_ENGINE_DEFAULTS = {'r.Shadow.MinResolution': '32', 'r.StaticMeshLODDistanceScale': '1',
                        'foliage.LODDistanceScale': '1'}


def ini_sections(text):
    """{section: {lowercase key: value}} of an UE4 ini (the last value of a key wins)."""
    sections, current = {}, None
    for line in text.replace('\r', '').split('\n'):
        line = line.strip()
        if line.startswith('[') and line.endswith(']'):
            current = sections.setdefault(line[1:-1], {})
        elif current is not None and '=' in line and not line.startswith(';'):
            key, value = line.split('=', 1)
            current[key.strip().lstrip('+').lower()] = value.strip()
    return sections


def preset_cvars(game, quality):
    """What the live options' cvars are under `quality` at start: the pak's preset value, else the
    game's PS4Engine.ini [SystemSettings], else its BaseScalability.ini at level 3 (the PS4 starts
    every scalability group at 3), else the engine default. For the menu's "Preset" choice."""
    values = preset_values(quality)
    for item in os.environ.get('GOW3_UE4_CVARS', '').split(';'):
        name, sep, value = item.partition('=')
        if sep and name.strip():
            values[name.strip()] = value.strip()
    system = ini_sections(read_file(game, CONFIG).decode('utf-8-sig', 'replace')).get('SystemSettings', {})
    scalability = {}
    for section, entries in ini_sections(read_file(game, 'Engine/Config/BaseScalability.ini')
                                         .decode('utf-8-sig', 'replace')).items():
        if section.endswith('@3'):
            scalability.update(entries)
    result = {}
    for name in LIVE_CVARS:
        lower = {k.lower(): v for k, v in values.items()}
        value = lower.get(name.lower()) or system.get(name.lower()) or scalability.get(name.lower()) or \
            LIVE_ENGINE_DEFAULTS.get(name)
        if value:
            result[name] = value
    return result


def apply(game, output='1920x1080', scale=100, quality='console', fps=60, debug_menu=False,
          infinite_ammo=False, hzb=False, effects='75', aa='taa', voice='', graphics=None):
    """Writes or removes the override pak. Returns a short description of the active state.
    `graphics`: the graphics settings' console variables (graphics_options.graphics_cvars, run.py), which
    replace the quality preset's; a diagnostic preset (diag_*) still adds its own on top."""
    pak = paks_dir(game) / PAK_NAME
    # Earlier builds wrote the `_P` spelling: on Linux it stayed behind, unread.
    (paks_dir(game) / OLD_PAK_NAME).unlink(missing_ok=True)
    if graphics is not None:
        values = dict(graphics)
        if quality.startswith('diag_'):
            values.update(QUALITY.get(quality, {}))
    else:
        values = preset_values(quality)
    if scale != 100:
        values['r.ScreenPercentage'] = str(scale)
    if scale < 100:
        values['r.TemporalAA.Upsampling'] = '1'
    if output != '1920x1080':
        values['r.setres'] = output
    # Effects resolution (medievil.ini effects_resolution): UE4's separate translucency, where the
    # enemy death effects, smoke and fire draw, at 75 or 50% of the render size, blended in by
    # depth. At 4K several deaths at once cost 2-3 ms of GPU a frame at full size; at 50% none
    # showed (N-19, N-20). 50% at 4K is the PS4's 1080p effect detail. Default 75: the user saw
    # 50% only as slightly less sharp on a 32" 4K OLED up close (N-26).
    if effects in ('75', '50'):
        values['r.SeparateTranslucencyScreenPercentage'] = effects
    # Anti-aliasing (medievil.ini ue4_aa): temporal (the game's), FXAA or off; below 100% render
    # scale the temporal upsampler needs TAA.
    aa_quality = None if scale < 100 else ANTI_ALIASING.get(aa)
    if aa_quality is not None:
        values['r.PostProcessAAQuality'] = aa_quality
    # No memory cvars: this build has no r.GarlicHeapSizeMB, and the PS4 device profile's
    # r.Streaming.PoolSize=2000 outranks [SystemSettings]. Above 1080p the game's graphics memory
    # grows with GOW3_DMEM_MB (graphics_options.py), as it pools all of its direct memory
    # (research/docs/fix-inventory.md, Caveats).
    # The port answers the game's occlusion queries "visible" (shadPS4's synthetic counters), so
    # they cull nothing and their bounding-box draws cost ~150 draws a frame in busy views
    # (+~5% FPS without them, research: T-O1). hzb (the runtime's HZB remap, GOW3_HZB_REMAP=1,
    # analysed eboot only): UE4's HZB occlusion culls, served from finished copies without a GPU
    # wait (T-HZ4); the few queries it still issues are read 3 frames late, so the game never
    # waits for them (T-HZ4e).
    if hzb:
        values.update({'r.AllowOcclusionQueries': '1', 'r.HZBOcclusion': '1',
                       'r.NumBufferedOcclusionQueries': '3'})
    else:
        values['r.AllowOcclusionQueries'] = '0'
    # DBuffer decals force a full depth prepass in UE4 4.21 (the G-buffer drawn twice: ~630-740
    # draws a frame in heavy views, ~28%), whatever r.EarlyZPass says; the decals they serve were
    # invisible in the heavy view (1-4/255 on Dan's boots). Off, and the PS4's own partial prepass
    # (large occluders, r.EarlyZPass=1) keeps early depth for the GPU: heavy views ~600 fewer
    # draws, the best 1% lows so far (research: study-heavy-frame-map.md, T-Z2-T-Z5, 2026-10-21).
    # GOW3_DECAL_PREPASS=1: the PS4's settings (DBuffer decals, their full prepass).
    if os.environ.get('GOW3_DECAL_PREPASS') != '1':
        values.update({'r.DBuffer': '0', 'r.EarlyZPass': '1'})
    # GPU particles: one simulation pass a frame with the real frame time; the default fixed step
    # (1/30 s) added a second pass that, above 30 FPS, only copied the particles back (~190 draws,
    # study-heavy-frame-map.md, T-H3b). GOW3_PARTICLE_FIXED_STEP=1: the fixed step.
    if os.environ.get('GOW3_PARTICLE_FIXED_STEP') != '1':
        values['r.GPUParticle.FixDeltaSeconds'] = '0'
    # Extra console variables for experiments, "name=value;name=value" (GOW3_UE4_CVARS); they win
    # over the presets above.
    extra = {}
    for item in os.environ.get('GOW3_UE4_CVARS', '').split(';'):
        name, sep, value = item.partition('=')
        if sep and name.strip():
            extra[name.strip()] = value.strip()
    values.update(extra)
    engine = frame_rate_range(fps) if fps > 60 else {}
    if not values and not engine and not debug_menu and not infinite_ammo and not voice:
        pak.unlink(missing_ok=True)
        return '1920x1080 output, render scale 100%, console quality, 60 FPS'
    files = {CONFIG: set_values(read_file(game, CONFIG), values, engine)}
    # Native output: the base PS4 renderer makes its buffers at the display size at start and only
    # a live r.SetRes changes them; the runtime sets it (GOW3_UE4_SETRES), and the debug menu's
    # first resolution button follows the setting.
    # A game build whose assets differ (another version) loses only that cheat, not the rest.
    if debug_menu:
        try:
            files[GAME_CONFIG] = debug_menu_config(game, True)
            files[DEBUG_HUD] = debug_hud(game, output, scale > 100)
        except (ValueError, FileNotFoundError) as error:
            print(f'Debug menu: not available for this game build ({error})', flush=True)
            debug_menu = False
    if infinite_ammo:
        try:
            files[INFINITE_AMMO] = edited_asset(game, INFINITE_AMMO, INFINITE_AMMO_EDITS)
        except (ValueError, FileNotFoundError) as error:
            print(f'Unlimited ammo: not available for this game build ({error})', flush=True)
            infinite_ammo = False
    if voice:
        try:
            files[GAME_INSTANCE] = voice_language(game, voice)
        except (ValueError, FileNotFoundError) as error:
            print(f'Voice language: not available for this game build ({error})', flush=True)
            voice = ''
    if fps > 60:
        instance = files.get(GAME_INSTANCE) or read_file(game, GAME_INSTANCE)
        if any(command in instance for command in MAX_FPS_COMMANDS):
            files[GAME_INSTANCE] = without_max_fps(instance)
    write_pak(pak, files)
    return (f'{output} output, render scale {scale}%'
            f'{" (temporal upsampling)" if scale < 100 else " (supersampled)" if scale > 100 else ""}, {quality} quality, '
            f'{f"up to {fps}" if fps > 60 else 60} FPS{", debug menu (R3; 01.02: L1 + R1 + R3)" if debug_menu else ""}'
            f'{f", anti-aliasing {aa}" if aa_quality is not None else ""}'
            f'{", unlimited ammo" if infinite_ammo else ""}{f", voices {voice}" if voice else ""}'
            f'{"; extra cvars " + ", ".join(f"{k}={v}" for k, v in extra.items()) if extra else ""}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('game', type=Path)
    parser.add_argument('--output', default='1920x1080', help="the game's output size, WxH")
    parser.add_argument('--scale', type=int, default=100, help='render scale in percent (25-200)')
    parser.add_argument('--quality', choices=list(QUALITY), default='console')
    parser.add_argument('--fps', type=int, default=60, help='engine frame rate (60 = original)')
    parser.add_argument('--debug-menu', action='store_true', help='allow the developer debug menu (R3 in a level; 01.02: hold L1 + R1, then R3)')
    args = parser.parse_args()
    print('Render resolution:', apply(args.game, args.output, args.scale, args.quality, args.fps,
                                      args.debug_menu))
