"""Compile selected shadPS4/GoldHEN XML patches into out/patches.bin for the loader.

Patch addresses are PS4 virtual addresses (eboot base 0x400000); the loader's
image places eboot vaddr 0 at image offset 0. Only literal writes are supported
(bytes, bytes16/32/64, float32/64, utf8, utf16); pattern ("mask") patches are rejected.
"""
import argparse
import json
import os
import re
import struct
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

EBOOT_BASE=0x400000


def game_app_version(game):
    """APP_VER of the game folder's param.sfo ("01.02"), or None when it cannot be read."""
    try:
        from prepare import sfo
        return sfo((Path(game) / 'sce_sys/param.sfo').read_bytes()).get('APP_VER')
    except (OSError, ValueError, ImportError):
        return None


def eboot_segments(elf):
    phoff,=struct.unpack_from('<Q',elf,0x20)
    phentsize,phnum=struct.unpack_from('<HH',elf,0x36)
    segments=[]
    for i in range(phnum):
        kind,_,_,vaddr,_,_,memsz,_=struct.unpack_from('<IIQQQQQQ',elf,phoff+i*phentsize)
        if kind==1: segments.append((vaddr,vaddr+memsz))
    return segments


def eboot_bytes(elf, offset, length):
    """length bytes of the eboot file at image offset `offset` (vaddr - EBOOT_BASE), None when they
    are not all in one loaded segment's file data."""
    phoff,=struct.unpack_from('<Q',elf,0x20)
    phentsize,phnum=struct.unpack_from('<HH',elf,0x36)
    for i in range(phnum):
        kind,_,fileoff,vaddr,_,filesz,_,_=struct.unpack_from('<IIQQQQQQ',elf,phoff+i*phentsize)
        if kind==1 and vaddr<=offset and offset+length<=vaddr+filesz:
            return bytes(elf[fileoff+offset-vaddr:fileoff+offset-vaddr+length])
    return None


def original_mismatches(xml, names, app_version, elf):
    """The selected patches with a line whose Original attribute (the bytes the line replaces,
    optional, this port's addition to the format) differs from the eboot's bytes there."""
    bad=[]
    for meta in ET.parse(xml).getroot().iter('Metadata'):
        if meta.get('AppVer')!=app_version or meta.get('Name') not in names: continue
        for line in meta.iter('Line'):
            if line.get('Original') is None: continue
            original=bytes.fromhex(line.get('Original').replace(' ',''))
            if eboot_bytes(elf,int(line.get('Address'),0)-EBOOT_BASE,len(original))!=original:
                bad.append(meta.get('Name'))
                break
    return bad


def encode(line):
    kind,value=line.get('Type'),line.get('Value')
    if kind=='bytes': return bytes.fromhex(value.replace(' ',''))
    if kind in ('bytes16','bytes32','bytes64'):
        return int(value,0).to_bytes(int(kind[5:])//8,'little')
    if kind=='float32': return struct.pack('<f',float(value))
    if kind=='float64': return struct.pack('<d',float(value))
    if kind=='utf8': return value.encode()+b'\0'
    if kind=='utf16': return value.encode('utf-16-le')+b'\0\0'
    raise ValueError(f'unsupported patch type {kind!r}')


# A sized patch, "<name> WxH": the entry <name> with Type="size" lines, each written as W and H
# (two little-endian int32), e.g. "Boot display size 2560x1440" (MediEvil.xml).
SIZED=re.compile(r'(.+) (\d+)x(\d+)$')


def sized_patch(name):
    """(entry name, width, height) of a sized patch name, else (name, None, None)."""
    m=SIZED.match(name)
    return (m.group(1),int(m.group(2)),int(m.group(3))) if m else (name,None,None)


def compile_patches(xml, names, app_version, segments):
    found={}
    for meta in ET.parse(xml).getroot().iter('Metadata'):
        if meta.get('AppVer')==app_version and meta.get('AppElf','eboot.bin')=='eboot.bin':
            found[meta.get('Name')]=meta
    def entry(name):
        if name in found: return found[name],None
        base,w,h=sized_patch(name)
        meta=found.get(base) if w else None
        return (meta,(w,h)) if meta is not None and any(l.get('Type')=='size' for l in meta.iter('Line')) else (None,None)
    missing=[n for n in names if entry(n)[0] is None]
    if missing: raise ValueError(f'patches not found for app version {app_version}: {missing}')
    writes=[]
    for name in names:
        meta,size=entry(name)
        for line in meta.iter('Line'):
            offset=int(line.get('Address'),0)-EBOOT_BASE
            if line.get('Type')=='size':
                if not size: raise ValueError(f'{name}: needs a size ("{name} WxH")')
                data=struct.pack('<ii',*size)
            else:
                data=encode(line)
            if not any(start<=offset and offset+len(data)<=end for start,end in segments):
                raise ValueError(f'{name}: address {line.get("Address")} is outside the eboot')
            writes.append((offset,data))
    return writes


PATCHES_DIR=Path(__file__).resolve().parent.parent/'patches'
# Built-in patch file per game version, the app version its addresses are for and, when known, the
# sha256 of the eboot.bin they were checked against (another build of the same version has its code
# elsewhere: byte writes there corrupt it).
# MediEvil, America and Europe: only the analysed eboots (sha256) get these patches.
GAMES=[({'CUSA11227','CUSA12982'},PATCHES_DIR/'MediEvil.xml','01.00',
        'a7e976d13b9649c5cd1ccb617b02f4a27b95c45b9f5331470b90a1f014c3b828'),
       # 01.02 exists as at least three eboot builds with the patched code at the same addresses
       # (e50c9b36: the dump of PR #26; cd273b59: the European CUSA12982 update, 29 Oct 2026;
       # fe2f88ba: unpacked from the US update PKG UP9000-CUSA11227_00-MEDIEVILHD000001-A0102-V0100,
       # every patched and hooked site checked, research/docs/plan-0102.md).
       ({'CUSA11227','CUSA12982'},PATCHES_DIR/'MediEvil.xml','01.02',
        ('e50c9b36153416949d377ad0ddafac1a888c119987095615dc0f69b5ad15e401',
         'cd273b59a2220ecec970434edf022ff03aa298851dae28b0460814a996ff5634',
         'fe2f88baaa97f19c34047b8ec0c9bd41e29cc818f6b72e9e21fafe66a8d01b00'))]
DMEM_RETAIL_MB=5056  # runtime_memory.c POOL_SIZE default


def game_title_id(game):
    try:
        from prepare import sfo
        return sfo((Path(game) / 'sce_sys/param.sfo').read_bytes()).get('TITLE_ID')
    except (OSError, ValueError, ImportError):
        return None


def game_profile(title_id,app_version=None):
    """(title IDs, built-in XML, app version, eboot sha256 (one, or a tuple of builds) or None)
    for a known game version,
    else None; without app_version, the first entry for the game."""
    return next((g for g in GAMES if title_id in g[0] and (app_version is None or g[2]==app_version)),None)


def eboot_matches(game, profile):
    """Whether the game's eboot.bin is the build the profile's patches were checked against."""
    if not profile or not profile[3]:
        return True
    import hashlib
    try:
        known=profile[3] if isinstance(profile[3],tuple) else (profile[3],)
        return hashlib.sha256((Path(game)/'eboot.bin').read_bytes()).hexdigest() in known
    except OSError:
        return False


# God of War III notes: the texture fix and the resolution patches both resize the video arena;
# enable one, never both. A chosen resolution patch replaces the default texture fix.
EXCLUSIVE_PREFIX='Resolution Patch'
REPLACED_BY_EXCLUSIVE={'Bug Fix - Texture Corruption Fix'}


def selected_patches(xml, app_version, extra, only=False):
    """The file's isEnabled patches for this version, then the names in extra (";"-separated).
    only: extra is the whole selection (the launcher's), without the file's defaults."""
    names=[] if only else [m.get('Name') for m in ET.parse(xml).getroot().iter('Metadata')
                           if m.get('AppVer')==app_version and m.get('isEnabled','false').lower()=='true']
    names+=[n for n in (n.strip() for n in extra.split(';')) if n and n not in names]
    exclusive=[n for n in names if n.startswith(EXCLUSIVE_PREFIX)]
    if len(exclusive)>1:
        raise ValueError(f'choose one resolution patch, not {exclusive}')
    return [n for n in names if not (exclusive and n in REPLACED_BY_EXCLUSIVE)]


def patch_requirements(xml, names, app_version):
    """(direct memory MiB, VBlank Hz) the selected patches' notes ask for, None when not needed.
    shadPS4 notes say "Requires setting DMEM to N MB": N is on top of the retail pool."""
    dmem=vblank=None
    for meta in ET.parse(xml).getroot().iter('Metadata'):
        if meta.get('Name') not in names or meta.get('AppVer')!=app_version: continue
        note=meta.get('Note','')
        if m:=re.search(r'DMEM to (\d+) MB',note): dmem=max(dmem or 0,DMEM_RETAIL_MB+int(m.group(1)))
        if m:=re.search(r'VBlank to (\d+) FPS',note): vblank=max(vblank or 0,int(m.group(1)))
    return dmem,vblank


def external_patches(directory, app_version, exclude, ids):
    """[(key, file, metadata)] of eboot patches for this version in directory/*.xml (third-party
    shadPS4/GoldHEN files in the data directory's patches/ folder).
    key is "<file name>/<patch name>" (the launcher's selection, patches.json)."""
    found=[]
    directory=Path(directory)
    for path in sorted(directory.glob('*.xml')) if directory.is_dir() else []:
        if path.resolve()==Path(exclude).resolve(): continue  # the built-in file (patches/ in a checkout)
        try:
            root=ET.parse(path).getroot()
        except ET.ParseError as error:
            print(f'Patches: {path.name}: {error}',file=sys.stderr)
            continue
        file_ids={e.text.strip() for e in root.iter('ID') if e.text}
        if file_ids and not file_ids&ids: continue
        for meta in root.iter('Metadata'):
            if meta.get('AppVer')==app_version and meta.get('AppElf','eboot.bin')=='eboot.bin':
                found.append((f'{path.name}/{meta.get("Name")}',path,meta))
    return found


def external_selection(found, config):
    """Selected external patches: patches.json {"enabled": [...], "disabled": [...]} overrides
    each file's isEnabled."""
    settings={}
    if config and Path(config).is_file():
        settings=json.loads(Path(config).read_text())
    enabled,disabled=set(settings.get('enabled',[])),set(settings.get('disabled',[]))
    return [(key,path,meta) for key,path,meta in found
            if key in enabled or (key not in disabled and meta.get('isEnabled','false').lower()=='true')]


def compile_external(selected, segments):
    """Writes of the selected external patches; a patch with unsupported lines is skipped whole."""
    writes=[]
    for key,_,meta in selected:
        try:
            ours=[]
            for line in meta.iter('Line'):
                offset=int(line.get('Address') or '',0)-EBOOT_BASE
                data=encode(line)
                if not any(start<=offset and offset+len(data)<=end for start,end in segments):
                    raise ValueError(f'address {line.get("Address")} is outside the eboot')
                ours.append((offset,data))
        except ValueError as error:
            print(f'Patches: skipped {key}: {error}',file=sys.stderr)
            continue
        writes+=ours
        print(f'Patches: external {key} ({len(ours)} writes)')
    return writes


def write_patches(out, writes):
    # G3PATCH2: the patch base, so the loader can rebase pointers the patches write into
    # relocated slots.
    blob=struct.pack('<8sQQ',b'G3PATCH2',EBOOT_BASE,len(writes))
    for offset,data in writes: blob+=struct.pack('<QQ',offset,len(data))+data
    (out/'patches.bin').write_bytes(blob)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--patches-dir',type=Path,help='third-party patch XML files (shadPS4 format)')
    p.add_argument('--patches-config',type=Path,help='patches.json: enabled/disabled external patches')
    p.add_argument('--extra',default='',help='additional patch names, separated by ";"')
    p.add_argument('--out',type=Path,default=Path(__file__).resolve().parent.parent/'out')
    p.add_argument('--game-dir',type=Path,default=Path(os.environ.get('GOW3_GAME_DIR','../CUSA11227')))
    a=p.parse_args()
    title=game_title_id(a.game_dir)
    version=game_app_version(a.game_dir)
    profile=game_profile(title,version)
    if not profile and os.environ.get('GOW3_FORCE_PATCHES'):
        profile=game_profile(title)  # force: apply the first version's addresses anyway
    if not profile:
        write_patches(a.out,[])
        known=game_profile(title)
        if known:
            print(f'Patches: game version {version}, {known[1].name} is for {known[2]}: none applied')
        else:
            print(f'Patches: no patch profile for {title}: none applied')
        return
    ids,xml,needed,_sha=profile
    # The patches are byte writes at the addresses of one game build: on another they would
    # corrupt code, so such a game runs unpatched.
    if not eboot_matches(a.game_dir,profile) and not os.environ.get('GOW3_FORCE_PATCHES'):
        write_patches(a.out,[])
        print(f'Patches: eboot.bin is not the build {xml.name} was checked against: none applied')
        return
    names=selected_patches(xml,needed,a.extra,os.environ.get('GOW3_PATCHES_ONLY')=='1')
    elf=(a.out/'eboot.elf').read_bytes()
    segments=eboot_segments(elf)
    for name in original_mismatches(xml,names,needed,elf):
        print(f'Patches: {name}: the eboot has other bytes at its addresses: not applied')
        names.remove(name)
    writes=compile_patches(xml,names,needed,segments)
    if a.patches_dir:
        # After the built-in ones: an external patch of the same bytes wins.
        writes+=compile_external(external_selection(external_patches(a.patches_dir,needed,xml,ids),
                                                    a.patches_config),segments)
    write_patches(a.out,writes)
    print(f'Patches: {title}; {len(writes)} writes from {names or "none"}')


if __name__=='__main__':
    main()
