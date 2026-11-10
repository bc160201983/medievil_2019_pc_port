"""GPU particle single pass, fixed step kept (research/docs/study-particle-fixed-step.md): the
"GPU particle fixed step in the single pass" patch, its run.py switch, and the flip guard's hook
sites and stubs (src/particle_hook.h, src/runtime_extra.c, gpu/shim/gow3_toggles.h)."""
import re
import unittest

from paths import ROOT
from patches import compile_patches, selected_patches
import graphics_options as g

XML = ROOT / 'patches/MediEvil.xml'
NAME = 'GPU particle fixed step in the single pass'
SEGMENTS = [(0, 0x6000000)]
HOOK = (ROOT / 'src' / 'particle_hook.h').read_text(encoding='utf-8')
EXTRA = (ROOT / 'src' / 'runtime_extra.c').read_text(encoding='utf-8')
TOGGLES = (ROOT / 'gpu' / 'shim' / 'gow3_toggles.h').read_text(encoding='utf-8')


def stub_pattern(name):
    """The bytes of a stub's pattern array in particle_hook.h and its declared length."""
    body = HOOK[HOOK.index(f'void {name}('):]
    array = re.sub(r'/\*.*?\*/', '', body[body.index('pattern['):body.index('};')])
    data = bytes(int(v, 0) for v in re.findall(r'0x[0-9a-fA-F]+|\b0\b', array[array.index('{'):]))
    return data, body


class ParticleFixedStepTest(unittest.TestCase):
    def test_patch_writes(self):
        # Game thread: the fixed step from the registered 1/30 (lea of the constant, one slot);
        # render thread: the single pass is the fixed-step pass in the three collision variants.
        self.assertEqual(compile_patches(XML, [NAME], '01.00', SEGMENTS), [
            (0x2383a36, bytes.fromhex('488d1ddf0baa01')), (0x2383a5c, bytes.fromhex('c5fa102b90')),
            (0x23684da, bytes.fromhex('b00190')), (0x236851f, bytes.fromhex('b00190')),
            (0x236855b, bytes.fromhex('b00190'))])
        writes = compile_patches(XML, [NAME], '01.02', SEGMENTS)
        self.assertEqual([a for a, _ in writes], [0x2386806, 0x238682c, 0x236b2aa, 0x236b2ef, 0x236b32b])
        # lea rbx, [rip + d]: 01.00 -> 0x3e2461c, 01.02 -> 0x3e276dc (1/30 in each constant pool).
        for (at, code), target in ((compile_patches(XML, [NAME], '01.00', SEGMENTS)[0], 0x3e2461c),
                                   (writes[0], 0x3e276dc)):
            self.assertEqual(at + 7 + int.from_bytes(code[3:], 'little', signed=True), target)
        self.assertNotIn(NAME, selected_patches(XML, '01.00', ''))
        self.assertNotIn(NAME, selected_patches(XML, '01.02', ''))

    def test_environment(self):
        env = g.particle_single_fixed_environment({'GOW3_PATCHES': 'A', 'GOW3_PARTICLE_SINGLE_FIXED': '1'})
        self.assertEqual(env['GOW3_PATCHES'], f'A;{NAME}')
        env = g.particle_single_fixed_environment({'GOW3_PATCHES': f'A;{NAME}', 'GOW3_PARTICLE_SINGLE_FIXED': '0'})
        self.assertEqual(env['GOW3_PATCHES'], 'A')
        self.assertEqual(g.particle_single_fixed_environment({'GOW3_PATCHES': NAME}), {'GOW3_PATCHES': NAME})

    def test_hook_sites(self):
        rows = re.findall(r'\{(0x[0-9a-f]+), (0x[0-9a-f]+), (0x[0-9a-f]+), (0x[0-9a-f]+)\}, /\* (01\.0\d) \*/', HOOK)
        sites = {v: [int(x, 16) for x in r] for *r, v in rows}
        self.assertEqual(sites['01.00'], [0x230de02, 0x230df0a, 0x23667e0, 0x236a050])
        self.assertEqual(sites['01.02'], [a + 0x2dd0 for a in sites['01.00']])

    def test_stubs(self):
        for name, length_name, immediates in (
                ('medievil_particle_build_advance_stub', 'MEDIEVIL_PARTICLE_ADVANCE_STUB_LENGTH', (14, 42)),
                ('medievil_particle_build_pre_render_stub', 'MEDIEVIL_PARTICLE_PRE_RENDER_STUB_LENGTH', (14, 35, 59))):
            data, body = stub_pattern(name)
            length = int(re.search(length_name + r' = (\d+)', HOOK).group(1))
            self.assertEqual(len(data), length, name)
            found = tuple(int(o) for o in re.findall(r'memcpy\(code \+ (\d+), &', body[:body.index('#endif')
                                                                                          if '#endif' in body else None])[:len(immediates)])
            self.assertEqual(found, immediates, name)
            for at in immediates:
                self.assertEqual(data[at - 2:at], b'\x48\xb8', name)  # movabs rax, imm64
                self.assertEqual(data[at:at + 8], bytes(8), name)
        advance, _ = stub_pattern('medievil_particle_build_advance_stub')
        self.assertEqual(advance[38:40], b'\x74\x0c')        # jz to the ret
        self.assertEqual(advance[40 + 12], 0xc3)
        pre_render, _ = stub_pattern('medievil_particle_build_pre_render_stub')
        self.assertEqual(pre_render[26:28], b'\x74\x11')     # jz past the AdvanceGPUParticleFrame call
        self.assertEqual(pre_render[28 + 17:28 + 21], bytes.fromhex('4883c408'))
        self.assertEqual(pre_render[28:33], bytes.fromhex('488b7c2430'))  # the saved rdi (6 pushes, 8)

    def test_toggle_bit(self):
        second = TOGGLES[TOGGLES.index('namespace Gow3Toggle2'):]
        second = second[:second.index('};')]
        bit = int(re.search(r'ParticleFlipGuardOff = (\d+),', second).group(1))
        self.assertEqual(bit, 16384)
        self.assertEqual(bit, int(re.search(r'#define TOGGLE2_PARTICLE_FLIP_GUARD_OFF UINT64_C\((\d+)\)', EXTRA).group(1)))


if __name__ == '__main__':
    unittest.main()
