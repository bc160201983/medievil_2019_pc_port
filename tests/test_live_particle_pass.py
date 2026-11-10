"""The live single GPU particle pass (src/runtime_extra.c ue4_particle_pass_poll, gow3_toggles.h)."""
import re
import struct
import unittest

from paths import ROOT

EXTRA = (ROOT / 'src' / 'runtime_extra.c').read_text(encoding='utf-8')
TOGGLES = (ROOT / 'gpu' / 'shim' / 'gow3_toggles.h').read_text(encoding='utf-8')


def cvar_row(name):
    """{object, site, bytes} per build of a ue4_cvars[] row."""
    line = next(l for l in EXTRA.splitlines() if l.strip().startswith('{"%s",' % name))
    rows = re.findall(r'\{(0x[0-9a-f]+),(0x[0-9a-f]+),\{([^}]*)\}\}', line)
    return [(int(o, 16), int(s, 16), bytes(int(b, 16) for b in code.split(','))) for o, s, code in rows]


class LiveParticlePassTest(unittest.TestCase):
    def test_cvar_rows_load_the_name(self):
        # Each site is `lea rsi, [rip+X]` with X the UTF-16 name, found in each eboot
        # (research/docs/study-live-switches.md: 01.00 name at 0x3e24a7a, 01.02 at 0x3e27b3a).
        rows = cvar_row('r.GPUParticle.FixDeltaSeconds')
        self.assertEqual(len(rows), 2)
        for (obj, site, code), name_at, expected_obj in zip(rows, (0x3e24a7a, 0x3e27b3a), (0x5c14d18, 0x5c14d58)):
            self.assertEqual(code[:3], b'\x48\x8d\x35')
            self.assertEqual(site + 7 + struct.unpack('<i', code[3:])[0], name_at)
            self.assertEqual(obj, expected_obj)

    def test_fixed_step_is_the_registered_float(self):
        value = re.search(r'#define PARTICLE_FIXED_STEP "([^"]+)"', EXTRA).group(1)
        self.assertEqual(struct.pack('<f', float(value)), struct.pack('<I', 0x3d088889))
        # The value buffer (ue4_cvar_values) holds it with its terminator.
        size = int(re.search(r'ue4_cvar_values\[UE4_CVARS\]\[(\d+)\]', EXTRA).group(1))
        self.assertLess(len(value), size)

    def test_toggle_bit_matches_and_is_unique(self):
        second = TOGGLES[TOGGLES.index('namespace Gow3Toggle2'):]
        second = second[:second.index('};')]
        values = [int(v) for v in re.findall(r'^\s*\w+ = (\d+),', second, re.M)]
        self.assertEqual(len(values), len(set(values)))
        self.assertTrue(all(v & (v - 1) == 0 for v in values))
        bit = int(re.search(r'ParticleFixedStep = (\d+),', second).group(1))
        self.assertEqual(bit, int(re.search(r'#define TOGGLE2_PARTICLE_FIXED_STEP UINT64_C\((\d+)\)', EXTRA).group(1)))
        self.assertNotIn(64, values)  # reserved (particle-sim skip, impl-draw-reduction.md)

    def test_live_in_the_menu(self):
        header = (ROOT / 'gpu' / 'shim' / 'gow3_settings.h').read_text(encoding='utf-8')
        keys = re.search(r'PerfKeys\[PerfCount\] = \{([^}]*)\}', header).group(1)
        keys = re.findall(r'"(\w+)"', keys)
        live = re.search(r'PerfLive\[PerfCount\] = \{([^}]*)\}', header).group(1)
        live = [v.strip() == 'true' for v in live.split(',')]
        self.assertTrue(dict(zip(keys, live))['perf_particle_single_pass'])


if __name__ == '__main__':
    unittest.main()
