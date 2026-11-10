import os
import sys
import tempfile
import unittest
from pathlib import Path

from paths import ROOT
sys.path.insert(0, str(ROOT / 'launcher'))
import medievil_launcher as launcher  # noqa: E402
import graphics_options  # noqa: E402  (scripts/, on the path through the launcher)


class DeckPresetTest(unittest.TestCase):
    """Deck Optimized (plan-linux-deck.md): the user's flat-60 settings and the Deck's opt-in switches."""

    def test_preset_is_the_logging_ini_with_80_percent_fog_and_a_60_cap(self):
        logging = {
            'show_fps': '0', 'display_mode': 'borderless', 'vsync': '0', 'fps_limit': '0', 'async_shaders': '1',
            'deferred_readback': '1', 'stale_readback': '1', 'output_resolution': '1280x720', 'render_scale': '67',
            'engine_fps': '240', 'fsr1': '0', 'rcas': '0', 'rcas_strength': '29', 'frames_queued': '1',
            'ue4_quality': 'low', 'effects_resolution': '50', 'ue4_aa': 'taa', 'perf_hzb_occlusion': '1',
            'perf_no_decal_prepass': '1', 'perf_particle_single_pass': '1', 'perf_small_buffers': '1',
            'perf_texture_memo': '1', 'perf_lockfree_memory': '1', 'perf_cheap_frees': '1',
            'perf_gpu_data_no_waits': '1', 'perf_4k_memory': '0', 'perf_lean_barriers': '1', 'perf_lean_binding': '1',
            'perf_ring_reuse': '1', 'present_inline': '1', 'present_precise_tick': '0', 'present_frame_trace': '0',
            'present_no_frame_caps': '1', 'present_tick_hz': '900', 'present_mode': 'auto',
            'present_limiter': 'standard', 'display_vrr': '0', 'shadow_resolution': '256', 'shadow_distance': 'short',
            'shadow_quality': '1', 'view_distance': '0.7', 'lod_distance': 'low', 'ambient_occlusion': '2',
            'reflections': '1', 'volumetric_fog': '0', 'motion_blur': '0', 'depth_of_field': '0', 'bloom': '2',
            'lens_flares': '0', 'chromatic_aberration': '0', 'preshadows': '0',
            # Settings only Low quality changed before 2026-11-08 (now in the list): Low's values.
            'small_object_shadows': '0.15', 'sun_shadow_resolution': '1024', 'foliage_density': '0.6',
            'texture_filtering': '8', 'refraction': '0'}
        self.assertEqual(launcher.DECK_PRESET,
                         {**logging, 'render_scale': '80', 'volumetric_fog': '1', 'fps_limit': '60'})

    def test_every_value_is_one_the_launcher_and_game_accept(self):
        p = launcher.DECK_PRESET
        for key in p:
            self.assertTrue(key in launcher.INI_DEFAULTS or key in launcher.INI_OPTIONAL_KEYS, key)
        for key in launcher.INI_FLAGS & set(p):
            self.assertIn(p[key], ('0', '1'), key)
        lists = {'display_mode': launcher.DISPLAY_MODES, 'fps_limit': launcher.FPS_LIMITS,
                 'output_resolution': launcher.OUTPUT_CHOICES, 'ue4_quality': launcher.QUALITY_CHOICES,
                 'effects_resolution': launcher.EFFECTS_CHOICES, 'ue4_aa': launcher.AA_CHOICES,
                 'frames_queued': launcher.FRAMES_QUEUED, 'engine_fps': launcher.ENGINE_FPS_CHOICES}
        for key, choices in lists.items():
            self.assertIn(p[key], [v for v, _t in choices], key)
        for key, _title, _hint, choices in launcher.GRAPHICS_OPTIONS:
            self.assertIn(p[key], [v for v, _l in choices], key)
        self.assertIn(p['present_limiter'], graphics_options.PRESENT_LIMITERS)
        self.assertIn(p['present_mode'], graphics_options.PRESENT_MODES)
        low, high = graphics_options.RENDER_SCALE_RANGE
        self.assertTrue(low <= int(p['render_scale']) <= high)
        self.assertEqual(graphics_options.parse_output(p['output_resolution']), '1280x720')
        # The game's own quality stays a real one: "deck" is the launcher's name for the preset.
        self.assertEqual(launcher.QUALITY_PRESETS[-1], (launcher.DECK_QUALITY, ('Deck Optimized',)))
        self.assertNotIn(launcher.DECK_QUALITY, [v for v, _t in launcher.QUALITY_CHOICES])

    def test_the_60_limit_is_the_only_clock(self):
        p = launcher.DECK_PRESET
        env = graphics_options.present_environment(p)
        # No engine bound and no built-in present cap: only fps_limit (driver.cpp current_limit) paces.
        self.assertTrue(graphics_options.no_frame_caps(p))
        self.assertEqual(env.get('GOW3_NO_FRAME_CAP'), '1')
        # The tick is far above the limit (a limit at or above the tick is dropped), not a second clock.
        self.assertEqual(env['GOW3_VBLANK_HZ'], '900')
        self.assertGreater(int(env['GOW3_VBLANK_HZ']), int(p['fps_limit']))
        # Standard limiter: no Reflex (AMD: none) and no precise limiter on top; VSync off (Mailbox).
        for name in ('GOW3_REFLEX', 'GOW3_PRECISE_LIMIT', 'GOW3_LIMIT_HOLD', 'GOW3_PRESENT_MODE', 'GOW3_PRECISE_TICK'):
            self.assertNotIn(name, env)
        self.assertEqual((p['fps_limit'], p['vsync']), ('60', '0'))
        # Only the 4K memory patch is off (above 1080p only); every other performance patch stays on.
        self.assertEqual(graphics_options.perf_environment(p), {'GOW3_PS4_MEMORY': '1'})

    def test_switches_come_with_the_preset_only(self):
        self.assertEqual(launcher.DECK_SWITCHES, {
            'GOW3_EARLY_FRAME_EOP': '1', 'GOW3_DECODE_PACING': '1', 'GOW3_DECODE_PACING_LEAD': '2',
            'GOW3_ARENA_HOST_VISIBLE': '0', 'GOW3_DRAIN_HYBRID': '1', 'GOW3_TEXSET_WIDE': '1',
            'GOW3_REGION_INDEX': '1', 'GOW3_BACKING_COPY': '1', 'GOW3_PENDING_FILTER': '1',
            'GOW3_REG_HASH_PAIRED': '1', 'GOW3_PARTICLE_FLIP_GUARD': '1', 'GOW3_THREAD_NICE': '1'})
        self.assertFalse(launcher.APP_DEFAULTS['deck_switches'])
        added = lambda env: {k: v for k, v in env.items() if k not in os.environ}
        plain = added(launcher.game_environment({**launcher.APP_DEFAULTS}))
        for name in launcher.DECK_SWITCHES:
            self.assertNotIn(name, plain)
        deck = launcher.game_environment({**launcher.APP_DEFAULTS, 'deck_switches': True})
        for name, value in launcher.deck_switch_env({'deck_switches': True})[0].items():
            self.assertEqual(deck[name], os.environ.get(name, value))
        self.assertIn('GOW3_DECK_SWITCHES', deck)
        # A variable already set (a test) wins over the preset.
        saved = os.environ.get('GOW3_ARENA_HOST_VISIBLE')
        os.environ['GOW3_ARENA_HOST_VISIBLE'] = '1'
        try:
            env = launcher.game_environment({**launcher.APP_DEFAULTS, 'deck_switches': True})
            self.assertEqual(env['GOW3_ARENA_HOST_VISIBLE'], '1')
        finally:
            if saved is None:
                del os.environ['GOW3_ARENA_HOST_VISIBLE']
            else:
                os.environ['GOW3_ARENA_HOST_VISIBLE'] = saved


class DeckSwitchItemsTest(unittest.TestCase):
    """The switches one by one: all on Linux, the Linux-only ones never on Windows, unchecked ones left out."""

    def test_linux_sets_all_of_them(self):
        env, summary = launcher.deck_switch_env({'deck_switches': True}, windows=False)
        self.assertEqual(env, launcher.DECK_SWITCHES)
        self.assertTrue(summary.startswith('11 on'))

    def test_windows_leaves_out_what_does_nothing_there(self):
        env, summary = launcher.deck_switch_env({'deck_switches': True}, windows=True)
        for name in ('GOW3_BACKING_COPY', 'GOW3_THREAD_NICE', 'GOW3_ARENA_HOST_VISIBLE'):
            self.assertNotIn(name, env)
        self.assertEqual(len(env), len(launcher.DECK_SWITCHES) - 3)
        self.assertIn('Windows: untested', summary)

    def test_unchecked_ones_are_left_out(self):
        env, summary = launcher.deck_switch_env(
            {'deck_switches': True, 'deck_switches_off': 'decode_pacing,thread_nice'}, windows=False)
        for name in ('GOW3_DECODE_PACING', 'GOW3_DECODE_PACING_LEAD', 'GOW3_THREAD_NICE'):
            self.assertNotIn(name, env)
        self.assertIn('decode_pacing (unchecked)', summary)

    def test_off_sets_nothing(self):
        self.assertEqual(launcher.deck_switch_env({'deck_switches': False}, windows=False), ({}, ''))

    def test_items_cover_the_preset_switches(self):
        keys = [key for key, *_rest in launcher.DECK_SWITCH_ITEMS]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertEqual(len(launcher.DECK_SWITCHES), 12)


class SteamDeckDetectionTest(unittest.TestCase):
    def dmi(self, **files):
        folder = Path(self.enterContext(tempfile.TemporaryDirectory()))
        for name, text in files.items():
            (folder / name).write_text(text + '\n', encoding='utf-8')
        return folder

    def test_valve_boards(self):
        for product in ('Jupiter', 'Galileo'):
            self.assertTrue(launcher.steam_deck({}, self.dmi(board_vendor='Valve', product_name=product)))
        # Older kernels name the board only (board_name), the vendor in sys_vendor.
        self.assertTrue(launcher.steam_deck({}, self.dmi(sys_vendor='Valve', board_name='Galileo')))

    def test_other_machines(self):
        self.assertFalse(launcher.steam_deck({}, self.dmi(board_vendor='Valve', product_name='Fremont')))
        self.assertFalse(launcher.steam_deck({}, self.dmi(board_vendor='LENOVO', product_name='Jupiter')))
        self.assertFalse(launcher.steam_deck({}, self.dmi(board_vendor='ASUSTeK COMPUTER INC.',
                                                           product_name='ROG Ally RC71L')))
        self.assertFalse(launcher.steam_deck({}, Path(self.enterContext(tempfile.TemporaryDirectory())) / 'none'))
        self.assertFalse(launcher.steam_deck({'SteamDeck': '0'}, self.dmi()))

    def test_steam_environment(self):
        # Steam sets SteamDeck=1 for what it starts on a Deck (also under Windows, where there is no DMI here).
        self.assertTrue(launcher.steam_deck({'SteamDeck': '1'}, self.dmi()))

    def test_offer_once_and_never_silently_over_existing_settings(self):
        offer = launcher.deck_preset_offer
        fresh = {**launcher.APP_DEFAULTS}
        self.assertEqual(offer(fresh, first_run=True, on_deck=True), 'select')
        self.assertEqual(offer(fresh, first_run=False, on_deck=True), 'ask')
        self.assertIsNone(offer(fresh, first_run=True, on_deck=False))
        self.assertIsNone(offer({**fresh, 'deck_offered': True}, first_run=False, on_deck=True))
        self.assertIsNone(offer({**fresh, 'deck_offered': True}, first_run=True, on_deck=True))
        self.assertIsNone(offer({**fresh, 'graphics_preset': 'deck'}, first_run=False, on_deck=True))


if __name__ == '__main__':
    unittest.main()
