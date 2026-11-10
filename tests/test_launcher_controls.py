import sys
import unittest

from paths import ROOT
sys.path.insert(0, str(ROOT / 'launcher'))
import medievil_launcher as launcher  # noqa: E402


class ControlMapTest(unittest.TestCase):
    def test_new_cheats_default_to_disabled(self):
        self.assertEqual(launcher.INI_DEFAULTS['cheat_double_jumps'], '0')
        self.assertEqual(launcher.INI_DEFAULTS['cheat_very_hard'], '0')
        self.assertIn('cheat_double_jumps', launcher.INI_FLAGS)
        self.assertIn('cheat_very_hard', launcher.INI_FLAGS)

    def test_pipeline_binary_cache_switch(self):
        settings = {**launcher.APP_DEFAULTS}
        self.assertTrue(settings['pipeline_binary_cache'])
        self.assertEqual(launcher.game_environment(settings)['GOW3_PIPELINE_BINARIES'], '1')
        settings['pipeline_binary_cache'] = False
        self.assertEqual(launcher.game_environment(settings)['GOW3_PIPELINE_BINARIES'], '0')

    def test_performance_patches_are_ini_flags_with_switches(self):
        import graphics_options
        for key, _title, _hint, off in launcher.PERF_PATCHES:
            self.assertIn(key, launcher.INI_FLAGS)
            self.assertEqual(launcher.INI_DEFAULTS[key], '0' if key in graphics_options.PERF_DEFAULT_OFF else '1')
            self.assertEqual(graphics_options.PERF_SWITCHES[key], off)
        # The single particle pass is off without a key (plan-0102.md, T-0102-4).
        self.assertEqual(graphics_options.perf_environment({}), {'GOW3_PARTICLE_FIXED_STEP': '1'})
        self.assertEqual(graphics_options.perf_environment({'perf_particle_single_pass': '1'}), {})
        self.assertEqual(graphics_options.perf_environment({'perf_texture_memo': '0', 'perf_particle_single_pass': '1'}),
                         {'GOW3_TEXSET_REGIONS': '0', 'GOW3_TEXSET_TWO_WAY': '0'})
        env = launcher.game_environment({**launcher.APP_DEFAULTS})
        self.assertEqual(env.get('GOW3_FRAME_STATS'), launcher.os.environ.get('GOW3_FRAME_STATS'))

    def test_graphics_options_match_the_game(self):
        # gpu/shim/gow3_settings.h UE4LiveOptions is scripts/graphics_options.py GRAPHICS_SETTINGS: every field
        # (keys, titles, hints, live, cvars, labels, tokens, values, each quality level's choice).
        import re
        import graphics_options as g
        header = (ROOT / 'gpu' / 'shim' / 'gow3_settings.h').read_text(encoding='utf-8')
        start = header.index('UE4LiveOptions[] = {') + len('UE4LiveOptions[] = {')
        table = header[start:header.index('inline constexpr int UE4LiveCount')]

        def groups(text):  # the top-level {...} of text, without their braces
            out, depth, begin, quoted = [], 0, 0, False
            for i, ch in enumerate(text):
                if ch == '"' and text[i - 1] != '\\':
                    quoted = not quoted
                if quoted:
                    continue
                if ch == '{':
                    begin, depth = (i + 1 if depth == 0 else begin), depth + 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        out.append(text[begin:i])
            return out

        def strings(text):  # adjacent C++ literals are one string
            return [''.join(re.findall(r'"((?:[^"\\]|\\.)*)"', part)) for part in re.split(r'",\s*\n?\s*(?=")', text)]
        entries = groups(table)
        self.assertEqual(len(entries), len(g.GRAPHICS_SETTINGS))
        for entry, (key, title, hint, live, cvars, choices, levels) in zip(entries, g.GRAPHICS_SETTINGS):
            head = entry[:entry.index('{')]
            parts = re.findall(r'"((?:[^"\\]|\\.)*)"', head)
            self.assertEqual((parts[0], parts[1]), (key, title))
            self.assertEqual(''.join(parts[2:]), hint, key)
            self.assertIn(('true' if live else 'false') + ',', head.split('"')[-1], key)
            c_cvars, c_labels, c_tokens, c_values, c_levels = groups(entry)
            self.assertEqual([c for c in re.findall(r'"([^"]*)"', c_cvars)], list(cvars), key)
            self.assertEqual(re.findall(r'"([^"]*)"', c_labels), [label for _t, label, _v in choices], key)
            self.assertEqual(re.findall(r'"([^"]*)"', c_tokens), [t for t, _l, _v in choices], key)
            self.assertEqual([re.findall(r'"([^"]*)"', v) for v in groups(c_values)], [list(v) for _t, _l, v in choices], key)
            self.assertEqual(re.findall(r'"([^"]*)"', c_levels),
                             [levels[level] for level in (*g.QUALITY_LEVELS, g.DECK_LEVEL)], key)
            self.assertEqual(launcher.INI_DEFAULTS[key], levels['console'])  # a fresh ini is the PS4's
            for level, token in levels.items():
                self.assertIn(token, [t for t, _l, _v in choices], (key, level))
        # The level order of the tables' columns: the launcher's list and the game's UE4QualityKeys.
        self.assertEqual([v for v, _t in launcher.QUALITY_CHOICES], list(g.QUALITY_LEVELS))
        keys = header[header.index('UE4QualityKeys[] = {'):]
        self.assertEqual(re.findall(r'"([^"]*)"', keys[:keys.index('}')]), list(g.QUALITY_LEVELS))

    def test_quality_levels_and_migration(self):
        import graphics_options as g
        for level in (*g.QUALITY_LEVELS, g.DECK_LEVEL):
            self.assertEqual(g.matched_level(g.level_choices(level), level), level)
        # Every level differs from every other (the quality is told by the settings alone).
        columns = [tuple(g.level_choices(level).values()) for level in (*g.QUALITY_LEVELS, g.DECK_LEVEL)]
        self.assertEqual(len(set(columns)), len(columns))
        # An old ini: "preset" follows ue4_quality, old tokens are translated, the new keys come from the level.
        old = {'ue4_quality': 'high', 'shadow_resolution': 'preset', 'shadow_distance': 'far', 'lod_distance': 'console'}
        choices = g.graphics_choices(old)
        self.assertEqual(choices['shadow_resolution'], 'dynamic')
        self.assertEqual(choices['shadow_distance'], 'ps4')
        self.assertEqual(choices['lod_distance'], 'ps4')
        self.assertEqual(choices['texture_filtering'], '16')
        self.assertEqual(g.matched_level(choices, 'high'), '')  # LOD Normal at High: Custom
        self.assertEqual(g.graphics_migration(g.level_choices('console')), {})
        cvars = g.graphics_cvars(g.level_choices('console'))
        self.assertEqual((cvars['r.Shadow.MinResolution'], cvars['r.Shadow.MaxResolution']), ('32', '2048'))
        self.assertEqual(cvars['r.Shadow.CacheWholeSceneShadows'], '1')

    def test_loaded_ini_takes_missing_settings_from_its_level(self):
        # The launcher's own load (load_ini, then the migration as Launcher.__init__ does): a Low ini written
        # before the five new keys existed gets Low's, not the Console defaults (Deck, 2026-11-08: small object
        # shadows 0.01 and refraction 2 on a Low ini).
        import tempfile
        from pathlib import Path
        import graphics_options as g
        folder = Path(self.enterContext(tempfile.TemporaryDirectory()))
        path = folder / 'medievil.ini'
        path.write_text('ue4_quality=low\nshadow_resolution=256\n', encoding='utf-8')
        saved = launcher.ini_path
        launcher.ini_path = lambda: path
        try:
            ini, _lines = launcher.load_ini()
        finally:
            launcher.ini_path = saved
        ini.update(g.graphics_migration(ini))
        low = g.level_choices('low')
        for key in ('small_object_shadows', 'sun_shadow_resolution', 'foliage_density', 'texture_filtering',
                    'refraction'):
            self.assertEqual(ini[key], low[key], key)
        self.assertEqual(ini['shadow_resolution'], '256')

    def test_render_scale_range_matches_the_game(self):
        # gpu/shim/gow3_settings.h RenderScaleMin / RenderScaleMax: the launcher's box and the in-game
        # slider take the same whole percents; every preset is one of them.
        import re
        import graphics_options
        header = (ROOT / 'gpu' / 'shim' / 'gow3_settings.h').read_text(encoding='utf-8')
        low, high, ini_low = map(int, re.search(
            r'RenderScaleMin = (\d+), RenderScaleMax = (\d+), RenderScaleIniMin = (\d+);', header).groups())
        self.assertEqual(graphics_options.RENDER_SCALE_RANGE, (low, high))
        self.assertEqual((graphics_options.parse_render_scale(ini_low), graphics_options.parse_render_scale(ini_low - 1)),
                         (ini_low, 100))
        self.assertEqual({int(v) for v, _t in launcher.RENDER_SCALE_CHOICES}, set(graphics_options.RENDER_SCALES))
        self.assertEqual(launcher.render_scale_scene('1280x720', '72'), 'scene 921x518')
        self.assertEqual(launcher.render_scale_scene('2560x1440', 'x'), 'scene 2560x1440')
        self.assertEqual(launcher.Launcher.custom_label('render_scale')('72'), 'Custom (72%)')

    def test_changed_buttons_in_table_order(self):
        controls = {'key': {'circle': ' Left Shift ', 'cross': 'Q', 'square': ''}, 'pad': {'cross': 'b'}}
        self.assertEqual(launcher.control_map(controls, 'key'), 'cross=Q,circle=Left Shift')
        self.assertEqual(launcher.control_map(controls, 'pad'), 'cross=b')

    def test_rejects_separators_unknown_buttons_and_bad_shapes(self):
        controls = {'key': {'cross': 'a,b', 'circle': 'x=y', 'jump': 'Q', 'square': 5}}
        self.assertEqual(launcher.control_map(controls, 'key'), '')
        self.assertEqual(launcher.control_map([], 'key'), '')
        self.assertEqual(launcher.control_map({}, 'pad'), '')

    def test_only_changes_from_the_defaults_are_kept(self):
        keys = {b: k for b, k, _p in launcher.CONTROLS}
        keys.update(cross='q', circle='LEFT SHIFT', square='  ')
        self.assertEqual(launcher.control_changes(keys, 'key'), {'cross': 'q'})
        pads = {b: p or '' for b, _k, p in launcher.CONTROLS}
        pads.update(cross='b', l2='a')
        self.assertEqual(launcher.control_changes(pads, 'pad'), {'cross': 'b'})  # L2/R2: triggers only

    def test_both_styles_name_every_default_gamepad_button(self):
        for labels in launcher.PAD_STYLES.values():
            self.assertTrue(all(p in labels for _b, _k, p in launcher.CONTROLS if p))
            self.assertEqual(len(set(labels.values())), len(labels))

    def test_environment(self):
        settings = {**launcher.APP_DEFAULTS, 'controls': {'key': {'cross': 'Q'}}}
        env = launcher.game_environment(settings)
        self.assertEqual(env['GOW3_KEY_MAP'], 'cross=Q')
        self.assertNotIn('GOW3_PAD_MAP', env)
        self.assertEqual(env['GOW3_PARALLEL_WARMUP'], '1')
        self.assertEqual(env['GOW3_BACKGROUND_WARMUP'], '1')
        self.assertEqual(env['GOW3_PERF_DIAG'], '0')
        self.assertNotIn('GOW3_FRAME_CAPTURE_KEY', {k for k in env if k not in __import__('os').environ})
        self.assertEqual(launcher.game_environment({**settings, 'frame_capture_key': True})['GOW3_FRAME_CAPTURE_KEY'], '1')
        # Graphics options live in gow3.ini (run.py and the in-game menu), not in the environment.
        for name in ('GOW3_FULLSCREEN', 'GOW3_PRESENT_MODE', 'GOW3_FPS_LIMIT', 'GOW3_ASYNC_SHADERS',
                     'GOW3_DEFERRED_READBACK', 'GOW3_STALE_READBACK', 'GOW3_FSR1', 'GOW3_RCAS'):
            self.assertNotIn(name, {k for k in env if k not in __import__('os').environ})
        old = launcher.game_environment({**settings, 'extra_env': 'GOW3_DRAW_PIPE=0'})
        self.assertNotIn('GOW3_DRAW_PIPE', old)  # the old free-text field is ignored
        self.assertEqual(launcher.game_environment({**settings, 'parallel_warmup': False})['GOW3_PARALLEL_WARMUP'], '0')
        self.assertEqual(launcher.game_environment({**settings, 'background_warmup': False})['GOW3_BACKGROUND_WARMUP'], '0')

    def test_new_default_patches_start_on(self):
        from patches import game_profile
        profile = game_profile('CUSA11227', '01.02')
        pool, bounds = 'Native 4K framebuffer pool (512 MB)', 'GPU particle bounds'
        select = lambda **s: launcher.game_patch_selection(s, 'CUSA11227', profile)
        self.assertIsNone(select())  # no saved selection: run.py uses the file's defaults
        # Saved before game_patches_seen existed: the new default comes on, the chosen ones stay,
        # a default the user had switched off stays off.
        self.assertEqual(select(game_patches={'CUSA11227': [pool]}), [pool, bounds])
        self.assertEqual(select(game_patches={'CUSA11227': []}), [bounds])
        # Switched off after the page showed it: stays off.
        self.assertEqual(select(game_patches={'CUSA11227': []}, game_patches_seen={'CUSA11227': [pool, bounds]}), [])
        # Shown before it became a default (seen): not switched on behind the user's back.
        self.assertEqual(select(game_patches={'CUSA11227': [pool]}, game_patches_seen={'CUSA11227': [pool]}),
                         [pool, bounds])
        self.assertEqual(select(game_patches={'CUSA11227': [pool]},
                                game_patches_seen={'CUSA11227': [pool, bounds]}), [pool])
        env = launcher.game_environment({**launcher.APP_DEFAULTS, 'game_dir': '', 'game_patches': {}})
        self.assertNotIn('GOW3_PATCHES_ONLY', {k for k in env if k not in __import__('os').environ})

    def test_settings_keys_kept(self):
        # The redesign hides some options (player name) but keeps every saved setting working.
        for key in ('game_dir', 'user_dir', 'mods_dir', 'patches_dir', 'language', 'player_name', 'hdr',
                    'draw_pipe', 'readbacks', 'background_warmup', 'parallel_warmup', 'controls', 'pad_style'):
            self.assertIn(key, launcher.APP_DEFAULTS)
        env = launcher.game_environment({**launcher.APP_DEFAULTS, 'player_name': ' Kratos '})
        self.assertEqual(env['GOW3_USER_NAME'], 'Kratos')
        for key in ('display_mode', 'fps_limit', 'output_resolution', 'render_scale', 'engine_fps', 'rcas_strength'):
            self.assertIn(key, launcher.INI_DEFAULTS)

    def test_graphics_defaults_match_the_game(self):
        d = launcher.INI_DEFAULTS
        self.assertEqual((d['display_mode'], d['vsync'], d['fps_limit'], d['engine_fps']), ('windowed', '1', '-3', '60'))
        self.assertEqual([v for v, _t in launcher.FPS_LIMITS],
                         ['-2', '-3', '30', '60', '119', '120', '141', '237', '240', '0', '-1'])
        self.assertEqual((d['present_limiter'], d['present_mode'], d['present_tick_hz']), ('reflex', 'auto', '960'))
        self.assertEqual((d['fsr1'], d['rcas'], d['rcas_strength'], d['frames_queued']), ('0', '1', '75', '1'))
        self.assertEqual([v for v, _t in launcher.FRAMES_QUEUED], ['1', '2', '0'])
        self.assertNotIn('GOW3_FRAMES_AHEAD', launcher.game_environment({**launcher.APP_DEFAULTS, 'frames_ahead': '2'}))


class ShortcutTest(unittest.TestCase):
    def test_start_menu_shortcut_uses_the_game_icon(self):
        icon = launcher.PORT_DIR / 'launcher' / 'medievil.ico'
        self.assertTrue(icon.is_file())
        script = launcher.shortcut_script('Programs', r'C:\MediEvil\MediEvil.exe', '', icon)
        self.assertIn('[Environment]::GetFolderPath("Programs")+"\\MediEvil.lnk"', script)
        self.assertIn(f'$s.IconLocation="{icon}"', script)
        self.assertNotIn('$s.Arguments', script)
        script = launcher.shortcut_script('Desktop', 'pythonw.exe', 'launcher.py', icon)
        self.assertIn('GetFolderPath("Desktop")', script)
        self.assertIn("$s.Arguments='\"launcher.py\"'", script)


class ReleaseVersionTest(unittest.TestCase):
    def test_fork_release_is_newer_than_upstream_and_checks_the_fork(self):
        self.assertEqual(launcher.VERSION, '1.4.0')
        self.assertGreater(launcher.version_tuple(launcher.VERSION), launcher.version_tuple('v1.3.8'))
        self.assertGreater(launcher.version_tuple('v2.1.2'), launcher.version_tuple(launcher.VERSION))
        self.assertIn('/repos/bc160201983/', launcher.RELEASES_API)

    def test_first_start_is_in_english(self):
        # The translations are partial; 'System' stays a choice in Advanced.
        self.assertEqual(launcher.APP_DEFAULTS['ui_language'], 'en')


if __name__ == '__main__':
    unittest.main()
