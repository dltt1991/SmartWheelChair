import importlib.util
from pathlib import Path
import unittest


PROBE = Path(__file__).parents[4] / 'scripts' / 'probe_unified_control.py'
CONFIG = Path(__file__).parents[1] / 'config' / 'unified_control.yaml'
spec = importlib.util.spec_from_file_location('unified_probe', PROBE)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class UnifiedProbeTest(unittest.TestCase):
    def test_mppi_uses_physical_footprint_without_duplicating_guard_margin(self):
        config = CONFIG.read_text()

        self.assertIn('      footprint_padding: 0.0', config.splitlines())

    def test_moving_window_starts_at_first_nonzero_raw_command_and_keeps_stops(self):
        records = [
            {'t': 1.9, 'cmd_vel_raw': [0., 0.]},
            {'t': 2.05, 'cmd_vel_raw': [0., .3]},
            {'t': 2.2, 'cmd_vel_raw': [.5, .3]},
            {'t': 3.1, 'cmd_vel_raw': [0., 0.]},
        ]
        self.assertEqual(probe.moving_records(records), records[1:])
        self.assertEqual(probe.moving_records([{'t': 5.}]), [])
        self.assertEqual(probe.moving_records(records[:1]), [])

    def test_early_moving_wall_capture_fails_even_when_later_door_modes_succeed(self):
        records = [
            {'t': 1.9, 'cmd_vel_raw': [0., 0.], 'status': {'mode': 'wall'}},
            {'t': 2.05, 'cmd_vel_raw': [.5, .3], 'status': {'mode': 'wall'}},
            {'t': 2.2, 'cmd_vel_raw': [.5, .3], 'status': {'mode': 'door_align'}},
            {'t': 3.1, 'cmd_vel_raw': [.5, .3], 'status': {'mode': 'door_align'}},
            {'t': 3.5, 'cmd_vel_raw': [.5, .3], 'status': {'mode': 'door_pass'}},
        ]
        modes = [r['status']['mode'] for r in probe.moving_records(records)]
        with self.assertRaisesRegex(AssertionError, 'wall following captured'):
            probe.check_door_modes(modes)

    def test_door_modes_require_align_and_pass_and_reject_wall_after_align_too(self):
        probe.check_door_modes(['manual', 'door_align', 'door_pass', 'door_clear'])
        for modes, message in ((['door_pass'], 'alignment was not selected'),
                               (['door_align'], 'never committed to pass'),
                               (['door_align', 'wall', 'door_pass'], 'wall following captured')):
            with self.subTest(modes=modes), self.assertRaisesRegex(AssertionError, message):
                probe.check_door_modes(modes)
