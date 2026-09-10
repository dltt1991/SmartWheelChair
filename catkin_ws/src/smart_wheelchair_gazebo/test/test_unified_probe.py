import importlib.util
from pathlib import Path
import unittest


PROBE = Path(__file__).parents[4] / 'scripts' / 'probe_unified_control.py'
spec = importlib.util.spec_from_file_location('unified_probe', PROBE)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class UnifiedProbeTest(unittest.TestCase):
    def test_release_settling_requires_new_fresh_command_and_odometry(self):
        self.assertTrue(hasattr(probe, 'settled_after_release'),
                        'cached zero samples must not prove post-release settling')
        data = {'cmd_vel': [0., 0.], 'odom_velocity': [0., 0.],
                'cmd_vel_received': 9., 'odom_velocity_received': 9.}
        self.assertFalse(probe.settled_after_release(data, released=10., now=10.1))
        data['cmd_vel_received'] = 10.05
        self.assertFalse(probe.settled_after_release(data, released=10., now=10.1))
        data['odom_velocity_received'] = 10.06
        self.assertTrue(probe.settled_after_release(data, released=10., now=10.1))
        self.assertFalse(probe.settled_after_release(data, released=10., now=10.5))
        data['odom_velocity'] = [.03, 0.]
        self.assertFalse(probe.settled_after_release(data, released=10., now=10.1))
        data['odom_velocity'] = []
        self.assertFalse(probe.settled_after_release(data, released=10., now=10.1))

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
