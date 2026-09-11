import importlib.util
from pathlib import Path
import unittest


PROBE = Path(__file__).parents[4] / 'scripts' / 'probe_unified_control.py'
spec = importlib.util.spec_from_file_location('unified_probe', PROBE)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class UnifiedProbeTest(unittest.TestCase):
    def test_opening_gap_requires_turn_progress_and_no_prolonged_stop(self):
        self.assertTrue(hasattr(probe, 'check_opening_gap_result'))
        passing = {'junction_exit_time': 4., 'junction_yaw_change': 1.3,
                   'junction_forward_progress': .4, 'junction_longest_stop': .5}
        probe.check_opening_gap_result(passing, 1.)
        probe.check_opening_gap_result(dict(passing, junction_yaw_change=-1.3), -1.)
        for key, value in [('junction_exit_time', None), ('junction_yaw_change', .2),
                           ('junction_forward_progress', .1), ('junction_longest_stop', 3.1)]:
            with self.subTest(key=key), self.assertRaises(AssertionError):
                probe.check_opening_gap_result(dict(passing, **{key: value}), 1.)

    def test_junction_stop_metric_excludes_later_leg_but_keeps_stalls_before_exit(self):
        self.assertTrue(hasattr(probe, 'opening_gap_metrics'))
        records = [dict(t=t, odom=pose, odom_velocity=velocity) for t, pose, velocity in [
            (2., [0., 0., 0.], [.1, .1]), (3., [.3, .2, .5], [.3, .5]),
            (5., [.6, .7, 1.3], [.3, .5]), (10., [.6, 4., 1.5], [0., 0.]),
            (20., [.6, 4., 1.5], [0., 0.])]]
        result = probe.opening_gap_metrics(records, 1., (0., 0., 0.))
        self.assertEqual(result['junction_exit_time'], 5.)
        self.assertEqual(result['junction_longest_stop'], 0.)
        self.assertEqual(result['longest_stop'], 10.)
        probe.check_opening_gap_result(result, 1.)
        stalled = [dict(t=2., odom=[0., 0., 0.], odom_velocity=[0., 0.]),
                   dict(t=6., odom=[0., 0., 0.], odom_velocity=[0., 0.]),
                   dict(records[2], t=7.)] + records[3:]
        with self.assertRaisesRegex(AssertionError, 'prolonged stop'):
            probe.check_opening_gap_result(probe.opening_gap_metrics(stalled, 1., (0., 0., 0.)), 1.)

    def test_wall_acceptance_requires_mode_and_speed_recovery(self):
        self.assertTrue(hasattr(probe, 'check_wall_result'),
                        'wall probe must expose its real acceptance check')
        for max_speed in (0., .6):
            with self.subTest(max_speed=max_speed), \
                    self.assertRaisesRegex(AssertionError, 'wall speed did not recover'):
                probe.check_wall_result({'modes': ['wall'], 'max_speed': max_speed})
        probe.check_wall_result({'modes': ['wall'], 'max_speed': .600001})

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
