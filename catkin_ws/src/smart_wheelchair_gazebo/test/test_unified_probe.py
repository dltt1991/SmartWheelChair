import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch, MagicMock
import json


PROBE = Path(__file__).parents[4] / 'scripts' / 'probe_unified_control.py'
spec = importlib.util.spec_from_file_location('unified_probe', PROBE)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class UnifiedProbeTest(unittest.TestCase):
    def test_sample_row_pairs_later_odom_with_actual_snapshot_time(self):
        # HTTP/scan work started at 100.; callbacks advanced before copying data.
        data = dict(odom=[1., 0., 0.], odom_velocity=[0., 0.],
                    odom_velocity_received=100.15, cmd_vel=[0.,0.],
                    cmd_vel_received=100.12, cmd_vel_raw=[.5,0.],
                    cmd_vel_raw_received=100.13)
        scanned = [('left', (object(), 100.10)), ('right', (object(), 100.11))]
        with patch.object(probe.time, 'monotonic', return_value=100.16):
            row = probe.sample_row(data, scanned, 100.)
        self.assertAlmostEqual(row['t'], .16)
        self.assertEqual(row['sampled_monotonic'], 100.16)
        self.assertTrue(probe.fresh_row(row, 100., ('raw','cmd_vel','odom_velocity','left','right')))
        data['odom'] = [2.,0.,0.]
        self.assertEqual(row['odom'], [1.,0.,0.])

    def test_sample_row_uses_receipt_of_scans_actually_used_for_geometry(self):
        data = dict(left_received=100.49, right_received=100.49)
        scanned = [('left', (object(), 100.10)), ('right', (object(), 100.11))]
        with patch.object(probe.time, 'monotonic', return_value=100.50):
            row = probe.sample_row(data, scanned, 100.)
        self.assertEqual(row['left_received'], 100.10)
        self.assertEqual(row['right_received'], 100.11)
        self.assertFalse(probe.fresh_row(row, 100., ('left','right')),
                         'new callback receipt cannot make old scanned geometry fresh')

    def test_front_requires_forward_obstacle_progress_and_sustained_fresh_stop(self):
        rows = [dict(t=t, odom=[x, 0., 0.], cmd_vel_raw=[.5, 0.],
                     cmd_vel=[v, 0.], odom_velocity=[v, 0.], front_obstacle_gap=gap,
                     raw_received=t, cmd_vel_received=t, odom_velocity_received=t, left_received=t, right_received=t,
                     status=dict(mode='manual', planner_source='neupan'))
                for t, x, v, gap in [(1., 0., .2, .7)]+[(2.+i*.1, .5, 0., .1) for i in range(21)]]
        probe.check_front_stop(rows, started=0.)
        for change in (dict(front_obstacle_gap=None), dict(cmd_vel_raw=[0., 0.]),
                       dict(odom_velocity=[.1, 0.]), dict(cmd_vel_received=0.), dict(left_received=0.),
                       dict(odom=[0., 0., 0.])):
            with self.subTest(change=change), self.assertRaises(AssertionError):
                probe.check_front_stop([dict(row, **change) for row in rows], started=0.)
        with self.assertRaises(AssertionError):
            probe.check_front_stop(rows[:2], started=0.)

    def test_override_fixture_is_parallel_on_solid_wall_not_beneath_door_gap(self):
        x, y, yaw = probe.override_start_pose()
        axle_x = x-.33
        self.assertAlmostEqual(yaw, 0.)
        self.assertAlmostEqual(1.29-y-.4, .12)
        # Actual door_nw_horizontal_b occupies x=[-4,-1.35].
        self.assertGreater(axle_x-.25, -4.)
        self.assertGreater(-1.35-axle_x, 1.5)

    def test_override_side_wall_requires_visible_coverage_not_blind_rear_corner(self):
        from types import SimpleNamespace
        import math
        # At body-side gap .12, the left lidar sees the side wall at dy=.16.
        self.assertGreater(math.degrees(math.atan2(.52-.36, -.25-.79)), 170.)
        visible_min = .79+(.52-.36)/math.tan(math.radians(170.))
        self.assertGreater(visible_min, -.25)
        line = SimpleNamespace(heading=0., distance=.52, start=(visible_min,.52), end=(2.18,.52))
        self.assertIs(probe.override_wall_line([line]), line)
        for change in (dict(start=(.80,.52)), dict(end=(1.4,.52)), dict(distance=1.2)):
            bad = SimpleNamespace(**dict(vars(line), **change))
            self.assertIsNone(probe.override_wall_line([bad]))

    def test_override_requires_stable_current_wall_then_real_away_override(self):
        rows = [dict(t=t, odom=[t*.1, 0., 0.], cmd_vel_raw=[.5, .42],
                     wall_side=1, wall_heading=0., status=dict(mode='wall'),
                     raw_received=t, status_received=t, left_received=t, right_received=t)
                for t in (1., 1.2, 1.4, 1.6)]
        trigger = probe.override_trigger(rows, started=0.)
        self.assertIsNotNone(trigger)
        self.assertIsNone(probe.override_trigger([dict(row, wall_heading=-.2903) for row in rows], 0.))
        self.assertIsNone(probe.override_trigger([dict(row, status=dict(mode='opening_turn')) for row in rows], 0.))
        self.assertIsNone(probe.override_trigger([dict(row, wall_side=0) for row in rows], 0.))
        after = dict(t=2.6, odom=[.7, -.2, -.3], cmd_vel_raw=[.5, -.42],
                     cmd_vel=[.4, -.4], odom_velocity=[.4, -.35], status=dict(mode='override'),
                     raw_received=2.6, status_received=2.6, cmd_vel_received=2.6, odom_velocity_received=2.6)
        probe.check_override_response([after], trigger, 0.)
        for change in (dict(status=dict(mode='manual')), dict(odom_velocity=[.4, 0.]),
                       dict(odom=[.7, 0., .3]), dict(cmd_vel_raw=[.5, .42]),
                       dict(raw_received=0.)):
            with self.subTest(change=change), self.assertRaises(AssertionError):
                probe.check_override_response([dict(after, **change)], trigger, 0.)

    def test_staged_override_accepts_safe_slow_turn_and_continued_handoff(self):
        trigger = dict(t=1., pose=[0., 0., 0.], wall_side=1., wall_heading_world=0.)
        def row(t, mode, y, yaw):
            return dict(t=t, odom=[t*.1, y, yaw], cmd_vel_raw=[.5, -.42],
                        cmd_vel=[.2, -.1], odom_velocity=[.2, -.09],
                        status=dict(mode=mode, reason='clear'), raw_received=t,
                        status_received=t, cmd_vel_received=t, odom_velocity_received=t)
        rows = [row(1.1, 'override', -.001, -.01),
                row(1.2, 'manual', -.02, -.06), row(1.3, 'opening_turn', -.06, -.15)]
        probe.check_override_response(rows, trigger, 0.)
        events = [dict(monotonic=1.15, status=dict(mode='override', reason='clear', v=.2, w=-.1))]
        probe.check_override_response(rows, trigger, 0., events)
        events.append(dict(monotonic=1.25, status=dict(mode='manual', reason='clear', v=0., w=0.)))
        with self.assertRaises(AssertionError):
            probe.check_override_response(rows, trigger, 0., events)
        from copy import deepcopy
        for label in ('no_override', 'raw_changed', 'stale', 'reverse', 'timeout',
                      'no_motion', 'stop', 'bad_reason'):
            bad = deepcopy(rows)
            if label == 'no_override': bad[0]['status']['mode'] = 'manual'
            if label == 'raw_changed': bad[1]['cmd_vel_raw'] = [.6, -.42]
            if label == 'stale': bad[1]['raw_received'] = 0.
            if label == 'reverse': bad[1]['cmd_vel'][1] = .1
            if label == 'timeout':
                for r in bad: r['t'] += 4.
            if label == 'no_motion':
                for r in bad: r['odom'][1] = 0.
            if label == 'stop': bad[1]['cmd_vel'] = [0., 0.]
            if label == 'bad_reason': bad[1]['status']['reason'] = 'planner_timeout'
            with self.subTest(label=label), self.assertRaises(AssertionError):
                probe.check_override_response(bad, trigger, 0.)

    def test_fullrate_status_keeps_stop_between_sampled_rows(self):
        data, events = {}, []
        for stamp, v, w, reason in [(10., .2, .1, 'clear'),
                                     (10.05, 0., 0., 'emergency_stop'),
                                     (10.1, .2, -.1, 'clear')]:
            probe.record_control_status(data, events, dict(v=v, w=w, reason=reason), stamp)
        # The sampled endpoints both show forward motion, but the full stream
        # must retain the zero command and its original callback interval.
        self.assertTrue(all(events[i]['status']['v'] > 0 for i in (0, 2)))
        window = probe.control_event_window(events, 10., 10.1)
        metrics = probe.control_event_metrics(window)
        self.assertEqual(metrics['reason_counts'], {'clear': 2, 'emergency_stop': 1})
        self.assertEqual(metrics['max_linear_command_step'], .2)
        self.assertEqual(metrics['max_angular_command_step'], .1)
        self.assertAlmostEqual(metrics['max_linear_step_event']['interval_s'], .05)
        self.assertEqual(metrics['max_linear_step_event']['reason'], 'emergency_stop')
        self.assertEqual(len(window), 3)

    def test_control_events_are_deep_snapshots_and_case_windows_are_separate(self):
        data, events = {}, []
        status = dict(v=.1, w=0., reason='clear', input_age={'scan': .02})
        probe.record_control_status(data, events, status, 5.)
        status['input_age']['scan'] = .8
        status['reason'] = 'stale_input'
        probe.record_control_status(data, events, status, 6.)
        data['status']['input_age']['scan'] = 99.
        self.assertEqual(events[0]['status']['input_age']['scan'], .02)
        self.assertEqual(events[1]['status']['input_age']['scan'], .8)
        window = probe.control_event_window(events, 5.5, 6.5)
        self.assertEqual(len(window), 1)
        self.assertEqual(window[0]['monotonic'], 6.)
        self.assertEqual(window[0]['t'], .5)
        self.assertEqual(probe.control_event_metrics([])['reason_counts'], {})

    def test_commands_include_current_mode_epoch_after_switch(self):
        mode = MagicMock()
        mode.__enter__.return_value.read.return_value = json.dumps(
            dict(revision=3, session='new-session', assist_enabled=True)).encode()
        with patch.object(probe.urllib.request, 'urlopen', side_effect=[mode, MagicMock()]) as opened:
            probe.send_command('probe-test', .1, .4)
        payload = json.loads(opened.call_args.args[0].data)
        self.assertEqual(payload, dict(client_id='probe-test', x=.1, y=.4,
                                      mode_revision=3, mode_session='new-session'))

    def test_readiness_rejects_cached_or_missing_telemetry(self):
        data = {key+'_received': 10.1 for key in ('status', 'odom', 'left', 'right')}
        data['status'] = dict(reason='user_stop')
        self.assertTrue(probe.controller_ready(data, after=10., now=10.2))
        for key in ('status', 'odom', 'left', 'right'):
            with self.subTest(key=key):
                self.assertFalse(probe.controller_ready(dict(data, **{key+'_received': 9.9}), 10., 10.2))
                self.assertFalse(probe.controller_ready(dict(data, **{key+'_received': 10.1}), 10., 11.))
        self.assertFalse(probe.controller_ready({}, 10., 10.2))

    def test_readiness_waits_for_controller_to_accept_the_neutral_input(self):
        data = {key+'_received': 10.1 for key in ('status', 'odom', 'left', 'right')}
        for reason in ('stale_input', 'planner_timeout', 'clear'):
            with self.subTest(reason=reason):
                data['status'] = dict(reason=reason)
                self.assertFalse(probe.controller_ready(data, after=10., now=10.2))
        data['status'] = dict(reason='user_stop')
        self.assertTrue(probe.controller_ready(data, after=10., now=10.2))

    def test_wait_allows_slow_start_but_times_out_missing_sensor(self):
        clock = [0.]
        data = {}
        def sleep(seconds):
            clock[0] += seconds
        def command():
            if clock[0] > 1.5:
                data.update({key+'_received': clock[0] for key in ('status', 'odom', 'left', 'right')})
                data['status'] = dict(reason='user_stop')
        with patch.object(probe.time, 'monotonic', side_effect=lambda: clock[0]), \
                patch.object(probe.time, 'sleep', side_effect=sleep):
            probe.wait_for_controller(data, command, timeout=3.)
            self.assertGreater(clock[0], 1.5)
            with self.assertRaisesRegex(RuntimeError, 'both scans required'):
                probe.wait_for_controller({}, lambda: None, timeout=.2)

    def test_motion_metrics_keep_stops_and_use_command_receipt_intervals(self):
        rows = [dict(t=t, cmd=[.1, w], cmd_received=stamp, measured=[v, 0.],
                     status=dict(mode=mode, reason=reason))
                for t, w, stamp, v, mode, reason in [
                    (0., 0., 2., 0., 'waiting', 'planner_timeout'),
                    (1., .2, 2.1, 0., 'waiting', 'planner_timeout'),
                    (2., .2, 2.1, .2, 'door_pass', 'tracking')]]
        metrics = probe.motion_metrics(rows)
        self.assertAlmostEqual(metrics['max_angular_command_step'], .2)
        self.assertAlmostEqual(metrics['max_sampled_angular_acceleration'], 2.)
        self.assertEqual(metrics['longest_sampled_stop'], 1.)
        self.assertEqual(metrics['mode_switches'], 1)
        self.assertEqual(metrics['planner_wait_samples'], 2)

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
