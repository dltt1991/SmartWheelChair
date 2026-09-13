import math
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import check_door_matrix as matrix
from check_door_matrix import inspect_sweep, overlaps, rear_extent


class DoorMatrixTest(unittest.TestCase):
    def test_chain_requires_crossings_wall_turn_and_fresh_release(self):
        from check_door_chain import chain_passed
        result = dict(raw_seen=True, collision=None, stopped_samples_between=0,
                      crossed_second=True, turned_at_wall=True, released_stopped=True,
                      neupan_enabled=True, neupan_used=True)
        self.assertTrue(chain_passed(result))
        for key, value in [('raw_seen', False), ('collision', 'wall'),
                           ('stopped_samples_between', 1), ('crossed_second', False),
                           ('turned_at_wall', False), ('released_stopped', False), ('neupan_used', False)]:
            with self.subTest(key=key):
                self.assertFalse(chain_passed(dict(result, **{key: value})))

    def test_chain_neupan_evidence_requires_active_positive_output(self):
        from check_door_chain import neupan_forward_events, chain_passed
        events = [dict(monotonic=t, status=dict(planner_source=source, v=v, mode='wall'))
                  for t, source, v in [(1., 'local_follower', .2), (2., 'neupan', 0.),
                                       (3., 'neupan', -.1), (4., 'neupan', .1),
                                       (5., 'neupan', .2)]]
        events.insert(0, dict(monotonic=.5, status=dict(planner_source='neupan', v=.2, mode='manual')))
        self.assertEqual(neupan_forward_events(events, released=4.5), 1)
        self.assertEqual(neupan_forward_events(events, released=4.), 0)
        result = dict(raw_seen=True, collision=None, stopped_samples_between=0,
                      crossed_second=True, turned_at_wall=True, released_stopped=True,
                      neupan_enabled=True, neupan_used=False)
        self.assertFalse(chain_passed(result))
        self.assertTrue(chain_passed(dict(result, neupan_enabled=False)))

    def test_chain_wall_turn_requires_current_mode_heading_and_forward_motion(self):
        from check_door_chain import wall_turn_observed
        row = dict(pose=[-4.5, -4.3, -math.pi/2+1.21],
                   measured=[.026, .1], status=dict(mode='wall'))
        self.assertTrue(wall_turn_observed(row))
        for update in (dict(status=dict(mode='door_clear')), dict(measured=[0., .1]),
                       dict(measured=[-.1, .1]), dict(measured=[.025, .1]),
                       dict(pose=[-4.5, -4., -math.pi/2+1.21]),
                       dict(pose=[-4.5, -4.3, -math.pi/2+1.19]),
                       dict(pose=[-4.5, -4.3, -math.pi/2+2*math.pi])):
            with self.subTest(update=update):
                self.assertFalse(wall_turn_observed(dict(row, **update)))

    def test_chain_stop_requires_new_command_and_measured_samples(self):
        from check_door_chain import stopped
        sample = dict(cmd=[0., 0.], measured=[0., 0.], cmd_received=10.1, odom_received=10.1)
        self.assertTrue(stopped(sample, 10., 10.2))
        for updates in (dict(cmd_received=9.9), dict(odom_received=9.9),
                        dict(measured=[.1, 0.]), dict(cmd=[0., .1])):
            self.assertFalse(stopped(dict(sample, **updates), 10., 10.2))
        self.assertFalse(stopped(sample, 10., 10.5))

    door = ('test', 0., 0., 1, 1.)
    normal = np.array([1., 0.])

    def test_acceptance_requires_every_expected_case_to_pass(self):
        passing = [dict(result='passed') for _ in range(20)]
        self.assertTrue(matrix.matrix_passed(passing, expected=20))
        self.assertFalse(matrix.matrix_passed(passing[:-1], expected=20))
        self.assertFalse(matrix.matrix_passed(passing+[dict(result='passed')], expected=20))
        self.assertFalse(matrix.matrix_passed(passing[:-1]+[dict(result='stalled')], expected=20))
        self.assertFalse(matrix.matrix_passed([], expected=0))

    def test_lateral_offset_changes_position_not_heading_or_target_door(self):
        for door in matrix.DOORS:
            for direction in ('in', 'out'):
                axle, yaw, normal = matrix.start_pose(door, direction, 15.)
                shifted, shifted_yaw, shifted_normal = matrix.start_pose(door, direction, 15., .25)
                np.testing.assert_allclose(shifted-axle, .25*np.array([-normal[1], normal[0]]), atol=1e-12)
                np.testing.assert_allclose(shifted_normal, normal)
                self.assertEqual(shifted_yaw, yaw)

    def test_all_84_map_starts_have_four_centimetres_of_body_clearance(self):
        scene = matrix.M6AccessibilityTest()
        scene.setUp()
        checked = 0
        for door in matrix.DOORS:
            for direction in ('in', 'out'):
                for angle in (0, -15, 15, -30, 30, -45, 45):
                    with self.subTest(door=door[0], direction=direction, angle=angle):
                        axle, yaw, _ = matrix.start_pose(door, direction, angle)
                        self.assertIsNone(overlaps((*axle, yaw), scene.boxes, .04))
                        checked += 1
        self.assertEqual(checked, 84)

    def test_crossing_another_gap_does_not_count_for_requested_door(self):
        _, crossed = inspect_sweep([(-1., 2., 0.), (1., 2., 0.)], {}, self.door, self.normal)
        self.assertFalse(crossed)
        _, crossed = inspect_sweep([(-1., .1, 0.), (1., .1, 0.)], {}, self.door, self.normal)
        self.assertTrue(crossed)
        _, crossed = inspect_sweep([(1., .1, 0.), (-1., .1, 0.)], {}, self.door, self.normal)
        self.assertFalse(crossed)

    def test_rear_projection_accounts_for_long_front_when_facing_backwards(self):
        self.assertAlmostEqual(rear_extent(0.), .25)
        self.assertAlmostEqual(rear_extent(math.pi), .97)
        self.assertAlmostEqual(rear_extent(math.pi/2), .4)
        self.assertGreater(rear_extent(math.radians(135)), .9)

    def test_translation_between_clear_samples_cannot_skip_wall(self):
        boxes = {'wall': (-.02, -2., .02, 2.)}
        poses = [(-1.1, 0., 0.), (.4, 0., 0.)]
        self.assertTrue(all(overlaps(p, boxes, .02) is None for p in poses))
        hit, _ = inspect_sweep(poses, boxes, self.door, self.normal)
        self.assertEqual(hit, 'wall')

    def test_rotation_between_clear_samples_checks_front_corner_sweep(self):
        boxes = {'post': (.66, .66, .69, .69)}
        poses = [(0., 0., 0.), (0., 0., math.pi/2)]
        self.assertTrue(all(overlaps(p, boxes, .02) is None for p in poses))
        hit, _ = inspect_sweep(poses, boxes, self.door, self.normal)
        self.assertEqual(hit, 'post')


if __name__ == '__main__':
    unittest.main()
