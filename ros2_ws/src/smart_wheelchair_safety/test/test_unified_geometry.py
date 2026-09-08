import math
import unittest
import numpy as np

from smart_wheelchair_safety.unified_geometry import (
    Door, Segment, door_reference, wall_reference, extract_lines, find_door, braking_clear, arc_path, transform_points,
)


class UnifiedGeometryTest(unittest.TestCase):
    def test_corner_is_two_walls_not_an_imaginary_diagonal(self):
        points = [(x, .8) for x in np.linspace(0, 1, 30)]
        points += [(1, y) for y in np.linspace(.8, 1.8, 30)]
        lines = extract_lines(points)
        self.assertEqual(len(lines), 2)
        self.assertLess(abs(lines[0].heading), .01)
        self.assertAlmostEqual(abs(lines[1].heading), math.pi / 2, places=2)

    def test_detects_one_metre_door_and_rejects_too_narrow_gap(self):
        for width in (1., .84):
            points = [(2., y) for y in np.linspace(-2, -width / 2, 40)]
            points += [(2., y) for y in np.linspace(width / 2, 2, 40)]
            door = find_door(extract_lines(points))
            if width == 1.:
                self.assertIsNotNone(door)
                self.assertAlmostEqual(door.width, 1.)
                self.assertAlmostEqual(door.center[0], 2.)
            else:
                self.assertIsNone(door)

    def test_near_parallel_wall_does_not_force_slowdown(self):
        wall = [(x, .55) for x in np.linspace(-1, 4, 200)]
        self.assertTrue(braking_clear(wall, (.8, 0.), (.8, 0.)))

    def test_wall_reference_targets_twelve_cm_from_body_edge_on_both_sides(self):
        for heading in (0., .3, -.3):
            tangent = np.array([math.cos(heading), math.sin(heading)])
            normal = np.array([-tangent[1], tangent[0]])
            for side in (-1., 1.):
                distance = side*.8
                line = Segment(tuple(distance*normal), tuple(4*tangent+distance*normal), heading, distance)
                path, mode, _ = wall_reference([line], .6, side*.1)
                self.assertEqual(mode, 'wall')
                clearance = abs(distance-path[-1, :2] @ normal)-.4
                self.assertAlmostEqual(clearance, .12)

    def test_wall_clearance_can_be_calibrated_without_changing_guard(self):
        line = Segment((0., .8), (4., .8), 0., .8)
        path, _, _ = wall_reference([line], .6, 0., body_clearance=.10)
        self.assertAlmostEqual(.8-path[-1, 1]-.4, .10)
        self.assertTrue(braking_clear([(x, .52) for x in np.linspace(-1., 4., 200)],
                                      (.8, 0.), (.8, 0.), state_age=.02))

    def test_braking_tail_and_front_corner_are_checked(self):
        wall = [(1.4, y) for y in np.linspace(-2, 2, 100)]
        self.assertFalse(braking_clear(wall, (.8, 0.), (.8, 0.)))
        self.assertTrue(braking_clear(wall, (.1, 0.), (.1, 0.)))
        self.assertFalse(braking_clear([(1.02, .43)], (.5, .5), (.5, .5)))

    def test_reverse_and_rotation_check_rear_sweep(self):
        self.assertFalse(braking_clear([(-.45, 0.)], (-.5, 0.), (-.5, 0.)))
        self.assertFalse(braking_clear([(.0, -.47)], (0., 1.), (0., 1.)))

    def test_zero_command_cannot_instantly_remove_measured_momentum(self):
        self.assertFalse(braking_clear([(1.4, 0.)], (0., 0.), (.8, 0.)))
        self.assertTrue(braking_clear([(2.5, 0.)], (0., 0.), (.8, 0.)))

    def test_stale_pose_includes_unreported_motion_before_braking(self):
        self.assertTrue(braking_clear([(1.95, 0.)], (.8, 0.), (.8, 0.)))
        self.assertFalse(braking_clear([(1.95, 0.)], (.8, 0.), (.8, 0.), state_age=.35))

    def test_door_reference_does_not_turn_back_after_axle_crosses(self):
        path = door_reference(Door((-.2, .01), 0., 1.))
        self.assertTrue(np.all(np.diff(path[:, 0]) > 0.))
        self.assertGreater(path[-1, 0], 1.)

    def test_approaching_wall_is_selected_even_when_opposite_wall_is_closer(self):
        heading = -math.pi/6
        tangent = np.array([math.cos(heading), math.sin(heading)])
        normal = np.array([-tangent[1], tangent[0]])
        lines = [Segment(tuple(-tangent+d*normal), tuple(3*tangent+d*normal), heading, d)
                 for d in (1.45, -1.12)]
        _, mode, side = wall_reference(lines, .8, .42)
        self.assertEqual(mode, 'wall')
        self.assertEqual(side, 1)

    def test_rigid_transform_round_trip_and_axle_arc(self):
        points = np.array([[0., 0.], [1., .3]])
        pose = (2., 3., .7)
        np.testing.assert_allclose(transform_points(transform_points(points, pose), pose, inverse=True), points, atol=1e-12)
        path = arc_path(.5, .5, duration=2.)
        self.assertAlmostEqual(path[-1, 0], math.sin(1.), places=5)
        self.assertAlmostEqual(path[-1, 1], 1. - math.cos(1.), places=5)
