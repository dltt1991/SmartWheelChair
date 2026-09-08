import math
import unittest
import numpy as np

from smart_wheelchair_safety.unified_geometry import (
    Door, Opening, Segment, door_alignment_reference, door_entry_clearance,
    door_reference, wall_reference, extract_lines, find_door, find_openings,
    intended_side_opening, opening_matches, braking_clear, arc_path,
    transform_points, intended_front_door,
)


class UnifiedGeometryTest(unittest.TestCase):
    def test_active_aperture_targeting_covers_near_plane_arcs_on_both_sides(self):
        from smart_wheelchair_safety.unified_geometry import active_aperture_targeted
        for side in (-1., 1.):
            door = Opening((.75, side*.04), side*.04, 1.)
            for turn in (-.3, .3):
                with self.subTest(side=side, turn=turn):
                    self.assertTrue(active_aperture_targeted(door, .5, turn))
            self.assertTrue(active_aperture_targeted(door, .5, 0.))

    def test_active_aperture_targeting_rejects_turn_out_before_rear_clearance(self):
        from smart_wheelchair_safety.unified_geometry import active_aperture_targeted
        door = Opening((.75, 0.), 0., 1.)
        for turn in (-.5, .5):
            self.assertFalse(active_aperture_targeted(door, .5, turn))

    def test_active_aperture_targeting_does_not_invent_low_speed_crossing(self):
        from smart_wheelchair_safety.unified_geometry import active_aperture_targeted
        door = Opening((1.6, .45), .18, 1.)
        self.assertFalse(active_aperture_targeted(door, .1, .6))
        self.assertFalse(active_aperture_targeted(door, .5, -.5))
        self.assertFalse(active_aperture_targeted(Opening((-.1, 0.), 0., 1.), .5, .5))

    def test_active_aperture_analysis_distinguishes_traversal_departure_and_unknown(self):
        from smart_wheelchair_safety.unified_geometry import active_aperture_targeted
        for side in (-1., 1.):
            cases = (
                (Opening((1.6, side*.04), side*.3, 1.), .1, side*.6, None),
                (Opening((.75, side*.04), side*.04, 1.), .5, side*.3, True),
                (Opening((.75, side*.001), 0., 1.), .5, side*.5, False),
                (Opening((.75, -side*.6), 0., 1.), .5, side*.3, False),
                (Opening((-.1, side*.001), 0., 1.), .5, side*.5, None),
                (Opening((0., 0.), 0., 1.), .5, side*.5, None),
            )
            for door, v, w, expected in cases:
                with self.subTest(door=door, v=v, w=w):
                    self.assertIs(active_aperture_targeted(door, v, w), expected)

    def test_general_opening_detector_serves_side_passages_and_front_doors(self):
        side = [
            Segment((-1., .8), (.2, .8), 0., .8),
            Segment((2.8, .8), (4., .8), 0., .8),
        ]
        front = [
            Segment((2., -2.), (2., -.5), math.pi / 2, -2.),
            Segment((2., .5), (2., 2.), math.pi / 2, -2.),
        ]

        side_opening = find_openings(side)[0]
        front_opening = find_openings(front, max_width=1.5)[0]

        np.testing.assert_allclose(side_opening.center, [1.5, .8])
        self.assertAlmostEqual(side_opening.width, 2.6)
        self.assertGreater(math.sin(side_opening.heading), .99)
        np.testing.assert_allclose(front_opening.center, [2., 0.])
        self.assertAlmostEqual(front_opening.width, 1.)
        self.assertGreater(math.cos(front_opening.heading), .99)

    def test_opening_detector_rejects_one_jamb_bad_plane_and_out_of_range(self):
        lone = [Segment((-1., .8), (.2, .8), 0., .8)]
        bad_plane = lone + [Segment((1.2, .92), (3., .92), 0., .92)]
        far = [
            Segment((5., .8), (6., .8), 0., .8),
            Segment((7., .8), (8., .8), 0., .8),
        ]

        self.assertEqual(find_openings(lone), [])
        self.assertEqual(find_openings(bad_plane), [])
        self.assertEqual(find_openings(far), [])

    def test_opening_association_tolerates_scan_noise_but_not_a_new_gap(self):
        first = Opening((1.5, .8), math.pi / 2, 2.6, ())
        noisy = Opening((1.68, .84), math.pi / 2 + .08, 2.43, ())
        other = Opening((2.1, .8), math.pi / 2, 2.6, ())

        self.assertTrue(opening_matches(first, noisy))
        self.assertFalse(opening_matches(first, other))

    def test_driver_arc_selects_opening_on_followed_side(self):
        left = Opening((1.0, .8), math.pi / 2, 2.0, ())
        right = Opening((1.0, -.8), -math.pi / 2, 2.0, ())

        self.assertEqual(intended_side_opening([left], 1, .6, .5), left)
        self.assertEqual(intended_side_opening([right], -1, .6, -.5), right)
        self.assertIsNone(intended_side_opening([left], 1, .6, 0.))
        self.assertIsNone(intended_side_opening([left], -1, .6, .5))

    def test_driver_arc_must_cross_between_jambs(self):
        too_far = Opening((3.0, .8), math.pi / 2, 1.0, ())
        near_narrow_door = Opening((1.0, .8), math.pi / 2, 1.2, ())

        self.assertIsNone(intended_side_opening([too_far], 1, .4, .6))
        self.assertIsNone(intended_side_opening([near_narrow_door], 1, .6, .5))

    def test_oblique_joystick_arc_selects_front_door(self):
        for side in (-1., 1.):
            door = Opening((1.6, side*.45), side*.18, 1., ())
            self.assertEqual(intended_front_door([door], .6, side*.30), door)

    def test_front_door_rejects_opening_below_minimum_width(self):
        door = Opening((1.6, .45), .18, .84, ())
        self.assertIsNone(intended_front_door([door], .6, .30))

    def test_arc_turning_away_does_not_select_front_door(self):
        door = Opening((1.6, .45), .18, 1., ())
        self.assertIsNone(intended_front_door([door], .6, -.35))

    def test_wide_side_corridor_is_selected_early_for_sustained_turn_intent(self):
        corridor = Opening((3.5, -.52), -math.pi / 2, 2.7, ())

        self.assertEqual(intended_side_opening([corridor], -1, .8, -.6), corridor)

    def test_opening_pair_rejects_gap_occupied_by_another_coplanar_segment(self):
        lines = [
            Segment((2., -3.), (2., -.95), math.pi / 2, -2.),
            Segment((2., -3.), (2., -.5), math.pi / 2, -2.),
            Segment((2., .5), (2., 3.), math.pi / 2, -2.),
        ]

        openings = find_openings(lines, max_width=1.5)

        self.assertEqual(len(openings), 1)
        self.assertAlmostEqual(openings[0].width, 1.)

    def test_door_alignment_uses_more_setback_for_larger_error(self):
        centered_door = Opening((2., 0.), 0., 1., ())
        skewed_door = Opening((2., .25), .2, 1., ())
        centered = door_alignment_reference(centered_door, 'align')
        skewed = door_alignment_reference(skewed_door, 'align')
        skewed_normal = np.array([math.cos(.2), math.sin(.2)])

        self.assertLess(skewed[-1, :2] @ skewed_normal, centered[-1, 0])
        self.assertAlmostEqual(skewed[-1, 2], .2, places=2)
        np.testing.assert_allclose(skewed[0], [0., 0., 0.], atol=1e-9)
        self.assertLess(skewed[-1, :2] @ skewed_normal,
                        np.array(skewed_door.center) @ skewed_normal)

    def test_one_metre_entry_requires_body_and_margin_to_fit(self):
        self.assertAlmostEqual(door_entry_clearance(Opening((1., .05), 0., 1., ())), .01)
        aligned = (math.cos(.04), math.sin(.04))
        self.assertGreater(door_entry_clearance(Opening(aligned, .04, 1., ())), 0.)
        self.assertLess(door_entry_clearance(Opening((1., .06), .08, 1., ())), 0.)

    def test_pass_reference_starts_here_and_converges_to_frozen_centerline(self):
        door = Opening((.55, -.03), .04, 1., ())
        path = door_alignment_reference(door, 'pass')
        normal = np.array([math.cos(door.heading), math.sin(door.heading)])
        tangent = np.array([-normal[1], normal[0]])

        np.testing.assert_allclose(path[0], [0., 0., 0.], atol=1e-9)
        self.assertTrue(np.all(np.diff(path[:, :2] @ normal) > 0.))
        self.assertAlmostEqual((path[-1, :2] - door.center) @ tangent, 0., places=9)
        self.assertGreater((path[-1, :2] - door.center) @ normal, 1.4)

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
