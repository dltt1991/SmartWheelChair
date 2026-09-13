#!/usr/bin/env python3
import math
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from smart_wheelchair_safety import unified_geometry as g

from smart_wheelchair_safety.unified_geometry import (
    Door, Opening, Segment, collision_aware_door_reference,
    door_alignment_reference, door_entry_clearance,
    door_reference, wall_reference, extract_lines, extract_opening_lines,
    find_door, find_openings,
    intended_side_opening, opening_matches, braking_clear, arc_path,
    transform_points, intended_front_door,
)


def dense_door_clearance(path, points):
    # Independent 2.5 mm / 2.5 mrad oracle, no production clearance helper.
    result = math.inf
    for a, b in zip(path, path[1:]):
        yaw = math.atan2(math.sin(b[2]-a[2]), math.cos(b[2]-a[2]))
        count = max(2, math.ceil(np.linalg.norm(b[:2]-a[:2])/.0025)+1,
                    math.ceil(abs(yaw)/.0025)+1)
        for t in np.linspace(0., 1., count):
            x, y = a[:2]+t*(b[:2]-a[:2])
            theta = a[2]+t*yaw
            dx, dy = points[:, 0]-x, points[:, 1]-y
            px = math.cos(theta)*dx+math.sin(theta)*dy
            py = -math.sin(theta)*dx+math.cos(theta)*dy
            sx = np.maximum(-.25-px, px-.97)
            sy = np.abs(py)-.4
            distances = (np.hypot(np.maximum(sx, 0), np.maximum(sy, 0))
                         + np.minimum(np.maximum(sx, sy), 0))
            result = min(result, float(distances.min()))
    return result

class UnifiedGeometryTest(unittest.TestCase):
    def test_front_wall_turns_both_ways_and_rejects_blocked_sweep(self):
        from smart_wheelchair_safety.unified_geometry import front_wall_reference, reference_is_clear
        wall = np.column_stack((np.full(101, 2.), np.linspace(-4., 4., 101)))
        for side in (-1., 1.):
            result = front_wall_reference(extract_lines(wall), wall, .5, side*.3)
            self.assertIsNotNone(result)
            path, mode, wall_side = result
            self.assertEqual(mode, 'wall')
            self.assertEqual(wall_side, -side)
            self.assertGreater(side*path[-1, 1], 1.)
            self.assertTrue(reference_is_clear(path, wall))
            blocked = np.vstack((wall, path[len(path)//2, :2]))
            self.assertFalse(reference_is_clear(path, blocked))
        boxed = np.vstack((wall, [[.8, 0.]]))
        self.assertIsNone(front_wall_reference(extract_lines(wall), boxed, .5, 0.))

    def test_captured_nw_door_has_one_supported_feasible_aperture(self):
        capture = json.loads((Path(__file__).with_name('fixtures') /
                              'nw_door_wait_capture.json').read_text())
        lines = [line for group in capture['groups'] for line in extract_opening_lines(group)]
        for ordered in (lines, lines[::-1]):
            doors = find_openings(ordered, max_width=1.5)
            self.assertEqual(len(doors), 1)
            self.assertLess(doors[0].width, 1.23)
            path = collision_aware_door_reference(doors[0], capture['points'])
            self.assertIsNotNone(path)
            for pose in path:
                from smart_wheelchair_safety.unified_geometry import footprint_clearance
                self.assertGreater(np.min(footprint_clearance(
                    transform_points(capture['points'], pose, inverse=True))), .04)

    def test_foreground_occlusion_does_not_make_a_background_wall_door(self):
        capture = json.loads((Path(__file__).with_name('fixtures') /
                              'outer_wall_occlusion_capture.json').read_text())
        for side in (-1., 1.):
            groups = [np.asarray(group)*[1., side] for group in capture['groups']]
            lines = [line for group in groups for line in extract_opening_lines(group)]
            for ordered in (lines, lines[::-1]):
                with self.subTest(side=side, reversed=ordered is not lines):
                    openings = find_openings(ordered, max_width=1.5)
                    self.assertEqual(len(openings), 1, 'occluded background is not observed free space')
                    self.assertAlmostEqual(openings[0].width, 1.1234767432, places=5)
                    np.testing.assert_allclose(openings[0].center,
                                               [1.7084971777, side*.9689253145], atol=1e-5)

    def test_short_contiguous_jamb_piece_cannot_enlarge_aperture(self):
        lines = [Segment((2., y0), (2., y1), math.pi/2, -2.)
                 for y0, y1 in ((-2., -.55), (.55, .674), (.674, 2.), (.674, 2.))]
        for ordered in (lines, lines[::-1]):
            doors = find_openings(ordered, max_width=1.5)
            self.assertEqual(len(doors), 1)
            self.assertAlmostEqual(doors[0].width, 1.1)
            np.testing.assert_allclose(doors[0].center, [2., 0.], atol=1e-7)

    def test_pruning_rejects_invalid_envelope_parameters(self):
        for name in ('margin', 'reaction', 'deceleration', 'state_age'):
            for value in (math.nan, math.inf, -.1):
                with self.subTest(name=name, value=value):
                    self.assertFalse(braking_clear([[3., 0.]], [.03, 0.], [0., 0.], **{name: value}))
        self.assertFalse(braking_clear([[3., 0.]], [.03, 0.], [0., 0.], deceleration=0.))

    def test_normal_creep_does_not_roll_out_unreachable_distant_points(self):
        import smart_wheelchair_safety.unified_geometry as geometry
        points = np.column_stack((np.full(1400, 3.), np.linspace(-2., 2., 1400)))
        with patch.object(geometry, '_braking_envelope', wraps=geometry._braking_envelope) as envelope:
            self.assertTrue(braking_clear(points, [.05, .01], [.03, 0.]))
        envelope.assert_not_called()

    def test_recovery_rejects_nonfinite_motion_before_pruning(self):
        for value in (math.nan, math.inf, -math.inf):
            self.assertFalse(braking_clear([[3., 0.]], [value, 0.], [0., 0.], recovery=True))
            self.assertFalse(braking_clear([[3., 0.]], [0., 0.], [0., value], recovery=True))

    def test_recovery_does_not_roll_out_unreachable_distant_scan_points(self):
        import smart_wheelchair_safety.unified_geometry as geometry
        points = np.vstack(([[1., 0.]], np.column_stack((np.full(1400, 3.), np.linspace(-2., 2., 1400)))))
        with patch.object(geometry, '_braking_envelope', wraps=geometry._braking_envelope) as envelope:
            self.assertTrue(braking_clear(points, [-.03, 0.], [0., 0.], recovery=True))
        self.assertTrue(all(len(call.args[0]) == 1 for call in envelope.call_args_list))

    def test_recovery_pruning_matches_complete_rollouts(self):
        import smart_wheelchair_safety.unified_geometry as geometry
        random = np.random.default_rng(731)
        accepted = 0
        for index in range(160):
            command = random.uniform([- .05, -.1], [.05, .1])
            measured = np.zeros(2) if index % 2 else random.uniform([-.055, -.11], [.055, .11])
            points = random.uniform([-4., -4.], [4., 4.], (300, 2))
            points = points[geometry.footprint_clearance(points) > .01]
            kwargs = dict(recovery=True, reaction=random.uniform(.05, .5),
                          deceleration=random.uniform(.2, 1.), state_age=random.uniform(0., .1))
            actual = braking_clear(points, command, measured, **kwargs)
            with patch.object(geometry, '_reachable_points', side_effect=lambda p, *args: p):
                expected = braking_clear(points, command, measured, **kwargs)
            self.assertEqual(actual, expected, index)
            accepted += expected
        self.assertGreater(accepted, 10)

    def test_normal_pruning_matches_complete_rollouts(self):
        import smart_wheelchair_safety.unified_geometry as geometry
        random = np.random.default_rng(903)
        accepted = 0
        for index in range(160):
            scale = .1 if index % 2 else 1.
            command = scale*random.uniform([-.4, -.65], [.8, .65])
            measured = scale*random.uniform([-.4, -.65], [.8, .65])
            points = random.uniform([-4., -4.], [4., 4.], (300, 2))
            points = points[geometry.footprint_clearance(points) > .01]
            kwargs = dict(reaction=random.uniform(.05, .5), deceleration=random.uniform(.2, 1.),
                          state_age=random.uniform(0., .3))
            actual = braking_clear(points, command, measured, **kwargs)
            with patch.object(geometry, '_reachable_points', side_effect=lambda p, *args: p):
                expected = braking_clear(points, command, measured, **kwargs)
            self.assertEqual(actual, expected, index)
            accepted += expected
        self.assertGreater(accepted, 10)

    def test_recovery_oblique_wall_matrix_checks_both_body_ends(self):
        for degrees in (-75, -60, -45, -30, -15, 0, 15, 30, 45, 60, 75, 120, 150, 180, 210, 240):
            angle = math.radians(degrees)
            normal = np.array([math.cos(angle), math.sin(angle)])
            tangent = np.array([-normal[1], normal[0]])
            corner = np.array([.97 if normal[0] >= 0. else -.25,
                               .4 if normal[1] >= 0. else -.4])
            points = corner + .02*normal + np.linspace(-2., 2., 801)[:, None]*tangent
            away = -.03*math.copysign(1., normal[0])
            with self.subTest(degrees=degrees):
                self.assertTrue(braking_clear(points, [away, 0.], [0., 0.], recovery=True))
                self.assertFalse(braking_clear(points, [-away, 0.], [0., 0.], recovery=True))

    def test_recovery_does_not_exempt_points_in_age_uncertainty_padding(self):
        self.assertFalse(braking_clear([[0., .4401]], [.05, 0.], [.05, 0.],
                                       state_age=.1, recovery=True))

    def test_recovery_can_leave_stop_infeasible_uncertainty_boundary(self):
        points, measured = [[1.01003, 0.]], [.00013, .00024]
        self.assertFalse(braking_clear(points, [0., 0.], measured, state_age=.02))
        self.assertTrue(braking_clear(points, [-.02, 0.], measured,
                                      state_age=.02, recovery=True))
        self.assertFalse(braking_clear(points, [.02, 0.], measured,
                                       state_age=.02, recovery=True))

    def test_uncertainty_boundary_recovery_cannot_threaten_other_points(self):
        self.assertFalse(braking_clear([[1.01003, 0.], [-.2901, 0.]], [-.02, 0.],
                                       [.00013, .00024], state_age=.02, recovery=True))

    def test_uncertainty_boundary_requires_resolving_not_merely_reducing_deficit(self):
        self.assertFalse(braking_clear([[1.01003, 0.]], [-.000001, 0.],
                                       [.00013, .00024], state_age=.02, recovery=True))

    def test_recovery_rejects_stale_state_and_cannot_rotate_between_two_jambs(self):
        self.assertFalse(braking_clear([[1., 0.]], [-.03, 0.], [0., 0.],
                                       state_age=.101, recovery=True))
        points = [[.9, .42], [.9, -.42], [-.2, .42], [-.2, -.42]]
        for turn in (-.1, .1):
            self.assertFalse(braking_clear(points, [0., turn], [0., 0.], recovery=True))

    def test_near_wall_recovery_separates_but_never_pushes_inward(self):
        for point, away in (((1., 0.), -.05), ((-.28, 0.), .05)):
            with self.subTest(point=point):
                self.assertFalse(braking_clear([point], [away, 0.], [0., 0.]))
                self.assertTrue(braking_clear([point], [away, 0.], [0., 0.], recovery=True))
                self.assertFalse(braking_clear([point], [-away, 0.], [0., 0.], recovery=True))

    def test_recovery_does_not_lock_when_stop_drift_crosses_margin_boundary(self):
        points = [[1.00999, 0.], [1.01001, .2]]
        measured = [.00013, .00024]
        self.assertTrue(braking_clear(points, [-.02, 0.], measured, state_age=.02, recovery=True))
        self.assertFalse(braking_clear(points, [.02, 0.], measured, state_age=.02, recovery=True))
        self.assertFalse(braking_clear(points, [-.02, 0.], measured, state_age=.02))

    def test_stop_drift_recovery_cannot_approach_a_new_rear_obstacle(self):
        points = [[1.00999, 0.], [1.01001, .2], [-.2901, 0.]]
        self.assertFalse(braking_clear(points, [-.02, 0.], [.00013, .00024],
                                       state_age=.02, recovery=True))

    def test_recovery_never_increases_existing_footprint_penetration(self):
        self.assertTrue(braking_clear([[.96, 0.]], [-.05, 0.], [0., 0.], recovery=True))
        self.assertFalse(braking_clear([[.96, 0.]], [.05, 0.], [0., 0.], recovery=True))

    def test_recovery_retains_speed_and_other_obstacle_guards(self):
        self.assertFalse(braking_clear([[1., 0.]], [-.1, 0.], [0., 0.], recovery=True))
        self.assertFalse(braking_clear([[1., 0.]], [-.05, 0.], [.3, 0.], recovery=True))
        self.assertFalse(braking_clear([[1., 0.], [-.291, 0.]], [-.05, 0.], [0., 0.], recovery=True))
        self.assertFalse(braking_clear([[0., .42], [-.24, .41]], [0., .1], [0., 0.], recovery=True))

    def test_captured_near_wall_lock_can_retreat_at_creep_speed(self):
        capture = json.loads((Path(__file__).with_name('fixtures') /
                              'near_wall_lock_capture.json').read_text())
        self.assertFalse(braking_clear(capture['points'], [-.05, 0.], capture['measured'],
                                       state_age=capture['state_age']))
        self.assertTrue(braking_clear(capture['points'], [-.02, 0.], capture['measured'],
                                      state_age=capture['state_age'], recovery=True))
        self.assertFalse(braking_clear(capture['points'], [.05, 0.], capture['measured'],
                                       state_age=capture['state_age'], recovery=True))

    def test_oblique_door_can_stage_with_collision_checked_stationary_turns(self):
        for degrees in (-45, -30, 30, 45):
            yaw = math.radians(degrees)
            axle = (-1.35, -1.35*math.tan(yaw), yaw)
            points = np.array([(x, y) for x in (-.06, .06)
                               for y in np.r_[np.linspace(-4., -.5, 180), np.linspace(.5, 4., 180)]])
            # Opposite side of the real 2.58 m cross corridor.
            points = np.vstack((points, np.column_stack((np.full(300, -2.64), np.linspace(-4., 4., 300)))))
            local = transform_points(points, axle, inverse=True)
            center = transform_points([(0., 0.)], axle, inverse=True)[0]
            with self.subTest(degrees=degrees):
                path = collision_aware_door_reference(Door(tuple(center), -yaw, 1.), local)
                self.assertIsNotNone(path, 'a clear staging turn must not be reported as no path')
                for x, y, heading in path:
                    obstacle = transform_points(local, (x, y, heading), inverse=True)
                    dx = np.maximum(np.maximum(-.25-obstacle[:, 0], obstacle[:, 0]-.97), 0.)
                    dy = np.maximum(np.abs(obstacle[:, 1])-.4, 0.)
                    self.assertGreater(np.hypot(dx, dy).min(), .04)

    def test_captured_wall_guard_preserves_state_age_uncertainty(self):
        capture = json.loads((Path(__file__).with_name('fixtures') /
                              'wall_guard_capture.json').read_text())
        self.assertEqual(capture['reason'], 'emergency_stop')
        self.assertEqual(len(capture['candidates']), 6)
        for candidate in capture['candidates']:
            with self.subTest(command=candidate['command']):
                self.assertFalse(braking_clear(
                    capture['points'], candidate['command'], candidate['measured'],
                    margin=candidate['margin'], reaction=candidate['reaction'],
                    deceleration=candidate['deceleration'], state_age=candidate['state_age']))
        # Counterfactual diagnosis only: removing acquisition age would accept
        # this nominally clear stop. Production must retain the measured age.
        zero = capture['candidates'][-1]
        self.assertTrue(braking_clear(capture['points'], zero['command'],
                                      zero['measured'], state_age=0.))

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

    def test_corner_bounded_wide_opening_is_detected_on_both_sides(self):
        for side in (-1., 1.):
            wall = Segment((.3, side*.52), (2.44, side*.52), 0., side*.52)
            boundary = Segment((5.107, side*.939), (5.154, side*.55),
                               -side*1.451, -side*5.14)

            openings = find_openings([wall, boundary])

            self.assertEqual(len(openings), 1)
            self.assertAlmostEqual(openings[0].center[0], 3.80, places=2)
            self.assertAlmostEqual(openings[0].center[1], side*.52, places=2)
            self.assertAlmostEqual(openings[0].heading, side*math.pi/2, places=2)
            self.assertAlmostEqual(openings[0].width, 2.72, places=2)

    def test_corner_bounded_opening_requires_boundary_to_reach_wall(self):
        wall = Segment((.3, .52), (2.44, .52), 0., .52)
        disconnected = Segment((5.14, 1.), (5.14, 1.5), math.pi/2, -5.14)

        self.assertEqual(find_openings([wall, disconnected]), [])

    def test_corner_boundary_tolerance_is_measured_in_metres(self):
        wall = Segment((.3, .52), (2.44, .52), 0., .52)
        nine_cm_short = Segment((5.14, .61), (5.14, 2.61), math.pi/2, -5.14)

        self.assertEqual(find_openings([wall, nine_cm_short]), [])

    def test_corner_bounded_opening_rejects_occupied_gap(self):
        wall = Segment((.3, .52), (2.44, .52), 0., .52)
        occupied = Segment((3., .52), (4., .52), 0., .52)
        boundary = Segment((5.14, .55), (5.14, .95), math.pi/2, -5.14)

        self.assertEqual(find_openings([wall, occupied, boundary]), [])

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

    def test_far_door_cannot_capture_arc_that_turns_away_before_its_plane(self):
        # Measured wall-following regression: the closest arc pose is still
        # 2.28 m short of the door plane, then recedes instead of entering it.
        for side in (-1., 1.):
            door = Opening((4.08219, side*.75047), -side*.28981, 1.3434, ())
            with self.subTest(side=side):
                self.assertIsNone(intended_front_door(
                    [door], .83333, side*.42, corridor_tolerance=.60))

    def test_arc_may_turn_away_after_crossing_the_front_door_plane(self):
        for side in (-1., 1.):
            door = Opening((2.9, side*1.), side*.8, 1.4, ())
            with self.subTest(side=side):
                self.assertEqual(intended_front_door(
                    [door], 1.2, side*.4, corridor_tolerance=.60), door)

    def test_crossing_door_plane_outside_aperture_is_not_entry_intent(self):
        # This previously passed on corridor tolerance although the arc
        # crosses 1.17 m off-center, outside the 0.70 m half-aperture.
        for side in (-1., 1.):
            door = Opening((3.6, side*1.), side*.8, 1.4, ())
            with self.subTest(side=side):
                self.assertIsNone(intended_front_door(
                    [door], 1.2, side*.4, corridor_tolerance=.60))

    def test_oversteered_arc_cannot_capture_a_door_beyond_its_circle(self):
        # This formerly accepted target caused a false capture: the persistent
        # joystick circle falls over one metre short of the door plane.
        door = Opening((3.3168, 1.1490), .4428, 1.3008, ())

        selected = intended_front_door([door], .8977, .5182, corridor_tolerance=.60)

        self.assertIsNone(selected)

    def test_wall_following_cannot_capture_unreachable_door_at_horizon_boundary(self):
        for side in (-1., 1.):
            for x in (3.49, 3.4907612926, 3.51):
                door = Opening((x, side*1.3861148429), -side*.1284125524, 1.4081023897)
                with self.subTest(side=side, x=x):
                    self.assertIsNone(intended_front_door(
                        [door], .83333335, side*.42, corridor_tolerance=.60))

    def test_side_front_door_is_acquired_at_lidar_planning_range(self):
        door = Opening((3.75, -.27), math.radians(41.2), 1.23, ())

        self.assertEqual(intended_front_door([door], .5, 0., .35), door)

    def test_captured_departing_arc_cannot_acquire_the_same_door(self):
        from smart_wheelchair_safety.unified_geometry import active_aperture_targeted
        capture = json.loads((Path(__file__).with_name('fixtures') /
                              'wall_door_recapture.json').read_text())
        value = capture['door']
        for side in (-1., 1.):
            door = Opening((value['center'][0], side*value['center'][1]),
                           side*value['heading'], value['width'])
            v, w = capture['raw'][0], side*capture['raw'][1]
            with self.subTest(side=side):
                self.assertIs(active_aperture_targeted(door, v, w), False)
                self.assertIsNone(intended_front_door([door], v, w, .60))

    def test_captured_arc_must_be_able_to_clear_the_rear_after_entry(self):
        from smart_wheelchair_safety.unified_geometry import active_aperture_targeted
        capture = json.loads((Path(__file__).with_name('fixtures') /
                              'wall_incomplete_door_arc.json').read_text())
        for case in capture['cases']:
            value = case['door']
            for side in (-1., 1.):
                door = Opening((value['center'][0], side*value['center'][1]),
                               side*value['heading'], value['width'])
                v, w = case['raw'][0], side*case['raw'][1]
                with self.subTest(t=case['t'], side=side):
                    self.assertIs(active_aperture_targeted(door, v, w), False)
                    self.assertIsNone(intended_front_door([door], v, w, .60))

    def test_incomplete_door_arc_keeps_only_reachable_rear_clearance(self):
        from smart_wheelchair_safety.unified_geometry import active_aperture_targeted
        v, w = .83333335, .42
        for side in (-1., 1.):
            for penetration, expected in ((.28, False), (.30, True)):
                door = Opening((v/w-penetration, side*1.2), 0., 1.48)
                with self.subTest(side=side, penetration=penetration):
                    self.assertIs(active_aperture_targeted(door, v, side*w), expected)
                    self.assertEqual(intended_front_door([door], v, side*w, .60),
                                     door if expected else None)

    def test_wall_capture_cannot_hide_a_side_front_narrow_door(self):
        door = Opening((1.63, 1.20), math.radians(69.), 1.22, ())

        self.assertEqual(intended_front_door([door], .3, 0., .35), door)

    def test_pure_side_door_is_not_front_door_intent(self):
        door = Opening((1.2, 1.2), math.radians(80.), 1.2, ())

        self.assertIsNone(intended_front_door([door], .5, 0., .35))

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

    def test_short_jamb_fragments_prevent_oblique_door_width_inflation(self):
        points = np.vstack((
            np.column_stack((np.full(30, 2.), np.linspace(-3., -.6, 30))),
            np.column_stack((np.full(5, 2.), np.linspace(.6, .78, 5))),
            np.column_stack((np.full(15, 2.), np.linspace(1.08, 2., 15))),
        ))

        coarse = find_openings(extract_lines(points))
        detailed = find_openings(extract_opening_lines(points))

        self.assertAlmostEqual(coarse[0].width, 1.68, places=2)
        self.assertAlmostEqual(detailed[0].width, 1.2, places=2)

    def test_oblique_door_alignment_curves_into_and_through_centerline(self):
        door = Opening((2.32, .34), math.radians(45.), 1.22, ())
        path = door_alignment_reference(door, 'align')
        normal = np.array([math.cos(door.heading), math.sin(door.heading)])
        tangent = np.array([-normal[1], normal[0]])
        relative = path[:, :2] - door.center
        centerline = relative @ normal >= .35
        distance = np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1)
        heading_change = np.abs(np.arctan2(np.sin(np.diff(path[:, 2])),
                                           np.cos(np.diff(path[:, 2]))))

        np.testing.assert_allclose(path[0], [0., 0., 0.], atol=1e-9)
        self.assertGreater(relative[-1] @ normal, 1.4)
        self.assertGreaterEqual(np.min(path[:, 2]), -1e-9)
        self.assertLessEqual(np.max(path[:, 2]), door.heading+.01)
        self.assertLess(np.max(heading_change/distance), 1.5)
        np.testing.assert_allclose(relative[centerline] @ tangent, 0., atol=1e-9)
        np.testing.assert_allclose(path[centerline, 2], door.heading, atol=.06)
        self.assertTrue(np.all(np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1) > 0.))

    def test_collision_aware_door_path_clears_current_door_jambs(self):
        door = Opening((3.3168, 1.1490), .4428, 1.3008, ())
        obstacles = np.array([
            [.659, .488], [.688, .489], [.712, .497], [.729, .528],
            [1.056, -.417], [1.080, -.409], [1.104, -.402],
            [1.151, -.386], [1.178, -.719], [1.255, -.878],
            [1.347, -1.085], [1.455, -1.339],
        ])

        path = collision_aware_door_reference(door, obstacles)

        self.assertIsNotNone(path)
        normal = np.array([math.cos(door.heading), math.sin(door.heading)])
        tangent = np.array([-normal[1], normal[0]])
        relative = path[:, :2] - door.center
        self.assertGreater(relative[-1] @ normal, 1.4)
        self.assertLess(abs(relative[-1] @ tangent), .03)
        for x, y, yaw in path:
            local = transform_points(obstacles, (x, y, yaw), inverse=True)
            collision = ((local[:, 0] >= -.29) & (local[:, 0] <= 1.01)
                         & (np.abs(local[:, 1]) <= .44))
            self.assertFalse(np.any(collision), (x, y, yaw, local[collision]))

    def test_collision_aware_path_merges_gradually_into_oblique_door(self):
        door = Opening((3.4679, -.3507), .3981, 1.0038, ())
        segments = [
            ((3.6640, -.8127), (4.3395, -2.4189)),
            ((3.6617, -.8137), (3.7548, -.7608)),
            ((2.2651, 2.4902), (3.2724, .1093)),
        ]
        obstacles = np.vstack([
            np.linspace(start, end, max(3, int(np.linalg.norm(np.subtract(end, start))/.02)))
            for start, end in segments
        ])

        path = collision_aware_door_reference(door, obstacles)

        self.assertIsNotNone(path)
        for x, y, yaw in path:
            local = transform_points(obstacles, (x, y, yaw), inverse=True)
            outside_x = np.maximum(np.maximum(-.25-local[:, 0], local[:, 0]-.97), 0.)
            outside_y = np.maximum(np.abs(local[:, 1])-.40, 0.)
            self.assertGreater(np.min(np.hypot(outside_x, outside_y)), .04)

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

    def test_wall_reference_explicit_turn_uses_same_side_wall(self):
        lines = [
            Segment((0., .9), (4., .9), 0., .9),
            Segment((0., -.55), (4., -.55), 0., -.55),
        ]
        for angular, expected_side in ((.12, 1), (-.12, -1)):
            with self.subTest(angular=angular):
                _, mode, side = wall_reference(lines, .6, angular)
                self.assertEqual(mode, 'wall')
                self.assertEqual(side, expected_side)

    def test_wall_reference_straight_keeps_preferred_entry_side(self):
        lines = [
            Segment((0., .9), (4., .9), 0., .9),
            Segment((0., -.55), (4., -.55), 0., -.55),
        ]

        _, mode, side = wall_reference(lines, .6, 0., preferred_side=1)

        self.assertEqual(mode, 'wall')
        self.assertEqual(side, 1)

    def test_wall_reference_missing_preferred_side_does_not_swap_walls(self):
        lines = [Segment((0., -.55), (4., -.55), 0., -.55)]

        path, mode, side = wall_reference(lines, .6, 0., preferred_side=1)

        np.testing.assert_allclose(path, arc_path(.6, 0.))
        self.assertEqual((mode, side), ('manual', 0))

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

    def test_curvature_reversal_preserves_initial_front_and_rear_sweeps(self):
        # Steering against measured yaw cannot erase the initial sweep. Check
        # both travel directions and their reflected left/right counterparts.
        for side in (-1., 1.):
            with self.subTest(side=side):
                self.assertFalse(braking_clear([(1.1, -side*.5)],
                                               (.4, side*.6), (.4, -side*.6)))
                self.assertTrue(braking_clear([(1.1, side*.5)],
                                              (.4, side*.6), (.4, -side*.6)))
                self.assertFalse(braking_clear([(-.4, side*.5)],
                                               (-.3, side*.6), (-.3, -side*.6)))
                self.assertTrue(braking_clear([(-.4, -side*.5)],
                                              (-.3, side*.6), (-.3, -side*.6)))

    def test_stale_pose_includes_unreported_motion_before_braking(self):
        self.assertTrue(braking_clear([(1.95, 0.)], (.8, 0.), (.8, 0.)))
        self.assertFalse(braking_clear([(1.95, 0.)], (.8, 0.), (.8, 0.), state_age=.35))

    def test_stale_uncertainty_applies_to_pure_translation(self):
        self.assertTrue(braking_clear([(1.30, 0.)], (.4, 0.), (.4, 0.)))
        self.assertFalse(braking_clear([(1.30, 0.)], (.4, 0.), (.4, 0.), state_age=.05))

    def test_stationary_pose_age_does_not_invent_full_acceleration(self):
        obstacle = [(1.014, .443)]

        self.assertTrue(braking_clear(obstacle, (0., 0.), (0., 0.), state_age=.02))
        self.assertTrue(braking_clear(obstacle, (.02, 0.), (0., 0.), state_age=.02))

    def test_braking_margin_uses_radial_distance_at_footprint_corners(self):
        outside_round_corner = [(.97+.036, .40+.036)]
        inside_round_corner = [(.97+.025, .40+.025)]

        self.assertTrue(braking_clear(outside_round_corner, (0., 0.), (0., 0.)))
        self.assertFalse(braking_clear(inside_round_corner, (0., 0.), (0., 0.)))

    def test_margin_preserving_corner_escape_is_allowed(self):
        door_jamb = [(.9812854577, -.4383768388)]

        self.assertTrue(braking_clear(door_jamb, (.05, .0375), (0., 0.)))
        self.assertFalse(braking_clear(door_jamb, (.05, -.0375), (0., 0.)))

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

    def test_four_recorded_offset_failures_use_observed_inner_edges(self):
        fixture = Path(__file__).with_name('fixtures')/'observed_door_inner_edges.json'
        for row in json.loads(fixture.read_text()):
            with self.subTest(case=row['case']):
                opening = g.Opening(**row['opening'])
                refined = g.refine_opening(opening, np.array(row['points']))
                world = g.transform_points([refined.center], row['pose'])[0]
                self.assertLess(refined.width, opening.width-.07)
                self.assertAlmostEqual(refined.width, 1., delta=.002)
                self.assertAlmostEqual(world[0], row['expected_world_x'], delta=.005)
                self.assertEqual(refined.heading, opening.heading)

    def test_single_side_support_only_shrinks_observed_edge(self):
        opening = g.Opening((1., 0.), 0., 1.1, ())
        points = np.array([[1., .45], [1.03, .45], [1.07, .45]])
        refined = g.refine_opening(opening, points)
        self.assertAlmostEqual(refined.width, 1.)
        self.assertAlmostEqual(refined.center[1], -.05)
        self.assertAlmostEqual(refined.center[1]-refined.width/2, -.55)

    def test_no_widen_isolated_noise_far_plane_or_insufficient_support(self):
        opening = g.Opening((1., 0.), 0., 1.1, ())
        for points in ([], [[1., .2]], [[1., .45], [1.03, .45]],
                       [[1., .45]]*3,
                       [[1., .65], [1.03, .65], [1.07, .65]],
                       [[1.5, .45], [1.53, .45], [1.57, .45]]):
            with self.subTest(points=points):
                self.assertEqual(g.refine_opening(opening, np.array(points)), opening)

    def test_side_passages_and_wide_openings_are_unchanged(self):
        points = np.array([[1., .45], [1.03, .45], [1.07, .45]])
        for heading, width in ((np.pi/2, 1.1), (0., 2.)):
            opening = g.Opening((1., 0.), heading, width, ())
            self.assertEqual(g.refine_opening(opening, points), opening)

    def test_offset_door_direct_path_reserves_full_swept_clearance(self):
        cases = json.loads((Path(__file__).with_name('fixtures') /
                            'offset_door_staging.json').read_text())
        for row in cases:
            with self.subTest(case=row['case']):
                points = np.array(row['points'])
                door = Opening(**row['door'])
                direct = door_alignment_reference(door, 'align')
                self.assertLess(dense_door_clearance(direct, points), .05)
                with patch.object(g.heapq, 'heappop',
                                  side_effect=AssertionError('unexpected lattice search')):
                    route = collision_aware_door_reference(door, points)
                self.assertIsNotNone(route)
                self.assertGreater(dense_door_clearance(route, points), .06)


    def test_observed_near_door_wins_over_more_centered_far_door(self):
        records = json.loads((Path(__file__).with_name('fixtures')/'offset_door_selection.json').read_text())
        for record in records:
            for mirror in (1., -1.):
                doors = [g.Opening((o['center_local'][0], mirror*o['center_local'][1]),
                                          mirror*o['heading'], o['width'])
                         for o in record['openings']]
                near, far = doors
                with self.subTest(case=record['case'], mirror=mirror):
                    for door in doors:
                        self.assertIs(g.intended_front_door([door], 2./3., 0.), door)
                    for order in (doors, doors[::-1]):
                        self.assertIs(g.intended_front_door(order, 2./3., 0.), near)

    def test_near_door_without_crossing_intent_does_not_hide_far_door(self):
        for mirror in (1., -1.):
            near = g.Opening((1.3, mirror*.7), 0., 1.1)
            far = g.Opening((4., 0.), 0., 1.1)
            with self.subTest(mirror=mirror):
                # Near passes broad corridor matching but misses the aperture.
                self.assertIs(g.active_aperture_targeted(near, 2./3., 0.), False)
                self.assertIsNone(g.intended_front_door([near], 2./3., 0.))
                for order in ([near, far], [far, near]):
                    self.assertIs(g.intended_front_door(order, 2./3., 0.), far)

    def test_recorded_inward_update_matches_symmetrically(self):
        for name in ('v10_far_opening_update.json', 'v10_se_far_opening_update.json'):
            with self.subTest(fixture=name):
                row = json.loads((Path(__file__).with_name('fixtures')/name).read_text())
                old, new = Opening(**row['tracked']), Opening(**row['observed'])
                self.assertGreater(old.width-new.width, .25)
                self.assertTrue(opening_matches(old, new))
                self.assertTrue(opening_matches(new, old))

    def test_large_width_change_requires_stable_contained_narrow_aperture(self):
        old = Opening((2., 0.), 0., 1.4)
        for new in (Opening((2.031, 0.), 0., 1.), Opening((2., 0.), .031, 1.),
                    Opening((2., .23), 0., 1.), Opening((2., 0.), 0., .9),
                    Opening((2., 0.), 0., 1.7)):
            with self.subTest(new=new):
                self.assertFalse(opening_matches(old, new))
                self.assertFalse(opening_matches(new, old))
        self.assertTrue(opening_matches(old, Opening((2., .1), 0., 1.)))

    def test_existing_small_width_matching_is_unchanged(self):
        old = Opening((2., 0.), 0., 2.)
        self.assertTrue(opening_matches(old, Opening((2.1, 0.), .1, 2.2)))
        self.assertFalse(opening_matches(old, Opening((2.3, 0.), .1, 2.2)))
        self.assertFalse(opening_matches(old, Opening((float('nan'), 0.), .1, 2.2)))


class CrossingTests(unittest.TestCase):

    def test_recorded_turn270_raw_hits_but_actual_arc_misses(self):
        for mirror in (-1.0, 1.0):
            door = g.Opening((2.2982257435645965, mirror * 0.5269919203496637), mirror * 1.236335296800013, 1.1227339375132526)
            self.assertTrue(g.output_arc_targets_aperture(door, 1.6666667, mirror * 0.63))
            self.assertFalse(g.output_arc_targets_aperture(door, 0.6998741735441372, mirror * 0.029708163281233484))

    def test_straight_distant_door_has_no_time_horizon(self):
        for distance in (1.3, 4.0, 40.0):
            self.assertTrue(g.output_arc_targets_aperture(g.Opening((distance, 0.1), 0.0, 1.1), 0.2, 0.0))
        self.assertFalse(g.output_arc_targets_aperture(g.Opening((4.0, 1.0), 0.0, 1.1), 0.2, 0.0))
        self.assertFalse(g.output_arc_targets_aperture(g.Opening((4.0, 0.0), 0.0, 1.1), -0.2, 0.0))

    def test_first_crossing_is_used_not_later_reverse_crossing(self):
        for mirror in (-1.0, 1.0):
            self.assertTrue(g.output_arc_targets_aperture(g.Opening((1.0, mirror * (2 - math.sqrt(3))), 0.0, 1.1), 1.0, mirror * 0.5))
            self.assertFalse(g.output_arc_targets_aperture(g.Opening((1.0, mirror * (2 + math.sqrt(3))), 0.0, 1.1), 1.0, mirror * 0.5))
            self.assertFalse(g.output_arc_targets_aperture(g.Opening((2.0, mirror * 2.0), 0.0, 1.1), 1.0, mirror * 0.5))
            self.assertFalse(g.output_arc_targets_aperture(g.Opening((2.1, mirror * 2.0), 0.0, 1.1), 1.0, mirror * 0.5))

    def test_incomplete_three_second_arc_can_still_target_far_door(self):
        door = g.Opening((1.0, 2 - math.sqrt(3)), 0.0, 1.1)
        self.assertTrue(g.output_arc_targets_aperture(door, 0.1, 0.05))

    def test_nonfinite_inputs_are_rejected_before_trigonometry(self):
        for value in (float('inf'), float('-inf'), float('nan')):
            for door, v, w, tolerance in ((g.Opening((1., 0.), value, 1.), .2, .1, .25),
                                          (g.Opening((value, 0.), 0., 1.), .2, .1, .25),
                                          (g.Opening((1., 0.), 0., value), .2, .1, .25),
                                          (g.Opening((1., 0.), 0., 1.), value, .1, .25),
                                          (g.Opening((1., 0.), 0., 1.), .2, value, .25),
                                          (g.Opening((1., 0.), 0., 1.), .2, .1, value)):
                self.assertFalse(g.output_arc_targets_aperture(door, v, w, tolerance))

    def test_circle_cannot_recapture_already_crossed_plane(self):
        for mirror in (-1., 1.):
            for x in (-1., 0.):
                self.assertFalse(g.output_arc_targets_aperture(g.Opening((x, mirror*3.7), 0., 1.1), 1., mirror*.5))


if __name__ == '__main__':
    import rostest
    rostest.rosrun('smart_wheelchair_safety', 'unified_geometry', UnifiedGeometryTest)
