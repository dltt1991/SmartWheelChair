import importlib.util
import math
import time
from types import SimpleNamespace
import unittest
import numpy as np


@unittest.skipUnless(importlib.util.find_spec('rclpy'), 'requires ROS')
class UnifiedNodeTest(unittest.TestCase):
    def setUp(self):
        import rclpy
        from smart_wheelchair_safety.unified_control_node import UnifiedControlNode
        rclpy.init(args=['--ros-args', '-r', '__ns:=/unified_regression'])
        self.node = UnifiedControlNode()
        self.commands = []
        self.node.pub = SimpleNamespace(publish=self.commands.append)
        self.node.pose = (0., 0., 0.)
        now = time.monotonic()
        self.node.raw_time = self.node.odom_time = self.node.plan_time = now
        self.node.odom_stamp = self.node.get_clock().now().nanoseconds / 1e9
        self.node.scans = {side: (now, np.array([[3., 2.], [3., -2.]])) for side in ('left', 'right')}
        self.node.raw = np.array([.5, 0.])
        self.node.planned = np.array([.5, 0.])
        self.node.mode = 'wall'

    def tearDown(self):
        import rclpy
        self.node.destroy_node()
        rclpy.shutdown()

    def wall_opening(self, side=1):
        from smart_wheelchair_safety.unified_geometry import extract_lines, find_openings
        y = side * .8
        points = np.vstack((
            np.column_stack((np.linspace(-1., .2, 20), np.full(20, y))),
            np.column_stack((np.linspace(2.2, 4., 20), np.full(20, y))),
        ))
        now = time.monotonic()
        self.node.scans = {
            'left': (now, points),
            'right': (now, np.empty((0, 2))),
        }
        return find_openings(extract_lines(points))[0]

    def set_opening_turn(self, side=1):
        from smart_wheelchair_safety.unified_geometry import Opening
        self.node.opening_turn = Opening((1.2, side * .8), side * np.pi / 2, 2., ())
        self.node.opening_turn_side = side
        self.node.opening_turn_heading = 0.
        self.node.opening_turn_time = time.monotonic()
        self.node.wall_side = 0

    def front_opening(self, center=(2., 0.), heading=0., width=1.):
        from smart_wheelchair_safety.unified_geometry import Segment, find_openings
        normal = np.array([math.cos(heading), math.sin(heading)])
        tangent = np.array([-normal[1], normal[0]])
        center = np.asarray(center, dtype=float)
        near = center - width / 2 * tangent
        far = center + width / 2 * tangent
        line_heading = math.atan2(tangent[1], tangent[0])
        line_normal = np.array([-tangent[1], tangent[0]])
        lines = [
            Segment(tuple(near - 1.5 * tangent), tuple(near), line_heading,
                    float(near @ line_normal)),
            Segment(tuple(far), tuple(far + 1.5 * tangent), line_heading,
                    float(far @ line_normal)),
        ]
        return find_openings(lines, max_width=1.5)[0]

    def observe_front_door(self, **kwargs):
        self.node._observe_openings([self.front_opening(**kwargs)])

    def confirm_door(self, **kwargs):
        self.observe_front_door(**kwargs)
        self.observe_front_door(**kwargs)

    def send_raw(self, linear, angular):
        from geometry_msgs.msg import Twist
        msg = Twist()
        msg.linear.x, msg.angular.z = linear, angular
        self.node.on_raw(msg)

    def test_same_side_opening_needs_two_frames_and_blocks_old_wall_reacquisition(self):
        opening = self.wall_opening()
        self.node.raw = np.array([.6, .5])
        self.node.wall_side = 1
        self.node.mode = 'wall'

        self.node._observe_openings([opening])
        self.assertEqual(self.node.confirmed_openings, [])
        self.node._observe_openings([opening])
        self.node.update_reference()

        self.assertEqual(self.node.mode, 'opening_turn')
        self.assertIsNotNone(self.node.opening_turn)
        self.node.update_reference()
        self.assertEqual(self.node.mode, 'opening_turn')

    def test_confirmed_opening_survives_temporary_lidar_occlusion_until_passed(self):
        opening = self.wall_opening(side=-1)
        self.node._observe_openings([opening])
        self.node._observe_openings([opening])

        self.node._observe_openings([])
        self.assertEqual(len(self.node.confirmed_openings), 1)

        self.node.pose = (3., 0., 0.)
        self.node._observe_openings([])
        self.assertEqual(self.node.confirmed_openings, [])

    def test_confirmed_opening_world_position_does_not_walk_with_chained_matches(self):
        self.observe_front_door(center=(2., 0.), width=1.)
        self.observe_front_door(center=(2.1, 0.), width=1.)
        confirmed = self.node.confirmed_openings[0]

        self.observe_front_door(center=(2.25, 0.), width=1.)
        self.observe_front_door(center=(2.4, 0.), width=1.)

        self.assertEqual(self.node.confirmed_openings[0], confirmed)

    def test_opening_confirmation_tolerates_one_missing_lidar_fit(self):
        opening = self.wall_opening(side=-1)

        self.node._observe_openings([opening])
        self.node._observe_openings([])
        self.node._observe_openings([opening])

        self.assertEqual(len(self.node.confirmed_openings), 1)

    def test_stop_reverse_and_away_steering_cancel_opening_turn(self):
        from geometry_msgs.msg import Twist
        for linear, angular in ((0., 0.), (-.2, 0.), (.4, -.3)):
            with self.subTest(command=(linear, angular)):
                self.set_opening_turn()
                msg = Twist()
                msg.linear.x, msg.angular.z = linear, angular
                self.node.on_raw(msg)
                self.assertIsNone(self.node.opening_turn)

    def test_opening_turn_exits_on_heading_pass_timeout_and_stale_input(self):
        cases = ('heading', 'passed', 'timeout', 'stale')
        for case in cases:
            with self.subTest(case=case):
                self.set_opening_turn()
                if case == 'heading':
                    self.node.pose = (0., 0., 1.1)
                elif case == 'passed':
                    self.node.pose = (2.0, 0., 0.)
                elif case == 'timeout':
                    self.node.opening_turn_time -= 6.1
                else:
                    self.node.raw_time = 0.
                self.node.update_reference()
                self.assertIsNone(self.node.opening_turn)

    def test_opening_turn_preserves_clear_driver_steering_when_planner_understeers(self):
        self.set_opening_turn(side=1)
        self.node.mode = 'opening_turn'
        self.node.raw = np.array([.6, .5])
        self.node.planned = np.array([.5, .03])

        for _ in range(12):
            self.node.last_tick -= .1
            self.node.control()

        self.assertGreater(self.commands[-1].angular.z, .25)

    def test_opening_turn_suppresses_planner_yaw_until_front_clears_jamb(self):
        from smart_wheelchair_safety.unified_geometry import Opening
        self.node.opening_turn = Opening((3.5, -.52), -math.pi / 2, 2.7, ())
        self.node.opening_turn_side = -1
        self.node.opening_turn_heading = 0.
        self.node.opening_turn_time = time.monotonic()
        self.node.mode = 'opening_turn'
        self.node.raw = np.array([.6, -.5])
        self.node.planned = np.array([.5, -.08])

        for _ in range(8):
            self.node.last_tick -= .1
            self.node.control()

        self.assertAlmostEqual(self.commands[-1].angular.z, 0.)

    def test_opening_wait_segment_corrects_heading_drift_away_from_wall_end(self):
        from smart_wheelchair_safety.unified_geometry import Opening
        self.node.opening_turn = Opening((3.5, -.52), -math.pi / 2, 2.7, ())
        self.node.opening_turn_side = -1
        self.node.opening_turn_heading = 0.
        self.node.opening_turn_time = time.monotonic()
        self.node.mode = 'opening_turn'
        self.node.pose = (0., 0., -.08)
        self.node.raw = np.array([.6, -.5])
        self.node.planned = np.array([.5, -.08])

        for _ in range(8):
            self.node.last_tick -= .1
            self.node.control()

        self.assertGreater(self.commands[-1].angular.z, 0.)

    def test_opening_turn_reference_enters_wide_corridor_instead_of_turning_early(self):
        from smart_wheelchair_safety.unified_geometry import Opening
        self.node.opening_turn = Opening((1.7, -.52), -math.pi / 2, 2.7, ())
        self.node.opening_turn_side = -1
        self.node.opening_turn_heading = 0.
        self.node.opening_turn_time = time.monotonic()

        path = self.node._opening_turn_path()

        self.assertGreater(path[-1, 0], 1.5)
        self.assertLess(path[-1, 1], -1.)

    def test_opening_turn_reference_stays_straight_until_front_clears_near_jamb(self):
        from smart_wheelchair_safety.unified_geometry import Opening
        self.node.opening_turn = Opening((3.5, -.52), -math.pi / 2, 2.7, ())
        self.node.opening_turn_side = -1
        self.node.opening_turn_heading = 0.
        self.node.opening_turn_time = time.monotonic()

        path = self.node._opening_turn_path()

        np.testing.assert_allclose(path[:, 1], 0.)
        np.testing.assert_allclose(path[:, 2], 0.)

    def test_door_needs_two_observations_before_alignment(self):
        self.observe_front_door(center=(2., .2), heading=.12, width=1.)
        self.node.update_reference()
        self.assertNotEqual(self.node.mode, 'door_align')

        self.observe_front_door(center=(2.02, .18), heading=.10, width=1.02)
        self.node.update_reference()

        self.assertEqual(self.node.mode, 'door_align')

    def test_door_centerline_freezes_after_commit(self):
        self.confirm_door(center=(1., 0.), heading=0., width=1.)
        self.node.pose = (.10, 0., 0.)
        self.node.update_reference()
        self.assertEqual(self.node.mode, 'door_pass')
        frozen = self.node.door

        self.observe_front_door(center=(1.15, .12), heading=.1, width=1.)
        self.node.update_reference()

        self.assertEqual(self.node.door, frozen)
        self.assertEqual(self.node.mode, 'door_pass')

    def test_aligned_door_commits_within_staging_heading_control_band(self):
        self.confirm_door(center=(1.15, 0.), heading=0., width=1.)

        self.node.update_reference()

        self.assertEqual(self.node.mode, 'door_pass')

    def test_door_releases_only_after_rear_body_clears_plane(self):
        self.confirm_door(center=(1., 0.), heading=0., width=1.)
        self.node.door = self.front_opening(center=(1., 0.), heading=0., width=1.)
        self.node.door_phase = 'door_pass'
        self.node.pose = (1.20, 0., 0.)
        self.node.update_reference()
        self.assertEqual(self.node.mode, 'door_clear')

        self.node.pose = (1.30, 0., 0.)
        self.node.update_reference()

        self.assertIsNone(self.node.door)

    def test_stop_and_reverse_cancel_door_immediately(self):
        for command in ((0., 0.), (-.2, 0.)):
            with self.subTest(command=command):
                self.node.door = self.front_opening(center=(1., 0.), heading=0., width=1.)
                self.node.door_phase = 'door_pass'
                self.send_raw(*command)
                self.assertIsNone(self.node.door)

    def test_door_cancels_only_after_sustained_away_steering(self):
        self.node.door = self.front_opening(center=(1.4, .4), heading=.15, width=1.)
        self.node.door_phase = 'door_align'

        self.send_raw(.5, -.5)

        self.assertIsNotNone(self.node.door)
        self.node.door_away_since -= .36
        self.send_raw(.5, -.5)

        self.assertIsNone(self.node.door)
        self.assertTrue(self.node.override)

    def test_door_pass_retains_target_near_plane_despite_acquisition_range(self):
        self.node.door = self.front_opening(center=(.75, 0.), heading=0., width=1.)
        self.node.door_phase = 'door_pass'

        self.send_raw(.5, 0.)

        self.assertIsNotNone(self.node.door)
        self.assertEqual(self.node.door_away_since, 0.)

    def test_committed_door_near_plane_debounces_explicit_away_steering(self):
        for phase in ('door_pass', 'door_clear'):
            with self.subTest(phase=phase):
                self.node.door = self.front_opening(center=(.75, 0.), heading=0., width=1.)
                self.node.door_phase = phase

                self.send_raw(.5, .5)

                self.assertIsNotNone(self.node.door)
                self.assertGreater(self.node.door_away_since, 0.)
                self.node.door_away_since -= .36
                self.send_raw(.5, .5)

                self.assertIsNone(self.node.door)
                self.assertTrue(self.node.override)

    def test_door_pass_cancels_after_sustained_away_steering_before_plane(self):
        self.node.door = self.front_opening(center=(1.4, .4), heading=.15, width=1.)
        self.node.door_phase = 'door_pass'

        self.send_raw(.5, -.5)

        self.assertIsNotNone(self.node.door)
        self.node.door_away_since -= .36
        self.send_raw(.5, -.5)

        self.assertIsNone(self.node.door)
        self.assertTrue(self.node.override)

    def test_stop_clears_uncommitted_opening_observations(self):
        self.confirm_door(center=(2., 0.), heading=0., width=1.)

        self.send_raw(0., 0.)

        self.assertEqual(self.node.opening_candidates, [])
        self.assertEqual(self.node.confirmed_openings, [])

    def test_door_align_uses_heading_feedback_at_staging_instead_of_looping(self):
        self.node.door = self.front_opening(center=(1., 0.), heading=.08, width=1.)
        self.node.door_phase = 'door_align'
        self.node.mode = 'door_align'
        self.node.raw = np.array([.6, 0.])
        self.node.planned = np.array([.2, -.2])

        for _ in range(10):
            self.node.last_tick -= .1
            self.node.control()

        self.assertAlmostEqual(self.commands[-1].linear.x, 0.)
        self.assertGreater(self.commands[-1].angular.z, 0.)

    def test_stale_planner_never_replays_old_motion(self):
        self.node.output = np.array([.4, .1])
        self.node.plan_time = 0.
        self.node.control()
        self.assertEqual(self.commands[-1].linear.x, 0.)
        self.assertEqual(self.commands[-1].angular.z, 0.)

    def test_stale_lidar_or_joystick_stops(self):
        for field in ('raw_time', 'odom_time'):
            setattr(self.node, field, 0.)
            self.node.control()
            self.assertEqual(self.commands[-1].linear.x, 0.)
            setattr(self.node, field, time.monotonic())
        self.node.scans['left'] = (0., np.array([[3., 2.]]))
        self.node.control()
        self.assertEqual(self.commands[-1].linear.x, 0.)

    def test_away_input_marks_immediate_override(self):
        from geometry_msgs.msg import Twist
        self.node.wall_side = 1
        msg = Twist()
        msg.linear.x, msg.angular.z = .5, -.4
        self.node.on_raw(msg)
        self.assertTrue(self.node.override)

    def test_oblique_door_intent_beats_existing_wall_override(self):
        self.confirm_door(center=(1.6, .45), heading=.18, width=1.)
        self.node.wall_side = -1

        self.send_raw(.6, .30)
        self.node.update_reference()

        self.assertFalse(self.node.override)
        self.assertEqual(self.node.mode, 'door_align')

    def test_observed_obstacle_is_not_erased_when_body_reaches_it(self):
        self.node.scans['left'] = (time.monotonic(), np.array([[.9, .1]]))
        _, points = self.node.points()
        self.assertTrue(np.any(np.all(points == [.9, .1], axis=1)))
        self.node.control()
        self.assertEqual(self.commands[-1].linear.x, 0.)

    def test_driver_stop_is_immediate_and_clears_acceleration(self):
        self.node.raw[:] = 0.
        self.node.output[:] = .4
        self.node.acceleration[:] = .3
        self.node.control()
        np.testing.assert_array_equal(self.node.output, [0., 0.])
        np.testing.assert_array_equal(self.node.acceleration, [0., 0.])

    def test_front_stop_does_not_disable_manual_rotation(self):
        self.node.mode = 'front_stop'
        self.node.raw = np.array([0., .4])
        self.node.plan_time = 0.
        self.node.last_tick -= .05
        self.node.control()
        self.assertEqual(self.commands[-1].linear.x, 0.)
        self.assertGreater(self.commands[-1].angular.z, 0.)

    def test_override_remains_active_across_reference_refresh(self):
        from geometry_msgs.msg import Twist
        self.node.wall_side = 1
        msg = Twist()
        msg.linear.x, msg.angular.z = .5, -.4
        self.node.on_raw(msg)
        self.node.update_reference()
        self.node.on_raw(msg)
        self.assertTrue(self.node.override)
        self.assertEqual(self.node.mode, 'override')

    def test_forward_intent_never_accepts_negative_planner_velocity(self):
        self.node.planned = np.array([-.01, .2])
        self.node.last_tick -= .05
        self.node.control()
        self.assertGreaterEqual(self.commands[-1].linear.x, 0.)

    def test_front_stop_survives_temporary_line_fragmentation(self):
        self.node.front_blocked = True
        self.node.update_reference()
        self.assertEqual(self.node.mode, 'front_stop')

    def test_valid_infinite_scan_is_free_space_not_sensor_loss(self):
        from geometry_msgs.msg import TransformStamped
        from sensor_msgs.msg import LaserScan
        transform = TransformStamped()
        transform.transform.rotation.w = 1.
        self.node.tf = SimpleNamespace(lookup_transform=lambda *args: transform)
        scan = LaserScan()
        scan.header.stamp = self.node.get_clock().now().to_msg()
        scan.range_min, scan.range_max = .05, 8.
        scan.ranges = [float('inf')] * 10
        scan.angle_increment = .1
        self.node.scans.clear()
        self.node.on_scan(scan, 'left')
        self.assertIn('left', self.node.scans)
        self.assertEqual(len(self.node.scans['left'][1]), 0)

    def test_corrupt_scan_does_not_refresh_watchdog(self):
        from sensor_msgs.msg import LaserScan
        scan = LaserScan()
        scan.header.stamp = self.node.get_clock().now().to_msg()
        scan.range_min, scan.range_max = .05, 8.
        scan.ranges = [float('nan')] * 10
        self.node.scans.clear()
        self.node.on_scan(scan, 'left')
        self.assertNotIn('left', self.node.scans)

    def test_cancelling_door_assistance_keeps_nearby_jamb_observations(self):
        self.node.door_obstacles = np.array([[.2, .5], [8., .5]])
        self.node.door = None
        self.node.raw[:] = 0.
        self.node.update_reference()
        np.testing.assert_array_equal(self.node.door_obstacles, [[.2, .5]])

    def test_delayed_odometry_does_not_refresh_watchdog(self):
        from nav_msgs.msg import Odometry
        msg = Odometry()
        msg.pose.pose.orientation.w = 1.
        self.node.odom_time = 0.
        self.node.on_odom(msg)
        self.assertEqual(self.node.odom_time, 0.)

    def test_odom_dropout_stops_even_with_fresh_receipt_and_override(self):
        self.node.odom_stamp -= .15
        self.node.override = True
        self.node.output = np.array([.8, .2])
        self.node.control()
        np.testing.assert_array_equal(self.node.output, [0., 0.])
