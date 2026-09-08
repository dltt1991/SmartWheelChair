import importlib.util
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
