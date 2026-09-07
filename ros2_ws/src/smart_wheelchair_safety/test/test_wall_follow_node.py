import importlib.util
from types import SimpleNamespace
import unittest


@unittest.skipUnless(importlib.util.find_spec('rclpy'), 'requires the ROS environment')
class WallFollowNodeTest(unittest.TestCase):
    def test_driver_override_cancels_the_node_hold(self):
        import rclpy
        from geometry_msgs.msg import Twist
        from smart_wheelchair_safety.wall_follow_assist_node import WallFollowAssistNode

        rclpy.init(args=['--ros-args', '-r', '__ns:=/wall_follow_regression'])
        node = WallFollowAssistNode()
        commands = []
        node._pub = SimpleNamespace(publish=commands.append)
        try:
            for side in (-1, 1):
                for dropout in (False, True):
                    for linear, angular in ((.5, -side * .4), (-.3, 0.), (0., 0.)):
                        with self.subTest(side=side, dropout=dropout, linear=linear):
                            node._controller.reset()
                            points = [(0., side * .8), (1., side * .8)]
                            node._left_points = points if side > 0 else []
                            node._right_points = points if side < 0 else []
                            node._left_scan_time = node._right_scan_time = node.get_clock().now()
                            request = Twist()
                            request.linear.x = .5
                            node._on_cmd_vel(request)
                            self.assertEqual(commands[-1].linear.y, 1.)
                            self.assertTrue(node._can_hold_wall_follow(.5))
                            if dropout:
                                node._left_points = node._right_points = []
                            request.linear.x = linear
                            request.angular.z = angular
                            node._on_cmd_vel(request)
                            self.assertEqual(commands[-1].linear.x, linear)
                            self.assertEqual(commands[-1].angular.z, angular)
                            self.assertEqual(commands[-1].linear.y, 0.)
                            self.assertFalse(node._can_hold_wall_follow(.5))
        finally:
            node.destroy_node()
            rclpy.shutdown()
