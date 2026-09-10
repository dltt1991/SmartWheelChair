#!/usr/bin/env python3
import importlib.util
import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch


ROS_AVAILABLE = importlib.util.find_spec("rospy") is not None

if ROS_AVAILABLE:
    import rospy

    rospy.init_node("test_local_path_follower_node", anonymous=True,
                    disable_signals=True)


@unittest.skipUnless(ROS_AVAILABLE, "requires ROS")
class LocalPathFollowerNodeTest(unittest.TestCase):
    def test_fresh_inputs_publish_stamped_plan_and_stale_inputs_stop(self):
        import rospy
        from geometry_msgs.msg import PoseStamped
        from nav_msgs.msg import Odometry, Path
        from sensor_msgs.msg import LaserScan
        from std_msgs.msg import Float32
        from smart_wheelchair_safety.local_path_follower_node import (
            LocalPathFollowerNode,
        )

        now = [10.0]
        with patch.object(rospy, "Timer"):
            node = LocalPathFollowerNode(clock=lambda: now[0])
        messages = []
        node.publisher = SimpleNamespace(publish=messages.append)

        odom = Odometry()
        odom.pose.pose.position.x = 2.0
        odom.pose.pose.position.y = 3.0
        odom.pose.pose.orientation.z = math.sin(math.pi / 4.0)
        odom.pose.pose.orientation.w = math.cos(math.pi / 4.0)
        node.on_odom(odom)

        path = Path()
        for distance in [index * 0.1 for index in range(31)]:
            pose = PoseStamped()
            pose.pose.position.x = 2.0
            pose.pose.position.y = 3.0 + distance
            pose.pose.orientation.z = math.sin(math.pi / 4.0)
            pose.pose.orientation.w = math.cos(math.pi / 4.0)
            path.poses.append(pose)
        node.on_reference(path)

        scan = LaserScan(angle_min=0.0, angle_increment=0.1,
                         ranges=[4.0, float("nan"), float("inf")])
        node.on_scan_left(scan)
        node.on_scan_right(scan)
        node.on_speed_limit(Float32(data=0.8))

        node.publish()

        self.assertGreater(messages[-1].twist.linear.x, 0.2)
        self.assertEqual(messages[-1].header.frame_id, "rear_axle")
        self.assertGreater(messages[-1].header.stamp.to_sec(), 0.0)

        now[0] += 0.251
        node.publish()

        self.assertEqual((messages[-1].twist.linear.x,
                          messages[-1].twist.angular.z), (0.0, 0.0))


if __name__ == "__main__":
    import rostest

    rostest.rosrun("smart_wheelchair_safety", "local_path_follower_node",
                   LocalPathFollowerNodeTest)
