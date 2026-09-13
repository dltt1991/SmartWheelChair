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
    def test_sim_input_receipts_age_on_ros_time_but_hardware_uses_wall(self):
        from geometry_msgs.msg import PoseStamped
        from nav_msgs.msg import Odometry, Path
        from sensor_msgs.msg import LaserScan
        from std_msgs.msg import Float32
        from smart_wheelchair_safety.local_path_follower_node import LocalPathFollowerNode
        now=[100.]
        with patch.object(rospy,'Timer'):
            node=LocalPathFollowerNode(clock=lambda:now[0])
        node.use_sim_time=True
        path=Path();path.header.stamp=rospy.Time.from_sec(10.)
        path.poses=[PoseStamped() for _ in range(31)]
        for index,pose in enumerate(path.poses):
            pose.pose.position.x=index*.1;pose.pose.orientation.w=1.
        odom=Odometry();odom.header.stamp=path.header.stamp;odom.pose.pose.orientation.w=1.
        scan=LaserScan(angle_increment=.1,ranges=[4.,4.,4.]);scan.header.stamp=path.header.stamp
        with patch.object(rospy.Time,'now',return_value=rospy.Time.from_sec(10.)):
            node.on_reference(path);node.on_odom(odom);node.on_speed_limit(Float32(data=.8))
            node.on_scan_left(scan);node.on_scan_right(scan)
        messages=[];node.publisher=SimpleNamespace(publish=messages.append)
        now[0]=100.27
        with patch.object(rospy.Time,'now',return_value=rospy.Time.from_sec(10.2)):node.publish()
        self.assertGreater(messages[-1].twist.linear.x,0.)
        with patch.object(rospy.Time,'now',return_value=rospy.Time.from_sec(10.251)):node.publish()
        self.assertEqual(messages[-1].twist.linear.x,0.)
        node.use_sim_time=False
        with patch.object(rospy.Time,'now',return_value=rospy.Time.from_sec(10.2)):node.publish()
        self.assertEqual(messages[-1].twist.linear.x,0.)
        for subscription in node._subscriptions:subscription.unregister()

    def test_sim_sensor_future_tolerance_is_bounded_and_reference_remains_strict(self):
        import numpy as np
        from smart_wheelchair_safety.local_path_follower_node import LocalPathFollowerNode
        with patch.object(rospy,'Timer'):node=LocalPathFollowerNode(clock=lambda:100.)
        node.use_sim_time=True
        def value(stamp):return (100.,np.zeros(2),rospy.Time.from_sec(stamp),10.)
        base=(value(10.),value(10.),(100.,.8,None,10.),value(10.),value(10.))
        with patch.object(rospy.Time,'now',return_value=rospy.Time.from_sec(10.)):
            for delta,expected in ((.008,True),(.049,True),(.051,False)):
                inputs=list(base);inputs[1]=value(10.+delta)
                self.assertEqual(node._fresh(inputs,100.),expected)
            inputs=list(base);inputs[0]=value(10.008)
            self.assertFalse(node._fresh(inputs,100.))
            node.use_sim_time=False
            self.assertTrue(node._fresh(inputs,100.),'hardware local follower keeps original receipt-only rule')
        for subscription in node._subscriptions:subscription.unregister()

    def test_private_timing_parameters_are_used_and_validated(self):
        from smart_wheelchair_safety.local_path_follower_node import LocalPathFollowerNode

        with patch.object(rospy, 'get_param', side_effect=[10.0, 0.1, False]), \
                patch.object(rospy, 'Timer') as timer:
            node = LocalPathFollowerNode()
        self.assertAlmostEqual(timer.call_args[0][0].to_sec(), 0.1)
        self.assertEqual(node.input_timeout_s, 0.1)
        for value in (0.0, -1.0, float('nan'), float('inf')):
            for parameters in ([value, 0.25], [20.0, value]):
                with self.subTest(parameters=parameters), \
                        patch.object(rospy, 'get_param', side_effect=parameters), \
                        patch.object(rospy, 'Timer'), \
                        self.assertRaises(ValueError):
                    LocalPathFollowerNode()

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
        path.header.stamp = rospy.Time(12, 34)
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
        self.assertEqual(messages[-1].header.stamp, path.header.stamp)

        now[0] += 0.251
        node.publish()

        self.assertEqual((messages[-1].twist.linear.x,
                          messages[-1].twist.angular.z), (0.0, 0.0))

    def test_inflight_result_keeps_consumed_reference_stamp_after_replacement(self):
        import io
        import rospy.msg
        from geometry_msgs.msg import PoseStamped, TwistStamped
        from nav_msgs.msg import Odometry, Path
        from sensor_msgs.msg import LaserScan
        from std_msgs.msg import Float32
        from smart_wheelchair_safety.local_path_follower_node import LocalPathFollowerNode

        with patch.object(rospy, 'Timer'):
            node = LocalPathFollowerNode()
        old = Path()
        old.header.stamp = rospy.Time(12, 34)
        old.poses = [PoseStamped(), PoseStamped()]
        old.poses[-1].pose.position.x = 3.
        new = Path()
        new.header.stamp = rospy.Time(12, 35)
        new.poses = old.poses
        node.on_reference(old)
        node.on_odom(Odometry())
        node.on_speed_limit(Float32(data=.5))
        scan = LaserScan(angle_increment=.1, ranges=[4., 4., 4.])
        node.on_scan_left(scan)
        node.on_scan_right(scan)
        messages = []

        def replace_then_publish(message):
            node.on_reference(new)
            buffer = io.BytesIO()
            rospy.msg.serialize_message(buffer, 999, message)
            messages.append(TwistStamped().deserialize(buffer.getvalue()[4:]))

        node.publisher = SimpleNamespace(publish=replace_then_publish)
        node.publish()

        self.assertEqual(messages[-1].header.stamp, old.header.stamp)
        self.assertEqual(messages[-1].header.frame_id, 'rear_axle')
        node.publish()
        self.assertEqual(messages[-1].header.stamp, new.header.stamp)


if __name__ == "__main__":
    import rostest

    rostest.rosrun("smart_wheelchair_safety", "local_path_follower_node",
                   LocalPathFollowerNodeTest)
