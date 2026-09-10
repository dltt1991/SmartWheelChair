#!/usr/bin/env python3
import importlib.util
import io
import unittest


@unittest.skipUnless(importlib.util.find_spec('rospy'), 'requires ROS')
class OdomFrameRelayTest(unittest.TestCase):
    def test_only_parent_frame_changes(self):
        from nav_msgs.msg import Odometry
        from smart_wheelchair_safety.odom_frame_relay import odom_frame_message

        message = Odometry()
        message.header.seq = 42
        message.header.stamp.secs = 12
        message.header.stamp.nsecs = 345
        message.header.frame_id = 'world'
        message.child_frame_id = 'rear_axle'
        message.pose.pose.position.x = -7.03
        message.pose.pose.orientation.w = 1.0
        message.twist.twist.linear.x = .5
        message.twist.twist.angular.z = .3
        message.pose.covariance = [float(i) for i in range(36)]
        message.twist.covariance = [float(i + 36) for i in range(36)]
        result = odom_frame_message(message)
        self.assertEqual(message.header.frame_id, 'world')
        self.assertEqual(result.header.frame_id, 'odom')
        result.header.frame_id = 'world'
        before, after = io.BytesIO(), io.BytesIO()
        message.serialize(before)
        result.serialize(after)
        self.assertEqual(before.getvalue(), after.getvalue())


if __name__ == '__main__':
    import rostest
    rostest.rosrun('smart_wheelchair_safety', 'odom_frame_relay', OdomFrameRelayTest)
