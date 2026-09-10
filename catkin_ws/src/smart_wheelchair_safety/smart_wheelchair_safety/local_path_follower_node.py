import math
import threading
import time

from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry, Path
import numpy as np
import rospy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32

from smart_wheelchair_safety.local_path_follower import select_velocity


LIDAR_X_M = 0.79
LIDAR_Y_M = {"left": 0.26, "right": -0.26}


def _yaw(quaternion):
    return math.atan2(
        2.0 * (quaternion.w * quaternion.z
               + quaternion.x * quaternion.y),
        1.0 - 2.0 * (quaternion.y ** 2 + quaternion.z ** 2),
    )


class LocalPathFollowerNode:
    def __init__(self, clock=time.monotonic):
        control_rate_hz = float(rospy.get_param('~control_rate_hz', 20.0))
        self.input_timeout_s = float(rospy.get_param('~input_timeout_s', 0.25))
        if any(not math.isfinite(value) or value <= 0.0
               for value in (control_rate_hz, self.input_timeout_s)):
            raise ValueError('control_rate_hz and input_timeout_s must be finite and positive')
        self._clock = clock
        self._lock = threading.Lock()
        self._reference = None
        self._odom = None
        self._speed_limit = None
        self._scans = {"left": None, "right": None}
        self._previous_velocity = np.zeros(2)

        self.publisher = rospy.Publisher(
            "/cmd_vel_planned", TwistStamped, queue_size=10, latch=False)
        self._subscriptions = [
            rospy.Subscriber("/shared_control/reference", Path,
                             self.on_reference, queue_size=10),
            rospy.Subscriber("/speed_limit", Float32,
                             self.on_speed_limit, queue_size=10),
            rospy.Subscriber("/odom", Odometry, self.on_odom, queue_size=10),
            rospy.Subscriber("/unified_scan_left", LaserScan,
                             self.on_scan_left, queue_size=10),
            rospy.Subscriber("/unified_scan_right", LaserScan,
                             self.on_scan_right, queue_size=10),
        ]
        self._timer = rospy.Timer(rospy.Duration(1.0 / control_rate_hz), self.publish)

    def on_reference(self, message):
        path = np.asarray([
            (pose.pose.position.x, pose.pose.position.y,
             _yaw(pose.pose.orientation))
            for pose in message.poses
        ], dtype=float).reshape((-1, 3))
        with self._lock:
            self._reference = (self._clock(), path,
                               rospy.Time(message.header.stamp.secs, message.header.stamp.nsecs))

    def on_speed_limit(self, message):
        with self._lock:
            self._speed_limit = (self._clock(), float(message.data))

    def on_odom(self, message):
        pose = message.pose.pose
        odom = np.asarray((pose.position.x, pose.position.y,
                           _yaw(pose.orientation)), dtype=float)
        with self._lock:
            self._odom = (self._clock(), odom)

    def on_scan_left(self, message):
        self._store_scan(message, "left")

    def on_scan_right(self, message):
        self._store_scan(message, "right")

    def _store_scan(self, message, side):
        ranges = np.asarray(message.ranges, dtype=float)
        angles = (message.angle_min
                  + np.arange(len(ranges)) * message.angle_increment)
        valid = np.isfinite(ranges) & np.isfinite(angles)
        points = np.column_stack((
            LIDAR_X_M + ranges[valid] * np.cos(angles[valid]),
            LIDAR_Y_M[side] + ranges[valid] * np.sin(angles[valid]),
        ))
        with self._lock:
            self._scans[side] = (self._clock(), points)

    def publish(self, _event=None):
        now = self._clock()
        with self._lock:
            inputs = (self._reference, self._odom, self._speed_limit,
                      self._scans["left"], self._scans["right"])
            reference_stamp = self._reference[2] if self._reference else rospy.Time()
            if any(value is None or now - value[0] > self.input_timeout_s
                   for value in inputs):
                command = np.zeros(2)
            else:
                path, odom, speed_limit, left, right = (
                    value[1] for value in inputs)
                delta = path[:, :2] - odom[:2]
                cosine, sine = math.cos(odom[2]), math.sin(odom[2])
                local_path = np.column_stack((
                    cosine * delta[:, 0] + sine * delta[:, 1],
                    -sine * delta[:, 0] + cosine * delta[:, 1],
                    np.arctan2(np.sin(path[:, 2] - odom[2]),
                               np.cos(path[:, 2] - odom[2])),
                ))
                command = select_velocity(
                    local_path, np.vstack((left, right)), speed_limit,
                    self._previous_velocity)
            self._previous_velocity = np.asarray(command, dtype=float)

        message = TwistStamped()
        # Preserve the identity of the reference consumed by this computation,
        # including when a replacement arrives before publication.
        message.header.stamp = reference_stamp
        message.header.frame_id = "rear_axle"
        message.twist.linear.x = float(command[0])
        message.twist.angular.z = float(command[1])
        self.publisher.publish(message)


def main():
    rospy.init_node("local_path_follower_node")
    node = LocalPathFollowerNode()
    rospy.spin()
