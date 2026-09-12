"""ROS bridge from dual unified scans and shared reference to NeuPAN."""

import math, threading, time
import numpy as np

try:
    import rospy
    from geometry_msgs.msg import TwistStamped, PoseStamped
    from nav_msgs.msg import Odometry, Path
    from sensor_msgs.msg import LaserScan
    from std_msgs.msg import String
except ImportError:  # geometry helpers remain testable without ROS
    rospy = None
from .neupan_adapter import NeuPANAdapter


def _yaw(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def scan_points(scan, side, max_range=5.0):
    ranges = np.asarray(scan.ranges, dtype=float)
    a = scan.angle_min + np.arange(len(ranges)) * scan.angle_increment
    lo, hi = (-30, 170) if side == "left" else (-170, 30)
    deg = np.degrees(a)
    ok = (
        np.isfinite(ranges)
        & (ranges >= max(float(getattr(scan, "range_min", 0)), 0.02))
        & (ranges <= min(max_range, float(getattr(scan, "range_max", max_range))))
        & (deg >= lo - 1e-9)
        & (deg <= hi + 1e-9)
    )
    return np.column_stack((ranges[ok] * np.cos(a[ok]), ranges[ok] * np.sin(a[ok])))


def transform_points(points, transform):
    p = np.asarray(points, dtype=float).reshape((-1, 2))
    q = transform.rotation
    yaw = _yaw(q)
    c, s = math.cos(yaw), math.sin(yaw)
    t = transform.translation
    return p.dot(np.array([[c, s], [-s, c]])) + [t.x, t.y]


def reference_points(path):
    return np.asarray(
        [
            (p.pose.position.x, p.pose.position.y, _yaw(p.pose.orientation))
            for p in path.poses
        ],
        dtype=float,
    ).reshape((-1, 3))


class NeuPANWheelchairNode:
    def __init__(self, clock=time.monotonic):
        self._clock = clock
        self._lock = threading.Lock()
        self._solve_lock = threading.Lock()
        hz = float(rospy.get_param("~control_rate_hz", 10))
        self.input_timeout = float(rospy.get_param("~input_timeout", 0.25))
        self.max_range = float(rospy.get_param("~max_range", 5))
        if any(
            not math.isfinite(v) or v <= 0
            for v in (hz, self.input_timeout, self.max_range)
        ):
            raise ValueError("rate, timeout and range must be positive and finite")
        import tf2_ros

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self._ros_now = lambda: rospy.Time.now().to_sec()
        self.adapter = NeuPANAdapter(
            config_file=rospy.get_param("~config_file", ""),
            dune_checkpoint=rospy.get_param("~dune_checkpoint", ""),
            max_linear=float(rospy.get_param("~max_linear", 0.8)),
            max_angular=float(rospy.get_param("~max_angular", 0.65)),
            action_timeout=self.input_timeout,
            max_range=self.max_range,
        )
        self._reference = self._odom = None
        self._scans = {"left": None, "right": None}
        self.cmd_pub = rospy.Publisher("/neupan/cmd_vel", TwistStamped, queue_size=10)
        self.plan_pub = rospy.Publisher("/neupan/plan", Path, queue_size=10)
        self.status_pub = rospy.Publisher("/neupan/status", String, queue_size=10)
        rospy.Subscriber("/shared_control/reference", Path, self.on_reference)
        rospy.Subscriber("/odom", Odometry, self.on_odom)
        rospy.Subscriber(
            "/unified_scan_left", LaserScan, lambda m: self.on_scan(m, "left")
        )
        rospy.Subscriber(
            "/unified_scan_right", LaserScan, lambda m: self.on_scan(m, "right")
        )
        rospy.Timer(rospy.Duration(1 / hz), self.tick)

    def on_reference(self, m):
        path = reference_points(m)
        valid = (
            m.header.frame_id == "odom" and len(path) >= 2 and np.isfinite(path).all()
        )
        with self._lock:
            self._reference = (self._clock(), m.header.stamp, path) if valid else None

    def on_odom(self, m):
        p = m.pose.pose
        stamp = m.header.stamp
        state = np.array([p.position.x, p.position.y, _yaw(p.orientation)])
        with self._lock:
            self._odom = (
                (self._clock(), stamp, state)
                if m.header.frame_id == "odom" and np.isfinite(state).all()
                else None
            )

    def on_scan(self, m, side):
        received = self._clock()
        try:
            if m.header.stamp.to_sec() <= 0:
                raise ValueError("unstamped scan")
            transform = self.tf_buffer.lookup_transform(
                "odom", m.header.frame_id, m.header.stamp, rospy.Duration(0.03)
            ).transform
            points = transform_points(scan_points(m, side, self.max_range), transform)
            if not np.isfinite(points).all():
                raise ValueError("nonfinite transform")
            value = (received, m.header.stamp, points)
        except Exception:
            value = None
        with self._lock:
            self._scans[side] = value

    def _publish(self, action, trajectory, stamp, reason):
        msg = TwistStamped()
        msg.header.stamp = stamp or rospy.Time()
        msg.header.frame_id = "odom"
        msg.twist.linear.x = float(action[0])
        msg.twist.angular.z = float(action[1])
        self.cmd_pub.publish(msg)
        self.status_pub.publish(String(data=reason))
        plan = Path()
        plan.header.stamp = msg.header.stamp
        plan.header.frame_id = "odom"
        for state in trajectory:
            pose = PoseStamped()
            pose.header = plan.header
            pose.pose.position.x = float(state[0])
            pose.pose.position.y = float(state[1])
            yaw = float(state[2]) if len(state) > 2 else 0.0
            pose.pose.orientation.z = math.sin(yaw / 2)
            pose.pose.orientation.w = math.cos(yaw / 2)
            plan.poses.append(pose)
        self.plan_pub.publish(plan)

    def _fresh(self, values):
        now = self._clock()
        ros_now = self._ros_now()
        for value in values:
            if value is None:
                return False
            stamp = value[1].to_sec() if hasattr(value[1], "to_sec") else value[1]
            if (
                stamp <= 0
                or not 0 <= now - value[0] <= self.input_timeout
                or not 0 <= ros_now - stamp <= self.input_timeout
            ):
                return False
        return True

    def tick(self, _event=None):
        now = self._clock()
        with self._lock:
            ref, odom, left, right = (
                self._reference,
                self._odom,
                self._scans["left"],
                self._scans["right"],
            )
        vals = (ref, odom, left, right)
        if not self._fresh(vals):
            self._publish([0, 0], [], None, "invalid input")
            return
        if not self._solve_lock.acquire(False):
            return
        try:
            self.adapter.set_obstacles(np.vstack((left[2], right[2])), now=now)
            self.adapter.set_initial_path(ref[2])
            action, traj = self.adapter.step(odom[2], now=now)
            valid = self.adapter.reason == "ok" and self._fresh(vals)
            self._publish(
                action if valid else [0, 0],
                traj if valid else [],
                ref[1] if valid else None,
                self.adapter.reason if valid else "stale result",
            )
        except Exception as exc:
            self._publish([0, 0], [], None, "inference error: %s" % exc)
        finally:
            self._solve_lock.release()


def main():
    rospy.init_node("neupan_wheelchair_node")
    NeuPANWheelchairNode()
    rospy.spin()
