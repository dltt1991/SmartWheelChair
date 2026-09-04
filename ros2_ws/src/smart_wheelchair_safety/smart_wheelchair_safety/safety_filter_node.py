from geometry_msgs.msg import Twist
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan

from smart_wheelchair_safety.limiter import limit_forward_speed


class SafetyFilterNode(Node):
    def __init__(self):
        super().__init__("safety_filter_node")
        self.declare_parameter("stop_distance_m", 0.45)
        self.declare_parameter("slow_distance_m", 1.20)
        self.declare_parameter("scan_timeout_s", 0.50)
        self._validate_parameters()

        self._left_ranges = []
        self._right_ranges = []
        self._left_scan_time = None
        self._right_scan_time = None
        self._pub = self.create_publisher(Twist, "cmd_vel", 10)
        self.create_subscription(Twist, "cmd_vel_raw", self._on_cmd_vel, 10)
        self.create_subscription(LaserScan, "scan_left", self._on_left_scan, 10)
        self.create_subscription(LaserScan, "scan_right", self._on_right_scan, 10)

    def _on_left_scan(self, msg):
        self._left_ranges = self._valid_scan_ranges(msg)
        self._left_scan_time = self.get_clock().now()

    def _on_right_scan(self, msg):
        self._right_ranges = self._valid_scan_ranges(msg)
        self._right_scan_time = self.get_clock().now()

    def _on_cmd_vel(self, msg):
        stop_distance = self.get_parameter("stop_distance_m").value
        slow_distance = self.get_parameter("slow_distance_m").value
        ranges = []
        if self._scans_are_fresh():
            ranges = self._left_ranges + self._right_ranges

        filtered = Twist()
        filtered.linear.x = limit_forward_speed(
            msg.linear.x,
            ranges,
            stop_distance,
            slow_distance,
        )
        filtered.linear.y = 0.0
        filtered.linear.z = 0.0
        filtered.angular.x = msg.angular.x
        filtered.angular.y = msg.angular.y
        filtered.angular.z = msg.angular.z
        self._pub.publish(filtered)

    def _scans_are_fresh(self):
        if self._left_scan_time is None or self._right_scan_time is None:
            return False
        timeout = self.get_parameter("scan_timeout_s").value
        now = self.get_clock().now()
        return (
            (now - self._left_scan_time).nanoseconds / 1e9 <= timeout
            and (now - self._right_scan_time).nanoseconds / 1e9 <= timeout
        )

    def _valid_scan_ranges(self, msg):
        return [
            value
            for value in msg.ranges
            if msg.range_min <= value <= msg.range_max
        ]

    def _validate_parameters(self):
        stop_distance = self.get_parameter("stop_distance_m").value
        slow_distance = self.get_parameter("slow_distance_m").value
        scan_timeout = self.get_parameter("scan_timeout_s").value
        if slow_distance <= stop_distance:
            raise ValueError("slow_distance_m must be greater than stop_distance_m")
        if scan_timeout <= 0.0:
            raise ValueError("scan_timeout_s must be positive")


def main(args=None):
    rclpy.init(args=args)
    node = SafetyFilterNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
