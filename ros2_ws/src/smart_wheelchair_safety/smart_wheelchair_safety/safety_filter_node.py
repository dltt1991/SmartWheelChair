from geometry_msgs.msg import Twist
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan

from smart_wheelchair_safety.limiter import (
    limit_speed_with_scan_state,
    safety_clearances_in_sector,
)


class SafetyFilterNode(Node):
    def __init__(self):
        super().__init__("safety_filter_node")
        self.declare_parameter("stop_distance_m", 0.10)
        self.declare_parameter("slow_distance_m", 1.20)
        self.declare_parameter("scan_timeout_s", 2.0)
        self.declare_parameter("front_sector_half_angle_rad", 0.70)
        self.declare_parameter("rear_sector_half_angle_rad", 0.70)
        self.declare_parameter("side_sector_half_angle_rad", 0.80)
        self.declare_parameter("body_min_x_m", -0.58)
        self.declare_parameter("body_max_x_m", 0.64)
        self.declare_parameter("body_min_y_m", -0.40)
        self.declare_parameter("body_max_y_m", 0.40)
        self.declare_parameter("left_lidar_x_m", 0.46)
        self.declare_parameter("left_lidar_y_m", 0.26)
        self.declare_parameter("right_lidar_x_m", 0.46)
        self.declare_parameter("right_lidar_y_m", -0.26)
        self.declare_parameter("body_filter_margin_m", 0.02)
        self._validate_parameters()

        self._left_front_ranges = []
        self._right_front_ranges = []
        self._left_rear_ranges = []
        self._right_rear_ranges = []
        self._left_side_ranges = []
        self._right_side_ranges = []
        self._left_scan_time = None
        self._right_scan_time = None
        self._pub = self.create_publisher(Twist, "cmd_vel", 10)
        self.create_subscription(Twist, "cmd_vel_raw", self._on_cmd_vel, 10)
        self.create_subscription(LaserScan, "scan_left", self._on_left_scan, 10)
        self.create_subscription(LaserScan, "scan_right", self._on_right_scan, 10)

    def _on_left_scan(self, msg):
        sectors = self._valid_sector_ranges(msg, "left")
        self._left_front_ranges = sectors["front"]
        self._left_rear_ranges = sectors["rear"]
        self._left_side_ranges = sectors["left"]
        self._left_scan_time = self.get_clock().now()

    def _on_right_scan(self, msg):
        sectors = self._valid_sector_ranges(msg, "right")
        self._right_front_ranges = sectors["front"]
        self._right_rear_ranges = sectors["rear"]
        self._right_side_ranges = sectors["right"]
        self._right_scan_time = self.get_clock().now()

    def _on_cmd_vel(self, msg):
        stop_distance = self.get_parameter("stop_distance_m").value
        slow_distance = self.get_parameter("slow_distance_m").value
        scans_are_fresh = self._scans_are_fresh()
        linear_ranges = []
        angular_ranges = []
        if scans_are_fresh:
            if msg.linear.x >= 0.0:
                linear_ranges = self._left_front_ranges + self._right_front_ranges
            else:
                linear_ranges = self._left_rear_ranges + self._right_rear_ranges

            if msg.angular.z > 0.0:
                angular_ranges = self._left_side_ranges
            elif msg.angular.z < 0.0:
                angular_ranges = self._right_side_ranges

        filtered = Twist()
        filtered.linear.x = limit_speed_with_scan_state(
            msg.linear.x,
            linear_ranges,
            scans_are_fresh,
            stop_distance,
            slow_distance,
        )
        filtered.linear.y = 0.0
        filtered.linear.z = 0.0
        filtered.angular.x = msg.angular.x
        filtered.angular.y = msg.angular.y
        filtered.angular.z = limit_speed_with_scan_state(
            msg.angular.z,
            angular_ranges,
            scans_are_fresh,
            stop_distance,
            slow_distance,
        )
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

    def _valid_sector_ranges(self, msg, lidar_prefix):
        return {
            "front": self._valid_scan_ranges_in_sector(
                msg, lidar_prefix, 0.0, "front_sector_half_angle_rad"
            ),
            "rear": self._valid_scan_ranges_in_sector(
                msg, lidar_prefix, 3.14159265, "rear_sector_half_angle_rad"
            ),
            "left": self._valid_scan_ranges_in_sector(
                msg, lidar_prefix, 1.57079633, "side_sector_half_angle_rad"
            ),
            "right": self._valid_scan_ranges_in_sector(
                msg, lidar_prefix, -1.57079633, "side_sector_half_angle_rad"
            ),
        }

    def _valid_scan_ranges_in_sector(
        self, msg, lidar_prefix, center_angle_rad, half_angle_parameter
    ):
        return safety_clearances_in_sector(
            msg.ranges,
            msg.angle_min,
            msg.angle_increment,
            center_angle_rad,
            self.get_parameter(half_angle_parameter).value,
            msg.range_min,
            msg.range_max,
            self.get_parameter(f"{lidar_prefix}_lidar_x_m").value,
            self.get_parameter(f"{lidar_prefix}_lidar_y_m").value,
            self.get_parameter("body_min_x_m").value,
            self.get_parameter("body_max_x_m").value,
            self.get_parameter("body_min_y_m").value,
            self.get_parameter("body_max_y_m").value,
            self.get_parameter("body_filter_margin_m").value,
        )

    def _validate_parameters(self):
        stop_distance = self.get_parameter("stop_distance_m").value
        slow_distance = self.get_parameter("slow_distance_m").value
        scan_timeout = self.get_parameter("scan_timeout_s").value
        front_sector_half_angle = self.get_parameter("front_sector_half_angle_rad").value
        rear_sector_half_angle = self.get_parameter("rear_sector_half_angle_rad").value
        side_sector_half_angle = self.get_parameter("side_sector_half_angle_rad").value
        body_min_x = self.get_parameter("body_min_x_m").value
        body_max_x = self.get_parameter("body_max_x_m").value
        body_min_y = self.get_parameter("body_min_y_m").value
        body_max_y = self.get_parameter("body_max_y_m").value
        body_filter_margin = self.get_parameter("body_filter_margin_m").value
        if slow_distance <= stop_distance:
            raise ValueError("slow_distance_m must be greater than stop_distance_m")
        if scan_timeout <= 0.0:
            raise ValueError("scan_timeout_s must be positive")
        if front_sector_half_angle <= 0.0:
            raise ValueError("front_sector_half_angle_rad must be positive")
        if rear_sector_half_angle <= 0.0:
            raise ValueError("rear_sector_half_angle_rad must be positive")
        if side_sector_half_angle <= 0.0:
            raise ValueError("side_sector_half_angle_rad must be positive")
        if body_min_x >= body_max_x:
            raise ValueError("body_min_x_m must be less than body_max_x_m")
        if body_min_y >= body_max_y:
            raise ValueError("body_min_y_m must be less than body_max_y_m")
        if body_filter_margin < 0.0:
            raise ValueError("body_filter_margin_m must be non-negative")


def main(args=None):
    rclpy.init(args=args)
    node = SafetyFilterNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
