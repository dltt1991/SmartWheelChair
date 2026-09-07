from geometry_msgs.msg import Twist
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool

from smart_wheelchair_safety.limiter import (
    limit_forward_speed_for_arc,
    limit_forward_speed_for_wall_follow,
    limit_turn_speed_for_forward_arc,
    safety_clearances_in_forward_corridor,
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
        self.declare_parameter("body_sector_half_angle_rad", 3.14159265)
        self.declare_parameter("body_min_x_m", -0.58)
        self.declare_parameter("body_max_x_m", 0.64)
        self.declare_parameter("body_min_y_m", -0.40)
        self.declare_parameter("body_max_y_m", 0.40)
        self.declare_parameter("left_lidar_x_m", 0.46)
        self.declare_parameter("left_lidar_y_m", 0.26)
        self.declare_parameter("right_lidar_x_m", 0.46)
        self.declare_parameter("right_lidar_y_m", -0.26)
        self.declare_parameter("body_filter_margin_m", 0.02)
        self.declare_parameter("input_topic", "cmd_vel_raw")
        self.declare_parameter("arc_front_bypass_angular_rps", 0.20)
        self.declare_parameter("arc_front_corridor_half_width_m", 0.40)
        self.declare_parameter("wall_follow_min_body_clearance_m", 0.05)
        self.declare_parameter("wall_follow_slow_body_clearance_m", 0.35)
        self.declare_parameter("wall_follow_min_linear_mps", 0.12)
        self._validate_parameters()

        self._left_body_ranges = []
        self._left_front_ranges = []
        self._left_front_arc_ranges = []
        self._right_body_ranges = []
        self._right_front_ranges = []
        self._right_front_arc_ranges = []
        self._left_rear_ranges = []
        self._right_rear_ranges = []
        self._left_side_ranges = []
        self._right_side_ranges = []
        self._left_scan_time = None
        self._right_scan_time = None
        self._wall_follow_active = False
        self._pub = self.create_publisher(Twist, "cmd_vel", 10)
        self.create_subscription(
            Twist,
            self.get_parameter("input_topic").value,
            self._on_cmd_vel,
            10,
        )
        self.create_subscription(Bool, "wall_follow_active", self._on_wall_follow_active, 10)
        self.create_subscription(LaserScan, "scan_left", self._on_left_scan, 10)
        self.create_subscription(LaserScan, "scan_right", self._on_right_scan, 10)

    def _on_left_scan(self, msg):
        sectors = self._valid_sector_ranges(msg, "left")
        self._left_body_ranges = sectors["body"]
        self._left_front_ranges = sectors["front"]
        self._left_front_arc_ranges = sectors["front_arc"]
        self._left_rear_ranges = sectors["rear"]
        self._left_side_ranges = sectors["left"]
        self._left_scan_time = self.get_clock().now()

    def _on_right_scan(self, msg):
        sectors = self._valid_sector_ranges(msg, "right")
        self._right_body_ranges = sectors["body"]
        self._right_front_ranges = sectors["front"]
        self._right_front_arc_ranges = sectors["front_arc"]
        self._right_rear_ranges = sectors["rear"]
        self._right_side_ranges = sectors["right"]
        self._right_scan_time = self.get_clock().now()

    def _on_wall_follow_active(self, msg):
        self._wall_follow_active = msg.data

    def _on_cmd_vel(self, msg):
        stop_distance = self.get_parameter("stop_distance_m").value
        slow_distance = self.get_parameter("slow_distance_m").value
        arc_threshold = self.get_parameter("arc_front_bypass_angular_rps").value
        wall_follow_active = (self._wall_follow_active or msg.linear.y > 0.5) and msg.linear.x > 0.0
        scans_are_fresh = self._scans_are_fresh()
        linear_ranges = []
        angular_ranges = []
        if scans_are_fresh:
            if msg.linear.x >= 0.0:
                if wall_follow_active or abs(msg.angular.z) >= arc_threshold:
                    linear_ranges = self._left_front_arc_ranges + self._right_front_arc_ranges
                else:
                    linear_ranges = self._left_front_ranges + self._right_front_ranges
            else:
                linear_ranges = self._left_rear_ranges + self._right_rear_ranges

            if msg.angular.z > 0.0:
                angular_ranges = self._left_side_ranges
            elif msg.angular.z < 0.0:
                angular_ranges = self._right_side_ranges

        filtered = Twist()
        forward_limiter_angular = msg.angular.z
        if wall_follow_active and abs(forward_limiter_angular) < arc_threshold:
            forward_limiter_angular = arc_threshold
        if wall_follow_active:
            filtered.linear.x = limit_forward_speed_for_wall_follow(
                msg.linear.x,
                forward_limiter_angular,
                linear_ranges,
                self._left_body_ranges + self._right_body_ranges,
                scans_are_fresh,
                stop_distance,
                slow_distance,
                arc_threshold,
                self.get_parameter("wall_follow_min_body_clearance_m").value,
                self.get_parameter("wall_follow_slow_body_clearance_m").value,
                self.get_parameter("wall_follow_min_linear_mps").value,
            )
        else:
            filtered.linear.x = limit_forward_speed_for_arc(
                msg.linear.x,
                forward_limiter_angular,
                linear_ranges,
                scans_are_fresh,
                stop_distance,
                slow_distance,
                arc_threshold,
            )
        filtered.linear.y = 0.0
        filtered.linear.z = 0.0
        filtered.angular.x = msg.angular.x
        filtered.angular.y = msg.angular.y
        filtered.angular.z = limit_turn_speed_for_forward_arc(
            msg.angular.z,
            filtered.linear.x,
            angular_ranges,
            scans_are_fresh,
            stop_distance,
            slow_distance,
            arc_threshold,
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
            "body": self._valid_scan_ranges_in_sector(
                msg, lidar_prefix, 0.0, "body_sector_half_angle_rad"
            ),
            "front": self._valid_scan_ranges_in_sector(
                msg, lidar_prefix, 0.0, "front_sector_half_angle_rad"
            ),
            "front_arc": self._valid_scan_ranges_in_forward_corridor(msg, lidar_prefix),
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

    def _valid_scan_ranges_in_forward_corridor(self, msg, lidar_prefix):
        return safety_clearances_in_forward_corridor(
            msg.ranges,
            msg.angle_min,
            msg.angle_increment,
            msg.range_min,
            msg.range_max,
            self.get_parameter(f"{lidar_prefix}_lidar_x_m").value,
            self.get_parameter(f"{lidar_prefix}_lidar_y_m").value,
            self.get_parameter("body_max_x_m").value,
            self.get_parameter("body_min_y_m").value,
            self.get_parameter("body_max_y_m").value,
            self.get_parameter("arc_front_corridor_half_width_m").value,
        )

    def _validate_parameters(self):
        stop_distance = self.get_parameter("stop_distance_m").value
        slow_distance = self.get_parameter("slow_distance_m").value
        scan_timeout = self.get_parameter("scan_timeout_s").value
        front_sector_half_angle = self.get_parameter("front_sector_half_angle_rad").value
        rear_sector_half_angle = self.get_parameter("rear_sector_half_angle_rad").value
        side_sector_half_angle = self.get_parameter("side_sector_half_angle_rad").value
        body_sector_half_angle = self.get_parameter("body_sector_half_angle_rad").value
        body_min_x = self.get_parameter("body_min_x_m").value
        body_max_x = self.get_parameter("body_max_x_m").value
        body_min_y = self.get_parameter("body_min_y_m").value
        body_max_y = self.get_parameter("body_max_y_m").value
        body_filter_margin = self.get_parameter("body_filter_margin_m").value
        arc_front_corridor_half_width = self.get_parameter("arc_front_corridor_half_width_m").value
        wall_follow_min_body_clearance = self.get_parameter(
            "wall_follow_min_body_clearance_m"
        ).value
        wall_follow_slow_body_clearance = self.get_parameter(
            "wall_follow_slow_body_clearance_m"
        ).value
        wall_follow_min_linear = self.get_parameter("wall_follow_min_linear_mps").value
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
        if body_sector_half_angle <= 0.0:
            raise ValueError("body_sector_half_angle_rad must be positive")
        if body_min_x >= body_max_x:
            raise ValueError("body_min_x_m must be less than body_max_x_m")
        if body_min_y >= body_max_y:
            raise ValueError("body_min_y_m must be less than body_max_y_m")
        if body_filter_margin < 0.0:
            raise ValueError("body_filter_margin_m must be non-negative")
        if arc_front_corridor_half_width <= 0.0:
            raise ValueError("arc_front_corridor_half_width_m must be positive")
        if wall_follow_min_body_clearance < 0.0:
            raise ValueError("wall_follow_min_body_clearance_m must be non-negative")
        if wall_follow_slow_body_clearance <= wall_follow_min_body_clearance:
            raise ValueError(
                "wall_follow_slow_body_clearance_m must be greater than "
                "wall_follow_min_body_clearance_m"
            )
        if wall_follow_min_linear < 0.0:
            raise ValueError("wall_follow_min_linear_mps must be non-negative")


def main(args=None):
    rclpy.init(args=args)
    node = SafetyFilterNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
