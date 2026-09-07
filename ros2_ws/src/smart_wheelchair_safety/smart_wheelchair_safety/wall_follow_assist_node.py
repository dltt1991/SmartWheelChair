from geometry_msgs.msg import Twist
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool

from smart_wheelchair_safety.wall_follow import (
    WallFollowController,
    scan_points_in_base,
    split_wall_points,
)


class WallFollowAssistNode(Node):
    def __init__(self):
        super().__init__("wall_follow_assist_node")
        self.declare_parameter("scan_timeout_s", 2.0)
        self.declare_parameter("target_wall_distance_m", 0.80)
        self.declare_parameter("wall_follow_enter_distance_m", 1.20)
        self.declare_parameter("min_follow_speed_mps", 0.10)
        self.declare_parameter("max_follow_linear_mps", 0.45)
        self.declare_parameter("max_follow_angular_rps", 0.80)
        self.declare_parameter("wall_follow_hold_s", 0.60)
        self.declare_parameter("wall_filter_alpha", 0.25)
        self.declare_parameter("side_switch_margin_m", 0.30)
        self.declare_parameter("distance_deadband_m", 0.06)
        self.declare_parameter("heading_deadband_rad", 0.03)
        self.declare_parameter("away_distance_margin_m", 0.12)
        self.declare_parameter("max_angular_step_rps", 0.10)
        self.declare_parameter("angular_deadband_rps", 0.05)
        self.declare_parameter("lookahead_m", 1.0)
        self.declare_parameter("k_distance", 1.0)
        self.declare_parameter("k_heading", 1.2)
        self.declare_parameter("min_away_angular_rps", 0.25)
        self.declare_parameter("body_min_x_m", -0.58)
        self.declare_parameter("body_max_x_m", 0.64)
        self.declare_parameter("body_min_y_m", -0.40)
        self.declare_parameter("body_max_y_m", 0.40)
        self.declare_parameter("left_lidar_x_m", 0.46)
        self.declare_parameter("left_lidar_y_m", 0.26)
        self.declare_parameter("right_lidar_x_m", 0.46)
        self.declare_parameter("right_lidar_y_m", -0.26)
        self.declare_parameter("body_filter_margin_m", 0.02)

        self._left_points = []
        self._right_points = []
        self._left_scan_time = None
        self._right_scan_time = None
        self._last_follow_time = None
        self._last_follow_angular_z = 0.0
        self._last_follow_linear_x = 0.0
        self._controller = WallFollowController(
            self.get_parameter("wall_filter_alpha").value,
            self.get_parameter("side_switch_margin_m").value,
            self.get_parameter("distance_deadband_m").value,
            self.get_parameter("heading_deadband_rad").value,
            self.get_parameter("away_distance_margin_m").value,
            self.get_parameter("max_angular_step_rps").value,
            self.get_parameter("angular_deadband_rps").value,
            self.get_parameter("lookahead_m").value,
        )
        self._pub = self.create_publisher(Twist, "cmd_vel_assisted", 10)
        self._active_pub = self.create_publisher(Bool, "wall_follow_active", 10)
        self.create_subscription(Twist, "cmd_vel_raw", self._on_cmd_vel, 10)
        self.create_subscription(LaserScan, "scan_left", self._on_left_scan, 10)
        self.create_subscription(LaserScan, "scan_right", self._on_right_scan, 10)

    def _on_left_scan(self, msg):
        self._left_points = self._wall_points_from_scan(msg, "left")
        self._left_scan_time = self.get_clock().now()

    def _on_right_scan(self, msg):
        self._right_points = self._wall_points_from_scan(msg, "right")
        self._right_scan_time = self.get_clock().now()

    def _on_cmd_vel(self, msg):
        assisted = Twist()
        assisted.linear.x = msg.linear.x
        assisted.angular.z = msg.angular.z
        assisted.angular.x = msg.angular.x
        assisted.angular.y = msg.angular.y
        active = False
        if self._scans_are_fresh():
            all_points = self._left_points + self._right_points
            left_points, right_points = split_wall_points(all_points)
            assisted.linear.x, assisted.angular.z, active = self._controller.assist(
                msg.linear.x,
                msg.angular.z,
                left_points,
                right_points,
                all_points,
                self.get_parameter("target_wall_distance_m").value,
                self.get_parameter("wall_follow_enter_distance_m").value,
                self.get_parameter("min_follow_speed_mps").value,
                self.get_parameter("max_follow_linear_mps").value,
                self.get_parameter("max_follow_angular_rps").value,
                self.get_parameter("k_distance").value,
                self.get_parameter("k_heading").value,
                self.get_parameter("min_away_angular_rps").value,
            )
            if active:
                self._last_follow_time = self.get_clock().now()
                self._last_follow_angular_z = assisted.angular.z
                self._last_follow_linear_x = assisted.linear.x
            elif self._can_hold_wall_follow(msg.linear.x):
                assisted.linear.x = min(
                    msg.linear.x,
                    self._last_follow_linear_x,
                )
                assisted.angular.z = self._last_follow_angular_z
                active = True
        elif msg.linear.x <= self.get_parameter("min_follow_speed_mps").value:
            self._controller.reset()
        assisted.linear.y = 1.0 if active else 0.0
        self._active_pub.publish(Bool(data=active))
        self._pub.publish(assisted)

    def _wall_points_from_scan(self, msg, lidar_prefix):
        points = scan_points_in_base(
            msg.ranges,
            msg.angle_min,
            msg.angle_increment,
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
        return points

    def _scans_are_fresh(self):
        if self._left_scan_time is None or self._right_scan_time is None:
            return False
        timeout = self.get_parameter("scan_timeout_s").value
        now = self.get_clock().now()
        return (
            (now - self._left_scan_time).nanoseconds / 1e9 <= timeout
            and (now - self._right_scan_time).nanoseconds / 1e9 <= timeout
        )

    def _can_hold_wall_follow(self, linear_x):
        if not self._controller.following:
            return False
        if linear_x <= self.get_parameter("min_follow_speed_mps").value:
            return False
        if self._last_follow_time is None:
            return False
        hold_s = self.get_parameter("wall_follow_hold_s").value
        return (self.get_clock().now() - self._last_follow_time).nanoseconds / 1e9 <= hold_s


def main(args=None):
    rclpy.init(args=args)
    node = WallFollowAssistNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
