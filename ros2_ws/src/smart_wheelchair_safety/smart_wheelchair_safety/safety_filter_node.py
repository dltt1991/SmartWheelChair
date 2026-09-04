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

        self._left_ranges = []
        self._right_ranges = []
        self._pub = self.create_publisher(Twist, "cmd_vel", 10)
        self.create_subscription(Twist, "cmd_vel_raw", self._on_cmd_vel, 10)
        self.create_subscription(LaserScan, "scan_left", self._on_left_scan, 10)
        self.create_subscription(LaserScan, "scan_right", self._on_right_scan, 10)

    def _on_left_scan(self, msg):
        self._left_ranges = list(msg.ranges)

    def _on_right_scan(self, msg):
        self._right_ranges = list(msg.ranges)

    def _on_cmd_vel(self, msg):
        stop_distance = self.get_parameter("stop_distance_m").value
        slow_distance = self.get_parameter("slow_distance_m").value

        filtered = Twist()
        filtered.linear.x = limit_forward_speed(
            msg.linear.x,
            self._left_ranges + self._right_ranges,
            stop_distance,
            slow_distance,
        )
        filtered.linear.y = msg.linear.y
        filtered.linear.z = msg.linear.z
        filtered.angular.x = msg.angular.x
        filtered.angular.y = msg.angular.y
        filtered.angular.z = msg.angular.z
        self._pub.publish(filtered)


def main(args=None):
    rclpy.init(args=args)
    node = SafetyFilterNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
