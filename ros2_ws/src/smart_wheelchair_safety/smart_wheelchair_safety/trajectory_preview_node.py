import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

from geometry_msgs.msg import Twist
import rclpy
from rclpy.node import Node

from smart_wheelchair_safety.trajectory_preview import (
    hide_trajectory_visual_config_text,
    preview_paths,
    trajectory_visual_config_text,
)


class TrajectoryPreviewNode(Node):
    def __init__(self):
        super().__init__("trajectory_preview_node")
        self.declare_parameter("prediction_seconds", 3.0)
        self.declare_parameter("step_seconds", 0.3)
        self.declare_parameter("publish_period_s", 0.2)
        self.declare_parameter("command_hold_s", 0.8)
        self.declare_parameter("command_update_epsilon", 0.03)
        self.declare_parameter("command_topic", "cmd_vel_raw")
        self.declare_parameter("visual_config_service", "/world/m6_room/visual_config")
        self.declare_parameter("max_segments", 10)
        self.declare_parameter("marker_z_m", 0.18)
        self.declare_parameter("rear_axle_x_m", -0.33)
        self.declare_parameter("wheel_separation_m", 0.72)

        self._cmd = Twist()
        self._last_nonzero_command_time = 0.0
        self._last_rendered_command = None
        self._preview_visible = False
        self._visual_pool = ThreadPoolExecutor(max_workers=12)
        self._visual_future = None
        self.create_subscription(
            Twist,
            str(self.get_parameter("command_topic").value),
            self._on_cmd_vel,
            10,
        )
        self.create_timer(
            float(self.get_parameter("publish_period_s").value), self._publish_preview
        )

    def _on_cmd_vel(self, msg):
        self._cmd = msg
        if abs(msg.linear.x) >= 1e-4 or abs(msg.angular.z) >= 1e-4:
            self._last_nonzero_command_time = time.monotonic()

    def _publish_preview(self):
        if self._visual_future is not None and not self._visual_future.done():
            return

        linear_x = self._cmd.linear.x
        angular_z = self._cmd.angular.z
        has_preview = abs(linear_x) >= 1e-4 or abs(angular_z) >= 1e-4
        if has_preview:
            command = (linear_x, angular_z)
            if not self._command_changed(command):
                return
            self._last_rendered_command = command
            self._visual_future = self._visual_pool.submit(self._show_preview, command)
            self._preview_visible = True
        elif self._preview_visible:
            hold_time = float(self.get_parameter("command_hold_s").value)
            if time.monotonic() - self._last_nonzero_command_time < hold_time:
                return
            self._last_rendered_command = None
            self._visual_future = self._visual_pool.submit(self._hide_preview)
            self._preview_visible = False

    def _command_changed(self, command):
        if self._last_rendered_command is None:
            return True
        epsilon = float(self.get_parameter("command_update_epsilon").value)
        return (
            abs(command[0] - self._last_rendered_command[0]) >= epsilon
            or abs(command[1] - self._last_rendered_command[1]) >= epsilon
        )

    def _show_preview(self, command):
        linear_x, angular_z = command
        paths = preview_paths(
            linear_x,
            angular_z,
            0.0,
            0.0,
            0.0,
            float(self.get_parameter("prediction_seconds").value),
            float(self.get_parameter("step_seconds").value),
            float(self.get_parameter("marker_z_m").value),
            float(self.get_parameter("rear_axle_x_m").value),
            float(self.get_parameter("wheel_separation_m").value),
        )
        max_segments = int(self.get_parameter("max_segments").value)
        for name in ("rear_axle", "left_wheel", "right_wheel"):
            segments = list(zip(paths[name], paths[name][1:]))[:max_segments]
            requests = [
                trajectory_visual_config_text(name, index, start, end)
                for index, (start, end) in enumerate(segments)
            ]
            requests.extend(
                hide_trajectory_visual_config_text(name, index)
                for index in range(len(segments), max_segments)
            )
            self._update_visuals(requests)

    def _hide_preview(self):
        max_segments = int(self.get_parameter("max_segments").value)
        requests = []
        for name in ("rear_axle", "left_wheel", "right_wheel"):
            for index in range(max_segments):
                requests.append(hide_trajectory_visual_config_text(name, index))
        self._update_visuals(requests)

    def _update_visuals(self, requests):
        futures = [
            self._visual_pool.submit(self._update_visual, request) for request in requests
        ]
        for future in futures:
            future.result()

    def _update_visual(self, request):
        try:
            result = subprocess.run(
                [
                    "gz",
                    "service",
                    "-s",
                    str(self.get_parameter("visual_config_service").value),
                    "--reqtype",
                    "gz.msgs.Visual",
                    "--reptype",
                    "gz.msgs.Boolean",
                    "--timeout",
                    "1000",
                    "--req",
                    request,
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=3.0,
            )
            return result.returncode == 0 and "data: true" in result.stdout
        except (OSError, subprocess.TimeoutExpired) as exc:
            self.get_logger().warning(f"Failed to update Gazebo trajectory visual: {exc}")
            return False


def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryPreviewNode()
    try:
        rclpy.spin(node)
    finally:
        node._visual_pool.shutdown(wait=False)
        node.destroy_node()
        rclpy.shutdown()
