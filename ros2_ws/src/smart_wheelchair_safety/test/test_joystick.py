import unittest
from pathlib import Path

from smart_wheelchair_safety.joystick import (
    camera_frame_payload,
    joystick_to_velocity,
    reverse_wheel_paths,
)


class JoystickMappingTest(unittest.TestCase):
    def test_maps_forward_and_right_turn(self):
        linear, angular = joystick_to_velocity(0.5, 1.0, 0.8, 1.4)
        self.assertAlmostEqual(linear, 0.8)
        self.assertAlmostEqual(angular, -0.7)

    def test_clamps_input_and_applies_deadzone(self):
        self.assertEqual(joystick_to_velocity(0.01, 0.01, 0.8, 1.4), (0.0, 0.0))
        self.assertEqual(joystick_to_velocity(2.0, -2.0, 0.8, 1.4), (-0.8, -1.4))

    def test_uses_separate_forward_and_reverse_speed_limits(self):
        forward, _ = joystick_to_velocity(
            0.0,
            1.0,
            1.6666667,
            1.4,
            max_reverse_linear=0.8333333,
        )
        reverse, _ = joystick_to_velocity(
            0.0,
            -1.0,
            1.6666667,
            1.4,
            max_reverse_linear=0.8333333,
        )

        self.assertAlmostEqual(forward, 1.6666667)
        self.assertAlmostEqual(reverse, -0.8333333)

    def test_camera_frame_payload_keeps_rgb_pixels(self):
        payload = camera_frame_payload(1, 2, "rgb8", bytes([255, 0, 0, 0, 255, 0]))

        self.assertEqual(payload["width"], 1)
        self.assertEqual(payload["height"], 2)
        self.assertEqual(payload["encoding"], "rgb8")
        self.assertEqual(payload["data"], "/wAAAP8A")

    def test_reverse_wheel_paths_are_parallel_when_backing_straight(self):
        paths = reverse_wheel_paths(-0.5, 0.0, wheel_width_m=0.72, max_length_m=2.0)

        self.assertAlmostEqual(paths["left"][0][1], 0.36)
        self.assertAlmostEqual(paths["right"][0][1], -0.36)
        self.assertAlmostEqual(paths["left"][-1][0], -2.0)
        self.assertAlmostEqual(paths["right"][-1][0], -2.0)

    def test_reverse_wheel_paths_are_empty_unless_reversing(self):
        self.assertEqual(reverse_wheel_paths(0.0, 0.5), {"left": [], "right": []})
        self.assertEqual(reverse_wheel_paths(0.3, 0.5), {"left": [], "right": []})

    def test_rear_camera_overlay_uses_rear_view_turn_direction(self):
        page_source = (
            Path(__file__).parents[1]
            / "smart_wheelchair_safety"
            / "web_joystick_node.py"
        ).read_text()

        self.assertIn("const angular = current.x * MAX_ANGULAR_RPS;", page_source)
