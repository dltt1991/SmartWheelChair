import unittest

from smart_wheelchair_safety.joystick import camera_frame_payload, joystick_to_velocity


class JoystickMappingTest(unittest.TestCase):
    def test_maps_forward_and_right_turn(self):
        linear, angular = joystick_to_velocity(0.5, 1.0, 0.8, 1.4)
        self.assertAlmostEqual(linear, 0.8)
        self.assertAlmostEqual(angular, -0.7)

    def test_clamps_input_and_applies_deadzone(self):
        self.assertEqual(joystick_to_velocity(0.01, 0.01, 0.8, 1.4), (0.0, 0.0))
        self.assertEqual(joystick_to_velocity(2.0, -2.0, 0.8, 1.4), (-0.8, -1.4))

    def test_camera_frame_payload_keeps_rgb_pixels(self):
        payload = camera_frame_payload(1, 2, "rgb8", bytes([255, 0, 0, 0, 255, 0]))

        self.assertEqual(payload["width"], 1)
        self.assertEqual(payload["height"], 2)
        self.assertEqual(payload["encoding"], "rgb8")
        self.assertEqual(payload["data"], "/wAAAP8A")
