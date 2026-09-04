import math
import unittest

from smart_wheelchair_safety.limiter import limit_forward_speed


class LimitForwardSpeedTest(unittest.TestCase):
    def test_stops_when_obstacle_is_inside_stop_distance(self):
        self.assertEqual(limit_forward_speed(0.8, [0.3, 0.6], 0.45, 1.2), 0.0)

    def test_scales_forward_speed_between_stop_and_slow_distance(self):
        speed = limit_forward_speed(1.0, [0.825], 0.45, 1.2)

        self.assertAlmostEqual(speed, 0.5)

    def test_keeps_forward_speed_when_path_is_clear(self):
        self.assertEqual(limit_forward_speed(0.7, [2.0, 1.8], 0.45, 1.2), 0.7)

    def test_keeps_reverse_speed_without_lidar_limiting(self):
        self.assertEqual(limit_forward_speed(-0.4, [0.2], 0.45, 1.2), -0.4)

    def test_ignores_invalid_ranges(self):
        speed = limit_forward_speed(0.5, [math.nan, math.inf, -1.0, 2.0], 0.45, 1.2)

        self.assertEqual(speed, 0.5)

    def test_rejects_invalid_thresholds(self):
        with self.assertRaisesRegex(ValueError, "slow_distance_m must be greater"):
            limit_forward_speed(0.5, [1.0], 1.2, 0.45)


if __name__ == "__main__":
    unittest.main()
