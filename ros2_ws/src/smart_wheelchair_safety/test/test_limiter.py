import math
import unittest

from smart_wheelchair_safety.limiter import (
    front_sector_ranges,
    limit_forward_speed,
    limit_forward_speed_with_scan_state,
    limit_speed_with_scan_state,
    safety_ranges,
    safety_clearances_in_sector,
    safety_ranges_in_sector,
    sector_ranges,
)


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

    def test_stops_forward_motion_when_no_valid_ranges_exist(self):
        speed = limit_forward_speed(0.5, [math.nan, math.inf, -1.0, 0.0], 0.45, 1.2)

        self.assertEqual(speed, 0.0)

    def test_stops_forward_motion_when_scan_is_missing(self):
        self.assertEqual(limit_forward_speed(0.5, [], 0.45, 1.2), 0.0)

    def test_allows_forward_motion_when_fresh_scan_has_no_obstacle_returns(self):
        speed = limit_forward_speed_with_scan_state(0.5, [], True, 0.45, 1.2)

        self.assertEqual(speed, 0.5)

    def test_stops_forward_motion_when_scan_is_stale(self):
        speed = limit_forward_speed_with_scan_state(0.5, [], False, 0.45, 1.2)

        self.assertEqual(speed, 0.0)

    def test_limits_reverse_motion_from_rear_ranges(self):
        speed = limit_speed_with_scan_state(-0.5, [0.3], True, 0.45, 1.2)

        self.assertEqual(speed, 0.0)

    def test_limits_turning_motion_from_side_ranges(self):
        speed = limit_speed_with_scan_state(1.0, [0.825], True, 0.45, 1.2)

        self.assertAlmostEqual(speed, 0.5)

    def test_allows_turning_when_side_scan_is_clear(self):
        speed = limit_speed_with_scan_state(-1.0, [], True, 0.45, 1.2)

        self.assertEqual(speed, -1.0)

    def test_rejects_invalid_thresholds(self):
        with self.assertRaisesRegex(ValueError, "slow_distance_m must be greater"):
            limit_forward_speed(0.5, [1.0], 1.2, 0.45)

    def test_extracts_front_sector_from_360_degree_scan(self):
        ranges = [3.0] * 720
        ranges[0] = 0.2
        ranges[360] = 0.6
        ranges[719] = 0.3

        front = front_sector_ranges(ranges, -math.pi, math.radians(0.5), math.radians(30))

        self.assertIn(0.6, front)
        self.assertNotIn(0.2, front)
        self.assertNotIn(0.3, front)

    def test_filters_body_intersection_returns_before_limiting(self):
        ranges = [math.inf] * 401
        ranges[90] = 0.32
        ranges[100] = 1.0

        filtered = safety_ranges(
            ranges,
            angle_min=-0.87266,
            angle_increment=math.radians(0.5),
            half_width_rad=0.70,
            range_min=0.08,
            range_max=5.0,
            body_filter_distance_m=0.34,
        )

        self.assertEqual(filtered, [1.0])

    def test_extracts_rear_and_side_sector_ranges(self):
        ranges = [3.0] * 720
        ranges[0] = 0.4
        ranges[540] = 0.6
        ranges[360] = 0.8

        rear = sector_ranges(ranges, -math.pi, math.radians(0.5), math.pi, math.radians(10))
        left = sector_ranges(ranges, -math.pi, math.radians(0.5), math.pi / 2, math.radians(10))

        self.assertIn(0.4, rear)
        self.assertNotIn(0.8, rear)
        self.assertIn(0.6, left)

    def test_filters_ranges_in_named_sector(self):
        ranges = [math.inf] * 720
        ranges[180] = 0.3
        ranges[190] = 0.9

        filtered = safety_ranges_in_sector(
            ranges,
            angle_min=-math.pi,
            angle_increment=math.radians(0.5),
            center_angle_rad=-math.pi / 2,
            half_width_rad=math.radians(10),
            range_min=0.08,
            range_max=5.0,
            body_filter_distance_m=0.34,
        )

        self.assertEqual(filtered, [0.9])

    def test_converts_lidar_ranges_to_body_outline_clearances(self):
        ranges = [math.inf] * 401
        ranges[100] = 0.30

        clearances = safety_clearances_in_sector(
            ranges,
            angle_min=-0.87266,
            angle_increment=math.radians(0.5),
            center_angle_rad=0.0,
            half_width_rad=0.70,
            range_min=0.08,
            range_max=5.0,
            lidar_x_m=0.46,
            lidar_y_m=0.26,
            body_min_x_m=-0.58,
            body_max_x_m=0.64,
            body_min_y_m=-0.40,
            body_max_y_m=0.40,
            body_filter_margin_m=0.02,
        )

        self.assertEqual(len(clearances), 1)
        self.assertAlmostEqual(clearances[0], 0.12, places=2)


if __name__ == "__main__":
    unittest.main()
