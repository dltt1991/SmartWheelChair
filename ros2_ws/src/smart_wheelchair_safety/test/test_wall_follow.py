import math
import unittest
from pathlib import Path

from smart_wheelchair_safety.wall_follow import (
    WallFollowController,
    assist_wall_follow,
    assist_wall_follow_with_state,
)


class WallFollowAssistTest(unittest.TestCase):
    def test_away_turn_releases_wall_follow_even_during_scan_dropout(self):
        for side in (-1, 1):
            for dropout in (False, True):
                with self.subTest(side=side, dropout=dropout):
                    controller = WallFollowController()
                    points = [(0.0, side * .7), (1.0, side * .7)]
                    left, right = (points, []) if side > 0 else ([], points)
                    self.assertTrue(controller.assist(.5, 0., left, right)[2])
                    command = controller.assist(.5, -side * .4,
                                                [] if dropout else left,
                                                [] if dropout else right)
                    self.assertEqual(command, (.5, -side * .4, False))
                    self.assertFalse(controller.following)

    def test_turn_limit_slows_translation_to_preserve_wall_follow_curvature(self):
        commands = []
        for max_turn in (.3, 2.):
            controller = WallFollowController(max_angular_step_rps=10.)
            commands.append(controller.assist(
                .6, 0., [(0., .85), (.4, .55)], [],
                max_follow_linear_mps=.6, max_follow_angular_rps=max_turn,
                target_wall_distance_m=.7,
            ))
        slow, fast = commands
        self.assertLess(slow[0], fast[0])
        self.assertAlmostEqual(slow[1] / slow[0], fast[1] / fast[0])

    def test_user_turn_into_wall_does_not_cancel_tangent_tracking(self):
        for side in (-1, 1):
            with self.subTest(side=side):
                points = [(x, side * (0.625 - 0.29 * x)) for x in (0.0, 0.2, 0.4, 0.6)]
                controller = WallFollowController(
                    away_distance_margin_m=0.30, angular_deadband_rps=0.08,
                    lookahead_m=1.20,
                )
                for _ in range(30):
                    linear, angular, active = controller.assist(
                        0.833, side * 0.42, points if side > 0 else [],
                        points if side < 0 else [], target_wall_distance_m=0.70,
                        max_follow_linear_mps=0.30,
                        k_heading=1.0,
                    )
                self.assertTrue(active)
                self.assertEqual(linear, 0.30)
                self.assertLess(side * angular, -0.05)

    def test_lookahead_uses_the_speed_it_actually_commands(self):
        commands = []
        for speed in (0.30, 1.60):
            controller = WallFollowController(max_angular_step_rps=10.0)
            commands.append(controller.assist(
                speed, 0.0, [(0.0, 0.9), (1.0, 0.6)], [],
                max_follow_linear_mps=0.30,
            ))
        self.assertEqual(commands[0], commands[1])

    def test_small_steering_is_preserved_until_wall_heading_is_aligned(self):
        controller = WallFollowController(
            heading_deadband_rad=0.08, angular_deadband_rps=0.08,
            distance_deadband_m=0.20, away_distance_margin_m=0.30,
            lookahead_m=1.20,
        )
        _, angular, active = controller.assist(
            0.30, 0.0, [(0.0, 0.71), (1.0, 0.57)], [],
            target_wall_distance_m=0.70, k_heading=1.0,
        )
        self.assertTrue(active)
        self.assertLess(angular, 0.0)

    def test_heading_deadband_does_not_hide_a_persistent_approach_angle(self):
        controller = WallFollowController(
            heading_deadband_rad=0.08, angular_deadband_rps=0.08,
            distance_deadband_m=0.20, away_distance_margin_m=0.30,
            lookahead_m=1.20,
        )
        _, angular, active = controller.assist(
            0.30, 0.0, [(0.0, 0.70), (1.0, 0.63)], [],
            target_wall_distance_m=0.70, k_heading=1.0,
        )
        self.assertTrue(active)
        self.assertLess(angular, -0.01)

    def test_keeps_command_when_wall_is_not_close(self):
        linear, angular = assist_wall_follow(
            0.5,
            0.0,
            left_points=[(0.4, 1.4), (1.0, 1.4)],
            right_points=[],
        )

        self.assertEqual((linear, angular), (0.5, 0.0))

    def test_enters_wall_follow_before_safety_slowdown_range(self):
        linear, angular = assist_wall_follow(
            0.5,
            0.0,
            left_points=[],
            right_points=[],
            front_points=[(0.1, 1.02), (0.7, 0.73), (1.3, 0.44)],
        )

        self.assertEqual(linear, 0.45)
        self.assertLess(angular, -0.20)

    def test_turns_away_from_left_wall_when_inside_target_distance(self):
        linear, angular = assist_wall_follow(
            0.5,
            0.0,
            left_points=[(0.2, 0.62), (1.2, 0.62)],
            right_points=[],
        )

        self.assertGreater(linear, 0.0)
        self.assertLess(angular, 0.0)

    def test_turns_away_from_right_wall_when_inside_target_distance(self):
        linear, angular = assist_wall_follow(
            0.5,
            0.0,
            left_points=[],
            right_points=[(0.2, -0.62), (1.2, -0.62)],
        )

        self.assertGreater(linear, 0.0)
        self.assertGreater(angular, 0.0)

    def test_does_not_turn_toward_left_wall_when_outside_comfort_distance(self):
        linear, angular = assist_wall_follow(
            0.5,
            0.0,
            left_points=[(0.2, 0.95), (1.2, 0.95)],
            right_points=[],
        )

        self.assertGreater(linear, 0.0)
        self.assertEqual(angular, 0.0)

    def test_preserves_user_turn_direction_along_slanted_wall(self):
        linear, angular = assist_wall_follow(
            0.5,
            0.0,
            left_points=[(0.2, 0.75), (1.2, 0.50)],
            right_points=[],
        )

        self.assertGreater(linear, 0.0)
        self.assertLess(angular, 0.0)

    def test_keeps_command_for_front_cross_wall_from_side_points(self):
        command = assist_wall_follow(
            0.5,
            0.0,
            left_points=[(0.7, 0.55), (0.7, 0.95)],
            right_points=[(0.7, -0.55), (0.7, -0.95)],
        )

        self.assertEqual(command, (0.5, 0.0))

    def test_keeps_command_for_front_cross_wall_from_center_scan_points(self):
        command = assist_wall_follow(
            0.5,
            0.0,
            left_points=[],
            right_points=[],
            front_points=[(0.7, -0.40), (0.7, 0.0), (0.7, 0.40)],
        )

        self.assertEqual(command, (0.5, 0.0))

    def test_turns_along_oblique_wall_when_approaching_at_an_angle(self):
        linear, angular = assist_wall_follow(
            0.5,
            0.0,
            left_points=[],
            right_points=[],
            front_points=[(0.2, 0.62), (0.55, 0.42), (0.9, 0.22)],
        )

        self.assertGreater(linear, 0.0)
        self.assertLessEqual(linear, 0.45)
        self.assertLess(angular, -0.20)

    def test_prioritizes_heading_when_approaching_wall_obliquely(self):
        linear, angular = assist_wall_follow(
            0.8,
            0.0,
            left_points=[],
            right_points=[],
            front_points=[(0.1, 0.92), (0.7, 0.63), (1.3, 0.34)],
        )

        self.assertEqual(linear, 0.45)
        self.assertLess(angular, -0.45)

    def test_uses_closest_lateral_point_for_oblique_wall_clearance(self):
        linear, angular = assist_wall_follow(
            0.8,
            0.0,
            left_points=[],
            right_points=[],
            front_points=[(0.1, 0.85), (0.8, 0.72), (1.5, 0.59)],
        )

        self.assertEqual(linear, 0.45)
        self.assertLess(angular, -0.35)

    def test_reports_active_when_following_wall_even_if_command_change_is_small(self):
        linear, angular, active = assist_wall_follow_with_state(
            0.45,
            0.0,
            left_points=[(0.2, 0.80), (1.2, 0.80)],
            right_points=[],
        )

        self.assertEqual(linear, 0.45)
        self.assertAlmostEqual(angular, 0.0)
        self.assertTrue(active)

    def test_stateful_controller_does_not_force_away_turn_inside_target_deadband(self):
        controller = WallFollowController()

        linear, angular, active = controller.assist(
            0.45,
            0.0,
            left_points=[(0.2, 0.76), (1.2, 0.76)],
            right_points=[],
        )

        self.assertEqual(linear, 0.45)
        self.assertAlmostEqual(angular, 0.0)
        self.assertTrue(active)

    def test_stateful_controller_suppresses_small_angular_jitter_near_target(self):
        controller = WallFollowController(
            wall_filter_alpha=1.0,
            distance_deadband_m=0.08,
            heading_deadband_rad=0.08,
            angular_deadband_rps=0.05,
        )

        _, angular, active = controller.assist(
            0.45,
            0.0,
            left_points=[(0.2, 0.84), (1.2, 0.84)],
            right_points=[],
        )

        self.assertTrue(active)
        self.assertEqual(angular, 0.0)

    def test_stateful_controller_suppresses_small_heading_jitter_near_target(self):
        controller = WallFollowController(
            wall_filter_alpha=1.0,
            distance_deadband_m=0.20,
            heading_deadband_rad=0.08,
            angular_deadband_rps=0.08,
        )

        _, angular, active = controller.assist(
            0.45,
            0.0,
            left_points=[(0.2, 0.81), (1.2, 0.86)],
            right_points=[],
        )

        self.assertTrue(active)
        self.assertEqual(angular, 0.0)

    def test_stateful_controller_clears_residual_turn_when_comfortable(self):
        controller = WallFollowController(
            wall_filter_alpha=1.0,
            distance_deadband_m=0.20,
            heading_deadband_rad=0.08,
            away_distance_margin_m=0.30,
            max_angular_step_rps=0.04,
            angular_deadband_rps=0.08,
        )

        _, first_angular, first_active = controller.assist(
            0.45,
            0.0,
            left_points=[(0.2, 0.45), (1.2, 0.45)],
            right_points=[],
            min_away_angular_rps=0.12,
        )
        _, second_angular, second_active = controller.assist(
            0.45,
            0.0,
            left_points=[(0.2, 0.80), (1.2, 0.80)],
            right_points=[],
            min_away_angular_rps=0.12,
        )

        self.assertTrue(first_active)
        self.assertTrue(second_active)
        self.assertLess(first_angular, 0.0)
        self.assertEqual(second_angular, 0.0)

    def test_stateful_controller_uses_lookahead_for_smooth_wall_convergence(self):
        controller = WallFollowController(
            wall_filter_alpha=1.0,
            distance_deadband_m=0.05,
            heading_deadband_rad=0.0,
            away_distance_margin_m=0.0,
            max_angular_step_rps=10.0,
            angular_deadband_rps=0.0,
            lookahead_m=1.0,
        )

        _, close_angular, close_active = controller.assist(
            0.35,
            0.0,
            left_points=[(0.2, 0.50), (1.2, 0.50)],
            right_points=[],
            target_wall_distance_m=0.75,
            max_follow_linear_mps=0.35,
            max_follow_angular_rps=0.40,
            k_distance=0.25,
            k_heading=1.0,
            min_away_angular_rps=0.0,
        )
        _, centered_angular, centered_active = controller.assist(
            0.35,
            0.0,
            left_points=[(0.2, 0.75), (1.2, 0.75)],
            right_points=[],
            target_wall_distance_m=0.75,
            max_follow_linear_mps=0.35,
            max_follow_angular_rps=0.40,
            k_distance=0.25,
            k_heading=1.0,
            min_away_angular_rps=0.0,
        )

        self.assertTrue(close_active)
        self.assertTrue(centered_active)
        self.assertLess(close_angular, 0.0)
        self.assertAlmostEqual(centered_angular, 0.0)
        self.assertLess(abs(close_angular), 0.25)

    def test_stateful_controller_keeps_following_same_side_when_parallel_walls_are_close(self):
        controller = WallFollowController()

        _, first_angular, first_active = controller.assist(
            0.45,
            0.0,
            left_points=[(0.2, 0.80), (1.2, 0.80)],
            right_points=[],
        )
        _, second_angular, second_active = controller.assist(
            0.45,
            0.0,
            left_points=[(0.2, 0.82), (1.2, 0.82)],
            right_points=[(0.2, -0.78), (1.2, -0.78)],
        )

        self.assertTrue(first_active)
        self.assertTrue(second_active)
        self.assertGreaterEqual(first_angular, -0.001)
        self.assertGreaterEqual(second_angular, -0.001)

    def test_ignores_reverse_commands(self):
        self.assertEqual(
            assist_wall_follow(
                -0.2,
                0.0,
                left_points=[(0.2, 0.62), (1.2, 0.62)],
                right_points=[],
            ),
            (-0.2, 0.0),
        )

    def test_node_is_registered_and_safety_filter_consumes_assisted_topic(self):
        setup_source = (Path(__file__).parents[1] / "setup.py").read_text()
        safety_source = (
            Path(__file__).parents[1]
            / "smart_wheelchair_safety"
            / "safety_filter_node.py"
        ).read_text()
        assist_source = (
            Path(__file__).parents[1]
            / "smart_wheelchair_safety"
            / "wall_follow_assist_node.py"
        ).read_text()
        web_joystick_source = (
            Path(__file__).parents[1]
            / "smart_wheelchair_safety"
            / "web_joystick_node.py"
        ).read_text()
        launch_source = (
            Path(__file__).parents[2]
            / "smart_wheelchair_gazebo"
            / "launch"
            / "sim.launch.py"
        ).read_text()

        self.assertIn("wall_follow_assist_node", setup_source)
        self.assertIn('"input_topic", "cmd_vel_raw"', safety_source)
        self.assertIn('"wall_follow_active"', assist_source)
        self.assertIn('"wall_follow_active"', safety_source)
        self.assertIn("WallFollowController", assist_source)
        self.assertIn('"lookahead_m", 1.0', assist_source)
        self.assertIn("assisted.linear.y = 1.0 if active else 0.0", assist_source)
        self.assertIn("_can_hold_wall_follow", assist_source)
        self.assertNotIn("active = assisted.linear.x != msg.linear.x", assist_source)
        self.assertIn("msg.linear.y > 0.5", safety_source)
        self.assertIn('"arc_front_corridor_half_width_m": 0.40', launch_source)
        self.assertIn('"wall_follow_hold_s": 0.60', launch_source)
        self.assertIn('"target_wall_distance_m": 0.70', launch_source)
        self.assertIn('"max_follow_linear_mps": 0.60', launch_source)
        self.assertIn('"max_follow_angular_rps": 0.30', launch_source)
        self.assertIn('"wall_filter_alpha": 0.10', launch_source)
        self.assertIn('"side_switch_margin_m": 0.30', launch_source)
        self.assertIn('"distance_deadband_m": 0.20', launch_source)
        self.assertIn('"heading_deadband_rad": 0.08', launch_source)
        self.assertIn('"away_distance_margin_m": 0.30', launch_source)
        self.assertIn('"max_angular_step_rps": 0.04', launch_source)
        self.assertIn('"angular_deadband_rps": 0.08', launch_source)
        self.assertIn('"lookahead_m": 1.20', launch_source)
        self.assertIn('"k_distance": 0.25', launch_source)
        self.assertIn('"k_heading": 1.0', launch_source)
        self.assertIn('"min_away_angular_rps": 0.12', launch_source)
        self.assertIn('"wall_follow_min_body_clearance_m": 0.05', launch_source)
        self.assertIn('"wall_follow_slow_body_clearance_m": 0.35', launch_source)
        self.assertIn('"wall_follow_min_linear_mps": 0.12', launch_source)
        self.assertIn('"input_topic": "cmd_vel_assisted"', launch_source)
        self.assertIn('"command_timeout_s": 1.0', launch_source)
        self.assertIn("function hasCommand()", web_joystick_source)
        self.assertIn("client_id: clientId", web_joystick_source)
        self.assertIn("self._active_client_id", web_joystick_source)
        self.assertIn("if (dragging || hasCommand()) send(current.x, current.y);", web_joystick_source)
        self.assertNotIn("setInterval(() => send(current.x, current.y), 100)", web_joystick_source)
        self.assertIn("wall_follow_assist_node", launch_source)


if __name__ == "__main__":
    unittest.main()
