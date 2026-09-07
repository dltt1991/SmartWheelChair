"""Closed-loop corridor regression using the simulation's launch parameters."""

import ast
import inspect
import math
from pathlib import Path
import unittest

from smart_wheelchair_safety.limiter import (
    limit_forward_speed_for_wall_follow,
    limit_turn_speed_for_forward_arc,
    safety_clearances_in_forward_corridor,
    safety_clearances_in_sector,
)
from smart_wheelchair_safety.wall_follow import (
    WallFollowController,
    scan_points_in_base,
    split_wall_points,
)


def launch_parameters(name):
    path = Path(__file__).parents[2] / 'smart_wheelchair_gazebo/launch/sim.launch.py'
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            values = next(k.value for k in node.value.keywords if k.arg == 'parameters')
            return {k: v for value in values.elts if isinstance(value, ast.Dict)
                    for k, v in ast.literal_eval(value).items()}
    raise AssertionError(name)


class WallFollowClosedLoopTest(unittest.TestCase):
    def test_oblique_approaches_keep_moving_with_body_clearance(self):
        params = launch_parameters('wall_follow_assist')
        constructor = inspect.signature(WallFollowController).parameters
        arguments = inspect.signature(WallFollowController.assist).parameters
        kwargs = {k: v for k, v in params.items() if k in arguments}
        kwargs['enter_distance_m'] = params['wall_follow_enter_distance_m']
        for angle in (-60, -30, -15, 15, 30, 60):
            for turn in (0.0, math.copysign(0.42, angle)):
                with self.subTest(angle=angle, turn=turn):
                    controller = WallFollowController(**{k: v for k, v in params.items() if k in constructor})
                    yaw = math.radians(angle)
                    y = 0.0
                    travel = 0.0
                    for step in range(800):
                        points, body, front, left, right = [], [], [], [], []
                        for side in (-1, 1):
                            origin_y = y + .46 * math.sin(yaw) + side * .26 * math.cos(yaw)
                            angle_min = math.radians(-150 if side < 0 else -50)
                            increment = math.radians(1)
                            ranges = []
                            for i in range(201):
                                direction = math.sin(yaw + angle_min + i * increment)
                                distance = ((math.copysign(1.29, direction) - origin_y) / direction
                                            if abs(direction) > 1e-9 else math.inf)
                                ranges.append(distance if distance > 0 else math.inf)
                            scan = (ranges, angle_min, increment)
                            points += scan_points_in_base(*scan, .05, 10., .46, side * .26,
                                                          -.58, .64, -.4, .4, .02)
                            def sector(center, width):
                                return safety_clearances_in_sector(
                                    *scan, center, width, .05, 10., .46, side * .26,
                                    -.58, .64, -.4, .4, .02)
                            body += sector(0, math.pi)
                            if side > 0:
                                left += sector(math.pi / 2, .8)
                            else:
                                right += sector(-math.pi / 2, .8)
                            front += safety_clearances_in_forward_corridor(
                                *scan, .05, 10., .46, side * .26, .64, -.4, .4, .4)
                        v, w, active = controller.assist(.833, turn, *split_wall_points(points), points, **kwargs)
                        self.assertTrue(active)
                        v = limit_forward_speed_for_wall_follow(v, w, front, body, True,
                                                               .1, .9, .2, .05, .35, .12)
                        w = limit_turn_speed_for_forward_arc(w, v, left if w > 0 else right,
                                                             True, .1, .9, .2)
                        self.assertGreater(v, 0.0, f'stopped at {step * .1:.1f}s')
                        # Twist linear.x is the rear axle speed, 0.33 m behind base_link.
                        y += (v * math.sin(yaw) + .33 * w * math.cos(yaw)) * .1
                        yaw += w * .1
                        travel += v * .1
                        clearance = min(1.29 - abs(y + x * math.sin(yaw) + lateral * math.cos(yaw))
                                        for x in (-.58, .64) for lateral in (-.4, .4))
                        self.assertGreater(clearance, .05)
                    self.assertGreater(travel, 4.0)
                    self.assertLess(abs(yaw), .03)
                    self.assertGreater(v, .40)


if __name__ == '__main__':
    unittest.main()
