#!/usr/bin/env python3
import unittest

import numpy as np

from smart_wheelchair_safety.local_path_follower import (
    _path_score, _rollout, _trajectory_clear, select_velocity)


def scalar_selection(path, obstacles, speed_limit, previous):
    candidates = [(v, w) for v in np.linspace(0., min(speed_limit, .8), 9)
                  for w in np.linspace(-.65, .65, 15)]
    safe = [command for command in candidates
            if _trajectory_clear(_rollout(*command), obstacles)]
    return (np.asarray(min(safe, key=lambda command:
            _path_score(_rollout(*command), path)
            + .05*np.linalg.norm(np.asarray(command)-previous)))
            if safe else np.zeros(2))


class LocalPathFollowerTest(unittest.TestCase):
    def test_selection_matches_scalar_grid_with_varied_scans_and_paths(self):
        rng = np.random.RandomState(42)
        for case in range(16):
            angles = np.linspace(0., (-1)**case*.07*case, 31)
            path = np.column_stack((np.linspace(0., 3., 31),
                                    .6*np.sin(angles), angles))
            obstacles = rng.uniform([-2., -2.], [3., 2.], (case*17, 2))
            # Keep the initial footprint free except in deliberate stop cases.
            if case % 4:
                obstacles = obstacles[np.abs(obstacles[:, 1]) > .48]
            speed = [.25, .6, .8, 1.2][case % 4]
            previous = rng.uniform([0., -.65], [.8, .65])
            with self.subTest(case=case):
                np.testing.assert_array_equal(
                    select_velocity(path, obstacles, speed, previous),
                    scalar_selection(path, obstacles, speed, previous))

    def test_previous_velocity_resolves_near_equal_path_scores(self):
        angular = np.linspace(-.65, .65, 15)[8]
        path = np.array([[0., 0., angular/2.]])
        for previous in ([0., 0.], [0., angular]):
            with self.subTest(previous=previous):
                np.testing.assert_array_equal(
                    select_velocity(path, np.empty((0, 2)), .8, previous), previous)

    def test_selection_matches_scalar_at_footprint_edges(self):
        path = _rollout(.8, .65)
        for x, y in ((-.29, .44), (1.01, .44), (1.01, -.44), (-.29, -.44)):
            for direction in (-np.inf, np.inf):
                obstacles = np.array([[np.nextafter(x, direction), y]])
                with self.subTest(point=obstacles.tolist()):
                    np.testing.assert_array_equal(
                        select_velocity(path, obstacles, .8),
                        scalar_selection(path, obstacles, .8, np.zeros(2)))

    def test_straight_path_selects_forward_motion(self):
        path = np.column_stack((np.linspace(0., 3., 31), np.zeros(31), np.zeros(31)))
        command = select_velocity(path, np.empty((0, 2)), .8)
        self.assertGreater(command[0], .2)
        self.assertAlmostEqual(command[1], 0., delta=.11)

    def test_left_curve_selects_positive_angular_velocity(self):
        angles = np.linspace(0., .7, 31)
        path = np.column_stack((2*np.sin(angles), 2*(1-np.cos(angles)), angles))
        command = select_velocity(path, np.empty((0, 2)), .6)
        self.assertGreater(command[0], 0.)
        self.assertGreater(command[1], .1)

    def test_speed_limit_is_absolute(self):
        path = np.column_stack((np.linspace(0., 3., 31), np.zeros(31), np.zeros(31)))
        self.assertLessEqual(select_velocity(path, np.empty((0, 2)), .25)[0], .25)

    def test_obstacle_across_footprint_stops(self):
        path = np.column_stack((np.linspace(0., 3., 31), np.zeros(31), np.zeros(31)))
        obstacles = np.array([[x, y] for x in np.linspace(.9, 1.2, 5)
                              for y in np.linspace(-.4, .4, 9)])
        np.testing.assert_allclose(select_velocity(path, obstacles, .8), [0., 0.])

    def test_invalid_or_empty_inputs_stop(self):
        np.testing.assert_allclose(select_velocity(np.empty((0, 3)),
                                                   np.empty((0, 2)), .8), [0., 0.])
        path = np.array([[0., 0., float("nan")]])
        np.testing.assert_allclose(select_velocity(path, np.empty((0, 2)), .8), [0., 0.])


if __name__ == '__main__':
    import rostest
    rostest.rosrun('smart_wheelchair_safety', 'local_path_follower', LocalPathFollowerTest)
