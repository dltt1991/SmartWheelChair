import unittest

import numpy as np

from smart_wheelchair_safety.local_path_follower import select_velocity


class LocalPathFollowerTest(unittest.TestCase):
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
