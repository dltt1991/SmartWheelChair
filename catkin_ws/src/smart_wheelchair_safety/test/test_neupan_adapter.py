import unittest
import numpy as np
from smart_wheelchair_safety.neupan_adapter import NeuPANAdapter


class Planner:
    def __call__(self, state, points, velocities=None):
        return np.array([[2.], [-2.]]), {"opt_state_list": [np.array([[0.], [0.], [0.]])]} 


class AdapterTest(unittest.TestCase):
    def test_filters_and_merges_points(self):
        a = NeuPANAdapter(planner=Planner(), max_range=2)
        a.set_obstacles([[1, 0], [np.nan, 0], [3, 0]])
        self.assertEqual(a.obstacles.tolist(), [[1., 0.], [3., 0.]])
        self.assertEqual(NeuPANAdapter.merge_scans([[1, 2]], [[3, 4]]).shape, (2, 2))

    def test_path_normalizes_duplicates(self):
        a = NeuPANAdapter(planner=Planner())
        a.set_initial_path([[0, 0], [0, 0], [1, 0]])
        self.assertEqual(a.initial_path.tolist(), [[0., 0.], [1., 0.]])

    def test_action_clipped_and_trajectory_returned(self):
        a = NeuPANAdapter(planner=Planner())
        a.set_obstacles([[1, 0]])
        action, trajectory = a.step([0, 0, 0])
        self.assertEqual(action.tolist(), [.8, -.65])
        self.assertEqual(trajectory.shape, (1, 3))

    def test_stale_returns_zero(self):
        a = NeuPANAdapter(planner=Planner(), action_timeout=.1)
        a.set_obstacles([[1, 0]])
        action, trajectory = a.step([0, 0, 0], now=1.0)
        self.assertTrue(np.allclose(action, 0))
        self.assertEqual(len(trajectory), 0)


if __name__ == '__main__':
    unittest.main()
