import unittest
from types import SimpleNamespace
import numpy as np
from smart_wheelchair_safety.neupan_adapter import NeuPANAdapter

try:
    from neupan.blocks.initial_path import InitialPath
except ImportError:
    InitialPath = None


class Planner:
    def __init__(self):
        self.ipath = SimpleNamespace(arrive_flag=True)
        self.info = {"arrive": True}

    def set_initial_path(self, path):
        self.path = path

    def __call__(self, state, points, velocities=None):
        return np.array([[2.], [-2.]]), {"opt_state_list": [np.array([[0.], [0.], [0.]])]} 


class AdapterTest(unittest.TestCase):
    def test_reference_speed_updates_planner_and_bounds_action(self):
        planner = Planner()
        planner.set_reference_speed = lambda speed: setattr(planner, 'ref_speed', speed)
        a = NeuPANAdapter(planner=planner)
        for speed in (.35, .8, 0.):
            a.set_reference_speed(speed)
            self.assertEqual(planner.ref_speed, speed)
            a.set_obstacles([], now=10.)
            action, _ = a.step([0., 0., 0.], now=10.)
            self.assertEqual(action[0], speed)
        for invalid in (-.1, float('nan'), float('inf'), .81):
            with self.assertRaises(ValueError):
                a.set_reference_speed(invalid)

    @unittest.skipIf(InitialPath is None, 'requires the pinned NeuPAN runtime')
    def test_real_upstream_reference_progress_and_stationary_turn(self):
        class ReferencePlanner(Planner):
            def __init__(self):
                super().__init__()
                self.ipath = InitialPath(10, .1, .5, SimpleNamespace(kinematics='diff', L=0.))

            def set_initial_path(self, path):
                self.ipath.set_initial_path(path)

            def __call__(self, state, points):
                self.ipath.check_arrive(state)
                return np.zeros((2, 1)), {'opt_state_list': [state]}

        a = NeuPANAdapter(planner=ReferencePlanner())
        path = np.column_stack((np.linspace(0., 3., 121), np.zeros((121, 2))))
        for x in (1.5, 1.6, 1.7):
            a.set_initial_path(path.copy())
            a.set_obstacles([], now=10.)
            a.step([x, 0., 0.], now=10.)
            self.assertGreaterEqual(a.planner.ipath.cur_point[0, 0], x-.03)
        a.step([2.98, 0., 0.], now=10.)
        self.assertTrue(a.planner.ipath.arrive_flag)
        a.set_initial_path(path+1e-10)
        self.assertTrue(a.planner.ipath.arrive_flag)
        # A refreshed path must re-localize, including a changed travel heading.
        path[:, 1] = .01
        path[:, 2] = np.pi
        a.set_initial_path(path)
        a.step([2., .01, -np.pi], now=10.)
        self.assertAlmostEqual(a.planner.ipath.cur_point[0, 0], 2.)
        # Equal positions encode a turn, so XY-only nearest-point search stalls.
        pivot = np.column_stack((np.zeros((51, 2)), np.linspace(0., 1., 51)))
        a.set_initial_path(np.vstack((pivot, [[1., 0., 1.]])))
        a.step([0., 0., .6], now=10.)
        self.assertAlmostEqual(a.planner.ipath.cur_point[2, 0], .6)

    def test_filters_and_merges_points(self):
        a = NeuPANAdapter(planner=Planner(), max_range=2)
        a.set_obstacles([[1, 0], [np.nan, 0], [3, 0]])
        self.assertEqual(a.obstacles.tolist(), [[1., 0.], [3., 0.]])
        self.assertEqual(NeuPANAdapter.merge_scans([[1, 2]], [[3, 4]]).shape, (2, 2))

    def test_path_normalizes_duplicates(self):
        a = NeuPANAdapter(planner=Planner())
        a.set_initial_path([[0, 0], [0, 0], [1, 0]])
        self.assertEqual(a.initial_path.tolist(), [[0., 0.], [1., 0.]])
        self.assertEqual(a.planner.path[0].shape, (4, 1))
        self.assertFalse(a.planner.info["arrive"])

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

    def test_nan_action_and_inference_failure_stop(self):
        for action in ([float('nan'), 0], None):
            class Invalid(Planner):
                def __call__(self, state, points):
                    if action is None:
                        raise RuntimeError('failure')
                    return action, {}
            a = NeuPANAdapter(planner=Invalid())
            a.set_obstacles([], now=10)
            command, trajectory = a.step([0, 0, 0], now=10)
            self.assertTrue(np.allclose(command, 0))
            self.assertEqual(len(trajectory), 0)
            self.assertIn('inference failed', a.reason)

    def test_missing_config_does_not_import_or_train(self):
        a = NeuPANAdapter()
        self.assertFalse(a.available)

    def test_explicit_heading_and_stationary_turn_preserved(self):
        a = NeuPANAdapter(planner=Planner())
        path = [[0, 0, .1], [0, 0, .9], [1, 1, 1.1]]
        a.set_initial_path(path)
        np.testing.assert_allclose(np.stack(a.planner.path)[:, :3, 0], path)

    def test_nonfinite_state_and_bad_trajectory_rejected(self):
        a = NeuPANAdapter(planner=Planner())
        a.set_obstacles([], now=10)
        command, _ = a.step([0, float('nan'), 0], now=10)
        self.assertTrue(np.allclose(command, 0))
        with self.assertRaises(ValueError):
            a.set_initial_path([[0, 0, float('nan')]])
        with self.assertRaises(ValueError):
            a.set_obstacles([[1, 2, 3]])
        class Bad(Planner):
            def __call__(self, state, points):
                return [.5, 0], {'opt_state_list': [[1, 2, 3, 4]]}
        a.planner = Bad()
        command, _ = a.step([0, 0, 0], now=10)
        self.assertTrue(np.allclose(command, 0))

    def test_planner_stop_overrides_action(self):
        class Stopped(Planner):
            def __call__(self, state, points):
                return [.5, .1], {'stop': True}
        a = NeuPANAdapter(planner=Stopped())
        a.set_obstacles([], now=10)
        command, _ = a.step([0, 0, 0], now=10)
        self.assertTrue(np.allclose(command, 0))


if __name__ == '__main__':
    unittest.main()
