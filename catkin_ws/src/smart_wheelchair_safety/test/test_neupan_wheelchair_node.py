import math
import threading
import unittest
from types import SimpleNamespace as NS

import numpy as np

from smart_wheelchair_safety.neupan_wheelchair_node import (
    NeuPANWheelchairNode,
    scan_points,
    transform_points,
    reference_points,
)


class NodeTest(unittest.TestCase):
    def test_scan_fov_range_and_finite_filter(self):
        scan = NS(
            ranges=[1.0] * 37,
            angle_min=-math.pi,
            angle_increment=math.pi / 18,
            range_min=0.1,
            range_max=10.0,
        )
        for side, bounds in [("left", (-30, 170)), ("right", (-170, 30))]:
            points = scan_points(scan, side, 5.0)
            self.assertEqual(len(points), 21)
            angles = np.degrees(np.arctan2(points[:, 1], points[:, 0]))
            self.assertAlmostEqual(angles.min(), bounds[0])
            self.assertAlmostEqual(angles.max(), bounds[1])
        scan.ranges = [float("nan"), float("inf"), 0.01, 6.0]
        self.assertEqual(len(scan_points(scan, "right", 5.0)), 0)

    def test_world_transform_and_reference(self):
        transform = NS(
            translation=NS(x=10.0, y=20.0),
            rotation=NS(x=0.0, y=0.0, z=math.sin(math.pi / 4), w=math.cos(math.pi / 4)),
        )
        np.testing.assert_allclose(
            transform_points([[1.0, 0.0]], transform), [[10.0, 21.0]]
        )
        pose = NS(position=NS(x=10.0, y=21.0), orientation=transform.rotation)
        np.testing.assert_allclose(
            reference_points(NS(poses=[NS(pose=pose)])), [[10.0, 21.0, math.pi / 2]]
        )

    def node(self):
        node = NeuPANWheelchairNode.__new__(NeuPANWheelchairNode)
        self.now = 10.0
        node._clock = lambda: self.now
        node._lock = threading.Lock()
        node._solve_lock = threading.Lock()
        node.input_timeout = 0.25
        node._reference = (10.0, 10.0, np.array([[10.0, 20.0, 0.0], [11.0, 20.0, 0.0]]))
        node._odom = (10.0, 10.0, np.array([10.0, 20.0, 0.0]))
        node._scans = {
            side: (10.0, 10.0, np.array([[12.0, 20.0]])) for side in ("left", "right")
        }
        node._ros_now = lambda: self.now
        node.adapter = NS(
            reason="ok",
            set_obstacles=lambda *a, **kw: None,
            set_initial_path=lambda *a: None,
            step=lambda *a, **kw: (np.array([0.4, 0.1]), np.array([[10.0, 20.0, 0.0]])),
        )
        self.results = []
        node._publish = lambda *args: self.results.append(args)
        return node

    def test_fresh_result_keeps_reference_stamp(self):
        node = self.node()
        node.tick()
        np.testing.assert_allclose(self.results[-1][0], [0.4, 0.1])
        self.assertEqual(self.results[-1][2], 10.0)

    def test_missing_or_stale_each_input_invalidates_action(self):
        for field in ("reference", "odom", "left", "right"):
            node = self.node()
            if field in ("left", "right"):
                node._scans[field] = (9.0, 9.0, np.empty((0, 2)))
            else:
                setattr(node, "_" + field, None)
            node.tick()
            np.testing.assert_allclose(self.results[-1][0], [0.0, 0.0])
            self.assertIsNone(self.results[-1][2])

    def test_expired_during_solve_invalidates_action(self):
        node = self.node()

        def slow(*args, **kwargs):
            self.now += 0.3
            return np.array([0.4, 0.1]), np.array([[10.0, 20.0, 0.0]])

        node.adapter.step = slow
        node.tick()
        self.assertIsNone(self.results[-1][2])
        np.testing.assert_allclose(self.results[-1][0], [0.0, 0.0])

    def test_busy_solver_does_not_enqueue_work(self):
        node = self.node()
        node._solve_lock.acquire()
        node.tick()
        self.assertEqual(self.results, [])

    def test_receipt_of_old_or_future_stamp_does_not_refresh_input(self):
        node = self.node()
        for stamp in (9.0, 11.0, 0.0):
            node._odom = (10.0, stamp, np.zeros(3))
            node.tick()
            self.assertIsNone(self.results[-1][2])


if __name__ == "__main__":
    unittest.main()
