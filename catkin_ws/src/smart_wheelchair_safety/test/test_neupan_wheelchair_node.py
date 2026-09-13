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
        node._solve_requested = threading.Event()
        node._shutdown = threading.Event()
        node.input_timeout = 0.25
        node.use_sim_time = False
        node._reference = (10.0, 10.0, np.array([[10.0, 20.0, 0.0], [11.0, 20.0, 0.0]]))
        node._reference_speed = (10.0, 10.0, .8)
        node._odom = (10.0, 10.0, np.array([10.0, 20.0, 0.0]))
        node._scans = {
            side: (10.0, 10.0, np.array([[12.0, 20.0]])) for side in ("left", "right")
        }
        node._ros_now = lambda: self.now
        node.adapter = NS(
            reason="ok",
            max_linear=.8,
            set_reference_speed=lambda speed: setattr(node.adapter, 'reference_speed', speed),
            set_obstacles=lambda *a, **kw: None,
            set_initial_path=lambda *a: None,
            step=lambda *a, **kw: (np.array([0.4, 0.1]), np.array([[10.0, 20.0, 0.0]])),
        )
        self.results = []
        node._publish = lambda *args: self.results.append(args)
        return node

    def test_sim_clock_pause_does_not_expire_ros_fresh_snapshot(self):
        node=self.node();node.use_sim_time=True
        self.now=10.268609
        node._ros_now=lambda:10.2
        values=(node._reference,node._odom,*node._scans.values(),node._reference_speed)
        values=tuple((*value,10.,index in (1,2,3)) for index,value in enumerate(values))
        self.assertTrue(node._fresh(values))
        node._ros_now=lambda:10.251
        self.assertFalse(node._fresh(values))
        node.use_sim_time=False;node._ros_now=lambda:10.2
        self.assertFalse(node._fresh(values))

    def test_future_sim_sensor_tolerance_never_applies_to_reference(self):
        node=self.node();node.use_sim_time=True
        node._ros_now=lambda:10.
        for future,expected in ((.008,True),(.049,True),(.051,False)):
            value=(10.,10.+future,np.zeros(2),10.,True)
            self.assertEqual(node._fresh((value,)),expected)
        self.assertFalse(node._fresh(((10.,10.008,np.zeros(2),10.,False),)))
        node.use_sim_time=False
        self.assertFalse(node._fresh(((10.,10.008,np.zeros(2),10.,True),)))

    def start_worker(self, node):
        node._worker = threading.Thread(target=node._run_solver, daemon=True)
        node._worker.start()
        self.addCleanup(node.close)

    def path(self, stamp):
        poses = [NS(pose=NS(position=NS(x=x, y=20.),
                            orientation=NS(x=0., y=0., z=0., w=1.))) for x in (10., 11.)]
        return NS(header=NS(stamp=stamp, frame_id='odom'), poses=poses)

    def test_reference_pair_wakes_worker_without_timer_in_either_order(self):
        for speed_first in (False, True):
            node = self.node()
            node._reference = node._reference_speed = None
            solved = threading.Event()
            threads = []
            def solve(*args, **kwargs):
                threads.append(threading.current_thread())
                solved.set()
                return np.array([.4, .1]), np.empty((0, 3))
            node.adapter.step = solve
            self.start_worker(node)
            speed = NS(header=NS(stamp=10., frame_id='odom'), twist=NS(linear=NS(x=.8)))
            callbacks = [(node.on_reference, self.path(10.)), (node.on_reference_speed, speed)]
            for callback, message in callbacks[::-1] if speed_first else callbacks:
                callback(message)
            self.assertTrue(solved.wait(1.))
            self.assertTrue(threads and all(thread is node._worker for thread in threads))
            node.close()

    def test_busy_worker_coalesces_timer_requests(self):
        node = self.node()
        entered, release, second, extra = [threading.Event() for _ in range(4)]
        calls = []
        def solve(*args, **kwargs):
            calls.append(1)
            if len(calls) == 1:
                entered.set()
                release.wait(1.)
            elif len(calls) == 2:
                second.set()
            else:
                extra.set()
            return np.array([.4, .1]), np.empty((0, 3))
        node.adapter.step = solve
        self.start_worker(node)
        node.request_solve()
        self.assertTrue(entered.wait(1.))
        for _ in range(100):
            node.request_solve()
        release.set()
        self.assertTrue(second.wait(1.))
        self.assertFalse(extra.wait(.05))
        self.assertEqual(len(calls), 2)

    def test_speed_requires_matching_fresh_reference_and_sane_value(self):
        for stamp, speed, frame in ((9.9, .35, 'odom'), (10., float('nan'), 'odom'),
                                    (10., -.1, 'odom'), (10., .81, 'odom'),
                                    (10., .35, 'base_link')):
            node = self.node()
            node.on_reference_speed(NS(header=NS(stamp=stamp, frame_id=frame),
                                       twist=NS(linear=NS(x=speed))))
            node.tick()
            np.testing.assert_allclose(self.results[-1][0], [0., 0.])
            self.assertEqual(self.results[-1][3], 'invalid reference speed')
        for value in (None, (9., 10., .35)):
            node = self.node()
            node._reference_speed = value
            node.tick()
            self.assertEqual(self.results[-1][3], 'invalid reference speed')

    def test_reference_speed_updates_upstream(self):
        node = self.node()
        for speed in (.35, .8):
            node.on_reference_speed(NS(header=NS(stamp=10., frame_id='odom'),
                                       twist=NS(linear=NS(x=speed))))
            node.tick()
            self.assertEqual(node.adapter.reference_speed, speed)
            self.assertEqual(self.results[-1][2], 10.)

    def test_speed_replacement_during_solve_revokes_result(self):
        node = self.node()
        def solve(*args, **kwargs):
            node.on_reference_speed(NS(header=NS(stamp=10.1, frame_id='odom'),
                                       twist=NS(linear=NS(x=.8))))
            return np.array([.4, .1]), np.empty((0, 3))
        node.adapter.step = solve
        node.tick()
        self.assertIsNone(self.results[-1][2])
        np.testing.assert_allclose(self.results[-1][0], [0., 0.])

    def test_speed_arriving_before_path_waits_for_matching_stamp(self):
        node = self.node()
        self.now = 10.1
        node.on_reference_speed(NS(header=NS(stamp=10.1, frame_id='odom'),
                                   twist=NS(linear=NS(x=.35))))
        node.tick()
        self.assertEqual(self.results[-1][3], 'invalid reference speed')
        node._reference = (10.1, 10.1, node._reference[2])
        node.tick()
        self.assertEqual(self.results[-1][2], 10.1)
        self.assertEqual(node.adapter.reference_speed, .35)

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

    def test_genuine_stop_retains_valid_reference(self):
        node = self.node()
        node.adapter.step = lambda *a, **kw: (np.zeros(2), np.empty((0, 3)))
        node.tick()
        self.assertEqual(self.results[-1][2], 10.)

    def test_reference_cleared_while_solving_rejects_motion(self):
        node = self.node()
        def solve(*args, **kwargs):
            node._reference = None
            return np.array([.4, .1]), np.empty((0, 3))
        node.adapter.step = solve
        node.tick()
        self.assertIsNone(self.results[-1][2])

    def test_receipt_of_old_or_future_stamp_does_not_refresh_input(self):
        node = self.node()
        for stamp in (9.0, 11.0, 0.0):
            node._odom = (10.0, stamp, np.zeros(3))
            node.tick()
            self.assertIsNone(self.results[-1][2])


if __name__ == "__main__":
    unittest.main()
