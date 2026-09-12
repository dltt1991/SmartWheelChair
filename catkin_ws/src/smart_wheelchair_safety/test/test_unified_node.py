#!/usr/bin/env python3
import importlib.util
import json
import math
import os
import threading
import time
from types import SimpleNamespace
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np


ROS_AVAILABLE = importlib.util.find_spec('rospy') is not None
if ROS_AVAILABLE:
    import rospy
    # Spawn imports the test entry point; the geometry worker must not register
    # the rostest-remapped ROS node name again and shut down its parent.
    if __name__ != '__mp_main__':
        rospy.init_node('test_unified_control', anonymous=True, disable_signals=True)


def parameter(node, name):
    return node.parameter(name)


@unittest.skipUnless(ROS_AVAILABLE, 'requires ROS')
class UnifiedNodeTest(unittest.TestCase):
    def setUp(self):
        from smart_wheelchair_safety.unified_control_node import UnifiedControlNode
        with patch.object(rospy, 'Timer') as timer, patch.object(
                rospy, 'Subscriber', wraps=rospy.Subscriber) as subscriber:
            self.node = UnifiedControlNode()
        self.timer_calls = timer.call_args_list
        self.callbacks = {call.args[0]: call.args[2] for call in subscriber.call_args_list}
        self.commands = []
        self.node.pub = SimpleNamespace(publish=self.commands.append)
        self.node.pose = (0., 0., 0.)
        now = time.monotonic()
        self.node.raw_time = self.node.odom_time = self.node.plan_time = now
        self.node.odom_stamp = rospy.Time.now().to_sec()
        self.node.scans = {side: (now, np.array([[3., 2.], [3., -2.]])) for side in ('left', 'right')}
        self.node.raw = np.array([.5, 0.])
        self.node.planned = np.array([.5, 0.])
        self.node.mode = 'wall'

    def tearDown(self):
        self.node.destroy_node()

    def test_neupan_fresh_action_uses_existing_guard(self):
        from geometry_msgs.msg import TwistStamped
        self.node.use_neupan = True
        self.node.accept_planned = True
        self.node.active_reference_stamp = rospy.Time.now()
        message = TwistStamped()
        message.header.stamp = self.node.active_reference_stamp
        message.twist.linear.x = .3
        message.twist.angular.z = .1
        self.node.on_neupan(message)
        self.node.plan_time = 0.
        self.node.last_tick = rospy.Time.now().to_sec()-.05
        self.node.control()
        self.assertGreater(self.commands[-1].linear.x, 0.)
        self.assertGreater(self.commands[-1].angular.z, 0.)
        with patch('smart_wheelchair_safety.unified_control_node.braking_clear', return_value=False):
            self.node.control()
        self.assertEqual(self.commands[-1].linear.x, 0.)

    def test_neupan_rejects_wrong_reference_and_expired_action(self):
        from geometry_msgs.msg import TwistStamped
        self.node.use_neupan = True
        self.node.accept_planned = True
        self.node.active_reference_stamp = rospy.Time.now()
        message = TwistStamped()
        message.twist.linear.x = .8
        self.node.on_neupan(message)
        self.assertFalse(self.node.neupan_fresh(time.monotonic()))
        message.header.stamp = self.node.active_reference_stamp
        self.node.on_neupan(message)
        self.assertTrue(self.node.neupan_fresh(time.monotonic()))
        self.node.neupan_time -= 1.
        self.assertFalse(self.node.neupan_fresh(time.monotonic()))
        self.node.control()
        self.assertNotEqual(self.node.reason, 'planner_timeout')
        self.node.cancel()
        self.assertFalse(self.node.neupan_fresh(time.monotonic()))

    def test_door_search_runs_outside_control_interpreter(self):
        from smart_wheelchair_safety.unified_geometry import collision_aware_door_reference
        worker_pid = self.node.door_plan_executor.submit(os.getpid).result(timeout=10.)
        self.assertNotEqual(worker_pid, os.getpid())
        door = self.front_opening()
        result = self.node.door_plan_executor.submit(
            collision_aware_door_reference, door, np.empty((0, 2))).result(timeout=10.)
        np.testing.assert_allclose(result, collision_aware_door_reference(door, np.empty((0, 2))))

    def test_broken_search_worker_does_not_kill_reference_callback(self):
        from concurrent.futures import Future
        from concurrent.futures.process import BrokenProcessPool
        self.node.door_phase = 'door_align'
        failed = Future()
        failed.set_exception(BrokenProcessPool('worker exited'))
        self.node.door_plan_request = (failed, self.node.pose, self.node.door_generation)
        with patch.object(self.node.door_plan_executor, 'submit',
                          side_effect=BrokenProcessPool('pool unavailable')):
            for _ in range(2):
                self.node._local_door_path(self.front_opening(), np.empty((0, 2)))
                self.assertFalse(self.node.door_path_feasible)
                self.assertIsNone(self.node.door_path)
                self.assertIsNone(self.node.door_plan_request)

    def test_recovery_uses_capped_joystick_even_when_planner_is_blocked(self):
        for mode in ('wall', 'door_wait', 'door_align'):
            with self.subTest(mode=mode):
                self.node.mode = mode
                self.node.door_path_feasible = True
                self.node.plan_time = 0.
                self.node.planned[:] = 0.
                self.node.raw = np.array([.5, 0.])
                self.node.output[:] = 0.
                self.node.scans = {side: (time.monotonic(), np.array([[-.28, 0.]]))
                                   for side in ('left', 'right')}
                self.node.last_tick = rospy.Time.now().to_sec()-.05
                self.node.control()
                self.assertGreater(self.commands[-1].linear.x, 0.)
                self.assertLessEqual(self.commands[-1].linear.x, .05)
                self.assertEqual(self.commands[-1].angular.z, 0.)
                self.assertEqual(self.node.reason, 'clearance_recovery')

    def test_near_obstacle_recovery_does_not_start_distant_door_searches(self):
        self.node.raw = np.array([.5, 0.])
        self.node.scans = {side: (time.monotonic(), np.array([[-.28, 0.]]))
                           for side in ('left', 'right')}
        with patch.object(self.node, '_intended_door', return_value=(None, None)) as search:
            self.node.update_reference()
            search.assert_not_called()
        self.assertEqual(self.node.mode, 'clearance_recovery')
        self.assertFalse(self.node.accept_planned)

    def test_recovery_still_rejects_inward_stale_and_neutral_commands(self):
        self.node.scans = {side: (time.monotonic(), np.array([[1., 0.]]))
                           for side in ('left', 'right')}
        for raw, stale in (([.5, 0.], False), ([-.4, 0.], True), ([0., 0.], False)):
            with self.subTest(raw=raw, stale=stale):
                self.node.raw = np.array(raw)
                self.node.odom_time = 0. if stale else time.monotonic()
                self.node.control()
                self.assertEqual(self.commands[-1].linear.x, 0.)
                self.assertEqual(self.commands[-1].angular.z, 0.)

    def test_recovery_is_stably_limited_and_exits_with_clearance(self):
        self.node.raw = np.array([-.4, 0.])
        self.node.scans = {side: (time.monotonic(), np.array([[1., 0.]]))
                           for side in ('left', 'right')}
        for _ in range(30):
            self.node.raw_time = self.node.odom_time = time.monotonic()
            self.node.odom_stamp = rospy.Time.now().to_sec()
            self.node.last_tick = self.node.odom_stamp-.05
            self.node.control()
            self.node.measured = self.node.output.copy()
            self.assertLessEqual(abs(self.commands[-1].linear.x), .05)
            self.assertLess(self.commands[-1].linear.x, 0.)
        self.node.scans = {side: (time.monotonic(), np.array([[1.06, 0.]]))
                           for side in ('left', 'right')}
        self.node.last_tick = rospy.Time.now().to_sec()-.05
        self.node.control()
        self.assertFalse(self.node.recovering)
        self.assertLess(self.commands[-1].linear.x, -.05)

    def test_stale_input_stops_without_losing_recovery_hysteresis(self):
        self.node.recovering = True
        self.node.raw = np.array([-.4, 0.])
        self.node.odom_time = 0.
        self.node.control()
        self.assertTrue(self.node.recovering)
        self.assertEqual(self.node.reason, 'stale_input')
        self.assertEqual(self.commands[-1].linear.x, 0.)
        self.node.odom_time = time.monotonic()
        self.node.odom_stamp = rospy.Time.now().to_sec()
        self.node.scans = {side: (time.monotonic(), np.array([[1.02, 0.]]))
                           for side in ('left', 'right')}
        self.node.control()
        self.assertTrue(self.node.recovering)

    def test_stop_infeasible_margin_boundary_can_enter_recovery(self):
        self.node.recovering = False
        self.node.measured = np.array([.00013, .00024])
        self.node.odom_stamp = rospy.Time.now().to_sec()-.02
        self.assertTrue(self.node._near_recovery(np.array([[1.01003, 0.]])))
        self.assertFalse(self.node._near_recovery(np.array([[1.1, 0.]])))

    def test_recovery_start_does_not_repeat_infeasible_tiny_jerk_candidate(self):
        self.node.raw = np.array([-.4, 0.])
        self.node.measured = np.array([.002, 0.])
        points = np.array([[1.01058, 0.]])
        for _ in range(3):
            now, stamp = time.monotonic(), rospy.Time.now().to_sec()
            self.node.raw_time = self.node.odom_time = now
            self.node.odom_stamp = stamp-.08
            self.node.last_tick = stamp-.05
            self.node.scans = {side: (now, points) for side in ('left', 'right')}
            previous = self.node.output.copy()
            self.node.control()
            self.assertLess(self.node.output[0], -.00625)
            self.assertTrue(np.all(np.abs(self.node.output-previous) <= np.array([.5, .8])*.051))
            self.assertLessEqual(abs(self.node.output[0]), .05)

    def test_recovery_start_candidates_still_reject_a_new_rear_obstacle(self):
        self.node.raw = np.array([-.4, 0.])
        self.node.measured = np.array([.002, 0.])
        self.node.odom_stamp = rospy.Time.now().to_sec()-.08
        self.node.last_tick = rospy.Time.now().to_sec()-.05
        self.node.scans = {side: (time.monotonic(), np.array([[1.01058, 0.], [-.2901, 0.]]))
                           for side in ('left', 'right')}
        self.node.control()
        np.testing.assert_array_equal(self.node.output, [0., 0.])

    def test_forward_recovery_waits_for_a_moving_planner_before_handoff(self):
        self.node.recovering = True
        self.node.raw = np.array([.5, 0.])
        self.node.planned[:] = 0.
        self.node.mode = 'door_wait'
        self.node.scans = {side: (time.monotonic(), np.array([[-.33, 0.]]))
                           for side in ('left', 'right')}
        self.node.last_tick = rospy.Time.now().to_sec()-.05
        self.node.control()
        self.assertTrue(self.node.recovering)
        self.assertGreater(self.commands[-1].linear.x, 0.)
        self.node.planned = np.array([.2, 0.])
        self.node.plan_time = time.monotonic()
        self.node.mode = 'manual'
        self.node.control()
        self.assertFalse(self.node.recovering)

    def test_recovery_hands_clear_space_back_to_override_and_front_stop(self):
        points = np.array([[-.33, 0.]])
        for mode, override in (('front_stop', False), ('override', True)):
            with self.subTest(mode=mode):
                self.node.recovering = True
                self.node.planned[:] = 0.
                self.node.plan_time = 0.
                self.node.mode, self.node.override = mode, override
                self.assertFalse(self.node._near_recovery(points))

    def test_curved_recovery_can_request_a_door_plan(self):
        from concurrent.futures import Future
        self.node.recovering = True
        self.node.measured = self.node.output = np.array([.04, .08])
        self.node.door_phase = 'door_align'
        with patch.object(self.node.door_plan_executor, 'submit', return_value=Future()) as submit:
            self.node._local_door_path(self.front_opening(heading=.5), np.array([[1., .43]]))
            submit.assert_called_once()

    def test_recovery_slows_while_a_door_search_is_pending(self):
        from concurrent.futures import Future
        self.node.recovering = True
        self.node.raw = np.array([.5, .2])
        self.node.planned[:] = 0.
        self.node.mode = 'door_wait'
        self.node.door_plan_request = (Future(), self.node.pose, self.node.door_generation)
        self.node.scans = {side: (time.monotonic(), np.array([[-.33, 0.]]))
                           for side in ('left', 'right')}
        self.node.last_tick = rospy.Time.now().to_sec()-.1
        self.node.control()
        self.assertGreater(self.commands[-1].linear.x, 0.)
        self.assertLessEqual(self.commands[-1].linear.x, .01+1e-12)
        self.assertLessEqual(abs(self.commands[-1].angular.z), .02)

    def test_door_pivot_stays_latched_across_small_position_drift(self):
        for side in (-1., 1.):
            with self.subTest(side=side):
                self.node.mode = self.node.door_phase = 'door_align'
                self.node.door_path_feasible = True
                self.node.door_rotation_index = 0
                self.node.door_path = np.array([[0., 0., 0.], [.36, 0., 0.],
                                                [.36, 0., side*math.pi/2], [.36, side, side*math.pi/2]])
                self.node.pose = (.291, 0., side*1.45)
                first = self.node._safe_door_command(np.empty((0, 2)), self.node.odom_stamp)
                self.assertEqual(first[0], 0.)
                self.assertGreater(side*first[1], 0.)
                self.node.pose = (.290, side*.015, side*1.49)
                second = self.node._safe_door_command(np.empty((0, 2)), self.node.odom_stamp)
                self.assertEqual(second[0], 0.)
                self.assertGreater(side*second[1], 0.)
                self.node.pose = (.290, side*.015, side*math.pi/2)
                self.assertIsNone(self.node._pending_door_rotation())

    def test_excessive_pivot_drift_stops_and_invalidates_the_path(self):
        self.node.mode = self.node.door_phase = 'door_align'
        self.node.door_path_feasible = True
        self.node.door_path = np.array([[0., 0., 0.], [.36, 0., 0.], [.36, 0., math.pi/2]])
        self.node.pose = (.291, 0., 1.)
        self.node._safe_door_command(np.empty((0, 2)), self.node.odom_stamp)
        self.node.pose = (.20, 0., 1.1)
        np.testing.assert_array_equal(self.node._safe_door_command(np.empty((0, 2)), self.node.odom_stamp), [0., 0.])
        self.assertFalse(self.node.door_path_feasible)

    def test_invalidated_pivot_with_residual_output_stops_without_exception(self):
        self.node.mode = self.node.door_phase = 'door_align'
        self.node.door_path_feasible = True
        self.node.door_path = np.array([[0., 0., 0.], [.36, 0., 0.], [.36, 0., math.pi/2]])
        self.node.door_rotation_index = 1
        self.node.pose = (.20, 0., 1.1)
        self.node.output = np.array([.05, 0.])
        self.node.last_tick = rospy.Time.now().to_sec()-.05
        self.node.control()
        self.assertEqual(self.commands[-1].linear.x, 0.)
        self.assertEqual(self.commands[-1].angular.z, 0.)
        self.assertFalse(self.node.door_path_feasible)

    def test_recovery_does_not_override_missing_scan_or_high_measured_speed(self):
        self.node.raw = np.array([-.4, 0.])
        self.node.scans = {side: (time.monotonic(), np.array([[1., 0.]]))
                           for side in ('left', 'right')}
        self.node.scans['left'] = (0., np.array([[1., 0.]]))
        self.node.control()
        self.assertEqual(self.node.reason, 'stale_input')
        self.assertEqual(self.commands[-1].linear.x, 0.)
        self.node.scans['left'] = (time.monotonic(), np.array([[1., 0.]]))
        self.node.measured = np.array([.2, 0.])
        self.node.control()
        self.assertFalse(self.node.recovering)
        self.assertEqual(self.commands[-1].linear.x, 0.)

    def test_guard_keeps_remembered_door_points_in_addition_to_current_view(self):
        from smart_wheelchair_safety.unified_geometry import transform_points
        self.node.scans = {side: (time.monotonic(), np.array([[-.28, 0.], [2., 1.]]))
                           for side in ('left', 'right')}
        self.node.planning_scans = {side: value[1] for side, value in self.node.scans.items()}
        self.node.door_obstacles = transform_points([[1., .6]], self.node.pose)
        groups, guard = self.node.points()
        self.assertEqual(sum(len(g) for g in groups), 4)
        self.assertEqual(len(guard), 5)

    def test_scan_scope_and_self_filter_are_identical_for_planner_and_guard(self):
        from geometry_msgs.msg import TransformStamped
        from sensor_msgs.msg import LaserScan
        t = TransformStamped()
        t.transform.rotation.w = 1.
        t.transform.translation.x = .79
        self.node.tf = SimpleNamespace(lookup_transform=lambda *args: t)
        for side, lower, upper in [('left', -30, 170), ('right', -170, 30)]:
            for self_hit in (False, True):
                with self.subTest(side=side, self_hit=self_hit):
                    t.transform.translation.y = .36 if side == 'left' else -.36
                    published = []
                    self.node.scan_pubs[side] = SimpleNamespace(publish=published.append)
                    scan = LaserScan(angle_min=-math.pi, angle_max=math.pi,
                                     angle_increment=math.pi/360, range_min=.08,
                                     range_max=5., ranges=[2.]*721)
                    if self_hit:
                        scan.ranges[360] = .1
                    scan.header.stamp = rospy.Time.now()
                    self.node.on_scan(scan, side)
                    actual = np.isfinite(published[-1].ranges)
                    expected = np.zeros(721, dtype=bool)
                    expected[(lower+180)*2:(upper+180)*2+1] = True
                    if self_hit:
                        expected[360] = False
                    np.testing.assert_array_equal(actual, expected)
                    self.assertEqual(len(self.node.scans[side][1]), 401-int(self_hit))
                    np.testing.assert_array_equal(self.node.scans[side][1],
                                                  self.node.planning_scans[side])
                    self.assertEqual(scan.ranges[0], 2., 'raw scan must not be mutated')

    def test_relocated_lidar_tf_matches_follower_and_model(self):
        import xml.etree.ElementTree as ET
        from smart_wheelchair_safety.unified_control_node import UnifiedControlNode
        from smart_wheelchair_safety.local_path_follower_node import LIDAR_X_M, LIDAR_Y_M
        self.node.destroy_node()
        with patch('smart_wheelchair_safety.unified_control_node.StaticTransformBroadcaster') as broadcaster, patch.object(rospy, 'Timer'):
            self.node = UnifiedControlNode()
        transforms = broadcaster.return_value.sendTransform.call_args.args[0]
        model = ET.parse(Path(__file__).parents[2] / 'smart_wheelchair_gazebo/models/smart_wheelchair/model.sdf').getroot()
        for side, expected_y in [('left', .36), ('right', -.36)]:
            t = next(t for t in transforms if t.child_frame_id == 'unified_lidar_'+side)
            sensor_pose = list(map(float, model.findtext(f".//sensor[@name='{side}_lidar']/pose").split()))
            self.assertEqual(t.header.frame_id, 'rear_axle')
            self.assertAlmostEqual(t.transform.translation.x, sensor_pose[0]+.33)
            self.assertAlmostEqual(t.transform.translation.x, LIDAR_X_M)
            self.assertAlmostEqual(t.transform.translation.y, expected_y)
            self.assertAlmostEqual(t.transform.translation.y, sensor_pose[1])
            self.assertAlmostEqual(t.transform.translation.y, LIDAR_Y_M[side])

    def test_scan_health_uses_selected_view_not_ignored_rear(self):
        from geometry_msgs.msg import TransformStamped
        from sensor_msgs.msg import LaserScan
        t = TransformStamped()
        t.transform.rotation.w = 1.
        self.node.tf = SimpleNamespace(lookup_transform=lambda *args: t)
        for side, lower, upper in [('left', -30, 170), ('right', -170, 30)]:
            self.node.scans.pop(side, None)
            scan = LaserScan(angle_min=-math.pi, angle_increment=math.pi/360,
                             range_min=.08, range_max=5., ranges=[float('nan')]*721)
            start, end = (lower+180)*2, (upper+180)*2+1
            scan.ranges[start:end] = [float('inf')]*(end-start)
            scan.header.stamp = rospy.Time.now()
            self.node.on_scan(scan, side)
            self.assertIn(side, self.node.scans)
            accepted_time = self.node.scans[side][0]
            scan.ranges[start:start+50] = [float('nan')]*50
            scan.header.stamp = rospy.Time.now()
            self.node.on_scan(scan, side)
            self.assertEqual(self.node.scans[side][0], accepted_time,
                             'invalid active sector must not refresh sensor health')

    def wall_opening(self, side=1):
        from smart_wheelchair_safety.unified_geometry import extract_lines, find_openings
        y = side * .8
        points = np.vstack((
            np.column_stack((np.linspace(-1., .2, 20), np.full(20, y))),
            np.column_stack((np.linspace(2.2, 4., 20), np.full(20, y))),
        ))
        now = time.monotonic()
        self.node.scans = {
            'left': (now, points),
            'right': (now, np.empty((0, 2))),
        }
        return find_openings(extract_lines(points))[0]

    def set_opening_turn(self, side=1):
        from smart_wheelchair_safety.unified_geometry import Opening
        self.node.opening_turn = Opening((1.2, side * .8), side * np.pi / 2, 2., ())
        self.node.opening_turn_side = side
        self.node.opening_turn_heading = 0.
        self.node.opening_turn_time = time.monotonic()
        self.node.wall_side = 0

    def front_opening(self, center=(2., 0.), heading=0., width=1.):
        from smart_wheelchair_safety.unified_geometry import Segment, find_openings
        normal = np.array([math.cos(heading), math.sin(heading)])
        tangent = np.array([-normal[1], normal[0]])
        center = np.asarray(center, dtype=float)
        near = center - width / 2 * tangent
        far = center + width / 2 * tangent
        line_heading = math.atan2(tangent[1], tangent[0])
        line_normal = np.array([-tangent[1], tangent[0]])
        lines = [
            Segment(tuple(near - 1.5 * tangent), tuple(near), line_heading,
                    float(near @ line_normal)),
            Segment(tuple(far), tuple(far + 1.5 * tangent), line_heading,
                    float(far @ line_normal)),
        ]
        return find_openings(lines, max_width=1.5)[0]

    def observe_front_door(self, **kwargs):
        self.node._observe_openings([self.front_opening(**kwargs)])

    def confirm_door(self, **kwargs):
        self.observe_front_door(**kwargs)
        self.observe_front_door(**kwargs)

    def test_confirmed_door_refines_inward_jamb_without_chasing_wall_drift(self):
        self.confirm_door(center=(2., .06), width=1.34)
        self.confirm_door(center=(2., 0.), width=1.22)
        self.assertEqual(len(self.node.confirmed_openings), 1)
        self.assertAlmostEqual(self.node.confirmed_openings[0].width, 1.22)
        np.testing.assert_allclose(self.node.confirmed_openings[0].center, (2., 0.))
        # Occlusion cannot widen a previously supported opening again.
        self.confirm_door(center=(2., .06), width=1.34)
        self.assertAlmostEqual(self.node.confirmed_openings[0].width, 1.22)

    def test_captured_nw_pose_selects_feasible_door_instead_of_wall(self):
        from smart_wheelchair_safety.unified_geometry import extract_opening_lines, find_openings, transform_points
        capture = json.loads((Path(__file__).with_name('fixtures') /
                              'nw_door_wait_capture.json').read_text())
        self.node.pose = capture['pose']
        self.node.raw = np.array([.666, 0.])
        self.node.scans = {side: (time.monotonic(), transform_points(capture['points'], self.node.pose))
                           for side in ('left', 'right')}
        openings = find_openings([line for group in capture['groups']
                                  for line in extract_opening_lines(group)])
        self.node._observe_openings(openings)
        self.node._observe_openings(openings)
        self.node.raw_time = self.node.odom_time = time.monotonic()
        self.node.odom_stamp = rospy.Time.now().to_sec()
        self.node.update_reference()
        self.finish_door_plan()
        self.assertEqual(self.node.mode, 'door_align')
        self.assertTrue(self.node.door_path_feasible)
        self.assertEqual(len(self.node.confirmed_openings), 1)

    def test_inward_jamb_refinement_invalidates_active_and_inflight_alignment(self):
        from concurrent.futures import Future
        self.confirm_door(center=(2., .06), width=1.34)
        self.node.door = self.node.confirmed_openings[0]
        self.node.door_phase = 'door_align'
        self.node.door_path = np.array([[0., 0., 0.], [2., .06, 0.]])
        self.node.door_path_feasible = True
        self.node.accept_planned = True
        generation = self.node.door_generation
        old = Future()
        self.node.door_plan_request = (old, self.node.pose, generation)
        self.confirm_door(center=(2., 0.), width=1.22)
        self.assertIsNone(self.node.door_path)
        self.assertFalse(self.node.door_path_feasible)
        self.assertFalse(self.node.accept_planned)
        self.assertGreater(self.node.door_generation, generation)
        old.set_result(np.array([[0., 0., 0.], [2., .06, 0.]]))
        self.node._local_door_path(self.node.door, np.empty((0, 2)))
        self.assertIsNone(self.node.door_path)
        self.assertFalse(self.node.door_path_feasible)
        self.assertNotEqual(self.node.door_plan_request[2], generation)

    def test_door_visual_publishes_confirmed_world_jambs_while_neutral(self):
        messages = []
        self.node.door_detection_pub = SimpleNamespace(publish=messages.append)
        self.node.pose = (3., 4., math.pi/2)
        self.node.raw[:] = 0.
        self.observe_front_door(center=(2., 0.), width=1.1)
        self.node.update_reference()
        self.assertEqual(len(messages[-1].polygon.points), 0)
        self.observe_front_door(center=(2., 0.), width=1.1)
        self.node.update_reference()
        message = messages[-1]
        self.assertEqual(message.header.frame_id, 'odom')
        np.testing.assert_allclose([(p.x, p.y) for p in message.polygon.points],
                                   [(3.55, 6.), (2.45, 6.)], atol=1e-7)
        self.node._observe_openings([])
        self.node.update_reference()
        self.assertEqual(len(messages[-1].polygon.points), 0)

    def test_neutral_joystick_heartbeats_do_not_erase_visual_confirmation(self):
        messages = []
        self.node.door_detection_pub = SimpleNamespace(publish=messages.append)
        self.send_raw(0., 0.)
        self.observe_front_door(width=1.1)
        self.send_raw(0., 0.)
        self.observe_front_door(width=1.1)
        self.node.update_reference()
        self.assertEqual(len(messages[-1].polygon.points), 2)
        self.send_raw(0., 0.)
        self.node.update_reference()
        self.assertEqual(len(messages[-1].polygon.points), 2)
        self.assertIsNone(self.node.door)
        self.assertEqual(self.node.confirmed_openings, [])

    def test_door_visual_clears_on_stale_scan_and_observation(self):
        messages = []
        self.node.door_detection_pub = SimpleNamespace(publish=messages.append)
        self.confirm_door(width=1.1)
        self.node.update_reference()
        self.assertEqual(len(messages[-1].polygon.points), 2)
        self.node.scans['left'] = (0., self.node.scans['left'][1])
        self.node.update_reference()
        self.assertEqual(len(messages[-1].polygon.points), 0)
        self.node.scans['left'] = (time.monotonic(), self.node.scans['left'][1])
        self.node.door_detection_time = 0.
        self.node.update_reference()
        self.assertEqual(len(messages[-1].polygon.points), 0)

    def finish_door_plan(self):
        if self.node.door_plan_request is not None:
            self.node.door_plan_request[0].result(timeout=2.)
        self.node.raw_time = self.node.odom_time = time.monotonic()
        self.node.odom_stamp = rospy.Time.now().to_sec()
        self.node.update_reference()

    def send_raw(self, linear, angular):
        from geometry_msgs.msg import Twist
        msg = Twist()
        msg.linear.x, msg.angular.z = linear, angular
        self.node.on_raw(msg)

    def test_waiting_for_door_search_stops_before_losing_staging_space(self):
        self.node.mode = 'door_wait'
        self.node.planned = np.array([.6, 0.])
        self.node.control()
        self.assertEqual(self.commands[-1].linear.x, 0.)

    def test_aligned_next_door_gets_checked_path_while_moving(self):
        self.node.door_phase = 'door_align'
        self.node.measured = np.array([.3, 0.])
        self.node.output = np.array([.3, 0.])
        door = self.front_opening(center=(2., 0.), heading=0.)
        with patch.object(self.node.door_plan_executor, 'submit') as submit:
            self.node._local_door_path(door, np.empty((0, 2)))
        submit.assert_not_called()
        self.assertTrue(self.node.door_path_feasible)
        self.assertIsNotNone(self.node.door_path)

    def test_front_wall_can_take_over_after_door_clearance(self):
        wall = np.column_stack((np.full(81, 2.), np.linspace(-3., 3., 81)))
        self.node.scans = {side: (time.monotonic(), wall) for side in ('left', 'right')}
        self.node.mode = 'door_clear'
        self.node.update_reference()
        self.assertEqual(self.node.mode, 'wall')
        self.assertFalse(self.node.front_blocked)

    def test_checked_door_pivot_rotates_without_forward_creep(self):
        self.node.mode = 'door_align'
        self.node.door_path_feasible = True
        self.node.door_path = np.array([[0., 0., 0.], [0., 0., .5],
                                       [1., .5, .5], [1., .5, 0.], [3., .5, 0.]])
        self.node.door_rotation_index = 0
        self.node.scans = {side: (time.monotonic(), np.empty((0, 2))) for side in ('left', 'right')}
        self.node.control()
        self.assertEqual(self.commands[-1].linear.x, 0.)
        self.assertGreater(self.commands[-1].angular.z, 0.)

    def test_door_search_waits_for_measured_stop_before_fixing_pivot_origin(self):
        door = self.front_opening(center=(2., .3), heading=.4)
        self.node.door_phase = 'door_align'
        self.node.measured = np.array([.3, 0.])
        with patch.object(self.node.door_plan_executor, 'submit') as submit:
            self.node._local_door_path(door, np.empty((0, 2)))
        submit.assert_not_called()

    def test_initial_pivot_drift_invalidates_path_instead_of_driving_away(self):
        self.node.door_phase = 'door_align'
        self.node.door_path_feasible = True
        self.node.door_path = np.array([[0., 0., 0.], [0., 0., .5], [1., .5, .5]])
        self.node.pose = (.12, 0., 0.)
        self.node.measured = np.array([.08, 0.])
        self.node._local_door_path(self.front_opening(), np.empty((0, 2)))
        self.assertIsNone(self.node.door_path)
        self.assertFalse(self.node.door_path_feasible)

    def test_short_sensor_gap_stops_without_forgetting_active_door(self):
        self.node.door = self.front_opening(center=(2., .2), heading=.3)
        self.node.door_phase = 'door_align'
        self.node.door_path = np.array([[0., 0., 0.], [1., .2, .3], [3., .2, .3]])
        self.node.door_path_feasible = True
        original = self.node.door_path
        self.node.odom_time = time.monotonic()-.15
        self.node.update_reference()
        self.node.control()
        self.assertEqual(self.node.reason, 'stale_input')
        np.testing.assert_array_equal(self.node.output, [0., 0.])
        self.assertIsNotNone(self.node.door)
        self.assertIs(self.node.door_path, original)
        self.node.odom_time = self.node.raw_time = time.monotonic()
        self.node.odom_stamp = rospy.Time.now().to_sec()
        self.node.update_reference()
        self.assertEqual(self.node.mode, 'door_align')
        self.assertIs(self.node.door_path, original)

    def test_sustained_sensor_gap_discards_active_door(self):
        self.node.door = self.front_opening()
        self.node.odom_time = time.monotonic()-.15
        self.node.sensor_stale_since = time.monotonic()-1.1
        self.node.update_reference()
        self.assertIsNone(self.node.door)

    def test_same_side_opening_needs_two_frames_and_blocks_old_wall_reacquisition(self):
        opening = self.wall_opening()
        self.node.raw = np.array([.6, .5])
        self.node.wall_side = 1
        self.node.mode = 'wall'

        self.node._observe_openings([opening])
        self.assertEqual(self.node.confirmed_openings, [])
        self.node._observe_openings([opening])
        self.node.update_reference()

        self.assertEqual(self.node.mode, 'opening_turn')
        self.assertIsNotNone(self.node.opening_turn)
        self.node.update_reference()
        self.assertEqual(self.node.mode, 'opening_turn')

    def test_straight_reference_keeps_remembered_wall_side(self):
        now = time.monotonic()
        x = np.linspace(0., 4., 20)
        self.node.wall_preference = 1
        self.node.raw = np.array([.6, 0.])
        self.node.scans = {
            'left': (now, np.column_stack((x, np.full_like(x, .9)))),
            'right': (now, np.column_stack((x, np.full_like(x, -.55)))),
        }

        self.node.update_reference()

        self.assertEqual(self.node.mode, 'wall')
        self.assertEqual(self.node.wall_side, 1)
        self.assertEqual(self.node.wall_preference, 1)

    def test_missing_remembered_side_uses_manual_without_forgetting_entry_side(self):
        now = time.monotonic()
        x = np.linspace(0., 4., 20)
        self.node.wall_preference = 1
        self.node.raw = np.array([.6, 0.])
        self.node.scans = {
            'left': (now, np.empty((0, 2))),
            'right': (now, np.column_stack((x, np.full_like(x, -.55)))),
        }

        self.node.update_reference()

        self.assertEqual((self.node.mode, self.node.wall_side), ('manual', 0))
        self.assertEqual(self.node.wall_preference, 1)

    def test_requested_side_opening_releases_old_wall_override(self):
        opening = self.wall_opening(side=1)
        self.node._observe_openings([opening])
        self.node._observe_openings([opening])
        self.node.wall_side = -1
        self.node.wall_preference = -1
        self.node.mode = 'wall'

        self.send_raw(.6, .5)
        self.node.update_reference()

        self.assertFalse(self.node.override)
        self.assertEqual(self.node.mode, 'opening_turn')
        self.assertEqual(self.node.opening_turn_side, 1)

    def test_stop_clears_wall_preference(self):
        self.node.wall_preference = 1

        self.send_raw(0., 0.)

        self.assertEqual(self.node.wall_preference, 0)

    def test_mode_change_stops_and_clears_assistance_state(self):
        from std_msgs.msg import Bool
        self.set_opening_turn()
        self.node.wall_side = self.node.wall_preference = 1
        self.node.opening_candidates = [('candidate', 1, time.monotonic())]
        self.node.confirmed_openings = ['opening']
        self.node.door_obstacles = np.array([[.2, .5]])
        self.node.output[:] = [.4, .2]
        self.node.acceleration[:] = [.3, .4]

        self.node.on_assist_enabled(Bool(data=False))

        self.assertFalse(self.node.assist_enabled)
        self.assertFalse(self.node.mode_neutral_seen)
        self.assertIsNone(self.node.opening_turn)
        self.assertEqual((self.node.wall_side, self.node.wall_preference), (0, 0))
        self.assertEqual(self.node.opening_candidates, [])
        self.assertEqual(self.node.confirmed_openings, [])
        self.assertEqual(len(self.node.door_obstacles), 0)
        np.testing.assert_array_equal(self.node.output, [0., 0.])
        np.testing.assert_array_equal(self.node.acceleration, [0., 0.])
        self.assertEqual((self.commands[-1].linear.x,
                          self.commands[-1].angular.z), (0., 0.))

    def test_repeated_mode_heartbeat_does_not_disarm_active_mode(self):
        from std_msgs.msg import Bool
        self.node.mode_neutral_seen = True

        self.node.on_assist_enabled(Bool(data=True))

        self.assertTrue(self.node.mode_neutral_seen)

    def test_manual_mode_forwards_fresh_raw_without_sensors_or_smoothing(self):
        from std_msgs.msg import Bool
        self.node.on_assist_enabled(Bool(data=False))
        self.send_raw(0., 0.)
        self.send_raw(1.2, -.7)
        self.node.scans = {}
        self.node.odom_time = 0.

        self.node.control()

        self.assertEqual(self.node.mode, 'manual_direct')
        self.assertAlmostEqual(self.commands[-1].linear.x, 1.2)
        self.assertAlmostEqual(self.commands[-1].angular.z, -.7)

    def test_manual_mode_rejects_non_finite_raw_command(self):
        from std_msgs.msg import Bool
        self.node.on_assist_enabled(Bool(data=False))
        self.send_raw(0., 0.)

        self.send_raw(float('nan'), .4)
        self.node.control()

        self.assertEqual((self.commands[-1].linear.x,
                          self.commands[-1].angular.z), (0., 0.))
        self.assertEqual(self.node.reason, 'stale_input')

    def test_mode_transition_and_manual_timeout_publish_zero(self):
        from std_msgs.msg import Bool
        self.node.on_assist_enabled(Bool(data=False))
        self.send_raw(.6, .3)

        self.node.control()

        self.assertEqual(self.commands[-1].linear.x, 0.)
        self.send_raw(0., 0.)
        self.node.raw_time = 0.
        self.node.control()
        self.assertEqual(self.commands[-1].linear.x, 0.)

    def test_assist_restoration_waits_for_neutral_then_uses_existing_chain(self):
        from std_msgs.msg import Bool
        self.node.on_assist_enabled(Bool(data=False))
        self.send_raw(0., 0.)
        self.node.on_assist_enabled(Bool(data=True))
        self.send_raw(.5, 0.)

        self.node.control()

        self.assertEqual(self.commands[-1].linear.x, 0.)
        self.send_raw(0., 0.)
        self.send_raw(.5, 0.)
        now = time.monotonic()
        self.node.raw_time = self.node.odom_time = self.node.plan_time = now
        self.node.odom_stamp = rospy.Time.now().to_sec()
        self.node.scans = {
            side: (now, np.array([[3., 2.], [3., -2.]]))
            for side in ('left', 'right')
        }
        self.node.control()

        self.assertNotEqual(self.node.mode, 'manual_direct')

    def test_assist_restoration_ignores_old_plan_until_new_reference_is_published(self):
        from geometry_msgs.msg import TwistStamped
        from std_msgs.msg import Bool
        self.node.on_assist_enabled(Bool(data=False))
        self.send_raw(0., 0.)
        self.node.on_assist_enabled(Bool(data=True))
        self.send_raw(0., 0.)

        old = TwistStamped()
        old.twist.linear.x = .7
        self.node.on_planned(old)

        np.testing.assert_array_equal(self.node.planned, [0., 0.])
        self.assertEqual(self.node.plan_time, 0.)

        self.send_raw(.5, 0.)
        now = time.monotonic()
        self.node.raw_time = self.node.odom_time = self.node.plan_time = now
        self.node.odom_stamp = rospy.Time.now().to_sec()
        self.node.update_reference()

        self.node.on_planned(old)

        np.testing.assert_array_equal(self.node.planned, [0., 0.])
        fresh = TwistStamped()
        fresh.header.stamp = self.node.active_reference_stamp
        fresh.twist.linear.x = .7
        self.node.on_planned(fresh)

        self.assertAlmostEqual(self.node.planned[0], .7)

    def test_startup_and_cancel_ignore_plans_until_new_reference_is_published(self):
        from geometry_msgs.msg import TwistStamped
        self.node.planned[:] = 0.
        self.node.plan_time = 0.
        command = TwistStamped()
        command.header.stamp = rospy.Time.now()
        command.twist.linear.x = .7

        self.node.on_planned(command)

        np.testing.assert_array_equal(self.node.planned, [0., 0.])
        self.node.accept_planned = True
        self.node.active_reference_stamp = command.header.stamp
        self.node.on_planned(command)
        self.assertAlmostEqual(self.node.planned[0], .7)

        self.node.cancel()
        command.header.stamp = rospy.Time.now()
        self.node.on_planned(command)

        self.assertFalse(self.node.accept_planned)
        np.testing.assert_array_equal(self.node.planned, [0., 0.])
        self.assertEqual(self.node.plan_time, 0.)

    def test_replaced_reference_rejects_late_result_and_stamps_increase_when_time_stalls(self):
        from geometry_msgs.msg import TwistStamped
        references = []
        self.node.reference = SimpleNamespace(publish=references.append)
        instant = rospy.Time.now()
        with patch.object(rospy.Time, 'now', return_value=instant):
            self.node.update_reference()
            old = references[-1].header.stamp
            self.node.update_reference()
            current = references[-1].header.stamp

        self.assertGreater(current, old)
        late = TwistStamped()
        late.header.stamp = old
        late.twist.linear.x = .7
        self.node.planned[:] = 0.
        self.node.plan_time = 0.
        self.node.on_planned(late)
        np.testing.assert_array_equal(self.node.planned, [0., 0.])
        self.assertEqual(self.node.plan_time, 0.)

        # A computation stamped only at publication time has no valid
        # reference identity, even though it is newer than the replacement.
        late.header.stamp = current + rospy.Duration(nsecs=1)
        self.node.on_planned(late)
        np.testing.assert_array_equal(self.node.planned, [0., 0.])
        self.assertEqual(self.node.plan_time, 0.)

        late.header.stamp = current
        self.node.on_planned(late)
        self.assertAlmostEqual(self.node.planned[0], .7)
        self.node.cancel()
        self.node.on_planned(late)
        np.testing.assert_array_equal(self.node.planned, [0., 0.])

    def test_confirmed_opening_survives_temporary_lidar_occlusion_until_passed(self):
        opening = self.wall_opening(side=-1)
        self.node._observe_openings([opening])
        self.node._observe_openings([opening])

        self.node._observe_openings([])
        self.assertEqual(len(self.node.confirmed_openings), 1)

        self.node.pose = (3., 0., 0.)
        self.node._observe_openings([])
        self.assertEqual(self.node.confirmed_openings, [])

    def test_corner_bounded_opening_still_requires_two_observations(self):
        from smart_wheelchair_safety.unified_geometry import Segment, find_openings
        wall = Segment((.3, -.52), (2.44, -.52), 0., -.52)
        boundary = Segment((5.14, -.55), (5.14, -.95), math.pi/2, -5.14)
        opening = find_openings([wall, boundary])[0]

        self.node._observe_openings([opening])
        self.assertEqual(self.node.confirmed_openings, [])

        self.node._observe_openings([opening])
        self.assertEqual(len(self.node.confirmed_openings), 1)

    def test_confirmed_opening_world_position_does_not_walk_with_chained_matches(self):
        self.observe_front_door(center=(2., 0.), width=1.)
        self.observe_front_door(center=(2.1, 0.), width=1.)
        confirmed = self.node.confirmed_openings[0]

        self.observe_front_door(center=(2.25, 0.), width=1.)
        self.observe_front_door(center=(2.4, 0.), width=1.)

        self.assertEqual(self.node.confirmed_openings[0], confirmed)

    def test_opening_confirmation_tolerates_one_missing_lidar_fit(self):
        opening = self.wall_opening(side=-1)

        self.node._observe_openings([opening])
        self.node._observe_openings([])
        self.node._observe_openings([opening])

        self.assertEqual(len(self.node.confirmed_openings), 1)

    def test_stop_reverse_and_away_steering_cancel_opening_turn(self):
        from geometry_msgs.msg import Twist
        for linear, angular in ((0., 0.), (-.2, 0.), (.4, -.3)):
            with self.subTest(command=(linear, angular)):
                self.set_opening_turn()
                msg = Twist()
                msg.linear.x, msg.angular.z = linear, angular
                self.node.on_raw(msg)
                self.assertIsNone(self.node.opening_turn)

    def test_opening_turn_exits_on_heading_pass_timeout_and_stale_input(self):
        cases = ('heading', 'passed', 'timeout', 'stale')
        for case in cases:
            with self.subTest(case=case):
                self.set_opening_turn()
                if case == 'heading':
                    self.node.pose = (0., 0., 1.1)
                elif case == 'passed':
                    self.node.pose = (2.0, 0., 0.)
                elif case == 'timeout':
                    self.node.opening_turn_time -= 12.1
                else:
                    self.node.raw_time = 0.
                self.node.update_reference()
                self.assertIsNone(self.node.opening_turn)

    def test_opening_turn_preserves_clear_driver_steering_when_planner_understeers(self):
        from smart_wheelchair_safety.unified_geometry import Opening
        self.set_opening_turn(side=1)
        self.node.opening_turn = Opening((.4, .8), math.pi / 2, 2., ())
        self.node.mode = 'opening_turn'
        self.node.raw = np.array([.6, .5])
        self.node.planned = np.array([.5, .03])

        for _ in range(12):
            self.node.last_tick -= .1
            self.node.control()

        self.assertGreater(self.commands[-1].angular.z, .25)

    def test_opening_turn_suppresses_planner_yaw_until_front_clears_jamb(self):
        from smart_wheelchair_safety.unified_geometry import Opening
        self.node.opening_turn = Opening((3.5, -.52), -math.pi / 2, 2.7, ())
        self.node.opening_turn_side = -1
        self.node.opening_turn_heading = 0.
        self.node.opening_turn_time = time.monotonic()
        self.node.mode = 'opening_turn'
        self.node.raw = np.array([.6, -.5])
        self.node.planned = np.array([.5, -.08])

        for _ in range(8):
            self.node.last_tick -= .1
            self.node.control()

        self.assertAlmostEqual(self.commands[-1].angular.z, 0.)

    def test_opening_wait_segment_corrects_heading_drift_away_from_wall_end(self):
        from smart_wheelchair_safety.unified_geometry import Opening
        self.node.opening_turn = Opening((3.5, -.52), -math.pi / 2, 2.7, ())
        self.node.opening_turn_side = -1
        self.node.opening_turn_heading = 0.
        self.node.opening_turn_time = time.monotonic()
        self.node.mode = 'opening_turn'
        self.node.pose = (0., 0., -.08)
        self.node.raw = np.array([.6, -.5])
        self.node.planned = np.array([.5, -.08])

        for _ in range(8):
            self.node.last_tick -= .1
            self.node.control()

        self.assertGreater(self.commands[-1].angular.z, 0.)

    def test_opening_wait_segment_corrects_lateral_drift_toward_wall(self):
        from smart_wheelchair_safety.unified_geometry import Opening
        self.node.opening_turn = Opening((3.5, -.52), -math.pi / 2, 2.7, ())
        self.node.opening_turn_side = -1
        self.node.opening_turn_heading = 0.
        self.node.opening_turn_origin = np.array([0., 0.])
        self.node.opening_turn_time = time.monotonic()
        self.node.mode = 'opening_turn'
        self.node.pose = (1., -.05, 0.)
        self.node.raw = np.array([.6, -.5])
        self.node.planned = np.array([.5, 0.])

        for _ in range(8):
            self.node.last_tick -= .1
            self.node.control()

        self.assertGreater(self.commands[-1].angular.z, .03)

    def test_straight_wall_preview_enforces_correction_away_from_wall(self):
        for side in (-1, 1):
            with self.subTest(side=side):
                self.commands.clear()
                self.node.output[:] = 0.
                self.node.acceleration[:] = 0.
                self.node.mode = 'wall'
                self.node.wall_side = side
                self.node.wall_preview_heading = -side*.08
                self.node.pose = (0., 0., 0.)
                self.node.raw = np.array([.6, 0.])
                self.node.planned = np.array([.5, side*.04])

                for _ in range(8):
                    self.node.last_tick -= .1
                    self.node.control()

                self.assertGreater(-side*self.commands[-1].angular.z, .03)

    def test_opening_turn_reference_enters_wide_corridor_instead_of_turning_early(self):
        from smart_wheelchair_safety.unified_geometry import Opening
        self.node.opening_turn = Opening((.8, -.52), -math.pi / 2, 2.7, ())
        self.node.opening_turn_side = -1
        self.node.opening_turn_heading = 0.
        self.node.opening_turn_time = time.monotonic()

        path = self.node._opening_turn_path()

        self.assertGreater(path[-1, 0], .7)
        self.assertLess(path[-1, 1], -1.)

    def test_opening_wait_heading_comes_from_wall_tangent_not_current_drift(self):
        from smart_wheelchair_safety.unified_geometry import Opening
        self.node.pose = (1., 2., -.08)
        local_opening = Opening((3.5, -.52), -math.pi/2+.08, 2.7, ())

        self.node._start_opening_turn(local_opening, -1)

        self.assertAlmostEqual(self.node.opening_turn_heading, 0., places=6)

    def test_opening_turn_waits_until_rear_axle_clears_near_jamb(self):
        from smart_wheelchair_safety.unified_geometry import Opening
        self.node.opening_turn = Opening((1.7, -.52), -math.pi / 2, 2.7, ())
        self.node.opening_turn_side = -1
        self.node.opening_turn_heading = 0.
        self.node.opening_turn_time = time.monotonic()

        self.assertFalse(self.node._opening_turn_ready())

        self.node.opening_turn = Opening((.8, -.52), -math.pi / 2, 2.7, ())
        self.assertTrue(self.node._opening_turn_ready())

    def test_opening_turn_reference_stays_straight_until_front_clears_near_jamb(self):
        from smart_wheelchair_safety.unified_geometry import Opening
        self.node.opening_turn = Opening((3.5, -.52), -math.pi / 2, 2.7, ())
        self.node.opening_turn_side = -1
        self.node.opening_turn_heading = 0.
        self.node.opening_turn_time = time.monotonic()

        path = self.node._opening_turn_path()

        np.testing.assert_allclose(path[:, 1], 0.)
        np.testing.assert_allclose(path[:, 2], 0.)

    def test_door_needs_two_observations_before_alignment(self):
        self.observe_front_door(center=(2., .2), heading=.12, width=1.)
        self.node.update_reference()
        self.assertNotEqual(self.node.mode, 'door_align')

        self.observe_front_door(center=(2.02, .18), heading=.10, width=1.02)
        self.node.update_reference()
        self.assertEqual(self.node.mode, 'door_wait')
        self.finish_door_plan()

        self.assertEqual(self.node.mode, 'door_align')

    def test_pending_door_intent_does_not_briefly_capture_wall_following(self):
        self.observe_front_door(center=(3.3168, 1.149), heading=.4428, width=1.3008)
        self.node.raw = np.array([.8977, .5182])
        self.node.wall_side = 1

        self.node.update_reference()

        self.assertEqual(self.node.mode, 'manual')
        self.assertEqual(self.node.wall_side, 0)

    def test_door_centerline_freezes_after_commit(self):
        self.confirm_door(center=(1., 0.), heading=0., width=1.)
        self.node.pose = (.10, 0., 0.)
        self.node.update_reference()
        self.finish_door_plan()
        self.assertEqual(self.node.mode, 'door_pass')
        frozen = self.node.door

        self.observe_front_door(center=(1.15, .12), heading=.1, width=1.)
        self.node.odom_stamp = rospy.Time.now().to_sec()
        self.node.update_reference()

        self.assertEqual(self.node.door, frozen)
        self.assertEqual(self.node.mode, 'door_pass')

    def test_oblique_door_reference_stays_fixed_while_chair_advances(self):
        references = []
        self.node.reference = SimpleNamespace(publish=references.append)
        self.confirm_door(center=(2.32, .34), heading=math.radians(45.), width=1.22)

        self.node.update_reference()
        self.node.door_plan_request[0].result(timeout=1.)
        self.node.raw_time = self.node.odom_time = time.monotonic()
        self.node.odom_stamp = rospy.Time.now().to_sec()
        self.node.update_reference()
        first = np.array([(pose.pose.position.x, pose.pose.position.y)
                          for pose in references[-1].poses])
        self.node.pose = (.2, -.05, -.1)
        self.node.update_reference()
        second = np.array([(pose.pose.position.x, pose.pose.position.y)
                           for pose in references[-1].poses])

        np.testing.assert_allclose(second, first, atol=1e-9)

    def test_locked_door_reference_stays_fixed_after_takeover(self):
        references = []
        self.node.reference = SimpleNamespace(publish=references.append)
        self.confirm_door(center=(2.32, .34), heading=math.radians(45.), width=1.22)

        self.node.update_reference()
        self.finish_door_plan()
        frozen = references[-1]
        epoch = self.node.epoch
        self.node.update_reference()

        self.assertEqual(len(references), 3)
        self.assertEqual(self.node.epoch, epoch)
        self.assertEqual([pose.pose for pose in references[-1].poses],
                         [pose.pose for pose in frozen.poses])

    def test_door_acquisition_replaces_navigation_only_after_path_is_feasible(self):
        references = []
        self.node.reference = SimpleNamespace(publish=references.append)
        epoch = self.node.epoch
        self.confirm_door(center=(2.32, .34), heading=math.radians(45.), width=1.22)

        self.node.update_reference()
        self.assertEqual(self.node.epoch, epoch)
        self.assertEqual(self.node.mode, 'door_wait')
        self.node.door_plan_request[0].result(timeout=1.)
        self.node.raw_time = self.node.odom_time = time.monotonic()
        self.node.odom_stamp = rospy.Time.now().to_sec()
        self.node.update_reference()

        self.assertEqual(self.node.epoch, epoch + 1)
        self.assertEqual(len(references), 2)
        self.assertTrue(self.node.mode.startswith('door_'))
        self.assertTrue(self.node.accept_planned)
        np.testing.assert_array_equal(self.node.planned, [0., 0.])

    def test_door_takeover_waits_for_a_feasible_reference(self):
        from smart_wheelchair_safety.unified_geometry import arc_path
        epoch = self.node.epoch
        self.confirm_door(center=(4.32, -.68), heading=0., width=1.11)

        with patch.object(self.node, '_local_door_path', return_value=arc_path(.5, 0.)):
            self.node.door_path_feasible = False
            self.node.update_reference()

        self.assertEqual(self.node.mode, 'door_wait')
        self.assertEqual(self.node.epoch, epoch)

    def test_aligned_door_commits_within_staging_heading_control_band(self):
        self.confirm_door(center=(1.15, 0.), heading=0., width=1.)

        self.node.update_reference()
        self.finish_door_plan()

        self.assertEqual(self.node.mode, 'door_pass')

    def test_door_releases_only_after_rear_body_clears_plane(self):
        self.confirm_door(center=(1., 0.), heading=0., width=1.)
        self.node.door = self.front_opening(center=(1., 0.), heading=0., width=1.)
        self.node.door_phase = 'door_pass'
        self.node.pose = (1.20, 0., 0.)
        self.node.update_reference()
        self.assertEqual(self.node.mode, 'door_clear')

        self.node.pose = (1.30, 0., 0.)
        self.node.update_reference()

        self.assertIsNone(self.node.door)

    def test_successful_door_clear_discards_remembered_jambs(self):
        from smart_wheelchair_safety.unified_geometry import Opening
        self.node.door = Opening((-.30, 0.), 0., 1., ())
        self.node.door_phase = 'door_pass'
        self.node.door_obstacles = np.array([[.2, .5]])

        self.assertIsNone(self.node._local_tracked_door())

        self.assertEqual(len(self.node.door_obstacles), 0)

    def test_door_clear_progressively_returns_steering_to_joystick(self):
        from smart_wheelchair_safety.unified_geometry import Opening
        self.node.mode = 'door_clear'
        self.node.door_phase = 'door_clear'
        self.node.raw = np.array([.5, .5])
        with patch.object(self.node, '_door_preview_angular', return_value=.1):
            expected = ((0., .1), (.145, .3), (.29, .5), (.40, .5))
            for progress, angular in expected:
                with self.subTest(progress=progress):
                    self.node.door = Opening((-progress, 0.), 0., 1., ())
                    self.assertAlmostEqual(self.node._door_angular(.4), angular)

    def test_door_pass_keeps_locked_path_steering(self):
        from smart_wheelchair_safety.unified_geometry import Opening
        self.node.mode = 'door_pass'
        self.node.door_phase = 'door_pass'
        self.node.door = Opening((.20, 0.), 0., 1., ())
        self.node.raw = np.array([.5, .5])
        with patch.object(self.node, '_door_preview_angular', return_value=.1):
            self.assertAlmostEqual(self.node._door_angular(.4), .1)

    def test_stale_door_pass_after_plane_uses_clearance_handoff(self):
        from smart_wheelchair_safety.unified_geometry import Opening
        self.node.mode = 'door_pass'
        self.node.door_phase = 'door_pass'
        self.node.door = Opening((-.145, 0.), 0., 1., ())
        self.node.raw = np.array([.5, .5])
        with patch.object(self.node, '_door_preview_angular', return_value=.1):
            self.assertAlmostEqual(self.node._door_angular(.4), .3)

    def test_door_clear_handoff_applies_to_planner_and_fallback(self):
        from smart_wheelchair_safety.unified_geometry import Opening
        for planner_fresh in (True, False):
            with self.subTest(planner_fresh=planner_fresh):
                self.node.mode = 'door_clear'
                self.node.door_phase = 'door_clear'
                self.node.door = Opening((-.29, 0.), 0., 1., ())
                self.node.door_path = np.array([[0., 0., 0.], [1., 0., 0.]])
                self.node.door_path_feasible = True
                self.node.raw = np.array([.5, .5])
                self.node.planned = np.array([.3, 0.])
                self.node.plan_time = time.monotonic() if planner_fresh else 0.
                self.node.raw_time = self.node.odom_time = time.monotonic()
                self.node.scans = {
                    side: (self.node.raw_time, points)
                    for side, (_, points) in self.node.scans.items()
                }
                self.node.odom_stamp = rospy.Time.now().to_sec()
                self.node.output[:] = 0.
                self.node.acceleration[:] = 0.
                with patch.object(self.node, '_door_angular', return_value=.4) as angular:
                    for _ in range(12):
                        now = time.monotonic()
                        self.node.raw_time = self.node.odom_time = now
                        self.node.scans = {
                            side: (now, points)
                            for side, (_, points) in self.node.scans.items()
                        }
                        self.node.odom_stamp = (
                            rospy.Time.now().to_sec())
                        self.node.last_tick -= .1
                        self.node.control()
                self.assertTrue(angular.called)
                self.assertGreater(self.commands[-1].angular.z, 0.)

    def test_stop_and_reverse_cancel_door_immediately(self):
        for phase in ('door_align', 'door_pass', 'door_clear'):
            for command in ((0., 0.), (-.2, 0.)):
                with self.subTest(phase=phase, command=command):
                    self.node.door = self.front_opening(center=(1., 0.), heading=0., width=1.)
                    self.node.door_phase = phase
                    self.send_raw(*command)
                    self.assertIsNone(self.node.door)

    def test_active_door_retains_near_plane_aperture_crossing_for_sustained_steering(self):
        for phase in ('door_align', 'door_pass'):
            for side in (-1., 1.):
                with self.subTest(phase=phase, side=side):
                    door = self.front_opening(center=(.75, side*.04), heading=side*.04)
                    self.node.door = door
                    self.node.door_phase = phase
                    with patch('smart_wheelchair_safety.unified_control_node.time.monotonic') as now:
                        for stamp in (100., 100.2, 100.4, 100.8):
                            now.return_value = stamp
                            self.send_raw(.5, side*.3)
                    self.assertEqual(self.node.door, door)
                    self.assertEqual(self.node.door_phase, phase)
                    self.assertEqual(self.node.door_away_since, 0.)
                    self.assertFalse(self.node.override)

    def test_active_door_retains_low_speed_steering_toward_reference(self):
        for side in (-1., 1.):
            with self.subTest(side=side):
                door = self.front_opening(center=(1.6, side*.45), heading=side*.18)
                self.node.door = door
                self.node.door_phase = 'door_align'
                with patch('smart_wheelchair_safety.unified_control_node.time.monotonic') as now:
                    for stamp in (100., 100.2, 100.4, 100.8):
                        now.return_value = stamp
                        self.send_raw(.1, side*.6)
                self.assertEqual(self.node.door, door)
                self.assertEqual(self.node.door_phase, 'door_align')
                self.assertEqual(self.node.door_away_since, 0.)
                self.assertFalse(self.node.override)

    def test_millimetre_offset_does_not_veto_confirmed_aperture_departure(self):
        from smart_wheelchair_safety.unified_geometry import active_aperture_targeted
        for phase in ('door_pass', 'door_clear'):
            for side in (-1., 1.):
                with self.subTest(phase=phase, side=side):
                    door = self.front_opening(center=(.75, side*.001), heading=0.)
                    self.assertFalse(active_aperture_targeted(door, .5, side*.5))
                    self.node.door = door
                    self.node.door_phase = phase
                    self.node.door_away_since = 0.
                    with patch('smart_wheelchair_safety.unified_control_node.time.monotonic') as now:
                        for stamp in (100., 100.2, 100.349):
                            now.return_value = stamp
                            self.send_raw(.5, side*.5)
                            self.assertEqual(self.node.door, door)
                            self.assertFalse(self.node.override)
                        now.return_value = 100.351
                        self.send_raw(.5, side*.5)
                    self.assertIsNone(self.node.door)
                    self.assertTrue(self.node.override)

    def test_low_throttle_toward_heading_retains_despite_opposite_staging_bend(self):
        from smart_wheelchair_safety.unified_geometry import door_alignment_reference
        for side in (-1., 1.):
            with self.subTest(side=side):
                door = self.front_opening(center=(1.6, side*.04), heading=side*.3)
                reference = door_alignment_reference(door, 'align')
                self.assertLess(side*reference[1, 2], 0.)
                self.node.door = door
                self.node.door_phase = 'door_align'
                self.node.door_away_since = 0.
                with patch('smart_wheelchair_safety.unified_control_node.time.monotonic') as now:
                    for stamp in (100., 100.2, 100.4, 100.8):
                        now.return_value = stamp
                        self.send_raw(.1, side*.6)
                        self.assertEqual(self.node.door, door)
                        self.assertEqual(self.node.door_away_since, 0.)
                        self.assertFalse(self.node.override)

    def test_inconclusive_toward_intent_uses_heading_or_forward_center_bearing(self):
        from smart_wheelchair_safety.unified_geometry import Opening
        for phase in ('door_align', 'door_pass', 'door_clear'):
            for side in (-1., 1.):
                for center, heading in (((1.6, side*.45), 0.),
                                        ((1.6, 0.), side*.3),
                                        ((-.1, 0.), side*.3)):
                    with self.subTest(phase=phase, side=side, center=center, heading=heading):
                        door = Opening(center, heading, 1.)
                        self.node.door = door
                        self.node.door_phase = phase
                        self.node.door_away_since = 0.
                        with patch('smart_wheelchair_safety.unified_control_node.time.monotonic') as now:
                            for stamp in (100., 100.4, 100.8):
                                now.return_value = stamp
                                self.send_raw(.1, side*.6)
                                self.assertEqual(self.node.door, door)
                                self.assertEqual(self.node.door_away_since, 0.)

    def test_inconclusive_millimetre_offsets_do_not_count_as_toward_intent(self):
        from smart_wheelchair_safety.unified_geometry import Opening
        for phase in ('door_align', 'door_pass', 'door_clear'):
            for side in (-1., 1.):
                for x in (1.6, .04, -.1):
                    with self.subTest(phase=phase, side=side, x=x):
                        door = Opening((x, side*.001), 0., 1.)
                        self.node.door = door
                        self.node.door_phase = phase
                        self.node.door_away_since = 0.
                        with patch('smart_wheelchair_safety.unified_control_node.time.monotonic') as now:
                            now.return_value = 100.
                            self.send_raw(.021, side*.6)
                            self.assertEqual(self.node.door, door)
                            now.return_value = 100.4
                            self.send_raw(.021, side*.6)
                        self.assertIsNone(self.node.door)
                        self.assertTrue(self.node.override)

    def test_aperture_crossing_retains_door_even_opposite_to_reference_turn(self):
        for phase in ('door_align', 'door_pass'):
            with self.subTest(phase=phase):
                door = self.front_opening(center=(.75, .04), heading=.04)
                self.node.door = door
                self.node.door_phase = phase
                with patch('smart_wheelchair_safety.unified_control_node.time.monotonic') as now:
                    for stamp in (100., 100.4, 100.8):
                        now.return_value = stamp
                        self.send_raw(.5, -.3)
                self.assertEqual(self.node.door, door)
                self.assertEqual(self.node.door_away_since, 0.)

    def test_low_speed_staging_steering_uses_heading_feedback_direction(self):
        door = self.front_opening(center=(1., 0.), heading=.08)
        self.node.door = door
        self.node.door_phase = 'door_align'
        with patch('smart_wheelchair_safety.unified_control_node.time.monotonic') as now:
            for stamp in (100., 100.4, 100.8):
                now.return_value = stamp
                self.send_raw(.1, .6)
        self.assertEqual(self.node.door, door)
        self.assertEqual(self.node.door_away_since, 0.)

    def test_active_door_neutral_input_does_not_cancel_on_acquisition_filters(self):
        from smart_wheelchair_safety.unified_geometry import Opening
        for center, heading in (((.75, .2), .2), ((3.6, .2), .2), ((2., 1.6), .9)):
            with self.subTest(center=center, heading=heading):
                door = Opening(center, heading, 1.)
                self.node.door = door
                self.node.door_phase = 'door_align'
                with patch('smart_wheelchair_safety.unified_control_node.time.monotonic') as now:
                    for stamp in (100., 100.4, 100.8):
                        now.return_value = stamp
                        self.send_raw(.05, 0.)
                self.assertEqual(self.node.door, door)
                self.assertEqual(self.node.door_away_since, 0.)

    def test_toward_input_resets_away_debounce(self):
        self.node.door = self.front_opening(center=(1.6, .45), heading=.18)
        self.node.door_phase = 'door_align'
        with patch('smart_wheelchair_safety.unified_control_node.time.monotonic') as now:
            for stamp, command in ((100., (.5, -.5)), (100.2, (.1, .6)),
                                   (100.4, (.5, -.5)), (100.6, (.5, -.5))):
                now.return_value = stamp
                self.send_raw(*command)
                self.assertIsNotNone(self.node.door)
            now.return_value = 100.8
            self.send_raw(.5, -.5)
        self.assertIsNone(self.node.door)

    def test_low_speed_away_input_still_cancels_in_each_door_phase(self):
        from smart_wheelchair_safety.unified_geometry import Opening
        for phase in ('door_align', 'door_pass', 'door_clear'):
            with self.subTest(phase=phase):
                center = (-.1, .04) if phase == 'door_clear' else (1.6, .45)
                self.node.door = Opening(center, .18, 1.)
                self.node.door_phase = phase
                with patch('smart_wheelchair_safety.unified_control_node.time.monotonic') as now:
                    now.return_value = 100.
                    self.send_raw(.1, -.6)
                    self.assertIsNotNone(self.node.door)
                    now.return_value = 100.4
                    self.send_raw(.1, -.6)
                self.assertIsNone(self.node.door)
                self.assertTrue(self.node.override)

    def test_door_cancels_only_after_sustained_away_steering(self):
        self.node.door = self.front_opening(center=(1.4, .4), heading=.15, width=1.)
        self.node.door_phase = 'door_align'

        self.send_raw(.5, -.5)

        self.assertIsNotNone(self.node.door)
        self.node.door_away_since -= .36
        self.send_raw(.5, -.5)

        self.assertIsNone(self.node.door)
        self.assertTrue(self.node.override)

    def test_door_pass_retains_target_near_plane_despite_acquisition_range(self):
        self.node.door = self.front_opening(center=(.75, 0.), heading=0., width=1.)
        self.node.door_phase = 'door_pass'

        self.send_raw(.5, 0.)

        self.assertIsNotNone(self.node.door)
        self.assertEqual(self.node.door_away_since, 0.)

    def test_committed_door_near_plane_debounces_explicit_away_steering(self):
        for phase in ('door_pass', 'door_clear'):
            with self.subTest(phase=phase):
                self.node.door = self.front_opening(center=(.75, 0.), heading=0., width=1.)
                self.node.door_phase = phase

                self.send_raw(.5, .5)

                self.assertIsNotNone(self.node.door)
                self.assertGreater(self.node.door_away_since, 0.)
                self.node.door_away_since -= .36
                self.send_raw(.5, .5)

                self.assertIsNone(self.node.door)
                self.assertTrue(self.node.override)

    def test_door_pass_cancels_after_sustained_away_steering_before_plane(self):
        self.node.door = self.front_opening(center=(1.4, .4), heading=.15, width=1.)
        self.node.door_phase = 'door_pass'

        self.send_raw(.5, -.5)

        self.assertIsNotNone(self.node.door)
        self.node.door_away_since -= .36
        self.send_raw(.5, -.5)

        self.assertIsNone(self.node.door)
        self.assertTrue(self.node.override)

    def test_stop_clears_uncommitted_opening_observations(self):
        self.confirm_door(center=(2., 0.), heading=0., width=1.)

        self.send_raw(0., 0.)

        self.assertEqual(self.node.opening_candidates, [])
        self.assertEqual(self.node.confirmed_openings, [])

    def test_door_align_stops_instead_of_rotating_in_place(self):
        self.node.door = self.front_opening(center=(1., 0.), heading=.08, width=1.)
        self.node.door_phase = 'door_align'
        self.node.mode = 'door_align'
        self.node.raw = np.array([.6, 0.])
        self.node.planned = np.array([0., .3])

        for _ in range(10):
            self.node.last_tick -= .1
            self.node.control()

        self.assertAlmostEqual(self.commands[-1].linear.x, 0.)
        self.assertAlmostEqual(self.commands[-1].angular.z, 0.)

    def test_feasible_door_path_keeps_a_safe_minimum_alignment_speed(self):
        self.node.door = self.front_opening(center=(2., .2), heading=.2, width=1.)
        self.node.door_phase = 'door_align'
        self.node.mode = 'door_align'
        self.node.door_path = np.array([[0., 0., 0.], [.3, .02, .1], [2., .2, .2]])
        self.node.door_path_feasible = True
        self.node.raw = np.array([.8, 0.])
        self.node.planned = np.array([0., 0.])

        for _ in range(30):
            self.node.raw_time = self.node.odom_time = time.monotonic()
            self.node.odom_stamp = rospy.Time.now().to_sec()
            self.node.last_tick -= .1
            self.node.control()

        self.assertGreaterEqual(self.commands[-1].linear.x, .17)
        self.assertGreater(self.commands[-1].angular.z, 0.)

    def test_feasible_door_path_survives_planner_action_timeout(self):
        self.node.door = self.front_opening(center=(2., .2), heading=.2, width=1.)
        self.node.door_phase = 'door_align'
        self.node.mode = 'door_align'
        self.node.door_path = np.array([[0., 0., 0.], [.3, .02, .1], [2., .2, .2]])
        self.node.door_path_feasible = True
        self.node.raw = np.array([.8, 0.])
        self.node.plan_time = 0.

        for _ in range(30):
            self.node.raw_time = self.node.odom_time = time.monotonic()
            self.node.odom_stamp = rospy.Time.now().to_sec()
            self.node.last_tick -= .1
            self.node.control()

        self.assertGreaterEqual(self.commands[-1].linear.x, .17)
        self.assertNotEqual(self.node.reason, 'planner_timeout')

    def test_door_speed_slew_preserves_preview_curvature(self):
        self.node.door = self.front_opening(center=(2., .2), heading=.2, width=1.)
        self.node.door_phase = 'door_align'
        self.node.mode = 'door_align'
        self.node.door_path = np.array([[0., 0., 0.], [.3, .03, .1], [2., .2, .2]])
        self.node.door_path_feasible = True
        self.node.raw = np.array([.8, 0.])
        self.node.output = np.array([.05, -.1])
        self.node.acceleration = np.array([0., 0.])

        self.node.last_tick -= .1
        self.node.control()

        command = self.commands[-1]
        self.assertAlmostEqual(command.angular.z,
                               self.node._door_preview_angular(command.linear.x))
        self.assertGreater(command.angular.z, 0.)

    def test_door_tracking_keeps_short_initial_avoidance_curvature(self):
        self.node.door_path = np.array([
            [0., 0., 0.],
            [.07996, .00240, .06],
            [.15982, .00719, .06],
            [.23967, .01199, .06],
        ])
        self.node.pose = (.03999, .00060, .03)

        curvature = self.node._door_preview_angular(.1) / .1

        self.assertAlmostEqual(curvature, .75, delta=.12)

    def test_nearly_aligned_door_creeps_instead_of_stalling_on_map_noise(self):
        from smart_wheelchair_safety.unified_geometry import Opening
        self.node.door = Opening((.97, .184), .184, 1.221, ())
        self.node.door_phase = 'door_align'
        self.node.mode = 'door_align'
        self.node.door_path = np.array([[0., 0., 0.], [.25, .04, .16], [1., .2, .16]])
        self.node.raw = np.array([.6, 0.])
        self.node.planned = np.array([0., 0.])

        for _ in range(10):
            self.node.last_tick -= .1
            self.node.control()

        self.assertGreater(self.commands[-1].linear.x, 0.)
        self.assertGreater(self.commands[-1].angular.z, .02)

    def test_door_align_keeps_forward_motion_while_steering(self):
        self.node.door = self.front_opening(center=(2., .2), heading=.2, width=1.)
        self.node.door_phase = 'door_align'
        self.node.mode = 'door_align'
        self.node.raw = np.array([.6, 0.])
        self.node.planned = np.array([.2, .3])

        for _ in range(10):
            self.node.last_tick -= .1
            self.node.control()

        self.assertGreater(self.commands[-1].linear.x, 0.)
        self.assertGreater(self.commands[-1].angular.z, 0.)

    def test_door_align_uses_locked_path_preview_steering(self):
        from smart_wheelchair_safety.unified_geometry import door_alignment_reference
        door = self.front_opening(center=(2.32, .34), heading=math.radians(45.), width=1.22)
        self.node.door = door
        self.node.door_phase = 'door_align'
        self.node.mode = 'door_align'
        self.node.door_path = door_alignment_reference(door, 'align')
        self.node.raw = np.array([.6, 0.])
        self.node.planned = np.array([.2, 0.])

        for _ in range(10):
            self.node.last_tick -= .1
            self.node.control()

        self.assertGreater(self.commands[-1].linear.x, 0.)
        self.assertGreater(self.commands[-1].angular.z, .02)

    def test_stale_planner_never_replays_old_motion(self):
        self.node.output = np.array([.4, .1])
        self.node.plan_time = 0.
        self.node.control()
        self.assertEqual(self.commands[-1].linear.x, 0.)
        self.assertEqual(self.commands[-1].angular.z, 0.)

    def test_stale_lidar_or_joystick_stops(self):
        for field in ('raw_time', 'odom_time'):
            setattr(self.node, field, 0.)
            self.node.control()
            self.assertEqual(self.commands[-1].linear.x, 0.)
            setattr(self.node, field, time.monotonic())
        self.node.scans['left'] = (0., np.array([[3., 2.]]))
        self.node.control()
        self.assertEqual(self.commands[-1].linear.x, 0.)

    def test_away_input_marks_immediate_override(self):
        from geometry_msgs.msg import Twist
        self.node.wall_side = 1
        msg = Twist()
        msg.linear.x, msg.angular.z = .5, -.4
        self.node.on_raw(msg)
        self.assertTrue(self.node.override)

    def test_oblique_door_intent_beats_existing_wall_override(self):
        self.confirm_door(center=(1.6, .45), heading=.18, width=1.)
        self.node.wall_side = -1

        self.send_raw(.6, .30)
        self.node.update_reference()
        self.finish_door_plan()

        self.assertFalse(self.node.override)
        self.assertEqual(self.node.mode, 'door_align')

    def test_oversteered_current_door_intent_beats_wall_following(self):
        self.confirm_door(center=(3.3168, 1.1490), heading=.4428, width=1.3008)
        self.node.raw = np.array([.8977, .5182])
        self.node.wall_side = 1

        self.node.update_reference()
        self.finish_door_plan()

        self.assertEqual(self.node.mode, 'door_align')

    def test_side_front_door_recaptures_after_wall_has_turned_chair(self):
        self.confirm_door(center=(1.63, 1.20), heading=math.radians(69.), width=1.22)
        self.node.wall_side = -1
        self.node.mode = 'wall'

        self.send_raw(.3, 0.)
        self.node.update_reference()
        self.finish_door_plan()

        self.assertEqual(self.node.mode, 'door_align')
        self.assertIsNotNone(self.node.door)

    def test_straight_oblique_door_intent_tolerates_lidar_fit_variation(self):
        self.confirm_door(center=(2.6, .9), heading=.262, width=1.02)
        self.node.raw = np.array([.833, 0.])

        world, _ = self.node._intended_door()

        self.assertIsNotNone(world)

    def test_full_speed_straight_does_not_capture_distant_side_opening(self):
        self.confirm_door(center=(3.023, .985), heading=.466, width=1.18)
        self.node.raw = np.array([1.667, 0.])

        world, _ = self.node._intended_door()

        self.assertIsNone(world)

    def test_untargeted_offset_door_does_not_capture_straight_command(self):
        self.confirm_door(center=(2., 1.25), heading=0., width=1.)
        self.node.raw = np.array([.6, 0.])

        world, _ = self.node._intended_door()

        self.assertIsNone(world)

    def test_observed_obstacle_is_not_erased_when_body_reaches_it(self):
        self.node.scans['left'] = (time.monotonic(), np.array([[.9, .1]]))
        _, points = self.node.points()
        self.assertTrue(np.any(np.all(points == [.9, .1], axis=1)))
        self.node.control()
        self.assertEqual(self.commands[-1].linear.x, 0.)

    def test_door_memory_uses_latest_complete_frame_without_accumulating_noise(self):
        self.node.door_obstacles = np.array([[1., .4401], [1., -.5]])
        observed = np.array([[1., .452], [1.1, .452], [1.2, .452],
                             [1., -.5], [1.1, -.5], [1.2, -.5]])

        self.node._refresh_door_obstacles(observed)

        self.assertFalse(np.any(np.all(self.node.door_obstacles == [1., .4401], axis=1)))
        refreshed = self.node.door_obstacles.copy()
        self.node._refresh_door_obstacles(observed[:2])
        np.testing.assert_array_equal(self.node.door_obstacles, refreshed)

    def test_failed_door_path_is_retried_with_the_next_scan(self):
        # Local mocks are intentionally not serialized into a worker process.
        from concurrent.futures import ThreadPoolExecutor
        self.node.door_plan_executor.shutdown()
        self.node.door_plan_executor = ThreadPoolExecutor(max_workers=1)
        fallback = np.array([[0., 0., 0.], [1., .1, .1]])
        feasible = np.array([[0., 0., 0.], [.08, 0., 0.], [1., .1, .1]])
        self.node.door_phase = 'door_align'
        self.node.door_path = None
        with patch(
            'smart_wheelchair_safety.unified_control_node.collision_aware_door_reference',
            side_effect=[None, feasible],
        ) as planner, patch(
            'smart_wheelchair_safety.unified_control_node.door_alignment_reference',
            return_value=fallback,
        ):
            self.node._local_door_path(self.front_opening(), np.empty((0, 2)))
            self.node.door_plan_request[0].result(timeout=1.)
            self.node._local_door_path(self.front_opening(), np.empty((0, 2)))
            self.node.door_plan_request[0].result(timeout=1.)
            self.node._local_door_path(self.front_opening(), np.empty((0, 2)))

        self.assertEqual(planner.call_count, 2)
        self.assertTrue(self.node.door_path_feasible)

    def test_door_path_search_does_not_block_control_callbacks(self):
        from concurrent.futures import ThreadPoolExecutor
        self.node.door_plan_executor.shutdown()
        self.node.door_plan_executor = ThreadPoolExecutor(max_workers=1)
        started = threading.Event()
        release = threading.Event()
        feasible = np.array([[0., 0., 0.], [.08, 0., 0.], [1., .1, .1]])

        def delayed_plan(*_):
            started.set()
            release.wait(1.)
            return feasible

        self.node.door_phase = 'door_align'
        with patch(
            'smart_wheelchair_safety.unified_control_node.collision_aware_door_reference',
            side_effect=delayed_plan,
        ):
            before = time.monotonic()
            self.node._local_door_path(self.front_opening(), np.empty((0, 2)))
            elapsed = time.monotonic()-before
            release.set()

        self.assertTrue(started.is_set())
        self.assertLess(elapsed, .05)

    def test_reference_timer_runs_at_five_hz(self):
        self.assertAlmostEqual(self.timer_calls[0].args[0].to_sec(), .2)
        self.assertAlmostEqual(self.timer_calls[1].args[0].to_sec(), .05)

    def test_reference_publishes_ros1_path_and_float_speed_limit(self):
        from nav_msgs.msg import Path
        from std_msgs.msg import Float32
        references, limits = [], []
        self.node.reference = SimpleNamespace(publish=references.append)
        self.node.limit_pub = SimpleNamespace(publish=limits.append)

        self.node.update_reference()

        self.assertIsInstance(references[-1], Path)
        self.assertEqual(references[-1].header.frame_id, 'odom')
        self.assertIsInstance(limits[-1], Float32)
        self.assertAlmostEqual(limits[-1].data,
                               min(self.node.raw[0], parameter(self.node, 'max_speed')))
        self.assertTrue(self.node.accept_planned)
        self.assertEqual(self.node.active_reference_stamp, references[-1].header.stamp)

    def test_non_positive_or_non_finite_parameters_are_rejected(self):
        from smart_wheelchair_safety.unified_control_node import UnifiedControlNode
        for value in (0., -.1, float('nan'), float('inf')):
            with self.subTest(value=value), patch.object(rospy, 'get_param', return_value=value):
                with self.assertRaisesRegex(ValueError, 'finite and positive'):
                    UnifiedControlNode()

    def test_shutdown_stops_once_and_discards_queued_callbacks(self):
        from geometry_msgs.msg import Twist
        self.node.raw[:] = 0.
        queued = self.node._serialized(self.node.on_raw)
        command = Twist()
        command.linear.x = .7

        self.node.destroy_node()
        queued(command)
        self.node.destroy_node()

        np.testing.assert_array_equal(self.node.raw, [0., 0.])
        self.assertEqual(len(self.commands), 1)
        self.assertEqual(self.commands[-1].linear.x, 0.)

    def test_command_ramp_reaches_acceleration_limits_within_point_two_seconds(self):
        self.node.raw = np.array([.8, .65])
        self.node.planned = np.array([.8, .65])
        self.node.output[:] = 0.
        self.node.acceleration[:] = 0.
        samples = []
        increment_limits = []
        for _ in range(4):
            self.node.last_tick -= .05
            previous_tick = self.node.last_tick
            self.node.control()
            samples.append([self.commands[-1].linear.x,
                            self.commands[-1].angular.z])
            dt = min(.1, max(.001, self.node.last_tick-previous_tick))
            increment_limits.append(np.array([.5, .8])*dt)
        samples = np.asarray(samples)

        np.testing.assert_allclose(self.node.acceleration, [.5, .8], atol=.01)
        increments = np.diff(np.vstack(([0., 0.], samples)), axis=0)
        self.assertTrue(np.all(increments <= np.asarray(increment_limits) + 1e-6))

    def test_driver_stop_is_immediate_and_clears_acceleration(self):
        self.node.raw[:] = 0.
        self.node.output[:] = .4
        self.node.acceleration[:] = .3
        self.node.control()
        np.testing.assert_array_equal(self.node.output, [0., 0.])
        np.testing.assert_array_equal(self.node.acceleration, [0., 0.])

    def test_front_stop_does_not_disable_manual_rotation(self):
        self.node.mode = 'front_stop'
        self.node.raw = np.array([0., .4])
        self.node.plan_time = 0.
        self.node.raw_time = self.node.odom_time = time.monotonic()
        self.node.odom_stamp = rospy.Time.now().to_sec()
        self.node.last_tick -= .05
        self.node.control()
        self.assertEqual(self.commands[-1].linear.x, 0.)
        self.assertGreater(self.commands[-1].angular.z, 0.)

    def test_override_remains_active_across_reference_refresh(self):
        from geometry_msgs.msg import Twist
        self.node.wall_side = 1
        msg = Twist()
        msg.linear.x, msg.angular.z = .5, -.4
        self.node.on_raw(msg)
        self.node.update_reference()
        self.node.on_raw(msg)
        self.assertTrue(self.node.override)
        self.assertEqual(self.node.mode, 'override')

    def test_forward_intent_never_accepts_negative_planner_velocity(self):
        self.node.planned = np.array([-.01, .2])
        self.node.last_tick -= .05
        self.node.control()
        self.assertGreaterEqual(self.commands[-1].linear.x, 0.)

    def test_front_stop_survives_temporary_line_fragmentation(self):
        self.node.front_blocked = True
        self.node.update_reference()
        self.assertEqual(self.node.mode, 'front_stop')

    def test_valid_infinite_scan_is_free_space_not_sensor_loss(self):
        from geometry_msgs.msg import TransformStamped
        from sensor_msgs.msg import LaserScan
        transform = TransformStamped()
        transform.transform.rotation.w = 1.
        self.node.tf = SimpleNamespace(lookup_transform=lambda *args: transform)
        scan = LaserScan()
        scan.header.stamp = rospy.Time.now()
        scan.range_min, scan.range_max = .05, 8.
        scan.ranges = [float('inf')] * 10
        scan.angle_increment = .1
        self.node.scans.clear()
        self.node.on_scan(scan, 'left')
        self.assertIn('left', self.node.scans)
        self.assertEqual(len(self.node.scans['left'][1]), 0)

    def test_odometry_callback_runs_during_scan_tf_wait_then_scan_commits(self):
        from geometry_msgs.msg import TransformStamped
        from nav_msgs.msg import Odometry
        from sensor_msgs.msg import LaserScan
        waiting, release, odom_done = threading.Event(), threading.Event(), threading.Event()
        transform = TransformStamped()
        transform.transform.rotation.w = 1.

        def lookup(*_):
            waiting.set()
            release.wait(1.)
            return transform

        self.node.tf = SimpleNamespace(lookup_transform=lookup)
        scan = LaserScan(angle_increment=.1, range_min=.05, range_max=8.,
                         ranges=[float('inf')] * 10)
        scan.header.stamp = rospy.Time.now()
        odom = Odometry()
        odom.header.stamp = scan.header.stamp
        odom.pose.pose.position.x = .2
        odom.pose.pose.orientation.w = 1.
        observed_poses = []
        self.node._observe_openings = lambda _: observed_poses.append(self.node.pose)

        def receive_odom():
            self.callbacks['odom'](odom)
            odom_done.set()

        scanner = threading.Thread(target=self.callbacks['scan_left'], args=(scan,))
        odometry = threading.Thread(target=receive_odom)
        scanner.start()
        try:
            self.assertTrue(waiting.wait(.5))
            odometry.start()
            self.assertTrue(odom_done.wait(.1), 'TF lookup blocked odometry callback')
        finally:
            release.set()
            scanner.join(1.)
            if odometry.ident is not None:
                odometry.join(1.)
        self.assertFalse(scanner.is_alive())
        self.assertEqual(len(self.node.scans['left'][1]), 0)
        self.assertEqual(observed_poses[-1], (.2, 0., 0.))

    def test_scan_waiting_on_tf_does_not_commit_after_shutdown(self):
        from geometry_msgs.msg import TransformStamped
        from sensor_msgs.msg import LaserScan
        previous = self.node.scans['left']
        transform = TransformStamped()
        transform.transform.rotation.w = 1.

        def lookup(*_):
            self.node.destroy_node()
            return transform

        self.node.tf = SimpleNamespace(lookup_transform=lookup)
        scan = LaserScan(angle_increment=.1, range_min=.05, range_max=8.,
                         ranges=[float('inf')] * 10)
        scan.header.stamp = rospy.Time.now()
        self.callbacks['scan_left'](scan)

        self.assertIs(self.node.scans['left'], previous)
        self.assertTrue(self.node.stopped)

    def test_corrupt_scan_does_not_refresh_watchdog(self):
        from sensor_msgs.msg import LaserScan
        scan = LaserScan()
        scan.header.stamp = rospy.Time.now()
        scan.range_min, scan.range_max = .05, 8.
        scan.ranges = [float('nan')] * 10
        self.node.scans.clear()
        self.node.on_scan(scan, 'left')
        self.assertNotIn('left', self.node.scans)

    def test_cancelling_door_assistance_keeps_nearby_jamb_observations(self):
        self.node.door_obstacles = np.array([[.2, .5], [8., .5]])
        self.node.door_obstacle_time = time.monotonic()
        self.node.door = None
        self.node.raw[:] = 0.
        self.node.update_reference()
        np.testing.assert_array_equal(self.node.door_obstacles, [[.2, .5]])

    def test_cancelled_door_jamb_memory_expires(self):
        self.node.door_obstacles = np.array([[.2, .5]])
        self.node.door_obstacle_time = 100.
        self.node.door = None

        with patch('smart_wheelchair_safety.unified_control_node.time.monotonic',
                   return_value=101.51):
            self.node.update_reference()

        self.assertEqual(len(self.node.door_obstacles), 0)

    def test_delayed_odometry_does_not_refresh_watchdog(self):
        from nav_msgs.msg import Odometry
        msg = Odometry()
        msg.pose.pose.orientation.w = 1.
        self.node.odom_time = 0.
        self.node.on_odom(msg)
        self.assertEqual(self.node.odom_time, 0.)

    def test_odom_dropout_stops_even_with_fresh_receipt_and_override(self):
        self.node.odom_stamp -= .15
        self.node.override = True
        self.node.output = np.array([.8, .2])
        self.node.control()
        np.testing.assert_array_equal(self.node.output, [0., 0.])


if __name__ == '__main__':
    import rostest
    rostest.rosrun('smart_wheelchair_safety', 'unified_control', UnifiedNodeTest)
