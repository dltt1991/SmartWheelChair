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
from unittest.mock import patch, Mock
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

    def _successful_duplicate_fixture(self):
        n=self.node
        n.use_sim_time=True; n.assist_enabled=True; n.mode_neutral_seen=True
        n.mode='manual'; n.clock_stalled=False; n.clock_last_ros=2037.17
        n.clock_last_advance_wall=100.; n.raw_resume_wall=0.
        n.last_tick=2037.05; n.raw_time=n.odom_time=n.plan_time=100.
        n.odom_ros_time=n.plan_ros_time=n.odom_stamp=2037.17
        n.raw=np.array([.15,0.]); n.planned=n.raw.copy(); n.measured=n.raw.copy()
        n.output=n.raw.copy(); n.acceleration=np.zeros(2)
        n.use_neupan=False; n.accept_planned=True
        n.active_reference_stamp=rospy.Time.from_sec(2037.17)
        n.scans={side:(100.,np.array([[3.,2.],[3.,-2.]])) for side in ('left','right')}
        n.status=SimpleNamespace(publish=Mock())
        with patch.object(time,'monotonic',return_value=100.01),patch.object(
                rospy.Time,'now',return_value=rospy.Time.from_sec(2037.17)):
            n.control()
        self.assertEqual(n.reason,'clear')
        self.assertAlmostEqual(n.output[0],.15)

    def _same_ros_control(self,wall=100.010684375):
        with patch.object(time,'monotonic',return_value=wall),patch.object(
                rospy.Time,'now',return_value=rospy.Time.from_sec(2037.17)):
            self.node.control()

    def test_successful_tick_same_stamp_is_no_publish_no_integral_no_revocation(self):
        self._successful_duplicate_fixture();n=self.node
        output=n.output.copy();acc=n.acceleration.copy();epoch=n.epoch
        ref=n.active_reference_stamp;rawtime=n.raw_time;commands=len(self.commands)
        statuses=n.status.publish.call_count
        self._same_ros_control()
        self.assertEqual(len(self.commands),commands)
        self.assertEqual(n.status.publish.call_count,statuses)
        np.testing.assert_array_equal(n.output,output);np.testing.assert_array_equal(n.acceleration,acc)
        self.assertEqual(n.epoch,epoch);self.assertEqual(n.raw_time,rawtime)
        self.assertEqual(n.active_reference_stamp,ref);self.assertFalse(n.clock_stalled)
        with patch.object(time,'monotonic',return_value=100.04),patch.object(
                rospy.Time,'now',return_value=rospy.Time.from_sec(2037.20)):
            n.control()
        self.assertAlmostEqual(json.loads(n.status.publish.call_args.args[0].data)['control_dt_s'],.03)
        self.assertEqual(n.reason,'clear')

    def test_duplicate_new_input_or_safety_state_never_bypasses_original_stop(self):
        from geometry_msgs.msg import Twist
        from std_msgs.msg import Bool
        for change in ('raw','odom','scan','local','nn','reference','parameters','neutral','nan','cancel','assist','mode'):
            with self.subTest(change=change):
                self._successful_duplicate_fixture();n=self.node
                if change in ('raw','odom','local','nn'):
                    attr={'raw':'raw_time','odom':'odom_time','local':'plan_time','nn':'neupan_time'}[change]
                    setattr(n,attr,100.0101)
                elif change=='scan':n.scans['left']=(100.0101,np.array([[.1,0.]]))
                elif change=='reference':n.active_reference_stamp=rospy.Time.from_sec(2037.18)
                elif change=='parameters':n.parameters['hard_margin']+=.001
                elif change in ('neutral','nan'):
                    msg=Twist();msg.linear.x=float('nan') if change=='nan' else 0.
                    n.on_raw(msg,100.0101)
                elif change=='cancel':n.cancel()
                elif change=='assist':n.on_assist_enabled(Bool(data=False))
                elif change=='mode':n.mode='door_wait'
                self._same_ros_control()
                np.testing.assert_array_equal(n.output,[0.,0.]);self.assertTrue(n.clock_stalled)

    def test_duplicate_requires_prior_clear_and_fresh_clock_without_rollback(self):
        for change in ('no_success','stalled','rollback','raw_expired','clock_deadline'):
            with self.subTest(change=change):
                self._successful_duplicate_fixture();n=self.node
                if change=='no_success':n.last_clear_tick=None
                elif change=='stalled':n.clock_stalled=True
                elif change=='rollback':n.clock_last_ros=2037.18
                elif change=='raw_expired':n.raw_time=99.
                elif change=='clock_deadline':n.clock_last_advance_wall=99.
                self._same_ros_control();np.testing.assert_array_equal(n.output,[0.,0.])
                self.assertTrue(n.clock_stalled)

    def test_duplicate_does_not_refresh_independent_freeze_watchdog(self):
        self._successful_duplicate_fixture();n=self.node;epoch=n.epoch
        for wall in (100.02,100.10,100.20):self._same_ros_control(wall)
        self.assertEqual(n.epoch,epoch);self.assertEqual(n.clock_last_advance_wall,100.)
        n._clock_watchdog_tick(100.251,2037.17)
        self.assertTrue(n.clock_stalled);np.testing.assert_array_equal(n.output,[0.,0.])
        self.assertEqual(n.epoch,epoch+1)
        n._clock_watchdog_tick(100.30,2037.17);self.assertEqual(n.epoch,epoch+1)
        n._clock_watchdog_tick(100.31,2037.18)
        self.assertEqual(n.raw_time,0.);self.assertFalse(n.clock_stalled)

    def test_ordinary_spin_and_tiny_forward_variable_targets_keep_actual_jerk(self):
        for raw_v in (0.,.01,.02):
            self.node.output=np.zeros(2);self.node.acceleration=np.zeros(2)
            previous_a=np.zeros(2)
            for i in range(160):
                dt=(.001,.01,.02,.05,.09,.13)[i%6]
                target_w=.11 if i<50 else -.3 if i<100 else .02
                before=self.node.output.copy()
                command=self._ordinary_reverse_tick([raw_v,target_w],dt)
                a=(command-before)/dt
                self.assertTrue(np.all(np.abs(a-previous_a)<=np.array([2.5,4.])*dt+1e-8),(raw_v,i,dt,a,previous_a))
                self.assertTrue(np.all(np.abs(a)<=np.array([.5,.8])+1e-8))
                self.assertGreaterEqual(command[0],-1e-10)
                np.testing.assert_allclose(self.node.acceleration,a,atol=1e-10)
                previous_a=a

    def test_joystick_slew_does_not_change_direct_or_recovery_policy(self):
        self.node.assist_enabled=False
        with patch.object(self.node,'_assisted_slew',side_effect=AssertionError('manual_direct entered helper')):
            np.testing.assert_allclose(self._ordinary_reverse_tick([0.,.3],.05),[0.,.3])
        self.node.assist_enabled=True;self.node.output=np.zeros(2);self.node.acceleration=np.zeros(2)
        self.node.raw=np.array([0.,.3]);self.node.mode='manual'
        self.node.last_tick=99.95;self.node.odom_stamp=100.
        with patch.object(rospy.Time,'now',return_value=rospy.Time.from_sec(100.)),patch.object(
                self.node,'fresh',return_value=True),patch.object(self.node,'_near_recovery',return_value=True),patch.object(
                self.node,'_assisted_slew',side_effect=AssertionError('recovery entered helper')),patch(
                'smart_wheelchair_safety.unified_control_node.braking_clear',return_value=False):
            self.node.control()
        np.testing.assert_array_equal(self.node.output,[0.,0.])
        self.assertEqual(self.node.reason,'emergency_stop')

    def test_existing_reverse_inertia_survives_joystick_and_planned_handoffs(self):
        for switch_tick in (4,5,25):
            for next_target in ([0.,.2],[.01,.2],[.25,.2]):
                with self.subTest(switch_tick=switch_tick,next_target=next_target):
                    self.node.output=np.array([.08,0.]);self.node.acceleration=np.zeros(2)
                    previous_a=np.zeros(2)
                    for i in range(110):
                        target=[-.2,.2] if i<switch_tick else next_target
                        dt=(.01,.02,.05,.09,.13)[i%5]
                        before=self.node.output.copy()
                        self.node.planned=np.array(target);self.node.plan_time=time.monotonic();self.node.plan_ros_time=100.
                        command=self._ordinary_reverse_tick(target,dt)
                        a=(command-before)/dt
                        self.assertTrue(np.all(np.abs(a)<=np.array([.5,.8])+1e-8),(i,a))
                        self.assertTrue(np.all(np.abs(a-previous_a)<=np.array([2.5,4.])*dt+1e-8),(i,a,previous_a))
                        self.assertGreaterEqual(command[0],-.4-1e-10)
                        previous_a=a
                    self.assertGreaterEqual(self.node.output[0],-1e-9)

    def test_forward_only_keeps_original_zero_bound(self):
        self.node.output=np.zeros(2);self.node.acceleration=np.zeros(2)
        for i in range(120):
            dt=(.01,.02,.05,.09,.13)[i%5]
            target=np.array([.25,.1] if i<50 else [.01,.1])
            expected,_=self.node._assisted_slew(target,dt)
            self.node.planned=target;self.node.plan_time=time.monotonic();self.node.plan_ros_time=100.
            actual=self._ordinary_reverse_tick(target,dt)
            np.testing.assert_allclose(actual,expected,atol=1e-10)
            self.assertGreaterEqual(actual[0],0.)

    def _start_transition_test(self):
        from geometry_msgs.msg import Twist
        self.node.use_sim_time=True;self.node.mode='manual';self.node.mode_neutral_seen=True
        self.node.clock_stalled=False;self.node.raw_resume_wall=0.
        self.node.raw=np.array([-.1,0.]);self.node.raw_time=1000.
        self.node.output=np.array([-.1,0.]);self.node.acceleration=np.zeros(2)
        self.node.cancel();self.node.use_neupan=True
        self._transition_ros=50.;self._transition_wall=1000.01
        msg=Twist();msg.linear.x=.15
        with patch.object(time,'monotonic',return_value=self._transition_wall),patch.object(
                rospy.Time,'now',return_value=rospy.Time.from_sec(self._transition_ros)),patch.object(self.node,'fresh',return_value=True):
            self.node.on_raw(msg,received=self._transition_wall)

    def _transition_control_test(self,dt=.05,fresh=True,guard=True,recovering=False):
        self.node.last_tick=self._transition_ros;self._transition_ros+=dt;self._transition_wall+=dt
        self.node.measured=self.node.output.copy();self.node.odom_stamp=self._transition_ros
        with patch.object(time,'monotonic',return_value=self._transition_wall),patch.object(
                rospy.Time,'now',return_value=rospy.Time.from_sec(self._transition_ros)),patch.object(
                self.node,'fresh',return_value=fresh),patch.object(self.node,'_near_recovery',return_value=recovering),patch(
                'smart_wheelchair_safety.unified_control_node.braking_clear',return_value=guard):
            self.node.control()
        return json.loads(self.node.status.publish.call_args.args[0].data) if isinstance(self.node.status.publish,Mock) else None

    def test_initial_reverse_forward_wait_preserves_actual_slew_and_audit_source(self):
        self._start_transition_test();self.node.status=SimpleNamespace(publish=Mock())
        previous_a=np.zeros(2)
        for dt in (.05,.05,.05):
            before=self.node.output.copy();status=self._transition_control_test(dt)
            a=(self.node.output-before)/dt
            self.assertTrue(np.all(np.abs(a-previous_a)<=np.array([2.5,4.])*dt+1e-8))
            self.assertEqual(status['reason'],'clear');self.assertEqual(status['planner_source'],'transition_stop')
            self.assertLessEqual(self.node.output[0],0.)
            previous_a=a
        self.assertIsNone(self.node.active_reference_stamp)
        self.assertFalse(self.node.accept_planned)

    def test_initial_wait_deadline_cannot_be_renewed_by_forward_or_reference(self):
        from geometry_msgs.msg import Twist
        self._start_transition_test()
        for _ in range(4):
            self._transition_control_test(.05)
            msg=Twist();msg.linear.x=.15
            with patch.object(time,'monotonic',return_value=self._transition_wall),patch.object(self.node,'fresh',return_value=True):
                self.node.on_raw(msg,received=self._transition_wall)
            self.node.active_reference_stamp=rospy.Time.from_sec(self._transition_ros)
            self.node.accept_planned=True
        self._transition_control_test(.051)
        self.assertEqual(self.node.reason,'planner_timeout');np.testing.assert_array_equal(self.node.output,[0.,0.])

    def test_initial_wait_accepts_only_new_authorized_action_and_never_masks_later_expiry(self):
        from geometry_msgs.msg import TwistStamped
        self._start_transition_test()
        stale=TwistStamped();stale.header.stamp=rospy.Time.from_sec(49.);stale.twist.linear.x=.8
        self.node.on_planned(stale,1000.,49.)
        self.assertEqual(self.node.plan_time,0.)
        self._transition_control_test()
        self.assertEqual(self.node.reason,'clear')
        self.node.active_reference_stamp=rospy.Time.from_sec(self._transition_ros);self.node.accept_planned=True
        self.node.on_planned(stale,1000.,49.)
        self.assertEqual(self.node.plan_time,0.)
        new=TwistStamped();new.header.stamp=self.node.active_reference_stamp;new.twist.linear.x=.15
        self.node.on_planned(new,self._transition_wall,self._transition_ros)
        self._transition_control_test()
        self.assertEqual(self.node.reason,'clear')
        self.node.plan_time=0.;self.node.plan_ros_time=0.
        self._transition_control_test()
        self.assertEqual(self.node.reason,'planner_timeout')

    def test_initial_wait_cancel_sensor_clock_and_guard_priorities(self):
        from geometry_msgs.msg import Twist
        for boundary in ('cancel','sensor','clock','guard','neutral','recovery'):
            self._start_transition_test()
            if boundary=='cancel':self.node.cancel()
            if boundary=='clock':self.node._stop_for_clock(self._transition_wall,self._transition_ros)
            if boundary=='neutral':
                self.node.on_raw(Twist(),received=self._transition_wall)
            self._transition_control_test(fresh=boundary not in ('sensor','clock'),guard=boundary not in ('guard','recovery'),recovering=boundary=='recovery')
            np.testing.assert_array_equal(self.node.output,[0.,0.],err_msg=boundary)
            self.assertNotEqual(self.node.reason,'clear',boundary)

    def test_pre_resume_forward_does_not_open_initial_wait(self):
        from geometry_msgs.msg import Twist
        self._start_transition_test();self.node.cancel();self.node.raw=np.zeros(2);self.node.raw_time=0.
        self.node.raw_resume_wall=1001.
        msg=Twist();msg.linear.x=.15
        self.node.on_raw(msg,received=1000.9)
        self.assertEqual(self.node.raw[0],0.)
        self._transition_control_test()
        self.assertEqual(self.node.reason,'user_stop')

    def test_initial_wait_signed_zero_and_variable_dt_are_real_derivative_bounded(self):
        for output,periods in (([-.1,0.],(.01,.02,.05,.09,.02)),([-.005,.005],(.02,)*11)):
            self._start_transition_test();self.node.output=np.array(output);self.node.acceleration=np.zeros(2)
            previous_a=np.zeros(2)
            for dt in periods:
                before=self.node.output.copy();self._transition_control_test(dt)
                actual_a=(self.node.output-before)/dt
                self.assertEqual(self.node.reason,'clear')
                self.assertTrue(np.all(np.abs(actual_a)<=np.array([.5,.8])+1e-8))
                self.assertTrue(np.all(np.abs(actual_a-previous_a)<=np.array([2.5,4.])*dt+1e-8))
                self.assertLessEqual(self.node.output[0],0.)
                self.assertGreaterEqual(self.node.output[1],0.)
                previous_a=actual_a
            if output[0]==-.005:np.testing.assert_array_equal(self.node.output,[0.,0.])

    def test_initial_wait_new_neupan_authority_ends_window_immediately(self):
        from geometry_msgs.msg import TwistStamped
        self._start_transition_test()
        old=TwistStamped();old.header.stamp=rospy.Time.from_sec(49.);old.twist.linear.x=.8
        with patch.object(rospy.Time,'now',return_value=rospy.Time.from_sec(50.)):
            self.node.on_neupan(old,self._transition_wall,50.)
        self.assertIsNotNone(self.node.initial_plan_wait);self.assertEqual(self.node.neupan_time,0.)
        self.node.active_reference_stamp=rospy.Time.from_sec(50.);self.node.accept_planned=True
        old.header.stamp=self.node.active_reference_stamp;old.twist.linear.x=.15
        with patch.object(rospy.Time,'now',return_value=rospy.Time.from_sec(50.)):
            self.node.on_neupan(old,self._transition_wall,50.)
        self.assertIsNone(self.node.initial_plan_wait)
        self._transition_control_test()
        self.assertEqual(self.node.reason,'clear')
        self.node.neupan_time=0.;self.node.neupan_ros_time=0.
        self._transition_control_test()
        self.assertEqual(self.node.reason,'planner_timeout')

    def _ordinary_reverse_tick(self, target, dt, guard=True):
        self.node.use_sim_time=True;self.node.mode='manual';self.node.mode_neutral_seen=True
        self.node.raw=np.array(target);self.node.measured=self.node.output.copy()
        self.node.last_tick=100.-dt;self.node.odom_stamp=100.
        self.node.raw_time=time.monotonic()
        with patch.object(rospy.Time,'now',return_value=rospy.Time.from_sec(100.)),patch.object(
                self.node,'fresh',return_value=True),patch.object(self.node,'_near_recovery',return_value=False),patch(
                'smart_wheelchair_safety.unified_control_node.braking_clear',return_value=guard):
            self.node.control()
        return self.node.output.copy()

    def test_reverse_recorded_target_clip_preserves_actual_jerk(self):
        fixture=json.loads((Path(__file__).parent/'reverse_target_clamp_v19.json').read_text())
        self.node.output=np.array([-.05,0.]);self.node.acceleration=np.zeros(2)
        previous_a=np.zeros(2)
        for row in fixture['rows']:
            before=self.node.output.copy();dt=row['dt']
            output=self._ordinary_reverse_tick([row['target'],0.],dt)
            actual_a=(output-before)/dt
            self.assertTrue(np.all(np.abs(actual_a-previous_a)<=np.array([2.5,4.])*dt+1e-7),row)
            np.testing.assert_allclose(self.node.acceleration,actual_a,atol=1e-7)
            self.assertGreaterEqual(output[0],-.4)
            previous_a=actual_a

    def test_reverse_variable_targets_fixed_bounds_and_forward_inertia(self):
        for initial in (0.,.12):
            self.node.output=np.array([initial,0.]);self.node.acceleration=np.zeros(2)
            previous_a=np.zeros(2)
            for i in range(240):
                dt=(.001,.01,.02,.05,.09,.13)[i%6]
                target=([-.4,.5] if i<80 else [-.08,-.4] if i<160 else [-.25,0.])
                before=self.node.output.copy();output=self._ordinary_reverse_tick(target,dt)
                a=(output-before)/dt
                self.assertTrue(np.all(np.abs(a)<=np.array([.5,.8])+1e-8),(i,a))
                self.assertTrue(np.all(np.abs(a-previous_a)<=np.array([2.5,4.])*dt+1e-8),(i,a,previous_a))
                self.assertGreaterEqual(output[0],-.4-1e-10)
                self.assertLessEqual(output[0],self.node.parameter('max_speed')+1e-10)
                self.assertLessEqual(abs(output[1]),.65+1e-10)
                np.testing.assert_allclose(self.node.acceleration,a,atol=1e-10)
                previous_a=a

    def test_reverse_neutral_and_hardguard_remain_immediate(self):
        self.node.output=np.array([-.2,.1]);self.node.acceleration=np.array([-.2,.1])
        np.testing.assert_array_equal(self._ordinary_reverse_tick([0.,0.],.05),[0.,0.])
        self.assertEqual(self.node.reason,'user_stop')
        self.node.output=np.array([-.2,0.]);self.node.acceleration=np.zeros(2)
        np.testing.assert_array_equal(self._ordinary_reverse_tick([-.2,0.],.05,guard=False),[0.,0.])
        self.assertEqual(self.node.reason,'emergency_stop')

    def _actual_pivot_test_tick(self, error, dt, mode='door_align'):
        self.node.use_sim_time=True;self.node.mode=mode;self.node.door_phase=mode
        self.node.pose=(.03239062019767453,0.,-error)
        self.node.door_path=np.array([[0.,0.,-.2],[0.,0.,0.],[3.,0.,0.]])
        self.node.door_rotation_index=0;self.node.door_path_feasible=True
        self.node.mode_neutral_seen=True;self.node.raw=np.array([.8,0.])
        self.node.measured=self.node.output.copy();self.node.last_tick=100.-dt;self.node.odom_stamp=100.;self.node.plan_ros_time=100.
        with patch.object(rospy.Time,'now',return_value=rospy.Time.from_sec(100.)),patch.object(
                self.node,'fresh',return_value=True),patch.object(self.node,'_near_recovery',return_value=False),patch.object(
                self.node,'_door_near_entry',return_value=True),patch.object(self.node,'_door_angular',side_effect=AssertionError('tracking entered current pivot')),patch(
                'smart_wheelchair_safety.unified_control_node.braking_clear',return_value=True):
            self.node.control()

    def test_actual_pivot_short_tick_does_not_clip_target_after_jerk_recovery(self):
        self.node.output=np.array([0.,.12959405307277758])
        self.node.acceleration=np.array([0.,-.20098690973931427])
        for dt in (.09,.01):
            previous=self.node.output.copy();prior_a=self.node.acceleration.copy()
            self._actual_pivot_test_tick(.0732572175904942,dt,mode='door_pass')
            a=(self.node.output-previous)/dt
            self.assertTrue(np.all(np.abs(a-prior_a)<=np.array([2.5,4.])*dt+1e-8), (dt,a,prior_a))
            np.testing.assert_allclose(self.node.acceleration,a,atol=1e-8)
        self.assertLess(self.node.output[1],.10988582638574129,'target overshoot is necessary to preserve this actual derivative')

    def test_actual_pivot_variable_dt_target_changes_and_settling_keep_derivatives(self):
        periods=(.001,.01,.02,.05,.09,.13)
        for sign in (-1.,1.):
            self.node.output[:]=0.;self.node.acceleration[:]=0.
            prior_a=np.zeros(2)
            # Fixed deterministic target switches model changing angle estimates;
            # command/acceleration state is never reset during the trajectory.
            errors=[sign*.3]*24+[-sign*.18]*24+[sign*.09]*24+[sign*.025]*180
            for tick,error in enumerate(errors):
                dt=periods[tick%len(periods)];before=self.node.output.copy()
                self._actual_pivot_test_tick(error,dt)
                a=(self.node.output-before)/dt
                self.assertTrue(np.all(np.abs(a)<=[.5+1e-8,.8+1e-8]),(sign,tick,a))
                self.assertTrue(np.all(np.abs(a-prior_a)<=np.array([2.5,4.])*dt+1e-8),(sign,tick,dt,a,prior_a))
                self.assertEqual(self.node.output[0],0.)
                self.assertLessEqual(abs(self.node.output[1]),.65)
                prior_a=a
                if tick>=72 and np.max(np.abs(self.node.output))<1e-12 and np.max(np.abs(a))<1e-9:
                    break
            np.testing.assert_array_equal(self.node.output,[0.,0.])

    def _assert_watchdog_samples_after_lock(self, initial_ros, live_ros, newer_state):
        node=self.node;node.use_sim_time=True
        clock={'wall':100.3,'ros':initial_ros}
        node.clock_last_ros=10.;node.clock_last_advance_wall=100.;node.clock_stalled=False
        class QueuedLock:
            def __enter__(self):
                clock.update(wall=100.4,ros=live_ros)
                if newer_state is not None:
                    node.clock_last_ros=newer_state
            def __exit__(self,*args):
                pass
        with patch.object(node,'callback_lock',QueuedLock()),patch(
                'smart_wheelchair_safety.unified_control_node.time.monotonic',side_effect=lambda:clock['wall']),patch.object(
                rospy.Time,'now',side_effect=lambda:rospy.Time.from_sec(clock['ros'])),patch.object(node,'_stop_for_clock') as stop:
            node._clock_watchdog_tick()
        stop.assert_not_called()
        self.assertEqual(node.clock_last_ros,rospy.Time.from_sec(live_ros).to_sec())
        self.assertEqual(node.clock_last_advance_wall,100.4)

    def test_watchdog_lock_wait_cannot_make_clock_progress_a_false_freeze(self):
        self._assert_watchdog_samples_after_lock(10.,10.2,None)

    def test_watchdog_lock_wait_cannot_rewind_newer_serialized_clock_state(self):
        self._assert_watchdog_samples_after_lock(10.1,10.3,10.2)

    def test_recorded_align_to_pass_angular_handoff_preserves_actual_jerk(self):
        cases=json.loads((Path(__file__).parent/'fixtures/door_align_pass_v17.json').read_text())
        for case in cases:
            with self.subTest(case=case['case']):
                rows=case['rows'];self.node.output=np.array([rows[1]['v'],rows[1]['w']])
                self.node.acceleration=np.array([0.,(rows[1]['w']-rows[0]['w'])/rows[1]['control_dt_s']])
                previous_a=self.node.acceleration[1]
                for row in rows[2:]:
                    old=self.node.output[1];dt=row['control_dt_s']
                    self._door_angular_test_tick(row['w'],dt,mode=row['mode'])
                    a=(self.node.output[1]-old)/dt
                    self.assertLessEqual(abs(a),.8+1e-8)
                    self.assertLessEqual(abs(a-previous_a)/dt,4.+1e-7)
                    self.assertAlmostEqual(self.node.acceleration[1],a)
                    previous_a=a

    def test_tracking_overlay_never_replaces_wait_or_settling_pivot(self):
        for settling in (False,True):
            if settling:
                self.setup_recorded_pivot_residual()
            else:
                self.node.mode='door_wait';self.node.door_path_feasible=True
                self.node.door_path=np.array([[0.,0.,0.],[3.,0.,0.]])
                self.node.output=np.array([.3,.1]);self.node.acceleration[:]=0.
                self.node.mode_neutral_seen=True
            self.node.use_sim_time=True;self.node.last_tick=99.95;self.node.odom_stamp=100.
            expected,expected_a=self.node._door_wait_slew(.05)
            with patch.object(rospy.Time,'now',return_value=rospy.Time.from_sec(100.)),patch.object(
                    self.node,'fresh',return_value=True),patch.object(self.node,'_near_recovery',return_value=False),patch.object(
                    self.node,'_door_angular',side_effect=AssertionError('tracking overlay entered wait/pivot')),patch.object(
                    self.node,'_assisted_slew',wraps=self.node._assisted_slew) as assisted,patch(
                    'smart_wheelchair_safety.unified_control_node.braking_clear',return_value=True):
                self.node.control()
            assisted.assert_not_called()
            np.testing.assert_allclose(self.node.output,expected,atol=1e-9)
            np.testing.assert_allclose(self.node.acceleration,expected_a,atol=1e-9)

    def test_active_pivot_retains_original_angular_policy(self):
        self.setup_recorded_pivot_residual();self.node.pose=(0.,0.,-.2)
        self.node.output[:]=0.;self.node.acceleration[:]=0.;self.node.measured[:]=0.
        self.node.use_sim_time=True;self.node.last_tick=99.95;self.node.odom_stamp=100.
        with patch.object(rospy.Time,'now',return_value=rospy.Time.from_sec(100.)),patch.object(
                self.node,'fresh',return_value=True),patch.object(self.node,'_near_recovery',return_value=False),patch.object(
                self.node,'_door_angular',side_effect=AssertionError('tracking overlay entered active pivot')),patch(
                'smart_wheelchair_safety.unified_control_node.braking_clear',return_value=True):
            self.node.control()
        np.testing.assert_allclose(self.node.output,[0.,.01],atol=1e-9)
        np.testing.assert_allclose(self.node.acceleration,[0.,.2],atol=1e-9)

    def _door_angular_test_tick(self, target_w, dt=.05, mode='door_clear', guard=True):
        self.node.use_sim_time=True;self.node.mode=mode;self.node.door_phase=mode
        self.node.mode_neutral_seen=True;self.node.door_path_feasible=True
        self.node.door_path=np.array([[0.,0.,0.],[3.,0.,0.]])
        self.node.raw=np.array([.8,0.]);self.node.last_tick=100.-dt;self.node.odom_stamp=100.
        self.node.plan_ros_time=100.
        with patch.object(rospy.Time,'now',return_value=rospy.Time.from_sec(100.)),patch.object(
                self.node,'fresh',return_value=True),patch.object(self.node,'_near_recovery',return_value=False),patch.object(
                self.node,'_pending_door_rotation',return_value=None),patch.object(self.node,'_safe_door_command',return_value=np.array([.15,target_w])),patch.object(
                self.node,'_door_near_entry',return_value=True),patch.object(self.node,'_door_angular',return_value=target_w),patch(
                'smart_wheelchair_safety.unified_control_node.braking_clear',return_value=guard):
            self.node.control()

    def test_recorded_door_clear_angular_targets_preserve_actual_jerk(self):
        data=json.loads((Path(__file__).parent/'fixtures/door_clear_angular_v16.json').read_text())
        self.assertEqual(len(data),4)
        for case in data:
            with self.subTest(case=case['case']):
                rows=case['rows'];self.node.output=np.array([rows[1]['v'],rows[1]['w']])
                self.node.acceleration=np.array([0.,(rows[1]['w']-rows[0]['w'])/rows[1]['control_dt_s']])
                previous_a=self.node.acceleration[1]
                for row in rows[2:]:
                    old_w=self.node.output[1];dt=row['control_dt_s']
                    self._door_angular_test_tick(row['w'],dt)
                    actual_a=(self.node.output[1]-old_w)/dt
                    self.assertLessEqual(abs(actual_a),.8+1e-8,case['case'])
                    self.assertLessEqual(abs(actual_a-previous_a)/dt,4.+1e-7,case['case'])
                    previous_a=actual_a
                self.assertAlmostEqual(self.node.acceleration[1],previous_a)

    def test_door_clear_entry_uses_actual_previous_acceleration_once(self):
        self.node.output=np.array([.15,.03]);self.node.acceleration[:]=0.
        before=self.node.output.copy()
        first,first_a=self.node._assisted_slew(np.array([.15,.031]),.05)
        self._door_angular_test_tick(.031,mode='door_pass')
        self.assertAlmostEqual(self.node.output[1],first[1])
        self.assertAlmostEqual(self.node.acceleration[1],first_a[1])
        expected,expected_a=self.node._assisted_slew(np.array([.15,.025]),.05)
        self._door_angular_test_tick(.025)
        self.assertAlmostEqual(self.node.output[1],expected[1])
        self.assertAlmostEqual(self.node.acceleration[1],expected_a[1])

    def test_all_checked_tracking_phases_keep_actual_angular_jerk(self):
        self.node.output=np.array([.15,.02]);self.node.acceleration[:]=0.
        previous_a=0.
        for mode in ('door_align','door_pass','door_clear'):
            for target in (.03,.04,.02):
                old=self.node.output[1]
                self._door_angular_test_tick(target,mode=mode)
                a=(self.node.output[1]-old)/.05
                self.assertLessEqual(abs(a),.8+1e-8)
                self.assertLessEqual(abs(a-previous_a)/.05,4.+1e-7)
                self.assertAlmostEqual(self.node.acceleration[1],a)
                previous_a=a

    def test_door_clear_smoothing_keeps_final_guard_priority(self):
        self.node.output=np.array([.15,.03]);self.node.acceleration[:]=0.
        self._door_angular_test_tick(.02,guard=False)
        self.assertEqual(self.node.reason,'emergency_stop')
        np.testing.assert_array_equal(self.node.output,[0.,0.])

    def test_checked_tracking_target_jump_cannot_create_excessive_derivative(self):
        self.node.output=np.array([.15,.03]);self.node.acceleration[:]=0.
        self._door_angular_test_tick(.15,mode='door_pass')
        self.assertAlmostEqual(self.node.output[1],.04)
        self.assertAlmostEqual(self.node.acceleration[1],.2)

    def _door_translation_test_tick(self, target, dt):
        self.node.use_sim_time=True;self.node.mode='door_align';self.node.door_phase='door_align'
        self.node.mode_neutral_seen=True;self.node.door_path_feasible=True
        self.node.door_path=np.array([[0.,0.,0.],[3.,0.,0.]])
        self.node.raw=np.array([.8,0.]);self.node.last_tick=100.-dt;self.node.odom_stamp=100.
        with patch.object(rospy.Time,'now',return_value=rospy.Time.from_sec(100.)),patch.object(
                self.node,'fresh',return_value=True),patch.object(self.node,'_near_recovery',return_value=False),patch.object(
                self.node,'_pending_door_rotation',return_value=None),patch.object(self.node,'_safe_door_command',return_value=np.array(target)),patch.object(
                self.node,'_door_near_entry',return_value=True),patch.object(self.node,'_door_angular',side_effect=lambda v:.05*v),patch(
                'smart_wheelchair_safety.unified_control_node.braking_clear',return_value=True):
            self.node.control()

    def test_door_linear_stage_targets_and_zero_stop_preserve_actual_jerk(self):
        self.node.output[:]=0.;self.node.acceleration[:]=0.;previous_a=0.
        for target in (.35,.15,0.):
            for tick in range(90):
                dt=(.05,.13,.02,.001)[tick%4]
                old=self.node.output[0]
                self._door_translation_test_tick([target,.05*target],dt)
                a=(self.node.output[0]-old)/dt
                self.assertLessEqual(abs(a),.5+1e-8)
                self.assertLessEqual(abs(a-previous_a)/dt,2.5+1e-7)
                self.assertGreaterEqual(self.node.output[0],0.)
                self.assertLessEqual(self.node.output[0],self.node.parameter('max_speed'))
                previous_a=a
            self.assertAlmostEqual(self.node.output[0],target,places=5)
        np.testing.assert_array_equal(self.node.output,[0.,0.])

    def test_recorded_short_staging_distance_reaches_pivot_without_linear_snap(self):
        from smart_wheelchair_safety import unified_control_node as module
        stage=.111673379155862
        self.node.use_sim_time=True;self.node.mode='door_align';self.node.door_phase='door_align'
        self.node.mode_neutral_seen=True;self.node.door_path_feasible=True
        self.node.door=module.Opening((2.,0.),0.,1.)
        self.node.door_path=np.array([[0.,0.,0.],[stage,0.,0.],[stage,0.,math.pi/2],[stage,3.,math.pi/2]])
        self.node.door_rotation_index=0;self.node.pose=(0.,0.,0.)
        self.node.output[:]=0.;self.node.acceleration[:]=0.;self.node.measured[:]=0.
        self.node.raw=np.array([.8,0.]);previous_a=0.;rotation_started=False
        self.node.plan_ros_time=100.
        for tick in range(220):
            dt=(.05,.13,.02,.001)[tick%4]
            self.node.last_tick=100.-dt;self.node.odom_stamp=100.
            old=self.node.output[0]
            with patch.object(rospy.Time,'now',return_value=rospy.Time.from_sec(100.)),patch.object(
                    self.node,'fresh',return_value=True),patch.object(self.node,'_near_recovery',return_value=False),patch.object(
                    self.node,'points',return_value=([],np.array([[5.,2.],[5.,-2.]]))):
                self.node.control()
            self.assertEqual(self.node.reason,'clear')
            a=(self.node.output[0]-old)/dt
            self.assertLessEqual(abs(a),.5+1e-8)
            self.assertLessEqual(abs(a-previous_a)/dt,2.5+1e-7,(tick,old,self.node.output,a,previous_a))
            previous_a=a
            x,y,yaw=self.node.pose;v,w=self.node.output
            self.node.pose=(x+v*math.cos(yaw)*dt,y+v*math.sin(yaw)*dt,yaw+w*dt)
            self.node.measured=self.node.output.copy()
            rotation_started |= abs(w)>.05
            if rotation_started and v==0. and abs(a)<1e-9:
                break
        self.assertTrue(rotation_started,'checked pivot must take over')
        self.assertLess(np.linalg.norm(np.array(self.node.pose[:2])-[stage,0.]),.04)
        self.assertEqual(self.node.output[0],0.)

    def test_recorded_door_translation_target_changes_preserve_actual_linear_jerk(self):
        data=json.loads((Path(__file__).parent/'fixtures/door_translation_clamp_v16.json').read_text())
        self.assertEqual(len(data),3)
        for row in data:
            with self.subTest(case=row['case']):
                self.node.output=np.array(row['output']);self.node.acceleration=np.array(row['acceleration'])
                before=self.node.output.copy();old_a=self.node.acceleration[0]
                self._door_translation_test_tick(row['target'],row['dt'])
                actual_a=(self.node.output[0]-before[0])/row['dt']
                self.assertLessEqual(abs(actual_a-old_a)/row['dt'],2.5+1e-8)
                self.assertAlmostEqual(self.node.acceleration[0],actual_a)
                angular_a=(self.node.output[1]-before[1])/row['dt']
                self.assertLessEqual(abs(angular_a),.8+1e-8)
                self.assertLessEqual(abs(angular_a-row['acceleration'][1])/row['dt'],4.+1e-7)

    def test_recorded_capture_rejects_unsafe_complete_wait_process(self):
        data=json.loads((Path(__file__).parent/'fixtures/wait_capture_v15_2.json').read_text())
        self.node.use_sim_time=True
        self.node.output=np.array(data['output']);self.node.measured=np.array(data['measured'])
        self.node.acceleration=np.array(data['acceleration'])
        self.node.last_tick=data['ros'];self.node.odom_stamp=data['odom_stamp']
        old=(self.node.output.copy(),self.node.acceleration.copy(),self.node.last_tick)
        with patch.object(rospy.Time,'now',return_value=rospy.Time.from_sec(data['ros'])):
            self.assertFalse(self.node._door_capture_is_clear(np.array(data['points'])))
        np.testing.assert_array_equal(self.node.output,old[0])
        np.testing.assert_array_equal(self.node.acceleration,old[1])
        self.assertEqual(self.node.last_tick,old[2])

    def test_complete_wait_admission_is_bounded_and_rejects_nonfinite_state(self):
        self.node.output=np.array([.2,.1]);self.node.acceleration[:]=0.
        self.node.odom_stamp=100.;self.node.last_tick=99.95
        with patch.object(rospy.Time,'now',return_value=rospy.Time.from_sec(100.)),patch.object(
                self.node,'_door_wait_slew',return_value=(self.node.output.copy(),np.zeros(2))) as slew,patch(
                'smart_wheelchair_safety.unified_control_node.braking_clear',return_value=True):
            self.assertFalse(self.node._door_capture_is_clear(np.empty((0,2))))
            self.assertEqual(slew.call_count,80)
        self.node.acceleration[0]=float('nan')
        with patch.object(rospy.Time,'now',return_value=rospy.Time.from_sec(100.)):
            self.assertFalse(self.node._door_capture_is_clear(np.empty((0,2))))

    def test_complete_wait_admission_accepts_synthetic_open_corridor_and_stops(self):
        for points in (np.empty((0,2)),np.array([[x,y] for x in np.linspace(-2.,4.,50) for y in (-1.5,1.5)])):
            self.node.use_sim_time=True;self.node.last_tick=100.;self.node.odom_stamp=100.
            self.node.output=np.array([.35,-.18]);self.node.measured=self.node.output.copy()
            self.node.acceleration=np.array([.22,.08])
            with patch.object(rospy.Time,'now',return_value=rospy.Time.from_sec(100.)),patch.object(
                    self.node,'_door_wait_slew',wraps=self.node._door_wait_slew) as slew:
                self.assertTrue(self.node._door_capture_is_clear(points))
                self.assertGreater(slew.call_count,2,'must inspect full wait process, not only first tick')

    def test_sim_actual_dt_preserves_physical_jerk_after_delayed_tick(self):
        self.node.use_sim_time = True
        previous_a = np.zeros(2)
        for dt in [.05]*4 + [.13, .02, .05]:
            old = self.node.output.copy()
            self.node.plan_ros_time = 100.
            with patch.object(self.node, 'fresh', return_value=True):
                self._assisted_slew_tick([.8, .65], dt)
            actual_a = (self.node.output-old)/dt
            self.assertTrue(np.all(np.abs(actual_a-previous_a) <= np.array([2.5,4.])*dt+1e-8),
                            (dt, actual_a, previous_a))
            np.testing.assert_allclose(self.node.acceleration, actual_a, atol=1e-9)
            previous_a = actual_a

    def test_sim_wait_and_pivot_use_actual_delayed_and_short_intervals(self):
        for pivot in (False,True):
            self.node.use_sim_time=True
            if pivot:
                self.setup_recorded_pivot_residual()
            else:
                self.node.mode='door_wait';self.node.mode_neutral_seen=True
                self.node.output[:]=[.3,.1];self.node.acceleration[:]=0.
            previous_a=self.node.acceleration.copy()
            for dt in (.13,.02,.05,.001):
                old=self.node.output.copy()
                self.node.last_tick=100.-dt;self.node.odom_stamp=100.
                self.node.measured=old.copy()
                with patch.object(rospy.Time,'now',return_value=rospy.Time.from_sec(100.)), patch.object(
                        self.node,'fresh',return_value=True), patch.object(self.node,'_near_recovery',return_value=False):
                    self.node.control()
                actual_a=(self.node.output-old)/dt
                self.assertTrue(np.all(np.abs(actual_a)<=[.5+1e-8,.8+1e-8]))
                self.assertTrue(np.all(np.abs(actual_a-previous_a)<=np.array([2.5,4.])*dt+1e-8),
                                (pivot,dt,actual_a,previous_a))
                np.testing.assert_allclose(self.node.acceleration,actual_a,atol=1e-8)
                previous_a=actual_a

    def test_sim_actual_dt_still_yields_to_hard_guard(self):
        self.node.use_sim_time=True;self.node.mode_neutral_seen=True
        self.node.output[:]=[.3,.1];self.node.acceleration[:]=0.
        self.node.plan_ros_time=100.;self.node.last_tick=99.87;self.node.odom_stamp=100.
        with patch.object(rospy.Time,'now',return_value=rospy.Time.from_sec(100.)), patch.object(
                self.node,'fresh',return_value=True),patch.object(self.node,'_near_recovery',return_value=False),patch(
                'smart_wheelchair_safety.unified_control_node.braking_clear',return_value=False):
            self.node.control()
        np.testing.assert_array_equal(self.node.output,[0.,0.])
        self.assertEqual(self.node.reason,'emergency_stop')

    def test_sim_capture_preview_uses_actual_dt_and_same_tick_prediction(self):
        self.node.use_sim_time = True
        self.node.output[:] = [.1, .1]
        for elapsed, expected in ((.13,.13),(.02,.02),(0.,.05)):
            self.node.last_tick = 100.-elapsed
            old_tick = self.node.last_tick
            with patch.object(rospy.Time,'now',return_value=rospy.Time.from_sec(100.)), patch.object(
                    self.node, '_door_wait_slew', wraps=self.node._door_wait_slew) as slew, patch(
                    'smart_wheelchair_safety.unified_control_node.braking_clear',return_value=True):
                self.assertTrue(self.node._door_capture_is_clear(np.empty((0,2))))
            self.assertAlmostEqual(slew.call_args_list[0].args[0],expected)
            self.assertEqual(self.node.last_tick,old_tick)
            self.assertFalse(self.node.clock_stalled)

    def test_sim_invalid_control_interval_stops_revokes_and_requires_new_input(self):
        from geometry_msgs.msg import Twist
        self.node.use_sim_time = True
        self.node.assist_enabled = False
        self.node.mode_neutral_seen = True
        statuses=[]
        self.node.status=SimpleNamespace(publish=statuses.append)
        for elapsed in (0.,-.01,.0002,.25,.4):
            self.node.clock_stalled=False
            self.node.output[:]=[.5,.1]
            self.node.raw[:]=[.5,0.]
            self.node.raw_time=100.
            self.node.last_tick=10.-elapsed
            self.node.active_reference_stamp=rospy.Time.from_sec(9.9)
            with patch.object(rospy.Time,'now',return_value=rospy.Time.from_sec(10.)), patch(
                    'smart_wheelchair_safety.unified_control_node.time.monotonic',return_value=100.1):
                self.node.control()
            np.testing.assert_array_equal(self.node.output,[0.,0.])
            self.assertIsNone(self.node.active_reference_stamp)
            self.assertTrue(self.node.clock_stalled)
            self.assertEqual(json.loads(statuses[-1].data)['clock_fault'],'control_interval')
            self.assertAlmostEqual(json.loads(statuses[-1].data)['elapsed_s'],elapsed)
            self.node._clock_watchdog_tick(100.12,10.02)
            self.assertFalse(self.node.clock_stalled)
            self.assertEqual(self.node.raw_time,0.)
            msg=Twist();msg.linear.x=.5
            self.node.on_raw(msg,100.13)
            self.assertEqual(self.node.raw_time,100.13)

    def test_sim_invalid_capture_preview_does_not_revoke_or_integrate(self):
        self.node.use_sim_time=True
        for elapsed in (-.01,.0002,.25):
            self.node.last_tick=100.-elapsed
            with patch.object(rospy.Time,'now',return_value=rospy.Time.from_sec(100.)), patch.object(
                    self.node,'_door_wait_slew', return_value=(np.zeros(2),np.zeros(2))) as slew:
                self.assertFalse(self.node._door_capture_is_clear(np.empty((0,2))))
                slew.assert_not_called()
            self.assertFalse(self.node.clock_stalled)

    def test_status_exposes_actual_control_stamp_and_clamped_dt_in_both_paths(self):
        stamp = rospy.Time.from_sec(100.)
        for assist in (True, False):
            for elapsed, expected_dt in ((.05, .05), (.0002, .001), (.2, .1)):
                with self.subTest(assist=assist, elapsed=elapsed):
                    self.node.assist_enabled = assist
                    self.node.mode_neutral_seen = True
                    self.node.raw = np.zeros(2)
                    self.node.raw_time = time.monotonic()
                    self.node.last_tick = 100.-elapsed
                    statuses = []
                    self.node.status = SimpleNamespace(publish=statuses.append)
                    with patch.object(rospy.Time, 'now', return_value=stamp), patch.object(self.node, 'fresh', return_value=True):
                        self.node.control()
                    status = json.loads(statuses[-1].data)
                    self.assertEqual(status['control_stamp_s'], 100.)
                    self.assertAlmostEqual(status['control_dt_s'], expected_dt)
                    np.testing.assert_array_equal(self.node.output, [0., 0.])

    def _assisted_slew_tick(self, target, dt=.05):
        self.node.mode = 'wall'
        self.node.use_neupan = False
        self.node.raw = np.array([.8, 0.])
        self.node.plan_time = time.monotonic()
        self.node.planned = np.asarray(target, dtype=float)
        stamp = rospy.Time.from_sec(100.)
        self.node.odom_stamp = stamp.to_sec()
        self.node.raw_time = self.node.odom_time = time.monotonic()
        self.node.last_tick = stamp.to_sec()-dt
        self.node.scans = {side: (time.monotonic(), np.array([[3., 2.], [3., -2.]]))
                           for side in ('left', 'right')}
        with patch.object(rospy.Time, 'now', return_value=stamp), patch.object(
                self.node, '_safe_assisted_command', return_value=np.asarray(target, dtype=float)), patch(
                'smart_wheelchair_safety.unified_control_node.braking_clear', return_value=True):
            self.node.control()
        return self.node.output.copy()

    def test_assisted_real_target_change_preserves_actual_acceleration(self):
        rows = json.loads((Path(__file__).parent/'fixtures/assisted_target_trace.json').read_text())
        for when in (18.44, 20.14):
            row = next(row for row in rows if abs(row['t']-when)<1e-6)
            self.node.output = np.array(row['before']['output'])
            self.node.acceleration = np.array(row['before']['acceleration'])
            previous = self.node.output.copy()
            old_a = self.node.acceleration.copy()
            command = self._assisted_slew_tick(row['target'], row['dt'])
            actual_a = (command-previous)/row['dt']
            np.testing.assert_allclose(self.node.acceleration, actual_a, atol=1e-8)
            self.assertTrue(np.all(abs(actual_a-old_a) <= np.array([2.5,4.])*row['dt']+1e-8))

    def test_assisted_recorded_targets_continuous_actual_derivatives(self):
        rows = json.loads((Path(__file__).parent/'fixtures/assisted_target_trace.json').read_text())
        self.node.output = np.zeros(2)
        self.node.acceleration = np.zeros(2)
        actual_a = np.zeros(2)
        for row in rows:
            dt = float(np.clip(row['dt'], .001, .1))
            previous = self.node.output.copy()
            command = self._assisted_slew_tick(row['target'], dt)
            new_a = (command-previous)/dt
            with self.subTest(t=row['t']):
                self.assertTrue(np.all(abs(new_a) <= np.array([.5,.8])+1e-7))
                self.assertTrue(np.all(abs(new_a-actual_a) <= np.array([2.5,4.])*dt+1e-7))
                np.testing.assert_allclose(self.node.acceleration, new_a, atol=1e-7)
                self.assertTrue(0. <= command[0] <= .8 and abs(command[1]) <= .65)
            actual_a = new_a

    def test_assisted_physical_bounds_and_changed_targets_keep_actual_jerk(self):
        for dt in (.025, .05, .1):
            with self.subTest(dt=dt):
                self.node.output = np.zeros(2)
                self.node.acceleration = np.zeros(2)
                actual_a = np.zeros(2)
                for target in ([.8,.65], [.8,-.65], [.2,.1], [0.,0.]):
                    for _ in range(int(5/dt)):
                        previous = self.node.output.copy()
                        command = self._assisted_slew_tick(target, dt)
                        new_a = (command-previous)/dt
                        self.assertTrue(np.all(abs(new_a) <= np.array([.5,.8])+1e-7))
                        self.assertTrue(np.all(abs(new_a-actual_a) <= np.array([2.5,4.])*dt+1e-7))
                        self.assertTrue(0. <= command[0] <= .8 and abs(command[1]) <= .65)
                        actual_a = new_a
                    np.testing.assert_allclose(command, target, atol=1e-7)

    def test_assisted_variable_ticks_near_bounds_and_final_stationary_step(self):
        actual_a = np.zeros(2)
        ticks = [.1, .001, .025, .05]
        for target in ([.8,.65], [0.,-.65], [.2,.1], [0.,0.]):
            for i in range(240):
                dt = ticks[i % len(ticks)]
                old = self.node.output.copy()
                command = self._assisted_slew_tick(target, dt)
                new_a = (command-old)/dt
                self.assertTrue(np.all(abs(new_a-actual_a) <= np.array([2.5,4.])*dt+1e-7),
                                (target,i,dt,old,command,actual_a,new_a))
                self.assertTrue(0. <= command[0] <= .8 and abs(command[1]) <= .65)
                actual_a = new_a
            np.testing.assert_allclose(command,target,atol=1e-7)

    def test_assisted_infeasible_external_state_keeps_physical_bounds_and_true_acceleration(self):
        self.node.output = np.array([.799, .649])
        self.node.acceleration = np.array([.5, .8])
        old = self.node.output.copy()
        command = self._assisted_slew_tick([.8,.65])
        self.assertLessEqual(command[0], .8)
        self.assertLessEqual(command[1], .65)
        np.testing.assert_allclose(self.node.acceleration, (command-old)/.05, atol=1e-8)

    def test_turn_away_cancel_releases_mode_and_old_plan_before_next_reference(self):
        from geometry_msgs.msg import Twist, TwistStamped
        from smart_wheelchair_safety import unified_control_node as module
        for source in ('neupan', 'local_follower'):
            for mirror in (-1., 1.):
                self.node.mode=self.node.door_phase='door_align'
                self.node.mode_neutral_seen=True
                self.node.door=module.Opening((2.,0.),0.,1.)
                self.node.door_path=np.array([[0.,0.,0.],[2.,0.,0.]])
                self.node.door_path_feasible=True
                self.node.door_away_since=time.monotonic()-.4
                self.node.override=False;self.node.wall_side=0
                self.node.output=np.array([.00625, mirror*-.09504501131071233])
                self.node.acceleration=np.array([.125, mirror*.38868458261985184])
                self.node.active_reference_stamp=rospy.Time.from_sec(99.9)
                self.node.accept_planned=True;self.node.use_neupan=True
                old=TwistStamped();old.header.stamp=self.node.active_reference_stamp
                old.twist.linear.x=.3;old.twist.angular.z=mirror*-.43115711212158203
                with patch.object(rospy.Time,'now',return_value=rospy.Time.from_sec(100.)):
                    self.node.on_planned(old);self.node.on_neupan(old)
                    self.assertTrue(self.node.neupan_fresh())
                message=Twist();message.linear.x=.8;message.angular.z=mirror*.63
                with patch.object(module,'active_aperture_targeted',return_value=False):
                    self.node.on_raw(message)
                self.assertIsNone(self.node.door)
                self.assertEqual(self.node.mode,'manual')
                self.assertIsNone(self.node.active_reference_stamp)
                self.assertFalse(self.node.accept_planned)
                np.testing.assert_array_equal(self.node.neupan,[0.,0.])
                np.testing.assert_array_equal(self.node.planned,[0.,0.])
                statuses=[];self.node.status=SimpleNamespace(publish=statuses.append)
                previous_a=self.node.acceleration.copy()
                def tick(expected_source):
                    nonlocal previous_a
                    self.node.last_tick=99.95;self.node.odom_stamp=100.
                    old_output=self.node.output.copy()
                    with patch.object(rospy.Time,'now',return_value=rospy.Time.from_sec(100.)),patch.object(self.node,'fresh',return_value=True),patch.object(self.node,'_near_recovery',return_value=False),patch.object(self.node,'_safe_assisted_command',side_effect=lambda desired,points:desired) as safe,patch.object(module,'braking_clear',return_value=True):
                        self.node.control()
                    safe.assert_called_once()
                    actual_a=(self.node.output-old_output)/.05
                    self.assertTrue(np.all(abs(actual_a-previous_a)<=module.JERK_LIMITS*.05+1e-8))
                    np.testing.assert_allclose(self.node.acceleration,actual_a,atol=1e-8)
                    self.assertEqual(json.loads(statuses[-1].data)['planner_source'],expected_source)
                    previous_a=actual_a
                tick('joystick')
                # A following heartbeat releases override, but the cancelled
                # reference's in-flight messages cannot take ownership back.
                with patch.object(self.node,'_intended_door',return_value=(None,None)):
                    self.node.on_raw(message)
                with patch.object(rospy.Time,'now',return_value=rospy.Time.from_sec(100.)):
                    self.node.on_planned(old);self.node.on_neupan(old)
                    self.assertFalse(self.node.neupan_fresh())
                self.assertEqual(self.node.plan_time,0.)
                tick('joystick')
                self.node.active_reference_stamp=rospy.Time.from_sec(100.)
                self.node.accept_planned=True
                new=TwistStamped();new.header.stamp=self.node.active_reference_stamp
                new.twist.linear.x=.3;new.twist.angular.z=mirror*-.43
                with patch.object(rospy.Time,'now',return_value=rospy.Time.from_sec(100.)):
                    if source=='neupan':self.node.on_neupan(new)
                    else:self.node.on_planned(new)
                tick(source)

    def test_cancelled_door_neutral_and_reverse_keep_priority(self):
        from geometry_msgs.msg import Twist
        from smart_wheelchair_safety import unified_control_node as module
        for linear in (0.,-.2):
            self.node.mode='door_align';self.node.mode_neutral_seen=True
            self.node.door=module.Opening((2.,0.),0.,1.)
            self.node.door_away_since=time.monotonic()-.4
            positive=Twist();positive.linear.x=.8;positive.angular.z=.63
            with patch.object(module,'active_aperture_targeted',return_value=False):self.node.on_raw(positive)
            neutral_or_reverse=Twist();neutral_or_reverse.linear.x=linear
            self.node.on_raw(neutral_or_reverse)
            self.node.last_tick=99.95;self.node.odom_stamp=100.
            with patch.object(rospy.Time,'now',return_value=rospy.Time.from_sec(100.)),patch.object(self.node,'fresh',return_value=True),patch.object(self.node,'_near_recovery',return_value=False),patch.object(self.node,'_safe_assisted_command') as safe,patch.object(module,'braking_clear',return_value=True):self.node.control()
            safe.assert_not_called()
            self.assertIsNone(self.node.door)
            self.assertIsNone(self.node.active_reference_stamp)
            self.assertEqual(self.node.override_handoff_time,0.)
            if linear==0.:np.testing.assert_array_equal(self.node.output,[0.,0.])

    def test_pre_resume_cancel_motion_is_stale_but_still_cancels(self):
        from geometry_msgs.msg import Twist
        from smart_wheelchair_safety import unified_control_node as module
        for v,w in ((-.5,0.),(0.,.5),(.01,0.),(float('nan'),0.)):
            with self.subTest(v=v,w=w):
                self.node.use_sim_time=True
                self.node.assist_enabled=True;self.node.mode_neutral_seen=True
                self.node.clock_stalled=False
                self.node.clock_last_ros=10.;self.node.clock_last_advance_wall=100.
                self.node.latest_raw=self.node.latest_raw_cancel=self.node.consumed_raw=None
                self.node._clock_watchdog_tick(100.251,10.)
                self.node.door=module.Opening((2.,0.),0.,1.)
                msg=Twist();msg.linear.x=v;msg.angular.z=w
                with patch.object(module.time,'monotonic',return_value=100.29):self.node._receive_raw(msg)
                self.node._clock_watchdog_tick(100.32,10.02)
                self.node._consume_raw()
                self.assertIsNone(self.node.door)
                self.assertEqual(self.node.raw_time,0.)
                self.node.assist_enabled=False
                with patch.object(module.time,'monotonic',return_value=100.33):self.node.control()
                np.testing.assert_array_equal(self.node.output,[0.,0.])

    def test_pre_resume_nonzero_mailbox_cannot_restart_direct_control(self):
        from geometry_msgs.msg import Twist
        module='smart_wheelchair_safety.unified_control_node'
        self.node.use_sim_time=True;self.node.assist_enabled=False
        self.node.mode_neutral_seen=True
        self.node.clock_last_ros=10.;self.node.clock_last_advance_wall=100.
        self.node.last_tick=10.
        self.node.output=np.array([.5,0.])
        self.node._clock_watchdog_tick(100.251,10.)
        msg=Twist();msg.linear.x=.5
        with patch(module+'.time.monotonic',return_value=100.29):self.node._receive_raw(msg)
        self.node._clock_watchdog_tick(100.32,10.02)
        self.node._consume_raw()
        self.assertEqual(self.node.raw_time,0.)
        with patch(module+'.time.monotonic',return_value=100.33),patch.object(rospy.Time,'now',return_value=rospy.Time.from_sec(10.03)):
            self.node.control()
        np.testing.assert_array_equal(self.node.output,[0.,0.])
        with patch(module+'.time.monotonic',return_value=100.34):self.node._receive_raw(msg)
        self.node._consume_raw()
        with patch(module+'.time.monotonic',return_value=100.35),patch.object(rospy.Time,'now',return_value=rospy.Time.from_sec(10.04)):
            self.node.control()
        np.testing.assert_array_equal(self.node.output,[.5,0.])

    def test_resume_cutoff_keeps_old_cancel_and_concurrent_new_forward(self):
        from geometry_msgs.msg import Twist
        from smart_wheelchair_safety import unified_control_node as module
        self.node.use_sim_time=True;self.node.clock_last_ros=10.;self.node.clock_last_advance_wall=100.
        self.node.door=module.Opening((2.,0.),0.,1.)
        self.node._clock_watchdog_tick(100.251,10.)
        neutral=Twist();forward=Twist();forward.linear.x=.5
        with patch.object(module.time,'monotonic',return_value=100.28):self.node._receive_raw(neutral)
        with patch.object(module.time,'monotonic',return_value=100.29):self.node._receive_raw(forward)
        self.node._clock_watchdog_tick(100.32,10.02)
        original=self.node.on_raw
        def arrive_after_snapshot(message,received=None):
            result=original(message,received)
            if message is neutral:
                with patch.object(module.time,'monotonic',return_value=100.35):self.node._receive_raw(forward)
            return result
        with patch.object(self.node,'on_raw',side_effect=arrive_after_snapshot):self.node._consume_raw()
        self.assertIsNone(self.node.door,'pre-resume neutral must still cancel retained intent')
        self.assertLessEqual(self.node.raw[0],.02)
        self.node._consume_raw()
        self.assertEqual(self.node.raw[0],.5)
        self.assertEqual(self.node.raw_time,100.35)

    def test_sim_watchdog_wall_thread_runs_without_ros_timer_and_joins(self):
        from smart_wheelchair_safety.unified_control_node import UnifiedControlNode
        self.assertIsNone(self.node.clock_watchdog_thread)
        def params(name,default=None):
            return True if name=='/use_sim_time' else default
        with patch.object(rospy,'get_param',side_effect=params),patch.object(rospy,'Timer'),patch.object(rospy.Time,'now',return_value=rospy.Time.from_sec(100.)):
            node=UnifiedControlNode()
            stopped=threading.Event()
            node.pub=SimpleNamespace(publish=lambda message:stopped.set())
            try:
                node.clock_last_ros=100.
                node.clock_last_advance_wall=time.monotonic()-.251
                self.assertTrue(stopped.wait(.3),'independent wall thread must stop without any ROS Timer callback')
                self.assertTrue(node.clock_stalled)
            finally:
                node.destroy_node()
            self.assertFalse(node.clock_watchdog_thread.is_alive())

    def test_sim_wall_watchdog_stops_without_ros_timer_and_requires_new_input(self):
        from geometry_msgs.msg import Twist
        self.node.use_sim_time=True
        self.node.clock_last_ros=0.;self.node.clock_last_advance_wall=100.
        self.node.output=np.array([.4,.2]);self.node.raw=np.array([.8,0.])
        self.node.raw_time=100.;self.node.accept_planned=True
        self.node.active_reference_stamp=rospy.Time.from_sec(10.)
        self.node._clock_watchdog_tick(100.,0.)
        self.assertTrue(self.node.accept_planned,'startup without clock does not mutate state')
        self.node._clock_watchdog_tick(100.,10.)
        epoch=self.node.epoch
        self.node._clock_watchdog_tick(100.249,10.)
        self.assertTrue(self.node.accept_planned)
        self.node._clock_watchdog_tick(100.251,10.)
        self.assertTrue(self.node.clock_stalled)
        self.assertEqual(self.node.epoch,epoch+1)
        self.assertFalse(self.node.accept_planned)
        self.assertIsNone(self.node.active_reference_stamp)
        np.testing.assert_array_equal(self.node.output,[0.,0.])
        count=len(self.commands)
        self.node._clock_watchdog_tick(100.301,10.)
        self.assertEqual(self.node.epoch,epoch+1)
        self.assertGreater(len(self.commands),count,'each stalled check still publishes stop')
        # Frozen-clock heartbeats do not authorize motion on clock resumption.
        message=Twist();message.linear.x=.8
        self.node.on_raw(message)
        self.node._clock_watchdog_tick(100.32,10.02)
        self.assertFalse(self.node.clock_stalled)
        self.assertEqual(self.node.raw_time,0.)
        self.node.on_raw(message)
        self.assertGreater(self.node.raw_time,0.)
        self.assertFalse(self.node.accept_planned,'only a new reference can rearm the planner')
        self.node.use_sim_time=False
        self.node.output=np.array([.2,0.]);self.node._clock_watchdog_tick(101.,10.02)
        np.testing.assert_array_equal(self.node.output,[.2,0.])

    def test_sim_wall_watchdog_resets_on_progress_not_sample_receipt(self):
        self.node.use_sim_time=True
        self.node.clock_last_ros=10.;self.node.clock_last_advance_wall=100.
        self.node._clock_watchdog_tick(100.27,10.2)
        self.assertFalse(self.node.clock_stalled)
        self.assertEqual(self.node.clock_last_advance_wall,100.27)
        self.node._clock_watchdog_tick(100.52,10.2)
        self.assertTrue(self.node.clock_stalled)
        # Rewinding the clock revokes the old epoch immediately.
        self.node._clock_watchdog_tick(100.53,10.3)
        epoch=self.node.epoch
        self.node._clock_watchdog_tick(100.54,9.)
        self.assertTrue(self.node.clock_stalled)
        self.assertEqual(self.node.epoch,epoch+1)

    def test_sim_planner_mailbox_keeps_transport_ros_receipt_through_pause(self):
        from geometry_msgs.msg import TwistStamped
        module='smart_wheelchair_safety.unified_control_node'
        for source in ('planned','neupan'):
            for simulated,expected in ((True,True),(False,False)):
                self.node.cancel();self.node.use_sim_time=simulated;self.node.use_neupan=True
                self.node.active_reference_stamp=rospy.Time.from_sec(10.)
                self.node.accept_planned=True
                message=TwistStamped();message.header.stamp=self.node.active_reference_stamp
                message.twist.linear.x=.3
                with patch(module+'.time.monotonic',return_value=100.),patch.object(rospy.Time,'now',return_value=rospy.Time.from_sec(10.)):
                    getattr(self.node,'_receive_'+source)(message)
                with patch(module+'.time.monotonic',return_value=100.27),patch.object(rospy.Time,'now',return_value=rospy.Time.from_sec(10.2)):
                    self.node._consume_planner_actions()
                    accepted=getattr(self.node,'plan_time' if source=='planned' else 'neupan_time')==100.
                    self.assertEqual(accepted,expected)
                    if source=='neupan':self.assertEqual(self.node.neupan_fresh(),expected)
                if expected:
                    self.assertEqual(getattr(self.node,'plan_ros_time' if source=='planned' else 'neupan_ros_time'),10.)
                    # A new reference does not refresh a cached local action.
                    self.node.active_reference_stamp=rospy.Time.from_sec(10.24)
                    with patch(module+'.time.monotonic',return_value=100.3),patch.object(rospy.Time,'now',return_value=rospy.Time.from_sec(10.251)):
                        self.assertGreaterEqual(self.node._plan_age(),.25)
                        self.assertFalse(self.node.neupan_fresh())

    def test_neupan_fresh_action_uses_existing_guard(self):
        from geometry_msgs.msg import TwistStamped
        statuses = []
        self.node.status = SimpleNamespace(publish=statuses.append)
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
        self.assertEqual(json.loads(statuses[-1].data)['planner_source'], 'neupan')
        with patch('smart_wheelchair_safety.unified_control_node.braking_clear', return_value=False):
            self.node.control()
        self.assertEqual(self.commands[-1].linear.x, 0.)

    def test_checked_door_keeps_control_despite_valid_neupan_straight_or_opposing_turn(self):
        from geometry_msgs.msg import TwistStamped
        statuses = []
        self.node.status = SimpleNamespace(publish=statuses.append)
        self.node.mode = self.node.door_phase = 'door_pass'
        self.node.door_path_feasible = True
        self.node.door_path = np.array([[0., 0., 0.], [3., 0., 0.]])
        self.node.pose = (0., .1, 0.)
        self.node.door_rotation_index = 0
        stamp = rospy.Time.now()
        self.node.odom_stamp = stamp.to_sec()
        self.node.use_neupan = False
        self.node.last_tick = stamp.to_sec()-.05
        with patch.object(rospy.Time, 'now', return_value=stamp):
            self.node.control()
        checked = self.node.output.copy()
        self.assertGreater(checked[0], 0.)
        self.assertLess(checked[1], 0.)
        for angular in (0., .4):
            with self.subTest(neupan_angular=angular):
                self.node.output[:] = self.node.acceleration[:] = 0.
                self.node.use_neupan = self.node.accept_planned = True
                self.node.active_reference_stamp = stamp
                action = TwistStamped()
                action.header.stamp = stamp
                action.twist.linear.x, action.twist.angular.z = .55, angular
                self.node.on_neupan(action)
                self.assertTrue(self.node.neupan_fresh())
                self.node.last_tick = stamp.to_sec()-.05
                with patch.object(rospy.Time, 'now', return_value=stamp):
                    self.node.control()
                self.assertEqual(json.loads(statuses[-1].data)['planner_source'], 'door_follower')
                np.testing.assert_allclose(self.node.output, checked, atol=1e-10)

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

    def test_door_speed_rejects_acceleration_without_a_successor_stop(self):
        from smart_wheelchair_safety.unified_geometry import arc_path, braking_clear, transform_points
        points = np.column_stack((np.full(101, 1.017), np.linspace(-2., 2., 101)))
        self.node.mode = self.node.door_phase = 'door_pass'
        self.node.door_path_feasible = True
        self.node.door_path = np.array([[0., 0., 0.], [3., 0., 0.]])
        self.node.door_rotation_index = 0
        self.node.measured = np.zeros(2)
        old_candidate = np.array([.025, 0.])
        self.assertTrue(braking_clear(points, old_candidate, self.node.measured, state_age=.1))
        old_future = transform_points(points, arc_path(*old_candidate, duration=.05, steps=2)[-1],
                                      inverse=True)
        self.assertFalse(braking_clear(old_future, [0., 0.], old_candidate, state_age=.1))
        command = self.node._safe_door_command(points, self.node.odom_stamp)
        self.assertGreaterEqual(command[0], 0., 'insufficient planning clearance may require a stop')
        future = transform_points(points, arc_path(*command, duration=.05, steps=2)[-1], inverse=True)
        self.assertTrue(braking_clear(future, [0., 0.], command, state_age=.1))

    def test_assisted_speed_reserves_stop_without_changing_curvature(self):
        from smart_wheelchair_safety.unified_geometry import arc_path, braking_clear, transform_points
        points = np.column_stack((np.linspace(-2., 10., 161), np.full(161, .52)))
        desired = np.array([.8, .04])
        command = self.node._safe_assisted_command(desired, points)
        self.assertGreater(command[0], 0.)
        self.assertLess(command[0], desired[0])
        self.assertAlmostEqual(command[1]/command[0], desired[1]/desired[0])
        kwargs = dict(margin=self.node.parameter('hard_margin')+.01,
                      reaction=self.node.parameter('reaction_time'),
                      deceleration=self.node.parameter('braking_deceleration'),
                      state_age=self.node.parameter('odom_timeout'))
        self.assertTrue(braking_clear(points, command, self.node.measured, **kwargs))
        future = transform_points(points, arc_path(*command, duration=.05, steps=2)[-1], inverse=True)
        self.assertTrue(braking_clear(future, np.zeros(2), command, **kwargs))

    def test_wall_speed_planning_keeps_age_jitter_out_of_hard_guard(self):
        from smart_wheelchair_safety.unified_geometry import arc_path, transform_points
        obstacles = np.column_stack((np.linspace(-2., 10., 161), np.full(161, .52)))
        self.node.mode = 'wall'
        self.node.raw = self.node.planned = np.array([.8, 0.])
        self.node.pose = (0., 0., 0.)
        initial_stamp = rospy.Time.now().to_sec()
        previous_acceleration = np.zeros(2)
        for step in range(160):
            stamp = initial_stamp+.05*step
            now = time.monotonic()
            self.node.raw_time = self.node.odom_time = self.node.plan_time = now
            self.node.odom_stamp = stamp-(0., .04, .09)[step % 3]
            self.node.scans = {side: (now, obstacles) for side in ('left', 'right')}
            self.node.last_tick = stamp-.05
            previous = self.node.output.copy()
            with patch.object(rospy.Time, 'now', return_value=rospy.Time.from_sec(stamp)):
                self.node.control()
            self.assertEqual(self.node.reason, 'clear', (step, self.node.reason, self.node.output))
            self.assertGreater(self.node.output[0], 0.)
            actual_acceleration = (self.node.output-previous)/.05
            self.assertTrue(np.all(np.abs(actual_acceleration) <= [.500001, .800001]))
            self.assertTrue(np.all(np.abs(actual_acceleration-previous_acceleration) <= np.array([2.5, 4.])*.05+1e-6),
                            (step, actual_acceleration, previous_acceleration))
            previous_acceleration = actual_acceleration
            local = arc_path(*self.node.output, duration=.05, steps=2)[-1]
            xy = transform_points([local[:2]], self.node.pose)[0]
            self.node.pose = (*xy, self.node.pose[2]+local[2])
            self.node.measured = self.node.output.copy()
        self.assertGreater(self.node.pose[0], 1.)

    def test_assisted_wall_manual_front_transition_never_accelerates_outside_reserve(self):
        from smart_wheelchair_safety.unified_geometry import arc_path, transform_points
        obstacles = np.column_stack((np.linspace(-2., 10., 161), np.full(161, .52)))
        obstacles = np.vstack((obstacles, np.column_stack((np.full(161, 3.5), np.linspace(-2., 2., 161)))))
        self.node.mode = 'wall'
        self.node.raw = self.node.planned = np.array([.8, 0.])
        self.node.pose = (0., 0., 0.)
        initial_stamp = rospy.Time.now().to_sec()
        previous_acceleration = np.zeros(2)
        for step in range(240):
            self.node.mode = 'wall' if step < 20 else ('manual' if step < 80 else 'front_stop')
            stamp = initial_stamp+.05*step
            now = time.monotonic()
            self.node.raw_time = self.node.odom_time = self.node.plan_time = now
            self.node.odom_stamp = stamp-(0., .04, .09)[step % 3]
            self.node.scans = {side: (now, obstacles) for side in ('left', 'right')}
            self.node.last_tick = stamp-.05
            previous = self.node.output.copy()
            with patch.object(rospy.Time, 'now', return_value=rospy.Time.from_sec(stamp)):
                self.node.control()
            self.assertEqual(self.node.reason, 'clear', (step, self.node.reason, self.node.output))
            self.assertGreaterEqual(self.node.output[0], 0.)
            actual_acceleration = (self.node.output-previous)/.05
            self.assertTrue(np.all(np.abs(actual_acceleration) <= [.500001, .800001]))
            self.assertTrue(np.all(np.abs(actual_acceleration-previous_acceleration) <= np.array([2.5, 4.])*.05+1e-6),
                            (step, actual_acceleration, previous_acceleration))
            previous_acceleration = actual_acceleration
            local = arc_path(*self.node.output, duration=.05, steps=2)[-1]
            xy = transform_points([local[:2]], self.node.pose)[0]
            self.node.pose = (*xy, self.node.pose[2]+local[2])
            self.node.measured = self.node.output.copy()
        self.assertGreater(self.node.pose[0], 1.)

    def test_front_speed_planning_brakes_once_without_guard_reset(self):
        obstacles = np.column_stack((np.full(161, 3.5), np.linspace(-2., 2., 161)))
        self.node.mode = 'front_stop'
        self.node.raw = np.array([.8, 0.])
        self.node.pose = (0., 0., 0.)
        initial_stamp = rospy.Time.now().to_sec()
        braking = False
        previous_acceleration = 0.
        for step in range(240):
            stamp = initial_stamp+.05*step
            now = time.monotonic()
            self.node.raw_time = self.node.odom_time = self.node.plan_time = now
            self.node.odom_stamp = stamp-(0., .04, .09)[step % 3]
            self.node.scans = {side: (now, obstacles) for side in ('left', 'right')}
            self.node.last_tick = stamp-.05
            previous = self.node.output[0]
            with patch.object(rospy.Time, 'now', return_value=rospy.Time.from_sec(stamp)):
                self.node.control()
            self.assertEqual(self.node.reason, 'clear', (step, self.node.reason))
            braking |= self.node.output[0] < previous-1e-6
            if braking:
                self.assertLessEqual(self.node.output[0], previous+1e-6, (step, 'reaccelerated'))
            actual_acceleration = (self.node.output[0]-previous)/.05
            self.assertLessEqual(abs(actual_acceleration), .500001)
            self.assertLessEqual(abs(actual_acceleration-previous_acceleration), .125001,
                                 (step, actual_acceleration, previous_acceleration))
            previous_acceleration = actual_acceleration
            self.node.pose = (self.node.pose[0]+.05*self.node.output[0], 0., 0.)
            self.node.measured = self.node.output.copy()
        self.assertTrue(braking)
        self.assertLess(self.node.output[0], .02)
        self.assertGreater(self.node.pose[0], 1.)

    def test_narrow_door_acceleration_preserves_stop_margin_across_odom_age_jitter(self):
        from smart_wheelchair_safety.unified_geometry import arc_path, braking_clear, transform_points
        # One-metre aperture with 12 cm thick jambs; small post-pivot yaw error.
        side = np.r_[np.linspace(-2., -.5, 61), np.linspace(.5, 2., 61)]
        obstacles = np.vstack([np.column_stack((np.full(len(side), x), side))
                               for x in (1.04, 1.10, 1.16)])
        self.node.mode = self.node.door_phase = 'door_pass'
        self.node.door_path_feasible = True
        self.node.door_path = np.array([[0., 0., 0.], [3., 0., 0.]])
        self.node.door_rotation_index = 0
        self.node.pose = (0., 0., .02)
        self.node.raw = np.array([.6, 0.])
        self.node.measured = np.zeros(2)
        initial_stamp = rospy.Time.now().to_sec()
        for step in range(80):
            stamp = initial_stamp + .05*step
            age = .04 if step >= 20 and step % 2 else 0.
            now = time.monotonic()
            self.node.raw_time = self.node.odom_time = now
            self.node.odom_stamp = stamp-age
            self.node.scans = {side: (now, obstacles) for side in ('left', 'right')}
            self.node.last_tick = stamp-.05
            with patch.object(rospy.Time, 'now', return_value=rospy.Time.from_sec(stamp)):
                self.node.control()
            self.assertNotEqual(self.node.reason, 'emergency_stop',
                                (step, age, self.node.pose, self.node.measured.copy()))
            self.assertGreater(self.node.output[0], 0., (step, age, self.node.reason))
            self.node.measured = self.node.output.copy()
            local = transform_points(obstacles, self.node.pose, inverse=True)
            self.assertTrue(braking_clear(local, [0., 0.], self.node.measured,
                                         state_age=.04),
                            ('acceleration consumed the stop margin before fresh odometry aged',
                             step, self.node.output.copy()))
            future = transform_points(local, arc_path(*self.node.output, duration=.05, steps=2)[-1],
                                      inverse=True)
            self.assertTrue(braking_clear(future, [0., 0.], self.node.output, state_age=.1),
                            ('next control cycle lost its stopping certificate', step, self.node.output.copy()))
            x, y, yaw = self.node.pose
            v, w = self.node.measured
            self.node.pose = (x+.05*v*math.cos(yaw), y+.05*v*math.sin(yaw), yaw+.05*w)
        self.assertGreater(self.node.pose[0], .15)

    def test_safe_door_full_speed_uses_only_two_original_certificates(self):
        from smart_wheelchair_safety.unified_geometry import braking_clear
        self.node.mode = self.node.door_phase = 'door_pass'
        self.node.door_path = np.array([[0., 0., 0.], [3., 0., 0.]])
        points = np.column_stack((np.linspace(-2., 5., 101), np.full(101, .6)))
        with patch('smart_wheelchair_safety.unified_control_node.braking_clear', wraps=braking_clear) as guard:
            command = self.node._safe_door_command(points, self.node.odom_stamp)
        self.assertEqual(guard.call_count, 2)
        np.testing.assert_allclose(command, [.15, 0.])
        self.assertTrue(all(call.kwargs['margin'] == .05 for call in guard.call_args_list))
        self.assertTrue(all(call.kwargs['state_age'] == .1 for call in guard.call_args_list))

    def test_unsafe_door_full_speed_still_searches_for_certified_command(self):
        from smart_wheelchair_safety.unified_geometry import arc_path, braking_clear, transform_points
        self.node.mode = self.node.door_phase = 'door_pass'
        self.node.door_path = np.array([[0., 0., 0.], [3., 0., 0.]])
        points = np.column_stack((np.full(101, 1.08), np.linspace(-2., 2., 101)))
        with patch('smart_wheelchair_safety.unified_control_node.braking_clear', wraps=braking_clear) as guard:
            command = self.node._safe_door_command(points, self.node.odom_stamp)
        self.assertGreater(guard.call_count, 2)
        self.assertGreater(command[0], 0.)
        self.assertLess(command[0], .15)
        self.assertTrue(braking_clear(points, command, self.node.measured, margin=.05, state_age=.1))
        future = transform_points(points, arc_path(*command, duration=.05, steps=2)[-1], inverse=True)
        self.assertTrue(braking_clear(future, [0., 0.], command, margin=.05, state_age=.1))

    def test_door_alignment_keeps_cruise_when_far_at_any_heading(self):
        from smart_wheelchair_safety.unified_geometry import Opening
        self.node.mode = self.node.door_phase = 'door_align'
        self.node.raw = np.array([.6, 0.])
        self.node.door_path = np.array([[0., 0., 0.], [3., 0., 0.]])
        for center, heading in (((2., 0.), .04), ((2., 0.), .3)):
            with self.subTest(center=center, heading=heading):
                self.node.door = Opening(center, heading, 1., ())
                command = self.node._safe_door_command(np.empty((0, 2)), self.node.odom_stamp)
                self.assertGreater(command[0], .3)

    def test_nearly_aligned_door_limits_speed_before_positive_entry_clearance(self):
        from smart_wheelchair_safety.unified_geometry import Opening, arc_path, door_entry_clearance, transform_points
        self.node.door = Opening((1.265, .021), .04, 1.0075, ())
        self.assertLess(door_entry_clearance(self.node.door), 0.)
        side = np.r_[np.linspace(-2., -.50375, 61), np.linspace(.50375, 2., 61)]
        jambs = np.vstack([np.column_stack((np.full(len(side), x), side)) for x in (-.06, 0., .06)])
        obstacles = transform_points(jambs, (1.265, .021, .04))
        self.node.mode = self.node.door_phase = 'door_align'
        self.node.door_path_feasible = True
        self.node.door_path = np.array([[0., 0., 0.], [3., .12, .04]])
        self.node.raw = np.array([.6, 0.])
        initial_stamp = rospy.Time.now().to_sec()
        previous_acceleration = np.zeros(2)
        saw_pass = False
        for step in range(180):
            stamp = initial_stamp+.05*step
            now = time.monotonic()
            self.node.raw_time = self.node.odom_time = self.node.plan_time = now
            self.node.odom_stamp = stamp-(0., .04, .09)[step % 3]
            self.node.scans = {side: (now, obstacles) for side in ('left', 'right')}
            self.node.last_tick = stamp-.05
            if door_entry_clearance(self.node._local_opening(self.node.door)) > 0.:
                self.node.mode = self.node.door_phase = 'door_pass'
                saw_pass = True
            previous = self.node.output.copy()
            with patch.object(rospy.Time, 'now', return_value=rospy.Time.from_sec(stamp)):
                self.node.control()
            self.assertEqual(self.node.reason, 'clear', (step, self.node.reason))
            self.assertLessEqual(self.node.output[0], .15+1e-6)
            actual_acceleration = (self.node.output-previous)/.05
            self.assertTrue(np.all(np.abs(actual_acceleration) <= [.500001, .800001]))
            self.assertTrue(np.all(np.abs(actual_acceleration-previous_acceleration) <= np.array([2.5, 4.])*.05+1e-6),
                            (step, actual_acceleration, previous_acceleration))
            previous_acceleration = actual_acceleration
            local = arc_path(*self.node.output, duration=.05, steps=2)[-1]
            xy = transform_points([local[:2]], self.node.pose)[0]
            self.node.pose = (*xy, self.node.pose[2]+local[2])
            self.node.measured = self.node.output.copy()
        self.assertTrue(saw_pass)
        self.assertGreater(self.node.pose[0], 1.)

    def test_curved_door_alignment_limits_speed_before_entering_aperture(self):
        from smart_wheelchair_safety.unified_geometry import Opening, arc_path, door_alignment_reference, door_entry_clearance, transform_points
        self.node.door = Opening((1.4, 0.), .26, 1.1055, ())
        self.assertLess(door_entry_clearance(self.node.door), 0.)
        side = np.r_[np.linspace(-2., -.55275, 61), np.linspace(.55275, 2., 61)]
        jambs = np.vstack([np.column_stack((np.full(len(side), x), side)) for x in (-.06, 0., .06)])
        obstacles = transform_points(jambs, (1.4, 0., .26))
        self.node.mode = self.node.door_phase = 'door_align'
        self.node.door_path_feasible = True
        self.node.door_path = door_alignment_reference(self.node.door, 'align')
        self.node.raw = np.array([.6, 0.])
        initial_stamp = rospy.Time.now().to_sec()
        previous_acceleration = np.zeros(2)
        saw_pass = False
        for step in range(300):
            stamp = initial_stamp+.05*step
            now = time.monotonic()
            self.node.raw_time = self.node.odom_time = self.node.plan_time = now
            self.node.odom_stamp = stamp-(0., .04, .09)[step % 3]
            self.node.scans = {side: (now, obstacles) for side in ('left', 'right')}
            self.node.last_tick = stamp-.05
            if door_entry_clearance(self.node._local_opening(self.node.door)) > 0.:
                self.node.mode = self.node.door_phase = 'door_pass'
                saw_pass = True
            previous = self.node.output.copy()
            with patch.object(rospy.Time, 'now', return_value=rospy.Time.from_sec(stamp)):
                self.node.control()
            self.assertEqual(self.node.reason, 'clear', (step, self.node.reason))
            self.assertLessEqual(self.node.output[0], .15+1e-6)
            actual_acceleration = (self.node.output-previous)/.05
            self.assertTrue(np.all(np.abs(actual_acceleration) <= [.500001, .800001]))
            self.assertTrue(np.all(np.abs(actual_acceleration-previous_acceleration) <= np.array([2.5, 4.])*.05+1e-6),
                            (step, actual_acceleration, previous_acceleration))
            previous_acceleration = actual_acceleration
            local = arc_path(*self.node.output, duration=.05, steps=2)[-1]
            xy = transform_points([local[:2]], self.node.pose)[0]
            self.node.pose = (*xy, self.node.pose[2]+local[2])
            self.node.measured = self.node.output.copy()
        self.assertTrue(saw_pass)
        self.assertGreater(self.node.pose[0], 1.)

    def test_door_pivot_requires_arrival_before_rotation_and_completion(self):
        self.node.mode = self.node.door_phase = 'door_align'
        self.node.door_path_feasible = True
        self.node.door_rotation_index = 0
        self.node.door_path = np.array([[0., 0., 0.], [.36, 0., 0.],
                                       [.36, 0., math.pi/2], [.36, 1., math.pi/2]])
        self.node.pose = (.291, 0., 0.)  # Recorded failure: 6.9 cm short.
        self.node.measured = np.zeros(2)
        command = self.node._safe_door_command(np.empty((0, 2)), self.node.odom_stamp)
        self.assertEqual(self.node.door_rotation_index, 0)
        self.assertGreater(command[0], 0., 'finish translation to the checked pivot first')
        self.assertAlmostEqual(command[1], 0.)
        self.node.pose = (.34, 0., 0.)
        command = self.node._safe_door_command(np.empty((0, 2)), self.node.odom_stamp)
        self.assertEqual(self.node.door_rotation_index, 1)
        self.assertEqual(command[0], 0.)
        self.assertGreater(command[1], 0.)
        self.node.pose = (.291, 0., math.pi/2)
        self.assertIsNotNone(self.node._pending_door_rotation(),
                             'heading alone must not finish a pivot 6.9 cm away')
        self.node.pose = (.34, 0., math.pi/2)
        self.assertIsNone(self.node._pending_door_rotation())

    def test_door_pivot_stays_latched_across_small_position_drift(self):
        for side in (-1., 1.):
            with self.subTest(side=side):
                self.node.mode = self.node.door_phase = 'door_align'
                self.node.door_path_feasible = True
                self.node.door_rotation_index = 0
                self.node.door_path = np.array([[0., 0., 0.], [.36, 0., 0.],
                                                [.36, 0., side*math.pi/2], [.36, side, side*math.pi/2]])
                self.node.pose = (.341, 0., side*1.45)
                first = self.node._safe_door_command(np.empty((0, 2)), self.node.odom_stamp)
                self.assertEqual(first[0], 0.)
                self.assertGreater(side*first[1], 0.)
                self.node.pose = (.340, side*.015, side*1.49)
                second = self.node._safe_door_command(np.empty((0, 2)), self.node.odom_stamp)
                self.assertEqual(second[0], 0.)
                self.assertGreater(side*second[1], 0.)
                self.node.pose = (.340, side*.015, side*math.pi/2)
                self.assertIsNone(self.node._pending_door_rotation())

    def test_excessive_pivot_drift_stops_and_invalidates_the_path(self):
        self.node.mode = self.node.door_phase = 'door_align'
        self.node.door_path_feasible = True
        self.node.door_path = np.array([[0., 0., 0.], [.36, 0., 0.], [.36, 0., math.pi/2]])
        self.node.pose = (.341, 0., 1.)
        self.node._safe_door_command(np.empty((0, 2)), self.node.odom_stamp)
        self.node.pose = (.290, .015, math.pi/2)
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
        self.node.door = self.front_opening()
        self.node.door_obstacle_time = time.monotonic()
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
        self.node.scans = {side: (time.monotonic(), points)
                           for side, (_, points) in self.node.scans.items()}
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
        self.node.scans = {side: (time.monotonic(), points)
                           for side, (_, points) in self.node.scans.items()}
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

        from geometry_msgs.msg import TwistStamped
        self.node.use_neupan = self.node.accept_planned = True
        self.node.active_reference_stamp = rospy.Time.now()
        action = TwistStamped()
        action.header.stamp = self.node.active_reference_stamp
        action.twist.linear.x, action.twist.angular.z = .3, .65
        self.node.on_neupan(action)
        self.node.control()
        self.assertEqual(self.commands[-1].linear.x, 0.)
        self.assertGreater(self.commands[-1].angular.z, 0.)

    def test_moving_door_rebuild_cannot_bypass_full_sweep_margin(self):
        from smart_wheelchair_safety.unified_geometry import door_alignment_reference, reference_is_clear
        door = self.front_opening(center=(2., 0.), heading=0., width=1.1)
        points = np.column_stack((np.linspace(.2, 2.8, 81), np.full(81, .458)))
        direct = door_alignment_reference(door, 'align')
        self.assertTrue(reference_is_clear(direct, points, margin=.045))
        self.assertFalse(reference_is_clear(direct, points, margin=.06))
        self.node.door_phase = 'door_align'
        self.node.door_path = None
        self.node.door_path_feasible = False
        self.node.output = self.node.measured = np.array([.15, 0.])
        with patch.object(self.node.door_plan_executor, 'submit') as submit:
            self.node._local_door_path(door, points)
        self.assertIsNone(self.node.door_path)
        self.assertFalse(self.node.door_path_feasible)
        submit.assert_not_called()  # Stop before fixing a new staging origin.
        self.node.output = self.node.measured = np.zeros(2)
        with patch.object(self.node.door_plan_executor, 'submit') as submit:
            self.node._local_door_path(door, points)
        submit.assert_called_once()
        self.assertFalse(self.node.door_path_feasible)

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

    def test_short_raw_gap_stops_then_resumes_same_door_path(self):
        from geometry_msgs.msg import Twist
        self.node.door = self.front_opening(center=(2., .2), heading=.3)
        self.node.mode = self.node.door_phase = 'door_align'
        self.node.door_path = np.array([[0., 0., 0.], [1., .2, .3], [3., .2, .3]])
        self.node.door_path_feasible = True
        original = self.node.door_path
        self.node.output = np.array([.15, .02])
        self.node.raw_time = time.monotonic()-.4
        self.node.door_away_since = time.monotonic()-.2
        self.node.control()
        self.assertEqual(self.node.reason, 'stale_input')
        np.testing.assert_array_equal(self.node.output, [0., 0.])
        self.node.update_reference()
        self.assertIsNotNone(self.node.door)
        self.assertIs(self.node.door_path, original)
        self.assertEqual(self.node.door_away_since, 0.)
        self.assertIsNone(self.node.active_reference_stamp)
        self.assertFalse(self.node.accept_planned)
        msg = Twist()
        msg.linear.x = .5
        self.node.on_raw(msg)
        self.node.odom_time = time.monotonic()
        self.node.odom_stamp = rospy.Time.now().to_sec()
        self.node.update_reference()
        self.assertEqual(self.node.mode, 'door_align')
        self.assertIs(self.node.door_path, original)
        self.node.last_tick = rospy.Time.now().to_sec()-.05
        self.node.control()
        self.assertGreater(self.node.output[0], 0.)
        self.assertIn(self.node.reason, ('clear', 'planner_fallback'))

    def test_sustained_raw_gap_discards_door_after_existing_grace(self):
        self.node.door = self.front_opening()
        self.node.raw_time = 99.6
        with patch('smart_wheelchair_safety.unified_control_node.time.monotonic', return_value=100.):
            self.node.update_reference()
        self.assertIsNotNone(self.node.door)
        with patch('smart_wheelchair_safety.unified_control_node.time.monotonic', return_value=101.01):
            self.node.update_reference()
        self.assertIsNone(self.node.door)
        self.assertIsNone(self.node.door_path)

    def test_explicit_input_and_assist_cancel_door_without_grace(self):
        from geometry_msgs.msg import Twist
        from std_msgs.msg import Bool
        for linear in (0., -.2, float('nan')):
            with self.subTest(linear=linear):
                self.node.door = self.front_opening()
                self.node.door_path = np.array([[0., 0., 0.], [3., 0., 0.]])
                msg = Twist()
                msg.linear.x = linear
                self.node.on_raw(msg)
                self.node.update_reference()
                self.assertIsNone(self.node.door)
                self.assertIsNone(self.node.door_path)
        self.node.door = self.front_opening()
        self.node.on_assist_enabled(Bool(data=False))
        self.assertIsNone(self.node.door)

    def test_large_observed_inward_update_revokes_path_and_old_worker(self):
        from smart_wheelchair_safety.unified_geometry import Opening
        from unittest.mock import Mock
        row = json.loads((Path(__file__).with_name('fixtures')/'v10_far_opening_update.json').read_text())
        old, new = Opening(**row['tracked']), Opening(**row['observed'])
        self.node.door = old
        self.node.confirmed_openings = [old]
        self.node.mode = self.node.door_phase = 'door_align'
        self.node.door_path = np.array([[0., 0., 0.], [3., 0., 0.]])
        self.node.door_path_feasible = True
        self.node.accept_planned = True
        self.node.active_reference_stamp = rospy.Time.now()
        generation = self.node.door_generation
        future = SimpleNamespace(done=lambda: True, result=Mock(return_value=self.node.door_path.copy()), cancel=Mock())
        self.node.door_plan_request = (future, self.node.pose, generation)
        self.node._observe_openings([new])
        self.node._observe_openings([new])
        self.assertAlmostEqual(self.node.door.width, new.width)
        self.assertEqual(len(self.node.confirmed_openings), 1)
        self.assertIsNone(self.node.door_path)
        self.assertFalse(self.node.door_path_feasible)
        self.assertFalse(self.node.accept_planned)
        self.assertIsNone(self.node.active_reference_stamp)
        self.assertGreater(self.node.door_generation, generation)
        with patch.object(self.node.door_plan_executor, 'submit') as submit:
            self.node._local_door_path(new, np.empty((0, 2)))
        future.result.assert_not_called()
        submit.assert_called_once()
        self.assertIsNone(self.node.door_path)
        self.assertFalse(self.node.door_path_feasible)

    def test_se_observed_inward_update_revokes_path_and_old_worker(self):
        from smart_wheelchair_safety.unified_geometry import Opening
        from unittest.mock import Mock
        row = json.loads((Path(__file__).with_name('fixtures')/'v10_se_far_opening_update.json').read_text())
        old, new = Opening(**row['tracked']), Opening(**row['observed'])
        self.node.door = old
        self.node.confirmed_openings = [old]
        self.node.mode = self.node.door_phase = 'door_align'
        self.node.door_path = np.array([[0., 0., 0.], [3., 0., 0.]])
        self.node.door_path_feasible = True
        self.node.accept_planned = True
        self.node.active_reference_stamp = rospy.Time.now()
        generation = self.node.door_generation
        future = SimpleNamespace(done=lambda: True, result=Mock(return_value=self.node.door_path.copy()), cancel=Mock())
        self.node.door_plan_request = (future, self.node.pose, generation)
        self.node._observe_openings([new])
        self.node._observe_openings([new])
        self.assertAlmostEqual(self.node.door.width, new.width)
        self.assertEqual(len(self.node.confirmed_openings), 1)
        self.assertIsNone(self.node.door_path)
        self.assertFalse(self.node.door_path_feasible)
        self.assertFalse(self.node.accept_planned)
        self.assertIsNone(self.node.active_reference_stamp)
        self.assertGreater(self.node.door_generation, generation)
        with patch.object(self.node.door_plan_executor, 'submit') as submit:
            self.node._local_door_path(new, np.empty((0, 2)))
        future.result.assert_not_called()
        submit.assert_called_once()
        self.assertIsNone(self.node.door_path)
        self.assertFalse(self.node.door_path_feasible)

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
        self.finish_door_plan()

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

    def test_door_speed_slew_retains_curvature_target_and_guards_actual_output(self):
        self.node.door = self.front_opening(center=(2., .2), heading=.2, width=1.)
        self.node.door_phase = 'door_align'
        self.node.mode = 'door_align'
        self.node.door_path = np.array([[0., 0., 0.], [.3, .03, .1], [2., .2, .2]])
        self.node.door_path_feasible = True
        self.node.raw = np.array([.8, 0.])
        self.node.output = np.array([.05, -.1])
        self.node.acceleration = np.array([0., 0.])

        from smart_wheelchair_safety import unified_control_node as module
        before=self.node.output.copy();entry_a=self.node.acceleration.copy()
        self.node.last_tick=99.9;self.node.odom_stamp=100.
        with patch.object(rospy.Time,'now',return_value=rospy.Time.from_sec(100.)),patch.object(
                self.node,'fresh',return_value=True),patch.object(self.node,'_door_angular',wraps=self.node._door_angular) as curvature,patch.object(
                module,'braking_clear',wraps=module.braking_clear) as guard:
            self.node.control()
        command=self.node.output.copy()
        self.assertAlmostEqual(curvature.call_args.args[0],command[0])
        target_w=self.node._door_preview_angular(command[0])
        self.assertGreater(target_w,0.)
        with patch.object(self.node,'output',before):
            expected,expected_a=self.node._assisted_slew(np.array([command[0],target_w]),.1,acceleration=entry_a)
        self.assertAlmostEqual(command[1],expected[1])
        self.assertAlmostEqual(self.node.acceleration[1],expected_a[1])
        np.testing.assert_allclose(guard.call_args.args[1],command)

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

    def test_raw_skips_door_search_when_override_cannot_use_it(self):
        from geometry_msgs.msg import Twist
        for active_door, angular in ((False, 0.), (False, .1), (True, .3)):
            with self.subTest(active_door=active_door, angular=angular):
                self.node.door = self.front_opening() if active_door else None
                self.node.wall_side = 1
                self.node.mode = 'wall'
                self.node.door_away_since = 0.
                msg = Twist()
                msg.linear.x, msg.angular.z = .5, angular
                with patch.object(self.node, '_intended_door', side_effect=AssertionError('unused door search')), patch(
                        'smart_wheelchair_safety.unified_control_node.intended_side_opening',
                        side_effect=AssertionError('unused side search')):
                    self.node.on_raw(msg)
                self.assertFalse(self.node.override)

    def test_raw_cancelled_door_overrides_without_another_search(self):
        from geometry_msgs.msg import Twist
        self.node.door = self.front_opening()
        self.node.door_away_since = time.monotonic()-1.
        msg = Twist()
        msg.linear.x, msg.angular.z = .5, .4
        with patch.object(self.node, '_intended_door', side_effect=AssertionError('cancelled door search')), patch(
                'smart_wheelchair_safety.unified_control_node.active_aperture_targeted', return_value=False):
            self.node.on_raw(msg)
        self.assertTrue(self.node.override)
        self.assertIsNone(self.node.door)

    def test_cancelled_captured_door_is_not_reacquired_by_unchanged_input(self):
        from geometry_msgs.msg import Twist
        from smart_wheelchair_safety.unified_geometry import Opening
        capture = json.loads((Path(__file__).with_name('fixtures') /
                              'wall_door_recapture.json').read_text())
        value = capture['door']
        for side in (-1., 1.):
            with self.subTest(side=side):
                door = Opening((value['center'][0], side*value['center'][1]),
                               side*value['heading'], value['width'])
                self.node.pose = (0., 0., 0.)
                self.node.door = door
                self.node.confirmed_openings = [door]
                self.node.mode = 'door_wait'
                self.node.door_away_since = time.monotonic()-1.
                msg = Twist()
                msg.linear.x, msg.angular.z = capture['raw'][0], side*capture['raw'][1]
                self.node.on_raw(msg)
                self.assertIsNone(self.node.door)
                self.assertTrue(self.node.override)
                self.assertEqual(self.node._intended_door(), (None, None))
                self.node.mode = 'override'
                self.node.on_raw(msg)
                self.assertTrue(self.node.override, 'unchanged away input must not reacquire the door')

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

    def test_oversteered_arc_that_cannot_reach_door_plane_does_not_take_over(self):
        self.confirm_door(center=(3.3168, 1.1490), heading=.4428, width=1.3008)
        self.node.raw = np.array([.8977, .5182])
        self.node.wall_side = 1

        self.node.update_reference()

        # The joystick circle cannot reach this door plane; an apparent
        # lateral match inside the short horizon must not claim the door.
        self.assertFalse(self.node.mode.startswith('door_'))
        self.assertIsNone(self.node.door)

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
        references, limits, stamped_limits = [], [], []
        self.node.reference = SimpleNamespace(publish=references.append)
        self.node.limit_pub = SimpleNamespace(publish=limits.append)
        self.node.reference_speed_pub = SimpleNamespace(publish=stamped_limits.append)

        self.node.update_reference()

        self.assertIsInstance(references[-1], Path)
        self.assertEqual(references[-1].header.frame_id, 'odom')
        self.assertIsInstance(limits[-1], Float32)
        self.assertEqual(stamped_limits[-1].header.stamp, references[-1].header.stamp)
        self.assertEqual(stamped_limits[-1].twist.linear.x, limits[-1].data)
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

    def test_opening_extraction_allows_odometry_and_preserves_world_position(self):
        from geometry_msgs.msg import TransformStamped
        from nav_msgs.msg import Odometry
        from sensor_msgs.msg import LaserScan
        from smart_wheelchair_safety.unified_geometry import Opening
        waiting, release, odom_done = threading.Event(), threading.Event(), threading.Event()
        transform = TransformStamped()
        transform.transform.rotation.w = 1.
        self.node.tf = SimpleNamespace(lookup_transform=lambda *_: transform)
        scan = LaserScan(angle_increment=.1, range_min=.05, range_max=8.,
                         ranges=[float('inf')] * 10)
        scan.header.stamp = rospy.Time.now()
        odom = Odometry()
        odom.header.stamp = scan.header.stamp
        odom.pose.pose.position.x = .2
        odom.pose.pose.orientation.w = 1.
        observed = []
        self.node._observe_openings = lambda openings: observed.extend(
            self.node._world_opening(opening) for opening in openings)
        def extract(_):
            waiting.set()
            release.wait(1.)
            return [Opening((2., .3), .1, 1., ())]
        def receive_odom():
            self.callbacks['odom'](odom)
            odom_done.set()
        scanner = threading.Thread(target=self.callbacks['scan_left'], args=(scan,))
        odometry = threading.Thread(target=receive_odom)
        with patch('smart_wheelchair_safety.unified_control_node.find_openings', side_effect=extract):
            scanner.start()
            try:
                self.assertTrue(waiting.wait(.5))
                odometry.start()
                self.assertTrue(odom_done.wait(.1), 'opening extraction blocked odometry')
            finally:
                release.set()
                scanner.join(1.)
                if odometry.ident is not None:
                    odometry.join(1.)
        self.assertFalse(scanner.is_alive())
        self.assertEqual(len(observed), 1)
        np.testing.assert_allclose(observed[0].center, (2., .3))
        self.assertAlmostEqual(observed[0].heading, .1)

    def test_opening_extraction_rejects_revoked_or_expired_snapshot(self):
        from geometry_msgs.msg import TransformStamped
        from sensor_msgs.msg import LaserScan
        transform = TransformStamped()
        transform.transform.rotation.w = 1.
        self.node.tf = SimpleNamespace(lookup_transform=lambda *_: transform)
        for invalidation in ('epoch', 'generation', 'shutdown', 'expired', 'superseded'):
            with self.subTest(invalidation=invalidation):
                self.node.stopped = False
                self.node.opening_observation_time = 0.
                scan = LaserScan(angle_increment=.1, range_min=.05, range_max=8.,
                                 ranges=[float('inf')] * 10)
                scan.header.stamp = rospy.Time.now()
                def extract(_):
                    if invalidation == 'epoch':
                        self.node.cancel()
                    elif invalidation == 'generation':
                        self.node.door_generation += 1
                    elif invalidation == 'shutdown':
                        self.node.stopped = True
                    elif invalidation == 'expired':
                        scan.header.stamp = rospy.Time.now()-rospy.Duration(1.)
                    else:
                        self.node.opening_observation_time += .1
                    return []
                with patch('smart_wheelchair_safety.unified_control_node.find_openings', side_effect=extract), patch.object(
                        self.node, '_observe_openings') as observe:
                    self.node.on_scan(scan, 'left')
                observe.assert_not_called()
        self.node.stopped = False

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

    def test_odometry_receipt_follows_simulation_clock_with_wall_deadman(self):
        from nav_msgs.msg import Odometry
        msg = Odometry()
        msg.header.stamp = rospy.Time.from_sec(10.)
        msg.pose.pose.orientation.w = 1.
        with patch('smart_wheelchair_safety.unified_control_node.time.monotonic', return_value=100.), patch.object(
                rospy.Time, 'now', return_value=rospy.Time.from_sec(10.)):
            self.node.on_odom(msg)
        self.node.raw_time = 100.
        self.node.scans = {side: (100., np.array([[3., 2.]])) for side in ('left', 'right')}
        for simulation, wall_elapsed, ros_elapsed, expected in (
                (True, .125, .05, True),
                (True, .125, .100001, False),
                (False, .125, .05, False),
                (True, .251, 0., False)):
            with self.subTest(simulation=simulation, wall_elapsed=wall_elapsed, ros_elapsed=ros_elapsed):
                self.node.use_sim_time = simulation
                with patch('smart_wheelchair_safety.unified_control_node.time.monotonic', return_value=100.+wall_elapsed), patch.object(
                        rospy.Time, 'now', return_value=rospy.Time.from_sec(10.+ros_elapsed)):
                    self.assertEqual(self.node.fresh(), expected)
        # Independently lost odometry must also expire when clock freezes,
        # even if the joystick and scanners keep sending wall-clock heartbeats.
        self.node.raw_time = 100.3
        self.node.scans = {side: (100.3, np.array([[3., 2.]])) for side in ('left', 'right')}
        with patch('smart_wheelchair_safety.unified_control_node.time.monotonic', return_value=100.3), patch.object(
                rospy.Time, 'now', return_value=rospy.Time.from_sec(10.)):
            self.assertFalse(self.node.fresh())
            self.node.output = np.array([.15, 0.])
            self.node.control()
            np.testing.assert_array_equal(self.node.output, [0., 0.])
            self.assertEqual(self.node.reason, 'stale_input')

    def planner_mailbox_message(self, v=.3, stamp=None):
        from geometry_msgs.msg import TwistStamped
        msg = TwistStamped()
        msg.header.stamp = rospy.Time.now() if stamp is None else stamp
        msg.twist.linear.x = v
        return msg

    def test_planner_mailbox_transport_does_not_wait_for_controller(self):
        done = threading.Event()
        def receive():
            self.callbacks['cmd_vel_planned'](self.planner_mailbox_message(.3))
            self.callbacks['neupan/cmd_vel'](self.planner_mailbox_message(.4))
            done.set()
        with self.node.callback_lock:
            worker = threading.Thread(target=receive)
            worker.start()
            unblocked = done.wait(.3)
        worker.join(1.)
        self.assertTrue(unblocked)
        self.node.active_reference_stamp = rospy.Time.now()
        self.node.accept_planned = True
        self.node.use_neupan = True
        self.node._receive_planned(self.planner_mailbox_message(.3, self.node.active_reference_stamp))
        self.node._receive_neupan(self.planner_mailbox_message(.4, self.node.active_reference_stamp))
        self.node._serialized(lambda: self.node.control())()
        self.assertEqual(self.node.planned[0], .3)
        self.assertEqual(self.node.neupan[0], .4)
        self.assertNotEqual(self.node.reason, 'planner_timeout')

    def test_planner_mailbox_expired_receipt_and_wrong_stamp_never_refresh(self):
        self.node.active_reference_stamp = rospy.Time.now()
        self.node.accept_planned = True
        for receiver in (self.node._receive_planned, self.node._receive_neupan):
            with patch('smart_wheelchair_safety.unified_control_node.time.monotonic', return_value=100.):
                receiver(self.planner_mailbox_message(.4, self.node.active_reference_stamp))
        with patch('smart_wheelchair_safety.unified_control_node.time.monotonic', return_value=100.251):
            self.node._consume_planner_actions()
        self.assertEqual(self.node.neupan_time, 0.)
        self.assertEqual(self.node.planned[0], .5)
        wrong = self.node.active_reference_stamp-rospy.Duration(.01)
        self.node._receive_planned(self.planner_mailbox_message(.7, wrong))
        self.node._receive_neupan(self.planner_mailbox_message(.7, wrong))
        self.node._consume_planner_actions()
        self.assertEqual(self.node.neupan_time, 0.)
        self.assertEqual(self.node.planned[0], .5)

    def test_planner_mailbox_cancel_consumes_before_old_action(self):
        self.node.active_reference_stamp = rospy.Time.now()
        self.node.accept_planned = True
        self.node.use_neupan = True
        stamp = self.node.active_reference_stamp
        self.node._receive_raw(self.raw_mailbox_message(0.))
        self.node._receive_raw(self.raw_mailbox_message(.6))
        self.node._receive_planned(self.planner_mailbox_message(.7, stamp))
        self.node._receive_neupan(self.planner_mailbox_message(.7, stamp))
        self.node._serialized(lambda: None)()
        self.assertIsNone(self.node.active_reference_stamp)
        self.assertFalse(self.node.accept_planned)
        np.testing.assert_array_equal(self.node.planned, [0., 0.])
        np.testing.assert_array_equal(self.node.neupan, [0., 0.])

    def test_planner_mailbox_preserves_receipt_and_newer_during_consumption(self):
        self.node.active_reference_stamp = rospy.Time.now()
        self.node.accept_planned = True
        stamp = self.node.active_reference_stamp
        with patch('smart_wheelchair_safety.unified_control_node.time.monotonic', return_value=100.):
            self.node._receive_planned(self.planner_mailbox_message(.3, stamp))
        apply = self.node.on_planned
        def replace(*args):
            self.node._receive_planned(self.planner_mailbox_message(.6, stamp))
            apply(*args)
        with patch('smart_wheelchair_safety.unified_control_node.time.monotonic', return_value=100.1), patch.object(
                self.node, 'on_planned', side_effect=replace):
            self.node._consume_planner_actions()
        self.assertEqual(self.node.plan_time, 100.)
        self.assertEqual(self.node.planned[0], .3)
        with patch('smart_wheelchair_safety.unified_control_node.time.monotonic', return_value=100.11):
            self.node._consume_planner_actions()
        self.assertEqual(self.node.plan_time, 100.1)
        self.assertEqual(self.node.planned[0], .6)

    @staticmethod
    def raw_mailbox_message(v, w=0.):
        from geometry_msgs.msg import Twist
        msg = Twist()
        msg.linear.x, msg.angular.z = v, w
        return msg

    def test_raw_mailbox_receiver_does_not_wait_for_state_lock(self):
        done = threading.Event()
        def receive():
            self.callbacks['cmd_vel_raw'](self.raw_mailbox_message(.3))
            self.callbacks['cmd_vel_raw'](self.raw_mailbox_message(.6))
            done.set()
        with self.node.callback_lock:
            thread = threading.Thread(target=receive)
            thread.start()
            unblocked = done.wait(.3)
        thread.join(1.)
        self.assertTrue(unblocked)
        self.node._serialized(lambda: None)()
        self.assertEqual(self.node.raw[0], .6)

    def test_raw_mailbox_latches_every_cancel_before_new_forward(self):
        for v, w in ((0., 0.), (-.2, 0.), (0., .5), (float('nan'), 0.), (.5, float('inf'))):
            with self.subTest(v=v, w=w):
                self.node.active_reference_stamp = rospy.Time.now()
                self.node.accept_planned = True
                self.node.planned = np.array([.3, .2])
                self.node.neupan = np.array([.4, .1])
                self.node.door = self.front_opening()
                self.node.door_path = np.array([[0., 0., 0.], [3., 0., 0.]])
                self.node._receive_raw(self.raw_mailbox_message(v, w))
                self.node._receive_raw(self.raw_mailbox_message(.6))
                self.node._serialized(lambda: None)()
                self.assertIsNone(self.node.door)
                self.assertIsNone(self.node.door_path)
                self.assertIsNone(self.node.active_reference_stamp)
                self.assertFalse(self.node.accept_planned)
                np.testing.assert_array_equal(self.node.planned, [0., 0.])
                np.testing.assert_array_equal(self.node.neupan, [0., 0.])
                self.assertEqual(self.node.raw[0], .6)

    def test_raw_mailbox_keeps_replacement_and_never_replays_pre_cancel_forward(self):
        self.node._receive_raw(self.raw_mailbox_message(.4))
        older = self.node.latest_raw
        self.node._receive_raw(self.raw_mailbox_message(0.))
        # Receiver may publish the cancellation latch before replacing latest.
        self.node.latest_raw = older
        self.node._serialized(lambda: None)()
        self.assertEqual(self.node.raw[0], 0.)
        self.node._receive_raw(self.raw_mailbox_message(.5))
        apply = self.node.on_raw
        def replacement(*args):
            self.node._receive_raw(self.raw_mailbox_message(.7))
            apply(*args)
        with patch.object(self.node, 'on_raw', side_effect=replacement):
            self.node._serialized(lambda: None)()
        self.assertEqual(self.node.raw[0], .5)
        self.node._serialized(lambda: None)()
        self.assertEqual(self.node.raw[0], .7)
        with patch.object(self.node, 'on_raw') as duplicate:
            self.node._serialized(lambda: None)()
        duplicate.assert_not_called()

    def test_raw_mailbox_preserves_wall_receipt_and_rejects_shutdown_input(self):
        with patch('smart_wheelchair_safety.unified_control_node.time.monotonic', return_value=100.):
            self.node._receive_raw(self.raw_mailbox_message(.6))
        with patch('smart_wheelchair_safety.unified_control_node.time.monotonic', return_value=100.31):
            self.node.odom_time = 100.31
            self.node.odom_stamp = rospy.Time.now().to_sec()
            self.node.scans = {side: (100.31, np.array([[3., 2.]])) for side in ('left', 'right')}
            self.node._serialized(lambda: None)()
            self.assertEqual(self.node.raw_time, 100.)
            self.node.output = np.array([.15, 0.])
            self.node.control()
            self.assertEqual(self.node.reason, 'stale_input')
            np.testing.assert_array_equal(self.node.output, [0., 0.])
        pending = self.node.latest_raw
        self.node.destroy_node()
        self.node._receive_raw(self.raw_mailbox_message(.8))
        with patch.object(self.node, 'on_raw') as apply:
            self.node._consume_raw()
        self.assertIs(self.node.latest_raw, pending)
        apply.assert_not_called()

    def mailbox_message(self, x, stamp=None):
        from nav_msgs.msg import Odometry
        msg = Odometry()
        msg.header.stamp = rospy.Time.now() if stamp is None else rospy.Time.from_sec(stamp)
        msg.pose.pose.orientation.w = 1.
        msg.pose.pose.position.x = x
        return msg

    def test_mailbox_receiver_never_waits_for_control_lock_and_keeps_latest(self):
        first, second = self.mailbox_message(1.), self.mailbox_message(2.)
        done = threading.Event()
        def receive():
            self.callbacks['odom'](first)
            self.callbacks['odom'](second)
            done.set()
        with self.node.callback_lock:
            worker = threading.Thread(target=receive)
            worker.start()
            received_without_lock = done.wait(.5)
            self.assertEqual(self.node.pose, (0., 0., 0.))
        worker.join(1.)
        self.assertTrue(received_without_lock, 'transport waited for state-machine lock')
        self.assertIs(self.node.latest_odom[0], second)
        # Consume a fresh replacement at the next serialized callback.
        self.callbacks['odom'](self.mailbox_message(3.))
        self.node._serialized(lambda: None)()
        self.assertEqual(self.node.pose[0], 3.)

    def test_mailbox_replacement_during_consume_is_not_cleared(self):
        first, second = self.mailbox_message(1.), self.mailbox_message(2.)
        self.node._receive_odom(first)
        apply = self.node.on_odom
        def replacing_apply(*args):
            self.node._receive_odom(second)
            apply(*args)
        with patch.object(self.node, 'on_odom', side_effect=replacing_apply):
            self.node._serialized(lambda: None)()
        self.assertEqual(self.node.pose[0], 1.)
        self.node._serialized(lambda: None)()
        self.assertEqual(self.node.pose[0], 2.)
        with patch.object(self.node, 'on_odom') as duplicate:
            self.node._serialized(lambda: None)()
        duplicate.assert_not_called()

    def test_mailbox_preserves_receipt_clocks_and_rejects_expired_snapshot(self):
        for simulation, wall_age, ros_age, valid in (
                (False, .08, .02, True), (False, .125, .05, False),
                (True, .125, .05, True), (True, .26, .05, False),
                (True, .08, .11, False)):
            with self.subTest(simulation=simulation, wall_age=wall_age, ros_age=ros_age):
                self.node.use_sim_time = simulation
                self.node.pose = (-1., 0., 0.)
                with patch('smart_wheelchair_safety.unified_control_node.time.monotonic', return_value=100.), patch.object(
                        rospy.Time, 'now', return_value=rospy.Time.from_sec(10.)):
                    self.node._receive_odom(self.mailbox_message(2., 10.))
                with patch('smart_wheelchair_safety.unified_control_node.time.monotonic', return_value=100.+wall_age), patch.object(
                        rospy.Time, 'now', return_value=rospy.Time.from_sec(10.+ros_age)):
                    self.node._serialized(lambda: None)()
                self.assertEqual(self.node.pose[0], 2. if valid else -1.)
                if valid:
                    self.assertEqual(self.node.odom_time, 100.)
                    self.assertEqual(self.node.odom_ros_time, 10.)

    def test_mailbox_shutdown_rejects_pending_and_late_odometry(self):
        self.node._receive_odom(self.mailbox_message(1.))
        pending = self.node.latest_odom
        self.node.destroy_node()
        with patch.object(self.node.transforms, 'sendTransform') as publish:
            self.node._receive_odom(self.mailbox_message(2.))
            self.node._consume_odom()
        self.assertIs(self.node.latest_odom, pending)
        self.assertEqual(self.node.pose, (0., 0., 0.))
        publish.assert_not_called()

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


    def _depth_case_node(self, progress, angle=0., mirror=1., points=True):
        from smart_wheelchair_safety import unified_control_node as m
        n=m.UnifiedControlNode.__new__(m.UnifiedControlNode)
        # Actual NE front/back wall faces y=1.41/1.29, mirrored to opposite travel.
        heading=-mirror*math.pi/2;normal=np.array([math.cos(heading),math.sin(heading)])
        n.door=m.Opening((4.5,mirror*1.41),heading,1.1)
        n.pose=(* (np.array(n.door.center)+progress*normal),heading+angle)
        n.door_phase='door_clear';n.door_obstacle_time=1.
        n.door_obstacles=np.array([(x,mirror*y) for x in (3.95,5.05) for y in (1.41,1.29)]) if points else np.empty((0,2))
        n.raw=np.array([.5,.3]);n._door_preview_angular=lambda _: 0.
        n._clear_door=Mock(side_effect=lambda:setattr(n,'door',None))
        return n

    def test_wall_depth_retains_owner_and_partial_raw_blend(self):
        for mirror in (-1.,1.):
            n=self._depth_case_node(.30,mirror=mirror)
            self.assertLess(n._door_angular(.1),.3)
            self.assertIsNotNone(n._local_tracked_door())
            n._clear_door.assert_not_called()
            n=self._depth_case_node(.42,mirror=mirror)
            self.assertIsNone(n._local_tracked_door())
            n._clear_door.assert_called_once()

    def test_rotated_rear_corners_are_still_inside_at_depth_plus_point29(self):
        for mirror in (-1.,1.):
            for angle in (-math.pi/12,math.pi/12):
                n=self._depth_case_node(.42,angle,mirror)
                self.assertIsNotNone(n._local_tracked_door())
                n=self._depth_case_node(.51,angle,mirror)
                self.assertIsNone(n._local_tracked_door())

    def test_recorded_ne_scan_retains_observed_wall_back_face(self):
        data = json.loads((Path(__file__).with_name('fixtures') / 'observed_door_depth.json').read_text())
        world = data['world_points']
        for mirror in (-1.,1.):
            n=self._depth_case_node(.30,mirror=mirror)
            n.door_obstacles=np.asarray(world)*[1.,mirror]
            threshold=n._door_rear_clear_progress(n._local_opening(n.door))
            self.assertGreater(threshold,.40)
            self.assertLess(threshold,.43)
            self.assertIsNotNone(n._local_tracked_door())

    def test_aligned_without_observation_preserves_point29(self):
        self.assertIsNotNone(self._depth_case_node(.28,points=False)._local_tracked_door())
        self.assertIsNone(self._depth_case_node(.30,points=False)._local_tracked_door())

    def test_unrelated_points_do_not_extend_the_door(self):
        n=self._depth_case_node(.42)
        n.door_obstacles=np.vstack((n.door_obstacles,[[8.,1.15],[5.05,.8]]))
        self.assertIsNone(n._local_tracked_door())

    def test_nonfinite_observation_cannot_trigger_early_clear(self):
        for bad in (math.nan, math.inf, -math.inf):
            n=self._depth_case_node(.42)
            n.door_obstacles=np.vstack((n.door_obstacles,[[bad,1.29]]))
            self.assertEqual(n._door_angular(.1),0.)
            self.assertIsNotNone(n._local_tracked_door())


    def test_override_reserve_uses_user_curvature_but_excludes_direct_reverse_recovery(self):
        points = np.column_stack((np.linspace(-2.0, 10.0, 161), np.full(161, 0.52)))
        for assist, recovering, raw in ((True, False, [0.8, 0.04]), (False, False, [0.8, 0.04]), (True, False, [-0.2, 0.04]), (True, True, [0.2, 0.04])):
            with self.subTest(assist=assist, recovering=recovering, raw=raw):
                self.node.mode = 'override'
                self.node.override = True
                self.node.assist_enabled = assist
                self.node.recovering = recovering
                self.node.raw = np.array(raw)
                self.node.output[:] = self.node.acceleration[:] = 0.0
                now = time.monotonic()
                self.node.raw_time = self.node.odom_time = self.node.plan_time = now
                self.node.odom_stamp = rospy.Time.now().to_sec()
                self.node.last_tick = self.node.odom_stamp - 0.05
                trial_points = np.array([[-0.27, 0.0]]) if recovering else points
                self.node.scans = {side: (now, trial_points) for side in ('left', 'right')}
                original = self.node._safe_assisted_command
                reserved = []

                def observe(desired, obstacles):
                    answer = original(desired, obstacles)
                    reserved.append(answer)
                    return answer
                with patch.object(self.node, '_safe_assisted_command', side_effect=observe) as checked:
                    self.node.control()
                if assist and (not recovering) and (raw[0] > 0.02):
                    self.assertEqual(checked.call_count, 1)
                    np.testing.assert_allclose(checked.call_args.args[0], raw)
                    self.assertGreater(reserved[0][0], 0.0)
                    self.assertLess(reserved[0][0], raw[0])
                    self.assertAlmostEqual(reserved[0][1] / reserved[0][0], raw[1] / raw[0])
                    from smart_wheelchair_safety.unified_geometry import braking_clear
                    self.assertTrue(braking_clear(points, self.node.output, self.node.measured, margin=self.node.parameter('hard_margin'), reaction=self.node.parameter('reaction_time'), deceleration=self.node.parameter('braking_deceleration'), state_age=0.0))
                else:
                    checked.assert_not_called()
                if not assist:
                    np.testing.assert_allclose(self.node.output, raw)

    def test_override_reserve_prevents_acceleration_into_later_guard_cut(self):
        from smart_wheelchair_safety.unified_geometry import arc_path, transform_points
        obstacles = np.column_stack((np.linspace(-2.0, 10.0, 161), np.full(161, 0.52)))
        self.node.mode = 'override'
        self.node.override = True
        self.node.raw = self.node.planned = np.array([0.8, 0.0])
        self.node.pose = (0.0, 0.0, 0.0)
        initial_stamp = rospy.Time.now().to_sec()
        previous_acceleration = np.zeros(2)
        for step in range(160):
            stamp = initial_stamp + 0.05 * step
            now = time.monotonic()
            self.node.raw_time = self.node.odom_time = self.node.plan_time = now
            self.node.odom_stamp = stamp - (0.0, 0.04, 0.09)[step % 3]
            self.node.scans = {side: (now, obstacles) for side in ('left', 'right')}
            self.node.last_tick = stamp - 0.05
            previous = self.node.output.copy()
            with patch.object(rospy.Time, 'now', return_value=rospy.Time.from_sec(stamp)):
                self.node.control()
            self.assertEqual(self.node.reason, 'clear', (step, self.node.reason, self.node.output))
            self.assertGreater(self.node.output[0], 0.0)
            actual_acceleration = (self.node.output - previous) / 0.05
            self.assertTrue(np.all(np.abs(actual_acceleration) <= [0.500001, 0.800001]))
            self.assertTrue(np.all(np.abs(actual_acceleration - previous_acceleration) <= np.array([2.5, 4.0]) * 0.05 + 1e-06), (step, actual_acceleration, previous_acceleration))
            previous_acceleration = actual_acceleration
            local = arc_path(*self.node.output, duration=0.05, steps=2)[-1]
            xy = transform_points([local[:2]], self.node.pose)[0]
            self.node.pose = (*xy, self.node.pose[2] + local[2])
            self.node.measured = self.node.output.copy()
        self.assertGreater(self.node.pose[0], 1.0)

    def setup_capture(self, velocity):
        from smart_wheelchair_safety.unified_geometry import Opening
        self.node.raw = np.array([1.6666667, 0.63])
        self.node.output = np.array(velocity)
        self.node.confirmed_openings = [Opening((2.2982257435645965, 0.5269919203496637), 1.236335296800013, 1.1227339375132526)]
        self.node.mode_neutral_seen = True
        self.node.assist_enabled = True

    def route(self, door, points):
        from smart_wheelchair_safety.unified_geometry import arc_path
        self.node.door_path_feasible = True
        return arc_path(0.2, 0.0)

    def test_moving_raw_only_target_does_not_commit(self):
        self.setup_capture([0.6998741735441372, 0.029708163281233484])
        self.assertIsNotNone(self.node._intended_door()[0])
        with patch.object(self.node, '_local_door_path', side_effect=self.route) as plan:
            self.node.update_reference()
        self.assertIsNone(self.node.door)
        plan.assert_not_called()

    def test_stationary_initial_capture_is_unchanged(self):
        for velocity in ([0.0, 0.0], [0.1, 0.0]):
            self.setup_capture(velocity)
            self.node.door = None
            with patch.object(self.node, '_local_door_path', side_effect=self.route):
                self.node.update_reference()
            self.assertIsNotNone(self.node.door)

    def test_already_committed_door_is_not_cancelled(self):
        self.setup_capture([0.6998741735441372, 0.029708163281233484])
        self.node.door = self.node.confirmed_openings[0]
        self.node.door_phase = 'door_align'
        with patch.object(self.node, '_local_door_path', side_effect=self.route):
            self.node.update_reference()
        self.assertIsNotNone(self.node.door)

    def test_straight_chain_captures_next_door_while_moving(self):
        from smart_wheelchair_safety.unified_geometry import Opening
        self.node.mode_neutral_seen = True
        self.node.assist_enabled = True
        first, second = (Opening((1.3, 0.0), 0.0, 1.1), Opening((4.0, 0.0), 0.0, 1.1))
        self.node.door = first
        self.node.door_phase = 'door_clear'
        self.node.confirmed_openings = [first, second]
        self.node.pose = (1.62, 0.0, 0.0)
        self.node.raw = np.array([2.0 / 3.0, 0.0])
        self.node.output = np.array([0.3, 0.0])
        with patch.object(self.node, '_local_door_path', side_effect=self.route):
            self.node.update_reference()
        self.assertEqual(self.node.door, second)

    def setup_loss(self, side=True):
        from smart_wheelchair_safety import unified_control_node as node
        from geometry_msgs.msg import Twist, TwistStamped
        self.node.override = True
        self.node.mode = 'override'
        self.node.wall_side = 0
        self.node.output = np.array([0.6, -0.42])
        self.node.measured = self.node.output.copy()
        self.node.cancel()
        self.node.acceleration = np.zeros(2)
        self.statuses = []
        self.node.status = SimpleNamespace(publish=lambda m: self.statuses.append(json.loads(m.data)))
        msg = Twist()
        msg.linear.x = 0.83333335
        msg.angular.z = -0.42
        front = (None, None) if side else (node.Opening((2.0, -0.4), 0.0, 1.1), node.Opening((2.0, -0.4), 0.0, 1.1))
        with patch.object(self.node, '_intended_door', return_value=front), patch.object(node, 'intended_side_opening', return_value=node.Opening((2.0, -1.3), -1.57, 2.7)):
            self.node.on_raw(msg)
        self.assertFalse(self.node.override)

    def tick(self):
        from smart_wheelchair_safety import unified_control_node as node
        from geometry_msgs.msg import Twist, TwistStamped
        self.node.last_tick = rospy.Time.now().to_sec() - 0.05
        with patch.object(node, 'braking_clear', return_value=True):
            self.node.control()

    def test_side_or_front_intent_does_not_drop_live_override_into_timeout(self):
        from smart_wheelchair_safety import unified_control_node as node
        from geometry_msgs.msg import Twist, TwistStamped
        for side in (True, False):
            self.setup_loss(side)
            self.tick()
            self.assertEqual(self.node.reason, 'clear')
            self.assertEqual(self.statuses[-1]['planner_source'], 'joystick')
            self.assertGreater(self.node.output[0], 0.5)

    def test_new_mode_waits_for_matching_fresh_action_then_hands_off(self):
        from smart_wheelchair_safety import unified_control_node as node
        from geometry_msgs.msg import Twist, TwistStamped
        self.setup_loss()
        self.node.mode = 'opening_turn'
        self.tick()
        self.assertEqual(self.statuses[-1]['planner_source'], 'joystick')
        self.node.accept_planned = True
        self.node.active_reference_stamp = rospy.Time.now()
        action = TwistStamped()
        action.header.stamp = self.node.active_reference_stamp - rospy.Duration(0.01)
        action.twist.linear.x = 0.4
        action.twist.angular.z = -0.1
        self.node.on_planned(action)
        self.tick()
        self.assertEqual(self.statuses[-1]['planner_source'], 'joystick')
        action.header.stamp = self.node.active_reference_stamp
        self.node.on_planned(action)
        self.tick()
        self.assertEqual(self.statuses[-1]['planner_source'], 'local_follower')
        self.assertFalse(self.node.override)

    def test_neutral_or_stale_input_still_stops(self):
        from smart_wheelchair_safety import unified_control_node as node
        from geometry_msgs.msg import Twist, TwistStamped
        self.setup_loss()
        self.node.on_raw(Twist())
        self.tick()
        self.assertEqual(self.node.reason, 'user_stop')
        np.testing.assert_array_equal(self.node.output, [0.0, 0.0])
        self.setup_loss()
        self.node.raw_time = time.monotonic() - 1.0
        self.tick()
        self.assertEqual(self.node.reason, 'stale_input')
        np.testing.assert_array_equal(self.node.output, [0.0, 0.0])

    def test_door_capture_precheck_rejects_current_and_wait_independently(self):
        from smart_wheelchair_safety import unified_control_node as module
        self.node.output = np.array([.3, .1])
        for results, count in (([False], 1), ([True, False], 2), ([True, True], None)):
            answers = iter(results)
            with patch.object(module, 'braking_clear', side_effect=lambda *a, **kw: next(answers, True)) as guard:
                self.assertEqual(self.node._door_capture_is_clear(np.array([[3., 2.]])), all(results))
            if count is None:
                self.assertGreater(guard.call_count, 2)
            else:
                self.assertEqual(guard.call_count, count)
            for call in guard.call_args_list:
                self.assertEqual(call.kwargs['margin'], self.node.parameter('hard_margin'))
                self.assertGreaterEqual(call.kwargs['state_age'], self.node.parameter('odom_timeout'))

    def test_door_capture_wait_preview_matches_control_dt_without_state_mutation(self):
        from smart_wheelchair_safety import unified_control_node as module
        self.node.output = np.array([.3, .12])
        self.node.acceleration = np.array([.1, -.3])
        stamp = rospy.Time.from_sec(100.)
        for elapsed in (-.1, .05, .3):
            self.node.last_tick = stamp.to_sec()-elapsed
            dt = min(.1, max(.001, elapsed))
            previous = self.node.output.copy(), self.node.acceleration.copy(), self.node.last_tick
            expected, _ = self.node._door_wait_slew(dt)
            with patch.object(rospy.Time, 'now', return_value=stamp), patch.object(module, 'braking_clear', return_value=True) as guard:
                self.assertTrue(self.node._door_capture_is_clear(np.array([[3., 2.]])))
            np.testing.assert_allclose(guard.call_args_list[1].args[1], expected, atol=1e-9)
            np.testing.assert_array_equal(self.node.output, previous[0])
            np.testing.assert_array_equal(self.node.acceleration, previous[1])
            self.assertEqual(self.node.last_tick, previous[2])

    def test_unsafe_capture_defers_without_pending_manual_or_door_submission(self):
        from smart_wheelchair_safety import unified_control_node as module
        self.node.mode_neutral_seen = True
        self.node.output = np.array([.3, 0.])
        self.node.confirmed_openings = [module.Opening((2., 0.), 0., 1.1)]
        with patch.object(self.node, '_door_capture_is_clear', return_value=False), patch.object(self.node, '_pending_door_intent', return_value=True) as pending, patch.object(module, 'wall_reference', return_value=(module.arc_path(.3, 0.), 'wall', 1)), patch.object(self.node, '_local_door_path') as plan:
            self.node.update_reference()
        self.assertIsNone(self.node.door)
        self.assertEqual(self.node.mode, 'manual')
        pending.assert_not_called()
        plan.assert_not_called()

    def test_safe_capture_and_existing_commit_preserve_door_planning(self):
        from smart_wheelchair_safety import unified_control_node as module
        self.node.mode_neutral_seen = True
        self.node.output = np.array([.3, 0.])
        door = module.Opening((2., 0.), 0., 1.1)
        self.node.confirmed_openings = [door]
        with patch.object(self.node, '_door_capture_is_clear', return_value=True) as precheck, patch.object(self.node, '_local_door_path', side_effect=self.route):
            self.node.update_reference()
        precheck.assert_called_once()
        self.assertEqual(self.node.door, door)
        with patch.object(self.node, '_door_capture_is_clear', side_effect=AssertionError('committed door rechecked')), patch.object(self.node, '_local_door_path', side_effect=self.route):
            self.node.update_reference()
        self.assertEqual(self.node.door, door)

    def test_recorded_door_capture_is_deferred_before_full_age_guard_cut(self):
        fixture = json.loads((Path(__file__).parent/'fixtures/door_capture_transition.json').read_text())
        self.node.output = np.array(fixture['output'])
        self.node.acceleration = np.array(fixture['acceleration'])
        self.node.measured = np.array(fixture['measured'])
        stamp = rospy.Time.from_sec(100.)
        self.node.odom_stamp = stamp.to_sec()-.016
        self.node.last_tick = stamp.to_sec()-.05
        with patch.object(rospy.Time, 'now', return_value=stamp):
            self.assertFalse(self.node._door_capture_is_clear(np.array(fixture['points'])))

    def test_capture_precheck_does_not_change_wait_slew_or_jerk(self):
        from smart_wheelchair_safety import unified_control_node as module
        fixture = json.loads((Path(__file__).parent/'fixtures/door_capture_transition.json').read_text())
        self.node.output = np.array(fixture['output'])
        self.node.acceleration = np.array(fixture['acceleration'])
        self.node.measured = np.array(fixture['measured'])
        self.node.mode = 'door_wait'
        stamp = rospy.Time.from_sec(100.)
        self.node.last_tick = stamp.to_sec()-.05
        self.node.mode_neutral_seen = True
        with patch.object(rospy.Time, 'now', return_value=stamp), patch.object(self.node, 'fresh', return_value=True), patch.object(self.node, '_near_recovery', return_value=False), patch.object(module, 'braking_clear', return_value=True):
            self.node.control()
        np.testing.assert_allclose(self.node.output, [.7681979861376403, .25480152921241546], atol=1e-8)
        self.assertLessEqual(np.max(np.abs((self.node.acceleration-fixture['acceleration'])/.05)/module.JERK_LIMITS), 1.+1e-8)

    def prepare_capture_case(self, output=.7, measured=.7):
        from smart_wheelchair_safety import unified_control_node as module
        self.node.mode_neutral_seen = True
        self.node.raw = np.array([.8, 0.])
        self.node.output = np.array([output, 0.])
        self.node.measured = np.array([measured, 0.])
        self.node.confirmed_openings = [module.Opening((2., 0.), 0., 1.1)]

    def test_high_speed_capture_prepares_before_changing_door_policy(self):
        from smart_wheelchair_safety import unified_control_node as module
        self.prepare_capture_case()
        speeds=[]
        self.node.reference_speed_pub=SimpleNamespace(publish=speeds.append)
        with patch.object(self.node, '_door_capture_is_clear', return_value=True), patch.object(self.node, '_pending_door_intent', return_value=True) as pending, patch.object(module, 'wall_reference', return_value=(module.arc_path(.8,0.),'wall',1)), patch.object(self.node, '_local_door_path') as plan:
            self.node.update_reference()
        self.assertIsNone(self.node.door)
        self.assertIsNotNone(self.node.pending_door_capture)
        self.assertEqual(self.node.mode,'manual')
        self.assertEqual(speeds[-1].twist.linear.x,.35)
        plan.assert_not_called();pending.assert_not_called()

    def test_capture_requires_output_measured_and_existing_admission(self):
        from smart_wheelchair_safety import unified_control_node as module
        for output, measured, clear, expected in ((.36,.3,True,False),(.3,.36,True,False),(.3,-.36,True,False),(.35,.35,False,False),(.35,.35,True,True)):
            self.node._clear_door();self.prepare_capture_case(output,measured)
            with patch.object(self.node,'_door_capture_is_clear',return_value=clear), patch.object(module,'wall_reference',return_value=(module.arc_path(.3,0.),'wall',1)), patch.object(self.node,'_local_door_path',side_effect=self.route):
                self.node.update_reference()
            self.assertEqual(self.node.door is not None, expected)
            self.assertEqual(self.node.pending_door_capture is None, expected)

    def test_pending_capture_caps_live_target_proportionally_before_slew(self):
        from smart_wheelchair_safety import unified_control_node as module
        self.prepare_capture_case()
        self.node.pending_door_capture=self.node.confirmed_openings[0]
        self.node.planned=np.array([.8,.4]);self.node.plan_time=time.monotonic()
        self.node.accept_planned=True
        with patch.object(self.node,'_safe_assisted_command',side_effect=lambda desired,points:desired) as reserve, patch.object(module,'braking_clear',return_value=True), patch.object(self.node,'_near_recovery',return_value=False):
            self.node.control()
        np.testing.assert_allclose(reserve.call_args.args[0],[.35,.175])

    def test_pending_capture_clears_on_changed_or_cancelled_intent(self):
        from geometry_msgs.msg import Twist
        from smart_wheelchair_safety import unified_control_node as module
        for v,w in ((0.,0.),(-.2,0.),(0.,.4),(.8,1.),(float('nan'),0.)):
            self.prepare_capture_case();self.node.pending_door_capture=self.node.confirmed_openings[0]
            message=Twist();message.linear.x=v;message.angular.z=w
            self.node.on_raw(message)
            self.assertIsNone(self.node.pending_door_capture,(v,w))
        self.prepare_capture_case();self.node.pending_door_capture=self.node.confirmed_openings[0]
        self.node.confirmed_openings=[]
        with patch.object(module,'wall_reference',return_value=(module.arc_path(.3,0.),'wall',1)):
            self.node.update_reference()
        self.assertIsNone(self.node.pending_door_capture)

    def test_pending_capture_clears_on_disable_and_sensor_loss(self):
        from std_msgs.msg import Bool
        self.prepare_capture_case();self.node.pending_door_capture=self.node.confirmed_openings[0]
        self.node.on_assist_enabled(Bool(data=False))
        self.assertIsNone(self.node.pending_door_capture)
        self.node.assist_enabled=True;self.prepare_capture_case();self.node.pending_door_capture=self.node.confirmed_openings[0]
        with patch.object(self.node,'fresh',return_value=False):self.node.update_reference()
        self.assertIsNone(self.node.pending_door_capture)

    def test_pending_same_door_survives_temporary_output_arc_mismatch(self):
        from smart_wheelchair_safety import unified_control_node as module
        self.setup_capture([.6998741735441372,.029708163281233484])
        self.node.pending_door_capture=self.node.confirmed_openings[0]
        self.assertIsNotNone(self.node._intended_door()[0])
        self.assertFalse(module.output_arc_targets_aperture(self.node._intended_door()[1],*self.node.output))
        speeds=[];self.node.reference_speed_pub=SimpleNamespace(publish=speeds.append)
        with patch.object(module,'wall_reference',return_value=(module.arc_path(.3,0.),'wall',1)), patch.object(self.node,'_local_door_path') as plan, patch.object(self.node,'_pending_door_intent',return_value=True) as pending:
            self.node.update_reference()
        self.assertIsNone(self.node.door)
        self.assertIsNotNone(self.node.pending_door_capture)
        self.assertEqual(speeds[-1].twist.linear.x,.35)
        self.assertEqual(self.node.mode,'manual')
        plan.assert_not_called();pending.assert_not_called()

    def test_repeated_preparation_slew_deadband_can_commit_at_target_speed(self):
        from smart_wheelchair_safety import unified_control_node as module
        self.prepare_capture_case(.7242894059802214,.7242894059802214)
        self.node.acceleration=np.array([.006543288018758905,0.])
        self.node.pending_door_capture=self.node.confirmed_openings[0]
        self.node.planned=np.array([.8,0.]);self.node.accept_planned=True
        with patch.object(self.node,'fresh',return_value=True), patch.object(self.node,'_near_recovery',return_value=False), patch.object(self.node,'_safe_assisted_command',side_effect=lambda desired,points:desired), patch.object(module,'braking_clear',return_value=True):
            for tick in range(120):
                stamp=rospy.Time.from_sec(100.+.05*tick)
                self.node.last_tick=stamp.to_sec()-.05
                self.node.plan_time=time.monotonic()
                self.node.measured=self.node.output.copy()
                with patch.object(rospy.Time,'now',return_value=stamp):self.node.control()
            self.assertAlmostEqual(self.node.output[0],.35,places=7)
            self.assertLessEqual(self.node.output[0],.35+2.5*.1**2/8)
            speed=self.node.output[0]
            self.node.measured=self.node.output.copy()
            with patch.object(self.node,'_door_capture_is_clear',return_value=True), patch.object(self.node,'_local_door_path',side_effect=self.route):
                self.node.update_reference()
        self.assertIsNotNone(self.node.door,('deadband leaves preparation pending',speed))
        self.assertIsNone(self.node.pending_door_capture)

    def test_capture_deadband_tolerance_is_local_and_applies_to_measured_speed(self):
        from smart_wheelchair_safety import unified_control_node as module
        for output,measured,expected in ((.3525,.35,True),(.35,.3525,True),(.3532,.35,False),(.35,.3532,False),(.36,.35,False),(.35,.36,False)):
            self.node._clear_door();self.prepare_capture_case(output,measured)
            speeds=[];self.node.reference_speed_pub=SimpleNamespace(publish=speeds.append)
            with patch.object(self.node,'_door_capture_is_clear',return_value=True), patch.object(self.node,'_local_door_path',side_effect=self.route):
                self.node.update_reference()
            self.assertEqual(self.node.door is not None,expected,(output,measured))
            self.assertLessEqual(speeds[-1].twist.linear.x,.35)

    def setup_recorded_pivot_residual(self):
        from smart_wheelchair_safety import unified_control_node as module
        self.node.pose=(0.,0.,-.037690788)
        self.node.door=module.Opening((1.3,0.),0.,1.)
        self.node.door_path=np.array([[0.,0.,-.2],[0.,0.,0.],[3.,0.,0.]])
        self.node.door_rotation_index=0;self.node.door_path_feasible=True
        self.node.mode='door_align';self.node.door_phase='door_align'
        self.node.mode_neutral_seen=True
        self.node.output=np.array([.0007228159758065882,.06182497537819114])
        self.node.measured=np.array([.0006752688579569484,.05672805003368639])
        self.node.acceleration=np.array([0.,(.06182497537819114-.06601934833133001)/.05])

    def test_recorded_pivot_completion_does_not_overwrite_angular_slew(self):
        self.setup_recorded_pivot_residual()
        previous=self.node.output.copy();old_a=self.node.acceleration.copy()
        stamp=rospy.Time.from_sec(100.);self.node.last_tick=99.95;self.node.odom_stamp=100.
        with patch.object(rospy.Time,'now',return_value=stamp),patch.object(self.node,'fresh',return_value=True):self.node.control()
        actual_a=(self.node.output-previous)/.05
        self.assertLessEqual(abs(actual_a[1]),.8+1e-9)
        self.assertLessEqual(abs(actual_a[1]-old_a[1])/.05,4.+1e-8)
        self.assertEqual(self.node.door_rotation_index,0)

    def test_pivot_inside_angle_tolerance_requests_stop_before_release(self):
        self.setup_recorded_pivot_residual()
        command=self.node._safe_door_command(np.array([[5.,2.]]),100.)
        np.testing.assert_array_equal(command,[0.,0.])
        self.assertEqual(self.node.door_rotation_index,0)
        self.node.output[:]=0.;self.node.acceleration[:]=0.
        self.assertIsNotNone(self.node._pending_door_rotation())
        self.node.measured[:]=0.
        self.assertIsNone(self.node._pending_door_rotation())

    def test_pivot_settles_through_deadband_then_releases_without_jerk_jump(self):
        for dt in (.05,.1):
            self.setup_recorded_pivot_residual()
            self.node.output[0]=0.;self.node.measured[0]=0.
            previous_actual_a=self.node.acceleration.copy()
            for step in range(80):
                stamp=rospy.Time.from_sec(100.+step*dt)
                self.node.last_tick=stamp.to_sec()-dt;self.node.odom_stamp=stamp.to_sec()
                previous=self.node.output.copy();old_a=self.node.acceleration.copy()
                with patch.object(rospy.Time,'now',return_value=stamp),patch.object(self.node,'fresh',return_value=True):self.node.control()
                actual_a=(self.node.output-previous)/dt
                self.assertLessEqual(abs(actual_a[1]),.8+1e-8)
                self.assertLessEqual(abs(actual_a[1]-previous_actual_a[1])/dt,4.+1e-7)
                previous_actual_a=actual_a.copy()
                self.node.pose=(0.,0.,self.node.pose[2]+self.node.output[1]*dt)
                self.node.measured=self.node.output.copy()
                if self.node.door_rotation_index>0:break
            self.assertGreater(self.node.door_rotation_index,0,('pivot never settled',dt,self.node.output,self.node.acceleration))

    def test_pivot_zero_snap_reserves_jerk_for_next_stationary_tick(self):
        self.setup_recorded_pivot_residual()
        self.node.output=np.array([0.,.012])
        self.node.measured=self.node.output.copy()
        self.node.acceleration=np.array([0.,-.39])
        previous_actual_a=-.39
        for step in range(8):
            stamp=rospy.Time.from_sec(100.+step*.05)
            self.node.last_tick=stamp.to_sec()-.05;self.node.odom_stamp=stamp.to_sec()
            previous_w=self.node.output[1]
            with patch.object(rospy.Time,'now',return_value=stamp),patch.object(self.node,'fresh',return_value=True):self.node.control()
            actual_a=(self.node.output[1]-previous_w)/.05
            self.assertLessEqual(abs(actual_a-previous_actual_a)/.05,4.+1e-7)
            if step==0:self.assertGreater(self.node.output[1],0.,'premature snap makes following stationary step exceed jerk')
            previous_actual_a=actual_a
            self.node.measured=self.node.output.copy()
            if self.node.door_rotation_index>0:break
        self.assertGreater(self.node.door_rotation_index,0)

    def test_pending_door_reference_follows_live_intent_instead_of_old_wall(self):
        from smart_wheelchair_safety import unified_control_node as module
        self.prepare_capture_case();self.node.raw=np.array([.8,-.1])
        paths=[];self.node.reference=SimpleNamespace(publish=paths.append)
        with patch.object(module,'wall_reference',return_value=(module.arc_path(.8,.4),'wall',1)):
            self.node.update_reference()
        self.assertIsNotNone(self.node.pending_door_capture)
        self.assertIsNone(self.node.door)
        self.assertEqual(self.node.mode,'manual')
        self.assertLess(paths[-1].poses[-1].pose.position.y,0.)
        self.assertEqual(paths[-1].header.stamp,self.node.active_reference_stamp)

    def test_recorded_wait_entry_and_zero_finish_preserve_actual_acceleration_and_jerk(self):
        from smart_wheelchair_safety import unified_control_node as module
        for mirror in (-1.,1.):
            self.node.mode='door_wait';self.node.mode_neutral_seen=True
            self.node.output=np.array([.3310103304408235,mirror*-.15232966320182093])
            self.node.acceleration=np.array([(.3310103304408235-.32786780926653847)/.05,mirror*(-.15232966320182093+.13996787786178597)/.05])
            previous_a=self.node.acceleration.copy()
            for tick in range(60):
                stamp=rospy.Time.from_sec(100.+tick*.05)
                self.node.last_tick=stamp.to_sec()-.05;self.node.odom_stamp=stamp.to_sec()
                self.node.measured=self.node.output.copy();previous=self.node.output.copy()
                with patch.object(rospy.Time,'now',return_value=stamp),patch.object(self.node,'fresh',return_value=True),patch.object(self.node,'_near_recovery',return_value=False):self.node.control()
                actual_a=(self.node.output-previous)/.05
                self.assertTrue(np.all(np.abs(actual_a)<=[.5+1e-8,.8+1e-8]),actual_a)
                self.assertTrue(np.all(np.abs(actual_a-previous_a)/.05<=module.JERK_LIMITS+1e-7),(tick,actual_a,previous_a,self.node.output))
                self.assertGreaterEqual(self.node.output[0],0.)
                self.assertLessEqual(mirror*self.node.output[1],0.)
                previous_a=actual_a
            np.testing.assert_array_equal(self.node.output,[0.,0.])

    def test_wait_reachable_stop_survives_short_following_tick_without_reverse(self):
        from smart_wheelchair_safety import unified_control_node as module
        for initial in ([.3,0.], [.3,.3], [.3,-.3]):
            self.node.output=np.array(initial)
            self.node.acceleration=np.zeros(2)
            previous_a=np.zeros(2)
            for tick,dt in enumerate([.05]*16+[.001]+[.05]*80):
                previous=self.node.output.copy()
                command,acceleration=self.node._door_wait_slew(dt)
                actual_a=(command-previous)/dt
                self.assertTrue(np.all(np.abs(actual_a-previous_a)<=module.JERK_LIMITS*dt+1e-8),
                                (initial,tick,dt,previous,command,previous_a,actual_a))
                self.assertGreaterEqual(command[0],0.)
                self.assertGreaterEqual(command[1]*initial[1],0.)
                if initial[1]==0.:self.assertEqual(command[1],0.)
                self.node.output,self.node.acceleration=command,acceleration
                previous_a=actual_a
            np.testing.assert_array_equal(self.node.output,[0.,0.])

    def test_wait_and_pivot_variable_ticks_preserve_actual_derivatives_to_zero(self):
        from smart_wheelchair_safety import unified_control_node as module
        for pivot in (False,True):
            for mirror in (-1.,1.):
                if pivot:
                    self.setup_recorded_pivot_residual()
                    self.node.output[1]*=mirror
                    self.node.acceleration[1]*=mirror
                    self.node.pose=(0.,0.,-mirror*.02)
                    # Keep observed angular motion until after the zero-step
                    # checks so the normal tracking phase cannot mask settling.
                else:
                    self.node.mode='door_wait';self.node.mode_neutral_seen=True
                    self.node.door=None;self.node.door_path=None
                    self.node.output=np.array([.3,mirror*.3]);self.node.acceleration=np.zeros(2)
                previous_a=self.node.acceleration.copy()
                initial=self.node.output.copy()
                for tick in range(180):
                    dt=(.1,.001,.025,.05)[tick%4]
                    stamp=rospy.Time.from_sec(100.)
                    self.node.last_tick=100.-dt;self.node.odom_stamp=100.
                    self.node.measured=self.node.output.copy()
                    if pivot:self.node.measured[1]=mirror*.02
                    previous=self.node.output.copy()
                    with patch.object(rospy.Time,'now',return_value=stamp),patch.object(self.node,'fresh',return_value=True),patch.object(self.node,'_near_recovery',return_value=False):
                        self.node.control()
                    actual_a=(self.node.output-previous)/dt
                    self.assertTrue(np.all(np.abs(actual_a)<=np.array([.5,.8])+1e-8))
                    self.assertTrue(np.all(np.abs(actual_a-previous_a)<=module.JERK_LIMITS*dt+1e-7),
                                    (pivot,mirror,tick,dt,previous,self.node.output,previous_a,actual_a))
                    self.assertTrue(np.all(self.node.output*initial>=0.))
                    previous_a=actual_a
                np.testing.assert_array_equal(self.node.output,[0.,0.])

    def test_wait_preview_and_actual_control_share_the_same_slew(self):
        self.node.output=np.array([.3310103304408235,-.15232966320182093])
        self.node.acceleration=np.array([.0628504234857,-.2472357068007])
        before=self.node.output.copy(),self.node.acceleration.copy()
        command,acceleration=self.node._door_wait_slew(.05)
        np.testing.assert_array_equal(self.node.output,before[0]);np.testing.assert_array_equal(self.node.acceleration,before[1])
        self.node.mode='door_wait';self.node.mode_neutral_seen=True;self.node.last_tick=99.95;self.node.odom_stamp=100.
        with patch.object(rospy.Time,'now',return_value=rospy.Time.from_sec(100.)),patch.object(self.node,'fresh',return_value=True),patch.object(self.node,'_near_recovery',return_value=False):self.node.control()
        np.testing.assert_allclose(self.node.output,command,atol=1e-10)
        np.testing.assert_allclose(self.node.acceleration,acceleration,atol=1e-10)

    def test_door_capture_rejects_jerk_unreachable_zero_crossing(self):
        from smart_wheelchair_safety import unified_control_node as module
        self.node.output=np.array([.2,.001]);self.node.acceleration=np.array([0.,-.8])
        self.node.last_tick=99.95;self.node.odom_stamp=100.
        command,acceleration=self.node._door_wait_slew(.05)
        self.assertEqual(command[1],0.)
        self.assertGreater(abs(acceleration[1]-self.node.acceleration[1]),4.*.05)
        with patch.object(rospy.Time,'now',return_value=rospy.Time.from_sec(100.)),patch.object(module,'braking_clear',return_value=True):
            self.assertFalse(self.node._door_capture_is_clear(np.array([[5.,2.]])))

if __name__ == '__main__':
    import rostest
    rostest.rosrun('smart_wheelchair_safety', 'unified_control', UnifiedNodeTest)
