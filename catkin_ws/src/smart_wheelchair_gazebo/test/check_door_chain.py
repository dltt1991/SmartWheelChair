#!/usr/bin/env python3
"""Isolated M6 simulation only: two aligned doors followed by a turn along the front wall."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import time

from check_door_matrix import overlaps, send_command, wait_for_controller, motion_metrics
from probe_unified_control import (settled_after_release, record_control_status,
                                   control_event_window, control_event_metrics)
from test_m6_accessibility import M6AccessibilityTest


def chain_passed(result):
    return bool(result['raw_seen'] and not result['collision']
                and not result['stopped_samples_between'] and result['crossed_second']
                and result['turned_at_wall'] and result['released_stopped']
                and (not result['neupan_enabled'] or result['neupan_used']))


def neupan_forward_events(events, released):
    return sum(event['monotonic'] < released
               and event['status'].get('mode') == 'wall'
               and event['status'].get('planner_source') == 'neupan'
               and event['status'].get('v', 0.) > 0. for event in events)


def wall_turn_observed(row):
    yaw_change = row['pose'][2]+math.pi/2
    yaw_change = math.atan2(math.sin(yaw_change), math.cos(yaw_change))
    return bool(row['pose'][1] < -4. and abs(yaw_change) > 1.2
                and row.get('status', {}).get('mode') == 'wall'
                and row['measured'][0] > .025)


def stopped(data, after, now):
    return settled_after_release(dict(cmd_vel=data.get('cmd'),
                                     cmd_vel_received=data.get('cmd_received', 0.),
                                     odom_velocity=data.get('measured'),
                                     odom_velocity_received=data.get('odom_received', 0.)), after, now)


def main():
    import rospy, rosnode
    from gazebo_msgs.msg import ModelState
    from gazebo_msgs.srv import SetModelState
    from nav_msgs.msg import Odometry
    from std_msgs.msg import String
    from geometry_msgs.msg import Twist
    from sensor_msgs.msg import LaserScan
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--timeout', type=float, default=60.,
                        help='Motion observation limit in seconds (door pass speed is 0.15 m/s)')
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0.:
        parser.error('--timeout must be positive and finite')
    rospy.init_node('door_chain_check', disable_signals=True)
    data = {}
    status_events = []
    def odom(msg):
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        data.update(pose=[p.x, p.y, 2*math.atan2(q.z, q.w)],
                    measured=[msg.twist.twist.linear.x, msg.twist.twist.angular.z],
                    stamp=time.monotonic(), odom_received=time.monotonic())
    rospy.Subscriber('/odom', Odometry, odom, queue_size=100)
    rospy.Subscriber('/shared_control/status', String,
                     lambda msg: record_control_status(data, status_events, json.loads(msg.data)), queue_size=100)
    rospy.Subscriber('/cmd_vel', Twist, lambda m: data.update(
        cmd=[m.linear.x, m.angular.z], cmd_received=time.monotonic()))
    rospy.Subscriber('/cmd_vel_raw', Twist, lambda m: data.update(
        cmd_vel_raw=[m.linear.x, m.angular.z], raw_received=time.monotonic()))
    for side in ('left', 'right'):
        rospy.Subscriber('/scan_'+side, LaserScan, lambda m, side=side:
                         data.update({side+'_received': time.monotonic()}), queue_size=1)
    def command(v=0.):
        send_command('door-chain', y=v)
    command(0.)
    controller_names = ['/unified_control_node', '/local_path_follower_node']
    rosnode.kill_nodes(controller_names)
    deadline = time.monotonic()+10.
    while any(name in rosnode.get_node_names() for name in controller_names):
        if time.monotonic() >= deadline:
            raise RuntimeError('previous controllers did not release ownership')
        time.sleep(.05)
    rospy.wait_for_service('/gazebo/set_model_state', timeout=10)
    state = ModelState(model_name='smart_wheelchair', reference_frame='world')
    state.pose.position.x, state.pose.position.y, state.pose.position.z = -4.5, 2.37, .04
    state.pose.orientation.z, state.pose.orientation.w = math.sin(-math.pi/4), math.cos(-math.pi/4)
    assert rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)(state).success
    neupan_enabled = bool(rospy.get_param('/unified_control_node/use_neupan', False))
    children = [subprocess.Popen(['rosrun', 'smart_wheelchair_safety', name, '__name:='+name],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                for name in ('unified_control_node', 'local_path_follower_node')]
    geometry = M6AccessibilityTest()
    geometry.setUp()
    rows = []
    hit = None
    turned_at_wall = released_stopped = False
    try:
        wait_for_controller(data, command)
        status_events.clear()
        start = time.monotonic()
        while time.monotonic()-start < args.timeout:
            command(.4)
            time.sleep(.08)
            if 'pose' not in data:
                continue
            row = dict(data, t=time.monotonic()-start)
            rows.append(row)
            hit = overlaps(row['pose'], geometry.boxes, .02)
            if hit:
                break
            if wall_turn_observed(row):
                turned_at_wall = True
                break
    finally:
        command(0.)
        released = time.monotonic()
        deadline = released+3.
        while time.monotonic() < deadline:
            if stopped(data, released, time.monotonic()):
                released_stopped = True
                break
            time.sleep(.05)
        settled_at = time.monotonic()
        for child in children:
            child.terminate()
        for child in children:
            child.wait(timeout=10)
    between = [r for r in rows if -1.7 < r['pose'][1] < 1.]
    pauses = [r for r in between if abs(r['measured'][0]) < .025]
    raw_seen = any(r.get('raw_received', 0.) > start and any(r.get('cmd_vel_raw', [])) for r in rows)
    result = dict(raw_seen=raw_seen, motion_metrics=motion_metrics(rows), collision=hit, minimum_between_speed=min((r['measured'][0] for r in between), default=0.),
                  stopped_samples_between=len(pauses), final_pose=data.get('pose'),
                  modes=list(dict.fromkeys(r.get('status', {}).get('mode') for r in rows)),
                  crossed_second=any(r['pose'][1] < -1.7 for r in rows),
                  turned_at_wall=turned_at_wall, released_stopped=released_stopped, timeout=args.timeout)
    control_events = control_event_window(status_events, start, settled_at)
    neupan_count = neupan_forward_events(control_events, released)
    result.update(neupan_enabled=neupan_enabled, neupan_used=neupan_count > 0,
                  neupan_wall_forward_events=neupan_count)
    result['control_event_window'] = dict(started=start, released=released, ended=settled_at)
    result['control_event_metrics'] = control_event_metrics(control_events)
    result['active_control_event_metrics'] = control_event_metrics(
        [event for event in control_events if event['monotonic'] < released])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(summary=result, records=rows, control_events=control_events), indent=2))
    print(json.dumps(result), flush=True)
    return 0 if chain_passed(result) else 1


if __name__ == '__main__':
    raise SystemExit(main())
