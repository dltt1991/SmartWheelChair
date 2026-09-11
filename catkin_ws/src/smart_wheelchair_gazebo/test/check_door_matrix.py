#!/usr/bin/env python3
"""Exercise every real M6 doorway with fresh controllers and measured body sweeps.

Run in a Noetic container with m6_room already launched and no other driver.
Success requires rear-body clearance, no body/scene overlap, and fresh stopping
telemetry. All attempted cases, including timeout/stall, enter the denominator.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time
import urllib.request

import numpy as np

from test_m6_accessibility import DOORS, M6AccessibilityTest


def start_pose(door, direction, angle):
    _, x, y, axis, _ = door
    heading = (math.copysign(math.pi/2, y) if axis == 0
               else (0. if x > 0 else math.pi))
    heading += math.pi if direction == 'out' else 0.
    normal = np.array([math.cos(heading), math.sin(heading)])
    tangent = np.array([-normal[1], normal[0]])
    # Aim the initial joystick straight ahead through the aperture centre.
    # 1.35 m fits the cross corridor as well as the room side of each door.
    axle = np.array([x, y])-1.35*normal-1.35*math.tan(math.radians(angle))*tangent
    yaw = heading+math.radians(angle)
    return axle, yaw, normal


def overlaps(pose, boxes, margin=0.):
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    bx, by, hx, hy = x+.36*c, y+.36*s, .61+margin, .4+margin
    for name, (x0, y0, x1, y1) in boxes.items():
        dx, dy = (x0+x1)/2-bx, (y0+y1)/2-by
        ex, ey = (x1-x0)/2, (y1-y0)/2
        if (abs(dx) <= ex+hx*abs(c)+hy*abs(s)
                and abs(dy) <= ey+hx*abs(s)+hy*abs(c)
                and abs(dx*c+dy*s) <= hx+ex*abs(c)+ey*abs(s)
                and abs(-dx*s+dy*c) <= hy+ex*abs(s)+ey*abs(c)):
            return name
    return None


def rear_extent(heading_error):
    cosine = math.cos(heading_error)
    return max(.25*cosine, -.97*cosine)+.4*abs(math.sin(heading_error))


def inspect_sweep(poses, boxes, door, normal):
    """Check measured sweeps and directed crossing of this finite aperture.

    Interpolation bounds translation plus the farthest corner's rotational
    travel to 5 mm per sample; the collision rectangle retains its 2 cm margin.
    """
    crossed = False
    center = np.asarray(door[1:3])
    tangent = np.array([-normal[1], normal[0]])
    for first, second in zip(poses, poses[1:]):
        start, end = np.asarray(first[:2]), np.asarray(second[:2])
        before, after = float((start-center)@normal), float((end-center)@normal)
        if before <= 0. < after:
            crossing = start + (-before/(after-before))*(end-start)
            crossed = crossed or abs(float((crossing-center)@tangent)) < door[4]/2
        yaw_delta = math.atan2(math.sin(second[2]-first[2]), math.cos(second[2]-first[2]))
        travel = np.linalg.norm(end-start)+math.hypot(.99, .42)*abs(yaw_delta)
        steps = max(1, math.ceil(travel/.005))
        for fraction in np.linspace(0., 1., steps+1):
            xy = start+fraction*(end-start)
            hit = overlaps((*xy, first[2]+fraction*yaw_delta), boxes, .02)
            if hit:
                return hit, crossed
    return None, crossed


def main():
    import rosnode
    import rospy
    from gazebo_msgs.msg import ModelState
    from gazebo_msgs.srv import SetModelState
    from geometry_msgs.msg import Twist
    from nav_msgs.msg import Odometry, Path as RosPath
    from sensor_msgs.msg import LaserScan
    from std_msgs.msg import String

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--doors', nargs='+', default=[d[0] for d in DOORS])
    parser.add_argument('--angles', nargs='+', type=float, default=[0, -15, 15, -30, 30, -45, 45])
    parser.add_argument('--directions', nargs='+', choices=['in', 'out'], default=['in', 'out'])
    parser.add_argument('--timeout', type=float, default=35.)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if set(args.doors)-{d[0] for d in DOORS}:
        parser.error('unknown door')
    args.output.mkdir(parents=True, exist_ok=True)
    source_root = Path(__file__).parents[2]
    source_paths = [source_root/'smart_wheelchair_safety'/'smart_wheelchair_safety'/name
                    for name in ('unified_control_node.py', 'unified_geometry.py',
                                 'local_path_follower.py', 'local_path_follower_node.py')]
    source_paths += [Path(__file__), source_root/'smart_wheelchair_gazebo'/'models'/'smart_wheelchair'/'model.sdf']
    hashes = {str(p.relative_to(source_root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    (args.output/'manifest.json').write_text(json.dumps(dict(
        expected=len(args.doors)*len(args.directions)*len(args.angles),
        angles=args.angles, directions=args.directions, doors=args.doors,
        timeout=args.timeout, pose_margin=.02, sweep_step=.005, sources=hashes), indent=2))
    geometry = M6AccessibilityTest()
    geometry.setUp()
    rospy.init_node('door_matrix', disable_signals=True)
    rospy.wait_for_service('/gazebo/set_model_state', timeout=30)
    place = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
    data = {}
    poses = []
    refs = []

    def odom(msg):
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        pose = (p.x, p.y, math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z)))
        data.update(pose=pose, measured=[msg.twist.twist.linear.x, msg.twist.twist.angular.z],
                    odom_received=time.monotonic())
        poses.append(pose)

    rospy.Subscriber('/odom', Odometry, odom, queue_size=100)
    rospy.Subscriber('/cmd_vel', Twist, lambda m: data.update(
        cmd=[m.linear.x, m.angular.z], cmd_received=time.monotonic()), queue_size=10)
    rospy.Subscriber('/shared_control/status', String,
                     lambda m: data.update(status=json.loads(m.data)), queue_size=10)
    rospy.Subscriber('/shared_control/reference', RosPath,
                     lambda m: refs.append([(p.pose.position.x, p.pose.position.y,
                         2*math.atan2(p.pose.orientation.z, p.pose.orientation.w)) for p in m.poses]), queue_size=1)
    for side in ('left', 'right'):
        rospy.Subscriber('/scan_'+side, LaserScan, lambda m, side=side: data.update(
            {side: {'stamp': m.header.stamp.to_sec(), 'ranges': list(m.ranges),
                    'min': m.angle_min, 'increment': m.angle_increment}}), queue_size=1)

    def command(y=0.):
        req = urllib.request.Request('http://localhost:8090/cmd',
            data=json.dumps(dict(x=0., y=y, client_id='door-matrix')).encode(),
            headers={'Content-Type': 'application/json'})
        urllib.request.urlopen(req, timeout=2).close()

    children = []
    summaries = []
    try:
        for door in (d for d in DOORS if d[0] in args.doors):
            for direction in args.directions:
                for angle in args.angles:
                    if any(hashlib.sha256(p.read_bytes()).hexdigest() != hashes[str(p.relative_to(source_root))]
                           for p in source_paths):
                        raise RuntimeError('source changed during matrix; run is incomplete')
                    case = '{}-{}-{:+g}'.format(door[0], direction, angle)
                    axle, yaw, normal = start_pose(door, direction, angle)
                    assert not overlaps((*axle, yaw), geometry.boxes, .04), 'invalid start '+case
                    command()
                    rosnode.kill_nodes(['/unified_control_node', '/local_path_follower_node'])
                    for child in children:
                        child.wait(timeout=10)
                    children = []
                    time.sleep(.25)
                    state = ModelState(model_name='smart_wheelchair', reference_frame='world')
                    state.pose.position.x = axle[0]+.33*math.cos(yaw)
                    state.pose.position.y = axle[1]+.33*math.sin(yaw)
                    state.pose.position.z = .04
                    state.pose.orientation.z, state.pose.orientation.w = math.sin(yaw/2), math.cos(yaw/2)
                    assert place(state).success
                    time.sleep(.4)
                    log = (args.output/(case+'.log')).open('w')
                    for executable in ('local_path_follower_node', 'unified_control_node'):
                        children.append(subprocess.Popen(['rosrun', 'smart_wheelchair_safety', executable,
                                                          '__name:='+executable],
                                                         stdout=log, stderr=subprocess.STDOUT))
                    # New application nodes have no path, wall or door memory.
                    data.pop('status', None)
                    ready = time.monotonic()+8.
                    while time.monotonic() < ready and 'status' not in data:
                        command()
                        time.sleep(.1)
                    if 'status' not in data:
                        raise RuntimeError('fresh controller failed to start: '+case)
                    poses.clear()
                    refs.clear()
                    records, modes = [], []
                    checked_count = 0
                    last_checked_pose = data['pose']
                    crossed_aperture = False
                    hit = None
                    stopped_since = None
                    longest_stop = 0.
                    started = time.monotonic()
                    result = 'timeout'
                    while time.monotonic()-started < args.timeout:
                        command(.4)
                        time.sleep(.08)
                        row = {k: v for k, v in data.items() if k not in ('left', 'right')}
                        row['t'] = time.monotonic()-started
                        records.append(row)
                        mode = row.get('status', {}).get('mode')
                        if not modes or mode != modes[-1]:
                            modes.append(mode)
                        position = np.asarray(row['pose'][:2])-door[1:3]
                        progress = float(position@normal)
                        if max(abs(v) for v in row['measured']) < .025:
                            if stopped_since is None:
                                stopped_since = row['t']
                            longest_stop = max(longest_stop, row['t']-stopped_since)
                        else:
                            stopped_since = None
                        recent = poses[checked_count:]
                        checked_count += len(recent)
                        hit, crossed = inspect_sweep([last_checked_pose]+recent, geometry.boxes, door, normal)
                        crossed_aperture = crossed_aperture or crossed
                        if recent:
                            last_checked_pose = recent[-1]
                        if hit:
                            result = 'collision:'+hit
                            break
                        if longest_stop > 3.:
                            result = 'stalled'
                            break
                        # Entire rear body beyond jamb thickness plus 4 cm.
                        heading_error = row['pose'][2]-math.atan2(normal[1], normal[0])
                        if crossed_aperture and progress > rear_extent(heading_error)+.10:
                            result = 'passed'
                            break
                    command()
                    released = time.monotonic()
                    until = released+3.
                    while time.monotonic() < until:
                        time.sleep(.05)
                        fresh_stop = all(released < data.get(k+'_received', 0.)
                                         and time.monotonic()-data[k+'_received'] < .25 for k in ('odom', 'cmd'))
                        if fresh_stop and max(abs(v) for v in data['cmd']+data['measured']) < .02:
                            break
                    else:
                        result = 'stop_timeout'
                    after_hit, _ = inspect_sweep([last_checked_pose]+poses[checked_count:],
                                                 geometry.boxes, door, normal)
                    if after_hit:
                        result = 'collision:'+after_hit
                    progress = float((np.asarray(data['pose'][:2])-door[1:3])@normal)
                    heading_error = data['pose'][2]-math.atan2(normal[1], normal[0])
                    if result == 'passed' and progress <= rear_extent(heading_error)+.10:
                        result = 'incomplete_clearance'
                    summary = dict(case=case, result=result, elapsed=round(time.monotonic()-started, 3),
                                   modes=modes, final_pose=data['pose'], progress=progress,
                                   final_cmd=data['cmd'], final_measured=data['measured'],
                                   longest_stop=longest_stop, crossed_aperture=crossed_aperture)
                    summaries.append(summary)
                    (args.output/(case+'.json')).write_text(json.dumps(dict(summary=summary, records=records,
                        poses=poses, references=refs, final_scans={k:data.get(k) for k in ('left','right')})))
                    (args.output/'summary.json').write_text(json.dumps(summaries, indent=2))
                    print(json.dumps(summary), flush=True)
                    log.close()
    finally:
        command()
        rosnode.kill_nodes(['/unified_control_node', '/local_path_follower_node'])
        for child in children:
            child.wait(timeout=10)
        rospy.signal_shutdown('matrix finished')
    passed = sum(r['result'] == 'passed' for r in summaries)
    print('PASS {}/{} ({:.1%})'.format(passed, len(summaries), passed/max(1,len(summaries))), flush=True)
    return 0 if summaries and passed/len(summaries) > .9 else 1


if __name__ == '__main__':
    raise SystemExit(main())
