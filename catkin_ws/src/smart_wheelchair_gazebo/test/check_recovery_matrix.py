#!/usr/bin/env python3
"""Adversarial low-clearance starts at every M6 door jamb, both body ends.

The inward joystick is a required stopped negative case. Retreat must make
measured progress with no conservative-body collision, then stop on neutral.
Invalid overlapping starts are recorded separately, never counted as passes.
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
from check_door_matrix import overlaps
from test_m6_accessibility import DOORS, M6AccessibilityTest


def cases(angles):
    for door in DOORS:
        name, x, y, axis, width = door
        normal = np.array([0., math.copysign(1., y)]) if axis == 0 else np.array([math.copysign(1., x), 0.])
        tangent = np.array([-normal[1], normal[0]])
        for jamb in (-1, 1):
            face = np.array([x, y]) + jamb*(width/2+.25)*tangent - .06*normal
            for degrees in angles:
                yaw = math.atan2(normal[1], normal[0])+math.radians(degrees)
                c, s = math.cos(yaw), math.sin(yaw)
                rotation = np.array([[c, -s], [s, c]])
                n = normal @ rotation
                corner = np.array([.97 if n[0] >= 0. else -.25, .4 if n[1] >= 0. else -.4])
                # 2 cm from the actual face, deliberately inside 4 cm margin.
                axle = face - rotation@corner - .02*normal
                yield dict(name=f'{name}-j{jamb}-{degrees:+g}', door=door, normal=normal,
                           pose=(*axle, yaw), away=-math.copysign(1., n[0]))


def motion_extent(poses, origin):
    return (max((math.dist(p[:2], origin[:2]) for p in poses), default=0.),
            max((abs(math.atan2(math.sin(p[2]-origin[2]), math.cos(p[2]-origin[2])))
                 for p in poses), default=0.))


def main(case_factory=cases, extra_sources=()):
    import rospy, rosnode
    from gazebo_msgs.msg import ModelState
    from gazebo_msgs.srv import SetModelState
    from geometry_msgs.msg import Twist
    from nav_msgs.msg import Odometry
    from std_msgs.msg import String
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--angles', nargs='+', type=float, default=[-60, -30, 0, 30, 60, 150, 180, -150])
    parser.add_argument('--seconds', type=float, default=6.)
    parser.add_argument('--doors', nargs='+')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    selected = list(case_factory(args.angles))
    if args.doors:
        known = {case['door'][0] for case in selected if 'door' in case}
        if set(args.doors)-known:
            parser.error('unknown door')
        selected = [case for case in selected if case['door'][0] in args.doors]
    args.output.mkdir(parents=True, exist_ok=True)
    scene = M6AccessibilityTest(); scene.setUp()
    sources = [Path(__file__)] + list((Path(__file__).parents[2]/'smart_wheelchair_safety'/'smart_wheelchair_safety').glob('unified_*.py'))
    sources += [Path(__file__).with_name(name) for name in ('check_door_matrix.py', 'test_m6_accessibility.py')]
    sources += [Path(__file__).parents[1]/'models/smart_wheelchair/model.sdf',
                Path(__file__).parents[1]/'worlds/m6_room.world',
                Path(__file__).parents[2]/'smart_wheelchair_safety/smart_wheelchair_safety/local_path_follower.py']
    sources += list(extra_sources)
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    (args.output/'manifest.json').write_text(json.dumps(dict(angles=args.angles, seconds=args.seconds, hashes=hashes,
        expected=len(selected), doors=args.doors, footprint=[-.25, .97, -.4, .4], sweep_step=.002), indent=2))
    rospy.init_node('recovery_matrix', disable_signals=True)
    rospy.wait_for_service('/gazebo/set_model_state', timeout=30.)
    place = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
    data = {}; poses = []
    def odom(m):
        p, q = m.pose.pose.position, m.pose.pose.orientation
        data.update(pose=[p.x, p.y, 2*math.atan2(q.z, q.w)],
                    measured=[m.twist.twist.linear.x, m.twist.twist.angular.z], received=time.monotonic())
        poses.append(data['pose'])
    rospy.Subscriber('/odom', Odometry, odom, queue_size=100)
    rospy.Subscriber('/cmd_vel', Twist, lambda m: data.update(cmd=[m.linear.x, m.angular.z], cmd_received=time.monotonic()))
    rospy.Subscriber('/shared_control/status', String, lambda m: data.update(status=json.loads(m.data)))
    def command(v=0.):
        req = urllib.request.Request('http://localhost:8090/cmd', data=json.dumps(dict(x=0., y=v, client_id='recovery-matrix')).encode(), headers={'Content-Type':'application/json'})
        urllib.request.urlopen(req, timeout=2).close()
    def drive(v, seconds, history):
        until = time.monotonic()+seconds
        while time.monotonic() < until:
            command(v); time.sleep(.05)
            history.append(dict(data))
    results = []; children = []; active = None
    try:
        for case in selected:
            active = case['name']
            assert all(hashlib.sha256(p.read_bytes()).hexdigest() == hashes[str(p)] for p in sources), 'source changed'
            hit = overlaps(case['pose'], scene.boxes)
            if hit:
                results.append(dict(name=case['name'], invalid_start=hit, passed=False))
                active = None
                continue
            command()
            rosnode.kill_nodes(['/unified_control_node', '/local_path_follower_node'])
            for child in children: child.wait(timeout=10.)
            children = []
            x, y, yaw = case['pose']
            state = ModelState(model_name='smart_wheelchair', reference_frame='world')
            state.pose.position.x, state.pose.position.y = x+.33*math.cos(yaw), y+.33*math.sin(yaw)
            state.pose.position.z = .04
            state.pose.orientation.z, state.pose.orientation.w = math.sin(yaw/2), math.cos(yaw/2)
            assert place(state).success
            time.sleep(.5)
            log = (args.output/(case['name']+'.log')).open('w')
            for executable in ('local_path_follower_node', 'unified_control_node'):
                children.append(subprocess.Popen(['rosrun', 'smart_wheelchair_safety', executable, '__name:='+executable], stdout=log, stderr=log))
            data.pop('status', None)
            drive(0., 1.5, [])
            assert 'status' in data, 'controller not ready'
            poses.clear(); history = []
            start = data['pose'][:]
            drive(-case['away']*.12, 1., history)
            inward_distance, inward_yaw = motion_extent(poses, start)
            drive(0., .4, history)
            retreat_start = data['pose'][:]
            drive(case['away']*.12, args.seconds, history)
            retreat_end = data['pose'][:]
            drive(0., .6, history)
            # Unlike normal door tests, starts are already inside the margin;
            # still require no uninflated conservative rectangle overlap.
            collision = None
            for first, second in zip(poses, poses[1:]):
                delta = math.atan2(math.sin(second[2]-first[2]), math.cos(second[2]-first[2]))
                steps = max(1, math.ceil((math.dist(first[:2], second[:2])+1.06*abs(delta))/.002))
                for t in np.linspace(0., 1., steps+1):
                    xy = np.array(first[:2])+t*(np.array(second[:2])-first[:2])
                    collision = overlaps((*xy, first[2]+t*delta), scene.boxes)
                    if collision: break
                if collision: break
            heading = case['away']*np.array([math.cos(yaw), math.sin(yaw)])
            progress = float((np.array(retreat_end[:2])-retreat_start[:2])@heading)
            stopped = (time.monotonic()-data['received'] < .2 and time.monotonic()-data['cmd_received'] < .2
                       and max(abs(v) for v in data['cmd']) < .001 and abs(data['measured'][0]) < .01
                       and abs(data['measured'][1]) < .015)
            result = dict(name=case['name'], passed=bool(inward_distance < .012 and inward_yaw < .02 and progress > .08 and not collision and stopped),
                          inward_distance=inward_distance, inward_yaw=inward_yaw, progress=progress, collision=collision, stopped=stopped,
                          statuses={reason: sum(h.get('status', {}).get('reason') == reason for h in history)
                                    for reason in {h.get('status', {}).get('reason') for h in history}},
                          start=start, end=data['pose'])
            (args.output/(case['name']+'.json')).write_text(json.dumps(dict(result=result, history=history, poses=poses), indent=2))
            results.append(result)
            active = None
            (args.output/'summary.json').write_text(json.dumps(results, indent=2))
            print(json.dumps(result), flush=True)
            log.close()
    finally:
        command()
        if active is not None:
            results.append(dict(name=active, passed=False, error='interrupted before complete result'))
        (args.output/'summary.json').write_text(json.dumps(results, indent=2))
    assert len(results) == len(selected) and all(r['passed'] for r in results), 'recovery matrix failed or has invalid starts'


if __name__ == '__main__':
    main()
