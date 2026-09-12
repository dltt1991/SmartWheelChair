#!/usr/bin/env python3
"""Run the complete 20-case M6 junction matrix with a fixed 20 s horizon.

Run inside a sourced Noetic container with a fresh M6 world and no other driver.
Each case gets fresh application controllers. Full-horizon body sweeps and fresh
stopping remain mandatory; gap progress/continuity apply through the first exit.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT/'catkin_ws/src/smart_wheelchair_gazebo/test'))
from check_door_matrix import overlaps
from test_m6_accessibility import M6AccessibilityTest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cases', nargs='+', default=['opening_turn', 'opening_straight', 'opening_gap'])
    parser.add_argument('--angles', nargs='+', type=int, default=[0, 90, 180, 270])
    parser.add_argument('--sides', nargs='+', choices=['left', 'right'], default=['left', 'right'])
    parser.add_argument('--forward-y', type=float, default=.5)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not 0. < args.forward_y <= 1.:
        parser.error('--forward-y must be in (0, 1]')
    args.output.mkdir(parents=True, exist_ok=True)
    args.output = args.output.resolve()
    sources = [ROOT/'catkin_ws/src/smart_wheelchair_safety/smart_wheelchair_safety'/name
               for name in ('unified_control_node.py', 'unified_geometry.py', 'local_path_follower.py')]
    sources += [Path(__file__).resolve(), ROOT/'scripts/probe_unified_control.py']
    fingerprints = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in sources}
    (args.output/'source-hashes.json').write_text(json.dumps(fingerprints, indent=2))
    snapshots = args.output/'source-snapshots'
    snapshots.mkdir()
    for path in sources:
        (snapshots/path.name).write_bytes(path.read_bytes())
    subprocess.run(['rosservice', 'call', '/gazebo/get_world_properties', '{}'],
                   check=True, stdout=subprocess.DEVNULL, timeout=15)
    request = urllib.request.Request('http://localhost:8090/cmd',
        data=json.dumps(dict(x=0., y=0., client_id='junction-matrix')).encode(),
        headers={'Content-Type': 'application/json'})
    urllib.request.urlopen(request, timeout=2).close()
    subprocess.run(['rosservice', 'call', '/gazebo/unpause_physics', '{}'],
                   check=True, stdout=subprocess.DEVNULL, timeout=10)
    for topic in ('/odom', '/scan_left', '/scan_right'):
        subprocess.run(['rostopic', 'echo', '-n', '1', topic],
                       check=True, stdout=subprocess.DEVNULL, timeout=10)
    scene = M6AccessibilityTest()
    scene.setUp()
    summaries = []
    children = []
    expected = len(args.angles)*sum(1 if case == 'opening_straight' else len(args.sides)
                                  for case in args.cases)
    (args.output/'run-config.json').write_text(json.dumps(dict(
        cases=args.cases, angles=args.angles, sides=args.sides, expected=expected,
        horizon=20., forward_y=args.forward_y, interpolation_step=.005, footprint_margin=.02), indent=2))
    for case in args.cases:
        for angle in args.angles:
            for side in (['left'] if case == 'opening_straight' else args.sides):
                name = f'{case}-{side}-{angle}'
                assert all(hashlib.sha256(path.read_bytes()).hexdigest() == fingerprints[str(path.relative_to(ROOT))]
                           for path in sources), 'production sources changed during matrix'
                subprocess.run(['rosnode', 'kill', '/unified_control_node', '/local_path_follower_node'],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
                for child in children:
                    child.wait(timeout=10)
                children = []
                with (args.output/(name+'.controller.log')).open('w') as log:
                    for executable in ('local_path_follower_node', 'unified_control_node'):
                        children.append(subprocess.Popen(['rosrun', 'smart_wheelchair_safety', executable,
                                                          '__name:='+executable], stdout=log, stderr=log))
                    time.sleep(2.)
                    output = args.output/(name+'.json')
                    with (args.output/(name+'.log')).open('w') as probe_log:
                        try:
                            result = subprocess.run(['python3', 'scripts/probe_unified_control.py', case,
                                '--wall-side', side, '--approach-yaw-deg', str(angle), '--seconds', '20',
                                '--forward-y', str(args.forward_y),
                                '--output', str(output)], cwd=ROOT, stdout=probe_log, stderr=probe_log, timeout=65)
                            returncode = result.returncode
                        except subprocess.TimeoutExpired:
                            returncode = 124
                summary = dict(case=name, returncode=returncode, horizon=20., overlap=None)
                if output.exists():
                    data = json.loads(output.read_text())
                    summary.update(data['summary'])
                    summary['case'] = name
                    poses = [(*r['world_axle'], r['world_yaw']) for r in data['records']
                             if 'world_axle' in r]
                    checked = 0
                    for start, end in zip(poses, poses[1:]):
                        delta = np.array(end)-start
                        delta[2] = math.atan2(math.sin(delta[2]), math.cos(delta[2]))
                        steps = max(1, math.ceil((np.linalg.norm(delta[:2])+1.1*abs(delta[2]))/.005))
                        for fraction in np.linspace(0., 1., steps+1):
                            hit = overlaps(np.array(start)+fraction*delta, scene.boxes, .02)
                            checked += 1
                            if hit and summary['overlap'] is None:
                                summary['overlap'] = hit
                    summary['interpolated_poses_checked'] = checked
                    turn_sign = 1. if side == 'left' else -1.
                    initial_yaw = data['records'][0]['odom'][2] if data['records'] else 0.
                    moving = [r for r in data['records'] if any(r.get('cmd_vel_raw', []))]
                    turning = []
                    for row in moving:
                        turning.append(row)
                        delta_yaw = row['odom'][2]-initial_yaw
                        if turn_sign*math.atan2(math.sin(delta_yaw), math.cos(delta_yaw)) >= math.radians(70):
                            break
                    jumps = [abs(b['cmd_vel'][1]-a['cmd_vel'][1]) for a, b in zip(turning, turning[1:])]
                    # Rows and command callbacks are asynchronous; use the
                    # command receipt interval rather than the row interval.
                    accelerations = [jump/interval for jump, a, b
                                     in zip(jumps, turning, turning[1:])
                                     for interval in [b.get('cmd_vel_received', b['t'])
                                                      -a.get('cmd_vel_received', a['t'])]
                                     if interval > 0.]
                    summary['junction_max_angular_command_step'] = max(jumps, default=0.)
                    summary['junction_max_sampled_angular_acceleration'] = max(accelerations, default=0.)
                    summary.pop('last', None)
                summary['passed'] = returncode == 0 and output.exists() and summary['overlap'] is None
                summaries.append(summary)
                (args.output/'summary.json').write_text(json.dumps(summaries, indent=2))
                print(json.dumps(summary), flush=True)
    subprocess.run(['rosnode', 'kill', '/unified_control_node', '/local_path_follower_node'],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
    for child in children:
        child.wait(timeout=10)
    passed = sum(s['passed'] for s in summaries)
    print(f"PASS {passed}/{expected}", flush=True)
    return 0 if len(summaries) == expected and passed == expected else 1


if __name__ == '__main__':
    raise SystemExit(main())
