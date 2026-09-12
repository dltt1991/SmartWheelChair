#!/usr/bin/env python3
"""Isolated M6 simulation only: two aligned doors followed by a front wall."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import time
import urllib.request

import rospy
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from check_door_matrix import overlaps
from test_m6_accessibility import M6AccessibilityTest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rospy.init_node('door_chain_check', disable_signals=True)
    data = {}
    def odom(msg):
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        data.update(pose=[p.x, p.y, 2*math.atan2(q.z, q.w)],
                    measured=[msg.twist.twist.linear.x, msg.twist.twist.angular.z],
                    stamp=time.monotonic())
    rospy.Subscriber('/odom', Odometry, odom, queue_size=100)
    rospy.Subscriber('/shared_control/status', String,
                     lambda msg: data.update(status=json.loads(msg.data)), queue_size=10)
    def command(v):
        request = urllib.request.Request('http://localhost:8090/cmd',
            data=json.dumps(dict(x=0., y=v, client_id='door-chain')).encode(),
            headers={'Content-Type': 'application/json'})
        urllib.request.urlopen(request, timeout=2).close()
    command(0.)
    rospy.wait_for_service('/gazebo/set_model_state', timeout=10)
    state = ModelState(model_name='smart_wheelchair', reference_frame='world')
    state.pose.position.x, state.pose.position.y, state.pose.position.z = -4.5, 2.37, .04
    state.pose.orientation.z, state.pose.orientation.w = math.sin(-math.pi/4), math.cos(-math.pi/4)
    assert rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)(state).success
    children = [subprocess.Popen(['rosrun', 'smart_wheelchair_safety', name],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                for name in ('unified_control_node', 'local_path_follower_node')]
    geometry = M6AccessibilityTest()
    geometry.setUp()
    rows = []
    hit = None
    try:
        until = time.monotonic()+3.
        while time.monotonic() < until:
            command(0.)
            time.sleep(.08)
        start = time.monotonic()
        while time.monotonic()-start < 38.:
            command(.4)
            time.sleep(.08)
            if 'pose' not in data:
                continue
            row = dict(data, t=time.monotonic()-start)
            rows.append(row)
            hit = overlaps(row['pose'], geometry.boxes, .02)
            if hit:
                break
            if row['pose'][1] < -4. and abs(row['pose'][2]+math.pi/2) > 1.2:
                break
    finally:
        command(0.)
        time.sleep(.5)
        for child in children:
            child.terminate()
        for child in children:
            child.wait(timeout=10)
    between = [r for r in rows if -1.7 < r['pose'][1] < 1.]
    pauses = [r for r in between if abs(r['measured'][0]) < .025]
    result = dict(collision=hit, minimum_between_speed=min((r['measured'][0] for r in between), default=0.),
                  stopped_samples_between=len(pauses), final_pose=data.get('pose'),
                  modes=list(dict.fromkeys(r.get('status', {}).get('mode') for r in rows)),
                  crossed_second=any(r['pose'][1] < -1.7 for r in rows),
                  turned_at_wall=any(r['pose'][1] < -4. and abs(r['pose'][2]+math.pi/2) > 1.2 for r in rows))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(summary=result, records=rows), indent=2))
    print(json.dumps(result), flush=True)
    return 0 if not hit and not pauses and result['crossed_second'] and result['turned_at_wall'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
