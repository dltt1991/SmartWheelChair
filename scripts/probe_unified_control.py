"""Controlled Gazebo-only smoke tests. Runs movement; never use on hardware.

Run inside the sourced GUI container with no other joystick client active.
Door requires unified_door.sdf; other cases require a fresh m6_room.sdf.
Results are sampled sensor/odometry evidence, not a contact or safety proof.
"""

import argparse
import json
import math
from pathlib import Path
import subprocess
import time
import urllib.request

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from smart_wheelchair_safety.unified_geometry import extract_lines, transform_points


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case', choices=['wall', 'wall_gap', 'override', 'front', 'vertical', 'room_door', 'door'])
    parser.add_argument('--seconds', type=float, default=30.)
    parser.add_argument('--wall-side', choices=['left', 'right'], default='left')
    parser.add_argument('--output', default='/tmp/unified-probe.json')
    args = parser.parse_args()
    rclpy.init()
    node = Node('unified_sim_probe')
    data, scans, records = {}, {}, []
    for topic in ['cmd_vel_raw', 'cmd_vel_planned', 'cmd_vel']:
        node.create_subscription(Twist, topic, lambda m, t=topic: data.update({t: [m.linear.x, m.angular.z]}), 10)
    node.create_subscription(String, 'shared_control/status', lambda m: data.update(status=json.loads(m.data)), 10)
    node.create_subscription(Odometry, 'odom', lambda m: data.update(odom=[
        m.pose.pose.position.x, m.pose.pose.position.y,
        2*math.atan2(m.pose.pose.orientation.z, m.pose.pose.orientation.w)]), 10)
    for side in ['left', 'right']:
        node.create_subscription(LaserScan, 'scan_'+side, lambda m, s=side: scans.update({s: m}), qos_profile_sensor_data)

    def command(x=0., y=0.):
        request = urllib.request.Request('http://localhost:8090/cmd',
            data=json.dumps(dict(x=x, y=y, client_id='unified-sim-probe')).encode(),
            headers={'Content-Type': 'application/json'})
        urllib.request.urlopen(request, timeout=2).close()

    try:
        services = subprocess.run(['gz', 'service', '-l'], capture_output=True, text=True, check=True, timeout=5).stdout
        world = 'unified_door' if args.case == 'door' else 'm6_room'
        if f'/world/{world}/set_pose' not in services:
            raise RuntimeError(f'Requires running Gazebo world {world}; no motion sent')
        command()
        if args.case != 'door':
            yaw = math.pi/2 if args.case in ('front', 'vertical', 'room_door', 'wall_gap') else math.pi/6
            starts = {'vertical': (0., -4.3), 'front': (-2.5, 0.), 'room_door': (-4.5, 0.),
                      'wall_gap': (-.77, -4.3) if args.wall_side == 'left' else (.77, 1.7)}
            x, y = starts.get(args.case, (-4., 0.))
            req = (f'name: "smart_wheelchair", position: {{x: {x}, y: {y}, z: 0.04}}, '
                   f'orientation: {{z: {math.sin(yaw/2)}, w: {math.cos(yaw/2)}}}')
            subprocess.run(['gz', 'service', '-s', '/world/m6_room/set_pose',
                '--reqtype', 'gz.msgs.Pose', '--reptype', 'gz.msgs.Boolean',
                '--timeout', '2000', '--req', req], check=True, timeout=5)
            subprocess.run(['ros2', 'service', 'call', '/local_costmap/clear_entirely_local_costmap',
                'nav2_msgs/srv/ClearEntireCostmap', '{}'], check=True, timeout=10)
        start = time.monotonic()
        last_send = last_log = 0.
        initial_odom = None
        while time.monotonic()-start < args.seconds+2:
            now = time.monotonic()
            elapsed = now-start
            if now-last_send >= .08:
                turn = -.3 if args.case in ('wall', 'override') else 0.
                if args.case == 'override' and elapsed > 13:
                    turn = .3
                command(turn if elapsed > 2 else 0., .5 if elapsed > 2 else 0.)
                last_send = now
            rclpy.spin_once(node, timeout_sec=.005)
            if now-last_log < .1 or len(scans) < 2 or 'odom' not in data:
                continue
            last_log = now
            if initial_odom is None:
                initial_odom = data['odom']
            clearances = []
            wall_clearances = []
            for side, msg in scans.items():
                ranges = np.array(msg.ranges)
                angles = msg.angle_min + np.arange(len(ranges))*msg.angle_increment
                valid = np.isfinite(ranges) & (ranges >= msg.range_min) & (ranges <= msg.range_max)
                x = ranges[valid]*np.cos(angles[valid])+.79
                y = ranges[valid]*np.sin(angles[valid]) + (.26 if side == 'left' else -.26)
                dx = np.maximum(np.maximum(-.25-x, x-.97), 0.)
                dy = np.maximum(np.abs(y)-.4, 0.)
                clearances.extend(np.hypot(dx, dy).tolist())
                if args.case == 'wall_gap':
                    for line in extract_lines(np.column_stack((x, y))):
                        if (abs(line.heading) < .04 and min(line.start[0], line.end[0]) < .97
                                and max(line.start[0], line.end[0]) > .79):
                            nx, ny = -math.sin(line.heading), math.cos(line.heading)
                            sign = math.copysign(1., line.distance)
                            support = max(-.25*sign*nx, .97*sign*nx)+.4*abs(ny)
                            gap = abs(line.distance)-support
                            if 0. <= gap < .5:
                                wall_clearances.append(gap)
            row = dict(t=round(elapsed, 3), **data, clearance=min(clearances, default=99.))
            if args.case == 'wall_gap':
                row['wall_clearance'] = min(wall_clearances, default=None)
            if args.case == 'door':
                relative = transform_points([data['odom'][:2]], initial_odom, inverse=True)[0]
                yaw = math.radians(10.)
                axle_start = (-2.5-.33*math.cos(yaw), .15-.33*math.sin(yaw), yaw)
                row['world_axle'] = transform_points([relative], axle_start)[0].tolist()
                row['world_yaw'] = data['odom'][2]-initial_odom[2]+yaw
            records.append(row)
        command()
        moving = [r for r in records if r['t'] > 3]
        result = {'case': args.case, 'samples': len(moving),
                  'min_sampled_clearance': min((r['clearance'] for r in moving), default=0.),
                  'max_speed': max((r.get('cmd_vel', [0])[0] for r in moving), default=0.),
                  'modes': sorted({r.get('status', {}).get('mode', 'missing') for r in moving}),
                  'last': records[-1] if records else None}
        if args.case == 'wall_gap':
            gaps = [r['wall_clearance'] for r in moving if r['wall_clearance'] is not None
                    and r.get('status', {}).get('mode') == 'wall' and r.get('cmd_vel', [0])[0] > .1]
            result['steady_gap_samples'] = len(gaps)
            result['steady_gap_min_max'] = [min(gaps), max(gaps)] if gaps else None
        Path(args.output).write_text(json.dumps({'summary': result, 'records': records}, indent=2)+'\n')
        print(json.dumps(result, indent=2))
        assert moving and result['min_sampled_clearance'] > .02, 'missing data or insufficient sampled clearance'
        if args.case == 'door':
            assert any(r['world_axle'][0] > .6 for r in moving), 'rear axle did not clear doorway'
            assert 'door' in result['modes'], 'door assistance not observed'
        elif args.case == 'front':
            assert abs(moving[-1].get('cmd_vel', [99])[0]) < .02, 'did not stop at front wall'
        elif args.case == 'override':
            assert any(r['t'] > 13 and r.get('cmd_vel', [0, 0])[1] < -.3
                       and r.get('status', {}).get('mode') == 'override' for r in moving), 'override not observed'
        elif args.case == 'vertical':
            assert math.dist(initial_odom[:2], moving[-1]['odom'][:2]) > 2., 'vertical corridor progress too small'
        elif args.case == 'room_door':
            progress = transform_points([moving[-1]['odom'][:2]], initial_odom, inverse=True)[0, 0]
            assert progress > 2.5 and 'door' in result['modes'], 'NW room door was not cleared'
        elif args.case == 'wall_gap':
            assert len(gaps) >= 10 and max(gaps) <= .15, 'steady body-edge wall gap exceeds 15 cm'
        else:
            assert 'wall' in result['modes'] and result['max_speed'] > .6, 'wall speed did not recover'
    finally:
        try:
            command()
        finally:
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()
