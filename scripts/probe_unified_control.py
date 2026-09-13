"""Controlled Gazebo-only smoke tests. Runs movement; never use on hardware.

Run inside the sourced GUI container with no other joystick client active.
Door requires unified_door.world; other cases require a fresh m6_room.world.
Results are sampled sensor/odometry evidence, not a contact or safety proof.
"""

import argparse
from collections import Counter
from copy import deepcopy
import json
import math
from pathlib import Path
import subprocess
import time
import urllib.request


def record_control_status(data, events, status, received=None):
    """Keep every delivered status, independent of sampled data and mutations."""
    received = time.monotonic() if received is None else received
    events.append(dict(monotonic=received, status=deepcopy(status)))
    data.update(status=status, status_received=received)


def control_event_window(events, started, ended):
    # Snapshot the append-only callback buffer before selecting this case.
    return [dict(event, t=event['monotonic']-started) for event in list(events)
            if started <= event['monotonic'] <= ended]


def control_event_metrics(events):
    """Callback-rate diagnostics; includes deliberate stops inside the window."""
    result = dict(events=len(events), reason_counts=dict(Counter(
        event['status'].get('reason', 'missing') for event in events)))
    intervals = [b['monotonic']-a['monotonic'] for a, b in zip(events, events[1:])]
    result.update(min_event_interval_s=min(intervals, default=None),
                  max_event_interval_s=max(intervals, default=None))
    for name, key in (('linear', 'v'), ('angular', 'w')):
        steps = [dict(step=abs(b['status'][key]-a['status'][key]),
                      t=b['t'], interval_s=b['monotonic']-a['monotonic'],
                      before=a['status'][key], after=b['status'][key],
                      previous_reason=a['status'].get('reason'), reason=b['status'].get('reason'))
                 for a, b in zip(events, events[1:])
                 if key in a['status'] and key in b['status']]
        peak = max(steps, key=lambda step: step['step'], default=None)
        result['max_'+name+'_command_step'] = peak['step'] if peak else None
        result['max_'+name+'_step_event'] = peak
    return result


def send_command(client_id, x=0., y=0.):
    # HTTP 200 only acknowledges receipt: unversioned commands can be discarded.
    with urllib.request.urlopen('http://localhost:8090/mode', timeout=2) as response:
        mode = json.loads(response.read())
    if (x or y) and not mode['assist_enabled']:
        raise RuntimeError('assisted probe requires assist_enabled')
    request = urllib.request.Request('http://localhost:8090/cmd',
        data=json.dumps(dict(x=x, y=y, client_id=client_id,
                             mode_revision=mode['revision'], mode_session=mode['session'])).encode(),
        headers={'Content-Type': 'application/json'})
    urllib.request.urlopen(request, timeout=2).close()


def controller_ready(data, after, now):
    # Raw topic receipts can precede TF/scan processing inside the controller.
    # A fresh neutral acknowledgement proves its own input checks also passed.
    return data.get('status', {}).get('reason') == 'user_stop' and all(after < data.get(key+'_received', -math.inf) <= now
               and now-data[key+'_received'] <= .5
               for key in ('status', 'odom', 'left', 'right'))


def wait_for_controller(data, command, timeout=30.):
    started = time.monotonic()
    while time.monotonic()-started < timeout:
        command()
        if controller_ready(data, started, time.monotonic()):
            return
        time.sleep(.05)
    raise RuntimeError('controller not ready: accepted neutral, fresh status, odometry and both scans required')


def motion_metrics(records):
    """Diagnostics only; sampled stops include intentional stops and pivots."""
    steps, accelerations = [], []
    longest_stop, stopped_since = 0., None
    modes = []
    wait_samples = 0
    for row in records:
        mode = row.get('status', {}).get('mode')
        if not modes or modes[-1] != mode:
            modes.append(mode)
        wait_samples += (mode == 'waiting' or row.get('status', {}).get('reason') == 'planner_timeout')
        measured = row.get('measured', row.get('odom_velocity', []))
        if measured and max(abs(v) for v in measured) < .025:
            stopped_since = row['t'] if stopped_since is None else stopped_since
            longest_stop = max(longest_stop, row['t']-stopped_since)
        else:
            stopped_since = None
    for first, second in zip(records, records[1:]):
        key = 'cmd' if 'cmd' in first else 'cmd_vel'
        if key not in first or key not in second:
            continue
        step = abs(second[key][1]-first[key][1])
        steps.append(step)
        interval = second.get(key+'_received', 0.)-first.get(key+'_received', 0.)
        if interval > 0.:
            accelerations.append(step/interval)
    return dict(max_angular_command_step=max(steps, default=None),
                max_sampled_angular_acceleration=max(accelerations, default=None),
                longest_sampled_stop=longest_stop, mode_switches=max(0, len(modes)-1),
                planner_wait_samples=wait_samples)


def moving_records(records):
    for index, record in enumerate(records):
        if any(value != 0. for value in record.get('cmd_vel_raw', [])):
            return records[index:]
    return []


def check_door_modes(modes):
    assert 'wall' not in modes, 'wall following captured the doorway approach'
    assert 'door_align' in modes, 'door alignment was not selected'
    assert 'door_pass' in modes, 'door alignment never committed to pass'


def sample_row(data, scanned, started):
    """Pair copied telemetry with capture time and the scans used for geometry."""
    row = dict(data)
    sampled = time.monotonic()
    row.update(t=sampled-started, sampled_monotonic=sampled)
    row['raw_received'] = row.get('cmd_vel_raw_received', -math.inf)
    for side, (_, received) in scanned:
        row[side+'_received'] = received
    return row


def fresh_row(row, started, keys):
    now = started+row['t']
    return all(-.05 <= now-row.get(key+'_received', -math.inf) <= .25 for key in keys)


def check_front_stop(records, started):
    moving = moving_records(records)
    assert moving, 'front input was not observed'
    first, last = moving[0], moving[-1]
    yaw = first['odom'][2]
    progress = ((last['odom'][0]-first['odom'][0])*math.cos(yaw)
                +(last['odom'][1]-first['odom'][1])*math.sin(yaw))
    assert progress > .25, 'no measured approach to front obstacle'
    stable = [row for row in moving if row['t'] >= last['t']-2.]
    assert len(stable) >= 2 and stable[-1]['t']-stable[0]['t'] >= 1., 'front stop not sustained'
    assert all(b['t']-a['t'] <= .25 for a, b in zip(stable, stable[1:])), 'front stop observations have gaps'
    for row in stable:
        assert fresh_row(row, started, ('raw', 'cmd_vel', 'odom_velocity', 'left', 'right')), 'front stop telemetry stale'
        assert row.get('cmd_vel_raw', [0.])[0] > .02, 'front stop requires ongoing forward intent'
        assert abs(row['cmd_vel_raw'][1]) < .12, 'front stop requires straight intent'
        gap = row.get('front_obstacle_gap')
        assert gap is not None and .02 < gap <= .2, 'no nearby obstacle in forward body corridor'
        assert max(abs(v) for v in row['cmd_vel']+row['odom_velocity']) < .02, 'front command or body not stopped'


def override_start_pose():
    # Parallel to the long upper corridor wall, with 12 cm body-edge gap.
    return -3.2, .77, 0.


def override_wall_line(lines):
    # The 170-degree scan limit cannot see the near wall at the rear corner.
    # Require a real segment crossing behind the lidar and ahead of the body.
    sides = [line for line in lines if abs(line.heading) < .35
             and min(line.start[0], line.end[0]) < .79
             and max(line.start[0], line.end[0]) > 1.5
             and .4 < abs(line.distance) < .9]
    return min(sides, key=lambda line: abs(line.distance), default=None)


def override_trigger(records, started):
    if not records:
        return None
    last = records[-1]
    recent = [row for row in records if row['t'] >= last['t']-.6-1e-9]
    if not recent or recent[-1]['t']-recent[0]['t'] < .5:
        return None
    side = last.get('wall_side', 0)
    if not all(b['t']-a['t'] <= .25 for a, b in zip(recent, recent[1:])):
        return None
    if not side or not all(row.get('status', {}).get('mode') == 'wall'
            and row.get('wall_side') == side and abs(row.get('wall_heading', math.inf)) <= .1
            and row['cmd_vel_raw'][0] > .02
            and side*row['cmd_vel_raw'][1] > .15
            and fresh_row(row, started, ('raw', 'status', 'left', 'right')) for row in recent):
        return None
    return dict(t=last['t'], wall_side=side, pose=last['odom'],
                wall_heading_world=last['odom'][2]+last['wall_heading'],
                before_wall_records=recent)


def check_override_response(records, trigger, started, control_events=None):
    assert trigger is not None, 'stable wall override precondition not established'
    away = -trigger['wall_side']
    sent = trigger.get('sent_monotonic', started+trigger['t'])
    intent, entered = None, False
    entered_at = None
    previous = None
    for row in records:
        if row['t'] <= trigger['t']:
            continue
        if started+row['t'] > sent+4.:
            break
        # A pre-flip sample may still carry the old callback value.
        if intent is None and row.get('raw_received', 0.) <= sent:
            continue
        assert fresh_row(row, started, ('raw', 'status', 'cmd_vel', 'odom_velocity')), 'stale override response'
        raw = row['cmd_vel_raw']
        assert raw[0] > .02 and away*raw[1] > .3, 'away intent was interrupted'
        if intent is None:
            intent = list(raw)
        assert all(abs(a-b) < 1e-6 for a, b in zip(raw, intent)), 'override raw intent changed'
        if previous is not None:
            assert row['t']-previous <= .25, 'override evidence gap'
        previous = row['t']
        status = row.get('status', {})
        assert status.get('mode') in ('override', 'manual', 'wall', 'opening_turn'), 'invalid override handoff mode'
        assert status.get('reason', 'clear') == 'clear', 'override response had a stop or guard reason'
        assert away*row['cmd_vel'][1] >= -.025, 'opposite angular command during override response'
        assert row['cmd_vel'][0] >= -.025 and row['odom_velocity'][0] >= -.025, 'reverse motion during override response'
        yaw = math.atan2(math.sin(row['odom'][2]-trigger['pose'][2]),
                         math.cos(row['odom'][2]-trigger['pose'][2]))
        heading = trigger['wall_heading_world']
        lateral = (-(row['odom'][0]-trigger['pose'][0])*math.sin(heading)
                   +(row['odom'][1]-trigger['pose'][1])*math.cos(heading))
        if (status.get('mode') == 'override' and away*row['cmd_vel'][1] > .025
                and away*row['odom_velocity'][1] > .025 and away*yaw > 0.):
            if not entered:
                entered_at = started+row['t']
            entered = True
        if entered:
            assert max(abs(v) for v in row['cmd_vel']) > .02, 'stopped command during override response'
            assert max(abs(v) for v in row['odom_velocity']) > .02, 'stopped body during override response'
            if away*yaw > .1 and away*lateral > .05:
                if control_events is not None:
                    events = [e for e in control_events
                              if entered_at <= e['monotonic'] <= started+row['t']]
                    assert events, 'missing full-rate override evidence'
                    for event in events:
                        state = event['status']
                        assert state.get('reason') == 'clear', 'full-rate override stop or guard reason'
                        assert state.get('mode') in ('override', 'manual', 'wall', 'opening_turn'), 'invalid full-rate handoff mode'
                        assert away*state['w'] >= -.025 and state['v'] >= -.025, 'full-rate reverse command'
                        assert max(abs(state['v']), abs(state['w'])) > .02, 'full-rate override stop'
                return
    raise AssertionError('no fresh override followed by continuous away motion within four seconds')


def check_wall_result(result):
    assert 'wall' in result['modes'], 'wall following was not observed'
    assert result['max_speed'] > .6, 'wall speed did not recover'


def check_opening_gap_result(result, intended_sign):
    assert result['junction_exit_time'] is not None, 'junction exit was not reached'
    assert intended_sign*result['junction_yaw_change'] >= math.radians(70), 'joystick turn was not executed in the gap'
    assert result['junction_forward_progress'] > .25, 'junction forward progress too small'
    assert result['junction_longest_stop'] <= 3., 'prolonged stop after clearing the near wall'


def opening_gap_metrics(records, intended_sign, initial_pose):
    """Separate the junction turn from later travel toward unrelated walls."""
    progress, longest_stop, stopped_since = 0., 0., None
    result = {'junction_exit_time': None}
    c, s = math.cos(initial_pose[2]), math.sin(initial_pose[2])
    for row in records:
        x, y, yaw = row['odom']
        progress = max(progress, (x-initial_pose[0])*c+(y-initial_pose[1])*s)
        yaw = math.atan2(math.sin(yaw-initial_pose[2]), math.cos(yaw-initial_pose[2]))
        if max(abs(value) for value in row.get('odom_velocity', [0., 0.])) < .03:
            stopped_since = row['t'] if stopped_since is None else stopped_since
            longest_stop = max(longest_stop, row['t']-stopped_since)
        else:
            stopped_since = None
        if result['junction_exit_time'] is None:
            result.update(junction_yaw_change=yaw, junction_forward_progress=progress,
                          junction_longest_stop=longest_stop)
            if intended_sign*yaw >= math.radians(70) and progress > .25:
                result['junction_exit_time'] = row['t']
    result.update(forward_progress=progress, longest_stop=longest_stop)
    return result


def settled_after_release(data, released, now):
    return all(released < data.get(key + '_received', -math.inf) <= now
               and now - data[key + '_received'] <= .25
               and len(data.get(key, [])) == 2
               and all(math.isfinite(value) and abs(value) < .02 for value in data[key])
               for key in ('cmd_vel', 'odom_velocity'))


def main():
    import numpy as np
    import rospy
    import yaml
    from geometry_msgs.msg import Twist, TwistStamped
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import LaserScan
    from std_msgs.msg import String
    from smart_wheelchair_safety.unified_geometry import LIDAR_X_M, LIDAR_Y_M, extract_lines, find_openings, transform_points

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case', choices=['wall', 'wall_gap', 'override', 'front', 'vertical',
                                         'room_door', 'opening_turn', 'opening_straight', 'opening_gap', 'door'])
    parser.add_argument('--seconds', type=float, default=30.)
    parser.add_argument('--forward-y', type=float, default=.5,
                        help='Forward joystick fraction in (0, 1]')
    parser.add_argument('--wall-side', choices=['left', 'right'], default='left')
    parser.add_argument('--approach-yaw-deg', type=float, default=0.,
                        help='Rotate opening-case start position and heading around the junction')
    parser.add_argument('--lateral-m', type=float, default=.15,
                        help='Door fixture body-centre lateral offset in metres')
    parser.add_argument('--yaw-deg', type=float, default=10.,
                        help='Door fixture initial heading error in degrees')
    parser.add_argument('--active-align', action='store_true',
                        help='Steer toward the door for four seconds during takeover')
    parser.add_argument('--output', default='/tmp/unified-probe.json')
    args = parser.parse_args()
    if not 0. < args.forward_y <= 1.:
        parser.error('--forward-y must be in (0, 1]')
    rospy.init_node('unified_sim_probe')
    data, scans, records = {}, {}, []
    status_events = []
    motion_started = False
    mode_history = []
    override_evidence = None

    def on_status(msg):
        record_control_status(data, status_events, json.loads(msg.data))
        if motion_started:
            mode_history.append({'t': round(time.monotonic()-start, 3),
                                 'mode': data['status'].get('mode', 'missing')})

    for topic in ['cmd_vel_raw', 'cmd_vel']:
        rospy.Subscriber(topic, Twist, lambda m, t=topic: data.update(
            {t: [m.linear.x, m.angular.z], t + '_received': time.monotonic()}), queue_size=10)
    rospy.Subscriber(
        'cmd_vel_planned', TwistStamped,
        lambda m: data.update(cmd_vel_planned=[m.twist.linear.x,
                                                m.twist.angular.z]), queue_size=10)
    rospy.Subscriber('shared_control/status', String, on_status, queue_size=100)
    rospy.Subscriber('odom', Odometry, lambda m: data.update(odom=[
        m.pose.pose.position.x, m.pose.pose.position.y,
        2*math.atan2(m.pose.pose.orientation.z, m.pose.pose.orientation.w)],
        odom_velocity=[m.twist.twist.linear.x, m.twist.twist.angular.z],
        odom_velocity_received=time.monotonic(), odom_received=time.monotonic()), queue_size=10)
    def on_scan(msg, side):
        received = time.monotonic()
        scans[side] = (msg, received)
        data[side+'_received'] = received

    for side in ['left', 'right']:
        rospy.Subscriber('scan_'+side, LaserScan, lambda m, s=side: on_scan(m, s), queue_size=5)

    def command(x=0., y=0.):
        nonlocal motion_started
        send_command('unified-sim-probe', x, y)
        motion_started = motion_started or x != 0. or y != 0.

    try:
        properties = yaml.safe_load(subprocess.run(
            ['rosservice', 'call', '/gazebo/get_world_properties', '{}'],
            capture_output=True, text=True, check=True, timeout=5).stdout)
        world = 'unified_door' if args.case == 'door' else 'm6_room'
        # Classic's world-properties service exposes model names, not the world name.
        fixture_models = ({'left_jamb', 'right_jamb', 'exit_wall'} if args.case == 'door'
                          else {'outer_north_wall', 'outer_south_wall'})
        if not properties['success'] or not (fixture_models | {'smart_wheelchair'}).issubset(properties['model_names']):
            raise RuntimeError(f'Requires running Gazebo world {world}; no motion sent')
        command()
        if args.case == 'door':
            yaw = math.radians(args.yaw_deg)
            x, y = -2.5, args.lateral_m
        else:
            yaw = math.pi/2 if args.case in ('front', 'vertical', 'room_door', 'wall_gap') else math.pi/6
            if args.case.startswith('opening_'):
                yaw = 0.
            starts = {'vertical': (0., -4.3), 'front': (-2.5, 0.), 'room_door': (-4.5, 0.),
                      'wall_gap': (-.77, -4.3) if args.wall_side == 'left' else (.77, 1.7),
                      'opening_turn': (-3.5, .77 if args.wall_side == 'left' else -.77),
                      'opening_straight': (-3.5, .77 if args.wall_side == 'left' else -.77),
                      'opening_gap': (-.2, .77 if args.wall_side == 'left' else -.77)}
            x, y = starts.get(args.case, (-4., 0.))
            if args.case == 'override':
                x, y, yaw = override_start_pose()
            if args.case.startswith('opening_'):
                yaw = math.radians(args.approach_yaw_deg)
                x, y = x*math.cos(yaw)-y*math.sin(yaw), x*math.sin(yaw)+y*math.cos(yaw)
        opening_axle_start = (x-.33*math.cos(yaw), y-.33*math.sin(yaw), yaw)
        req = {'model_state': {'model_name': 'smart_wheelchair', 'reference_frame': 'world',
                              'pose': {'position': {'x': x, 'y': y, 'z': .04},
                                       'orientation': {'z': math.sin(yaw/2), 'w': math.cos(yaw/2)}}}}
        placed = yaml.safe_load(subprocess.run(
            ['rosservice', 'call', '/gazebo/set_model_state', json.dumps(req)],
            capture_output=True, text=True, check=True, timeout=5).stdout)
        if not placed['success']:
            raise RuntimeError(placed['status_message'])
        wait_for_controller(data, command)
        status_events.clear()
        start = time.monotonic()
        last_send = last_log = 0.
        initial_odom = None
        while not rospy.is_shutdown() and time.monotonic()-start < args.seconds+2:
            now = time.monotonic()
            elapsed = now-start
            if now-last_send >= .08:
                turn = -.3 if args.case in ('wall', 'override') else 0.
                if args.case == 'override':
                    if override_evidence is None:
                        override_evidence = override_trigger(records, start)
                        if override_evidence is not None:
                            override_evidence['sent_monotonic'] = now
                    if override_evidence is not None:
                        turn = .3*override_evidence['wall_side']
                if ((args.case == 'opening_turn' and elapsed > 4)
                        or (args.case == 'opening_gap' and elapsed > 2)):
                    # HTTP x is screen direction; joystick_to_velocity negates it.
                    turn = -.45 if args.wall_side == 'left' else .45
                if args.case == 'door' and args.active_align and 2. < elapsed < 6.:
                    turn = math.copysign(.22, args.lateral_m)
                command(turn if elapsed > 2 else 0., args.forward_y if elapsed > 2 else 0.)
                last_send = now
            time.sleep(.005)  # rospy dispatches subscriptions on its own threads.
            if now-last_log < .1 or len(scans) < 2 or 'odom' not in data:
                continue
            last_log = now
            clearances = []
            frontal_gaps = []
            wall_clearances = []
            scan_lines = []
            scanned = list(scans.items())
            for side, (msg, _) in scanned:
                ranges = np.array(msg.ranges)
                angles = msg.angle_min + np.arange(len(ranges))*msg.angle_increment
                valid = np.isfinite(ranges) & (ranges >= msg.range_min) & (ranges <= msg.range_max)
                x = ranges[valid]*np.cos(angles[valid])+LIDAR_X_M
                y = ranges[valid]*np.sin(angles[valid]) + LIDAR_Y_M[side]
                if args.case == 'front':
                    ahead = (x > .97) & (np.abs(y) < .4)
                    frontal_gaps.extend((x[ahead]-.97).tolist())
                dx = np.maximum(np.maximum(-.25-x, x-.97), 0.)
                dy = np.maximum(np.abs(y)-.4, 0.)
                clearances.extend(np.hypot(dx, dy).tolist())
                outside = ((x < -.25) | (x > .97) | (np.abs(y) > .4))
                scan_lines.extend(extract_lines(np.column_stack((x[outside], y[outside]))))
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
            row = sample_row(data, scanned, start)
            row['clearance'] = min(clearances, default=99.)
            if initial_odom is None:
                initial_odom = row['odom']
            if args.case == 'front':
                row['front_obstacle_gap'] = min(frontal_gaps, default=None)
            if args.case == 'override':
                line = override_wall_line(scan_lines)
                row['wall_side'] = math.copysign(1., line.distance) if line is not None else 0
                row['wall_heading'] = line.heading if line is not None else None
            if args.case.startswith('opening_') or args.case == 'door':
                row['openings'] = [{'center': list(opening.center), 'heading': opening.heading,
                                    'width': opening.width}
                                   for opening in find_openings(scan_lines)]
            if args.case == 'wall_gap':
                row['wall_clearance'] = min(wall_clearances, default=None)
            if args.case == 'door':
                relative = transform_points([row['odom'][:2]], initial_odom, inverse=True)[0]
                axle_start = (-2.5-.33*math.cos(yaw), args.lateral_m-.33*math.sin(yaw), yaw)
                row['world_axle'] = transform_points([relative], axle_start)[0].tolist()
                row['world_yaw'] = row['odom'][2]-initial_odom[2]+yaw
            elif args.case.startswith('opening_'):
                relative = transform_points([row['odom'][:2]], initial_odom, inverse=True)[0]
                row['world_axle'] = transform_points([relative], opening_axle_start)[0].tolist()
                row['world_yaw'] = row['odom'][2]-initial_odom[2]+yaw
            records.append(row)
        command()
        released = time.monotonic()
        # Observe physical settling after release, not just the sent zero request.
        time.sleep(.2)
        stop_deadline = time.monotonic() + 3.
        while time.monotonic() < stop_deadline:
            if settled_after_release(data, released, time.monotonic()):
                break
            time.sleep(.05)
        stopped = dict(data)
        settled_at = time.monotonic()
        control_events = control_event_window(status_events, start, settled_at)
        moving = moving_records(records)
        sequence = []
        for entry in mode_history:
            if not sequence or sequence[-1] != entry['mode']:
                sequence.append(entry['mode'])
        result = {'case': args.case, 'forward_y': args.forward_y, 'samples': len(moving),
                  'min_sampled_clearance': min((r['clearance'] for r in moving), default=0.),
                  'max_speed': max((r.get('cmd_vel', [0])[0] for r in moving), default=0.),
                  'modes': sorted({r['mode'] for r in mode_history}
                                  | {r.get('status', {}).get('mode', 'missing') for r in moving}),
                  'mode_sequence': sequence,
                  'final_stop': {key: stopped.get(key) for key in (
                      'cmd_vel', 'odom_velocity', 'cmd_vel_received', 'odom_velocity_received')},
                  'released': released,
                  'settled_at': settled_at,
                  'last': records[-1] if records else None}
        result['motion_metrics'] = motion_metrics(moving)
        if args.case == 'override':
            result['override_trigger'] = override_evidence
        result['control_event_metrics'] = control_event_metrics(control_events)
        result['active_control_event_metrics'] = control_event_metrics(
            [event for event in control_events if event['monotonic'] < released])
        result['control_event_window'] = dict(started=start, released=released, ended=settled_at)
        if args.case == 'wall_gap':
            gaps = [r['wall_clearance'] for r in moving if r['wall_clearance'] is not None
                    and r.get('status', {}).get('mode') == 'wall' and r.get('cmd_vel', [0])[0] > .1]
            result['steady_gap_samples'] = len(gaps)
            result['steady_gap_min_max'] = [min(gaps), max(gaps)] if gaps else None
        if args.case.startswith('opening_') and moving:
            result['yaw_change'] = math.atan2(math.sin(moving[-1]['odom'][2]-initial_odom[2]),
                                              math.cos(moving[-1]['odom'][2]-initial_odom[2]))
        if args.case == 'opening_gap' and moving:
            result.update(opening_gap_metrics(moving, 1. if args.wall_side == 'left' else -1., initial_odom))
        Path(args.output).write_text(json.dumps({'summary': result, 'records': records,
                                               'mode_history': mode_history,
                                               'control_events': control_events}, indent=2)+'\n')
        print(json.dumps(result, indent=2))
        assert moving and result['min_sampled_clearance'] > .02, 'missing data or insufficient sampled clearance'
        assert settled_after_release(stopped, released, settled_at), \
            'release did not produce fresh stopped command and odometry samples'
        assert result['max_speed'] <= .800001, 'assisted speed limit exceeded'
        if args.case == 'door':
            check_door_modes(result['modes'])
            assert 'door_clear' in sequence, 'door tail-clear phase was not observed'
            assert sequence.index('door_align') < sequence.index('door_pass') < sequence.index('door_clear'), \
                'door phases were observed out of order'
            assert any(r['world_axle'][0] > .6 for r in moving), 'rear axle did not clear doorway'
        elif args.case == 'front':
            check_front_stop(records, start)
        elif args.case == 'wall':
            check_wall_result(result)
        elif args.case == 'override':
            check_override_response(records, override_evidence, start, control_events)
        elif args.case == 'vertical':
            assert math.dist(initial_odom[:2], moving[-1]['odom'][:2]) > 2., 'vertical corridor progress too small'
        elif args.case == 'room_door':
            progress = transform_points([moving[-1]['odom'][:2]], initial_odom, inverse=True)[0, 0]
            assert progress > 2.5 and any(mode.startswith('door_') for mode in result['modes']), \
                'NW room door was not cleared'
        elif args.case == 'wall_gap':
            assert len(gaps) >= 10 and max(gaps) <= .15, 'steady body-edge wall gap exceeds 15 cm'
        elif args.case == 'opening_turn':
            intended_sign = 1. if args.wall_side == 'left' else -1.
            assert 'opening_turn' in result['modes'], 'intended opening turn was not accepted'
            assert intended_sign*result['yaw_change'] > .40, 'wheelchair did not turn into the opening'
        elif args.case == 'opening_straight':
            assert 'opening_turn' not in result['modes'], 'opening captured a straight command'
            assert abs(result['yaw_change']) < .25, 'straight command turned into the opening'
        elif args.case == 'opening_gap':
            check_opening_gap_result(result, 1. if args.wall_side == 'left' else -1.)
    finally:
        try:
            command()
        finally:
            rospy.signal_shutdown('probe complete')


if __name__ == '__main__':
    main()
