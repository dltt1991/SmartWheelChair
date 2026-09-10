"""Shared-control references for the local follower, with an independent guard."""

from concurrent.futures import ThreadPoolExecutor
import copy
import json
import math
import threading
import time
import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped, TransformStamped, Twist, TwistStamped
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, Float32, String
from tf2_ros import Buffer, TransformBroadcaster, StaticTransformBroadcaster, TransformListener, TransformException

from smart_wheelchair_safety.unified_geometry import (
    Opening, active_aperture_targeted, angle_difference, approach_path, arc_path, braking_clear,
    collision_aware_door_reference, door_alignment_reference, door_entry_clearance,
    extract_lines, extract_opening_lines,
    find_openings, intended_front_door, intended_side_opening, opening_matches, transform_points,
    wall_reference,
)


DOOR_INTENT_CORRIDOR_TOLERANCE = .47
DOOR_ENTRY_NOISE_TOLERANCE = .01
DOOR_CREEP_SPEED = .05
DOOR_OBSTACLE_MEMORY_TIMEOUT = 1.5
OPENING_TURN_TIMEOUT = 12.
OPENING_TURN_REAR_CLEARANCE = .50
REFERENCE_PERIOD = .2
JERK_LIMITS = np.array([2.5, 4.0])


class UnifiedControlNode:
    def __init__(self):
        defaults = {'max_speed': .8, 'scan_timeout': .45,
                    'odom_timeout': .10,
                    'command_timeout': .3, 'braking_deceleration': .5,
                    'reaction_time': .2, 'hard_margin': .04,
                    'wall_clearance': .12,
                    'front_stop_margin': .12}
        self.parameters = {
            name: float(rospy.get_param('~' + name, default))
            for name, default in defaults.items()
        }
        if any(not math.isfinite(value) or value <= 0.
               for value in self.parameters.values()):
            raise ValueError('control parameters must be finite and positive')
        # rospy dispatches subscribers and timers on separate threads. Preserve
        # the serialized state transitions of the original executor.
        self.callback_lock = threading.RLock()
        self.stopped = False
        self.raw = np.zeros(2)
        self.measured = np.zeros(2)
        self.planned = np.zeros(2)
        self.output = np.zeros(2)
        self.acceleration = np.zeros(2)
        self.pose = None
        self.raw_time = self.odom_time = self.plan_time = 0.
        self.planned_after_stamp = -1
        self.odom_stamp = None
        self.scans = {}
        self.door = None
        self.door_phase = None
        self.door_path = None
        self.door_path_feasible = False
        self.door_plan_executor = ThreadPoolExecutor(max_workers=1,
                                                     thread_name_prefix='door_path')
        self.door_plan_request = None
        self.door_generation = 0
        self.door_away_since = 0.
        self.door_obstacles = np.empty((0, 2))
        self.door_obstacle_time = 0.
        self.opening_candidates = []
        self.confirmed_openings = []
        self.opening_observation_time = 0.
        self.opening_turn = None
        self.opening_turn_side = 0
        self.opening_turn_heading = 0.
        self.opening_turn_origin = None
        self.opening_turn_time = 0.
        self.mode = 'waiting'
        self.front_blocked = False
        self.wall_side = 0
        self.wall_preference = 0
        self.wall_preview_heading = None
        self.assist_enabled = True
        self.mode_neutral_seen = True
        self.accept_planned = False
        self.override = False
        self.reason = 'starting'
        self.epoch = 0
        self.last_tick = rospy.Time.now().to_sec()
        self.tf = Buffer()
        self.listener = TransformListener(self.tf)
        self.transforms = TransformBroadcaster()
        self.static = StaticTransformBroadcaster()
        frames = []
        for side, y in [('left', .26), ('right', -.26)]:
            t = TransformStamped()
            t.header.frame_id = 'rear_axle'
            t.child_frame_id = f'unified_lidar_{side}'
            t.transform.translation.x = .79
            t.transform.translation.y = y
            t.transform.rotation.w = 1.
            frames.append(t)
        self.static.sendTransform(frames)
        self.pub = rospy.Publisher('cmd_vel', Twist, queue_size=10)
        self.status = rospy.Publisher('shared_control/status', String, queue_size=10)
        self.reference = rospy.Publisher('shared_control/reference', Path, queue_size=10)
        self.limit_pub = rospy.Publisher('speed_limit', Float32, queue_size=10)
        self.scan_pubs = {side: rospy.Publisher(f'unified_scan_{side}', LaserScan, queue_size=10)
                          for side in ('left', 'right')}
        self.publishers = [self.pub, self.status, self.reference, self.limit_pub,
                           *self.scan_pubs.values()]
        self.subscribers = [
            rospy.Subscriber(topic, message, self._serialized(callback), queue_size=10)
            for topic, message, callback in (
                ('cmd_vel_raw', Twist, self.on_raw),
                ('cmd_vel_planned', TwistStamped, self.on_planned),
                ('assist_enabled', Bool, self.on_assist_enabled),
                ('odom', Odometry, self.on_odom))
        ]
        for side in ('left', 'right'):
            self.subscribers.append(rospy.Subscriber(
                f'scan_{side}', LaserScan,
                self._serialized(lambda msg, side=side: self.on_scan(msg, side)), queue_size=1))
        self.reference_timer = rospy.Timer(
            rospy.Duration(REFERENCE_PERIOD), self._serialized(lambda _: self.update_reference()))
        self.control_timer = rospy.Timer(
            rospy.Duration(.05), self._serialized(lambda _: self.control()))

    def parameter(self, name):
        return self.parameters[name]

    def _serialized(self, callback):
        def invoke(*args):
            with self.callback_lock:
                if not self.stopped:
                    return callback(*args)
        return invoke

    def on_raw(self, msg):
        self.raw = np.array([msg.linear.x, msg.angular.z])
        self.raw_time = time.monotonic()
        if np.linalg.norm(self.raw) < .01:
            self.mode_neutral_seen = True
        if not self.assist_enabled or not self.mode_neutral_seen:
            return
        if self.raw[0] <= .02:
            self.wall_preference = 0
        elif abs(self.raw[1]) >= .12:
            self.wall_preference = math.copysign(1., self.raw[1])
        if (self.opening_turn is not None
                and (self.raw[0] <= .02 or self.opening_turn_side*self.raw[1] < -.12)):
            self._clear_opening_turn()
        if self.raw[0] <= .02:
            self.front_blocked = False
            self.opening_candidates = []
            self.confirmed_openings = []
        intended, _ = self._intended_door()
        side_opening = None
        if self.pose is not None and self.raw[0] > .02 and abs(self.raw[1]) >= .12:
            local_openings = [self._local_opening(opening)
                              for opening in self.confirmed_openings]
            side_opening = intended_side_opening(
                local_openings, math.copysign(1., self.raw[1]), *self.raw)
        door_cancelled = False
        if self.raw[0] <= .02:
            self._clear_door()
        elif self.door is not None:
            local = self._local_opening(self.door)
            targeted = active_aperture_targeted(local, *self.raw)
            if targeted is None:
                heading_error = angle_difference(local.heading, 0.)
                # Ignore millimetre offsets even near the plane; a center behind
                # the axle is no longer a forward steering target.
                bearing_error = (math.atan2(local.center[1], local.center[0])
                                 if local.center[0] > 0. and abs(local.center[1]) > .01
                                 else 0.)
                targeted = any(abs(error) > .02 and self.raw[1]*error > 0.
                               for error in (heading_error, bearing_error))
            away = abs(self.raw[1]) > .25 and not targeted
            if not away:
                self.door_away_since = 0.
            elif not self.door_away_since:
                self.door_away_since = time.monotonic()
            elif time.monotonic()-self.door_away_since >= .35:
                self._clear_door()
                door_cancelled = True
        self.override = (door_cancelled
                         or (self.door is None and intended is None
                             and side_opening is None
                             and self.wall_side * self.raw[1] < -.15)
                         or (self.door is None and intended is None
                             and side_opening is None
                             and self.mode == 'override' and abs(self.raw[1]) > .25))

    def on_assist_enabled(self, msg):
        enabled = bool(msg.data)
        if enabled == self.assist_enabled:
            return
        self.assist_enabled = enabled
        self.mode_neutral_seen = False
        self.raw[:] = 0.
        self.planned[:] = 0.
        self.plan_time = 0.
        self.accept_planned = False
        self.output[:] = 0.
        self.acceleration[:] = 0.
        self._clear_opening_turn()
        self._clear_door()
        self.cancel()
        self.opening_candidates = []
        self.confirmed_openings = []
        self.door_obstacles = np.empty((0, 2))
        self.door_obstacle_time = 0.
        self.front_blocked = False
        self.wall_side = 0
        self.wall_preference = 0
        self.wall_preview_heading = None
        self.override = False
        self.mode = 'manual' if enabled else 'manual_direct'
        self.pub.publish(Twist())

    def on_planned(self, msg):
        stamp = msg.header.stamp.to_sec()
        if (not self.assist_enabled or not self.mode_neutral_seen
                or not self.accept_planned
                or stamp <= self.planned_after_stamp):
            return
        self.planned = np.array([msg.twist.linear.x, msg.twist.angular.z])
        self.plan_time = time.monotonic()

    def on_odom(self, msg):
        q = msg.pose.pose.orientation
        p = msg.pose.pose.position
        age = (rospy.Time.now().to_sec() - msg.header.stamp.to_sec())
        values = [p.x, p.y, q.x, q.y, q.z, q.w, msg.twist.twist.linear.x, msg.twist.twist.angular.z]
        if not -.05 <= age <= self.parameter('odom_timeout') or not np.isfinite(values).all():
            return
        if abs(q.x*q.x+q.y*q.y+q.z*q.z+q.w*q.w-1.) > .01:
            return
        yaw = math.atan2(2*(q.w*q.z + q.x*q.y), 1-2*(q.y*q.y + q.z*q.z))
        self.pose = (p.x, p.y, yaw)
        self.measured = np.array([msg.twist.twist.linear.x, msg.twist.twist.angular.z])
        self.odom_time = time.monotonic()
        self.odom_stamp = msg.header.stamp.to_sec()
        t = TransformStamped()
        t.header = msg.header
        t.header.frame_id = 'odom'
        t.child_frame_id = 'rear_axle'
        t.transform.translation.x, t.transform.translation.y = p.x, p.y
        t.transform.rotation = q
        self.transforms.sendTransform(t)

    def on_scan(self, msg, side):
        scan_age = (rospy.Time.now().to_sec() - msg.header.stamp.to_sec())
        if not -.05 <= scan_age <= self.parameter('scan_timeout'):
            return
        if (not np.isfinite([msg.angle_min, msg.angle_increment, msg.range_min, msg.range_max]).all()
                or msg.angle_increment == 0. or not 0. <= msg.range_min < msg.range_max):
            return
        ranges = np.array(msg.ranges)
        healthy = np.isposinf(ranges) | (np.isfinite(ranges) & (ranges >= msg.range_min) & (ranges <= msg.range_max))
        if len(ranges) < 3 or np.mean(healthy) < .9:
            return
        outgoing = copy.deepcopy(msg)
        outgoing.header.frame_id = f'unified_lidar_{side}'
        self.scan_pubs[side].publish(outgoing)
        try:
            t = self.tf.lookup_transform('odom', outgoing.header.frame_id,
                                         msg.header.stamp, rospy.Duration(.15))
        except TransformException:
            return
        angles = msg.angle_min + np.arange(len(ranges)) * msg.angle_increment
        valid = np.isfinite(ranges) & (ranges >= msg.range_min) & (ranges <= min(msg.range_max, 4.5))
        local = np.column_stack((ranges[valid]*np.cos(angles[valid]), ranges[valid]*np.sin(angles[valid])))
        q = t.transform.rotation
        pose = (t.transform.translation.x, t.transform.translation.y, 2*math.atan2(q.z, q.w))
        world = transform_points(local, pose)
        # Remove known self returns at acquisition, never newly approached
        # obstacles using the current footprint after the chair has moved.
        axle_local = local + [.79, .26 if side == 'left' else -.26]
        outside = ((axle_local[:, 0] < -.25) | (axle_local[:, 0] > .97)
                   | (np.abs(axle_local[:, 1]) > .4))
        world = world[outside]
        self.scans[side] = (time.monotonic(), world)
        now = time.monotonic()
        if self.pose is not None and len(self.scans) == 2 and now-self.opening_observation_time >= .08:
            groups, _ = self.points()
            lines = [line for group in groups for line in extract_opening_lines(group)]
            self._observe_openings(find_openings(lines))
            self.opening_observation_time = now

    def _world_opening(self, opening):
        center = transform_points([opening.center], self.pose)[0]
        return Opening(tuple(center), opening.heading+self.pose[2], opening.width, ())

    def _local_opening(self, opening):
        center = transform_points([opening.center], self.pose, inverse=True)[0]
        heading = angle_difference(opening.heading, self.pose[2])
        return Opening(tuple(center), heading, opening.width, ())

    def _local_door_path(self, door, obstacles):
        if self.door_path is None:
            phase = 'align' if self.door_phase == 'door_align' else 'pass'
            if phase == 'pass':
                local = door_alignment_reference(door, phase)
                self.door_path_feasible = True
                plan_pose = self.pose
            else:
                local = None
                plan_pose = None
                if self.door_plan_request is not None:
                    future, requested_pose, generation = self.door_plan_request
                    if future.done():
                        self.door_plan_request = None
                        if generation == self.door_generation:
                            try:
                                local = future.result()
                                plan_pose = requested_pose
                            except Exception as error:
                                rospy.logerr('door path search failed: %s', error)
                if local is None and self.door_plan_request is None:
                    plan_pose = tuple(self.pose)
                    future = self.door_plan_executor.submit(
                        collision_aware_door_reference, door,
                        np.array(obstacles, dtype=float, copy=True))
                    self.door_plan_request = (future, plan_pose, self.door_generation)
                self.door_path_feasible = local is not None
                if local is None:
                    return door_alignment_reference(door, phase)
            world_xy = transform_points(local[:, :2], plan_pose)
            self.door_path = np.column_stack((world_xy, local[:, 2]+plan_pose[2]))
        local_xy = transform_points(self.door_path[:, :2], self.pose, inverse=True)
        headings = [angle_difference(heading, self.pose[2]) for heading in self.door_path[:, 2]]
        return np.column_stack((local_xy, headings))

    def _door_preview_angular(self, linear):
        local = transform_points(self.door_path[:, :2], self.pose, inverse=True)
        segments = np.linalg.norm(np.diff(self.door_path[:, :2], axis=0), axis=1)
        delta = np.diff(local, axis=0)
        fraction = np.clip(-np.sum(local[:-1]*delta, axis=1)/(segments*segments), 0., 1.)
        projections = local[:-1] + fraction[:, None]*delta
        nearest = int(np.argmin(np.linalg.norm(projections, axis=1)))
        turns = np.arctan2(np.sin(np.diff(self.door_path[:, 2])),
                           np.cos(np.diff(self.door_path[:, 2])))
        reference_heading = self.door_path[nearest, 2] + fraction[nearest]*turns[nearest]
        heading_error = angle_difference(reference_heading, self.pose[2])
        curvature = (turns[nearest]/segments[nearest]
                     + 2.*heading_error + 4.*projections[nearest, 1])
        angular = linear*curvature
        return float(np.clip(angular, -.65, .65))

    def _door_angular(self, linear):
        path_angular = self._door_preview_angular(linear)
        if self.door_phase not in ('door_pass', 'door_clear') or self.door is None:
            return path_angular
        local = self._local_opening(self.door)
        normal = np.array([math.cos(local.heading), math.sin(local.heading)])
        progress = -np.asarray(local.center) @ normal
        joystick_weight = float(np.clip(progress/.29, 0., 1.))
        return (1.-joystick_weight)*path_angular + joystick_weight*self.raw[1]

    def _safe_door_command(self, points, stamp):
        cruise = .35 if self.mode == 'door_align' else .55
        high = min(self.raw[0], cruise)
        best = np.zeros(2)
        state_age = max(0., stamp-self.odom_stamp)
        for _ in range(7):
            speed = (best[0]+high)/2
            candidate = np.array([speed, self._door_angular(speed)])
            if braking_clear(points, candidate, self.measured,
                             margin=self.parameter('hard_margin'),
                             reaction=self.parameter('reaction_time'),
                             deceleration=self.parameter('braking_deceleration'),
                             state_age=state_age):
                best = candidate
            else:
                high = speed
        return best

    def _observe_openings(self, openings):
        if self.pose is None:
            return
        now = time.monotonic()
        observed = [self._world_opening(opening) for opening in openings]
        updated = []
        confirmed = []
        for opening in self.confirmed_openings:
            local = self._local_opening(opening)
            if local.center[0] >= -.5 and np.linalg.norm(local.center) <= 5.:
                confirmed.append(opening)
        for opening in observed:
            previous = next(((candidate, hits) for candidate, hits, stamp in self.opening_candidates
                             if now-stamp <= .6 and opening_matches(candidate, opening)), None)
            hits = previous[1]+1 if previous is not None else 1
            updated.append((opening, hits, now))
            if hits >= 2:
                match = next((index for index, tracked in enumerate(confirmed)
                              if opening_matches(tracked, opening)), None)
                if match is None:
                    confirmed.append(opening)
        for candidate, hits, stamp in self.opening_candidates:
            if (now-stamp <= .6 and not any(opening_matches(candidate, opening)
                                            for opening in observed)):
                updated.append((candidate, hits, stamp))
        self.opening_candidates = updated
        self.confirmed_openings = confirmed

    def _clear_opening_turn(self):
        self.opening_turn = None
        self.opening_turn_side = 0
        self.opening_turn_heading = 0.
        self.opening_turn_origin = None
        self.opening_turn_time = 0.

    def _clear_door(self):
        if self.door is not None:
            self.door_generation += 1
            if len(self.door_obstacles):
                self.door_obstacle_time = time.monotonic()
        self.door = None
        self.door_phase = None
        self.door_path = None
        self.door_path_feasible = False
        self.door_away_since = 0.

    def _refresh_door_obstacles(self, observed):
        # While both jambs are visible, replace the frame instead of taking a
        # temporal union: one inward range outlier must not become permanent.
        if len(observed) < 4:
            return
        _, indices = np.unique(np.floor(observed/.025).astype(int), axis=0,
                               return_index=True)
        self.door_obstacles = observed[indices]
        self.door_obstacle_time = time.monotonic()

    @staticmethod
    def _filtered_opening(current, observed, alpha=.25):
        center = (1-alpha)*np.asarray(current.center) + alpha*np.asarray(observed.center)
        heading = current.heading + alpha*angle_difference(observed.heading, current.heading)
        width = (1-alpha)*current.width + alpha*observed.width
        return Opening(tuple(center), heading, width, ())

    def _intended_door(self):
        if self.pose is None:
            return None, None
        local = [self._local_opening(opening) for opening in self.confirmed_openings]
        tolerance = min(.60, DOOR_INTENT_CORRIDOR_TOLERANCE + .16*abs(self.raw[1]))
        selected = intended_front_door(
            local, *self.raw,
            corridor_tolerance=tolerance)
        if selected is None:
            return None, None
        return self.confirmed_openings[local.index(selected)], selected

    def _pending_door_intent(self):
        now = time.monotonic()
        local = [self._local_opening(opening) for opening, _, stamp in self.opening_candidates
                 if now-stamp <= .6]
        tolerance = min(.60, DOOR_INTENT_CORRIDOR_TOLERANCE + .16*abs(self.raw[1]))
        return intended_front_door(
            local, *self.raw,
            corridor_tolerance=tolerance) is not None

    def _local_tracked_door(self):
        if self.door is None:
            return None
        if self.door_phase == 'door_align':
            observed = next((opening for opening in self.confirmed_openings
                             if opening_matches(self.door, opening)), None)
            if observed is not None:
                self.door = self._filtered_opening(self.door, observed)
        local = self._local_opening(self.door)
        normal = np.array([math.cos(local.heading), math.sin(local.heading)])
        progress = -np.asarray(local.center) @ normal
        if self.door_phase in ('door_pass', 'door_clear'):
            if progress > .29:
                self._clear_door()
                self.door_obstacles = np.empty((0, 2))
                self.door_obstacle_time = 0.
                return None
            if progress > 0.:
                self.door_phase = 'door_clear'
        return local

    def _start_opening_turn(self, opening, side):
        self.opening_turn = self._world_opening(opening)
        self.opening_turn_side = side
        self.opening_turn_heading = angle_difference(
            self.opening_turn.heading-side*math.pi/2, 0.)
        self.opening_turn_origin = np.asarray(self.pose[:2])
        self.opening_turn_time = time.monotonic()
        self.wall_side = 0
        self.wall_preference = 0

    def _opening_turn_path(self):
        if self.opening_turn is None:
            return None
        local = self._local_opening(self.opening_turn)
        expired = time.monotonic()-self.opening_turn_time >= OPENING_TURN_TIMEOUT
        turned = abs(angle_difference(self.pose[2], self.opening_turn_heading)) >= 1.05
        if expired or turned or local.center[0] < -.25:
            self._clear_opening_turn()
            return None
        if not self._opening_turn_ready(local):
            return arc_path(max(self.raw[0], .1), 0.)
        normal = np.array([math.cos(local.heading), math.sin(local.heading)])
        return approach_path(np.asarray(local.center) + 1.2*normal, local.heading)

    def _opening_turn_ready(self, local=None):
        if self.opening_turn is None:
            return False
        local = local or self._local_opening(self.opening_turn)
        normal = np.array([math.cos(local.heading), math.sin(local.heading)])
        tangent = np.array([-normal[1], normal[0]])
        if tangent[0] < 0.:
            tangent = -tangent
        near_edge = np.asarray(local.center) @ tangent - local.width/2
        # The inner jamb must be behind the rear axle before steering into the
        # branch. Clearing it with only the front lets the rear corner sweep
        # into the wall during a differential-drive turn.
        return near_edge <= -OPENING_TURN_REAR_CLEARANCE

    def points(self):
        if self.pose is None:
            return [], []
        groups = []
        for _, world in self.scans.values():
            local = transform_points(world, self.pose, inverse=True)
            groups.append(local)
        guard_groups = groups + [transform_points(self.door_obstacles, self.pose, inverse=True)]
        return groups, np.vstack(guard_groups)

    def fresh(self):
        now = time.monotonic()
        return (np.isfinite(self.raw).all() and np.isfinite(self.measured).all()
                and self.pose is not None and np.isfinite(self.pose).all()
                and now-self.raw_time < self.parameter('command_timeout')
                and self.odom_stamp is not None
                and -.05 <= rospy.Time.now().to_sec()-self.odom_stamp < self.parameter('odom_timeout')
                and now-self.odom_time < self.parameter('odom_timeout') and len(self.scans) == 2
                and all(now-stamp < self.parameter('scan_timeout') for stamp, _ in self.scans.values()))

    def cancel(self):
        self.epoch += 1
        self.accept_planned = False
        self.planned_after_stamp = rospy.Time.now().to_sec()
        self.planned[:] = 0.
        self.plan_time = 0.

    def update_reference(self):
        self.wall_preview_heading = None
        if not self.assist_enabled:
            self.mode = 'manual_direct'
            return
        if not self.mode_neutral_seen:
            self.mode = 'manual'
            return
        if len(self.door_obstacles) and self.pose is not None:
            if (self.door is None
                    and time.monotonic()-self.door_obstacle_time >= DOOR_OBSTACLE_MEMORY_TIMEOUT):
                self.door_obstacles = np.empty((0, 2))
                self.door_obstacle_time = 0.
            else:
                nearby = np.linalg.norm(self.door_obstacles-np.array(self.pose[:2]), axis=1) < 4.5
                self.door_obstacles = self.door_obstacles[nearby]
        if not self.fresh() or self.raw[0] <= .02:
            self._clear_opening_turn()
            self._clear_door()
            self.cancel()
            self.mode = 'manual'
            return
        if self.override:
            self.cancel()
            self.mode = 'override'
            return
        groups, obstacle_points = self.points()
        lines = [line for group in groups for line in extract_lines(group)]
        opening_path = self._opening_turn_path()
        local_door = self._local_tracked_door()
        if local_door is None:
            world_door, local_door = self._intended_door()
            if world_door is not None:
                self.door_generation += 1
                self.door = world_door
                self.door_phase = 'door_align'
        if local_door is not None and self.door_phase == 'door_align':
            # Keep observed jambs through rear clearance even if the
            # forward-mounted scanners temporarily lose their inner edges.
            points = np.vstack(groups)
            normal = np.array([math.cos(local_door.heading), math.sin(local_door.heading)])
            near_plane = np.abs((points-local_door.center) @ normal) < .3
            observed = transform_points(points[near_plane], self.pose)
            self._refresh_door_obstacles(observed)
        frontal_wall = any(abs(line.heading) > 1.35 and .97 < -line.distance/math.sin(line.heading) < 3.
                           and min(line.start[1], line.end[1]) < -.4
                           and max(line.start[1], line.end[1]) > .4 for line in lines)
        if opening_path is not None:
            local_path = opening_path
            self.mode = 'opening_turn'
            self.wall_side = 0
            self.wall_preference = 0
        elif local_door is not None:
            self.front_blocked = False
            self.wall_side = 0
            self.wall_preference = 0
            door_path = self._local_door_path(local_door, obstacle_points)
            if not self.door_path_feasible:
                local_path = arc_path(max(self.raw[0], .1), self.raw[1])
                self.mode = 'manual'
            else:
                if not self.mode.startswith('door_'):
                    self.cancel()
                local_path = door_path
                if self.door_phase == 'door_align':
                    if door_entry_clearance(local_door) > 0.:
                        self.door_phase = 'door_pass'
                self.mode = self.door_phase
        elif self._pending_door_intent():
            self.front_blocked = False
            self.wall_side = 0
            self.wall_preference = 0
            local_path = arc_path(max(self.raw[0], .1), self.raw[1])
            self.mode = 'manual'
        elif frontal_wall or self.front_blocked:
            self.front_blocked = True
            self.wall_side = 0
            self.wall_preference = 0
            self.mode = 'front_stop'
            self.cancel()
            return
        else:
            local_path, self.mode, self.wall_side = wall_reference(
                lines, *self.raw,
                body_clearance=self.parameter('wall_clearance'),
                preferred_side=self.wall_preference)
            if self.mode == 'wall':
                self.wall_preference = self.wall_side
                distance = np.linalg.norm(local_path[:, :2], axis=1)
                preview = int(np.argmin(np.abs(distance-.8)))
                self.wall_preview_heading = angle_difference(
                    local_path[preview, 2]+self.pose[2], 0.)
            local_openings = [self._local_opening(opening) for opening in self.confirmed_openings]
            selected = intended_side_opening(local_openings, self.wall_side, *self.raw)
            if self.mode == 'wall' and selected is not None:
                side = self.wall_side
                self._start_opening_turn(selected, side)
                local_path = arc_path(max(self.raw[0], .1), self.raw[1])
                self.mode = 'opening_turn'
        world_xy = transform_points(local_path[:, :2], self.pose)
        path = Path()
        path.header.frame_id = 'odom'
        path.header.stamp = rospy.Time.now()
        for xy, angle in zip(world_xy, local_path[:, 2] + self.pose[2]):
            p = PoseStamped()
            p.header = path.header
            p.pose.position.x, p.pose.position.y = float(xy[0]), float(xy[1])
            p.pose.orientation.z, p.pose.orientation.w = math.sin(angle/2), math.cos(angle/2)
            path.poses.append(p)
        self.reference.publish(path)
        limit = min(self.raw[0], self.parameter('max_speed'),
                    .35 if self.mode == 'door_align'
                    else .55 if self.mode in ('door_pass', 'door_clear')
                    else .5 if self.mode == 'opening_turn' else 10.)
        self.limit_pub.publish(Float32(data=limit))
        self.accept_planned = True
        self.planned_after_stamp = rospy.Time.now().to_sec()

    def control(self):
        now = time.monotonic()
        stamp = rospy.Time.now().to_sec()
        dt = min(.1, max(.001, stamp-self.last_tick))
        self.last_tick = stamp
        command = np.zeros(2)
        self.reason = 'clear'
        if not self.assist_enabled or not self.mode_neutral_seen:
            fresh_raw = (np.isfinite(self.raw).all()
                         and now-self.raw_time
                         < self.parameter('command_timeout'))
            if not self.assist_enabled and self.mode_neutral_seen and fresh_raw:
                command = self.raw.copy()
            self.mode = 'manual_direct' if not self.assist_enabled else 'manual'
            if np.linalg.norm(command) >= .01:
                self.reason = 'clear'
            elif not self.mode_neutral_seen:
                self.reason = 'mode_transition'
            elif fresh_raw:
                self.reason = 'user_stop'
            else:
                self.reason = 'stale_input'
            self.acceleration[:] = 0.
            self.output = command
            msg = Twist()
            msg.linear.x, msg.angular.z = map(float, command)
            self.pub.publish(msg)
            self.status.publish(String(data=json.dumps({
                'mode': self.mode, 'reason': self.reason,
                'v': msg.linear.x, 'w': msg.angular.z,
            })))
            return
        planner_stale = now-self.plan_time >= .25 or not np.isfinite(self.planned).all()
        door_fallback = self.mode.startswith('door_') and self.door_path_feasible
        if not self.fresh():
            self.reason = 'stale_input'
        elif np.linalg.norm(self.raw) < .01:
            self.reason = 'user_stop'
        elif (self.raw[0] > .02 and not self.override and self.mode != 'front_stop'
              and planner_stale and not door_fallback):
            self.reason = 'planner_timeout'
        else:
            _, points = self.points()
            if self.raw[0] <= .02 or self.override or self.mode == 'front_stop':
                desired = self.raw.copy()
                if self.mode == 'front_stop' and self.raw[0] > .02 and not self.override:
                    desired[1] = 0.
                    front = points[(points[:, 0] > 0.) & (np.abs(points[:, 1]) <= .44)]
                    gap = (float(front[:, 0].min())-.97-self.parameter('front_stop_margin')
                           if len(front) else 0.)
                    a = self.parameter('braking_deceleration')
                    # Also reserve the ramp to comfortable deceleration at
                    # the normal linear jerk limit of 1 m/s^3.
                    ar = a*(self.parameter('reaction_time')+a/1.)
                    allowed = math.sqrt(ar*ar+2*a*max(0., gap))-ar if gap > .01 else 0.
                    desired[0] = min(desired[0], allowed)
            elif door_fallback:
                desired = self._safe_door_command(points, stamp)
                if planner_stale:
                    self.reason = 'planner_fallback'
            elif now-self.plan_time < .25 and np.isfinite(self.planned).all():
                desired = self.planned.copy()
                if self.mode.startswith('door_'):
                    if (desired[0] <= .02 and self.mode == 'door_align'
                            and self.door is not None
                            and door_entry_clearance(self._local_opening(self.door))
                            > -DOOR_ENTRY_NOISE_TOLERANCE):
                        desired[0] = min(self.raw[0], DOOR_CREEP_SPEED)
                    if desired[0] <= .02:
                        desired[1] = 0.
                    elif self.door_path is not None:
                        desired[1] = self._door_angular(desired[0])
                elif self.mode == 'opening_turn':
                    if not self._opening_turn_ready():
                        lateral_error = 0.
                        if self.opening_turn_origin is not None:
                            left = np.array([-math.sin(self.opening_turn_heading),
                                             math.cos(self.opening_turn_heading)])
                            lateral_error = ((np.asarray(self.pose[:2])-self.opening_turn_origin)
                                             @ left)
                        preview_heading = (self.opening_turn_heading
                                           + math.atan2(-lateral_error, .8))
                        heading_error = angle_difference(self.pose[2], preview_heading)
                        desired[1] = np.clip(-1.5*heading_error, -.2, .2)
                    elif self.opening_turn_side*self.raw[1] > .12:
                        intended = min(abs(self.raw[1]), .5)
                        desired[1] = self.opening_turn_side * max(
                            self.opening_turn_side*desired[1], intended)
                elif (self.mode == 'wall' and self.wall_preview_heading is not None
                      and abs(self.raw[1]) < .12):
                    heading_error = angle_difference(
                        self.pose[2], self.wall_preview_heading)
                    correction = float(np.clip(-1.5*heading_error, -.2, .2))
                    away = -self.wall_side
                    if away*correction > .01:
                        desired[1] = away*max(away*desired[1], away*correction)
            else:
                desired = np.zeros(2)
                self.reason = 'planner_timeout'
            desired[0] = np.clip(desired[0], -.4 if self.raw[0] < 0. else 0.,
                                 min(max(0., self.raw[0]), self.parameter('max_speed')))
            desired[1] = np.clip(desired[1], -.65, .65)
            # Slew-limit acceleration; snap at the target to avoid overshoot.
            # The independent safety rejection below bypasses comfort limits.
            target_a = np.clip((desired-self.output)/dt, [-.5, -.8], [.5, .8])
            self.acceleration += np.clip(target_a-self.acceleration,
                                         -JERK_LIMITS*dt, JERK_LIMITS*dt)
            command = self.output + self.acceleration*dt
            command = np.minimum(np.maximum(command, np.minimum(self.output, desired)), np.maximum(self.output, desired))
            if door_fallback:
                command[1] = self._door_angular(command[0]) if command[0] > .002 else 0.
                self.acceleration[1] = 0.
            for scale in (1., .8, .6, .4, .2, 0.):
                candidate = command * scale
                if braking_clear(points, candidate, self.measured,
                                 margin=self.parameter('hard_margin'),
                                 reaction=self.parameter('reaction_time'),
                                 deceleration=self.parameter('braking_deceleration'),
                                 state_age=max(0., stamp-self.odom_stamp)):
                    command = candidate
                    if scale < 1.:
                        self.reason = 'braking_envelope'
                        self.acceleration[:] = 0.
                    break
            else:
                command[:] = 0.
                self.reason = 'emergency_stop'
        if self.reason in ('stale_input', 'user_stop', 'emergency_stop', 'planner_timeout'):
            self.acceleration[:] = 0.
        self.output = command
        msg = Twist()
        msg.linear.x, msg.angular.z = map(float, command)
        self.pub.publish(msg)
        status = {'mode': self.mode, 'reason': self.reason, 'v': msg.linear.x, 'w': msg.angular.z}
        if self.opening_turn is not None and self.pose is not None:
            opening = self._local_opening(self.opening_turn)
            status['opening'] = {'center': opening.center, 'width': opening.width,
                                 'ready': bool(self._opening_turn_ready(opening))}
        if self.door is not None and self.pose is not None:
            door = self._local_opening(self.door)
            status['door'] = {'center': door.center, 'heading': door.heading,
                              'width': door.width, 'entry_clearance': door_entry_clearance(door)}
        self.status.publish(String(data=json.dumps(status)))

    def destroy_node(self):
        with self.callback_lock:
            if self.stopped:
                return
            self.stopped = True
            self.reference_timer.shutdown()
            self.control_timer.shutdown()
            for subscriber in self.subscribers:
                subscriber.unregister()
            self.cancel()
            self.pub.publish(Twist())
            if self.door_plan_request is not None:
                self.door_plan_request[0].cancel()
            self.door_plan_executor.shutdown(wait=False)
        self.listener.unregister()
        for publisher in self.publishers:
            publisher.unregister()
        self.transforms.pub_tf.unregister()
        self.static.pub_tf.unregister()


def main(args=None):
    rospy.init_node('unified_control', argv=args)
    node = UnifiedControlNode()
    rospy.on_shutdown(node.destroy_node)
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
