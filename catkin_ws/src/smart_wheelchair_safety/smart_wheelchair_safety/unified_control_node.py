"""Shared-control references for the local follower, with an independent guard."""

from concurrent.futures import ProcessPoolExecutor
import copy
import json
import math
import multiprocessing
import threading
import time
import numpy as np
import rospy
from geometry_msgs.msg import Point32, PolygonStamped, PoseStamped, TransformStamped, Twist, TwistStamped
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, Float32, String
from tf2_ros import Buffer, TransformBroadcaster, StaticTransformBroadcaster, TransformListener, TransformException

from smart_wheelchair_safety.unified_geometry import (
    LIDAR_X_M, LIDAR_Y_M, LIDAR_VIEW_DEG,
    Opening, active_aperture_targeted, angle_difference, approach_path, arc_path, braking_clear,
    collision_aware_door_reference, door_alignment_reference, door_entry_clearance,
    extract_lines, extract_opening_lines, footprint_clearance,
    find_openings, intended_front_door, intended_side_opening, opening_matches, transform_points,
    wall_reference, front_wall_reference, reference_is_clear,
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
        self.sensor_stale_since = None
        self.active_reference_stamp = None
        self.last_reference_stamp = rospy.Time()
        self.odom_stamp = None
        self.scans = {}
        self.planning_scans = {}
        self.recovering = False
        self.door = None
        self.door_phase = None
        self.door_path = None
        self.door_rotation_index = 0
        self.door_path_feasible = False
        # CPU-bound lattice search must not contend for the control process's
        # GIL. Spawn avoids inheriting rospy threads, sockets and held locks.
        self.door_plan_executor = ProcessPoolExecutor(
            max_workers=1, mp_context=multiprocessing.get_context('spawn'))
        self.door_plan_request = None
        self.door_generation = 0
        self.door_away_since = 0.
        self.door_obstacles = np.empty((0, 2))
        self.door_obstacle_time = 0.
        self.opening_candidates = []
        self.confirmed_openings = []
        self.door_detections = []
        self.door_detection_candidates = []
        self.door_detection_time = 0.
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
        self.use_neupan = bool(rospy.get_param('~use_neupan', False))
        self.neupan_action_timeout = float(rospy.get_param('~neupan_action_timeout', .25))
        if not math.isfinite(self.neupan_action_timeout) or not 0 < self.neupan_action_timeout <= .25:
            raise ValueError('neupan_action_timeout must be in (0, .25]')
        self.neupan = np.zeros(2)
        self.neupan_time = 0.
        self.neupan_stamp = None
        self.override = False
        self.reason = 'starting'
        self.epoch = 0
        self.last_tick = rospy.Time.now().to_sec()
        self.tf = Buffer()
        self.listener = TransformListener(self.tf)
        self.transforms = TransformBroadcaster()
        self.static = StaticTransformBroadcaster()
        frames = []
        for side, y in LIDAR_Y_M.items():
            t = TransformStamped()
            t.header.frame_id = 'rear_axle'
            t.child_frame_id = f'unified_lidar_{side}'
            t.transform.translation.x = LIDAR_X_M
            t.transform.translation.y = y
            t.transform.rotation.w = 1.
            frames.append(t)
        self.static.sendTransform(frames)
        self.pub = rospy.Publisher('cmd_vel', Twist, queue_size=10)
        self.status = rospy.Publisher('shared_control/status', String, queue_size=10)
        self.reference = rospy.Publisher('shared_control/reference', Path, queue_size=10)
        self.door_detection_pub = rospy.Publisher('shared_control/door_detections', PolygonStamped, queue_size=1)
        self.limit_pub = rospy.Publisher('speed_limit', Float32, queue_size=10)
        self.scan_pubs = {side: rospy.Publisher(f'unified_scan_{side}', LaserScan, queue_size=10)
                          for side in ('left', 'right')}
        self.publishers = [self.pub, self.status, self.reference, self.limit_pub, self.door_detection_pub,
                           *self.scan_pubs.values()]
        self.subscribers = [
            rospy.Subscriber(topic, message, self._serialized(callback), queue_size=10)
            for topic, message, callback in (
                ('cmd_vel_raw', Twist, self.on_raw),
                ('cmd_vel_planned', TwistStamped, self.on_planned),
                ('neupan/cmd_vel', TwistStamped, self.on_neupan),
                ('assist_enabled', Bool, self.on_assist_enabled),
                ('odom', Odometry, self.on_odom))
        ]
        for side in ('left', 'right'):
            self.subscribers.append(rospy.Subscriber(
                f'scan_{side}', LaserScan,
                lambda msg, side=side: self.on_scan(msg, side), queue_size=1))
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
        if (not self.assist_enabled or not self.mode_neutral_seen
                or not self.accept_planned
                or msg.header.stamp != self.active_reference_stamp):
            return
        self.planned = np.array([msg.twist.linear.x, msg.twist.angular.z])
        self.plan_time = time.monotonic()

    def on_neupan(self, msg):
        if (not self.use_neupan or not self.assist_enabled or not self.mode_neutral_seen
                or not self.accept_planned or msg.header.stamp != self.active_reference_stamp
                or not 0 <= rospy.Time.now().to_sec()-msg.header.stamp.to_sec() < self.neupan_action_timeout):
            return
        value = np.array([msg.twist.linear.x, msg.twist.angular.z], dtype=float)
        if np.isfinite(value).all():
            self.neupan = value
            self.neupan_time = time.monotonic()
            self.neupan_stamp = msg.header.stamp

    def neupan_fresh(self, now=None):
        now = time.monotonic() if now is None else float(now)
        return (self.use_neupan and self.accept_planned and self.assist_enabled
                and self.mode_neutral_seen and self.neupan_stamp is not None
                and self.neupan_stamp == self.active_reference_stamp
                and np.isfinite(self.neupan).all()
                and 0 <= now-self.neupan_time < self.neupan_action_timeout
                and 0 <= rospy.Time.now().to_sec()-self.neupan_stamp.to_sec() < self.neupan_action_timeout)

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
        received = time.monotonic()
        scan_age = (rospy.Time.now().to_sec() - msg.header.stamp.to_sec())
        if not -.05 <= scan_age <= self.parameter('scan_timeout'):
            return
        if (not np.isfinite([msg.angle_min, msg.angle_increment, msg.range_min, msg.range_max]).all()
                or msg.angle_increment == 0. or not 0. <= msg.range_min < msg.range_max):
            return
        ranges = np.array(msg.ranges)
        healthy = np.isposinf(ranges) | (np.isfinite(ranges) & (ranges >= msg.range_min) & (ranges <= msg.range_max))
        outgoing = copy.deepcopy(msg)
        outgoing.header.frame_id = f'unified_lidar_{side}'
        angles = msg.angle_min + np.arange(len(ranges)) * msg.angle_increment
        lower, upper = map(math.radians, LIDAR_VIEW_DEG[side])
        view = (angles >= lower-1e-5) & (angles <= upper+1e-5)
        if np.count_nonzero(view) < 3 or np.mean(healthy[view]) < .9:
            return
        # TF may need a fresh odometry callback to supply this transform.
        # Never hold the state-machine lock while waiting for it.
        try:
            t = self.tf.lookup_transform('odom', outgoing.header.frame_id,
                                         msg.header.stamp, rospy.Duration(.15))
        except TransformException:
            return
        # User-selected 200 degree view is shared by perception, planning and
        # the final guard. Unobserved rear sectors are NOT certified clear.
        valid = view & np.isfinite(ranges) & (ranges >= msg.range_min) & (ranges <= min(msg.range_max, 4.5))
        local = np.column_stack((ranges[valid]*np.cos(angles[valid]), ranges[valid]*np.sin(angles[valid])))
        q = t.transform.rotation
        pose = (t.transform.translation.x, t.transform.translation.y, 2*math.atan2(q.z, q.w))
        world = transform_points(local, pose)
        # Remove known self returns at acquisition, never newly approached
        # obstacles using the current footprint after the chair has moved.
        axle_local = local + [LIDAR_X_M, LIDAR_Y_M[side]]
        outside = ((axle_local[:, 0] < -.25) | (axle_local[:, 0] > .97)
                   | (np.abs(axle_local[:, 1]) > .4))
        world = world[outside]
        accepted = np.zeros(len(ranges), dtype=bool)
        accepted[np.flatnonzero(valid)[outside]] = True
        outgoing.ranges = np.where(accepted, ranges, np.inf).tolist()
        with self.callback_lock:
            scan_age = rospy.Time.now().to_sec() - msg.header.stamp.to_sec()
            if (self.stopped or not -.05 <= scan_age <= self.parameter('scan_timeout')
                    or received < self.scans.get(side, (0., None))[0]):
                return
            self.scans[side] = (received, world)
            self.planning_scans[side] = world
            self.scan_pubs[side].publish(outgoing)
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
        if (self.door_path is not None and self.door_rotation_index == 0
                and np.linalg.norm(self.door_path[1, :2]-self.door_path[0, :2]) < 1e-6
                and np.linalg.norm(self.door_path[0, :2]-self.pose[:2]) > .04):
            self.door_path = None
            self.door_path_feasible = False
            return door_alignment_reference(door, 'align')
        if self.door_path is None:
            phase = 'align' if self.door_phase == 'door_align' else 'pass'
            if (phase == 'align' and abs(door.heading) < .15
                    and self.door_plan_request is None
                    and self.output[0] > .02 and self.measured[0] > .02
                    and door_entry_clearance(door) > .01):
                direct = door_alignment_reference(door, 'align')
                if reference_is_clear(direct, obstacles, margin=.045):
                    xy = transform_points(direct[:, :2], self.pose)
                    self.door_path = np.column_stack((xy, direct[:, 2]+self.pose[2]))
                    self.door_path_feasible = True
                    return direct
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
                                if (np.linalg.norm(np.asarray(self.pose[:2])-requested_pose[:2]) > .03
                                        or abs(angle_difference(self.pose[2], requested_pose[2])) > .03):
                                    local = None
                            except Exception as error:
                                rospy.logerr('door path search failed: %s', error)
                search_speed = np.array([.055, .11]) if self.recovering else np.array([.025, .025])
                if (local is None and self.door_plan_request is None
                        and np.all(np.abs(self.measured) < search_speed)
                        and np.all(np.abs(self.output) < search_speed)):
                    plan_pose = tuple(self.pose)
                    try:
                        future = self.door_plan_executor.submit(
                            collision_aware_door_reference, door,
                            np.array(obstacles, dtype=float, copy=True))
                    except RuntimeError as error:
                        # A failed worker must not terminate the reference
                        # timer or mark its unchecked fallback as executable.
                        rospy.logerr_throttle(1., 'door path submission failed: %s', error)
                    else:
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
        fraction = np.clip(-np.sum(local[:-1]*delta, axis=1)/np.maximum(segments*segments, 1e-12), 0., 1.)
        projections = local[:-1] + fraction[:, None]*delta
        distances = np.linalg.norm(projections, axis=1)
        distances[segments < 1e-6] = math.inf
        distances[:self.door_rotation_index] = math.inf
        pending = self._pending_door_rotation()
        if pending is not None:
            distances[pending[0]:] = math.inf
        if not np.isfinite(distances).any():
            return 0.
        nearest = int(np.argmin(distances))
        turns = np.arctan2(np.sin(np.diff(self.door_path[:, 2])),
                           np.cos(np.diff(self.door_path[:, 2])))
        reference_heading = self.door_path[nearest, 2] + fraction[nearest]*turns[nearest]
        heading_error = angle_difference(reference_heading, self.pose[2])
        curvature = (turns[nearest]/segments[nearest]
                     + 2.*heading_error + 4.*projections[nearest, 1])
        angular = linear*curvature
        return float(np.clip(angular, -.65, .65))

    def _pending_door_rotation(self):
        if self.door_path is None:
            return None
        for index in range(self.door_rotation_index, len(self.door_path)-1):
            first, second = self.door_path[index:index+2]
            if np.linalg.norm(second[:2]-first[:2]) > 1e-6:
                continue
            distance = float(np.linalg.norm(first[:2]-self.pose[:2]))
            error = angle_difference(second[2], self.pose[2])
            if distance < .07:
                self.door_rotation_index = index
            if (index == self.door_rotation_index and distance < (.04 if index == 0 else .12)
                    and abs(error) < .04 and abs(self.measured[1]) < .08):
                self.door_rotation_index = index+1
                continue
            return index, distance, error
        return None

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
        pending = self._pending_door_rotation()
        if pending is not None:
            index, distance, error = pending
            if index == 0 and distance > .04:
                return np.zeros(2)
            if index == self.door_rotation_index:
                if distance > .12:
                    self.door_path = None
                    self.door_path_feasible = False
                    self.door_rotation_index = 0
                    self.door_generation += 1
                    self.cancel()
                    self.mode = 'door_wait'
                    return np.zeros(2)
                angular = float(np.clip(1.5*error, -.4, .4)) if abs(self.measured[0]) < .025 else 0.
                return np.array([0., angular])
            # Arrive stopped at the checked pivot; do not carry translation
            # into a stationary rotation's swept-footprint guarantee.
            high = min(high, max(.025, math.sqrt(.04+max(0., distance-.05))-.2))
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
                else:
                    tracked = confirmed[match]
                    normal = np.array([math.cos(tracked.heading), math.sin(tracked.heading)])
                    tangent = np.array([-normal[1], normal[0]])
                    delta = np.asarray(opening.center)-tracked.center
                    # Accept newly supported inward jamb edges, never widen a
                    # remembered gap because its nearest edge is now occluded.
                    # Preserve the fixed anchor for same-width/chained fits.
                    if (tracked.width-opening.width >= .04
                            and abs(angle_difference(tracked.heading, opening.heading)) <= .03
                            and abs(delta @ normal) <= .03
                            and abs(delta @ tangent) <= (tracked.width-opening.width)/2+.025):
                        confirmed[match] = opening
                        if (self.door is not None and self.door_phase == 'door_align'
                                and opening_matches(self.door, tracked)):
                            # Neither a cached path nor an old worker result
                            # may continue targeting the superseded jamb edges.
                            self.door = opening
                            self.door_generation += 1
                            self.door_path = None
                            self.door_rotation_index = 0
                            self.door_path_feasible = False
                            self.mode = 'door_wait'
                            self.cancel()
        for candidate, hits, stamp in self.opening_candidates:
            if (now-stamp <= .6 and not any(opening_matches(candidate, opening)
                                            for opening in observed)):
                updated.append((candidate, hits, stamp))
        self.opening_candidates = updated
        self.confirmed_openings = confirmed
        # Navigation observations are cleared by neutral/reverse input. Keep
        # the two-scan display confirmation independent of those user actions.
        narrow = [opening for opening in observed if .92 <= opening.width <= 1.5][:32]
        self.door_detections = [opening for opening in narrow
                                if now-self.door_detection_time <= .6
                                and any(opening_matches(opening, previous)
                                        for previous in self.door_detection_candidates)]
        self.door_detection_candidates = narrow
        self.door_detection_time = now

    def _publish_door_detections(self):
        message = PolygonStamped()
        message.header.frame_id = 'odom'
        message.header.stamp = rospy.Time.now()
        now = time.monotonic()
        # Visualization has no dependency on joystick motion. Memory used for
        # traversing an occluded doorway is deliberately not rendered as a fresh detection.
        if (now-self.door_detection_time < self.parameter('scan_timeout')
                and len(self.scans) == 2
                and all(now-stamp < self.parameter('scan_timeout') for stamp, _ in self.scans.values())
                and now-self.odom_time < self.parameter('odom_timeout')):
            for opening in self.door_detections:
                tangent = np.array([-math.sin(opening.heading), math.cos(opening.heading)])
                for sign in (-1., 1.):
                    edge = np.asarray(opening.center)+sign*opening.width/2*tangent
                    message.polygon.points.append(Point32(x=edge[0], y=edge[1], z=.12))
        self.door_detection_pub.publish(message)

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
        self.door_rotation_index = 0
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
        guard_groups = []
        for side, (_, world) in self.scans.items():
            local = transform_points(world, self.pose, inverse=True)
            guard_groups.append(local)
            groups.append(transform_points(self.planning_scans.get(side, world), self.pose, inverse=True))
        # Door jamb memory is only a guard while an active door handoff is in
        # progress. Retaining it after the door is cleared can falsely put the
        # chair into clearance recovery at a later doorway.
        if (self.door is not None and self.door_obstacle_time
                and time.monotonic()-self.door_obstacle_time <= DOOR_OBSTACLE_MEMORY_TIMEOUT):
            guard_groups.append(transform_points(self.door_obstacles, self.pose, inverse=True))
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
        self.active_reference_stamp = None
        self.planned[:] = 0.
        self.neupan[:] = 0.
        self.neupan_time = 0.
        self.plan_time = 0.

    def _near_recovery(self, points):
        limit = self.parameter('hard_margin') + (.02 if self.recovering else 0.)
        if not len(points) or not np.all(np.abs(self.measured) <= [.055, .11]):
            return False
        clearance = np.min(footprint_clearance(points))
        if clearance <= limit:
            return True
        # A low-speed stop can already be uncertifiable just outside 4 cm.
        # This only selects the bounded recovery policy; each actual command
        # still has to pass its per-point separation and braking checks.
        if (clearance <= self.parameter('hard_margin')+.02
                and not braking_clear(points, np.zeros(2), self.measured,
                                      margin=self.parameter('hard_margin'),
                                      reaction=self.parameter('reaction_time'),
                                      deceleration=self.parameter('braking_deceleration'),
                                      state_age=max(0., rospy.Time.now().to_sec()-self.odom_stamp))):
            return True
        if not self.recovering or self.raw[0] <= .02 or self.override or self.mode == 'front_stop':
            return False
        # Do not hand a now-clear chair back to a stationary/infeasible plan.
        # Keep guarded creep while the normal planner obtains a moving path.
        moving_plan = (time.monotonic()-self.plan_time < .25 and self.mode != 'door_wait'
                       and np.isfinite(self.planned).all() and self.planned[0] > .002)
        door_ready = self.mode.startswith('door_') and self.door_path_feasible
        return not (moving_plan or door_ready)

    def update_reference(self):
        self._publish_door_detections()
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
        now = time.monotonic()
        if (self.raw[0] <= .02 or not np.isfinite(self.raw).all()
                or now-self.raw_time >= self.parameter('command_timeout')):
            self.sensor_stale_since = None
            self._clear_opening_turn()
            self._clear_door()
            self.cancel()
            self.mode = 'manual'
            return
        if not self.fresh():
            if self.sensor_stale_since is None:
                self.sensor_stale_since = now
            if now-self.sensor_stale_since >= 1.:
                self._clear_opening_turn()
                self._clear_door()
            self.cancel()
            self.mode = 'waiting'
            return
        self.sensor_stale_since = None
        if self.override:
            self.cancel()
            self.mode = 'override'
            return
        groups, obstacle_points = self.points()
        if (self._near_recovery(obstacle_points)
                and np.min(footprint_clearance(obstacle_points)) <= self.parameter('hard_margin')+.02):
            # Recovery ignores the planned path; do not keep starting costly
            # distant-door searches that can starve the control callbacks.
            self.recovering = True
            self._clear_door()
            self._clear_opening_turn()
            self.cancel()
            self.mode = 'clearance_recovery'
            return
        lines = [line for group in groups for line in extract_lines(group)]
        opening_path = self._opening_turn_path()
        local_door = self._local_tracked_door()
        if (local_door is not None and self.door_phase == 'door_align'
                and abs(self.raw[1]) < .12
                and abs(local_door.center[1]) > .65
                and door_entry_clearance(local_door) < -DOOR_ENTRY_NOISE_TOLERANCE):
            # A remembered door can remain in the two-scan cache after the
            # chair has moved past it or beside it. With straight joystick
            # intent, do not wait forever for an infeasible angled entry.
            self._clear_door()
            local_door = None
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
                self.mode = 'door_wait'
            else:
                if self.mode not in ('door_align', 'door_pass', 'door_clear'):
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
            turn = front_wall_reference(lines, obstacle_points, *self.raw,
                                        preferred_side=self.wall_preference)
            if turn is None:
                self.front_blocked = True
                self.wall_side = 0
                self.wall_preference = 0
                self.mode = 'front_stop'
                self.cancel()
                return
            local_path, self.mode, self.wall_side = turn
            self.front_blocked = False
            self.wall_preference = self.wall_side
            preview = int(np.argmin(np.abs(np.linalg.norm(local_path[:, :2], axis=1)-.8)))
            self.wall_preview_heading = angle_difference(local_path[preview, 2]+self.pose[2], 0.)
        else:
            local_path, self.mode, self.wall_side = wall_reference(
                lines, *self.raw,
                body_clearance=self.parameter('wall_clearance'),
                preferred_side=self.wall_preference)
            # A narrow doorway interrupts the followed wall. Keep the tangent
            # motion across its opening; treating the near jamb as a continuous
            # wall makes the safety planner brake at the threshold.
            local_openings = [self._local_opening(opening) for opening in self.confirmed_openings]
            if self.mode == 'wall' and self.wall_side:
                through_door = any(
                    .92 <= opening.width <= 1.5
                    and opening.center[0] > .35
                    and opening.center[0] < 2.5
                    and abs(math.sin(opening.heading)) > .85
                    and self.wall_side * opening.center[1] > .25
                    for opening in local_openings)
                if through_door and abs(self.raw[1]) < .12:
                    local_path = arc_path(max(self.raw[0], .1), self.raw[1])
                    self.mode = 'manual'
                    self.wall_side = 0
            if self.mode == 'wall':
                self.wall_preference = self.wall_side
                distance = np.linalg.norm(local_path[:, :2], axis=1)
                preview = int(np.argmin(np.abs(distance-.8)))
                self.wall_preview_heading = angle_difference(
                    local_path[preview, 2]+self.pose[2], 0.)
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
        if path.header.stamp <= self.last_reference_stamp:
            path.header.stamp = self.last_reference_stamp + rospy.Duration(nsecs=1)
        self.last_reference_stamp = path.header.stamp
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
        self.active_reference_stamp = path.header.stamp

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
        neupan_fresh = self.neupan_fresh(now)
        planner_stale = planner_stale and not neupan_fresh
        door_fallback = self.mode.startswith('door_') and self.door_path_feasible
        inputs_fresh = self.fresh()
        _, points = self.points() if inputs_fresh else (None, np.empty((0, 2)))
        if inputs_fresh:
            self.recovering = self._near_recovery(points)
        if self.recovering:
            door_fallback = False
        if neupan_fresh:
            door_fallback = False
        if not inputs_fresh:
            self.reason = 'stale_input'
        elif np.linalg.norm(self.raw) < .01:
            self.reason = 'user_stop'
        elif (self.raw[0] > .02 and not self.override and self.mode != 'front_stop'
              and self.mode != 'door_wait' and planner_stale and not door_fallback and not self.recovering):
            self.reason = 'planner_timeout'
        else:
            if self.recovering:
                desired = self.raw / max(1., abs(self.raw[0])/.05, abs(self.raw[1])/.1)
                if self.door_plan_request is not None and not self.door_plan_request[0].done():
                    desired *= .2
                self.reason = 'clearance_recovery'
            elif neupan_fresh and self.raw[0] > .02 and not self.override and self.mode not in ('door_wait', 'front_stop'):
                desired = self.neupan.copy()
            elif self.mode == 'door_wait':
                desired = np.zeros(2)
            elif self.raw[0] <= .02 or self.override or self.mode == 'front_stop':
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
            elif neupan_fresh or (now-self.plan_time < .25 and np.isfinite(self.planned).all()):
                desired = self.neupan.copy() if neupan_fresh else self.planned.copy()
                if (self.mode == 'wall' and self.raw[0] > .02
                        and desired[0] < .02):
                    # The lattice follower can briefly select its zero sample
                    # when a wall segment is refreshed. Preserve a guarded
                    # human-like creep unless a real frontal obstacle is near.
                    front = points[(points[:, 0] > 0.) & (np.abs(points[:, 1]) <= .44)]
                    nearest = float(front[:, 0].min()) if len(front) else math.inf
                    if nearest > 1.15:
                        desired[0] = min(self.raw[0], .08)
                if self.mode.startswith('door_'):
                    if (desired[0] <= .02 and self.door is not None
                            and self.raw[0] > .02
                            and door_entry_clearance(self._local_opening(self.door))
                            > -DOOR_ENTRY_NOISE_TOLERANCE):
                        # Preserve forward intent through a temporarily sparse
                        # planner update; the independent guard still decides
                        # whether this creep is geometrically safe.
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
            if door_fallback and (not self.door_path_feasible or self.door_path is None):
                command[:] = 0.
                self.acceleration[:] = 0.
                self.reason = 'invalid_door_path'
            elif door_fallback:
                pending = self._pending_door_rotation()
                if pending is None or pending[0] != self.door_rotation_index:
                    command[1] = self._door_angular(command[0]) if command[0] > .002 else 0.
                    self.acceleration[1] = 0.
            candidates = [(command*scale, None if scale == 1. else 'braking_envelope')
                          for scale in (1., .8, .6, .4, .2)]
            if self.recovering:
                # A tiny jerk-limited step can fail to resolve an existing
                # stop-certificate deficit, even when a larger retreat passes.
                # Try intent-preserving low-speed starts before falling back
                # to zero. Only comfort jerk is bypassed: both commanded and
                # measured velocity changes retain the acceleration bound,
                # and every candidate still goes through the same full guard.
                for scale in (.2, .4, .6, .8, 1.):
                    start = desired*scale
                    if (np.linalg.norm(start) > np.linalg.norm(command)
                            and np.all(np.abs(start-self.output) <= np.array([.5, .8])*dt)
                            and np.all(np.abs(start-self.measured) <= np.array([.5, .8])*dt)):
                        candidates.append((start, 'clearance_recovery'))
            candidates.append((np.zeros(2), 'braking_envelope'))
            for candidate, adjusted_reason in candidates:
                if braking_clear(points, candidate, self.measured,
                                 margin=self.parameter('hard_margin'),
                                 reaction=self.parameter('reaction_time'),
                                 deceleration=self.parameter('braking_deceleration'),
                                 state_age=max(0., stamp-self.odom_stamp), recovery=self.recovering):
                    command = candidate
                    if adjusted_reason is not None:
                        self.reason = adjusted_reason
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
        # Reap the spawned worker after publishing stop and releasing the ROS
        # callback lock (Python 3.8 cannot safely close a live pool asynchronously).
        self.door_plan_executor.shutdown(wait=True)
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
