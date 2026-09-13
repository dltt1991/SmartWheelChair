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
    LIDAR_X_M, LIDAR_Y_M, LIDAR_VIEW_DEG, output_arc_targets_aperture,
    Opening, active_aperture_targeted, angle_difference, approach_path, arc_path, braking_clear,
    collision_aware_door_reference, door_alignment_reference, door_entry_clearance,
    extract_lines, extract_opening_lines, footprint_clearance,
    find_openings, refine_opening, intended_front_door, intended_side_opening, opening_matches, transform_points,
    wall_reference, front_wall_reference, reference_is_clear,
)


DOOR_INTENT_CORRIDOR_TOLERANCE = .47
DOOR_ENTRY_NOISE_TOLERANCE = .01
DOOR_TRACKING_RESERVE = .01
DOOR_PASS_SPEED = .15
DOOR_CREEP_SPEED = .05
DOOR_OBSTACLE_MEMORY_TIMEOUT = 1.5
OPENING_TURN_TIMEOUT = 12.
OPENING_TURN_REAR_CLEARANCE = .50
REFERENCE_PERIOD = .2
JERK_LIMITS = np.array([2.5, 4.0])
# The approach limiter settles within J*dt^2/8; control caps dt at .1 s.
DOOR_CAPTURE_SPEED_TOLERANCE = JERK_LIMITS[0]*.1**2/8


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
        self.reverse_slew_active = False
        self.initial_plan_wait = None
        self.pose = None
        self.raw_time = self.odom_time = self.plan_time = 0.
        self.use_sim_time = bool(rospy.get_param('/use_sim_time', False))
        self.odom_ros_time = 0.
        self.plan_ros_time = self.neupan_ros_time = 0.
        self.clock_last_ros = 0.
        self.clock_last_advance_wall = time.monotonic()
        self.clock_stalled = False
        self.raw_resume_wall = 0.
        self.clock_watchdog_stop = threading.Event()
        self.clock_watchdog_thread = None
        self.latest_odom = self.consumed_odom = None
        self.latest_raw = self.latest_raw_cancel = self.consumed_raw = None
        self.latest_planned = self.consumed_planned = None
        self.latest_neupan = self.consumed_neupan = None
        self.sensor_stale_since = None
        self.active_reference_stamp = None
        self.last_reference_stamp = rospy.Time()
        self.odom_stamp = None
        self.scans = {}
        self.planning_scans = {}
        self.recovering = False
        self.door = None
        self.pending_door_capture = None
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
        self.override_handoff_time = 0.
        self.reason = 'starting'
        self.epoch = 0
        self.last_clear_tick = None
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
        self.reference_speed_pub = rospy.Publisher('shared_control/reference_speed', TwistStamped, queue_size=10)
        self.scan_pubs = {side: rospy.Publisher(f'unified_scan_{side}', LaserScan, queue_size=10)
                          for side in ('left', 'right')}
        self.publishers = [self.pub, self.status, self.reference, self.limit_pub, self.reference_speed_pub, self.door_detection_pub,
                           *self.scan_pubs.values()]
        receivers = {'odom': self._receive_odom, 'cmd_vel_raw': self._receive_raw,
                     'cmd_vel_planned': self._receive_planned, 'neupan/cmd_vel': self._receive_neupan}
        self.subscribers = [
            rospy.Subscriber(topic, message, receivers.get(topic) or self._serialized(callback),
                             queue_size=1 if topic == 'odom' else 10)
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

        if self.use_sim_time:
            # ROS timers themselves stop with /clock; this final-output watchdog
            # must run on an independent wall-clock wait.
            self.clock_watchdog_thread = threading.Thread(
                target=self._clock_watchdog_loop, name='sim_clock_watchdog', daemon=True)
            self.clock_watchdog_thread.start()

    def _clock_watchdog_loop(self):
        while not self.clock_watchdog_stop.wait(.05):
            self._clock_watchdog_tick()

    def _clock_watchdog_tick(self, wall=None, ros=None):
        if not self.use_sim_time:
            return
        with self.callback_lock:
            # Sample after acquiring the state lock: queued snapshots can
            # falsely report a freeze/rewind after control has already advanced.
            wall = time.monotonic() if wall is None else wall
            ros = rospy.Time.now().to_sec() if ros is None else ros
            if self.stopped or (self.clock_last_ros <= 0. and ros <= 0. and not self.clock_stalled):
                return
            if ros > self.clock_last_ros:
                self.clock_last_ros = ros
                self.clock_last_advance_wall = wall
                if self.clock_stalled:
                    # Input received while the world was frozen cannot restart
                    # motion by itself; require a fresh post-resume heartbeat.
                    self.raw_time = 0.
                    self.raw_resume_wall = wall
                self.clock_stalled = False
                return
            rewind = ros < self.clock_last_ros
            self.clock_last_ros = ros
            if not self.clock_stalled and not rewind and wall-self.clock_last_advance_wall < .25:
                return
            self._stop_for_clock(wall, ros)

    def _stop_for_clock(self, wall, ros, fault='clock_stalled', elapsed=None):
        """Caller holds callback_lock; revoke once, publish zero on every fault."""
        if not self.clock_stalled:
            self.clock_stalled = True
            self.cancel()
            self.raw_time = 0.
        self.output[:] = 0.
        self.acceleration[:] = 0.
        self.reason = 'stale_input'
        self.pub.publish(Twist())
        self.status.publish(String(data=json.dumps({
            'control_stamp_s': ros, 'control_dt_s': 0., 'mode': self.mode,
            'reason': self.reason, 'v': 0., 'w': 0., 'planner_source': 'stop',
            'clock_stalled': True, 'clock_wall_age_s': wall-self.clock_last_advance_wall,
            'clock_fault': fault, 'elapsed_s': elapsed})))

    def _control_interval(self, stamp, preview=False):
        elapsed = stamp-self.last_tick
        if not self.use_sim_time:
            return min(.1, max(.001, elapsed))
        if preview and elapsed == 0.:
            # Predict the next timer step without advancing state or authority.
            return .05
        if math.isfinite(elapsed) and .001-1e-9 <= elapsed < .25:
            return elapsed
        return None

    def parameter(self, name):
        return self.parameters[name]

    def _serialized(self, callback):
        def invoke(*args):
            with self.callback_lock:
                if not self.stopped:
                    self._consume_odom()
                    self._consume_raw()
                    self._consume_planner_actions()
                    return callback(*args)
        return invoke

    def _receive_raw(self, msg):
        if self.stopped:
            return
        snapshot = (msg, time.monotonic())
        if (msg.linear.x <= .02
                or not math.isfinite(msg.linear.x) or not math.isfinite(msg.angular.z)):
            # Publish cancellation first: a concurrent consumer must never
            # replay an older forward input after observing this cancellation.
            self.latest_raw_cancel = snapshot
        self.latest_raw = snapshot

    def _consume_raw(self):
        if self.stopped:
            return
        latest, cancellation = self.latest_raw, self.latest_raw_cancel
        for snapshot in (cancellation, latest):
            if snapshot is None or snapshot is self.consumed_raw:
                continue
            if self.consumed_raw is not None and snapshot[1] <= self.consumed_raw[1]:
                continue
            self.consumed_raw = snapshot
            self.on_raw(*snapshot)

    def on_raw(self, msg, received=None):
        received = time.monotonic() if received is None else received
        # Preserve cancellation even when queued during a clock pause. A forward
        # heartbeat, however, must have actually arrived after clock recovery.
        if (self.use_sim_time and received <= self.raw_resume_wall
                and msg.linear.x > .02 and math.isfinite(msg.linear.x)
                and math.isfinite(msg.angular.z)):
            return
        previous_raw, previous_received = self.raw.copy(), self.raw_time
        was_override = self.override
        self.raw = np.array([msg.linear.x, msg.angular.z])
        # Old cancellation still clears intent, but its nonzero reverse/spin
        # component cannot become a fresh motion command after clock recovery.
        self.raw_time = (0. if self.use_sim_time and received <= self.raw_resume_wall
                         else received)
        if not np.isfinite(self.raw).all():
            self._clear_opening_turn()
            self._clear_door()
            self.cancel()
            return
        if self.pending_door_capture is not None:
            tolerance = min(.60, DOOR_INTENT_CORRIDOR_TOLERANCE + .16*abs(self.raw[1]))
            if (self.raw[0] <= .02 or self.pose is None
                    or intended_front_door([self._local_opening(self.pending_door_capture)],
                                           *self.raw, corridor_tolerance=tolerance) is None):
                self.pending_door_capture = None
        if np.linalg.norm(self.raw) < .01:
            self.mode_neutral_seen = True
        if not self.assist_enabled or not self.mode_neutral_seen:
            return
        if (np.isfinite(previous_raw).all() and previous_raw[0] <= .02 < self.raw[0]
                and previous_received > self.raw_resume_wall
                and 0 <= received-previous_received < self.parameter('command_timeout')
                and self.active_reference_stamp is None and not self.accept_planned
                and not self.clock_stalled and self.fresh()):
            self.initial_plan_wait = (self.epoch, received)
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
        door_cancelled = False
        if self.raw[0] <= .02:
            self._clear_door()
            self.cancel()
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
                self.cancel()
                # The next reference timer may not run for another .2 s.
                # Release door-mode ownership now so live input uses ordinary
                # assisted slew; cancelled-path actions cannot reclaim it.
                self.mode = 'manual'
                door_cancelled = True
        self.override = door_cancelled
        if (not self.override and self.door is None
                and (self.wall_side*self.raw[1] < -.15
                     or (self.mode == 'override' and abs(self.raw[1]) > .25))):
            # Straight heartbeats and an active door cannot enter override;
            # only compute door intent when it can change this decision.
            intended, _ = self._intended_door()
            if intended is None:
                side_opening = None
                if self.pose is not None and self.raw[0] > .02 and abs(self.raw[1]) >= .12:
                    local_openings = [self._local_opening(opening)
                                      for opening in self.confirmed_openings]
                    side_opening = intended_side_opening(
                        local_openings, math.copysign(1., self.raw[1]), *self.raw)
                self.override = side_opening is None
        if self.override or self.raw[0] <= .02 or abs(self.raw[1]) <= .25:
            self.override_handoff_time = 0.
        elif was_override:
            # Release reference ownership now, but retain live joystick control
            # until a new accepted action can take over without a timeout gap.
            self.override_handoff_time = time.monotonic()

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
        self.override_handoff_time = 0.
        self.mode = 'manual' if enabled else 'manual_direct'
        self.pub.publish(Twist())

    def _receive_planned(self, msg):
        if not self.stopped:
            self.latest_planned = (msg, time.monotonic(), rospy.Time.now().to_sec())

    def _receive_neupan(self, msg):
        if not self.stopped:
            self.latest_neupan = (msg, time.monotonic(), rospy.Time.now().to_sec())

    def _consume_planner_actions(self):
        if self.stopped:
            return
        for name in ('planned', 'neupan'):
            snapshot = getattr(self, 'latest_'+name)
            if snapshot is None or snapshot is getattr(self, 'consumed_'+name):
                continue
            setattr(self, 'consumed_'+name, snapshot)
            msg, received, received_ros = snapshot
            timeout = self.neupan_action_timeout if name == 'neupan' else .25
            ros_now = rospy.Time.now().to_sec()
            age = ros_now-received_ros if self.use_sim_time else time.monotonic()-received
            if (not 0 <= age < timeout
                    or not 0 <= ros_now-msg.header.stamp.to_sec() < timeout):
                continue
            getattr(self, 'on_'+name)(msg, received, received_ros)

    def on_planned(self, msg, received=None, received_ros=None):
        if (not self.assist_enabled or not self.mode_neutral_seen
                or not self.accept_planned
                or msg.header.stamp != self.active_reference_stamp):
            return
        self.planned = np.array([msg.twist.linear.x, msg.twist.angular.z])
        self.plan_time = time.monotonic() if received is None else received
        self.plan_ros_time = rospy.Time.now().to_sec() if received_ros is None else received_ros
        if np.isfinite(self.planned).all():
            self.initial_plan_wait = None

    def on_neupan(self, msg, received=None, received_ros=None):
        if (not self.use_neupan or not self.assist_enabled or not self.mode_neutral_seen
                or not self.accept_planned or msg.header.stamp != self.active_reference_stamp
                or not 0 <= rospy.Time.now().to_sec()-msg.header.stamp.to_sec() < self.neupan_action_timeout):
            return
        value = np.array([msg.twist.linear.x, msg.twist.angular.z], dtype=float)
        if np.isfinite(value).all():
            self.neupan = value
            self.neupan_time = time.monotonic() if received is None else received
            self.neupan_ros_time = rospy.Time.now().to_sec() if received_ros is None else received_ros
            self.neupan_stamp = msg.header.stamp
            self.initial_plan_wait = None

    def _plan_age(self, now=None):
        # A replacement reference never renews this cached action's receipt.
        if self.use_sim_time:
            return rospy.Time.now().to_sec()-self.plan_ros_time
        return (time.monotonic() if now is None else now)-self.plan_time

    def neupan_fresh(self, now=None):
        now = time.monotonic() if now is None else float(now)
        return (self.use_neupan and self.accept_planned and self.assist_enabled
                and self.mode_neutral_seen and self.neupan_stamp is not None
                and self.neupan_stamp == self.active_reference_stamp
                and np.isfinite(self.neupan).all()
                and 0 <= (rospy.Time.now().to_sec()-self.neupan_ros_time if self.use_sim_time
                          else now-self.neupan_time) < self.neupan_action_timeout
                and 0 <= rospy.Time.now().to_sec()-self.neupan_stamp.to_sec() < self.neupan_action_timeout)

    def _receive_odom(self, msg):
        # Never wait for the state-machine lock in the transport callback.
        if not self.stopped:
            self.latest_odom = (msg, time.monotonic(), rospy.Time.now().to_sec())

    def _consume_odom(self):
        snapshot = self.latest_odom
        if self.stopped or snapshot is None or snapshot is self.consumed_odom:
            return
        # Do not clear latest: the receiver may replace it during consumption.
        self.consumed_odom = snapshot
        msg, received, received_ros = snapshot
        now, ros_now = time.monotonic(), rospy.Time.now().to_sec()
        receipt_age = ros_now-received_ros if self.use_sim_time else now-received
        if (not 0 <= receipt_age < self.parameter('odom_timeout')
                or (self.use_sim_time and now-received >= min(.25, self.parameter('command_timeout')))):
            return
        self.on_odom(msg, received, received_ros)

    def on_odom(self, msg, received=None, received_ros=None):
        q = msg.pose.pose.orientation
        p = msg.pose.pose.position
        ros_now = rospy.Time.now().to_sec()
        received_ros = ros_now if received_ros is None else received_ros
        received = time.monotonic() if received is None else received
        age = ros_now - msg.header.stamp.to_sec()
        if not -.05 <= received_ros-msg.header.stamp.to_sec() <= self.parameter('odom_timeout'):
            return
        values = [p.x, p.y, q.x, q.y, q.z, q.w, msg.twist.twist.linear.x, msg.twist.twist.angular.z]
        if not -.05 <= age <= self.parameter('odom_timeout') or not np.isfinite(values).all():
            return
        if abs(q.x*q.x+q.y*q.y+q.z*q.z+q.w*q.w-1.) > .01:
            return
        yaw = math.atan2(2*(q.w*q.z + q.x*q.y), 1-2*(q.y*q.y + q.z*q.z))
        self.pose = (p.x, p.y, yaw)
        self.measured = np.array([msg.twist.twist.linear.x, msg.twist.twist.angular.z])
        self.odom_time = received
        self.odom_ros_time = received_ros
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
            self._consume_odom()
            self._consume_raw()
            self._consume_planner_actions()
            self.scans[side] = (received, world)
            self.planning_scans[side] = world
            self.scan_pubs[side].publish(outgoing)
            now = time.monotonic()
            if self.pose is None or len(self.scans) != 2 or now-self.opening_observation_time < .08:
                return
            groups, _ = self.points()
            observation_pose = self.pose
            epoch, generation = self.epoch, self.door_generation
            scan_times = [stamp for stamp, _ in self.scans.values()]
            # Reserve this observation before releasing the lock. A newer
            # reservation supersedes an older extraction that finishes late.
            self.opening_observation_time = now
        lines = [line for group in groups for line in extract_opening_lines(group)]
        observation_points = np.vstack(groups)
        openings = [refine_opening(opening, observation_points) for opening in find_openings(lines)]
        observed = [Opening(tuple(transform_points([opening.center], observation_pose)[0]),
                            opening.heading+observation_pose[2], opening.width, ())
                    for opening in openings]
        with self.callback_lock:
            scan_age = rospy.Time.now().to_sec()-msg.header.stamp.to_sec()
            if (self.stopped or self.pose is None or self.epoch != epoch
                    or self.door_generation != generation or self.opening_observation_time != now
                    or not -.05 <= scan_age <= self.parameter('scan_timeout')
                    or any(time.monotonic()-stamp >= self.parameter('scan_timeout') for stamp in scan_times)):
                return
            # Odometry can advance during extraction. Preserve acquisition
            # geometry in world coordinates before using the current pose.
            self._observe_openings([self._local_opening(opening) for opening in observed])

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
                if reference_is_clear(direct, obstacles, margin=.06):
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
        # Translation samples are dense; only coincident positions can be a
        # pivot. Find those in one array operation instead of scanning every
        # path point repeatedly in the 20 Hz control callback.
        segments = np.diff(self.door_path[self.door_rotation_index:, :2], axis=0)
        pivots = np.flatnonzero(np.hypot(segments[:, 0], segments[:, 1]) <= 1e-6)
        for index in pivots + self.door_rotation_index:
            first, second = self.door_path[index:index+2]
            distance = float(np.linalg.norm(first[:2]-self.pose[:2]))
            error = angle_difference(second[2], self.pose[2])
            if distance < .025:
                self.door_rotation_index = index
            if (index == self.door_rotation_index and distance < .04
                    and abs(error) < .04 and abs(self.measured[1]) < .01
                    and abs(self.output[1]) < 1e-6 and abs(self.acceleration[1]) < 1e-6):
                self.door_rotation_index = index+1
                continue
            return index, distance, error
        return None

    def _door_rear_clear_progress(self, local):
        normal = np.array([math.cos(self.door.heading), math.sin(self.door.heading)])
        tangent = np.array([-normal[1], normal[0]])
        points = np.asarray(self.door_obstacles, dtype=float).reshape(-1, 2)
        if not np.isfinite(points).all():
            return math.inf
        relative = points - self.door.center
        across, along = relative @ normal, relative @ tangent
        # Use observed doorway depth, excluding distant or unrelated wall points.
        jamb = ((np.abs(across) <= .3)
                & (np.abs(np.abs(along) - self.door.width/2) <= .25))
        depth = max(0., float(np.max(across[jamb]))) if np.any(jamb) else 0.
        cosine, sine = math.cos(local.heading), math.sin(local.heading)
        rear = max(.25*cosine, -.97*cosine) + .4*abs(sine)
        return depth + rear + .04

    def _door_angular(self, linear):
        path_angular = self._door_preview_angular(linear)
        if self.door_phase not in ('door_pass', 'door_clear') or self.door is None:
            return path_angular
        local = self._local_opening(self.door)
        normal = np.array([math.cos(local.heading), math.sin(local.heading)])
        progress = -np.asarray(local.center) @ normal
        joystick_weight = float(np.clip(progress/self._door_rear_clear_progress(local), 0., 1.))
        return (1.-joystick_weight)*path_angular + joystick_weight*self.raw[1]

    def _guard_state_age(self, stamp):
        age = max(0., stamp-self.odom_stamp)
        # In a narrow aperture, accelerating against a freshly received pose
        # can consume the clearance needed by the next, older pose. Reserve
        # the full accepted odometry delay before entering that speed state.
        return max(age, self.parameter('odom_timeout')) if self.mode.startswith('door_') else age

    def _assisted_slew(self, desired, dt, acceleration=None, linear_lower_bound=0.):
        """Recover acceleration across changing targets and fixed speed bounds."""
        previous_a = self.acceleration if acceleration is None else acceleration
        limits = np.array([.5, .8])
        lower = np.array([linear_lower_bound, -.65])
        upper = np.array([self.parameter('max_speed'), .65])
        delta = desired-self.output
        approach = np.maximum(0., np.sqrt(2*JERK_LIMITS*np.abs(delta))
                              -.5*JERK_LIMITS*dt)
        # A timer may run as soon as .001 s later. Finish only when that
        # next stationary step can also recover acceleration within jerk.
        finish_limit = JERK_LIMITS*min(dt, .001)
        approach = np.maximum(approach, np.minimum(np.abs(delta)/dt, finish_limit))
        target_a = np.sign(delta)*np.minimum(limits, approach)
        acceleration = previous_a+np.clip(
            target_a-previous_a, -JERK_LIMITS*dt, JERK_LIMITS*dt)
        # A positive acceleration needs a*dt now and at most a**2/(2*J)
        # extra speed while subsequent steps recover it with maximum jerk.
        # Reserve that space at physical bounds, independent of a changing target.
        bound_a = lambda distance: (np.sqrt((JERK_LIMITS*dt)**2
                                            +2*JERK_LIMITS*np.maximum(0., distance))
                                    -JERK_LIMITS*dt)
        acceleration = np.clip(acceleration, -bound_a(self.output-lower),
                               bound_a(upper-self.output))
        finish_a = delta/dt
        finish = ((np.abs(finish_a) <= np.minimum(limits, finish_limit))
                  & (np.abs(finish_a-previous_a) <= JERK_LIMITS*dt))
        acceleration = np.where(finish, finish_a, acceleration)
        command = np.clip(self.output+acceleration*dt, lower, upper)
        # Safety bounds win for externally imposed, already infeasible states.
        # Never leave a stored acceleration different from the emitted step.
        return command, (command-self.output)/dt

    def _door_wait_slew(self, dt, output=None, acceleration=None):
        """Preview comfortable stopping without mutating the current command."""
        output = self.output if output is None else output
        previous_a = self.acceleration if acceleration is None else acceleration
        limits = np.array([.5, .8])
        finish_limit = JERK_LIMITS*min(dt, .001)
        approach = np.maximum(0., np.sqrt(2*JERK_LIMITS*np.abs(output))-.5*JERK_LIMITS*dt)
        approach = np.maximum(approach, np.minimum(np.abs(output)/dt, finish_limit))
        target_a = -np.sign(output)*np.minimum(limits, approach)
        acceleration = previous_a + np.clip(target_a-previous_a,
                                                   -JERK_LIMITS*dt, JERK_LIMITS*dt)
        # Before reaching zero reserve room to recover acceleration, including
        # when the next control interval is shorter than this one.
        toward_zero = (np.sqrt((JERK_LIMITS*dt)**2+2*JERK_LIMITS*np.abs(output))
                       -JERK_LIMITS*dt)
        bounded = np.sign(output)*np.maximum(
            np.sign(output)*acceleration, -toward_zero)
        # An imposed state may lack the future reserve yet still stop smoothly
        # at this dt. Recover as fast as jerk permits; only actual zero crossing
        # below takes precedence over comfort and makes admission fail.
        acceleration = np.clip(bounded, previous_a-JERK_LIMITS*dt,
                               previous_a+JERK_LIMITS*dt)
        stop_a = -output/dt
        finish = ((np.abs(stop_a) <= np.minimum(limits, finish_limit))
                  & (np.abs(stop_a-previous_a) <= JERK_LIMITS*dt))
        acceleration = np.where(finish, stop_a, acceleration)
        command = output + acceleration*dt
        # Initial acceleration may carry output away from zero while recovering.
        # Never cross zero or restart from it. Infeasible imposed states remain
        # visible as a jerk failure to the shared admission preview.
        command = np.where((output == 0.) | (command*output < 0.), 0., command)
        command = np.where(finish & (np.abs(command) < 1e-12), 0., command)
        return command, (command-output)/dt

    def _door_capture_is_clear(self, points):
        """Check a bounded full wait stop before changing door ownership.

        The first state uses actual odometry. Later states predict constant
        command integration and ideal end-of-step tracking on the frozen cloud.
        This admission screen does not replace the actual per-tick hard guard.
        """
        stamp = rospy.Time.now().to_sec()
        dt = self._control_interval(stamp, preview=True)
        if dt is None:
            return False
        kwargs = dict(margin=self.parameter('hard_margin'),
                      reaction=self.parameter('reaction_time'),
                      deceleration=self.parameter('braking_deceleration'),
                      state_age=max(0., stamp-self.odom_stamp, self.parameter('odom_timeout')))
        output, acceleration = self.output.copy(), self.acceleration.copy()
        measured = self.measured.copy()
        if not np.isfinite([output, acceleration, measured]).all():
            return False
        if not braking_clear(points, output, measured, **kwargs):
            return False
        for _ in range(80):
            command, next_a = self._door_wait_slew(dt, output, acceleration)
            if (np.any(np.abs(next_a) > np.array([.5, .8])+1e-9)
                    or np.any(np.abs(next_a-acceleration) > JERK_LIMITS*dt+1e-9)
                    or not braking_clear(points, command, measured, **kwargs)):
                return False
            if np.max(np.abs(command)) < 1e-10 and np.max(np.abs(next_a)) < 1e-10:
                return True
            v, w = command
            theta = w*dt
            pose = ([v*dt, 0., theta] if abs(w) < 1e-10 else
                    [v/w*math.sin(theta), v/w*(1.-math.cos(theta)), theta])
            points = transform_points(points, pose, inverse=True)
            output, acceleration, measured = command, next_a, command
            dt = .05
        # Unsettled predictions cannot authorize a wait that has no checked end.
        return False

    def _door_near_entry(self):
        if self.door is None:
            return False
        local = self._local_opening(self.door)
        normal = np.array([math.cos(local.heading), math.sin(local.heading)])
        # Start passage speed before the front reaches the jambs. At 1.6 m
        # the .97 m front overhang still leaves .63 m to settle acceleration.
        return np.asarray(local.center) @ normal < 1.6

    def _safe_door_command(self, points, stamp):
        cruise = .35 if self.mode == 'door_align' and not self._door_near_entry() else DOOR_PASS_SPEED
        high = min(self.raw[0], cruise)
        pending = self._pending_door_rotation()
        if pending is not None:
            index, distance, error = pending
            if index == 0 and distance > .04:
                return np.zeros(2)
            if index == self.door_rotation_index:
                if distance > .04:
                    self.door_path = None
                    self.door_path_feasible = False
                    self.door_rotation_index = 0
                    self.door_generation += 1
                    self.cancel()
                    self.mode = 'door_wait'
                    return np.zeros(2)
                angular = (float(np.clip(1.5*error, -.4, .4))
                           if abs(error) >= .04 and abs(self.measured[0]) < .025 else 0.)
                return np.array([0., angular])
            # Arrive stopped at the checked pivot; do not carry translation
            # into a stationary rotation's swept-footprint guarantee.
            high = min(high, max(.025, math.sqrt(.04+max(0., distance-.05))-.2))
        best = np.zeros(2)
        state_age = self._guard_state_age(stamp)
        # Keep tracking below the hard guard boundary so scan/pose updates do
        # not turn small clearance changes into abrupt safety intervention.
        planning_margin = self.parameter('hard_margin') + DOOR_TRACKING_RESERVE
        for attempt in range(8):
            speed = high if attempt == 0 else (best[0]+high)/2
            candidate = np.array([speed, self._door_angular(speed)])
            clear = braking_clear(points, candidate, self.measured,
                                  margin=planning_margin,
                                  reaction=self.parameter('reaction_time'),
                                  deceleration=self.parameter('braking_deceleration'),
                                  state_age=state_age)
            if clear:
                # The next cycle must still be able to stop at the requested
                # speed. A safe transition alone can consume that reserve as
                # the newly measured speed renews the reaction-delay travel.
                next_pose = arc_path(*candidate, duration=.05, steps=2)[-1]
                next_points = transform_points(points, next_pose, inverse=True)
                clear = braking_clear(next_points, np.zeros(2), candidate,
                                      margin=planning_margin,
                                      reaction=self.parameter('reaction_time'),
                                      deceleration=self.parameter('braking_deceleration'),
                                      state_age=state_age)
            if clear:
                if attempt == 0:
                    return candidate
                best = candidate
            else:
                high = speed
        return best

    def _safe_assisted_command(self, desired, points):
        """Reserve one control cycle and the full odometry age before acceleration."""
        if not len(points) or not np.any(desired):
            return desired
        kwargs = dict(margin=self.parameter('hard_margin')+.01,
                      reaction=self.parameter('reaction_time'),
                      deceleration=self.parameter('braking_deceleration'),
                      state_age=self.parameter('odom_timeout'))
        def clear(candidate):
            if not braking_clear(points, candidate, self.measured, **kwargs):
                return False
            next_pose = arc_path(*candidate, duration=.05, steps=2)[-1]
            next_points = transform_points(points, next_pose, inverse=True)
            return braking_clear(next_points, np.zeros(2), candidate, **kwargs)
        if clear(desired):
            return desired
        low, high = 0., 1.
        for _ in range(7):
            scale = (low+high)/2
            if clear(desired*scale):
                low = scale
            else:
                high = scale
        return desired*low

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
        self.pending_door_capture = None
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
            if progress > self._door_rear_clear_progress(local):
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
        ros_now = rospy.Time.now().to_sec()
        # Simulated motion ages on ROS time, but a frozen clock must never
        # hide an independently lost odometry stream beyond the wall deadman.
        receipt_age = ros_now-self.odom_ros_time if self.use_sim_time else now-self.odom_time
        return (not self.clock_stalled
                and np.isfinite(self.raw).all() and np.isfinite(self.measured).all()
                and self.pose is not None and np.isfinite(self.pose).all()
                and now-self.raw_time < self.parameter('command_timeout')
                and self.odom_stamp is not None
                and -.05 <= ros_now-self.odom_stamp < self.parameter('odom_timeout')
                and 0 <= receipt_age < self.parameter('odom_timeout')
                and (not self.use_sim_time or now-self.odom_time < min(.25, self.parameter('command_timeout')))
                and len(self.scans) == 2
                and all(now-stamp < self.parameter('scan_timeout') for stamp, _ in self.scans.values()))

    def cancel(self):
        self.initial_plan_wait = None
        self.epoch += 1
        self.accept_planned = False
        self.active_reference_stamp = None
        self.planned[:] = 0.
        self.neupan[:] = 0.
        self.neupan_time = 0.
        self.plan_time = 0.
        self.plan_ros_time = self.neupan_ros_time = 0.

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
        moving_plan = (0 <= self._plan_age() < .25 and self.mode != 'door_wait'
                       and np.isfinite(self.planned).all() and self.planned[0] > .002)
        door_ready = self.mode.startswith('door_') and self.door_path_feasible
        return not (moving_plan or door_ready)

    def update_reference(self):
        preparing_capture = self.pending_door_capture
        self.pending_door_capture = None
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
        if self.raw[0] <= .02 or not np.isfinite(self.raw).all():
            self.sensor_stale_since = None
            self._clear_opening_turn()
            self._clear_door()
            self.cancel()
            self.mode = 'manual'
            return
        if not self.fresh():
            # A missing heartbeat stops motion without becoming a new intent.
            # Keep the checked door through the same bounded sensor grace;
            # absent input must not count as sustained steering away either.
            self.door_away_since = 0.
            if now-self.raw_time >= self.parameter('command_timeout'):
                self._clear_opening_turn()
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
                and self.door_plan_request is None and not self.door_path_feasible
                and abs(self.raw[1]) < .12
                and abs(local_door.center[1]) > .65
                and door_entry_clearance(local_door) < -DOOR_ENTRY_NOISE_TOLERANCE):
            # A remembered door can remain in the two-scan cache after the
            # chair has moved past it or beside it. With straight joystick
            # intent, do not wait forever for an infeasible angled entry.
            # A pending search or a checked staging path can resolve the angle.
            self._clear_door()
            local_door = None
        capture_deferred = False
        if local_door is None:
            world_door, local_door = self._intended_door()
            if (world_door is not None and abs(self.output[0]) > .1
                    and not output_arc_targets_aperture(local_door, *self.output)):
                if (preparing_capture is not None
                        and opening_matches(preparing_capture, world_door)):
                    # A transient slew curvature must not lift the speed cap
                    # while the same door remains selected by live raw intent.
                    capture_deferred = True
                    self.pending_door_capture = world_door
                world_door, local_door = None, None
            if (world_door is not None
                    and (max(abs(self.output[0]), abs(self.measured[0])) > .35+DOOR_CAPTURE_SPEED_TOLERANCE
                         or ((abs(self.output[0]) > .1 or preparing_capture is not None)
                             and not self._door_capture_is_clear(obstacle_points)))):
                # Reach the existing alignment speed under the current
                # controller before entering the door waiting/age policy.
                capture_deferred = True
                self.pending_door_capture = world_door
                world_door, local_door = None, None
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
        elif (self.pending_door_capture is not None
              or (not capture_deferred and self._pending_door_intent())):
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
                    else DOOR_PASS_SPEED if self.mode in ('door_pass', 'door_clear')
                    else .5 if self.mode == 'opening_turn' else 10.)
        if self.pending_door_capture is not None:
            limit = min(limit, .35)
        self.limit_pub.publish(Float32(data=limit))
        reference_speed = TwistStamped()
        reference_speed.header = path.header
        reference_speed.twist.linear.x = limit
        self.reference_speed_pub.publish(reference_speed)
        self.accept_planned = True
        self.active_reference_stamp = path.header.stamp

    def _control_input_signature(self):
        # Receipts change only when accepted callbacks consume new data.
        return (self.epoch, self.mode, self.assist_enabled, self.mode_neutral_seen,
                self.raw_time, self.odom_time, self.plan_time, self.neupan_time,
                self.active_reference_stamp, self.accept_planned, tuple(sorted(self.parameters.items())),
                tuple(sorted((side, received) for side, (received, _) in self.scans.items())))

    def _duplicate_clear_tick(self, stamp, now):
        return (self.use_sim_time and self.assist_enabled and self.mode_neutral_seen
                and math.isfinite(stamp) and stamp == self.last_tick
                and not self.clock_stalled and stamp >= self.clock_last_ros
                and 0. <= now-self.clock_last_advance_wall < .25
                and self.last_clear_tick == (stamp, self._control_input_signature())
                and self.fresh())

    def control(self):
        now = time.monotonic()
        stamp = rospy.Time.now().to_sec()
        if self._duplicate_clear_tick(stamp, now):
            return
        self.last_clear_tick = None
        elapsed = stamp-self.last_tick
        dt = self._control_interval(stamp)
        if dt is None:
            if math.isfinite(stamp):
                self.last_tick = stamp
                self.clock_last_ros = stamp
            self._stop_for_clock(now, stamp, fault='control_interval', elapsed=elapsed)
            return
        self.last_tick = stamp
        tick_acceleration = self.acceleration.copy()
        command = np.zeros(2)
        self.reason = 'clear'
        planner_source = 'stop'
        if not self.assist_enabled or not self.mode_neutral_seen:
            fresh_raw = (not self.clock_stalled and np.isfinite(self.raw).all()
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
                'control_stamp_s': stamp, 'control_dt_s': dt,
                'v': msg.linear.x, 'w': msg.angular.z,
                'planner_source': 'joystick' if fresh_raw and self.mode_neutral_seen else 'stop',
            })))
            return
        planner_stale = not 0 <= self._plan_age(now) < .25 or not np.isfinite(self.planned).all()
        neupan_fresh = self.neupan_fresh(now)
        if (self.mode.startswith('door_') and self.door_path_feasible
                and self.door_path is not None):
            # Keep one controller throughout the checked aperture trajectory.
            # Switching curvature whenever a neural reference is refreshed
            # repeatedly consumes the narrow doorway's braking clearance.
            neupan_fresh = False
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
        if self.override_handoff_time:
            if (not inputs_fresh or self.raw[0] <= .02 or abs(self.raw[1]) <= .25
                    or self.mode in ('front_stop', 'door_wait') or self.recovering):
                self.override_handoff_time = 0.
            elif (self.mode != 'override' and (door_fallback or neupan_fresh
                    or (self.accept_planned and not planner_stale
                        and self.plan_time >= self.override_handoff_time))):
                self.override_handoff_time = 0.
        joystick_override = self.override or bool(self.override_handoff_time)
        if (self.initial_plan_wait is not None and (
                self.initial_plan_wait[0] != self.epoch
                or not 0 <= now-self.initial_plan_wait[1] < .25
                or not inputs_fresh or self.raw[0] <= .02 or self.clock_stalled
                or self.recovering or joystick_override or self.mode.startswith('door_')
                or not planner_stale)):
            self.initial_plan_wait = None
        transition_stop = self.initial_plan_wait is not None
        if not inputs_fresh:
            self.reason = 'stale_input'
        elif np.linalg.norm(self.raw) < .01:
            self.reason = 'user_stop'
        elif (self.raw[0] > .02 and not joystick_override and self.mode != 'front_stop'
              and self.mode != 'door_wait' and planner_stale and not door_fallback and not self.recovering
              and not transition_stop):
            self.reason = 'planner_timeout'
        else:
            if self.recovering:
                planner_source = 'recovery'
                desired = self.raw / max(1., abs(self.raw[0])/.05, abs(self.raw[1])/.1)
                if self.door_plan_request is not None and not self.door_plan_request[0].done():
                    desired *= .2
                self.reason = 'clearance_recovery'
            elif neupan_fresh and self.raw[0] > .02 and not joystick_override and self.mode not in ('door_wait', 'front_stop'):
                planner_source = 'neupan'
                desired = self.neupan.copy()
            elif transition_stop:
                planner_source = 'transition_stop'
                desired = np.zeros(2)
            elif self.mode == 'door_wait':
                desired = np.zeros(2)
            elif self.raw[0] <= .02 or joystick_override or self.mode == 'front_stop':
                planner_source = 'joystick'
                desired = self.raw.copy()
                if self.mode == 'front_stop' and self.raw[0] > .02 and not joystick_override:
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
                planner_source = 'door_follower'
                desired = self._safe_door_command(points, stamp)
                if planner_stale:
                    self.reason = 'planner_fallback'
            elif neupan_fresh or (0 <= self._plan_age(now) < .25 and np.isfinite(self.planned).all()):
                planner_source = 'neupan' if neupan_fresh else 'local_follower'
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
            if (self.pending_door_capture is not None and not self.recovering
                    and desired[0] > .35):
                desired *= .35/desired[0]
            planned_speed = (not self.recovering
                             and self.mode in ('manual', 'wall', 'front_stop', 'opening_turn', 'override') and self.raw[0] > .02)
            ordinary_slew = (not self.recovering and not door_fallback
                             and self.mode != 'door_wait'
                             and (planned_speed or planner_source == 'joystick'))
            self.reverse_slew_active = bool(ordinary_slew and (
                self.raw[0] < 0. or self.output[0] < 0.
                or (self.reverse_slew_active and self.acceleration[0] < 0.)))
            ordinary_lower_bound = -.4 if self.reverse_slew_active else 0.
            if planned_speed:
                desired = self._safe_assisted_command(desired, points)
            # Recover acceleration using the active controller's slew policy.
            # The independent safety rejection below bypasses comfort limits.
            settling_pivot = False
            if door_fallback and not np.any(desired):
                rotation = self._pending_door_rotation()
                settling_pivot = (rotation is not None
                                  and rotation[0] == self.door_rotation_index
                                  and rotation[1] < .04 and abs(rotation[2]) < .04)
            if transition_stop:
                command, self.acceleration = self._door_wait_slew(dt)
            elif planned_speed:
                command, self.acceleration = self._assisted_slew(
                    desired, dt, linear_lower_bound=ordinary_lower_bound)
            elif (self.mode == 'door_wait' or settling_pivot) and not self.recovering:
                command, self.acceleration = self._door_wait_slew(dt)
            elif door_fallback and not self.recovering:
                command, self.acceleration = self._assisted_slew(desired, dt)
            elif not self.recovering and planner_source == 'joystick':
                # Joystick targets remain soft while acceleration recovers;
                # keep directional physical bounds and the final guard.
                command, self.acceleration = self._assisted_slew(
                    desired, dt, linear_lower_bound=ordinary_lower_bound)
            else:
                target_a = np.clip((desired-self.output)/dt, [-.5, -.8], [.5, .8])
                self.acceleration += np.clip(target_a-self.acceleration,
                                             -JERK_LIMITS*dt, JERK_LIMITS*dt)
                command = self.output + self.acceleration*dt
                command = np.minimum(np.maximum(command, np.minimum(self.output, desired)), np.maximum(self.output, desired))
            if door_fallback and (not self.door_path_feasible or self.door_path is None):
                command[:] = 0.
                self.acceleration[:] = 0.
                self.reason = 'invalid_door_path'
            elif (door_fallback and not settling_pivot
                  and self.mode in ('door_align', 'door_pass', 'door_clear')):
                pending = self._pending_door_rotation()
                if pending is None or pending[0] != self.door_rotation_index:
                    target_w = self._door_angular(command[0]) if command[0] > .002 else 0.
                    # The first slew has already updated acceleration;
                    # use the real tick-entry state, never integrate twice.
                    smooth, _ = self._assisted_slew(
                        np.array([command[0], target_w]), dt,
                        acceleration=tick_acceleration)
                    command[1] = smooth[1]
                    # Preserve the actual derivative across checked tracking phases.
                    self.acceleration[1] = (command[1]-self.output[1])/dt
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
                                 state_age=self._guard_state_age(stamp), recovery=self.recovering):
                    command = candidate
                    if adjusted_reason is not None:
                        self.reason = adjusted_reason
                        self.acceleration[:] = 0.
                    break
            else:
                command[:] = 0.
                self.reason = 'emergency_stop'
        if self.reason != 'clear':
            self.initial_plan_wait = None
        if self.reason in ('stale_input', 'user_stop', 'emergency_stop', 'planner_timeout'):
            self.acceleration[:] = 0.
        self.output = command
        msg = Twist()
        msg.linear.x, msg.angular.z = map(float, command)
        self.pub.publish(msg)
        status = {'control_stamp_s': stamp, 'control_dt_s': dt,
                  'mode': self.mode, 'reason': self.reason, 'v': msg.linear.x, 'w': msg.angular.z,
                  'planner_source': planner_source}
        if self.reason == 'stale_input':
            status['input_age_s'] = {
                'raw': now-self.raw_time,
                'odom_receipt': stamp-self.odom_ros_time if self.use_sim_time else now-self.odom_time,
                'odom_receipt_clock': 'ros' if self.use_sim_time else 'monotonic',
                'odom_receipt_monotonic': now-self.odom_time,
                'odom_stamp': None if self.odom_stamp is None else stamp-self.odom_stamp,
                **{f'scan_{side}': now-received for side, (received, _) in self.scans.items()},
            }
        if self.opening_turn is not None and self.pose is not None:
            opening = self._local_opening(self.opening_turn)
            status['opening'] = {'center': opening.center, 'width': opening.width,
                                 'ready': bool(self._opening_turn_ready(opening))}
        if self.door is not None and self.pose is not None:
            door = self._local_opening(self.door)
            status['door'] = {'center': door.center, 'heading': door.heading,
                              'width': door.width, 'entry_clearance': door_entry_clearance(door)}
        self.status.publish(String(data=json.dumps(status)))
        if self.reason == 'clear':
            self.last_clear_tick = (stamp, self._control_input_signature())

    def destroy_node(self):
        self.clock_watchdog_stop.set()
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
        if (self.clock_watchdog_thread is not None
                and self.clock_watchdog_thread is not threading.current_thread()):
            self.clock_watchdog_thread.join(timeout=1.)
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
