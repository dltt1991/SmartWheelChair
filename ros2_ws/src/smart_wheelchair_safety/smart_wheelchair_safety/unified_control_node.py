"""Shared-control references for Nav2 MPPI, followed by an independent guard."""

import copy
import json
import math
import time
import numpy as np
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.time import Time
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import PoseStamped, TransformStamped, Twist
from nav_msgs.msg import Odometry, Path
from nav2_msgs.action import FollowPath
from nav2_msgs.msg import SpeedLimit
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from tf2_ros import Buffer, TransformBroadcaster, StaticTransformBroadcaster, TransformListener, TransformException

from smart_wheelchair_safety.unified_geometry import (
    Door, braking_clear, door_reference, extract_lines,
    find_door, transform_points, wall_reference,
)


class UnifiedControlNode(Node):
    def __init__(self):
        super().__init__('unified_control')
        for name, value in {'max_speed': .8, 'scan_timeout': .45,
                            'odom_timeout': .10,
                            'command_timeout': .3, 'braking_deceleration': .5,
                            'reaction_time': .2, 'hard_margin': .04,
                            'wall_clearance': .12,
                            'front_stop_margin': .12}.items():
            self.declare_parameter(name, value)
            if not math.isfinite(self.get_parameter(name).value) or self.get_parameter(name).value <= 0:
                raise ValueError(f'{name} must be finite and positive')
        self.raw = np.zeros(2)
        self.measured = np.zeros(2)
        self.planned = np.zeros(2)
        self.output = np.zeros(2)
        self.acceleration = np.zeros(2)
        self.pose = None
        self.raw_time = self.odom_time = self.plan_time = 0.
        self.odom_stamp = None
        self.scans = {}
        self.door = None
        self.door_obstacles = np.empty((0, 2))
        self.mode = 'waiting'
        self.front_blocked = False
        self.wall_side = 0
        self.override = False
        self.reason = 'starting'
        self.goal = None
        self.pending_goal = False
        self.epoch = 0
        self.last_tick = self.get_clock().now().nanoseconds / 1e9
        self.tf = Buffer()
        self.listener = TransformListener(self.tf, self)
        self.transforms = TransformBroadcaster(self)
        self.static = StaticTransformBroadcaster(self)
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
        self.pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.status = self.create_publisher(String, 'shared_control/status', 10)
        self.reference = self.create_publisher(Path, 'shared_control/reference', 10)
        self.limit_pub = self.create_publisher(SpeedLimit, 'speed_limit', 10)
        self.scan_pubs = {side: self.create_publisher(LaserScan, f'unified_scan_{side}', 10)
                          for side in ('left', 'right')}
        self.create_subscription(Twist, 'cmd_vel_raw', self.on_raw, 10)
        self.create_subscription(Twist, 'cmd_vel_planned', self.on_planned, 10)
        self.create_subscription(Odometry, 'odom', self.on_odom, 10)
        for side in ('left', 'right'):
            self.create_subscription(LaserScan, f'scan_{side}',
                                     lambda msg, side=side: self.on_scan(msg, side), qos_profile_sensor_data)
        self.client = ActionClient(self, FollowPath, 'follow_path')
        self.create_timer(.5, self.update_reference)
        self.create_timer(.05, self.control)

    def on_raw(self, msg):
        self.raw = np.array([msg.linear.x, msg.angular.z])
        self.raw_time = time.monotonic()
        if self.raw[0] <= .02:
            self.front_blocked = False
        self.override = (self.wall_side * self.raw[1] < -.15
                         or (self.mode in ('door', 'override') and abs(self.raw[1]) > .25))
        if self.raw[0] <= .02 or abs(self.raw[1]) > .25:
            self.door = None

    def on_planned(self, msg):
        self.planned = np.array([msg.linear.x, msg.angular.z])
        self.plan_time = time.monotonic()

    def on_odom(self, msg):
        q = msg.pose.pose.orientation
        p = msg.pose.pose.position
        age = (self.get_clock().now().nanoseconds-Time.from_msg(msg.header.stamp).nanoseconds)/1e9
        values = [p.x, p.y, q.x, q.y, q.z, q.w, msg.twist.twist.linear.x, msg.twist.twist.angular.z]
        if not -.05 <= age <= self.get_parameter('odom_timeout').value or not np.isfinite(values).all():
            return
        if abs(q.x*q.x+q.y*q.y+q.z*q.z+q.w*q.w-1.) > .01:
            return
        yaw = math.atan2(2*(q.w*q.z + q.x*q.y), 1-2*(q.y*q.y + q.z*q.z))
        self.pose = (p.x, p.y, yaw)
        self.measured = np.array([msg.twist.twist.linear.x, msg.twist.twist.angular.z])
        self.odom_time = time.monotonic()
        self.odom_stamp = Time.from_msg(msg.header.stamp).nanoseconds / 1e9
        t = TransformStamped()
        t.header = msg.header
        t.header.frame_id = 'odom'
        t.child_frame_id = 'rear_axle'
        t.transform.translation.x, t.transform.translation.y = p.x, p.y
        t.transform.rotation = q
        self.transforms.sendTransform(t)

    def on_scan(self, msg, side):
        scan_age = (self.get_clock().now().nanoseconds - Time.from_msg(msg.header.stamp).nanoseconds) / 1e9
        if not -.05 <= scan_age <= self.get_parameter('scan_timeout').value:
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
            t = self.tf.lookup_transform('odom', outgoing.header.frame_id, Time.from_msg(msg.header.stamp))
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
                and now-self.raw_time < self.get_parameter('command_timeout').value
                and self.odom_stamp is not None
                and -.05 <= self.get_clock().now().nanoseconds/1e9-self.odom_stamp < self.get_parameter('odom_timeout').value
                and now-self.odom_time < self.get_parameter('odom_timeout').value and len(self.scans) == 2
                and all(now-stamp < self.get_parameter('scan_timeout').value for stamp, _ in self.scans.values()))

    def cancel(self):
        self.epoch += 1
        if self.goal is not None:
            self.goal.cancel_goal_async()
            self.goal = None

    def update_reference(self):
        if len(self.door_obstacles) and self.pose is not None:
            nearby = np.linalg.norm(self.door_obstacles-np.array(self.pose[:2]), axis=1) < 4.5
            self.door_obstacles = self.door_obstacles[nearby]
        if not self.fresh() or self.raw[0] <= .02:
            self.cancel()
            self.mode = 'manual'
            return
        if self.override:
            self.cancel()
            self.mode = 'override'
            return
        groups, _ = self.points()
        lines = [line for group in groups for line in extract_lines(group)]
        local_door = None
        if self.door is not None:
            center = transform_points([self.door.center], self.pose, inverse=True)[0]
            heading = self.door.heading-self.pose[2]
            normal = np.array([math.cos(heading), math.sin(heading)])
            if -center @ normal > .6:
                self.door = None
            else:
                local_door = Door(tuple(center), heading, self.door.width)
        elif abs(self.raw[1]) < .25:
            local_door = find_door(lines)
            if local_door is not None:
                center = transform_points([local_door.center], self.pose)[0]
                self.door = Door(tuple(center), local_door.heading+self.pose[2], local_door.width)
                # Keep observed jambs through rear clearance even if the
                # forward-mounted scanners temporarily lose their inner edges.
                points = np.vstack(groups)
                normal = np.array([math.cos(local_door.heading), math.sin(local_door.heading)])
                near_plane = np.abs((points-local_door.center) @ normal) < .3
                observed = transform_points(points[near_plane], self.pose)
                memory = np.vstack((self.door_obstacles, observed))
                _, indices = np.unique(np.floor(memory/.025).astype(int), axis=0, return_index=True)
                self.door_obstacles = memory[indices]
        frontal_wall = any(abs(line.heading) > 1.35 and .97 < -line.distance/math.sin(line.heading) < 3.
                           and min(line.start[1], line.end[1]) < -.4
                           and max(line.start[1], line.end[1]) > .4 for line in lines)
        if local_door is not None:
            self.front_blocked = False
            self.wall_side = 0
            local_path, self.mode = door_reference(local_door), 'door'
        elif frontal_wall or self.front_blocked:
            self.front_blocked = True
            self.wall_side = 0
            self.mode = 'front_stop'
            self.cancel()
            return
        else:
            local_path, self.mode, self.wall_side = wall_reference(
                lines, *self.raw, body_clearance=self.get_parameter('wall_clearance').value)
        world_xy = transform_points(local_path[:, :2], self.pose)
        path = Path()
        path.header.frame_id = 'odom'
        path.header.stamp = self.get_clock().now().to_msg()
        for xy, angle in zip(world_xy, local_path[:, 2] + self.pose[2]):
            p = PoseStamped()
            p.header = path.header
            p.pose.position.x, p.pose.position.y = float(xy[0]), float(xy[1])
            p.pose.orientation.z, p.pose.orientation.w = math.sin(angle/2), math.cos(angle/2)
            path.poses.append(p)
        self.reference.publish(path)
        limit = SpeedLimit()
        limit.speed_limit = min(self.raw[0], self.get_parameter('max_speed').value,
                                .45 if self.mode == 'door' else 10.)
        self.limit_pub.publish(limit)
        if not self.pending_goal and self.client.server_is_ready():
            request = FollowPath.Goal()
            request.path = path
            request.controller_id, request.goal_checker_id, request.progress_checker_id = 'FollowPath', 'goal', 'progress'
            self.pending_goal = True
            epoch = self.epoch
            def accepted(future):
                self.pending_goal = False
                handle = future.result()
                if handle.accepted:
                    if epoch != self.epoch:
                        handle.cancel_goal_async()
                    else:
                        self.goal = handle
            self.client.send_goal_async(request).add_done_callback(accepted)

    def control(self):
        now = time.monotonic()
        stamp = self.get_clock().now().nanoseconds / 1e9
        dt = min(.1, max(.001, stamp-self.last_tick))
        self.last_tick = stamp
        command = np.zeros(2)
        self.reason = 'clear'
        if not self.fresh():
            self.reason = 'stale_input'
        elif np.linalg.norm(self.raw) < .01:
            self.reason = 'user_stop'
        elif (self.raw[0] > .02 and not self.override and self.mode != 'front_stop'
              and (now-self.plan_time >= .25 or not np.isfinite(self.planned).all())):
            self.reason = 'planner_timeout'
        else:
            _, points = self.points()
            if self.raw[0] <= .02 or self.override or self.mode == 'front_stop':
                desired = self.raw.copy()
                if self.mode == 'front_stop' and self.raw[0] > .02 and not self.override:
                    desired[1] = 0.
                    front = points[(points[:, 0] > 0.) & (np.abs(points[:, 1]) <= .44)]
                    gap = (float(front[:, 0].min())-.97-self.get_parameter('front_stop_margin').value
                           if len(front) else 0.)
                    a = self.get_parameter('braking_deceleration').value
                    # Also reserve the ramp to comfortable deceleration at
                    # the normal linear jerk limit of 1 m/s^3.
                    ar = a*(self.get_parameter('reaction_time').value+a/1.)
                    allowed = math.sqrt(ar*ar+2*a*max(0., gap))-ar if gap > .01 else 0.
                    desired[0] = min(desired[0], allowed)
            elif now-self.plan_time < .25 and np.isfinite(self.planned).all():
                desired = self.planned.copy()
            else:
                desired = np.zeros(2)
                self.reason = 'planner_timeout'
            desired[0] = np.clip(desired[0], -.4 if self.raw[0] < 0. else 0.,
                                 min(max(0., self.raw[0]), self.get_parameter('max_speed').value))
            desired[1] = np.clip(desired[1], -.65, .65)
            # Slew-limit acceleration; snap at the target to avoid overshoot.
            # The independent safety rejection below bypasses comfort limits.
            target_a = np.clip((desired-self.output)/dt, [-.5, -.8], [.5, .8])
            self.acceleration += np.clip(target_a-self.acceleration, -np.array([1., 2.])*dt, np.array([1., 2.])*dt)
            command = self.output + self.acceleration*dt
            command = np.minimum(np.maximum(command, np.minimum(self.output, desired)), np.maximum(self.output, desired))
            for scale in (1., .8, .6, .4, .2, 0.):
                candidate = command * scale
                if braking_clear(points, candidate, self.measured,
                                 margin=self.get_parameter('hard_margin').value,
                                 reaction=self.get_parameter('reaction_time').value,
                                 deceleration=self.get_parameter('braking_deceleration').value,
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
        self.status.publish(String(data=json.dumps({'mode': self.mode, 'reason': self.reason,
                                                   'v': msg.linear.x, 'w': msg.angular.z})))


def main(args=None):
    rclpy.init(args=args)
    node = UnifiedControlNode()
    try:
        rclpy.spin(node)
    finally:
        node.pub.publish(Twist())
        node.destroy_node()
        rclpy.shutdown()
