# ROS1 Noetic Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the complete SmartWheelChair simulation on ROS Noetic, Ubuntu 20.04, and Gazebo 11 without losing its browser controls, assisted-control modes, safety behavior, scenes, or visual feedback.

**Architecture:** Convert both packages and the workspace to catkin, port the ROS-facing Python shells from rclpy to rospy while retaining the existing pure geometry and state-machine logic, and replace Nav2 MPPI with a small tested differential-drive trajectory sampler. Replace Gazebo Harmonic systems and bridging with Gazebo 11 ROS plugins; keep the unified controller as the sole final velocity authority.

**Tech Stack:** ROS Noetic, Ubuntu 20.04, Python 3, rospy, tf2_ros, catkin, NumPy, Gazebo 11/gazebo_ros_pkgs, C++14, Docker Compose, unittest.

## Global Constraints

- Workspace path is `catkin_ws`; no operational `ros2_ws` path remains.
- Runtime is ROS Noetic on Ubuntu 20.04 with Gazebo 11.
- Public topic names and user-visible behavior remain unchanged except `/speed_limit`, whose unavailable ROS2 type becomes `std_msgs/Float32`.
- Nav2, ROS2 launch, ament, rclpy, ros_gz, and Gazebo Harmonic dependencies are removed.
- The final independent braking envelope remains in `unified_control_node`.
- Existing `.vscode/` content is never staged or modified.

---

## File map

- `catkin_ws/src/smart_wheelchair_safety/CMakeLists.txt`, `package.xml`, `setup.py`: catkin Python package metadata and executable installation.
- `catkin_ws/src/smart_wheelchair_safety/nodes/*.py`: thin rospy executable entry points.
- `catkin_ws/src/smart_wheelchair_safety/smart_wheelchair_safety/local_path_follower.py`: ROS-independent trajectory sampling and scoring.
- `catkin_ws/src/smart_wheelchair_safety/smart_wheelchair_safety/local_path_follower_node.py`: rospy message/time/TF adapter.
- `catkin_ws/src/smart_wheelchair_safety/smart_wheelchair_safety/web_joystick_node.py`: existing web server with rospy publishers and subscriber.
- `catkin_ws/src/smart_wheelchair_safety/smart_wheelchair_safety/unified_control_node.py`: existing state machine with rospy, TF2, reference, and plan interfaces.
- `catkin_ws/src/smart_wheelchair_gazebo/CMakeLists.txt`, `package.xml`: catkin Gazebo plugin package.
- `catkin_ws/src/smart_wheelchair_gazebo/launch/sim.launch`: ROS1 launch composition and parameters.
- `catkin_ws/src/smart_wheelchair_gazebo/config/local_path_follower.yaml`: sampler values corresponding to current MPPI bounds.
- `catkin_ws/src/smart_wheelchair_gazebo/models/smart_wheelchair/model.sdf`: Gazebo 11 drive, lidar, camera, and preview plugins.
- `catkin_ws/src/smart_wheelchair_gazebo/worlds/*.world`: Gazebo 11 worlds with unchanged geometry.
- `catkin_ws/src/smart_wheelchair_gazebo/src/trajectory_preview_plugin.cc`: Gazebo 11 ModelPlugin and marker transport.
- `docker/*`, `docker-compose.yml`: Noetic image, VNC launch, and workspace sourcing.
- `scripts/probe_unified_control.py`: ROS1 integration probe.
- `README.md`, `docs/architecture-and-algorithms.md`: ROS1 usage and architecture.

---

### Task 1: Rename the workspace and establish catkin packages

**Files:**
- Rename: `ros2_ws` → `catkin_ws`
- Create: `catkin_ws/src/smart_wheelchair_safety/CMakeLists.txt`
- Modify: `catkin_ws/src/smart_wheelchair_safety/package.xml`
- Modify: `catkin_ws/src/smart_wheelchair_safety/setup.py`
- Delete: `catkin_ws/src/smart_wheelchair_safety/setup.cfg`
- Delete: `catkin_ws/src/smart_wheelchair_safety/resource/smart_wheelchair_safety`
- Modify: `catkin_ws/src/smart_wheelchair_gazebo/CMakeLists.txt`
- Modify: `catkin_ws/src/smart_wheelchair_gazebo/package.xml`
- Create before rename, then move with workspace: `ros2_ws/src/smart_wheelchair_gazebo/test/test_ros1_layout.py` → `catkin_ws/src/smart_wheelchair_gazebo/test/test_ros1_layout.py`

**Interfaces:**
- Produces: catkin packages discoverable as `smart_wheelchair_safety` and `smart_wheelchair_gazebo`.
- Produces: importable Python package `smart_wheelchair_safety`.

- [ ] **Step 1: Write the failing ROS1 layout test**

```python
# ros2_ws/src/smart_wheelchair_gazebo/test/test_ros1_layout.py
import pathlib
import unittest

ROOT = pathlib.Path(__file__).parents[4]


class Ros1LayoutTest(unittest.TestCase):
    def test_workspace_and_packages_use_catkin_only(self):
        self.assertTrue((ROOT / "catkin_ws/src").is_dir())
        self.assertFalse((ROOT / "ros2_ws").exists())
    def test_both_packages_export_catkin(self):
        for package in ("smart_wheelchair_safety", "smart_wheelchair_gazebo"):
            directory = ROOT / "catkin_ws/src" / package
            self.assertIn("catkin_package(", (directory / "CMakeLists.txt").read_text())
            self.assertIn("<buildtool_depend>catkin</buildtool_depend>",
                          (directory / "package.xml").read_text())
```

- [ ] **Step 2: Run the test and verify the old layout fails**

Run: `python3 ros2_ws/src/smart_wheelchair_gazebo/test/test_ros1_layout.py`

Expected: FAIL because `catkin_ws/src` does not exist.

- [ ] **Step 3: Rename the workspace and replace ament metadata**

Run: `git mv ros2_ws catkin_ws`

Create the safety CMake file:

```cmake
cmake_minimum_required(VERSION 3.0.2)
project(smart_wheelchair_safety)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs nav_msgs rospy sensor_msgs std_msgs tf2_ros
)

catkin_python_setup()
catkin_package(CATKIN_DEPENDS geometry_msgs nav_msgs rospy sensor_msgs std_msgs tf2_ros)

catkin_install_python(PROGRAMS
  nodes/local_path_follower_node
  nodes/unified_control_node
  nodes/web_joystick_node
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

if(CATKIN_ENABLE_TESTING)
  find_package(rostest REQUIRED)
  catkin_add_nosetests(test/test_joystick.py)
  catkin_add_nosetests(test/test_unified_geometry.py)
endif()
```

Change safety `setup.py` to:

```python
from setuptools import find_packages, setup

setup(
    name="smart_wheelchair_safety",
    version="0.1.0",
    packages=find_packages(),
)
```

Use package format 2 and declare `catkin` as build tool plus `geometry_msgs`, `nav_msgs`, `rospy`, `sensor_msgs`, `std_msgs`, `tf2_ros`, and `python3-numpy` as dependencies. Remove every ament/Nav2 dependency. Convert the Gazebo package metadata to catkin now, leaving its actual Gazebo targets to Task 5.

- [ ] **Step 4: Run the layout tests and host-safe baseline**

Run: `PYTHONPATH=catkin_ws/src/smart_wheelchair_safety python3 -m unittest catkin_ws/src/smart_wheelchair_gazebo/test/test_ros1_layout.py catkin_ws/src/smart_wheelchair_safety/test/test_joystick.py catkin_ws/src/smart_wheelchair_safety/test/test_unified_geometry.py`

Expected: PASS with no failures.

- [ ] **Step 5: Commit**

```bash
git add catkin_ws
git commit -m "build: convert workspace packages to catkin"
```

---

### Task 2: Implement the ROS-independent local path follower

**Files:**
- Create: `catkin_ws/src/smart_wheelchair_safety/smart_wheelchair_safety/local_path_follower.py`
- Create: `catkin_ws/src/smart_wheelchair_safety/test/test_local_path_follower.py`

**Interfaces:**
- Produces: `select_velocity(path, obstacles, speed_limit, previous_velocity=None) -> numpy.ndarray`.
- Consumes: local-frame `path` shaped `(N, 3)`, local obstacle points shaped `(M, 2)`, positive speed limit, optional previous `[v, w]`.

- [ ] **Step 1: Write failing behavior tests**

```python
import unittest
import numpy as np

from smart_wheelchair_safety.local_path_follower import select_velocity


class LocalPathFollowerTest(unittest.TestCase):
    def test_straight_path_selects_forward_motion(self):
        path = np.column_stack((np.linspace(0., 3., 31), np.zeros(31), np.zeros(31)))
        command = select_velocity(path, np.empty((0, 2)), .8)
        self.assertGreater(command[0], .2)
        self.assertAlmostEqual(command[1], 0., delta=.11)

    def test_left_curve_selects_positive_angular_velocity(self):
        angles = np.linspace(0., .7, 31)
        path = np.column_stack((2*np.sin(angles), 2*(1-np.cos(angles)), angles))
        command = select_velocity(path, np.empty((0, 2)), .6)
        self.assertGreater(command[0], 0.)
        self.assertGreater(command[1], .1)

    def test_speed_limit_is_absolute(self):
        path = np.column_stack((np.linspace(0., 3., 31), np.zeros(31), np.zeros(31)))
        self.assertLessEqual(select_velocity(path, np.empty((0, 2)), .25)[0], .25)

    def test_obstacle_across_footprint_stops(self):
        path = np.column_stack((np.linspace(0., 3., 31), np.zeros(31), np.zeros(31)))
        obstacles = np.array([[x, y] for x in np.linspace(.9, 1.2, 5)
                              for y in np.linspace(-.4, .4, 9)])
        np.testing.assert_allclose(select_velocity(path, obstacles, .8), [0., 0.])

    def test_invalid_or_empty_inputs_stop(self):
        np.testing.assert_allclose(select_velocity(np.empty((0, 3)),
                                                   np.empty((0, 2)), .8), [0., 0.])
        path = np.array([[0., 0., float("nan")]])
        np.testing.assert_allclose(select_velocity(path, np.empty((0, 2)), .8), [0., 0.])
```

- [ ] **Step 2: Run and verify RED**

Run: `PYTHONPATH=catkin_ws/src/smart_wheelchair_safety python3 -m unittest catkin_ws/src/smart_wheelchair_safety/test/test_local_path_follower.py -v`

Expected: ERROR with `ModuleNotFoundError: smart_wheelchair_safety.local_path_follower`.

- [ ] **Step 3: Implement the minimum sampler**

Use fixed current requirements: `v` samples from zero to the absolute limit, `w` samples from `-0.65` to `0.65`, a 1.0 s horizon at 0.1 s, footprint `x=[-0.25,0.97]`, `y=[-0.40,0.40]`, and 0.04 m margin. Integrate each constant command with differential-drive kinematics, reject swept-footprint collisions, and score nearest-path distance plus wrapped heading error minus forward progress. Always include zero as a valid fallback and break score ties in favor of the smaller command change.

```python
import math
import numpy as np


def _rollout(linear, angular, horizon=1., dt=.1):
    times = np.arange(0., horizon + dt/2., dt)
    headings = angular*times
    if abs(angular) < 1e-9:
        x = linear*times
        y = np.zeros_like(times)
    else:
        radius = linear/angular
        x = radius*np.sin(headings)
        y = radius*(1.-np.cos(headings))
    return np.column_stack((x, y, headings))


def _trajectory_clear(trajectory, obstacles):
    if not len(obstacles):
        return True
    for x, y, heading in trajectory:
        cosine, sine = math.cos(heading), math.sin(heading)
        delta = obstacles - [x, y]
        local_x = cosine*delta[:, 0] + sine*delta[:, 1]
        local_y = -sine*delta[:, 0] + cosine*delta[:, 1]
        if np.any((local_x >= -.29) & (local_x <= 1.01)
                  & (np.abs(local_y) <= .44)):
            return False
    return True


def _path_score(trajectory, path):
    final = trajectory[-1]
    nearest = int(np.argmin(np.linalg.norm(path[:, :2]-final[:2], axis=1)))
    distance = np.linalg.norm(path[nearest, :2]-final[:2])
    heading = abs(math.atan2(math.sin(path[nearest, 2]-final[2]),
                             math.cos(path[nearest, 2]-final[2])))
    progress = nearest/max(1, len(path)-1)
    return 4.*distance + 2.*heading - 2.*progress


def select_velocity(path, obstacles, speed_limit, previous_velocity=None):
    path = np.asarray(path, dtype=float)
    obstacles = np.asarray(obstacles, dtype=float)
    previous = np.zeros(2) if previous_velocity is None else np.asarray(previous_velocity, dtype=float)
    if (path.ndim != 2 or path.shape[1:] != (3,) or not len(path)
            or obstacles.ndim != 2 or obstacles.shape[1:] != (2,)
            or not np.isfinite(path).all() or not np.isfinite(obstacles).all()
            or not np.isfinite(speed_limit) or speed_limit <= 0.):
        return np.zeros(2)
    candidates = [(v, w)
                  for v in np.linspace(0., min(float(speed_limit), .8), 9)
                  for w in np.linspace(-.65, .65, 15)]
    safe = [candidate for candidate in candidates
            if _trajectory_clear(_rollout(*candidate), obstacles)]
    if not safe:
        return np.zeros(2)
    return np.asarray(min(safe, key=lambda command:
                          _path_score(_rollout(*command), path)
                          + .05*np.linalg.norm(np.asarray(command)-previous)))
```

- [ ] **Step 4: Verify GREEN and regression tests**

Run: `PYTHONPATH=catkin_ws/src/smart_wheelchair_safety python3 -m unittest catkin_ws/src/smart_wheelchair_safety/test/test_local_path_follower.py catkin_ws/src/smart_wheelchair_safety/test/test_unified_geometry.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add catkin_ws/src/smart_wheelchair_safety
git commit -m "feat: add ROS1 local path follower"
```

---

### Task 3: Add the rospy follower node and port the web joystick

**Files:**
- Create: `catkin_ws/src/smart_wheelchair_safety/smart_wheelchair_safety/local_path_follower_node.py`
- Create: `catkin_ws/src/smart_wheelchair_safety/nodes/local_path_follower_node`
- Create: `catkin_ws/src/smart_wheelchair_safety/nodes/web_joystick_node`
- Modify: `catkin_ws/src/smart_wheelchair_safety/smart_wheelchair_safety/web_joystick_node.py`
- Modify: `catkin_ws/src/smart_wheelchair_safety/test/test_web_joystick.py`
- Create: `catkin_ws/src/smart_wheelchair_safety/test/test_local_path_follower_node.py`
- Create: `catkin_ws/src/smart_wheelchair_safety/test/web_and_follower.test`

**Interfaces:**
- Follower subscribes: `/shared_control/reference nav_msgs/Path`, `/speed_limit std_msgs/Float32`, `/odom nav_msgs/Odometry`, `/unified_scan_left sensor_msgs/LaserScan`, `/unified_scan_right sensor_msgs/LaserScan`.
- Follower publishes: `/cmd_vel_planned geometry_msgs/TwistStamped` at 20 Hz.
- Web publishes: `/cmd_vel_raw geometry_msgs/Twist`, `/assist_enabled std_msgs/Bool`; subscribes to rear `sensor_msgs/Image`.

- [ ] **Step 1: Port tests to require rospy and add stale-input node test**

Replace the availability guard with:

```python
ROS_AVAILABLE = importlib.util.find_spec("rospy") is not None
```

Initialize each test module once with `rospy.init_node(..., anonymous=True, disable_signals=True)`. Preserve all existing web assertions. Add a follower test that seeds a valid path/odom/scans/limit, observes a nonzero stamped plan, advances the injected monotonic clock beyond 0.25 s, and observes a zero plan. Give ROS test files a Python 3 shebang and executable bit, register them in `web_and_follower.test` with `<test pkg="smart_wheelchair_safety" type="...">` entries, and add `add_rostest(test/web_and_follower.test)` under `CATKIN_ENABLE_TESTING`.

- [ ] **Step 2: Run and verify RED inside Noetic**

Run: `rostest smart_wheelchair_safety web_and_follower.test`

Expected: FAIL because the follower node/module does not exist.

- [ ] **Step 3: Implement thin rospy adapters**

The follower node stores each input with `time.monotonic()`, converts the world path into the rear-axle frame from odometry, converts finite scan rays using fixed lidar poses `(0.79, ±0.26)`, calls `select_velocity`, and publishes:

```python
message = TwistStamped()
message.header.stamp = rospy.Time.now()
message.header.frame_id = "rear_axle"
message.twist.linear.x = float(command[0])
message.twist.angular.z = float(command[1])
self.publisher.publish(message)
```

Port `WebJoystickNode` by replacing declaration/get-parameter calls with private rospy parameters, `create_publisher` with `rospy.Publisher(..., queue_size=10, latch=...)`, subscriptions with `rospy.Subscriber`, and timers with `rospy.Timer`. Preserve the HTTP locking, session/revision protocol, neutral interlock, image handling, and shutdown zero publication.

Executable wrappers have exactly this shape:

```python
#!/usr/bin/env python3
from smart_wheelchair_safety.web_joystick_node import main

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run web and follower tests**

Run: `rostest smart_wheelchair_safety web_and_follower.test`

Expected: all migrated web tests and follower-node tests PASS.

- [ ] **Step 5: Commit**

```bash
git add catkin_ws/src/smart_wheelchair_safety
git commit -m "feat: port web and planner nodes to rospy"
```

---

### Task 4: Port the unified controller from rclpy/Nav2 to rospy

**Files:**
- Create: `catkin_ws/src/smart_wheelchair_safety/nodes/unified_control_node`
- Modify: `catkin_ws/src/smart_wheelchair_safety/smart_wheelchair_safety/unified_control_node.py`
- Modify: `catkin_ws/src/smart_wheelchair_safety/test/test_unified_node.py`
- Create: `catkin_ws/src/smart_wheelchair_safety/test/unified_control.test`
- Create: `catkin_ws/src/smart_wheelchair_safety/test/safety_regression.test`

**Interfaces:**
- Preserves all existing subscribers and publishers.
- Produces `/shared_control/reference nav_msgs/Path` and `/speed_limit std_msgs/Float32`.
- Consumes `/cmd_vel_planned geometry_msgs/TwistStamped`; publishes final `/cmd_vel geometry_msgs/Twist`.

- [ ] **Step 1: Change the node tests first**

Replace rclpy setup/spin/shutdown with one anonymous rospy node and direct callback invocation. Replace mocked ROS2 parameters with a test helper:

```python
def parameter(node, name):
    return node.parameter(name)
```

Keep every existing state-machine assertion, including mode transition, old-plan rejection, front stop, wall/opening intent, door phases, scan quality, data timeouts, jerk limiting, and braking behavior. Change `nav2_msgs/SpeedLimit` expectations to `std_msgs/Float32.data`. Register the single node suite in `unified_control.test`; register all safety test files in `safety_regression.test`.

- [ ] **Step 2: Run and verify RED**

Run: `rostest smart_wheelchair_safety unified_control.test`

Expected: FAIL on the remaining `rclpy` and `nav2_msgs` imports.

- [ ] **Step 3: Port the ROS shell without changing control math**

Implement a small parameter accessor and use it at all existing call sites:

```python
def parameter(self, name):
    return self.parameters[name]
```

Initialize parameters with:

```python
self.parameters = {
    name: float(rospy.get_param("~" + name, default))
    for name, default in defaults.items()
}
if any(not math.isfinite(value) or value <= 0.
       for value in self.parameters.values()):
    raise ValueError("control parameters must be finite and positive")
```

Use `rospy.Time.now().to_sec()` for simulation time, `stamp.to_sec()` for message time, `rospy.Publisher`/`Subscriber`/`Timer` for I/O, `tf2_ros.Buffer.lookup_transform(..., rospy.Duration(.15))` for scans, and ROS1 broadcasters. Keep `time.monotonic()` for receipt freshness.

Delete action-client state (`client`, `goal`, `pending_goal`, callbacks, and cancellation requests). `cancel()` becomes a plan barrier:

```python
def cancel(self):
    self.epoch += 1
    self.accept_planned = False
    self.planned_after_stamp = rospy.Time.now().to_sec()
    self.planned[:] = 0.
    self.plan_time = 0.
```

After publishing each reference, publish `Float32(data=limit)` and set `accept_planned = True` with the new barrier stamp. Preserve every subsequent control branch and final safety check.

- [ ] **Step 4: Run the complete safety package tests**

Run: `rostest smart_wheelchair_safety safety_regression.test`

Expected: all geometry, joystick, web, follower, and unified-controller tests PASS.

- [ ] **Step 5: Commit**

```bash
git add catkin_ws/src/smart_wheelchair_safety
git commit -m "feat: port unified controller to rospy"
```

---

### Task 5: Convert simulation assets and trajectory preview to Gazebo 11

**Files:**
- Modify: `catkin_ws/src/smart_wheelchair_gazebo/CMakeLists.txt`
- Modify: `catkin_ws/src/smart_wheelchair_gazebo/package.xml`
- Rename: `worlds/m6_room.sdf` → `worlds/m6_room.world`
- Rename: `worlds/unified_door.sdf` → `worlds/unified_door.world`
- Modify: `catkin_ws/src/smart_wheelchair_gazebo/models/smart_wheelchair/model.sdf`
- Delete: `catkin_ws/src/smart_wheelchair_gazebo/config/top_down_gui.config`
- Delete: `catkin_ws/src/smart_wheelchair_gazebo/config/unified_control.yaml`
- Create: `catkin_ws/src/smart_wheelchair_gazebo/config/local_path_follower.yaml`
- Rename/modify: `src/trajectory_preview_system.cc` → `src/trajectory_preview_plugin.cc`
- Rename/modify: `launch/sim.launch.py` → `launch/sim.launch`
- Modify: `catkin_ws/src/smart_wheelchair_gazebo/test/test_model_visuals.py`
- Modify: `catkin_ws/src/smart_wheelchair_gazebo/test/test_world_layout.py`

**Interfaces:**
- Gazebo subscribes `/cmd_vel`; publishes `/odom`, scans, image, camera info, and `/clock`.
- Launch accepts `world` and `gui` arguments and starts all three nodes.

- [ ] **Step 1: Rewrite structural expectations first**

Tests must parse SDF 1.6/1.7, expect `gpu_ray` sensors with `libgazebo_ros_gpu_laser.so`, `libgazebo_ros_camera.so`, and `libgazebo_ros_diff_drive.so`, and reject `gz-sim-`, `ros_gz`, `ament`, and `nav2`. Launch tests must parse XML and assert inclusion of `$(find gazebo_ros)/launch/empty_world.launch`, the three safety nodes, `use_sim_time=true`, the selected world, and `gui` forwarding. Geometry, door widths, visual names, range values, and accessibility assertions remain unchanged.

- [ ] **Step 2: Run and verify RED**

Run: `PYTHONPATH=catkin_ws/src/smart_wheelchair_safety python3 -m unittest discover -s catkin_ws/src/smart_wheelchair_gazebo/test -v`

Expected: FAIL on Harmonic plugin names and Python launch/config expectations.

- [ ] **Step 3: Convert worlds, model, config, and launch**

Use Gazebo 11 SDF and remove Harmonic world-system plugins. Add a world GUI camera:

```xml
<gui fullscreen="0">
  <camera name="top_down">
    <pose>-6.7 0 9.0 0 1.5708 0</pose>
    <view_controller>orbit</view_controller>
  </camera>
</gui>
```

Change each lidar to `type="gpu_ray"` with `<ray>` and a ROS GPU laser plugin containing `topicName`, `frameName`, and `robotNamespace`. Add the camera ROS plugin with `cameraName`, `imageTopicName`, `cameraInfoTopicName`, and `frameName`. Replace drive system with:

```xml
<plugin name="differential_drive_controller" filename="libgazebo_ros_diff_drive.so">
  <alwaysOn>true</alwaysOn>
  <updateRate>50</updateRate>
  <leftJoint>left_rear_wheel_joint</leftJoint>
  <rightJoint>right_rear_wheel_joint</rightJoint>
  <wheelSeparation>0.72</wheelSeparation>
  <wheelDiameter>0.36</wheelDiameter>
  <commandTopic>/cmd_vel</commandTopic>
  <odometryTopic>/odom</odometryTopic>
  <odometryFrame>odom</odometryFrame>
  <robotBaseFrame>rear_axle</robotBaseFrame>
  <publishWheelTF>false</publishWheelTF>
  <publishOdomTF>false</publishOdomTF>
  <odometrySource>world</odometrySource>
</plugin>
```

Create an XML `sim.launch` that loads `local_path_follower.yaml`, includes Gazebo's empty-world launch, and starts `local_path_follower_node`, `unified_control_node`, and `web_joystick_node`.

- [ ] **Step 4: Port the preview plugin and catkin target**

Derive `TrajectoryPreviewPlugin` from `gazebo::ModelPlugin`; subscribe with a dedicated ROS callback queue to raw/final Twist topics. Preserve `prediction_seconds=3.0`, `step_seconds=0.3`, timeout 0.5 s, rear axle x -0.33 m, track 0.72 m, raw diameter 0.03 m, final diameter 0.09 m, colors, and constant-curvature equations. Advertise a Gazebo transport publisher on `~/visual`. For each segment, publish a `gazebo::msgs::Visual` named `smart_wheelchair_trajectory_{raw|filtered}_{wheel}_{index}`, parented to the model scoped name, with cylinder radius equal to half the configured diameter, length equal to segment length, pose at the segment midpoint, and orientation aligned to the segment. Set `delete_me=true` on segment visuals that are no longer active.

CMake uses:

```cmake
find_package(catkin REQUIRED COMPONENTS gazebo_ros geometry_msgs roscpp)
find_package(gazebo REQUIRED)
catkin_package(LIBRARIES SmartWheelChairTrajectoryPreview
               CATKIN_DEPENDS gazebo_ros geometry_msgs roscpp)
include_directories(${catkin_INCLUDE_DIRS} ${GAZEBO_INCLUDE_DIRS})
add_library(SmartWheelChairTrajectoryPreview src/trajectory_preview_plugin.cc)
target_compile_features(SmartWheelChairTrajectoryPreview PRIVATE cxx_std_14)
target_link_libraries(SmartWheelChairTrajectoryPreview
  ${catkin_LIBRARIES} ${GAZEBO_LIBRARIES})
```

- [ ] **Step 5: Verify structural and geometry tests**

Run: `PYTHONPATH=catkin_ws/src/smart_wheelchair_safety python3 -m unittest discover -s catkin_ws/src/smart_wheelchair_gazebo/test -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add catkin_ws/src/smart_wheelchair_gazebo
git commit -m "feat: migrate simulation to Gazebo 11"
```

---

### Task 6: Migrate Docker, VNC startup, and the integration probe

**Files:**
- Modify: `docker/Dockerfile`
- Modify: `docker/entrypoint.sh`
- Modify: `docker/gazebo-gui-vnc.sh`
- Modify: `docker-compose.yml`
- Modify: `scripts/probe_unified_control.py`
- Create: `catkin_ws/src/smart_wheelchair_gazebo/test/test_runtime_wiring.py`

**Interfaces:**
- Image tag: `smart-wheelchair-ros:noetic`.
- Shell build: `cd /workspaces/SmartWheelChair/catkin_ws && catkin_make`.
- GUI service still exposes noVNC 6080, VNC 5900, and web joystick 8090.

- [ ] **Step 1: Add failing runtime-wiring assertions**

```python
class RuntimeWiringTest(unittest.TestCase):
    def test_container_uses_noetic_catkin_and_roslaunch(self):
        dockerfile = (ROOT / "docker/Dockerfile").read_text()
        entrypoint = (ROOT / "docker/entrypoint.sh").read_text()
        gui = (ROOT / "docker/gazebo-gui-vnc.sh").read_text()
        self.assertIn("noetic", dockerfile)
        self.assertIn("/opt/ros/noetic/setup.bash", entrypoint)
        self.assertIn("/catkin_ws/devel/setup.bash", entrypoint)
        self.assertIn("catkin_make", gui)
        self.assertIn("roslaunch smart_wheelchair_gazebo sim.launch", gui)
        for forbidden in ("jazzy", "colcon", "ros2 launch", "ros-gz"):
            self.assertNotIn(forbidden, dockerfile + entrypoint + gui)
```

- [ ] **Step 2: Run and verify RED**

Run: `python3 catkin_ws/src/smart_wheelchair_gazebo/test/test_runtime_wiring.py`

Expected: FAIL on Jazzy/colcon/ros2 launch.

- [ ] **Step 3: Convert image and scripts**

Base the image on `ros:noetic-ros-base-focal` (through the existing configurable mirror argument), use Focal apt sources, and install `python3-catkin-tools` or `ros-noetic-catkin`, `ros-noetic-gazebo-ros-pkgs`, `ros-noetic-gazebo-plugins`, `ros-noetic-teleop-twist-keyboard`, `ros-noetic-tf2-ros`, NumPy, and existing VNC packages. Source Noetic/devel in entrypoint.

The GUI script builds with `catkin_make`, sources `devel/setup.bash`, and runs:

```bash
args=(gui:=true)
if [[ -n "${WHEELCHAIR_WORLD:-}" ]]; then
  args+=("world:=${WHEELCHAIR_WORLD}")
fi
exec roslaunch smart_wheelchair_gazebo sim.launch "${args[@]}"
```

Remove the Harmonic `gz service` camera loop because the top-down pose now lives in the world GUI block.

Port the probe from rclpy to rospy while retaining its exact published commands, topic observations, timeouts, status JSON assertions, and exit codes.

- [ ] **Step 4: Run structural tests**

Run: `python3 catkin_ws/src/smart_wheelchair_gazebo/test/test_runtime_wiring.py -v`

Expected: PASS.

- [ ] **Step 5: Build the image and catkin workspace**

Run: `docker compose build sim`

Expected: exit 0.

Run: `docker compose run --rm sim bash -lc 'cd catkin_ws && catkin_make'`

Expected: exit 0 and both packages built.

- [ ] **Step 6: Run all tests inside Noetic**

Run: `docker compose run --rm sim bash -lc 'cd catkin_ws && source devel/setup.bash && catkin_make run_tests && catkin_test_results --verbose'`

Expected: zero failed tests.

- [ ] **Step 7: Commit**

```bash
git add docker docker-compose.yml scripts catkin_ws/src/smart_wheelchair_gazebo/test/test_runtime_wiring.py
git commit -m "build: run simulation in ROS Noetic container"
```

---

### Task 7: Update documentation and perform end-to-end verification

**Files:**
- Modify: `README.md`
- Modify: `docs/architecture-and-algorithms.md`
- Modify: any source/test file found by the final stale-reference scan

**Interfaces:**
- Documents the final ROS1 commands, paths, topics, limitations, and verification evidence.

- [ ] **Step 1: Add a stale-reference documentation test**

Extend `test_ros1_layout.py` with:

```python
def test_operational_docs_have_no_ros2_instructions(self):
    docs = (ROOT / "README.md").read_text() + (
        ROOT / "docs/architecture-and-algorithms.md").read_text()
    for forbidden in ("ros2 ", "ROS 2", "Jazzy", "Nav2", "colcon",
                      "ros_gz", "ros2_ws"):
        self.assertNotIn(forbidden, docs)
    self.assertIn("ROS Noetic", docs)
    self.assertIn("Gazebo 11", docs)
    self.assertIn("catkin_make", docs)
```

- [ ] **Step 2: Run and verify RED**

Run: `python3 catkin_ws/src/smart_wheelchair_gazebo/test/test_ros1_layout.py -v`

Expected: FAIL on current ROS2 operational text.

- [ ] **Step 3: Rewrite operational documentation**

Update build/shell/launch/topic/teleop/test commands, workspace paths, component tables, control-flow prose, planner description, dependencies, source links, and limitations. Describe the bounded trajectory sampler accurately and retain the warning that simulation is not a certified real-chair safety guarantee.

- [ ] **Step 4: Run the complete static suite**

Run: `PYTHONPATH=catkin_ws/src/smart_wheelchair_safety python3 -m unittest discover -s catkin_ws/src/smart_wheelchair_safety/test -v && PYTHONPATH=catkin_ws/src/smart_wheelchair_safety python3 -m unittest discover -s catkin_ws/src/smart_wheelchair_gazebo/test -v`

Expected: PASS; ROS-only tests may skip on the host but must pass in the container command below.

- [ ] **Step 5: Run fresh container verification**

Run: `docker compose build sim && docker compose run --rm sim bash -lc 'cd catkin_ws && catkin_make && source devel/setup.bash && catkin_make run_tests && catkin_test_results --verbose'`

Expected: all commands exit 0 and test results report zero failures.

- [ ] **Step 6: Verify headless runtime and topics**

Run: `docker compose run -d --name smart-wheelchair-verify sim bash -lc 'cd catkin_ws && source devel/setup.bash && roslaunch smart_wheelchair_gazebo sim.launch gui:=false'`

Run: `docker exec smart-wheelchair-verify bash -lc 'source /opt/ros/noetic/setup.bash; source catkin_ws/devel/setup.bash; timeout 30 rostopic list'`

Expected topics include `/cmd_vel`, `/cmd_vel_raw`, `/cmd_vel_planned`, `/odom`, both scans, rear image/camera info, `/shared_control/status`, `/shared_control/reference`, and `/speed_limit`.

Run: `docker exec smart-wheelchair-verify bash -lc 'source /opt/ros/noetic/setup.bash; source catkin_ws/devel/setup.bash; python3 scripts/probe_unified_control.py'`

Expected: exit 0 after manual, assisted, stale-stop, wall, and doorway checks.

- [ ] **Step 7: Stop verification containers and scan the repository**

Run: `docker stop smart-wheelchair-verify && docker rm smart-wheelchair-verify && docker compose down`

Run: `rg -n 'rclpy|ament_|ros_gz|nav2_|ros2 launch|ros2 run|ros2 topic|ros2_ws|Jazzy' --glob '!docs/superpowers/**' --glob '!.vscode/**' .`

Expected: no matches.

Run: `git diff --check && git status --short`

Expected: no whitespace errors; only intended migration files plus untracked `.vscode/`.

- [ ] **Step 8: Commit documentation and final fixes**

```bash
git add README.md docs/architecture-and-algorithms.md catkin_ws docker docker-compose.yml scripts
git commit -m "docs: document ROS1 simulation workflow"
```
