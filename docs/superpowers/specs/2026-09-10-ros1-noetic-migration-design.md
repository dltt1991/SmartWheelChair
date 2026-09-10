# ROS1 Noetic Migration Design

## Goal

Migrate the complete SmartWheelChair simulation from ROS 2 Jazzy and Gazebo Harmonic to ROS Noetic, Ubuntu 20.04, and Gazebo 11 while preserving user-visible functions, topic semantics, simulation scenes, control modes, and safety behavior.

## Scope and constraints

- Rename `ros2_ws` to the conventional ROS1 workspace name `catkin_ws`.
- Use ROS Noetic on Ubuntu 20.04 with Gazebo 11 inside Docker.
- Preserve the browser joystick, rear-camera display, manual/assist toggle, wall following, doorway assistance, side-opening turns, comfort limiting, stale-data stops, independent braking checks, and trajectory previews.
- Preserve public topic names and message semantics wherever ROS1 provides an equivalent message.
- Replace the ROS2-only Nav2 MPPI dependency with a small ROS1-native local path follower; preserving MPPI itself is out of scope.
- Do not introduce the ROS1 navigation stack, a global map, SLAM, hardware protocols, or autonomous navigation.
- Preserve the user's untracked `.vscode/` directory and exclude it from migration commits.

## Repository and build layout

The workspace becomes `catkin_ws/src` with the existing packages:

- `smart_wheelchair_safety`: Python control, geometry, browser joystick, and local path follower.
- `smart_wheelchair_gazebo`: Gazebo models, worlds, launch/config assets, tests, and the trajectory preview plugin.

Both packages use catkin. Python modules remain importable through `catkin_python_setup()`; ROS executables are thin scripts installed with `catkin_install_python()`. The Gazebo package builds its C++ plugin against Gazebo 11, `gazebo_ros`, and `roscpp`.

Docker uses a ROS Noetic Ubuntu 20.04 base image and installs catkin, NumPy, `gazebo_ros_pkgs`, teleoperation, TF2, and the existing VNC desktop dependencies. Entry scripts source `/opt/ros/noetic/setup.bash` and `catkin_ws/devel/setup.bash`. Builds use `catkin_make`; launches use `roslaunch`.

## ROS node migration

### Browser joystick

`web_joystick_node` moves from `rclpy` to `rospy` without changing the HTTP API, page behavior, port 8090, speed mapping, command timeout, camera encoding, or mode-toggle semantics. It continues publishing `/cmd_vel_raw` and `/assist_enabled` and consuming `/camera/rear/image`.

### Unified controller

`unified_control_node` keeps its geometry, state machines, intent arbitration, doorway planning, opening memory, velocity shaping, and final braking envelope. ROS integration changes as follows:

- ROS2 parameters become private rospy parameters with the same names and defaults.
- ROS2 publishers/subscriptions/timers become rospy equivalents.
- ROS time and message stamps use `rospy.Time`; process freshness still uses monotonic time.
- TF2 listener and broadcasters use the ROS1 Python TF2 interfaces.
- The ROS2 FollowPath action and lifecycle state are removed.
- The controller publishes its selected path to `/shared_control/reference` and an absolute linear limit to `/speed_limit`.
- The local path follower publishes `/cmd_vel_planned`; the controller remains the sole publisher of final `/cmd_vel`.

`/speed_limit` changes from the unavailable `nav2_msgs/SpeedLimit` type to `std_msgs/Float32`. Other public topics retain their names and ROS1-equivalent message types.

### Local path follower

A focused local follower replaces Nav2 MPPI. It consumes the reference path, speed limit, odometry, and filtered left/right scans. It samples a bounded set of differential-drive `(linear, angular)` commands, predicts each over a short horizon, and ranks valid candidates using:

- distance and heading error to the reference path;
- forward-progress preference;
- adherence to the current absolute speed limit;
- collision rejection against the wheelchair footprint and observed obstacles.

It publishes the best command as a stamped twist on `/cmd_vel_planned`. Missing/stale inputs, an empty reference, invalid numeric values, or no collision-free candidate produce a zero command. The algorithm is a pure Python unit beneath a thin rospy node so its behavior can be tested without a ROS master.

## Gazebo 11 migration

Gazebo Harmonic system plugins and the `ros_gz_bridge` process are removed. The model uses standard Gazebo 11 ROS plugins directly:

- `libgazebo_ros_diff_drive.so` subscribes to `/cmd_vel` and publishes `/odom`.
- Gazebo ROS laser plugins publish `/scan_left` and `/scan_right` with the existing fields of view, ranges, rates, and frames.
- `libgazebo_ros_camera.so` publishes rear image and camera-info topics.

World files are converted to Gazebo 11-compatible SDF while preserving geometry, materials, spawn poses, physics intent, and both the M6 room and one-metre-door scenarios. ROS1 launch XML includes `gazebo_ros/empty_world.launch`, loads the selected world, starts both Python nodes, and supports the existing headless/GUI choice.

The trajectory preview becomes a Gazebo 11 ModelPlugin using ROS subscribers for raw and final twist commands and Gazebo rendering/transport primitives for the same two three-line constant-curvature previews. Colors, horizons, wheel separation, visibility, and command timeout remain unchanged.

## Data flow and safety behavior

1. Gazebo publishes odometry, two laser scans, and the rear camera stream directly to ROS1.
2. The web node publishes joystick intent and the assist-mode state.
3. The unified controller validates timestamps and values, selects a control mode, and publishes a local reference and speed limit.
4. The local follower converts the reference and obstacle data into `/cmd_vel_planned`.
5. The unified controller rejects stale plans, applies joystick bounds, door/opening corrections, comfort acceleration and jerk limits, and the independent braking envelope before publishing `/cmd_vel`.
6. Gazebo applies `/cmd_vel`; the web page and trajectory plugin display feedback.

Joystick, odometry, either scan, or planned-command timeout retains the current stop behavior. Manual-direct mode continues bypassing assistance after the neutral transition guard. The final braking check remains independent of the local follower.

## Tests and acceptance

The migration retains and updates the existing 191 unit and structural tests. ROS-dependent node tests move from `rclpy` to `rospy`; tests that do not require ROS remain runnable on the host.

New pure Python tests cover local following of a straight path, curved path steering, absolute speed limiting, obstacle rejection, malformed/stale inputs, and stopping when no candidate is safe. Model and launch tests assert the Gazebo 11 plugins, ROS1 topic configuration, catkin metadata, and absence of ROS2/Nav2 dependencies.

Acceptance requires fresh evidence for all of the following:

- Docker image builds for the configured platform.
- `catkin_make` completes in the container.
- Both package test suites pass, including ROS-dependent tests.
- A headless `roslaunch smart_wheelchair_gazebo sim.launch` starts without fatal errors.
- Core topics are present and publish valid data.
- The integration probe verifies manual mode, assist mode, stale-input stop, and the wall/door control chain.
- README and architecture documentation contain ROS1 commands, paths, components, and retest instructions, with no stale operational ROS2 guidance.

## Branch and delivery

All work is performed on `feature/ros1-noetic-migration`. The design is committed separately before the implementation plan and code changes so the migration remains reviewable.
