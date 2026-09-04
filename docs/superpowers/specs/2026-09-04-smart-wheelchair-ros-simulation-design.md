# Smart Wheelchair ROS Simulation Design

## Goal

Build a Docker-based ROS 2 simulation environment for the M6-style smart wheelchair described in the teardown notes.

## Hardware Model

The simulator models the confirmed platform shape: two passive front caster wheels, two independently driven rear wheels, automatic rear wheel lock semantics represented as a parking/drive state, two front single-line LiDAR sensors, and a rear camera. The first simulation does not model battery chemistry, motor phase currents, GD32 firmware, RK3568 UI, 4G, Wi-Fi, Bluetooth, GPS, or the unknown auxiliary PCB.

## Software Architecture

Use ROS 2 Jazzy inside Docker and Gazebo Harmonic through `ros_gz`. Gazebo owns physics, collision, wheel joints, the differential-drive plugin, and simulated sensors. ROS owns command input, topic bridging, visualization, and shared-control logic.

The first control chain is:

`teleop / autonomy cmd_vel -> safety_filter_node -> /cmd_vel -> ros_gz_bridge -> Gazebo diff drive`

Sensor feedback is:

`Gazebo left/right LiDAR + rear camera -> ros_gz_bridge -> ROS sensor topics`

## Packages

- `smart_wheelchair_gazebo`: launch files, SDF world, SDF wheelchair model, bridge setup.
- `smart_wheelchair_safety`: pure Python safety limiter plus ROS node wrapper.

## Safety Behavior

The safety filter subscribes to raw velocity commands and the two front LiDAR scans. It scales forward speed down when the closest front obstacle enters a configurable slowdown distance, and commands a stop when the obstacle is inside a configurable stop distance. Reverse and turning commands are not modified in the first version.

Defaults:

- `stop_distance_m`: `0.45`
- `slow_distance_m`: `1.20`
- Front sector is the available LiDAR field because the simulated sensors are mounted facing forward with a `200 deg` horizontal scan.

## Docker

The environment uses `osrf/ros:jazzy-desktop-full` as the base image and installs `ros-jazzy-ros-gz`, `colcon`, `xacro`, `teleop_twist_keyboard`, and common debugging tools. The repository mounts into `/workspaces/SmartWheelChair`.

## Validation

Minimum validation for the first version:

- Python unit tests cover the velocity limiting math.
- Docker Compose config validates.
- The ROS workspace package metadata is present and buildable inside the container.
- README includes exact build and launch commands.

## Out Of Scope

- Real motor control firmware.
- Real GD32/RK3568 protocol emulation.
- Real LiDAR protocol drivers.
- SLAM, route planning, or destination autonomous navigation.
- Production safety certification.
