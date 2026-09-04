# SmartWheelChair ROS Simulation

Dockerized ROS 2 Jazzy + Gazebo simulation for an M6-style smart wheelchair.

## What Is Modeled

- Rear differential drive with two powered rear wheels.
- Two passive front caster wheels.
- Left and right front 2D LiDAR sensors.
- Rear camera.
- A ROS safety filter that slows or stops forward motion near front obstacles.

This first version intentionally skips real GD32/RK3568 protocols, SLAM, autonomous navigation, and certified safety behavior.

## Build Docker Image

```bash
docker compose build
```

## Open A ROS Shell

```bash
docker compose run --rm sim
```

Inside the shell:

```bash
cd /workspaces/SmartWheelChair/ros2_ws
colcon build --symlink-install
source install/setup.bash
```

## Launch Simulation

Inside the Docker shell:

```bash
ros2 launch smart_wheelchair_gazebo sim.launch.py
```

In another Docker shell, drive through the safety filter:

```bash
cd /workspaces/SmartWheelChair/ros2_ws
source install/setup.bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -r cmd_vel:=cmd_vel_raw
```

Useful topics:

```bash
ros2 topic list
ros2 topic echo /scan_left
ros2 topic echo /scan_right
ros2 topic echo /odom
```

## Run Local Unit Tests

These tests cover the pure speed-limiter logic and do not require ROS:

```bash
PYTHONPATH=ros2_ws/src/smart_wheelchair_safety python3 -m unittest discover ros2_ws/src/smart_wheelchair_safety/test
```

## Notes From Teardown

The model follows the teardown summary:

- Front wheels are passive casters.
- Rear wheels are independently driven and form a differential-drive base.
- Two front single-line LiDARs provide local obstacle input.
- Rear camera is modeled as a video sensor only.
- The simulated shared-control behavior is local obstacle speed limiting, not destination navigation.
