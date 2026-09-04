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

The Dockerfile defaults to DaoCloud's public Docker Hub mirror for an ARM64 ROS base image and Tsinghua mirrors for Ubuntu/ROS apt packages. `docker-compose.yml` pins `linux/arm64` for Apple Silicon Macs; running the Gazebo GUI through an amd64 ROS desktop image under emulation can leave the VNC desktop black because the Gazebo window never maps correctly.

For Docker Desktop on macOS, add this to `Settings -> Docker Engine` and restart Docker:

```json
{
  "registry-mirrors": [
    "https://docker.m.daocloud.io"
  ]
}
```

The same JSON is also saved in `docker/daemon-cn-mirror.json`.

```bash
docker compose build
```

To bypass the mirror:

```bash
docker compose build --build-arg BASE_IMAGE=ros:jazzy-ros-base
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

The launch file starts Gazebo in headless server mode by default, which works inside Docker without display forwarding.

To show the Gazebo window on macOS:

Recommended path:

```bash
cd /Users/guotao/Work/code/SmartWheelChair
docker compose up -d gui
```

Then open:

```text
http://localhost:6080/vnc.html
```

Click `Connect`. The VNC path runs Gazebo inside a container desktop with software OpenGL, avoiding XQuartz GLX issues. If the browser tab was already open from an older failed run, reconnect or hard refresh the page.

`docker compose up gui` starts `ros2 launch smart_wheelchair_gazebo sim.launch.py gui:=true` through `docker/gazebo-gui-vnc.sh`. `docker compose run --rm sim` only opens a ROS shell unless you launch Gazebo manually.

XQuartz path:

1. Install and open XQuartz.
2. In XQuartz, enable `Settings -> Security -> Allow connections from network clients`.
3. Restart XQuartz.
4. Run:

```bash
xhost + 127.0.0.1
docker compose run --rm sim
```

Keep the host terminal `DISPLAY=:0` only for running `xhost`. Docker uses `X11_DISPLAY`, defaulting to `host.docker.internal:0`, so Gazebo connects back to XQuartz instead of looking for a display inside the container.

Inside the Docker shell:

```bash
cd /workspaces/SmartWheelChair/ros2_ws
source install/setup.bash
ros2 launch smart_wheelchair_gazebo sim.launch.py gui:=true
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

These tests cover the pure speed-limiter logic and do not require ROS. The limiter fails safe for forward motion when no fresh, valid LiDAR ranges are available:

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
