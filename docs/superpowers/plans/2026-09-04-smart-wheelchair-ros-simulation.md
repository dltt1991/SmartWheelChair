# Smart Wheelchair ROS Simulation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Dockerized ROS 2 Jazzy + Gazebo simulation for an M6-style smart wheelchair with differential drive, dual front LiDAR, rear camera, and a basic shared-control safety filter.

**Architecture:** Gazebo owns the physical model and sensors; ROS talks to Gazebo through `ros_gz_bridge`. A small Python package filters requested velocity commands using the nearest front LiDAR obstacle before forwarding safe `cmd_vel` to Gazebo.

**Tech Stack:** Docker Compose, ROS 2 Jazzy, Gazebo Harmonic via `ros_gz`, Python 3, pytest, SDF.

## Global Constraints

- Use ROS inside Docker.
- Keep the first version local-simulation only.
- Do not implement SLAM or destination autonomous navigation.
- Model the teardown-derived platform: rear differential drive, passive front casters, two front single-line LiDARs, rear camera.
- Keep real hardware protocols out of scope until bus captures or device models are available.

---

### Task 1: Project Scaffold

**Files:**
- Create: `README.md`
- Create: `docker/Dockerfile`
- Create: `docker/entrypoint.sh`
- Create: `docker-compose.yml`
- Create: `.gitignore`

**Interfaces:**
- Consumes: none.
- Produces: Docker workspace mounted at `/workspaces/SmartWheelChair` and launch instructions for users.

- [ ] Create Docker and Compose files.
- [ ] Add README with build, test, and launch commands.
- [ ] Validate with `docker compose config`.

### Task 2: Gazebo Wheelchair Simulation

**Files:**
- Create: `ros2_ws/src/smart_wheelchair_gazebo/package.xml`
- Create: `ros2_ws/src/smart_wheelchair_gazebo/CMakeLists.txt`
- Create: `ros2_ws/src/smart_wheelchair_gazebo/launch/sim.launch.py`
- Create: `ros2_ws/src/smart_wheelchair_gazebo/worlds/m6_room.sdf`
- Create: `ros2_ws/src/smart_wheelchair_gazebo/models/smart_wheelchair/model.config`
- Create: `ros2_ws/src/smart_wheelchair_gazebo/models/smart_wheelchair/model.sdf`

**Interfaces:**
- Consumes: Docker ROS/Gazebo environment from Task 1.
- Produces: Gazebo topics bridged to ROS: `/cmd_vel`, `/scan_left`, `/scan_right`, `/camera/rear/image`, `/odom`.

- [ ] Add SDF model with body, rear driven wheels, front caster visuals, dual LiDAR, rear camera, and diff-drive plugin.
- [ ] Add a small room world with a doorway-like obstacle.
- [ ] Add ROS launch file that starts Gazebo and bridge nodes.
- [ ] Validate package metadata with `xmllint` or Python XML parsing.

### Task 3: Safety Filter

**Files:**
- Create: `ros2_ws/src/smart_wheelchair_safety/package.xml`
- Create: `ros2_ws/src/smart_wheelchair_safety/setup.py`
- Create: `ros2_ws/src/smart_wheelchair_safety/setup.cfg`
- Create: `ros2_ws/src/smart_wheelchair_safety/resource/smart_wheelchair_safety`
- Create: `ros2_ws/src/smart_wheelchair_safety/smart_wheelchair_safety/__init__.py`
- Create: `ros2_ws/src/smart_wheelchair_safety/smart_wheelchair_safety/limiter.py`
- Create: `ros2_ws/src/smart_wheelchair_safety/smart_wheelchair_safety/safety_filter_node.py`
- Create: `ros2_ws/src/smart_wheelchair_safety/test/test_limiter.py`

**Interfaces:**
- Consumes: `geometry_msgs/Twist` on `/cmd_vel_raw`, `sensor_msgs/LaserScan` on `/scan_left` and `/scan_right`.
- Produces: filtered `geometry_msgs/Twist` on `/cmd_vel`.
- Pure function: `limit_forward_speed(requested_speed: float, ranges: Sequence[float], stop_distance_m: float, slow_distance_m: float) -> float`.

- [ ] Write failing pytest tests for stop, slowdown, clear path, reverse pass-through, invalid ranges, and invalid thresholds.
- [ ] Run tests and confirm expected failure.
- [ ] Implement the pure limiter.
- [ ] Add ROS node wrapper around the pure limiter.
- [ ] Run tests and confirm pass.

### Task 4: Integration Docs

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: packages from Tasks 2 and 3.
- Produces: exact developer commands.

- [ ] Document build: `docker compose build`.
- [ ] Document shell entry: `docker compose run --rm sim`.
- [ ] Document workspace build: `colcon build --symlink-install`.
- [ ] Document launch: `ros2 launch smart_wheelchair_gazebo sim.launch.py`.
- [ ] Document teleop and safety-filter command flow.

## Self-Review

- Spec coverage: all first-version requirements map to Tasks 1-4.
- Placeholder scan: no deferred implementation placeholders remain in the first-version scope.
- Type consistency: the safety function signature is defined once and reused by the node and tests.
