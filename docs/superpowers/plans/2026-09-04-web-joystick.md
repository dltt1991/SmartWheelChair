# Web Joystick Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a browser-based virtual joystick that publishes safe raw velocity commands for the smart wheelchair simulation.

**Architecture:** Reuse the existing `smart_wheelchair_safety` Python package. A new ROS2 node serves a small HTML joystick page and an HTTP command endpoint, converts normalized drag input into `Twist`, and publishes `/cmd_vel_raw`; the existing safety filter continues publishing `/cmd_vel`.

**Tech Stack:** ROS2 Jazzy, `rclpy`, `geometry_msgs`, Python `http.server`, vanilla HTML/CSS/JavaScript.

## Global Constraints

- Keep the control path as `/cmd_vel_raw -> safety_filter -> /cmd_vel`.
- Use `docker compose up -d gui` as the normal startup path.
- Expose the joystick UI at `http://localhost:8090`.
- Stop automatically when the joystick is released or command input becomes stale.

---

### Task 1: Joystick Command Mapping

**Files:**
- Create: `ros2_ws/src/smart_wheelchair_safety/smart_wheelchair_safety/joystick.py`
- Test: `ros2_ws/src/smart_wheelchair_safety/test/test_joystick.py`

**Interfaces:**
- Produces: `joystick_to_velocity(x: float, y: float, max_linear: float, max_angular: float, deadzone: float = 0.08) -> tuple[float, float]`

- [ ] **Step 1: Write the failing tests**

```python
import unittest

from smart_wheelchair_safety.joystick import joystick_to_velocity


class JoystickMappingTest(unittest.TestCase):
    def test_maps_forward_and_right_turn(self):
        linear, angular = joystick_to_velocity(0.5, 1.0, 0.8, 1.4)
        self.assertAlmostEqual(linear, 0.8)
        self.assertAlmostEqual(angular, -0.7)

    def test_clamps_input_and_applies_deadzone(self):
        self.assertEqual(joystick_to_velocity(0.01, 0.01, 0.8, 1.4), (0.0, 0.0))
        self.assertEqual(joystick_to_velocity(2.0, -2.0, 0.8, 1.4), (-0.8, -1.4))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=ros2_ws/src/smart_wheelchair_safety python3 -m unittest ros2_ws/src/smart_wheelchair_safety/test/test_joystick.py`

- [ ] **Step 3: Implement the mapping**

Clamp `x` and `y` to `[-1.0, 1.0]`, apply radial deadzone, return `(linear_x, angular_z)` where `linear_x = y * max_linear` and `angular_z = -x * max_angular`.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=ros2_ws/src/smart_wheelchair_safety python3 -m unittest discover ros2_ws/src/smart_wheelchair_safety/test`

### Task 2: Web Joystick ROS Node

**Files:**
- Create: `ros2_ws/src/smart_wheelchair_safety/smart_wheelchair_safety/web_joystick_node.py`
- Modify: `ros2_ws/src/smart_wheelchair_safety/setup.py`
- Modify: `ros2_ws/src/smart_wheelchair_safety/package.xml`
- Modify: `ros2_ws/src/smart_wheelchair_gazebo/launch/sim.launch.py`
- Modify: `docker/Dockerfile`
- Modify: `docker-compose.yml`
- Modify: `README.md`

**Interfaces:**
- Consumes: `joystick_to_velocity(...)`
- Produces: ROS executable `web_joystick_node`, HTTP UI and command endpoint on port `8090`.

- [ ] **Step 1: Add node and dependencies**

Serve one HTML page and accept `POST /cmd` JSON input `{"x": number, "y": number}`.

- [ ] **Step 2: Launch it with the simulator**

Add the node to `sim.launch.py` so both headless and GUI launches provide the joystick service.

- [ ] **Step 3: Verify**

Run unit tests, relaunch GUI, confirm `http://localhost:8090` is served and `/cmd_vel_raw` receives messages while dragging or through `curl`.
