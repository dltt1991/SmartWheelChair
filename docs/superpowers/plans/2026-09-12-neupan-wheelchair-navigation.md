# NeuPAN Wheelchair Navigation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a ROS1 NeuPAN adapter for the wheelchair while preserving the existing safety controller and graceful fallback behavior.

**Architecture:** A focused adapter node converts dual lidar scans and the existing local reference into NeuPAN inputs, then publishes action and predicted path topics. The unified controller optionally consumes fresh actions and always applies its existing limits and collision guard.

**Tech Stack:** ROS Noetic, rospy, NumPy, geometry_msgs, nav_msgs, sensor_msgs, Python 3.8-compatible optional NeuPAN import.

## Global Constraints

- Keep lidar fields of view at left `-30°～170°` and right `-170°～30°`.
- NeuPAN is optional; missing package/checkpoint must not prevent safe startup.
- Do not commit model weights.
- All external actions pass existing independent braking and acceleration checks.

### Task 1: Add pure NeuPAN input/output adapter

**Files:**
- Create: `catkin_ws/src/smart_wheelchair_safety/smart_wheelchair_safety/neupan_adapter.py`
- Test: `catkin_ws/src/smart_wheelchair_safety/test/test_neupan_adapter.py`

- [ ] Write tests for finite point filtering, dual-scan merge, path normalization, action clipping, and stale action fallback.
- [ ] Run `PYTHONPATH=catkin_ws/src/smart_wheelchair_safety python3 -m unittest -q ...test_neupan_adapter` and verify failure before implementation.
- [ ] Implement dependency-optional `NeuPANAdapter` with `set_obstacles(points)`, `set_initial_path(path)`, `step(state, now)`, and `available`/`reason` properties. If NeuPAN is unavailable or raises during inference, return zero action and an empty trajectory with a reason.
- [ ] Re-run the focused tests and commit `feat: add optional NeuPAN adapter`.

### Task 2: Add ROS wheelchair NeuPAN node and configuration

**Files:**
- Create: `catkin_ws/src/smart_wheelchair_safety/smart_wheelchair_safety/neupan_wheelchair_node.py`
- Create: `catkin_ws/src/smart_wheelchair_safety/nodes/neupan_wheelchair_node`
- Create: `catkin_ws/src/smart_wheelchair_safety/config/neupan_wheelchair.yaml`
- Create: `catkin_ws/src/smart_wheelchair_safety/launch/neupan_wheelchair.launch`
- Modify: `catkin_ws/src/smart_wheelchair_safety/CMakeLists.txt`
- Modify: `catkin_ws/src/smart_wheelchair_safety/package.xml`
- Test: `catkin_ws/src/smart_wheelchair_safety/test/test_neupan_wheelchair_node.py`

- [ ] Test scan conversion in both configured fields of view, reference-path conversion, and publication of zero action when odometry or scan is stale.
- [ ] Implement the node at 10 Hz with topics `/scan_left`, `/scan_right`, `/odom`, `/shared_control/reference`, `/neupan/cmd_vel`, `/neupan/plan`, and `/neupan/status`.
- [ ] Add YAML values for the wheelchair footprint, differential kinematics, 0.8 m/s and 0.65 rad/s assisted limits, 5 m scan range, and 0.25 s action timeout.
- [ ] Install the executable and launch file; add required ROS dependencies.
- [ ] Run focused tests and `python3 -m py_compile` on new modules; commit `feat: add ROS NeuPAN wheelchair node`.

### Task 3: Integrate optional action into unified safety controller

**Files:**
- Modify: `catkin_ws/src/smart_wheelchair_safety/smart_wheelchair_safety/unified_control_node.py`
- Modify: `catkin_ws/src/smart_wheelchair_safety/config` or launch wiring as needed
- Test: `catkin_ws/src/smart_wheelchair_safety/test/test_unified_node.py`

- [ ] Add tests proving a fresh NeuPAN action is accepted, an expired action is rejected, and braking still overrides a clear-looking action.
- [ ] Add `~use_neupan` and `~neupan_action_timeout` parameters, subscribe to `neupan/cmd_vel`, and select the action only when fresh and assist is enabled.
- [ ] Route the selected action through the existing acceleration, speed, and collision checks without changing manual-direct behavior.
- [ ] Run the full existing unittest command plus the new tests; commit `feat: gate NeuPAN actions through safety controller`.

### Task 4: Add documentation and end-to-end checks

**Files:**
- Modify: `README.md`
- Create: `docs/neupan-wheelchair.md`
- Create: `catkin_ws/src/smart_wheelchair_gazebo/test/check_neupan_adapter.py`

- [ ] Document installation of the Python 3.8-compatible NeuPAN branch, DUNE training inputs, launch usage, topics, fallback, and tuning.
- [ ] Add a deterministic scenario check for straight corridor, wall following, door approach, two-door continuity, and stale-input stop using the adapter interface.
- [ ] Run all safety unit tests, the deterministic check, `git diff --check`, and compile checks; commit `docs: document NeuPAN wheelchair integration`.

### Task 5: Verify and push branch

- [ ] Run the complete validation suite and record results.
- [ ] Inspect `git status`, confirm `.vscode/` remains untracked, and confirm no model weights are staged.
- [ ] Push `feat/neupan-wheelchair-navigation` to `origin` and verify with `git ls-remote`.
