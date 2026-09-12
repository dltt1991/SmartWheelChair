# Task 3 Report: rospy follower and web joystick

## What I implemented

- Added a thin `LocalPathFollowerNode` rospy adapter.
  - Subscribes to `/shared_control/reference`, `/speed_limit`, `/odom`,
    `/unified_scan_left`, and `/unified_scan_right`.
  - Timestamps every receipt with an injectable monotonic clock.
  - Converts world path poses into the odometry rear-axle frame.
  - Converts finite scan rays with fixed lidar poses `(0.79, +/-0.26)`.
  - Calls the existing pure `select_velocity` planner and publishes a stamped
    `/cmd_vel_planned` message in `rear_axle` at 20 Hz.
  - Publishes zero whenever any required receipt is older than 0.25 seconds.
- Ported `WebJoystickNode` from rclpy to rospy while retaining its page, HTTP
  API, lock boundaries, session/revision checks, neutral interlock, image
  handling, command timeout, and mode publication behavior.
- Added a shutdown hook that publishes a zero raw velocity.
- Added executable wrappers for the follower and web joystick and installed
  only those two wrappers through catkin.
- Added a two-process rostest launch and migrated the web test lifecycle to
  one anonymous rospy node per test module.
- Removed one Python 3.10-only return annotation from `unified_geometry.py`
  after the full Noetic package run exposed it as a Python 3.8 import blocker.

## TDD evidence

### RED: rospy adapters

Command, in a clean ROS Noetic container:

```text
rostest smart_wheelchair_safety web_and_follower.test
```

Relevant result before implementation:

```text
RESULT: FAIL
TESTS: 14
ERRORS: 14
No module named 'smart_wheelchair_safety.local_path_follower_node'
No module named 'rclpy'
```

This was expected: the follower module did not exist and the web node still
depended on ROS2 rclpy.

### GREEN: rospy adapters

```text
rostest smart_wheelchair_safety web_and_follower.test
RESULT: SUCCESS
TESTS: 15
ERRORS: 0
FAILURES: 0
```

### RED: full Noetic package compatibility

The first registered package test run passed the 8 joystick tests and all 15
Task 3 rostests, but the geometry suite failed during import:

```text
TypeError: unsupported operand type(s) for |: 'type' and 'NoneType'
unified_geometry.py:255: def active_aperture_targeted(...) -> bool | None
```

This was a real Python 3.8 incompatibility. The minimal correction was to
remove the optional return annotation without changing function behavior.

### GREEN: affected and full registered suites

Commands, in ROS Noetic/Python 3.8:

```text
python3 -m unittest test/test_unified_geometry.py -v
rostest smart_wheelchair_safety web_and_follower.test
catkin_make run_tests_smart_wheelchair_safety
catkin_test_results catkin_ws/build/test_results
```

Results:

```text
Unified geometry: 47 tests, OK
Task 3 rostest: 15 tests, 0 errors, 0 failures
Registered package summary: 72 tests, 0 errors, 0 failures, 0 skipped
```

The package also configured and built successfully with:

```text
catkin_make --pkg smart_wheelchair_safety -DCMAKE_BUILD_TYPE=Release
```

The build generated devel-space wrappers for exactly
`local_path_follower_node` and `web_joystick_node`.

## Files changed

- `catkin_ws/src/smart_wheelchair_safety/CMakeLists.txt`
- `catkin_ws/src/smart_wheelchair_safety/nodes/local_path_follower_node`
- `catkin_ws/src/smart_wheelchair_safety/nodes/web_joystick_node`
- `catkin_ws/src/smart_wheelchair_safety/smart_wheelchair_safety/local_path_follower_node.py`
- `catkin_ws/src/smart_wheelchair_safety/smart_wheelchair_safety/unified_geometry.py`
- `catkin_ws/src/smart_wheelchair_safety/smart_wheelchair_safety/web_joystick_node.py`
- `catkin_ws/src/smart_wheelchair_safety/test/test_local_path_follower_node.py`
- `catkin_ws/src/smart_wheelchair_safety/test/test_web_joystick.py`
- `catkin_ws/src/smart_wheelchair_safety/test/web_and_follower.test`
- `.superpowers/sdd/task-3-report.md`

## Self-review

- Rechecked every Task 3 interface, rate, timeout, fixed sensor pose, stamped
  frame, wrapper, executable bit, CMake install, and rostest registration.
- Confirmed the existing web assertions remain present and pass; the only new
  web assertion covers shutdown zero publication.
- Kept ROS plumbing in the adapters and reused the existing pure planner.
- Checked Python syntax and `git diff --check`.
- Left the user's untracked `.vscode/` directory untouched.

## Issues or concerns

None remaining for Task 3. The Python 3.8 compatibility blocker found by the
full package run was corrected and all registered package tests now pass.
