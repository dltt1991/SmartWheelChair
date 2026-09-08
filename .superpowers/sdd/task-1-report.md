# Task 1 Report: Pure Door Intent Geometry

## Result

Implemented `intended_front_door(openings, v, w, corridor_tolerance=.25)` using the joystick arc trajectory and aperture scoring specified in the task brief.

## TDD Red

Added the required tests:

- `test_oblique_joystick_arc_selects_front_door`
- `test_arc_turning_away_does_not_select_front_door`

Host focused command:

```text
$ python3 -m pytest ros2_ws/src/smart_wheelchair_safety/test/test_unified_geometry.py -k 'oblique_joystick or turning_away' -q
exit_code=1
/opt/homebrew/opt/python@3.14/bin/python3.14: No module named pytest
```

The same focused command in the running ROS container reached test collection and failed for the expected missing import/function:

```text
$ docker compose exec -T gui python3 -m pytest ros2_ws/src/smart_wheelchair_safety/test/test_unified_geometry.py -k 'oblique_joystick or turning_away' -q
exit_code=2

ERROR collecting test/test_unified_geometry.py
ImportError: cannot import name 'intended_front_door' from 'smart_wheelchair_safety.unified_geometry'
!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
```

## TDD Green

Focused geometry tests in the running ROS container:

```text
$ docker compose exec -T gui python3 -m pytest ros2_ws/src/smart_wheelchair_safety/test/test_unified_geometry.py -k 'oblique_joystick or turning_away' -q
..                                                                       [100%]
2 passed, 22 deselected in 0.11s
```

Complete geometry tests in the running ROS container:

```text
$ docker compose exec -T gui python3 -m pytest ros2_ws/src/smart_wheelchair_safety/test/test_unified_geometry.py -q
........................                                                 [100%]
24 passed in 0.15s
```

## Self-review

- `git diff --check` passed.
- The selector uses the specified `arc_path`, front-door filters, trajectory progress, lateral miss, corridor tolerance, and `(miss, distance)` ordering.
- No controller node or unrelated file was modified.

## Changed files

- `ros2_ws/src/smart_wheelchair_safety/smart_wheelchair_safety/unified_geometry.py`
- `ros2_ws/src/smart_wheelchair_safety/test/test_unified_geometry.py`
- `.superpowers/sdd/task-1-report.md`
