# Door and Wall Intent Arbitration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make oblique joystick steering toward a confirmed narrow door select and retain doorway assistance ahead of wall following.

**Architecture:** Add one pure geometry selector that scores the existing three-second joystick arc against confirmed narrow-door apertures. Use that selector in the ROS coordinator both to suppress a stale wall override and to acquire a door, while retaining a short away-intent debounce for cancellation.

**Tech Stack:** Python 3, NumPy, ROS 2 Jazzy/rclpy, unittest, Nav2 MPPI

## Global Constraints

- Preserve the existing 0.92-1.50 m narrow-door validity range and two-observation confirmation.
- Do not relax door_entry_clearance, footprint costs, speed limits, watchdogs, jamb memory, or braking_clear.
- Stop/reverse still cancels immediately; sustained steering away cancels within 0.35 s.
- Use the existing 3.0 s arc_path prediction and add no dependencies.

---

### Task 1: Pure Door Intent Geometry

**Files:**
- Modify: `ros2_ws/src/smart_wheelchair_safety/smart_wheelchair_safety/unified_geometry.py`
- Test: `ros2_ws/src/smart_wheelchair_safety/test/test_unified_geometry.py`

**Interfaces:**
- Consumes: `Opening`, `arc_path(v, w, duration=3., steps=41)`
- Produces: `intended_front_door(openings, v, w, corridor_tolerance=.25) -> Opening | None`

- [ ] **Step 1: Write failing left/right oblique and rejection tests**

```python
def test_oblique_joystick_arc_selects_front_door(self):
    for side in (-1., 1.):
        door = Opening((1.6, side*.45), side*.18, 1., ())
        self.assertEqual(intended_front_door([door], .6, side*.30), door)

def test_arc_turning_away_does_not_select_front_door(self):
    door = Opening((1.6, .45), .18, 1., ())
    self.assertIsNone(intended_front_door([door], .6, -.35))
```

- [ ] **Step 2: Run the focused tests and verify the missing import/function fails**

Run: `python3 -m pytest ros2_ws/src/smart_wheelchair_safety/test/test_unified_geometry.py -k 'oblique_joystick or turning_away' -q`

Expected: FAIL because `intended_front_door` is not defined.

- [ ] **Step 3: Implement trajectory-to-aperture scoring**

```python
def intended_front_door(openings, v, w, corridor_tolerance=.25):
    path = arc_path(max(v, .1), w)[:, :2]
    matches = []
    for opening in openings:
        center = np.asarray(opening.center)
        if not (opening.width <= 1.5 and math.cos(opening.heading) > .65
                and .8 < center[0] < 3.5 and abs(center[1]) < 1.5):
            continue
        normal = np.array([math.cos(opening.heading), math.sin(opening.heading)])
        tangent = np.array([-normal[1], normal[0]])
        relative = path-center
        across, along = relative@normal, relative@tangent
        closest = int(np.argmin(np.abs(across)))
        progress = abs(across[0])-abs(across[closest])
        miss = abs(along[closest])
        if progress >= .25 and miss <= opening.width/2+corridor_tolerance:
            matches.append(((miss, np.linalg.norm(center)), opening))
    return min(matches, key=lambda item: item[0], default=(None, None))[1]
```

- [ ] **Step 4: Run the focused and complete geometry tests**

Run: `python3 -m pytest ros2_ws/src/smart_wheelchair_safety/test/test_unified_geometry.py -q`

Expected: all geometry tests PASS.

- [ ] **Step 5: Commit the geometry selector**

```bash
git add ros2_ws/src/smart_wheelchair_safety/smart_wheelchair_safety/unified_geometry.py ros2_ws/src/smart_wheelchair_safety/test/test_unified_geometry.py
git commit -m "feat: infer narrow-door intent from joystick path"
```

### Task 2: Door-First Coordinator Arbitration

**Files:**
- Modify: `ros2_ws/src/smart_wheelchair_safety/smart_wheelchair_safety/unified_control_node.py`
- Test: `ros2_ws/src/smart_wheelchair_safety/test/test_unified_node.py`

**Interfaces:**
- Consumes: `intended_front_door(openings, v, w)` from Task 1
- Produces: `_intended_door() -> tuple[Opening | None, Opening | None]` and `door_away_since: float`

- [ ] **Step 1: Write failing arbitration and debounce tests**

```python
def test_oblique_door_intent_beats_existing_wall_override(self):
    self.confirm_door(center=(1.6, .45), heading=.18, width=1.)
    self.node.wall_side = -1
    self.send_raw(.6, .30)
    self.node.update_reference()
    self.assertFalse(self.node.override)
    self.assertEqual(self.node.mode, 'door_align')

def test_door_cancels_only_after_sustained_away_steering(self):
    self.node.door = self.front_opening(center=(1.4, .4), heading=.15, width=1.)
    self.node.door_phase = 'door_align'
    self.send_raw(.5, -.5)
    self.assertIsNotNone(self.node.door)
    self.node.door_away_since -= .36
    self.send_raw(.5, -.5)
    self.assertIsNone(self.node.door)
```

- [ ] **Step 2: Run the focused node tests and verify failure**

Run: `python3 -m pytest ros2_ws/src/smart_wheelchair_safety/test/test_unified_node.py -k 'oblique_door_intent or sustained_away' -q`

Expected: FAIL because the old 0.25 rad/s override cancels or excludes the door.

- [ ] **Step 3: Integrate intent into input handling and reference priority**

```python
def _intended_door(self):
    local = [self._local_opening(opening) for opening in self.confirmed_openings]
    selected = intended_front_door(local, *self.raw)
    if selected is None:
        return None, None
    index = local.index(selected)
    return self.confirmed_openings[index], selected
```

In `on_raw`, calculate the targeted door before wall override. Do not set wall
override when the current command targets a confirmed door. For an active door,
start `door_away_since` when the command no longer targets its local aperture,
clear it when intent returns, and cancel only after 0.35 s. Stop/reverse remains
immediate. In `update_reference`, replace the absolute-angular-rate acquisition
gate with `_intended_door()` and keep the door branch ahead of front stop and
wall reference.

- [ ] **Step 4: Update the existing immediate-away test for debounce semantics**

Keep immediate stop/reverse assertions. Assert that one away sample retains
the door and a second sample after 0.35 s clears it and enters override.

- [ ] **Step 5: Run node tests and the complete safety package suite**

Run: `python3 -m pytest ros2_ws/src/smart_wheelchair_safety/test/test_unified_node.py -q`

Run: `python3 -m pytest ros2_ws/src/smart_wheelchair_safety/test -q`

Expected: all tests PASS.

- [ ] **Step 6: Commit coordinator arbitration**

```bash
git add ros2_ws/src/smart_wheelchair_safety/smart_wheelchair_safety/unified_control_node.py ros2_ws/src/smart_wheelchair_safety/test/test_unified_node.py
git commit -m "fix: prioritize intended narrow doors over wall following"
```

### Task 3: Integration Verification

**Files:**
- Modify if observations differ: `docs/unified-control-v1.md`
- Test: `scripts/probe_unified_control.py`

**Interfaces:**
- Consumes: coordinator status `mode` and `door`
- Produces: evidence that both oblique directions enter stable door modes

- [ ] **Step 1: Compile the package Python sources**

Run: `python3 -m compileall -q ros2_ws/src/smart_wheelchair_safety/smart_wheelchair_safety`

Expected: exit code 0.

- [ ] **Step 2: Build the ROS package**

Run: `source /opt/ros/jazzy/setup.bash && colcon build --packages-select smart_wheelchair_safety`

Expected: package finishes successfully.

- [ ] **Step 3: Run both Gazebo oblique door probes**

Run each case from a fresh `unified_door.sdf` simulation:

```bash
python3 scripts/probe_unified_control.py door --lateral-m .20 --yaw-deg 15 --seconds 35
python3 scripts/probe_unified_control.py door --lateral-m -.20 --yaw-deg -15 --seconds 35
```

Expected: each rear axle clears the door; mode history enters `door_align`
before any sustained `wall` mode; sampled hard clearance remains positive.

- [ ] **Step 4: Review the final diff and commit any verification documentation**

Run: `git diff --check && git status --short`

If runtime observations required documentation changes, commit only those
changes with `git commit -m "docs: record door intent arbitration checks"`.
