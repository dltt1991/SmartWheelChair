# Post-door Steering Handoff

## Problem

After the wheelchair has crossed a narrow doorway, `door_clear` continues to
track the locked straight-through reference until the 2 Hz reference update
observes that the rear body has cleared the door plane. At the maximum door
speed of 0.55 m/s, this can suppress joystick steering for up to 0.5 s and add
about 0.275 m of unwanted straight travel.

## Decision

Keep the existing door phases and safety geometry. During `door_clear`, blend
the angular command from the locked door-path curvature to the driver's raw
angular command according to rear-clearance progress:

- at the door plane (`progress <= 0`), use only the door-path angular command;
- between the plane and the existing 0.29 m rear-clear threshold, linearly
  increase the joystick contribution;
- at or beyond 0.29 m, use the joystick angular command completely, even if
  the next 2 Hz reference update has not yet cleared the door state.

Linear speed remains supplied by the planner or the existing door fallback.
The existing angular acceleration and jerk limits smooth the handoff. The
resulting linear/angular command continues through the independent whole-body
braking check, so blending does not authorize an unsafe rear-corner sweep.

## Code Shape

Add one small door angular-command helper in `unified_control_node.py`. It
computes the existing preview curvature and applies the progress blend only in
`door_clear`; `door_align` and `door_pass` remain unchanged. Use the helper in
both the normal planner path and the planner-timeout door fallback so their
behavior stays consistent.

Do not change timer frequencies, door detection, path generation, clear
distance, footprint dimensions, margins, or speed limits.

## Edge Cases

- Zero joystick steering continues to follow the straight door reference.
- Steering before the axle reaches the door plane does not alter door
  tracking.
- A large turn near a jamb can still be reduced or rejected by the independent
  braking envelope.
- Stop and reverse retain their existing immediate door-cancellation behavior.
- Once the normal reference update clears the door, ordinary manual, wall, and
  opening arbitration resumes unchanged.

## Verification

Add deterministic node tests that prove:

- `door_pass` still ignores post-door handoff logic;
- `door_clear` has no joystick contribution at the plane;
- joystick contribution increases midway through rear clearance;
- the full joystick angular command is available at the 0.29 m threshold;
- planner fallback and normal planner output use the same handoff;
- existing safety and package tests remain green.

Build the ROS package, restart the Gazebo GUI service, and perform a narrow-door
run with a turn command applied immediately after crossing. Record mode,
command, rear-clear progress, extra straight distance, and minimum lidar range.
