# Responsive Planning and Stable Wall-Side Selection Design

## Goal

Reduce perceived start and steering latency while preserving the existing MPPI,
wall-following, narrow-door, opening-turn, and independent braking behavior. At a
cross intersection, left steering must select the left wall and right steering
must select the right wall; straight travel must retain the wall side used when
entering the intersection.

## Current Causes

- `update_reference()` runs every 500 ms. A new joystick command can therefore
  wait almost half a second before MPPI receives a matching path.
- MPPI already applies linear and angular acceleration limits, then the unified
  node applies a second jerk-limited acceleration ramp. The current jerk limits
  take another 400-500 ms to reach the configured acceleration limits.
- `wall_reference()` chooses a wall from the current scan only. It has no stable
  side preference for straight travel, and its weak turn-side filter can admit
  the opposite side at small steering inputs.
- The persistent wall escape override is derived from the previous wall side.
  At an intersection it can suppress reacquisition of the new, explicitly
  requested turn side.

## Design

### Reference response

Run reference generation at 5 Hz instead of 2 Hz, reducing worst-case joystick
to-reference delay from 500 ms to 200 ms. Keep MPPI at 20 Hz and retain its
three-second prediction horizon. This is intentionally a timer adjustment, not
a second local controller or a new planning thread.

Keep the existing acceleration limits of `0.5 m/s^2` linear and `0.8 rad/s^2`
angular. Increase the unified output jerk limits from `1.0` to `2.5 m/s^3` and
from `2.0` to `4.0 rad/s^3`. The output remains continuous, reaches its allowed
acceleration sooner, and all immediate stop and independent braking paths keep
their current priority.

### Wall-side intent

`wall_reference()` will accept a preferred side and apply these rules in order:

1. With `|w| >= 0.12 rad/s`, joystick intent is authoritative: positive angular
   velocity considers only left-wall candidates and negative angular velocity
   considers only right-wall candidates.
2. With weaker steering, a valid remembered side is considered exclusively.
3. With neither explicit turn intent nor remembered side, the existing predicted
   clearance score chooses the initial wall.
4. If the requested or remembered side has no candidate in the current frame,
   return the joystick arc in manual mode instead of attaching to the opposite
   wall.

The node remembers the last successfully selected wall side across short scan
dropouts and intersection openings. Seeing that same side refreshes the memory.
Stopping, reversing, explicit opposite steering, narrow-door capture, or opening
turn capture may replace or clear it. The memory is selection state only: it
must not itself create a wall-follow command when no matching wall is visible.

The wall escape override remains available so steering away from a nearby wall
responds immediately. A confirmed wide opening in the newly commanded direction
releases stale override state, allowing the opening-turn logic to take ownership
and preventing the old wall side from blocking a cross-intersection turn.

### Safety and fallback

No footprint dimensions, clearances, obstacle ranges, speed caps, MPPI collision
critics, stale-input stops, or `braking_clear()` behavior change. If MPPI output
is stale, normal modes still stop as they do today. If wall-side evidence is
ambiguous, the system follows the guarded joystick reference rather than making
an opposite-side wall-follow decision.

## Implementation Scope

- `unified_geometry.py`: add preferred-side filtering to `wall_reference()`.
- `unified_control_node.py`: run reference updates at 5 Hz, maintain wall-side
  selection memory, release stale escape override for an explicitly targeted
  side opening, and use the higher jerk limits.
- `test_unified_geometry.py`: cover explicit left/right selection, straight-side
  retention, and absence of fallback to the opposite wall.
- `test_unified_node.py`: cover timer period, wall-side memory lifecycle,
  intersection override release, and faster but still bounded command ramps.
- Update the unified-control documentation with the final timing and selection
  rules.

No new package, node, topic, parameter, or controller is introduced.

## Acceptance Criteria

- A newly commanded reference is generated within 200 ms under normal executor
  scheduling.
- From rest, output acceleration remains at or below `0.5 m/s^2` and angular
  acceleration remains at or below `0.8 rad/s^2`, while reaching those limits
  within 200 ms.
- At a cross intersection, positive steering cannot select a right wall and
  negative steering cannot select a left wall.
- Straight steering retains the entry wall side through temporary visibility of
  a closer opposite wall.
- Missing evidence for the intended side produces manual guarded travel, not
  opposite-side wall following.
- Existing wall following, opening turns, door traversal, reverse motion,
  emergency stop, stale-input handling, and braking-envelope tests remain green.
