# Door and Wall Intent Arbitration

## Problem

The unified controller currently treats steering above 0.25 rad/s as both a
reason not to acquire a narrow doorway and a reason to cancel an active door
traversal. During an oblique approach, the driver naturally steers toward the
door, so the doorway branch is excluded and the jamb is subsequently selected
as a wall-following surface.

The controller must distinguish steering toward a doorway from steering away
from an assist mode. Narrow-door assistance should win while the driver's
predicted path targets a traversable doorway, without weakening front-wall
stopping or the independent braking guard.

## Decision

Use geometric intent arbitration based on the existing three-second joystick
arc. For each confirmed narrow opening:

- retain the existing width, range, orientation, and two-observation validity
  checks;
- project the joystick arc into the opening frame;
- treat the doorway as intended when the arc approaches or crosses the door
  plane inside an acceptance corridor around the clear aperture;
- rank valid candidates by trajectory miss distance and forward distance.

The acceptance corridor may be wider than the final collision-free aperture:
it represents approximate driver intent, not permission to pass. The existing
door alignment reference, MPPI footprint costs, entry-clearance check, and
independent braking guard remain responsible for precise alignment and safety.

## Priority and State

Reference selection uses this order:

1. an already committed wide-opening turn;
2. an active narrow-door traversal;
3. a confirmed narrow door targeted by the joystick trajectory;
4. a frontal transverse wall stop;
5. wall following or manual joystick trajectory.

Once a doorway is selected, steering that continues toward its corridor is
allowed and the door target remains latched through short scan occlusions.
The latch is released by stop/reverse, passing the door, or sustained steering
whose predicted trajectory misses the doorway on the away side. A short
debounce prevents scan noise or a single joystick sample from switching
between door and wall modes. Steering above 0.25 rad/s that departs from the
aperture must persist for 0.35 s before cancellation; stop/reverse remains
immediate.

Active-door arc analysis has three outcomes: in-aperture traversal, confirmed
departure on reaching the plane (including turning out before rear clearance),
and inconclusive when the three-second horizon cannot reach the plane or the
axle has already crossed. Confirmed departure starts the debounce regardless
of the alignment reference's initial turn direction.

For inconclusive arcs, steering toward either a meaningful doorway heading
error or a forward center-bearing error retains the target. Both angular
errors use a 0.02 rad deadband; center bearing additionally requires more than
0.01 m lateral offset and a center ahead of the axle. This ignores 1 mm offsets
even close to the plane. The initial staging-path bend is not evidence of
driver intent. Neutral steering and renewed toward intent reset the debounce.

Wall-follow override remains available when no doorway is intended. A door
candidate that is visible but not targeted does not suppress wall following.

## Safety Boundaries

- Door intent never bypasses the minimum-width validation.
- Intent corridor tolerance does not relax the body projection required by
  door_entry_clearance.
- Front-wall stopping remains active when no valid targeted doorway exists.
- The local costmap, full rectangular footprint, remembered jamb obstacles,
  speed limits, watchdogs, and final braking guard are unchanged.
- Stopping or reversing clears uncommitted opening observations as before.

## Verification

Add deterministic tests for:

- left and right oblique approaches with steering magnitude above 0.25 rad/s;
- a door candidate and a wall-follow candidate present simultaneously;
- approximate steering that initially misses the exact centerline but clearly
  targets the aperture;
- steering away from an active doorway long enough to cancel;
- a transient away sample that does not cause mode oscillation;
- a visible but untargeted side door that leaves wall following active;
- a transverse front wall without a valid doorway that still stops.

Run the safety package test suite, Python compilation, and the existing Gazebo
door probes for both approach directions. Inspect mode histories to confirm
that door_align wins before wall during oblique approaches and remains stable
until pass or explicit cancellation.
