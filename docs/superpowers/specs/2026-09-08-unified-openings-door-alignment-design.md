# Unified Openings and Door Alignment Design

## Goal

Extend the existing unified shared-control chain so that a driver can turn into
a same-side corridor opening without fighting wall following, and so that the
wheelchair actively aligns to a narrow doorway before passing through it. Keep
the existing MPPI planner, comfort limits, full-body checks, and independent
braking guard.

## Scope and Constraints

- Reuse the existing line fitting, local reference, MPPI, and braking pipeline.
- Keep the Gazebo lidar maximum range at 5.0 m. The later hardware value of
  12 m must remain a sensor/configuration concern, not a hard-coded assumption
  in opening geometry.
- Keep the rear-axle footprint `x=[-0.25, 0.97] m`, `y=[-0.40, 0.40] m` and the
  0.04 m hard guard margin unchanged.
- Keep a frontal wall without a valid opening on the existing controlled-stop
  path.
- Do not add autonomous reverse recovery. A layout that cannot be aligned to
  safely results in a stop.

## Unified Geometry Model

Both features consume one generalized opening extractor built on the existing
fitted `Segment` values. An opening is represented by its center, wall tangent,
crossing normal, measured width, and the two supporting jamb segments.

The extractor pairs approximately collinear, coplanar segments and measures the
gap between their nearest endpoints. A candidate must:

- have two observed jambs; a lone segment endpoint is not sufficient;
- be at least 0.92 m wide, covering the 0.80 m body width, two 0.04 m hard
  margins, and the existing fitting tolerance;
- be no wider than 1.50 m when classified as a narrow door, or 3.20 m when
  classified as a corridor opening;
- lie within the current 5 m scan support and in front of the rear axle;
- remain geometrically associated for two consecutive 10 Hz scan updates.

Association accepts center movement up to 0.25 m, heading movement up to
0.12 rad, and width movement up to 0.25 m between updates. These tolerances
filter scan flicker without introducing a long turn delay.

The existing frontal door selector becomes a filtered view of this generalized
opening set. The same-side turn selector filters the same set by the currently
followed wall and driver intent. No second wall/door fitting implementation is
introduced.

## Same-Side Corridor Turns

While following a wall, the controller evaluates the driver's existing 3 s
constant-twist intent arc. A same-side opening is selected only when:

- its jambs belong to the currently followed wall side;
- its mouth lies ahead of the rear axle;
- the commanded angular velocity points toward that side by at least
  0.12 rad/s; and
- the intent arc crosses the wall band through the measured opening rather
  than through either jamb.

On selection, the coordinator enters `opening_turn`. It follows the driver's
intent arc and suppresses reacquisition of the old wall. This state exits when
the opening is behind the rear axle, the heading has changed by at least
0.45 rad, the driver steers away from the opening, the driver releases or
reverses, or 3 s elapse. Normal wall/manual selection then resumes using the
new pose and visible geometry.

With no toward-opening command, the existing wall reference continues through
the opening without turning automatically. A command away from the wall keeps
the existing immediate driver-override behavior. All commands still pass
through MPPI, comfort limiting, and the independent braking guard.

## Narrow-Door Alignment

A frontal opening between 0.92 m and 1.50 m can engage door assistance when the
driver commands forward motion. The observed center and heading are filtered
while approaching. The door is then managed in three states:

1. `door_align`: place a staging point 0.9-1.8 m before the door plane. Increase
   the setback with lateral and heading error. Generate a tangent-continuous
   path from the current heading to this point, ending before the reserved door
   plane. Do not publish a through-door reference in this phase.
2. `door_pass`: enter only when the available lateral tolerance and projected
   body width show that the body plus hard margins fit and the staging point has
   been reached. For a 1.0 m doorway,
   this requires approximately 0.06 m or less center error and 0.08 rad or less
   heading error at the plane. Freeze the accepted centerline and continue at
   the current 0.45 m/s door speed limit so disappearing jamb returns cannot
   move the path.
3. `door_clear`: release assistance only after the rear axle and the 0.25 m
   rear body extent, including the hard margin, have crossed the door plane.

Before commitment, updated jamb observations refine the door with bounded
low-pass filtering. After commitment, observations may add conservative guard
points but cannot shift the centerline. If alignment cannot meet the geometric
entry condition before the front body reaches the reserved door plane, the
reference stops before the jambs; the guard remains the final authority.

The forward joystick magnitude remains the speed intent. Large alignment error
temporarily limits speed to 0.25 m/s; the existing 0.45 m/s door limit resumes
after alignment. Releasing the joystick, commanding reverse, or clearly
steering away cancels assistance immediately. The controller never reduces the
hard margin to complete a passage.

## State and Data Flow

The geometry module remains ROS-independent and produces opening candidates,
intent-intersection results, and reference paths. The coordinator owns temporal
association and the `opening_turn`, `door_align`, `door_pass`, and `door_clear`
states because it already owns odometry-frame persistence and driver override.

Data continues through:

`lidars -> fitted lines -> openings -> intent/state selector -> local reference
-> MPPI -> comfort limiter -> braking guard -> cmd_vel`

Stale joystick, odometry, scan, or planner data retains the current stop
behavior. State is cleared on stale input, user stop, reverse, explicit cancel,
or completion so an old wall/opening/door cannot capture a later command.

## Verification

ROS-independent tests cover:

- generalized detection of 0.92-1.50 m doors and up to 3.20 m corridor gaps;
- rejection of lone endpoints, narrow gaps, inconsistent jambs, and candidates
  outside the active 5 m geometry;
- same-side intent-arc intersection on left and right openings;
- straight intent retaining wall behavior and toward-opening intent releasing
  it;
- tangent-continuous alignment paths for positive and negative lateral/heading
  errors;
- full-body plus 0.04 m margin containment through 1.0 m doors;
- frozen centerline during pass and rear-body clearance before release;
- stop/reverse/away steering cancellation and stale-data clearing.

Gazebo smoke tests use a fresh simulation per scenario and cover:

- left-wall and right-wall approaches followed by a commanded turn into a side
  corridor, with no old-wall reacquisition;
- straight travel past the same opening without an autonomous turn;
- at least four 1.0 m door approaches spanning left/right lateral error and
  positive/negative heading error;
- frontal-wall stopping with no valid opening;
- minimum sampled clearance, traversal time, speed, and mode transitions.

The complete safety, map, trajectory, and build suites run afterward. Success
requires collision-free traversal, immediate response to a valid turn intent,
continuous door references, and unchanged safety margins. Geometry tests for
all map doors remain separate from closed-loop claims; only scenarios actually
run in Gazebo are reported as closed-loop passes.
