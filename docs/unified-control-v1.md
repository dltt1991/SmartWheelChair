# Unified shared control, version 1

Baseline: `a1b7745`, pushed before this implementation.

## Implemented chain

1. Tested, ROS-independent geometry for segmented wall fitting, doorway
   references and whole-footprint braking checks. Reuse NumPy and Nav2 MPPI;
   do not implement another stochastic optimizer.
2. A ROS coordinator for joystick intent, timestamped laser/odometry
   transforms, reference paths and the final independent command watchdog.
   Use the rear axle as the differential-drive control frame.
3. Nav2 controller server / MPPI and a rolling local costmap. Keep
   the old chain behind `unified_control:=false`. New command chain:
   joystick -> reference path -> MPPI -> comfort limiter -> braking guard -> cmd_vel.
4. An isolated one-metre doorway test world and automated closed-loop
   Gazebo checks for wall approaches, driver override, frontal stopping and
   doorway traversal. Docker includes the Nav2 dependencies.

The original wall follower and safety filter are unchanged and only run when
the unified chain is disabled; the chains never both publish final commands.
The new coordinator is intentionally not a global navigator. It produces a
short reference from the joystick and observed geometry; Nav2 MPPI chooses
the motion, and the independent guard can reject even a planner output.

## Parameters and frames

- Planner: 20 Hz simulation time, 60 x 0.05 s horizon, 600 samples, one iteration.
- Local map: 8 x 8 m, 0.025 m cells, 10 Hz, full rectangular footprint checks.
- Forward/reverse/angular command bounds: 0.8 m/s, 0.4 m/s, 0.65 rad/s.
- Door alignment/pass speed limits: 0.25/0.45 m/s. Wide-opening turns are
  limited to 0.5 m/s while entering. No autonomous reverse recovery.
- Both simulated lidars use a 5.0 m maximum range. The hardware sensor's
  stated 12 m maximum is reserved for later hardware calibration.
- Wall following: `wall_clearance=0.12` m from the body edge, equivalently
  0.52 m from the rear axle when parallel. The 0.04 m hard guard margin is
  unchanged. Approach, corner and door transitions are not forced inside
  the 0.15 m steady-following acceptance band.
- Normal acceleration limits: 0.5 m/s^2, 0.8 rad/s^2, with acceleration slew
  limits of 1 m/s^3 and 2 rad/s^3. Target snapping and emergency overrides
  mean these are smoothing settings, not a formal jerk guarantee.
- Braking check: hold measured motion through 0.2 s reaction time, transition
  within acceleration limits, then brake to rest. Add 0.04 m geometric margin
  and sampling padding. A zero command does not erase measured momentum.
- Front-wall soft stop: target 0.12 m clearance, including deceleration ramp
  time. Keep the stop intent latched until the driver stops/reverses/rotates.
- Watchdogs: joystick 0.3 s, odometry 0.1 s, each laser 0.45 s, planner 0.25 s.
  These are ROS message receipt deadlines; the web input node has its own
  upstream timeout. Odom/scan acquisition stamps are checked separately.
  Odometry acquisition age is checked again on every control tick, including
  manual override. The guard includes travel since that pose and an additional
  acceleration-uncertainty margin over its braking horizon.

Gazebo wheel odometry integrates rear-axle differential drive motion. The new
TF exposes it as `odom -> rear_axle`; the Gazebo odometry child-frame label
is not used as a physical body origin. From this axle, each lidar is at
x=0.79 m, y=+/-0.26 m; body bounds are x=[-0.25,0.97], y=[-0.40,0.40].
Transforms are looked up at scan acquisition time. Footprint self filtering
also uses acquisition-time geometry, not the later position of the chair.
Change the geometry in the coordinator, guard and Nav2 YAML together when
calibrating a different chassis. Hardware odometry needs its own frame audit.

`/shared_control/reference` is the selected geometric reference, not the
optimizer's final time-varying trajectory. Existing Gazebo preview cylinders
still show constant-twist predictions for raw and final commands.

## Reproduce the checks

Run only in Gazebo, with no other joystick client active. Wait for the web
control and ROS nodes to finish starting. Inside the GUI container:

```bash
source /opt/ros/jazzy/setup.bash
source /workspaces/SmartWheelChair/ros2_ws/install/setup.bash
python3 scripts/probe_unified_control.py wall --seconds 18
python3 scripts/probe_unified_control.py front --seconds 16
python3 scripts/probe_unified_control.py override --seconds 18
python3 scripts/probe_unified_control.py wall_gap --wall-side left --seconds 7
python3 scripts/probe_unified_control.py wall_gap --wall-side right --seconds 7
python3 scripts/probe_unified_control.py opening_turn --wall-side left --seconds 14
python3 scripts/probe_unified_control.py opening_turn --wall-side right --seconds 14
python3 scripts/probe_unified_control.py opening_straight --wall-side left --seconds 10
```

These cases require `m6_room.sdf`. Restart the simulation before each case:
teleporting and clearing Nav2 alone does not reset coordinator wall-side and
jamb-memory state. Each probe moves the chair to a controlled start pose and
clears the local costmap. Results default to
`/tmp/unified-probe.json`; use `--output` to retain separate JSON records.

For the door test, start a fresh `unified_door.sdf` via the README command,
then run the following inside the sourced container:

```bash
python3 scripts/probe_unified_control.py door --lateral-m .20 --yaw-deg 15 --seconds 35
python3 scripts/probe_unified_control.py door --lateral-m -.20 --yaw-deg -15 --seconds 35
```

Restart the door scene before each repetition. A stopped command clears
uncommitted opening observations, so a Gazebo pose reset cannot reuse geometry
from the previous pose. World-coordinate reports use relative wheel odometry
and the pose explicitly set by the probe.
The fixture is exactly 1.0 m wide and 0.2 m deep. The pre-redesign room's
named "narrow door" was much wider. The redesigned `m6_room` now has six
actual 1.0/1.1/1.2 m doors; this isolated fixture remains separate.

The scripts assert sampled clearance and scenario-specific progress/stop
conditions, always send a final zero, and preserve the sampled data. Door
world positions are reconstructed from wheel odometry and the known spawn;
they are not independent Gazebo contact/ground-truth measurements.

Observed local runs (2026-09-07/08, before the `m6_room` layout redesign;
not statistical safety guarantees):

| Scenario | Result |
| --- | --- |
| 30-degree wall approach, sustained steering toward wall | Wall mode, speed recovered to 0.8 m/s; minimum sampled clearance 0.368 m |
| Frontal transverse wall | Remained in front-stop mode; final speed zero, sampled clearance 0.129 m |
| Wide left/right corridor turn | Heading changed +1.586/-1.650 rad; minimum sampled clearance 0.111/0.080 m |
| Straight past the same opening | No opening-turn mode; heading drift 0.0073 rad |
| 1.0 m door, +/-15-degree heading and +/-0.20 m body offset | Both rear axles cleared the door; minimum sampled clearance 0.087/0.086 m |
| Steering away during wall following | Exited to override mode and followed -0.42 rad/s requested steering |

The automated suite also covers corners versus fictitious diagonal walls,
too-narrow door rejection, rear sweep, braking momentum, valid infinite scans,
stale inputs/planner, driver release, persistent override and jamb memory.

Door alignment intentionally slows near a one-metre opening. At the staging
point, heading closes under bounded feedback before pass is committed; the
whole-body entry projection must retain the hard margin. Confirmed static
openings persist through short lidar occlusions but are cleared by stop/reverse
or after the wheelchair passes them. The close-wall smoke checks
sampled body-edge gaps of 0.104-0.123 m on the left and 0.102-0.111 m on the
right. These short steady-following checks do not impose a 0.15 m limit during
initial approach or obstacle transitions.

## Version 1 boundaries

Indoor planar static geometry. A doorway must have two observed, approximately
coplanar jamb segments, no measured wall support inside the proposed gap, and
enough measured width for the inflated footprint. Side openings at least 1.8 m
wide can become an intended corridor turn; narrower gaps remain door candidates.
No automatic reverse recovery; stopping/reverse/explicit steering cancels door
assistance. Front cross walls remain stop obstacles unless a valid doorway in
the requested direction is identified. The doorway target persists in odom
until the rear of the wheelchair is clear. Nearby observed jamb points remain
in the guard even when assistance is cancelled; they are spatially pruned
beyond 4.5 m. This static memory is intentionally conservative and has no
dynamic-door clearing model in V1.

Safety validation includes the rectangular body envelope and a braking tail,
not just the axle/wheel preview lines. Margins, latency and achievable braking
must be calibrated on hardware; simulated checks are not a hardware safety
certification. Missing or stale input messages and planner loss cause a stop.

Two planar front-mounted scanners do not establish visibility of every rear
blind spot, transparent surface, low obstacle, drop-off or person. Valid
infinite returns are treated as free to the configured range. No dynamic
obstacle prediction, scan deskew, independent motor watchdog or hardware
fault certification is provided. Do not treat these simulation checks as
approval to run the controller on an occupied wheelchair. Hardware deployment
requires measured braking/delay bounds, uncertainty margins, watchdogs and
broader repeated scenario/contact tests. RK3568 timing is not benchmarked.

The new implementation remains local until separately requested to push.
