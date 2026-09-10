# Manual and Assist Mode Toggle Design

## Goal

Add a two-state control to the web joystick on port 8090. Assist mode keeps the
existing wall following, obstacle avoidance, wide-opening turns, narrow-door
alignment, MPPI planning, smoothing, and independent braking guard. Manual mode
uses the web joystick command directly and does not use any of those assistance
or safety behaviors.

The simulation starts in assist mode. Switching modes always stops the chair and
requires the joystick to return to center before motion resumes.

## Architecture

Keep `unified_control_node` as the only ROS publisher that commands `/cmd_vel`.
The web joystick publishes the requested mode on `/assist_enabled` alongside its
existing `/cmd_vel_raw` output. This avoids competing `/cmd_vel` publishers and
keeps mode transitions in one command owner.

The web server exposes one mode resource:

- `GET /mode` returns the current server-side mode.
- `POST /mode` accepts an `assist_enabled` boolean and returns the resulting mode.

The browser reads `/mode` when it loads and periodically while open, so a refresh
or a second tab displays the actual shared mode rather than stale local state.

## Transition Safety

On every actual mode change, the web node:

1. Clears its stored joystick input and active client.
2. Marks the input as requiring a neutral command.
3. Publishes a zero raw command and the new mode.
4. Rejects nonzero web commands until a neutral command has been received.

The browser also recenters the visible joystick when the mode control is used.
The server-side neutral gate is authoritative if a browser disconnects, another
tab is active, or HTTP and ROS callbacks arrive in an unexpected order.

When the unified controller receives a changed mode, it publishes no carried-over
motion. It cancels the active Nav2 goal and clears wall-side memory, opening-turn
state, door state, front-stop latches, output acceleration, and prior output. The
controller also requires a fresh neutral `/cmd_vel_raw` sample before accepting
motion in the new mode.

## Control Behavior

### Assist Mode

Assist mode is the existing unified chain:

`/cmd_vel_raw -> references and MPPI -> smoothing -> braking guard -> /cmd_vel`

All existing watchdogs, obstacle checks, wall following, opening selection, and
door handling remain unchanged.

### Manual Mode

After the neutral transition gate opens, manual mode publishes the fresh
`/cmd_vel_raw` command directly to `/cmd_vel`. It bypasses lidar and odometry
freshness checks, MPPI, obstacle avoidance, wall following, wide-opening and door
logic, acceleration smoothing, and the independent braking envelope.

The web command timeout remains active and publishes zero if browser commands
stop arriving. The joystick mapping and its configured forward, reverse, and
angular limits remain the source of manual commands. This is not an obstacle-safe
mode.

The shared-control status reports `manual_direct` while this path owns the output.

## Web Interface

Add a compact two-state mode control above the joystick. It displays explicit
`辅助模式` and `手动模式` text and exposes pressed/state semantics for keyboard and
screen-reader users.

- Assist mode keeps the joystick center knob teal with its current teal shadow.
- Manual mode changes the knob to gray and removes the teal shadow.
- Connection status remains separate from mode status.
- A failed mode request leaves the displayed state unchanged and shows a
  connection error.

The control uses the existing page and native HTML, CSS, and JavaScript. No new
frontend framework, icon package, or build step is introduced.

## Testing

Tests cover:

- `GET /mode` and valid/invalid `POST /mode` behavior;
- switching modes clears motion and rejects nonzero input until neutral;
- manual mode publishes the raw joystick command without planner, obstacle, or
  smoothing intervention;
- command timeout still stops manual mode;
- returning to assist mode clears manual output and restores the existing chain;
- the page contains the mode control and teal/gray knob states; and
- existing unified-control and Gazebo tests remain green.

Gazebo verification will confirm that switching in either direction first stops,
manual mode can drive without assistance ownership, and assist mode again reports
normal wall/manual/door states after a neutral release.

## Scope

Modify only the web joystick node, unified control node, their focused tests, and
the unified-control documentation. Do not add a node, package, dependency, dynamic
launch restart, or second `/cmd_vel` publisher.
