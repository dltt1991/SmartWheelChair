#!/usr/bin/env bash
set -e

export DISPLAY="${VNC_DISPLAY:-:1}"
export LIBGL_ALWAYS_SOFTWARE=1
# llvmpipe otherwise consumes every Docker CPU and starves ROS callbacks.
export LP_NUM_THREADS="${LP_NUM_THREADS:-2}"
export MESA_GL_VERSION_OVERRIDE="${MESA_GL_VERSION_OVERRIDE:-3.3}"
export QT_X11_NO_MITSHM=1

launch_pid=
xvfb_pid=
desktop_pids=()
cleanup() {
  trap '' TERM INT
  # roslaunch signals and reaps gzserver/gzclient while their display still exists.
  if [[ -n "$launch_pid" ]]; then
    kill -INT "$launch_pid" 2>/dev/null || true
    wait "$launch_pid" || true
  fi
  for pid in "${desktop_pids[@]}"; do
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  done
  if [[ -n "$xvfb_pid" ]]; then
    kill "$xvfb_pid" 2>/dev/null || true
    wait "$xvfb_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

rm -f "/tmp/.X${DISPLAY#:}-lock"

Xvfb "$DISPLAY" -screen 0 "${VNC_GEOMETRY:-1600x1000x24}" -ac +extension GLX +render -noreset >/tmp/xvfb.log 2>&1 &
xvfb_pid=$!
for _ in {1..40}; do
  xdpyinfo -display "$DISPLAY" >/dev/null 2>&1 && break
  sleep 0.25
done
xdpyinfo -display "$DISPLAY" >/dev/null 2>&1

mkdir -p "$HOME/.fluxbox"
printf 'session.screen0.rootCommand:\n' > "$HOME/.fluxbox/init"
fluxbox >/tmp/fluxbox.log 2>&1 &
desktop_pids+=($!)
(
  sleep 1
  pkill -f '^xmessage .*fbsetbg' || true
) &
desktop_pids+=($!)
x11vnc -display "$DISPLAY" -forever -shared -nopw -noxdamage -repeat -rfbport 5900 >/tmp/x11vnc.log 2>&1 &
desktop_pids+=($!)
websockify --web=/usr/share/novnc/ 0.0.0.0:6080 localhost:5900 >/tmp/novnc.log 2>&1 &
desktop_pids+=($!)

cd /workspaces/SmartWheelChair/catkin_ws
catkin_make
source devel/setup.bash

args=(gui:=true)
if [[ -n "${WHEELCHAIR_WORLD:-}" ]]; then
  args+=("world:=$WHEELCHAIR_WORLD")
fi
roslaunch smart_wheelchair_gazebo sim.launch "${args[@]}" &
launch_pid=$!
wait "$launch_pid"
