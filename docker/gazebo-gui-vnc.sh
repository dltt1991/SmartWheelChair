#!/usr/bin/env bash
set -e

export DISPLAY="${VNC_DISPLAY:-:1}"
export LIBGL_ALWAYS_SOFTWARE=1
export MESA_GL_VERSION_OVERRIDE="${MESA_GL_VERSION_OVERRIDE:-3.3}"
export QT_X11_NO_MITSHM=1
export HOME="${HOME:-/root}"

rm -f "/tmp/.X${DISPLAY#:}-lock"

Xvfb "$DISPLAY" -screen 0 "${VNC_GEOMETRY:-1600x1000x24}" -ac +extension GLX +render -noreset >/tmp/xvfb.log 2>&1 &
for _ in {1..40}; do
  xdpyinfo -display "$DISPLAY" >/dev/null 2>&1 && break
  sleep 0.25
done

mkdir -p "$HOME/.fluxbox"
printf 'session.screen0.rootCommand:\n' > "$HOME/.fluxbox/init"
fluxbox >/tmp/fluxbox.log 2>&1 &
(
  sleep 1
  pkill -f '^xmessage .*fbsetbg' || true
) &
x11vnc -display "$DISPLAY" -forever -shared -nopw -noxdamage -repeat -rfbport 5900 >/tmp/x11vnc.log 2>&1 &
websockify --web=/usr/share/novnc/ 0.0.0.0:6080 localhost:5900 >/tmp/novnc.log 2>&1 &

cd /workspaces/SmartWheelChair/ros2_ws
colcon build --symlink-install
source install/setup.bash

exec ros2 launch smart_wheelchair_gazebo sim.launch.py gui:=true
