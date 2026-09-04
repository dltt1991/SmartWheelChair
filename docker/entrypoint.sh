#!/usr/bin/env bash
set -e

source /opt/ros/jazzy/setup.bash

if [ -f /workspaces/SmartWheelChair/ros2_ws/install/setup.bash ]; then
  source /workspaces/SmartWheelChair/ros2_ws/install/setup.bash
fi

exec "$@"
