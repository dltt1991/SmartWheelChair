#!/usr/bin/env bash
set -e

source /opt/ros/noetic/setup.bash

if [ -f /workspaces/SmartWheelChair/catkin_ws/devel/setup.bash ]; then
  source /workspaces/SmartWheelChair/catkin_ws/devel/setup.bash
fi

exec "$@"
