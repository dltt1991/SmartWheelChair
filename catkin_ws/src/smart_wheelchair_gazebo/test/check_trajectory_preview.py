#!/usr/bin/env python3
"""Native scene/lidar check; source Gazebo setup.sh and catkin setup.bash,
then run under xvfb-run while roscore is available. Keep Xvfb alive until exit.
"""
import copy
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

import rospy
from rosgraph_msgs.msg import Clock


package = Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory(prefix='swc-preview-check-') as directory:
    root = ET.parse(package / 'worlds/m6_room.world')
    world = root.find('world')
    include = world.find('include')
    model = copy.deepcopy(ET.parse(package / 'models/smart_wheelchair/model.sdf').find('model'))
    model.find('pose').text = include.findtext('pose')
    # Freeze every articulated link; a fixed world joint still permits ODE drift.
    ET.SubElement(model, 'static').text = 'true'
    for plugin in list(model.findall('plugin')):
        if plugin.attrib['filename'] != 'libSmartWheelChairTrajectoryPreview.so':
            model.remove(plugin)
    world.remove(include)
    world.append(model)
    fixture = Path(directory) / 'preview.world'
    root.write(fixture)
    with tempfile.TemporaryFile(mode='w+') as log:
        server = subprocess.Popen(['gzserver', '-s', 'libgazebo_ros_api_plugin.so', str(fixture)],
                                  stdout=log, stderr=subprocess.STDOUT)
        try:
            rospy.init_node('preview_check_launcher', anonymous=True, disable_signals=True)
            rospy.wait_for_message('/clock', Clock, timeout=30)
            subprocess.run(['rosrun', 'smart_wheelchair_gazebo', 'trajectory_preview_render_check', *sys.argv[1:]],
                           check=True, timeout=90)
        finally:
            server.send_signal(signal.SIGINT)
            try:
                server.wait(timeout=20)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
            log.seek(0)
            print(log.read())
            if server.returncode != 0:
                raise RuntimeError(f'Gazebo exit status {server.returncode}')
