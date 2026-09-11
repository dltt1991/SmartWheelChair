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
    # Non-zero yaw makes the render check exercise world-to-parent conversion.
    model.find('pose').text = '-6.7 0 0 0 0 0.45'
    # GUI-only body occluders: no collision, so ROS scans must remain unchanged.
    for side, y in ([] if '--body-clearance' in sys.argv else [('left', .36), ('right', -.36)]):
        visual = ET.SubElement(model.find("link[@name='base_link']"), 'visual',
                               name=side + '_test_occluder_visual')
        lateral = y + (.212132034356 if side == 'left' else -.212132034356)
        ET.SubElement(visual, 'pose').text = f'.247867965644 {lateral} .56 0 0 0'
        box = ET.SubElement(ET.SubElement(visual, 'geometry'), 'box')
        ET.SubElement(box, 'size').text = '.1 .1 .1'
        for z in (.2, 1.):
            visual = ET.SubElement(model.find("link[@name='base_link']"), 'visual',
                                   name=f'{side}_off_plane_{z}_visual')
            ET.SubElement(visual, 'pose').text = f'.76 {y} {z} 0 0 .4'
            box = ET.SubElement(ET.SubElement(visual, 'geometry'), 'box')
            ET.SubElement(box, 'size').text = '.1 .1 .1'
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
        subprocess.run(['rosparam', 'set', '/use_sim_time', 'true'], check=True, timeout=10)
        server = subprocess.Popen(['gzserver', '-s', 'libgazebo_ros_api_plugin.so', str(fixture)],
                                  stdout=log, stderr=subprocess.STDOUT)
        try:
            rospy.init_node('preview_check_launcher', anonymous=True, disable_signals=True)
            rospy.wait_for_message('/clock', Clock, timeout=30)
            subprocess.run(['rosservice', 'call', '/gazebo/unpause_physics'], check=True, timeout=10)
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
