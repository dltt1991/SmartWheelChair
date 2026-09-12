from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
import re
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[4]


class RuntimeWiringTest(unittest.TestCase):
    def test_registered_rostests_do_not_share_result_files(self):
        package = ROOT / 'catkin_ws/src/smart_wheelchair_safety'
        names = []

        def collect(path):
            launch = ET.parse(path).getroot()
            names.extend(test.attrib['test-name'] for test in launch.iter('test'))
            for include in launch.iter('include'):
                collect(Path(include.attrib['file'].replace(
                    '$(find smart_wheelchair_safety)', str(package))))

        for filename in re.findall(r'add_rostest\(([^)]+)\)',
                                   (package / 'CMakeLists.txt').read_text()):
            collect(package / filename)
        self.assertEqual(len(names), len(set(names)),
                         'parallel rostests must not overwrite the same rosunit XML')
        self.assertEqual(set(names), {'joystick', 'unified_geometry',
                                     'local_path_follower', 'local_path_follower_node',
                                     'web_joystick', 'unified_control', 'wall_end'})

    def test_runtime_checks_are_registered_with_catkin(self):
        cmake = (ROOT / 'catkin_ws/src/smart_wheelchair_gazebo/CMakeLists.txt').read_text()
        self.assertIn('catkin_add_nosetests(test)', cmake)

    def test_container_uses_noetic_catkin_and_roslaunch(self):
        dockerfile = (ROOT / 'docker/Dockerfile').read_text()
        entrypoint = (ROOT / 'docker/entrypoint.sh').read_text()
        gui = (ROOT / 'docker/gazebo-gui-vnc.sh').read_text()
        self.assertIn('noetic-ros-base-focal', dockerfile)
        self.assertIn('/opt/ros/noetic/setup.bash', entrypoint)
        self.assertIn('/catkin_ws/devel/setup.bash', entrypoint)
        self.assertIn('catkin_make', gui)
        self.assertIn('roslaunch smart_wheelchair_gazebo sim.launch', gui)
        for forbidden in ('jazzy', 'colcon', 'ros2 launch', 'ros-gz', 'gz service'):
            self.assertNotIn(forbidden, dockerfile + entrypoint + gui)

    def test_compose_preserves_ports_and_allows_ordered_shutdown(self):
        compose = (ROOT / 'docker-compose.yml').read_text()
        self.assertEqual(compose.count('smart-wheelchair-ros:noetic'), 2)
        for port in (6080, 5900, 8090):
            self.assertIn('"{0}:{0}"'.format(port), compose)
        self.assertIn('stop_grace_period: 60s', compose)

    def test_gui_waits_for_roslaunch_before_stopping_display(self):
        gui = (ROOT / 'docker/gazebo-gui-vnc.sh').read_text()
        self.assertIn('trap cleanup EXIT', gui)
        self.assertIn("trap 'exit 143' TERM", gui)
        self.assertIn("trap 'exit 130' INT", gui)
        self.assertIn('kill -INT "$launch_pid"', gui)
        self.assertLess(gui.index('wait "$launch_pid"'), gui.index('kill "$xvfb_pid"'))
        self.assertNotIn('exec roslaunch', gui)

    def test_shutdown_traps_wait_for_children_on_term_int_and_exit(self):
        # Exercise the production traps with real processes, without requiring X/ROS.
        prefix = (ROOT / 'docker/gazebo-gui-vnc.sh').read_text().split('rm -f ', 1)[0]
        worker = '''import pathlib, signal, sys, time
events, role = pathlib.Path(sys.argv[1]), sys.argv[2]
def stop(*unused):
    time.sleep(.1)
    with events.open('a') as log:
        log.write(role + '\\n')
    sys.exit(0)
signal.signal(signal.SIGINT, stop)
signal.signal(signal.SIGTERM, stop)
events.with_suffix('.' + role).touch()
while True:
    time.sleep(.01)
'''
        for ending, code in (('kill -TERM $$', 143), ('kill -INT $$', 130), ('exit 7', 7)):
            with self.subTest(ending=ending), tempfile.TemporaryDirectory() as temporary:
                events = Path(temporary) / 'events'
                command = ' '.join(map(shlex.quote, [sys.executable, '-c', worker, str(events)]))
                script = prefix + '\n' + '\n'.join([
                    command + ' display &', 'xvfb_pid=$!',
                    command + ' ros &', 'launch_pid=$!',
                    'while [[ ! -e {0}.display || ! -e {0}.ros ]]; do sleep .01; done'.format(shlex.quote(str(events))),
                    ending,
                ])
                result = subprocess.run(['bash', '-c', script], capture_output=True, text=True, timeout=5)
                self.assertEqual(result.returncode, code, result.stderr)
                self.assertEqual(events.read_text().splitlines(), ['ros', 'display'])

    def test_probe_uses_rospy_and_classic_model_services(self):
        probe = (ROOT / 'scripts/probe_unified_control.py').read_text()
        self.assertIn('import rospy', probe)
        self.assertIn('/gazebo/get_world_properties', probe)
        self.assertIn('/gazebo/set_model_state', probe)
        for forbidden in ('rclpy', "['gz',", "['ros2',", 'ClearEntireCostmap'):
            self.assertNotIn(forbidden, probe)


if __name__ == '__main__':
    unittest.main()
