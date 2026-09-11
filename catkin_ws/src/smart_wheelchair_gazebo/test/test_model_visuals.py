import math
import pathlib
import unittest
import xml.etree.ElementTree as ET


MODEL = pathlib.Path(__file__).parents[1] / "models" / "smart_wheelchair" / "model.sdf"


class ModelVisualsTest(unittest.TestCase):
    def test_gui_software_renderer_leaves_cpu_for_ros_callbacks(self):
        launcher = pathlib.Path(__file__).parents[4] / 'docker/gazebo-gui-vnc.sh'
        self.assertIn('export LP_NUM_THREADS="${LP_NUM_THREADS:-2}"', launcher.read_text())

    def setUp(self):
        self.root = ET.parse(MODEL).getroot()

    def test_backrest_does_not_roll_sideways(self):
        pose = self._visual_pose("backrest_visual")
        roll, pitch = pose[3], pose[4]
        self.assertAlmostEqual(roll, 0.0)
        self.assertLess(pitch, 0.0)

    def test_wheelchair_has_expected_visual_detail(self):
        names = {visual.attrib["name"] for visual in self.root.findall(".//visual")}
        expected = {
            "left_armrest_visual",
            "right_armrest_visual",
            "footplate_visual",
            "left_rear_hub_visual",
            "right_rear_hub_visual",
            "left_caster_fork_visual",
            "right_caster_fork_visual",
        }
        self.assertTrue(expected.issubset(names))

    def test_battery_sits_under_chair_inside_body_outline(self):
        visual = self.root.find(".//visual[@name='battery_visual']")
        self.assertIsNotNone(visual)

        pose = [float(value) for value in visual.findtext("pose").split()]
        size = [float(value) for value in visual.findtext(".//size").split()]
        diffuse = [float(value) for value in visual.findtext(".//diffuse").split()]

        self.assertGreaterEqual(pose[0] - size[0] / 2, -0.475)
        self.assertLessEqual(pose[0] + size[0] / 2, 0.475)
        self.assertGreaterEqual(pose[1] - size[1] / 2, -0.31)
        self.assertLessEqual(pose[1] + size[1] / 2, 0.31)
        self.assertLessEqual(pose[2] + size[2] / 2, 0.33)
        self.assertLess(diffuse[0], 0.25)

    def test_lidar_scans_are_visible_in_gazebo(self):
        sensors = self.root.findall(".//sensor[@type='ray']")
        self.assertEqual({sensor.attrib["name"] for sensor in sensors}, {"left_lidar", "right_lidar"})
        self.assertTrue(all(sensor.findtext("visualize") == "true" for sensor in sensors))
        self.assertTrue(all(sensor.findtext("update_rate") == "10" for sensor in sensors))

        left = self.root.find(".//sensor[@name='left_lidar']//horizontal")
        self.assertEqual(left.findtext("samples"), "721")
        self.assertAlmostEqual(float(left.findtext("min_angle")), -math.pi, places=4)
        self.assertAlmostEqual(float(left.findtext("max_angle")), math.pi, places=4)

        right = self.root.find(".//sensor[@name='right_lidar']//horizontal")
        self.assertEqual(right.findtext("samples"), "721")
        self.assertAlmostEqual(float(right.findtext("min_angle")), -math.pi, places=4)
        self.assertAlmostEqual(float(right.findtext("max_angle")), math.pi, places=4)

        names = {visual.attrib["name"] for visual in self.root.findall(".//visual")}
        expected = {
            "left_lidar_body_visual",
            "right_lidar_body_visual",
            "left_lidar_ray_minus50_visual",
            "left_lidar_ray_50_visual",
            "left_lidar_ray_150_visual",
            "right_lidar_ray_minus150_visual",
            "right_lidar_ray_minus50_visual",
            "right_lidar_ray_50_visual",
        }
        self.assertTrue(expected.issubset(names))

    def test_classic_ros_sensor_and_drive_interfaces(self):
        self.assertIn(self.root.attrib['version'], ('1.6', '1.7'))
        for side in ('left', 'right'):
            sensor = self.root.find(f".//sensor[@name='{side}_lidar']")
            self.assertIsNotNone(sensor.find('ray'))
            self.assertEqual(sensor.attrib['type'], 'ray')
            plugin = sensor.find("plugin[@filename='libgazebo_ros_laser.so']")
            self.assertIsNotNone(plugin)
            self.assertEqual(plugin.findtext('topicName'), f'/scan_{side}')
            self.assertEqual(plugin.findtext('frameName'), f'{side}_lidar')
            self.assertEqual(plugin.findtext('robotNamespace'), '/')
        camera = self.root.find(".//sensor[@name='rear_camera']/plugin[@filename='libgazebo_ros_camera.so']")
        self.assertIsNotNone(camera)
        self.assertEqual(camera.findtext('cameraName'), 'camera/rear')
        self.assertEqual(camera.findtext('imageTopicName'), 'image')
        self.assertEqual(camera.findtext('cameraInfoTopicName'), 'camera_info')
        self.assertEqual(camera.findtext('frameName'), 'rear_camera')
        drive = self.root.find(".//plugin[@filename='libgazebo_ros_diff_drive.so']")
        self.assertIsNotNone(drive)
        for tag, value in {'alwaysOn': 'true', 'updateRate': '50',
                           'leftJoint': 'left_rear_wheel_joint',
                           'rightJoint': 'right_rear_wheel_joint',
                           'wheelSeparation': '0.72', 'wheelDiameter': '0.36',
                           'commandTopic': '/cmd_vel', 'odometryTopic': '/diff_drive/odom',
                           'odometryFrame': 'odom', 'robotBaseFrame': 'rear_axle',
                           'publishWheelTF': 'false', 'publishOdomTF': 'false',
                           'odometrySource': 'world'}.items():
            self.assertEqual(drive.findtext(tag), value, tag)

    def test_odometry_measures_actual_rear_axle_link(self):
        axle = self.root.find("model/link[@name='rear_axle']")
        self.assertIsNotNone(axle)
        self.assertEqual(list(map(float, axle.findtext('pose').split())), [-0.33, 0, 0, 0, 0, 0])
        joint = self.root.find("model/joint[@name='rear_axle_joint']")
        self.assertEqual(joint.attrib['type'], 'fixed')
        self.assertEqual(joint.findtext('parent'), 'base_link')
        self.assertEqual(joint.findtext('child'), 'rear_axle')
        p3d = self.root.find("model/plugin[@filename='libgazebo_ros_p3d.so']")
        self.assertIsNotNone(p3d)
        self.assertEqual(p3d.findtext('bodyName'), 'rear_axle')
        self.assertEqual(p3d.findtext('topicName'), '/rear_axle_ground_truth')
        self.assertEqual(p3d.findtext('frameName'), 'world')
        self.assertEqual(p3d.findtext('localTwist'), 'true')
        self.assertEqual(p3d.findtext('updateRate'), '50')

    def test_lidar_ray_visuals_are_above_scan_plane(self):
        lidar_height = self._sensor_pose("left_lidar")[2]
        ray_poses = [
            self._visual_pose(visual.attrib["name"])
            for visual in self.root.findall(".//visual")
            if "_lidar_ray_" in visual.attrib["name"]
        ]

        self.assertTrue(ray_poses)
        for pose in ray_poses:
            self.assertGreaterEqual(pose[2], lidar_height + 0.08)

    def test_rear_camera_is_centered_wide_angle(self):
        sensor = self.root.find(".//sensor[@name='rear_camera']")
        self.assertIsNotNone(sensor)
        pose = [float(value) for value in sensor.findtext("pose").split()]
        self.assertAlmostEqual(pose[1], 0.0)
        self.assertAlmostEqual(pose[4], 0.60, places=2)
        self.assertAlmostEqual(float(sensor.findtext(".//horizontal_fov")), 2.094, places=3)

    def test_model_does_not_keep_obsolete_preallocated_trajectory_visuals(self):
        base_link = self.root.find(".//link[@name='base_link']")
        names = {visual.attrib["name"] for visual in base_link.findall("visual")}

        self.assertFalse(any(name.startswith("trajectory_") for name in names))

    def test_trajectory_preview_gazebo_plugin_is_loaded(self):
        plugin = self.root.find(".//plugin[@filename='libSmartWheelChairTrajectoryPreview.so']")

        self.assertIsNotNone(plugin)
        self.assertEqual(plugin.findtext("cmd_topic"), "/cmd_vel_raw")
        self.assertEqual(plugin.findtext("filtered_cmd_topic"), "/cmd_vel")
        self.assertEqual(plugin.findtext("prediction_seconds"), "3.0")

    def _visual_pose(self, name):
        visual = self.root.find(f".//visual[@name='{name}']")
        self.assertIsNotNone(visual)
        return [float(value) for value in visual.findtext("pose").split()]

    def _sensor_pose(self, name):
        sensor = self.root.find(f".//sensor[@name='{name}']")
        self.assertIsNotNone(sensor)
        return [float(value) for value in sensor.findtext("pose").split()]


if __name__ == "__main__":
    unittest.main()
