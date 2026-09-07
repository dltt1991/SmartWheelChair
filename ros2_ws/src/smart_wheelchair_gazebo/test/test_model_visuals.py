import pathlib
import unittest
import xml.etree.ElementTree as ET


MODEL = pathlib.Path(__file__).parents[1] / "models" / "smart_wheelchair" / "model.sdf"


class ModelVisualsTest(unittest.TestCase):
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
        sensors = self.root.findall(".//sensor[@type='gpu_lidar']")
        self.assertEqual({sensor.attrib["name"] for sensor in sensors}, {"left_lidar", "right_lidar"})
        self.assertTrue(all(sensor.findtext("visualize") == "true" for sensor in sensors))
        self.assertTrue(all(sensor.findtext("update_rate") == "10" for sensor in sensors))

        left = self.root.find(".//sensor[@name='left_lidar']//horizontal")
        self.assertEqual(left.findtext("samples"), "401")
        self.assertAlmostEqual(float(left.findtext("min_angle")), -0.87266, places=4)
        self.assertAlmostEqual(float(left.findtext("max_angle")), 2.61799, places=4)

        right = self.root.find(".//sensor[@name='right_lidar']//horizontal")
        self.assertEqual(right.findtext("samples"), "401")
        self.assertAlmostEqual(float(right.findtext("min_angle")), -2.61799, places=4)
        self.assertAlmostEqual(float(right.findtext("max_angle")), 0.87266, places=4)

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

    def test_trajectory_preview_visuals_are_preallocated_on_base_link(self):
        base_link = self.root.find(".//link[@name='base_link']")
        names = {visual.attrib["name"] for visual in base_link.findall("visual")}

        for path_name in ("rear_axle", "left_wheel", "right_wheel"):
            for index in range(10):
                self.assertIn(f"trajectory_{path_name}_{index}", names)

    def test_trajectory_preview_gazebo_plugin_is_loaded(self):
        plugin = self.root.find(".//plugin[@filename='libSmartWheelChairTrajectoryPreview.so']")

        self.assertIsNotNone(plugin)
        self.assertEqual(plugin.findtext("cmd_topic"), "/cmd_vel_raw")
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
