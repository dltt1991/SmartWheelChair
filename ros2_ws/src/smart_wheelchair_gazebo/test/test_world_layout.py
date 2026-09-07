import pathlib
import unittest
import xml.etree.ElementTree as ET


WORLD = pathlib.Path(__file__).parents[1] / "worlds" / "m6_room.sdf"
LAUNCH = pathlib.Path(__file__).parents[1] / "launch" / "sim.launch.py"
CMAKE = pathlib.Path(__file__).parents[1] / "CMakeLists.txt"
GUI_CONFIG = pathlib.Path(__file__).parents[1] / "config" / "top_down_gui.config"
GUI_SCRIPT = pathlib.Path(__file__).parents[4] / "docker" / "gazebo-gui-vnc.sh"
TRAJECTORY_PLUGIN = pathlib.Path(__file__).parents[1] / "src" / "trajectory_preview_system.cc"


class WorldLayoutTest(unittest.TestCase):
    def setUp(self):
        self.root = ET.parse(WORLD).getroot()

    def test_floor_is_large_enough_for_maze_testing(self):
        size = self._model_size("floor")

        self.assertGreaterEqual(size[0], 16.0)
        self.assertGreaterEqual(size[1], 12.0)

    def test_world_contains_core_maze_scenarios(self):
        names = {model.attrib["name"] for model in self.root.findall(".//model")}

        expected = {
            "long_corridor_north_wall",
            "long_corridor_south_wall",
            "narrow_door_left_jamb",
            "narrow_door_right_jamb",
            "small_room_back_wall",
            "large_room_back_wall",
        }
        self.assertTrue(expected.issubset(names))

    def test_floor_has_wood_plank_texture(self):
        names = {model.attrib["name"] for model in self.root.findall(".//model")}

        self.assertIn("wood_floor_plank_0", names)
        self.assertIn("wood_floor_grain_0", names)

        diffuse = self._model_diffuse("wood_floor_plank_0")
        self.assertGreater(diffuse[0], diffuse[1])
        self.assertGreater(diffuse[1], diffuse[2])

    def test_passages_are_wide_enough_for_wheelchair(self):
        north_y_min = self._model_bounds("long_corridor_north_wall")[1]
        south_y_max = self._model_bounds("long_corridor_south_wall")[3]
        door_left_min = self._model_bounds("narrow_door_left_jamb")[1]
        door_right_max = self._model_bounds("narrow_door_right_jamb")[3]
        small_entry_min = self._model_bounds("small_room_entry_left_wall")[0]
        small_entry_max = self._model_bounds("small_room_entry_right_wall")[2]

        self.assertGreaterEqual(north_y_min - south_y_max, 2.4)
        self.assertGreaterEqual(door_left_min - door_right_max, 1.4)
        self.assertGreaterEqual(small_entry_min - small_entry_max, 1.4)

    def test_launch_uses_ten_centimeter_stop_distance(self):
        launch_source = LAUNCH.read_text()

        self.assertIn('"stop_distance_m": 0.10', launch_source)

    def test_launch_bridges_raw_joystick_for_gazebo_preview_plugin(self):
        launch_source = LAUNCH.read_text()

        self.assertIn("/cmd_vel_raw@geometry_msgs/msg/Twist]gz.msgs.Twist", launch_source)
        self.assertIn("GZ_SIM_SYSTEM_PLUGIN_PATH", launch_source)
        self.assertNotIn("trajectory_preview_node", launch_source)

    def test_gui_launch_uses_top_down_config(self):
        launch_source = LAUNCH.read_text()
        cmake_source = CMAKE.read_text()

        self.assertIn("top_down_gui.config", launch_source)
        self.assertIn("--gui-config", launch_source)
        self.assertIn("config", cmake_source)

    def test_gui_config_camera_is_top_down_over_wheelchair(self):
        config_root = ET.parse(GUI_CONFIG).getroot()
        scene = config_root.find(".//plugin[@filename='MinimalScene']")
        self.assertIsNotNone(scene)

        pose = [float(value) for value in scene.findtext("camera_pose").split()]
        self.assertAlmostEqual(pose[0], -6.7, places=1)
        self.assertAlmostEqual(pose[1], 0.0, places=1)
        self.assertGreaterEqual(pose[2], 8.0)
        self.assertAlmostEqual(pose[4], 1.5708, places=3)

    def test_gui_config_loads_marker_manager(self):
        config_root = ET.parse(GUI_CONFIG).getroot()

        marker_manager = config_root.find(".//plugin[@filename='MarkerManager']")
        self.assertIsNotNone(marker_manager)

    def test_trajectory_preview_uses_visible_cylinder_markers(self):
        source = TRAJECTORY_PLUGIN.read_text()

        self.assertIn("gz::msgs::Marker::CYLINDER", source)
        self.assertIn("smart_wheelchair_trajectory", source)
        self.assertNotIn("gz::msgs::Marker::LINE_STRIP", source)

    def test_vnc_startup_forces_top_down_camera_pose(self):
        script_source = GUI_SCRIPT.read_text()

        self.assertIn("/gui/move_to/pose", script_source)
        self.assertIn("x: -6.7", script_source)
        self.assertIn("z: 9.0", script_source)
        self.assertIn("y: 0.7071068", script_source)
        self.assertIn("data: true", script_source)

    def _model_size(self, name):
        model = self.root.find(f".//model[@name='{name}']")
        self.assertIsNotNone(model)
        size = model.findtext(".//visual//size")
        self.assertIsNotNone(size)
        return [float(value) for value in size.split()]

    def _model_pose(self, name):
        model = self.root.find(f".//model[@name='{name}']")
        self.assertIsNotNone(model)
        pose = model.findtext("pose", "0 0 0 0 0 0")
        return [float(value) for value in pose.split()]

    def _model_bounds(self, name):
        pose = self._model_pose(name)
        size = self._model_size(name)
        return (
            pose[0] - size[0] / 2,
            pose[1] - size[1] / 2,
            pose[0] + size[0] / 2,
            pose[1] + size[1] / 2,
        )

    def _model_diffuse(self, name):
        model = self.root.find(f".//model[@name='{name}']")
        self.assertIsNotNone(model)
        diffuse = model.findtext(".//diffuse")
        self.assertIsNotNone(diffuse)
        return [float(value) for value in diffuse.split()]


if __name__ == "__main__":
    unittest.main()
