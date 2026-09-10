import pathlib
import unittest
import xml.etree.ElementTree as ET


WORLD = pathlib.Path(__file__).parents[1] / "worlds" / "m6_room.sdf"
LAUNCH = pathlib.Path(__file__).parents[1] / "launch" / "sim.launch.py"
CMAKE = pathlib.Path(__file__).parents[1] / "CMakeLists.txt"
GUI_CONFIG = pathlib.Path(__file__).parents[1] / "config" / "top_down_gui.config"
UNIFIED_CONFIG = pathlib.Path(__file__).parents[1] / "config" / "unified_control.yaml"
GUI_SCRIPT = pathlib.Path(__file__).parents[4] / "docker" / "gazebo-gui-vnc.sh"
TRAJECTORY_PLUGIN = pathlib.Path(__file__).parents[1] / "src" / "trajectory_preview_system.cc"
WHEELCHAIR_MODEL = pathlib.Path(__file__).parents[1] / "models" / "smart_wheelchair" / "model.sdf"


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
            "door_nw_horizontal_a",
            "door_ne_horizontal_a",
            "door_sw_horizontal_a",
            "door_se_horizontal_a",
            "vertical_corridor_west_south",
            "vertical_corridor_east_north",
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
        north_y_min = self._model_bounds("door_nw_horizontal_a")[1]
        south_y_max = self._model_bounds("door_sw_horizontal_a")[3]
        west_x_max = self._model_bounds("vertical_corridor_west_south")[2]
        east_x_min = self._model_bounds("vertical_corridor_east_north")[0]

        self.assertAlmostEqual(north_y_min - south_y_max, 2.58)
        self.assertAlmostEqual(east_x_min - west_x_max, 2.58)

    def test_door_fixture_has_exact_one_metre_collision_gap(self):
        root = ET.parse(WORLD.with_name('unified_door.sdf')).getroot()
        edges = []
        for name, sign in [('left_jamb', -1), ('right_jamb', 1)]:
            model = root.find(f".//model[@name='{name}']")
            y = float(model.findtext('pose').split()[1])
            width = float(model.findtext('.//collision/geometry/box/size').split()[1])
            edges.append(y+sign*width/2)
        self.assertAlmostEqual(edges[0]-edges[1], 1.)

    def test_both_lidars_use_five_metre_simulation_range(self):
        root = ET.parse(WHEELCHAIR_MODEL).getroot()
        ranges = [float(sensor.findtext("lidar/range/max"))
                  for sensor in root.findall(".//sensor[@type='gpu_lidar']")]

        self.assertEqual(ranges, [5.0, 5.0])

    def test_launch_uses_only_unified_control_stack(self):
        source = LAUNCH.read_text()

        self.assertNotIn('DeclareLaunchArgument("unified_control"', source)
        self.assertNotIn('wall_follow_assist_node', source)
        self.assertNotIn('safety_filter_node', source)
        self.assertIn('executable="unified_control_node"', source)
        self.assertIn('("cmd_vel", "cmd_vel_planned")', source)
        self.assertIn('/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock', source)

    def test_planner_output_is_stamped_for_mode_transition_barrier(self):
        config = UNIFIED_CONFIG.read_text()

        self.assertIn("enable_stamped_cmd_vel: true", config)

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
        self.assertIn("smart_wheelchair_trajectory_raw", source)
        self.assertIn("smart_wheelchair_trajectory_filtered", source)
        self.assertIn("rawMarkerDiameter{0.03}", source)
        self.assertIn("filteredMarkerDiameter{0.09}", source)
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
