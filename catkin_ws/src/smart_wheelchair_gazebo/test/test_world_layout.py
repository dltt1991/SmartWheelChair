import pathlib
import unittest
import xml.etree.ElementTree as ET


WORLD = pathlib.Path(__file__).parents[1] / "worlds" / "m6_room.world"
LAUNCH = pathlib.Path(__file__).parents[1] / "launch" / "sim.launch"
CMAKE = pathlib.Path(__file__).parents[1] / "CMakeLists.txt"
GUI_CONFIG = pathlib.Path(__file__).parents[1] / "config" / "top_down_gui.config"
FOLLOWER_CONFIG = pathlib.Path(__file__).parents[1] / "config" / "local_path_follower.yaml"
TRAJECTORY_PLUGIN = pathlib.Path(__file__).parents[1] / "src" / "trajectory_preview_plugin.cc"
WHEELCHAIR_MODEL = pathlib.Path(__file__).parents[1] / "models" / "smart_wheelchair" / "model.sdf"


class WorldLayoutTest(unittest.TestCase):
    def setUp(self):
        self.assertTrue(WORLD.is_file(), 'Gazebo 11 .world asset is missing')
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
        root = ET.parse(WORLD.with_name('unified_door.world')).getroot()
        edges = []
        for name, sign in [('left_jamb', -1), ('right_jamb', 1)]:
            model = root.find(f".//model[@name='{name}']")
            y = float(model.findtext('pose').split()[1])
            width = float(model.findtext('.//collision/geometry/box/size').split()[1])
            edges.append(y+sign*width/2)
        self.assertAlmostEqual(edges[0]-edges[1], 1.)

    def test_both_lidars_use_five_metre_simulation_range(self):
        root = ET.parse(WHEELCHAIR_MODEL).getroot()
        ranges = [float(sensor.findtext("ray/range/max"))
                  for sensor in root.findall(".//sensor[@type='gpu_ray']")]

        self.assertEqual(ranges, [5.0, 5.0])

    def test_launch_uses_only_unified_control_stack(self):
        self.assertTrue(LAUNCH.is_file(), 'ROS1 XML launch is missing')
        root = ET.parse(LAUNCH).getroot()
        nodes = root.findall('node')
        self.assertEqual({node.attrib['type'] for node in nodes}, {
            'local_path_follower_node', 'unified_control_node', 'web_joystick_node', 'odom_frame_relay'})
        self.assertTrue(all(node.attrib['pkg'] == 'smart_wheelchair_safety' for node in nodes))
        self.assertEqual(root.find("param[@name='use_sim_time']").attrib['value'], 'true')
        self.assertEqual(root.find("arg[@name='world']").attrib['default'],
                         '$(find smart_wheelchair_gazebo)/worlds/m6_room.world')
        self.assertIsNotNone(root.find("arg[@name='gui']"))
        include = root.find("include[@file='$(find gazebo_ros)/launch/empty_world.launch']")
        self.assertIsNotNone(include)
        self.assertEqual(include.find("arg[@name='world_name']").attrib['value'], '$(arg world)')
        self.assertEqual(include.find("arg[@name='gui']").attrib['value'], '$(arg gui)')
        config = root.find(".//rosparam[@command='load']")
        self.assertEqual(config.attrib['file'], '$(find smart_wheelchair_gazebo)/config/local_path_follower.yaml')
        self.assertTrue(FOLLOWER_CONFIG.is_file())

    def test_worlds_use_classic_sdf_and_top_down_camera(self):
        for path in (WORLD, WORLD.with_name('unified_door.world')):
            root = ET.parse(path).getroot()
            self.assertIn(root.attrib['version'], ('1.6', '1.7'))
            self.assertEqual(root.find('world/physics').attrib['type'], 'ode')
            camera = root.find("world/gui/camera[@name='top_down']")
            self.assertIsNotNone(camera)
            self.assertEqual(list(map(float, camera.findtext('pose').split())),
                             [-6.7, 0., 9., 0., 1.5708, 0.])
            self.assertEqual(camera.findtext('view_controller'), 'orbit')
        self.assertFalse(GUI_CONFIG.exists())

    def test_runtime_assets_reject_retired_dependencies(self):
        package = WORLD.parents[1]
        files = [CMAKE, package / 'package.xml', WHEELCHAIR_MODEL, LAUNCH,
                 WORLD, WORLD.with_name('unified_door.world'), FOLLOWER_CONFIG, TRAJECTORY_PLUGIN]
        for path in files:
            for retired in ('gz-sim-', 'ros_gz', 'ament', 'nav2'):
                self.assertNotIn(retired, path.read_text(), str(path))
        self.assertFalse((package / 'launch/sim.launch.py').exists())
        self.assertFalse((package / 'config/unified_control.yaml').exists())

    def test_trajectory_preview_uses_classic_visual_transport(self):
        self.assertTrue(TRAJECTORY_PLUGIN.is_file(), 'Gazebo 11 plugin is missing')
        source = TRAJECTORY_PLUGIN.read_text()
        self.assertIn('gazebo::ModelPlugin', source)
        self.assertIn('gazebo::msgs::Visual', source)
        self.assertIn('gazebo::msgs::Geometry::CYLINDER', source)
        self.assertIn('"~/visual"', source)
        self.assertIn('set_delete_me(true)', source)
        self.assertIn('ros::CallbackQueue', source)
        self.assertIn("smart_wheelchair_trajectory_raw", source)
        self.assertIn("smart_wheelchair_trajectory_filtered", source)
        self.assertIn("rawMarkerDiameter{0.03}", source)
        self.assertIn("filteredMarkerDiameter{0.09}", source)

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
