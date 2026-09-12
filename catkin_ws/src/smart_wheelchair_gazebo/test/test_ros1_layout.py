import pathlib
import unittest

ROOT = pathlib.Path(__file__).parents[4]


class Ros1LayoutTest(unittest.TestCase):
    def test_operational_docs_have_no_ros2_instructions(self):
        docs = (ROOT / "README.md").read_text() + (
            ROOT / "docs/architecture-and-algorithms.md").read_text()
        for forbidden in ("ros2 ", "ROS 2", "Jazzy", "Nav2", "colcon",
                          "ros_gz", "ros2_ws"):
            self.assertNotIn(forbidden, docs)
        for required in ("ROS Noetic", "Gazebo 11", "catkin_make"):
            self.assertIn(required, docs)

    def test_workspace_and_packages_use_catkin_only(self):
        self.assertTrue((ROOT / "catkin_ws/src").is_dir())
        self.assertFalse((ROOT / "ros2_ws").exists())

    def test_both_packages_export_catkin(self):
        for package in ("smart_wheelchair_safety", "smart_wheelchair_gazebo"):
            directory = ROOT / "catkin_ws/src" / package
            self.assertIn("catkin_package(", (directory / "CMakeLists.txt").read_text())
            self.assertIn("<buildtool_depend>catkin</buildtool_depend>",
                          (directory / "package.xml").read_text())


if __name__ == "__main__":
    unittest.main()
