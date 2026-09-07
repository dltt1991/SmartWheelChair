import math
import unittest
from pathlib import Path

from smart_wheelchair_safety.trajectory_preview import (
    delete_marker_text,
    gazebo_model_pose_from_text,
    gazebo_marker_text,
    preview_paths,
    predict_trajectory,
    trajectory_model_sdf,
    trajectory_visual_config_text,
)


class TrajectoryPreviewTest(unittest.TestCase):
    def test_predicts_three_seconds_straight_from_pose(self):
        points = predict_trajectory(1.0, 0.0, -6.7, 0.0, 0.0, seconds=3.0, step=1.0)

        self.assertEqual(points, [(-6.7, 0.0, 0.03), (-5.7, 0.0, 0.03), (-4.7, 0.0, 0.03), (-3.7, 0.0, 0.03)])

    def test_predicts_circular_arc_from_current_yaw(self):
        points = predict_trajectory(1.0, 1.0, 0.0, 0.0, 0.0, seconds=1.0, step=1.0)

        self.assertAlmostEqual(points[-1][0], math.sin(1.0))
        self.assertAlmostEqual(points[-1][1], 1.0 - math.cos(1.0))
        self.assertAlmostEqual(points[-1][2], 0.03)

    def test_skips_preview_when_stopped(self):
        self.assertEqual(predict_trajectory(0.0, 0.0, 0.0, 0.0, 0.0), [])

    def test_gazebo_marker_text_is_line_strip(self):
        text = gazebo_marker_text([(0.0, 0.0, 0.03), (1.0, 0.0, 0.03)], 0.0, 0.85, 1.0)

        self.assertIn("ns: \"trajectory_preview\"", text)
        self.assertIn("type: LINE_STRIP", text)
        self.assertIn("point {x: 1.000", text)
        self.assertIn("diffuse {r: 0.000 g: 0.850 b: 1.000 a: 1.000}", text)

    def test_gazebo_marker_text_can_use_distinct_marker_ids(self):
        text = gazebo_marker_text([(0.0, 0.0, 0.03)], 1.0, 0.45, 0.0, marker_id=2)

        self.assertIn("id: 2", text)
        self.assertIn("diffuse {r: 1.000 g: 0.450 b: 0.000 a: 1.000}", text)

    def test_delete_marker_text_removes_preview(self):
        self.assertIn("action: DELETE_MARKER", delete_marker_text())

    def test_node_uses_gazebo_visual_config_service(self):
        source = (
            Path(__file__).parents[1]
            / "smart_wheelchair_safety"
            / "trajectory_preview_node.py"
        ).read_text()

        self.assertIn('"command_topic", "cmd_vel_raw"', source)
        self.assertIn('"/world/m6_room/visual_config"', source)
        self.assertNotIn("EntityFactory", source)
        self.assertIn('"gz.msgs.Visual"', source)
        self.assertIn('"gz.msgs.Boolean"', source)
        self.assertIn('"marker_z_m", 0.18', source)
        self.assertIn('"command_hold_s", 0.8', source)
        self.assertIn('"command_update_epsilon", 0.03', source)
        self.assertIn("data: true", source)
        self.assertIn("ThreadPoolExecutor", source)

    def test_preview_paths_include_rear_axle_and_two_drive_wheels(self):
        paths = preview_paths(
            1.0,
            0.0,
            start_x_m=0.0,
            start_y_m=0.0,
            start_yaw_rad=0.0,
            seconds=1.0,
            step=1.0,
            rear_axle_x_m=-0.33,
            wheel_separation_m=0.72,
        )

        self.assertEqual(set(paths), {"rear_axle", "left_wheel", "right_wheel"})
        self.assertEqual(paths["rear_axle"][0], (-0.33, 0.0, 0.03))
        self.assertEqual(paths["left_wheel"][0], (-0.33, 0.36, 0.03))
        self.assertEqual(paths["right_wheel"][0], (-0.33, -0.36, 0.03))
        self.assertAlmostEqual(paths["rear_axle"][-1][0], 0.67)
        self.assertAlmostEqual(paths["rear_axle"][-1][1], 0.0)
        self.assertAlmostEqual(paths["rear_axle"][-1][2], 0.03)

    def test_trajectory_model_sdf_contains_three_visible_paths(self):
        paths = {
            "rear_axle": [(0.0, 0.0, 0.18), (1.0, 0.0, 0.18)],
            "left_wheel": [(0.0, 0.3, 0.18), (1.0, 0.3, 0.18)],
            "right_wheel": [(0.0, -0.3, 0.18), (1.0, -0.3, 0.18)],
        }

        sdf = trajectory_model_sdf(paths)

        self.assertIn('name="trajectory_preview_model"', sdf)
        self.assertIn('name="rear_axle_0"', sdf)
        self.assertIn('name="left_wheel_0"', sdf)
        self.assertIn('name="right_wheel_0"', sdf)
        self.assertIn("<cylinder>", sdf)
        self.assertIn("<emissive>0.000 0.850 1.000 1</emissive>", sdf)

    def test_trajectory_visual_config_updates_existing_base_link_visual(self):
        text = trajectory_visual_config_text(
            "rear_axle",
            0,
            (-0.33, 0.0, 0.18),
            (0.0, 0.0, 0.18),
        )

        self.assertIn('name: "smart_wheelchair::base_link::trajectory_rear_axle_0"', text)
        self.assertIn('parent_name: "smart_wheelchair::base_link"', text)
        self.assertIn("type: VISUAL", text)
        self.assertIn("visible: true", text)
        self.assertIn("cylinder", text)

    def test_reads_named_gazebo_world_pose(self):
        text = """
pose {
  name: "wood_floor"
  position { x: 0 y: 0 z: 0.031 }
  orientation { w: 1 }
}
pose {
  name: "smart_wheelchair"
  position { x: -5.2 y: -0.3 z: 0.025 }
  orientation { z: 0.075 w: 0.997 }
}
"""

        pose = gazebo_model_pose_from_text(text, "smart_wheelchair")

        self.assertEqual(pose, (-5.2, -0.3, 0.025, 0.0, 0.0, 0.075, 0.997))


if __name__ == "__main__":
    unittest.main()
