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

    def _visual_pose(self, name):
        visual = self.root.find(f".//visual[@name='{name}']")
        self.assertIsNotNone(visual)
        return [float(value) for value in visual.findtext("pose").split()]


if __name__ == "__main__":
    unittest.main()
