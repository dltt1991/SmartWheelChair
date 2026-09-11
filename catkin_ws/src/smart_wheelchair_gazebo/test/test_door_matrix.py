import math
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import check_door_matrix as matrix
from check_door_matrix import inspect_sweep, overlaps, rear_extent


class DoorMatrixTest(unittest.TestCase):
    door = ('test', 0., 0., 1, 1.)
    normal = np.array([1., 0.])

    def test_all_84_map_starts_have_four_centimetres_of_body_clearance(self):
        scene = matrix.M6AccessibilityTest()
        scene.setUp()
        checked = 0
        for door in matrix.DOORS:
            for direction in ('in', 'out'):
                for angle in (0, -15, 15, -30, 30, -45, 45):
                    with self.subTest(door=door[0], direction=direction, angle=angle):
                        axle, yaw, _ = matrix.start_pose(door, direction, angle)
                        self.assertIsNone(overlaps((*axle, yaw), scene.boxes, .04))
                        checked += 1
        self.assertEqual(checked, 84)

    def test_crossing_another_gap_does_not_count_for_requested_door(self):
        _, crossed = inspect_sweep([(-1., 2., 0.), (1., 2., 0.)], {}, self.door, self.normal)
        self.assertFalse(crossed)
        _, crossed = inspect_sweep([(-1., .1, 0.), (1., .1, 0.)], {}, self.door, self.normal)
        self.assertTrue(crossed)
        _, crossed = inspect_sweep([(1., .1, 0.), (-1., .1, 0.)], {}, self.door, self.normal)
        self.assertFalse(crossed)

    def test_rear_projection_accounts_for_long_front_when_facing_backwards(self):
        self.assertAlmostEqual(rear_extent(0.), .25)
        self.assertAlmostEqual(rear_extent(math.pi), .97)
        self.assertAlmostEqual(rear_extent(math.pi/2), .4)
        self.assertGreater(rear_extent(math.radians(135)), .9)

    def test_translation_between_clear_samples_cannot_skip_wall(self):
        boxes = {'wall': (-.02, -2., .02, 2.)}
        poses = [(-1.1, 0., 0.), (.4, 0., 0.)]
        self.assertTrue(all(overlaps(p, boxes, .02) is None for p in poses))
        hit, _ = inspect_sweep(poses, boxes, self.door, self.normal)
        self.assertEqual(hit, 'wall')

    def test_rotation_between_clear_samples_checks_front_corner_sweep(self):
        boxes = {'post': (.66, .66, .69, .69)}
        poses = [(0., 0., 0.), (0., 0., math.pi/2)]
        self.assertTrue(all(overlaps(p, boxes, .02) is None for p in poses))
        hit, _ = inspect_sweep(poses, boxes, self.door, self.normal)
        self.assertEqual(hit, 'post')


if __name__ == '__main__':
    unittest.main()
