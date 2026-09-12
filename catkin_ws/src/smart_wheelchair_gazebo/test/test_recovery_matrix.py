import math
import unittest
from check_recovery_matrix import cases, motion_extent
from check_door_matrix import overlaps
from test_m6_accessibility import M6AccessibilityTest


class RecoveryMatrixTest(unittest.TestCase):
    def test_tight_starts_are_clear_but_within_four_mm(self):
        from check_tight_recovery import tight_cases
        scene = M6AccessibilityTest(); scene.setUp()
        matrix = list(tight_cases([0, 180]))
        self.assertEqual(len(matrix), 24)
        for case in matrix:
            with self.subTest(case=case['name']):
                self.assertIsNone(overlaps(case['pose'], scene.boxes))
                self.assertIsNotNone(overlaps(case['pose'], scene.boxes, .004))

    def test_all_24_corner_starts_are_non_overlapping(self):
        from check_corner_recovery import corner_cases
        scene = M6AccessibilityTest(); scene.setUp()
        matrix = list(corner_cases([-30, 0, 30, 150, 180, 210]))
        self.assertEqual(len(matrix), 24)
        for case in matrix:
            with self.subTest(case=case['name']):
                self.assertIsNone(overlaps(case['pose'], scene.boxes))

    def test_all_96_jamb_starts_are_non_overlapping(self):
        scene = M6AccessibilityTest(); scene.setUp()
        matrix = list(cases([-60, -30, 0, 30, 60, 150, 180, -150]))
        self.assertEqual(len(matrix), 96)
        self.assertEqual(len({c['name'] for c in matrix}), 96)
        for case in matrix:
            with self.subTest(case=case['name']):
                self.assertIsNone(overlaps(case['pose'], scene.boxes))

    def test_inward_excursion_cannot_be_hidden_by_returning_to_start(self):
        distance, yaw = motion_extent([(0., 0., 0.), (.1, 0., .3), (0., 0., 0.)], (0., 0., 0.))
        self.assertAlmostEqual(distance, .1)
        self.assertAlmostEqual(yaw, .3)
        self.assertAlmostEqual(motion_extent([(0., 0., -math.pi+.01)], (0., 0., math.pi-.01))[1], .02)
