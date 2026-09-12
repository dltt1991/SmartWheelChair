#!/usr/bin/env python3
import unittest
import math
import time
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

from smart_wheelchair_safety.unified_geometry import Segment, arc_path, wall_reference
sys.path.insert(0, str(Path(__file__).parent))
import test_unified_node


class WallEndTest(unittest.TestCase):
    def test_turn_releases_far_collinear_wall_across_junction(self):
        for side in (-1, 1):
            with self.subTest(side=side):
                lines = [Segment((-3., side*.52), (-.6, side*.52), 0., side*.52),
                         Segment((2.8, side*.52), (5., side*.52), 0., side*.52)]
                path, mode, selected = wall_reference(lines, .6, side*.5,
                                                      preferred_side=side)
                self.assertEqual((mode, selected), ('manual', 0))
                np.testing.assert_allclose(path, arc_path(.6, side*.5))

    def test_near_jamb_remains_constrained_until_rear_clears(self):
        for side in (-1, 1):
            line = Segment((-3., side*.52), (0., side*.52), 0., side*.52)
            _, mode, selected = wall_reference([line], .6, side*.5)
            self.assertEqual((mode, selected), ('wall', side))

    def test_straight_input_stays_straight_after_wall_ends(self):
        for side in (-1, 1):
            lines = [Segment((-3., side*.8), (-.6, side*.8), 0., side*.8),
                     Segment((3.5, side*.8), (5., side*.8), 0., side*.8)]
            path, mode, selected = wall_reference(lines, .6, 0., preferred_side=side)
            self.assertEqual((mode, selected), ('manual', 0))
            np.testing.assert_allclose(path, arc_path(.6, 0.))

    def test_actual_wall_in_requested_arc_is_still_captured(self):
        for side in (-1, 1):
            line = Segment((.8, side*.8), (4., side*.8), 0., side*.8)
            _, mode, selected = wall_reference([line], .6, side*.5)
            self.assertEqual((mode, selected), ('wall', side))

    def test_far_jamb_uses_slower_tighter_joystick_arc(self):
        for side in (-1, 1):
            line = Segment((1.82, side*.52), (4., side*.52), 0., side*.52)
            path, mode, selected = wall_reference([line], .8, side*.63)
            self.assertEqual((mode, selected), ('manual', 0))
            np.testing.assert_allclose(path, arc_path(.6, side*.63))

    def test_full_forward_joystick_still_turns_through_gap(self):
        for side in (-1, 1):
            line = Segment((1.82, side*.52), (4., side*.52), 0., side*.52)
            path, mode, selected = wall_reference([line], 1.667, side*.63)
            self.assertEqual((mode, selected), ('manual', 0))
            np.testing.assert_allclose(path, arc_path(1.667*.25, side*.63))

    def test_tighter_arc_checks_wall_outside_original_arc(self):
        for side in (-1, 1):
            heading = side*1.2
            tangent = np.array([math.cos(heading), math.sin(heading)])
            normal = np.array([-tangent[1], tangent[0]])
            lines = [Segment((1.82, side*.52), (4., side*.52), 0., side*.52),
                     Segment(tuple(2*tangent+side*.63*normal),
                             tuple(3*tangent+side*.63*normal), heading, side*.63)]
            path, mode, selected = wall_reference(lines, .8, side*.63)
            self.assertEqual((mode, selected), ('manual', 0))
            np.testing.assert_allclose(path, arc_path(.4, side*.63))


@unittest.skipUnless(test_unified_node.ROS_AVAILABLE, 'requires ROS')
class WallEndNodeTest(unittest.TestCase):
    setUp = test_unified_node.UnifiedNodeTest.setUp
    tearDown = test_unified_node.UnifiedNodeTest.tearDown

    def test_unconfirmed_junction_releases_wall_reference_for_joystick_turn(self):
        for side in (-1, 1):
            with self.subTest(side=side):
                references = []
                self.node.reference = SimpleNamespace(publish=references.append)
                self.node.raw = np.array([.6, side*.5])
                self.node.wall_side = self.node.wall_preference = side
                self.node.mode = 'wall'
                points = np.array([(x, side*.52) for x in np.r_[
                    np.linspace(-3., -.6, 60), np.linspace(2.8, 5., 60)]])
                self.node.scans = {sensor: (time.monotonic(), points.copy())
                                   for sensor in ('left', 'right')}

                self.node.update_reference()

                self.assertEqual(self.node.mode, 'manual')
                self.assertEqual(self.node.wall_side, 0)
                self.assertIsNone(self.node.wall_preview_heading)
                published = np.array([(pose.pose.position.x, pose.pose.position.y)
                                      for pose in references[-1].poses])
                np.testing.assert_allclose(published, arc_path(.6, side*.5)[:, :2])


if __name__ == '__main__':
    import rostest
    rostest.rosrun('smart_wheelchair_safety', 'wall_end', __name__)
