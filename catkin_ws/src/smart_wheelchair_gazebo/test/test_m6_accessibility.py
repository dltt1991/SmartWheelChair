"""Constructive connectivity checks against actual SDF collision geometry.

Each room has an explicit path from the cross corridor and space to turn.
The rear-axle rectangle is inflated by 6 cm: 4 cm clearance plus 2 cm to
cover the 2 cm / 0.02 rad sweep sampling intervals. No point-robot flood fill.
"""

import math
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET


WORLD = Path(__file__).parents[1] / 'worlds' / 'm6_room.sdf'
DOORS = [
    ('nw_horizontal', -4.5, 1.35, 0, 1.0),
    ('ne_horizontal', 4.5, 1.35, 0, 1.1),
    ('sw_horizontal', -4.5, -1.35, 0, 1.2),
    ('se_horizontal', 4.5, -1.35, 0, 1.0),
    ('nw_vertical', -1.35, 3.5, 1, 1.1),
    ('se_vertical', 1.35, -3.5, 1, 1.2),
]


class M6AccessibilityTest(unittest.TestCase):
    def setUp(self):
        self.root = ET.parse(WORLD).getroot()
        self.boxes = {}
        for model in self.root.findall('./world/model'):
            collisions = model.findall('./link/collision')
            if not collisions:
                continue
            self.assertEqual(len(collisions), 1)
            box = model.find('./link/collision/geometry/box/size')
            self.assertIsNotNone(box, 'Extend the sweep check before adding non-box obstacles')
            sx, sy, sz = map(float, box.text.split())
            if model.attrib['name'] == 'floor':
                continue
            self.assertEqual(model.findtext('static'), 'true')
            x, y, _, roll, pitch, yaw = map(float, model.findtext('pose').split())
            self.assertEqual((roll, pitch, yaw), (0., 0., 0.))
            self.assertIsNone(model.find('./link/pose'))
            self.assertIsNone(model.find('./link/collision/pose'))
            self.boxes[model.attrib['name']] = (x-sx/2, y-sy/2, x+sx/2, y+sy/2)
            visual = list(map(float, model.findtext('./link/visual/geometry/box/size').split()))
            self.assertEqual(visual, [sx, sy, sz])

    def assert_clear(self, x, y, yaw):
        c, s = math.cos(yaw), math.sin(yaw)
        # Rear axle x=[-.25,.97], y=[-.40,.40], plus conservative margin.
        bx, by, hx, hy = x+.36*c, y+.36*s, .67, .46
        for name, (x0, y0, x1, y1) in self.boxes.items():
            dx, dy = (x0+x1)/2-bx, (y0+y1)/2-by
            ex, ey = (x1-x0)/2, (y1-y0)/2
            overlap = (abs(dx) <= ex+hx*abs(c)+hy*abs(s)
                       and abs(dy) <= ey+hx*abs(s)+hy*abs(c)
                       and abs(dx*c+dy*s) <= hx+ex*abs(c)+ey*abs(s)
                       and abs(-dx*s+dy*c) <= hy+ex*abs(s)+ey*abs(c))
            self.assertFalse(overlap, f'body hits {name} at axle {(x, y, yaw)}')

    def sweep(self, start, end, yaw):
        count = math.ceil(math.dist(start, end)/.02)
        for i in range(count+1):
            t = i/max(count, 1)
            self.assert_clear(start[0]+t*(end[0]-start[0]), start[1]+t*(end[1]-start[1]), yaw)

    def rotate(self, x, y):
        for i in range(316):
            self.assert_clear(x, y, i*2*math.pi/315)

    def test_six_actual_door_gaps_and_widths(self):
        self.assertEqual(len([name for name in self.boxes if name.startswith('door_')]), 12)
        for name, x, y, axis, width in DOORS:
            with self.subTest(door=name):
                first, second = (self.boxes[f'door_{name}_{suffix}'] for suffix in ('a', 'b'))
                self.assertAlmostEqual(second[axis]-first[axis+2], width)
                self.assertAlmostEqual((second[axis]+first[axis+2])/2, (x, y)[axis])
                for jamb in (first, second):
                    other = 1-axis
                    self.assertAlmostEqual((jamb[other]+jamb[other+2])/2, (x, y)[other])
                    self.assertAlmostEqual(jamb[other+2]-jamb[other], .12)

    def test_horizontal_and_vertical_long_corridors_in_both_directions(self):
        spawn = list(map(float, self.root.findtext('./world/include/pose').split()))
        self.assertEqual(spawn, [-6.7, 0., 0., 0., 0., 0.])
        # SDF spawn is the body origin, 0.33 m ahead of the rear axle.
        # Leave the west wall before a full turn; reverse back to park.
        self.sweep((spawn[0]-.33, 0.), (-6.7, 0.), 0.)
        self.sweep((-6.7, 0.), (spawn[0]-.33, 0.), 0.)
        self.sweep((-6.7, 0), (6.7, 0), 0.)
        self.sweep((6.7, 0), (-6.7, 0), math.pi)
        self.sweep((0, -4.7), (0, 4.7), math.pi/2)
        self.sweep((0, 4.7), (0, -4.7), -math.pi/2)
        self.rotate(0., 0.)
        self.rotate(-6.7, 0.)
        self.rotate(6.7, 0.)
        self.rotate(0., -4.7)
        self.rotate(0., 4.7)

    def test_every_room_has_full_body_entry_exit_and_turning_space(self):
        for name, x, y, axis, _ in DOORS:
            with self.subTest(door=name):
                if axis == 0:
                    sign = math.copysign(1., y)
                    entry, room, yaw = (x, 0.), (x, sign*3.), sign*math.pi/2
                    self.sweep((0., 0.), entry, 0. if x > 0 else math.pi)
                else:
                    sign = math.copysign(1., x)
                    entry, room, yaw = (0., y), (sign*3., y), 0. if x > 0 else math.pi
                    self.sweep((0., 0.), entry, math.copysign(math.pi/2, y))
                self.rotate(*entry)
                self.sweep(entry, room, yaw)
                self.rotate(*room)
                self.sweep(room, entry, yaw+math.pi)

    def test_two_entrance_rooms_have_internal_loop_connections(self):
        for start, end in [((-4.5, 3.), (-3., 3.5)), ((4.5, -3.), (3., -3.5))]:
            yaw = math.atan2(end[1]-start[1], end[0]-start[0])
            self.sweep(start, end, yaw)
            self.sweep(end, start, yaw+math.pi)


if __name__ == '__main__':
    unittest.main()
