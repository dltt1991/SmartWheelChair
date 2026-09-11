"""Conservative 3D self-occlusion check against visual AND collision geometry."""
import math
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

import numpy as np

MODEL = Path(__file__).parents[1] / 'models/smart_wheelchair/model.sdf'


def pose(element):
    pose_element = element.find('pose')
    if pose_element is not None and pose_element.attrib.get('relative_to'):
        raise ValueError('relative_to frames require explicit resolution in this check')
    x, y, z, r, p, a = map(float, element.findtext('pose', '0 0 0 0 0 0').split())
    cr, sr, cp, sp, ca, sa = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(a), math.sin(a)
    rotation = np.array([[ca*cp, ca*sp*sr-sa*cr, ca*sp*cr+sa*sr],
                         [sa*cp, sa*sp*sr+ca*cr, sa*sp*cr-ca*sr],
                         [-sp, cp*sr, cp*cr]])
    return np.array([x, y, z]), rotation


class LidarClearanceTest(unittest.TestCase):
    def test_selected_sectors_have_no_body_intersections(self):
        model = ET.parse(MODEL).getroot().find('model')
        base = model.find("link[@name='base_link']")
        np.testing.assert_allclose(pose(base)[0], np.zeros(3))
        np.testing.assert_allclose(pose(base)[1], np.eye(3))
        for side, lower, upper in [('left', -30, 170), ('right', -170, 30)]:
            sensor = base.find(f"sensor[@name='{side}_lidar']")
            origin, sensor_rotation = pose(sensor)
            angles = np.radians(np.linspace(lower, upper, 10001))
            directions = np.column_stack((np.cos(angles), np.sin(angles), np.zeros_like(angles))) @ sensor_rotation.T
            blocked = []
            for link in model.findall('link'):
                link_pos, link_rotation = pose(link)
                for shape in link.findall('visual') + link.findall('collision'):
                    if shape.attrib['name'] == f'{side}_lidar_body_visual':
                        continue  # A ray originates inside its own instrument housing.
                    geom = shape.find('geometry')
                    if geom.find('box') is not None:
                        half = np.fromstring(geom.findtext('box/size'), sep=' ') / 2
                    elif geom.find('cylinder') is not None:
                        radius = float(geom.findtext('cylinder/radius'))
                        half = np.array([radius, radius, float(geom.findtext('cylinder/length'))/2])
                    else:
                        self.assertIsNotNone(geom.find('sphere'), 'new geometry needs an occlusion check')
                        half = np.full(3, float(geom.findtext('sphere/radius')))
                    # Bound cylinders/spheres with enclosing oriented boxes.
                    # 2 mm inflation exceeds the <=0.873 mm deviation to the
                    # nearest sampled ray at 5 m for this 0.02-degree sweep.
                    half += .002
                    part_pos, part_rotation = pose(shape)
                    rotation = link_rotation @ part_rotation
                    center = link_pos + link_rotation @ part_pos
                    local_origin = rotation.T @ (origin-center)
                    local_directions = directions @ rotation
                    enter, leave = np.zeros(len(angles)), np.full(len(angles), 5.)
                    for axis in range(3):
                        d = local_directions[:, axis]
                        parallel = np.abs(d) < 1e-12
                        if abs(local_origin[axis]) > half[axis]:
                            leave[parallel] = -1.
                        safe_d = np.where(parallel, 1., d)
                        a = (-half[axis]-local_origin[axis])/safe_d
                        b = (half[axis]-local_origin[axis])/safe_d
                        enter = np.maximum(enter, np.where(parallel, -np.inf, np.minimum(a, b)))
                        leave = np.minimum(leave, np.where(parallel, np.inf, np.maximum(a, b)))
                    hits = enter <= leave
                    if np.any(hits):
                        blocked.append((link.attrib['name'], shape.attrib['name'],
                                        float(np.degrees(angles[hits][0])),
                                        float(np.degrees(angles[hits][-1]))))
            self.assertEqual(blocked, [], f'{side}: body blocks selected field')

    def test_housing_matches_sensor_and_stays_within_existing_width(self):
        model = ET.parse(MODEL).getroot().find('model')
        for side, y in [('left', .36), ('right', -.36)]:
            sensor = model.find(f".//sensor[@name='{side}_lidar']")
            housing = model.find(f".//visual[@name='{side}_lidar_body_visual']")
            np.testing.assert_allclose(pose(sensor)[0], [.46, y, .56])
            np.testing.assert_allclose(pose(sensor)[0], pose(housing)[0])
            half_y = float(housing.findtext('geometry/box/size').split()[1])/2
            self.assertLessEqual(abs(y)+half_y, .4+1e-12)


if __name__ == '__main__':
    unittest.main()
