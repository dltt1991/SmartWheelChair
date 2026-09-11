#!/usr/bin/env python3
"""Use the same recovery acceptance checks at all four outer room corners."""
import math
from pathlib import Path
import numpy as np
import check_recovery_matrix as matrix


def corner_cases(angles):
    for sx, sy in ((1, 1), (-1, 1), (-1, -1), (1, -1)):
        for degrees in angles:
            yaw = math.atan2(sy, sx)+math.radians(degrees)
            c, s = math.cos(yaw), math.sin(yaw)
            rotation = np.array([[c, -s], [s, c]])
            corners = np.array([[-.25, -.4], [-.25, .4], [.97, -.4], [.97, .4]]) @ rotation.T
            x = sx*(7.93-.02)-sx*np.max(sx*corners[:, 0])
            y = sy*(5.93-.02)-sy*np.max(sy*corners[:, 1])
            # +/-30 degrees to the corner bisector: both walls recede.
            away = -math.copysign(1., sx*c+sy*s)
            yield dict(name=f'corner-{sx}-{sy}-{degrees:+g}', pose=(x, y, yaw), away=away)


if __name__ == '__main__':
    # Keep the common trial driver and every acceptance predicate identical.
    matrix.main(case_factory=corner_cases, extra_sources=(Path(__file__),))
