import math

import numpy as np


def _rollout(linear, angular, horizon=1., dt=.1):
    times = np.arange(0., horizon + dt / 2., dt)
    headings = angular * times
    if abs(angular) < 1e-9:
        x = linear * times
        y = np.zeros_like(times)
    else:
        radius = linear / angular
        x = radius * np.sin(headings)
        y = radius * (1. - np.cos(headings))
    return np.column_stack((x, y, headings))


def _trajectory_clear(trajectory, obstacles):
    if not len(obstacles):
        return True
    for x, y, heading in trajectory:
        cosine, sine = math.cos(heading), math.sin(heading)
        delta = obstacles - [x, y]
        local_x = cosine * delta[:, 0] + sine * delta[:, 1]
        local_y = -sine * delta[:, 0] + cosine * delta[:, 1]
        if np.any((local_x >= -.29) & (local_x <= 1.01)
                  & (np.abs(local_y) <= .44)):
            return False
    return True


def _path_score(trajectory, path):
    final = trajectory[-1]
    nearest = int(np.argmin(np.linalg.norm(path[:, :2] - final[:2], axis=1)))
    distance = np.linalg.norm(path[nearest, :2] - final[:2])
    heading = abs(math.atan2(math.sin(path[nearest, 2] - final[2]),
                             math.cos(path[nearest, 2] - final[2])))
    progress = nearest / max(1, len(path) - 1)
    return 4. * distance + 2. * heading - 2. * progress


def select_velocity(path, obstacles, speed_limit, previous_velocity=None):
    path = np.asarray(path, dtype=float)
    obstacles = np.asarray(obstacles, dtype=float)
    previous = (np.zeros(2) if previous_velocity is None
                else np.asarray(previous_velocity, dtype=float))
    if (path.ndim != 2 or path.shape[1:] != (3,) or not len(path)
            or obstacles.ndim != 2 or obstacles.shape[1:] != (2,)
            or not np.isfinite(path).all() or not np.isfinite(obstacles).all()
            or not np.isfinite(speed_limit) or speed_limit <= 0.):
        return np.zeros(2)
    candidates = [(v, w)
                  for v in np.linspace(0., min(float(speed_limit), .8), 9)
                  for w in np.linspace(-.65, .65, 15)]
    safe = [candidate for candidate in candidates
            if _trajectory_clear(_rollout(*candidate), obstacles)]
    if not safe:
        return np.zeros(2)
    return np.asarray(min(safe, key=lambda command:
                          _path_score(_rollout(*command), path)
                          + .05 * np.linalg.norm(np.asarray(command) - previous)))
