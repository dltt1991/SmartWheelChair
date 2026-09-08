"""Local geometry and independent braking checks, in the rear-axle frame."""

from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class Segment:
    start: tuple
    end: tuple
    heading: float
    distance: float


@dataclass(frozen=True)
class Opening:
    center: tuple
    heading: float
    width: float
    jambs: tuple = ()


Door = Opening


def transform_points(points, pose, inverse=False):
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    rotation = np.array([[c, -s], [s, c]])
    return (points - [x, y]) @ rotation if inverse else points @ rotation.T + [x, y]


def extract_lines(points, gap=.25, residual=.035, min_length=.30):
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(points) < 4:
        return []
    breaks = np.flatnonzero(np.linalg.norm(np.diff(points, axis=0), axis=1) > gap) + 1
    result = []

    def fit(group):
        if len(group) < 4:
            return
        center = group.mean(axis=0)
        _, _, vectors = np.linalg.svd(group - center, full_matrices=False)
        tangent = vectors[0]
        if tangent[0] < -1e-6 or (abs(tangent[0]) < 1e-6 and tangent[1] < 0):
            tangent = -tangent
        normal = np.array([-tangent[1], tangent[0]])
        errors = np.abs((group - center) @ normal)
        if errors.max() > residual:
            # Split ordered scans at the furthest departure from their chord.
            chord = group[-1] - group[0]
            chord_normal = np.array([-chord[1], chord[0]])
            deviations = np.abs((group - group[0]) @ chord_normal)
            split = int(np.argmax(deviations))
            if split < 3 or split > len(group) - 4:
                split = len(group) // 2
            fit(group[:split + 1])
            fit(group[split:])
            return
        along = (group - center) @ tangent
        if np.ptp(along) >= min_length:
            result.append(Segment(tuple(center + along.min() * tangent),
                                  tuple(center + along.max() * tangent),
                                  math.atan2(tangent[1], tangent[0]), float(center @ normal)))

    for group in np.split(points, breaks):
        fit(group)
    return result


def angle_difference(first, second):
    return math.atan2(math.sin(first-second), math.cos(first-second))


def opening_matches(first, second):
    return (np.linalg.norm(np.asarray(first.center)-second.center) <= .25
            and abs(angle_difference(first.heading, second.heading)) <= .12
            and abs(first.width-second.width) <= .25)


def find_openings(lines, min_width=.92, max_width=3.20, max_distance=4.5):
    openings = []
    for i, first in enumerate(lines):
        tangent = np.array([math.cos(first.heading), math.sin(first.heading)])
        left_normal = np.array([-tangent[1], tangent[0]])
        first_points = np.array([first.start, first.end])
        first_interval = sorted(first_points @ tangent)
        first_plane = float(np.mean(first_points @ left_normal))
        for j, second in enumerate(lines[i + 1:], start=i + 1):
            if abs(math.sin(first.heading - second.heading)) > .06:
                continue
            second_points = np.array([second.start, second.end])
            second_plane = float(np.mean(second_points @ left_normal))
            if abs(first_plane-second_plane) > .06:
                continue
            second_interval = sorted(second_points @ tangent)
            lower, upper = sorted(((first_interval, first_points),
                                   (second_interval, second_points)), key=lambda item: item[0][0])
            gap = upper[0][0] - lower[0][1]
            if not min_width <= gap <= max_width:
                continue
            gap_start, gap_end = lower[0][1], upper[0][0]
            occupied = False
            for k, third in enumerate(lines):
                if k in (i, j) or abs(math.sin(first.heading-third.heading)) > .06:
                    continue
                third_points = np.array([third.start, third.end])
                third_plane = float(np.mean(third_points @ left_normal))
                if abs(first_plane-third_plane) > .06:
                    continue
                third_interval = sorted(third_points @ tangent)
                overlap = (min(third_interval[1], gap_end-.05)
                           - max(third_interval[0], gap_start+.05))
                if overlap > .08:
                    occupied = True
                    break
            if occupied:
                continue
            lower_edge = lower[1][np.argmax(lower[1] @ tangent)]
            upper_edge = upper[1][np.argmin(upper[1] @ tangent)]
            center = (lower_edge + upper_edge) / 2
            if center[0] <= 0. or np.linalg.norm(center) > max_distance:
                continue
            crossing = math.copysign(1., (first_plane+second_plane)/2.) * left_normal
            openings.append(Opening(tuple(center), math.atan2(crossing[1], crossing[0]),
                                    float(gap), (first, second)))
    return sorted(openings, key=lambda opening: np.linalg.norm(opening.center))


def find_door(lines, min_width=.92, max_width=1.50):
    doors = []
    for opening in find_openings(lines, min_width, max_width):
        center = np.asarray(opening.center)
        if math.cos(opening.heading) > .65 and .8 < center[0] < 3.5 and abs(center[1]) < 1.5:
            doors.append(opening)
    return min(doors, key=lambda door: np.linalg.norm(door.center), default=None)


def arc_path(v, w, duration=3., steps=41):
    t = np.linspace(0., duration, steps)
    yaw = w * t
    if abs(w) < 1e-6:
        return np.column_stack((v * t, np.zeros(steps), yaw))
    return np.column_stack((v / w * np.sin(yaw), v / w * (1 - np.cos(yaw)), yaw))


def intended_front_door(openings, v, w, corridor_tolerance=.25):
    path = arc_path(max(v, .1), w)[:, :2]
    matches = []
    for opening in openings:
        center = np.asarray(opening.center)
        if not (0.92 <= opening.width <= 1.5 and math.cos(opening.heading) > .65
                and .8 < center[0] < 3.5 and abs(center[1]) < 1.5):
            continue
        normal = np.array([math.cos(opening.heading), math.sin(opening.heading)])
        tangent = np.array([-normal[1], normal[0]])
        relative = path-center
        across, along = relative@normal, relative@tangent
        closest = int(np.argmin(np.abs(across)))
        progress = abs(across[0])-abs(across[closest])
        miss = abs(along[closest])
        if progress >= .25 and miss <= opening.width/2+corridor_tolerance:
            matches.append(((miss, np.linalg.norm(center)), opening))
    return min(matches, key=lambda item: item[0], default=(None, None))[1]


def active_aperture_targeted(door, v, w):
    """Intent only: cross the aperture without turning out before rear clearance.

    No acquisition range/progress filters or speed floor apply to an active
    door. Failure to reach the plane is inconclusive; reference direction
    determines whether such a command is affirmative steering away.
    """
    normal = np.array([math.cos(door.heading), math.sin(door.heading)])
    tangent = np.array([-normal[1], normal[0]])
    relative = arc_path(v, w)[:, :2] - door.center
    across, along = relative @ normal, relative @ tangent
    hits = np.flatnonzero((across[:-1] < 0.) & (across[1:] >= 0.))
    if across[0] >= 0. or not len(hits):
        return False
    start = int(hits[0])
    fraction = -across[start] / (across[start+1]-across[start])
    entry = along[start] + fraction*(along[start+1]-along[start])
    if abs(entry) > door.width/2:
        return False
    for index in range(start+1, len(across)):
        # Match the existing rear-clear plane, not a new entry/safety margin.
        if across[index] >= .29:
            fraction = (.29-across[index-1]) / (across[index]-across[index-1])
            exit_lateral = along[index-1] + fraction*(along[index]-along[index-1])
            return abs(exit_lateral) <= door.width/2
        if across[index] < 0. or abs(along[index]) > door.width/2:
            return False
    return True


def intended_side_opening(openings, wall_side, v, w):
    if not wall_side or wall_side*w < .12:
        return None
    path = arc_path(max(v, .1), w)
    for opening in openings:
        normal = np.array([math.cos(opening.heading), math.sin(opening.heading)])
        if wall_side*normal[1] < .65 or opening.center[0] <= .4:
            continue
        if opening.width < 1.8:
            continue
        if opening.center[0] <= 4.5 and .35 < wall_side*opening.center[1] < 1.8:
            return opening
        tangent = np.array([-normal[1], normal[0]])
        relative = path[:, :2] - opening.center
        across, along = relative @ normal, relative @ tangent
        hits = np.flatnonzero((across[:-1] < 0.) & (across[1:] >= 0.))
        if any(abs(along[index+1]) <= opening.width/2-.40 for index in hits):
            return opening
    return None


def approach_path(target, heading, length=3., steps=41):
    """A tangent-continuous reference; MPPI and the guard check feasibility."""
    end = np.asarray(target, dtype=float)
    t = np.linspace(0., 1., steps)[:, None]
    scale = min(length, float(np.linalg.norm(end)))
    start_tangent = np.array([scale, 0.])
    end_tangent = scale * np.array([math.cos(heading), math.sin(heading)])
    xy = (t**3 - 2*t**2 + t) * start_tangent + (-2*t**3 + 3*t**2) * end + (t**3 - t**2) * end_tangent
    delta = np.gradient(xy, axis=0)
    return np.column_stack((xy, np.arctan2(delta[:, 1], delta[:, 0])))


def door_entry_clearance(door):
    normal = np.array([math.cos(door.heading), math.sin(door.heading)])
    tangent = np.array([-normal[1], normal[0]])
    lateral_error = float(np.asarray(door.center) @ tangent)
    projected = .40*abs(math.cos(door.heading)) + .97*abs(math.sin(door.heading))
    return door.width/2 - .04 - projected - abs(lateral_error)


def door_alignment_reference(door, phase):
    center = np.asarray(door.center, dtype=float)
    normal = np.array([math.cos(door.heading), math.sin(door.heading)])
    tangent = np.array([-normal[1], normal[0]])
    if phase == 'align':
        lateral_error = float(center @ tangent)
        heading_error = angle_difference(door.heading, 0.)
        setback = np.clip(.9 + .6*abs(lateral_error) + .6*abs(heading_error), .9, 1.8)
        target = center - setback*normal
        path = approach_path(target, door.heading)
    elif phase == 'pass':
        target = center + 1.5*normal
        path = approach_path(target, door.heading, steps=50)
    else:
        raise ValueError(f'unknown door phase: {phase}')
    path[0, 2] = 0.
    path[-1, 2] = door.heading
    return path


def wall_reference(lines, v, w, body_clearance=.12):
    raw_path = arc_path(max(v, .1), w)
    candidates = [line for line in lines if abs(line.heading) < 1.30
                  and .45 < abs(line.distance) < 1.80
                  and max(line.start[0], line.end[0]) > .5
                  and math.copysign(1., line.distance) * w >= -.15]
    def intended_clearance(line):
        normal = np.array([-math.sin(line.heading), math.cos(line.heading)])
        return float(np.min(math.copysign(1., line.distance) * (line.distance-raw_path[:, :2] @ normal)))
    wall = min(candidates, key=intended_clearance, default=None)
    if wall is None:
        return raw_path, 'manual', 0
    tangent = np.array([math.cos(wall.heading), math.sin(wall.heading)])
    normal = np.array([-tangent[1], tangent[0]])
    offset = wall.distance - math.copysign(.40+body_clearance, wall.distance)
    return approach_path(3. * tangent + offset * normal, wall.heading), 'wall', math.copysign(1., wall.distance)


def door_reference(door):
    normal = np.array([math.cos(door.heading), math.sin(door.heading)])
    center = np.array(door.center)
    pre = center - .9 * normal
    if pre[0] > .25:
        entry = approach_path(pre, door.heading, steps=25)
        exit_xy = pre + np.linspace(.05, 2.4, 35)[:, None] * normal
    else:
        return approach_path(center + 1.5 * normal, door.heading, steps=50)
    return np.vstack((entry, np.column_stack((exit_xy, np.full(len(exit_xy), door.heading)))))


def braking_clear(points, command, measured, margin=.04, reaction=.20, deceleration=.5, state_age=0.):
    """Check current motion, command transition and proportional braking.

    The axle-relative footprint is x=[-.25,.97], y=[-.40,.40]. Sampling
    inflation bounds corner motion between samples; point density and sensor
    uncertainty still need to be covered by the configured margin.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    if not len(points):
        return True
    command = np.asarray(command, dtype=float)
    measured = np.asarray(measured, dtype=float)
    stop_time = max(abs(command[0]) / deceleration, abs(command[1]) / .8)
    dt = .02
    delay = np.tile(measured, (max(1, math.ceil((reaction+state_age) / dt)), 1))
    transition_time = max(abs(command[0]-measured[0]) / deceleration,
                          abs(command[1]-measured[1]) / .8)
    # The pose is at acquisition, not receipt. Delay includes its unreported
    # travel; padding bounds acceleration uncertainty accrued during that age
    # and its displacement through the remaining transition/braking horizon.
    uncertainty_rate = deceleration + 1.05*.8
    margin += uncertainty_rate * state_age * (reaction+transition_time+stop_time)
    margin += .5 * uncertainty_rate * state_age**2
    transition = np.linspace(measured, command, max(2, math.ceil(transition_time / dt) + 1))
    brake = np.linspace(command, [0., 0.], max(2, math.ceil(stop_time / dt) + 1))
    velocities = np.vstack((delay, transition, brake))
    poses = [(0., 0., 0.)]
    x = y = yaw = 0.
    for v, w in velocities:
        x += v * math.cos(yaw + w * dt / 2) * dt
        y += v * math.sin(yaw + w * dt / 2) * dt
        yaw += w * dt
        poses.append((x, y, yaw))
    poses = np.asarray(poses)
    padding = margin + dt * .5 * np.max(np.abs(velocities[:, 0]) + 1.05 * np.abs(velocities[:, 1]))
    dx = points[None, :, 0] - poses[:, None, 0]
    dy = points[None, :, 1] - poses[:, None, 1]
    c, s = np.cos(poses[:, None, 2]), np.sin(poses[:, None, 2])
    local_x, local_y = c * dx + s * dy, -s * dx + c * dy
    collision = ((local_x >= -.25 - padding) & (local_x <= .97 + padding)
                 & (np.abs(local_y) <= .40 + padding))
    return not bool(np.any(collision))
