"""Local geometry and independent braking checks, in the rear-axle frame."""

from dataclasses import dataclass
import heapq
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

DOOR_MAX_DISTANCE = 4.5
DOOR_INTENT_HORIZON = 7.0
DOOR_MIN_FORWARD_NORMAL = .30


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


def extract_opening_lines(points):
    """Retain short collinear jamb pieces that stabilize aperture edges."""
    return extract_lines(points, min_length=.08)


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

    # A side passage at a T/cross intersection can be bounded by the end of the
    # followed wall and a perpendicular far wall, rather than a coplanar pair.
    if max_width >= 1.8:
        corner_min_width = max(min_width, 1.8)
        for i, wall in enumerate(lines):
            if abs(wall.heading) > .35 or not .45 <= abs(wall.distance) <= 1.8:
                continue
            tangent = np.array([math.cos(wall.heading), math.sin(wall.heading)])
            normal = np.array([-tangent[1], tangent[0]])
            wall_points = np.array([wall.start, wall.end])
            wall_end = wall_points[np.argmax(wall_points @ tangent)]
            wall_plane = float(np.mean(wall_points @ normal))
            for j, boundary in enumerate(lines):
                if i == j:
                    continue
                perpendicular_error = abs(
                    abs(angle_difference(boundary.heading, wall.heading)) - math.pi/2)
                if perpendicular_error > .15:
                    continue
                boundary_points = np.array([boundary.start, boundary.end])
                boundary_vector = boundary_points[1] - boundary_points[0]
                matrix = np.column_stack((tangent, -boundary_vector))
                determinant = float(np.linalg.det(matrix))
                if abs(determinant) < 1e-6:
                    continue
                along_wall, _ = np.linalg.solve(
                    matrix, boundary_points[0] - wall_points[0])
                intersection = wall_points[0] + along_wall * tangent
                boundary_fraction = np.clip(
                    (intersection-boundary_points[0]) @ boundary_vector
                    / (boundary_vector @ boundary_vector), 0., 1.)
                nearest_boundary = boundary_points[0] + boundary_fraction*boundary_vector
                if np.linalg.norm(intersection-nearest_boundary) > .08:
                    continue
                width = float((intersection-wall_end) @ tangent)
                if not corner_min_width <= width <= max_width:
                    continue
                center = wall_end + .5 * width * tangent
                if center[0] <= 0. or np.linalg.norm(center) > max_distance:
                    continue

                gap_start = float(wall_end @ tangent)
                gap_end = float(intersection @ tangent)
                occupied = False
                for k, third in enumerate(lines):
                    if k in (i, j) or abs(math.sin(wall.heading-third.heading)) > .06:
                        continue
                    third_points = np.array([third.start, third.end])
                    third_plane = float(np.mean(third_points @ normal))
                    if abs(wall_plane-third_plane) > .06:
                        continue
                    third_interval = sorted(third_points @ tangent)
                    overlap = (min(third_interval[1], gap_end-.05)
                               - max(third_interval[0], gap_start+.05))
                    if overlap > .08:
                        occupied = True
                        break
                if occupied:
                    continue

                crossing = math.copysign(1., wall.distance) * normal
                opening = Opening(tuple(center), math.atan2(crossing[1], crossing[0]),
                                  width, (wall, boundary))
                duplicate = any(
                    np.linalg.norm(np.asarray(existing.center)-center) <= .10
                    and abs(angle_difference(existing.heading, opening.heading)) <= .10
                    and abs(existing.width-width) <= .15
                    for existing in openings)
                if not duplicate:
                    openings.append(opening)
    return sorted(openings, key=lambda opening: np.linalg.norm(opening.center))


def find_door(lines, min_width=.92, max_width=1.50):
    doors = []
    for opening in find_openings(lines, min_width, max_width):
        center = np.asarray(opening.center)
        if (math.cos(opening.heading) > DOOR_MIN_FORWARD_NORMAL
                and .8 < center[0] < DOOR_MAX_DISTANCE and abs(center[1]) < 1.5):
            doors.append(opening)
    return min(doors, key=lambda door: np.linalg.norm(door.center), default=None)


def arc_path(v, w, duration=3., steps=41):
    t = np.linspace(0., duration, steps)
    yaw = w * t
    if abs(w) < 1e-6:
        return np.column_stack((v * t, np.zeros(steps), yaw))
    return np.column_stack((v / w * np.sin(yaw), v / w * (1 - np.cos(yaw)), yaw))


def intended_front_door(openings, v, w, corridor_tolerance=.25):
    matches = []
    for opening in openings:
        center = np.asarray(opening.center)
        if not (0.92 <= opening.width <= 1.5
                and math.cos(opening.heading) > DOOR_MIN_FORWARD_NORMAL
                and .8 < center[0] < DOOR_MAX_DISTANCE and abs(center[1]) < 1.5):
            continue
        duration = DOOR_INTENT_HORIZON if center[0] > 3.5 else 3.
        path = arc_path(max(v, .1), w, duration=duration,
                        steps=81 if duration > 3. else 41)[:, :2]
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
    """Return True for in-aperture traversal, False for departure, None if unknown.

    No acquisition range/progress filters or speed floor apply to an active
    door. An arc that cannot reach the plane, or starts beyond it, is
    inconclusive. This is intent only, not a footprint safety check.
    """
    normal = np.array([math.cos(door.heading), math.sin(door.heading)])
    tangent = np.array([-normal[1], normal[0]])
    relative = arc_path(v, w)[:, :2] - door.center
    across, along = relative @ normal, relative @ tangent
    hits = np.flatnonzero((across[:-1] < 0.) & (across[1:] >= 0.))
    if across[0] >= 0. or not len(hits):
        return None
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
            return bool(abs(exit_lateral) <= door.width/2)
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
    """A tangent-continuous reference; the local follower and the guard check feasibility."""
    end = np.asarray(target, dtype=float)
    t = np.linspace(0., 1., steps)[:, None]
    scale = min(length, float(np.linalg.norm(end)))
    start_tangent = np.array([scale, 0.])
    end_tangent = scale*np.array([math.cos(heading), math.sin(heading)])
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
    if phase == 'align':
        entry = approach_path(center + .3*normal, door.heading, length=2., steps=50)
        exit_xy = center + np.linspace(.35, 1.5, 25)[:, None]*normal
        path = np.vstack((entry, np.column_stack((exit_xy,
                                                  np.full(len(exit_xy), door.heading)))))
    elif phase == 'pass':
        target = center + 1.5*normal
        path = approach_path(target, door.heading, steps=50)
    else:
        raise ValueError(f'unknown door phase: {phase}')
    path[0, 2] = 0.
    path[-1, 2] = door.heading
    return path


def collision_aware_door_reference(door, obstacles, margin=.045):
    """Find a forward-only reference to the door centerline.

    This deterministic lattice only supplies the local follower with a collision-free
    geometric reference. The controller and braking guard remain responsible
    for dynamics and final command safety.
    """
    all_obstacles = np.asarray(obstacles, dtype=float).reshape(-1, 2)
    if not len(all_obstacles):
        return door_alignment_reference(door, 'align')
    _, unique = np.unique(np.floor(all_obstacles/.04).astype(int), axis=0, return_index=True)
    obstacles = all_obstacles[unique]

    center = np.asarray(door.center, dtype=float)
    normal = np.array([math.cos(door.heading), math.sin(door.heading)])
    tangent = np.array([-normal[1], normal[0]])
    setback = float(np.clip(.70 + .25*abs(door.heading), .70, 1.0))
    target = center - setback*normal
    step = .08
    curvatures = np.array([-1.5, -.75, 0., .75, 1.5])

    def advance(state, curvature, distance=step):
        x, y, yaw, _ = state
        middle = yaw + curvature*distance/2
        return (x + distance*math.cos(middle),
                y + distance*math.sin(middle),
                yaw + curvature*distance, curvature)

    def clearance(state, points=obstacles):
        x, y, yaw, _ = state
        px, py = points.T
        dx, dy = px-x, py-y
        c, s = math.cos(yaw), math.sin(yaw)
        local_x, local_y = c*dx+s*dy, -s*dx+c*dy
        outside_x = np.maximum(np.maximum(-.25-local_x, local_x-.97), 0.)
        outside_y = np.maximum(np.abs(local_y)-.40, 0.)
        return float(np.min(np.hypot(outside_x, outside_y)))

    def clear(state):
        return clearance(state) > margin

    def key(state):
        return (round(state[0]/step), round(state[1]/step),
                round(state[2]/.10))

    direct = door_alignment_reference(door, 'align')
    if all(clear((x, y, yaw, 0.)) for x, y, yaw in direct[1:]):
        return direct

    # A differential-drive chair can turn while stationary in a clear staging
    # area. Preserve those turns as zero-length segments for the controller;
    # checking only their endpoints would miss the front/rear corner sweep.
    across = float(-center @ normal)
    staging = center + min(across, -1.15)*normal
    bearing = math.atan2(staging[1], staging[0])
    route = [(0., 0., 0.)]
    checked = []

    def rotate_at(position, initial, final):
        delta = angle_difference(final, initial)
        for theta in np.linspace(initial, initial+delta, max(2, math.ceil(abs(delta)/.02)+1)):
            checked.append((*position, theta))
        route.append((*position, initial+delta))

    rotate_at((0., 0.), 0., bearing)
    for t in np.linspace(0., 1., max(2, math.ceil(np.linalg.norm(staging)/.025)+1))[1:]:
        route.append((*(t*staging), bearing))
        checked.append(route[-1])
    rotate_at(staging, bearing, door.heading)
    exit_point = center+1.2*normal
    for t in np.linspace(0., 1., max(2, math.ceil(np.linalg.norm(exit_point-staging)/.025)+1))[1:]:
        route.append((*(staging+t*(exit_point-staging)), door.heading))
        checked.append(route[-1])
    if all(clearance((x, y, theta, 0.), all_obstacles) > max(margin, .06)
           for x, y, theta in checked):
        return np.asarray(route)

    start = (0., 0., 0., 0.)
    start_key = key(start)
    states = {start_key: start}
    parents = {}
    costs = {start_key: 0.}
    queue = [(float(np.linalg.norm(target)), 0, start_key)]
    sequence = 0
    goal_key = None
    max_x = max(1.5, center[0]+.5)
    lateral_bound = max(2., abs(center[1])+.8)

    while queue and sequence < 7000:
        _, _, current_key = heapq.heappop(queue)
        current = states[current_key]
        relative = np.asarray(current[:2])-center
        across, lateral = relative@normal, relative@tangent
        if (-setback-.16 <= across <= -setback+.06 and abs(lateral) <= .06
                and abs(angle_difference(current[2], door.heading)) <= .12):
            goal_key = current_key
            break
        for curvature in curvatures:
            candidate = advance(current, curvature)
            midpoint = advance(current, curvature, step/2)
            if (candidate[0] < -.25 or candidate[0] > max_x
                    or abs(candidate[1]) > lateral_bound
                    or not clear(midpoint) or not clear(candidate)):
                continue
            candidate_key = key(candidate)
            gap = clearance(candidate)
            cost = (costs[current_key] + step + .006*abs(curvature)
                    + .08*max(0., .12-gap)/.08)
            if cost >= costs.get(candidate_key, math.inf):
                continue
            costs[candidate_key] = cost
            states[candidate_key] = candidate
            parents[candidate_key] = current_key
            heading_error = abs(angle_difference(candidate[2], door.heading))
            heuristic = float(np.linalg.norm(np.asarray(candidate[:2])-target) + .25*heading_error)
            sequence += 1
            heapq.heappush(queue, (cost+heuristic, sequence, candidate_key))

    if goal_key is None:
        return None
    route = []
    while True:
        route.append(states[goal_key])
        if goal_key == start_key:
            break
        goal_key = parents[goal_key]
    route.reverse()
    route = np.asarray(route)[:, :3]

    final = route[-1]
    lateral = float((final[:2]-center)@tangent)
    across = float((final[:2]-center)@normal)
    travel = np.arange(step, 1.5-across+step/2, step)
    line_across = across+travel
    blend_distance = max(.8, -.2-across)
    blend = np.clip(travel/blend_distance, 0., 1.)
    blend = blend*blend*(3.-2.*blend)
    line_xy = (center + line_across[:, None]*normal
               + (lateral*(1.-blend))[:, None]*tangent)
    delta = np.diff(np.vstack((final[:2], line_xy)), axis=0)
    line_heading = np.arctan2(delta[:, 1], delta[:, 0])
    line = np.column_stack((line_xy, line_heading))
    path = np.vstack((route, line))
    if any(clearance((x, y, yaw, 0.), all_obstacles) <= .04
           for x, y, yaw in path[1:]):
        return None
    return path


def _wall_in_requested_sweep(line, path, body_clearance):
    """Keep finite nearby walls, or walls the requested footprint approaches."""
    tangent = np.array([math.cos(line.heading), math.sin(line.heading)])
    normal = np.array([-tangent[1], tangent[0]])
    endpoints = np.array([line.start, line.end])
    along = endpoints @ tangent
    corners = np.array([[-.5, -.4], [-.5, .4], [.97, -.4], [.97, .4]])
    body_along = corners @ tangent
    if along.min() <= body_along.max() and along.max() >= body_along.min():
        return True

    # Separating-axis test between the finite segment and each inflated
    # requested footprint. A supporting line alone also spans the opening.
    c, s = np.cos(path[:, 2]), np.sin(path[:, 2])
    delta = endpoints[None, :, :] - path[:, None, :2]
    x = delta[:, :, 0]*c[:, None] + delta[:, :, 1]*s[:, None]
    y = -delta[:, :, 0]*s[:, None] + delta[:, :, 1]*c[:, None]
    overlap = ((x.min(axis=1) <= .97+body_clearance)
               & (x.max(axis=1) >= -.25-body_clearance)
               & (y.min(axis=1) <= .4+body_clearance)
               & (y.max(axis=1) >= -.4-body_clearance))
    center = path[:, :2] + .36*np.column_stack((c, s))
    extent = ((.61+body_clearance)*np.abs(normal[0]*c+normal[1]*s)
              + (.4+body_clearance)*np.abs(-normal[0]*s+normal[1]*c))
    return bool(np.any(overlap & (np.abs(line.distance-center @ normal) <= extent)))


def wall_reference(lines, v, w, body_clearance=.12, preferred_side=0):
    raw_path = arc_path(max(v, .1), w)
    candidates = [line for line in lines if abs(line.heading) < 1.30
                  and .45 < abs(line.distance) < 1.80]
    intended_side = math.copysign(1., w) if abs(w) >= .12 else preferred_side
    if intended_side:
        candidates = [line for line in candidates
                      if math.copysign(1., line.distance) == intended_side]
    side_lines = candidates
    candidates = [line for line in side_lines
                  if _wall_in_requested_sweep(line, raw_path, body_clearance)]

    # Inside a gap, a fast requested arc may reach the far jamb even though
    # no wall remains alongside the chair. Preserve the turn by reducing its
    # radius before considering a wall tangent that would cross the opening.
    if (candidates and abs(w) >= .12
            and not any(_wall_in_requested_sweep(line, raw_path[:1], body_clearance)
                        for line in candidates)):
        for scale in (.75, .5, .25):
            tighter = arc_path(max(v*scale, .1), w)
            if not any(_wall_in_requested_sweep(line, tighter, body_clearance)
                       for line in side_lines):
                return tighter, 'manual', 0

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
    return door_alignment_reference(door, 'align')


def footprint_clearance(points):
    """Signed distance to the axle-relative conservative rectangle."""
    points = np.asarray(points, dtype=float)
    dx = np.maximum(-.25-points[..., 0], points[..., 0]-.97)
    dy = np.abs(points[..., 1])-.40
    return np.hypot(np.maximum(dx, 0.), np.maximum(dy, 0.)) + np.minimum(np.maximum(dx, dy), 0.)


def _reachable_points(points, command, measured, margin, reaction, deceleration, state_age, dt):
    """Discard only points unreachable by the complete braking rollouts."""
    speeds = np.maximum(np.abs(command), np.abs(measured))
    transition = max((abs(command[0])+abs(measured[0]))/deceleration,
                     (abs(command[1])+abs(measured[1]))/.8)
    stop = max(abs(command[0])/deceleration, abs(command[1])/.8)
    # Total translation plus corner rotation bounds signed-distance change.
    # 1.05 encloses the rectangle; seven steps cover every rounded phase
    # and half-step sampling padding. Activity=1 bounds uncertainty from above.
    horizon = reaction+state_age+transition+stop+7*dt
    uncertainty = (deceleration+1.05*.8)*state_age*(reaction+transition+stop+.5*state_age)
    reach = margin+uncertainty+(speeds[0]+1.05*speeds[1])*horizon
    return points[footprint_clearance(points) <= reach+1e-9]


def braking_clear(points, command, measured, margin=.04, reaction=.20, deceleration=.5, state_age=0., recovery=False):
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
    if (not np.isfinite(command).all() or not np.isfinite(measured).all()
            or not np.isfinite([margin, reaction, deceleration, state_age]).all()
            or min(margin, reaction, state_age) < 0. or deceleration <= 0.):
        return False
    if recovery:
        if (np.any(np.abs(command) > np.array([.05, .1])+1e-9)
                or np.any(np.abs(measured) > [.055, .11]) or not 0. <= state_age <= .1):
            return False
    points = _reachable_points(points, command, measured, margin, reaction, deceleration, state_age,
                               .002 if recovery else .02)
    if not len(points):
        return True
    signed, padding, effective_margin = _braking_envelope(
        points, command, measured, margin, reaction, deceleration, state_age, recovery)
    clearance = np.maximum(signed, 0.)
    collision = clearance <= padding
    near_margin = (clearance[0] > effective_margin) & (clearance[0] <= padding)
    separating = near_margin & np.all(clearance[1:] >= clearance[0]-1e-9, axis=0)
    collision[:, separating] = clearance[:, separating] <= effective_margin
    if recovery:
        stopped, stop_padding, stop_margin = _braking_envelope(
            points, np.zeros(2), measured, margin, reaction, deceleration, state_age, True)
        # A point just outside the margin can already violate it during the
        # unavoidable measured-motion stop. Treat that nominal stop violation
        # like an initial violation, not like arbitrary uncertainty padding.
        near = stopped.min(axis=0) <= margin
        # Tiny measured drift can leave nominal stop clearance above the margin
        # while the complete stop certificate is already infeasible. Permit
        # only a strictly separating nominal recovery that resolves that deficit
        # at its endpoint. This does NOT restore a robust uncertainty guarantee
        # along the shared initial delay; never exempt a stop-safe new threat.
        stop_separating = ((stopped[0] > stop_margin) & (stopped[0] <= stop_padding)
                           & np.all(stopped[1:] >= stopped[0]-1e-9, axis=0))
        stop_limit = np.where(stop_separating, stop_margin, stop_padding)
        uncertified = ~near & np.any(stopped <= stop_limit, axis=0)
        if np.any(uncertified):
            if (np.any(signed[:, uncertified].min(axis=0) < margin)
                    or np.any(signed[-1, uncertified] <= padding)
                    or np.any(signed[-1, uncertified] <= stopped[-1, uncertified]+1e-9)):
                return False
        near |= uncertified
        if np.any(near):
            # Compare inevitable measured-motion drift with braking to zero,
            # not an arbitrary penetration allowance renewed every cycle.
            stopped = stopped[:, near]
            ideal, _, _ = _braking_envelope(
                points[near], command, np.zeros(2), margin, reaction, deceleration, 0., True)
            safe = (np.all(signed[:, near].min(axis=0) >= stopped.min(axis=0)-1e-9)
                    and np.all(signed[-1, near] >= stopped[-1]-1e-9)
                    and np.all(ideal >= ideal[0]-1e-9))
            if not safe:
                return False
            collision[:, near] = False
    return not bool(np.any(collision))


def _braking_envelope(points, command, measured, margin, reaction, deceleration, state_age, independent=False):
    """Signed clearances along the same envelope used by both guard policies."""
    stop_time = max(abs(command[0]) / deceleration, abs(command[1]) / .8)
    dt = .002 if independent else .02
    delay = np.tile(measured, (max(1, math.ceil((reaction+state_age) / dt)), 1))
    transition_time = max(abs(command[0]-measured[0]) / deceleration,
                          abs(command[1]-measured[1]) / .8)
    # The pose is at acquisition, not receipt. Delay includes its unreported
    # travel; padding bounds acceleration uncertainty accrued during that age
    # and its displacement through the remaining transition/braking horizon.
    activity = min(1., max(abs(measured[0])/.4, abs(measured[1])/.65))
    uncertainty_rate = activity*(deceleration + 1.05*.8)
    margin += uncertainty_rate * state_age * (reaction+transition_time+stop_time)
    margin += .5 * uncertainty_rate * state_age**2
    transition = np.linspace(measured, command, max(2, math.ceil(transition_time / dt) + 1))
    brake = np.linspace(command, [0., 0.], max(2, math.ceil(stop_time / dt) + 1))
    if independent:
        # At creep speed each axis settles independently: a microscopic yaw
        # estimate must not persist for the entire linear acceleration ramp.
        rates = np.array([deceleration, .8])
        t = np.arange(len(transition))[:, None]*dt
        transition = measured + np.sign(command-measured)*np.minimum(np.abs(command-measured), t*rates)
        t = np.arange(len(brake))[:, None]*dt
        brake = np.sign(command)*np.maximum(0., np.abs(command)-t*rates)
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
    return footprint_clearance(np.stack((local_x, local_y), axis=-1)), padding, margin
