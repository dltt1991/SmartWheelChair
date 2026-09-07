import math
from collections.abc import Sequence
from dataclasses import dataclass


Point = tuple[float, float]


@dataclass(frozen=True)
class WallLine:
    heading: float
    signed_distance: float
    closest_lateral_distance: float
    length: float


class WallFollowController:
    def __init__(
        self,
        wall_filter_alpha: float = 0.25,
        side_switch_margin_m: float = 0.30,
        distance_deadband_m: float = 0.06,
        heading_deadband_rad: float = 0.03,
        away_distance_margin_m: float = 0.12,
        max_angular_step_rps: float = 0.10,
        angular_deadband_rps: float = 0.05,
        lookahead_m: float = 1.0,
    ):
        self._wall_filter_alpha = wall_filter_alpha
        self._side_switch_margin_m = side_switch_margin_m
        self._distance_deadband_m = distance_deadband_m
        self._heading_deadband_rad = heading_deadband_rad
        self._away_distance_margin_m = away_distance_margin_m
        self._max_angular_step_rps = max_angular_step_rps
        self._angular_deadband_rps = angular_deadband_rps
        self._lookahead_m = lookahead_m
        self._follow_side = 0
        self._filtered_distance = None
        self._filtered_signed_distance = None
        self._filtered_heading = None
        self._last_angular_z = 0.0

    def assist(
        self,
        linear_x: float,
        angular_z: float,
        left_points: Sequence[Point],
        right_points: Sequence[Point],
        front_points: Sequence[Point] | None = None,
        target_wall_distance_m: float = 0.80,
        enter_distance_m: float = 1.20,
        min_follow_speed_mps: float = 0.10,
        max_follow_linear_mps: float = 0.45,
        max_follow_angular_rps: float = 0.80,
        k_distance: float = 1.0,
        k_heading: float = 1.2,
        min_away_angular_rps: float = 0.25,
    ) -> tuple[float, float, bool]:
        if (linear_x <= min_follow_speed_mps
                or self._follow_side * angular_z < -self._angular_deadband_rps):
            self.reset()
            return linear_x, angular_z, False

        wall = _nearest_wall(
            left_points,
            right_points,
            front_points or (),
            enter_distance_m,
            self._follow_side,
            self._side_switch_margin_m,
        )
        if wall is None:
            return linear_x, angular_z, False

        side = _wall_side(wall)
        if side * angular_z < -self._angular_deadband_rps:
            self.reset()
            return linear_x, angular_z, False
        if side != self._follow_side:
            self._filtered_distance = None
            self._filtered_signed_distance = None
            self._filtered_heading = None
        self._follow_side = side

        distance = self._smooth("_filtered_distance", wall.closest_lateral_distance)
        signed_distance = self._smooth("_filtered_signed_distance", wall.signed_distance)
        heading = self._smooth("_filtered_heading", wall.heading)
        follow_linear = min(linear_x, max_follow_linear_mps)
        # Tracking replaces the incoming turn; adding it would bias the wall
        # heading equilibrium whenever the joystick keeps pointing into the wall.
        desired_angular = _lookahead_wall_follow_angular(
            follow_linear,
            distance,
            signed_distance,
            heading,
            target_wall_distance_m,
            k_distance,
            k_heading,
            min_away_angular_rps,
            self._distance_deadband_m,
            self._away_distance_margin_m,
            self._lookahead_m,
        )
        if abs(desired_angular) > max_follow_angular_rps:
            follow_linear *= max_follow_angular_rps / abs(desired_angular)
            desired_angular = math.copysign(max_follow_angular_rps, desired_angular)
        # A small steering command can still be needed to remove a persistent
        # approach angle; suppress it only after the wall pose has converged.
        settled = (
            abs(heading) < self._heading_deadband_rad
            and abs(abs(signed_distance) - target_wall_distance_m) < self._distance_deadband_m
            and heading * signed_distance >= 0.0
        )
        if settled and abs(desired_angular) < self._angular_deadband_rps:
            corrected_angular = 0.0
        else:
            corrected_angular = _limit_step(
                desired_angular,
                self._last_angular_z,
                self._max_angular_step_rps,
            )
        corrected_angular = _clamp(corrected_angular, -max_follow_angular_rps, max_follow_angular_rps)
        self._last_angular_z = corrected_angular
        return follow_linear, corrected_angular, True

    @property
    def following(self) -> bool:
        return self._follow_side != 0

    def reset(self):
        self._follow_side = 0
        self._filtered_distance = None
        self._filtered_signed_distance = None
        self._filtered_heading = None
        self._last_angular_z = 0.0

    def _smooth(self, field_name: str, measured: float) -> float:
        previous = getattr(self, field_name)
        if previous is None:
            setattr(self, field_name, measured)
            return measured
        filtered = previous + self._wall_filter_alpha * (measured - previous)
        setattr(self, field_name, filtered)
        return filtered


def assist_wall_follow(
    linear_x: float,
    angular_z: float,
    left_points: Sequence[Point],
    right_points: Sequence[Point],
    front_points: Sequence[Point] | None = None,
    target_wall_distance_m: float = 0.80,
    enter_distance_m: float = 1.20,
    min_follow_speed_mps: float = 0.10,
    max_follow_linear_mps: float = 0.45,
    max_follow_angular_rps: float = 0.80,
    k_distance: float = 1.0,
    k_heading: float = 1.2,
    min_away_angular_rps: float = 0.25,
) -> tuple[float, float]:
    assisted_linear, assisted_angular, _ = assist_wall_follow_with_state(
        linear_x,
        angular_z,
        left_points,
        right_points,
        front_points,
        target_wall_distance_m,
        enter_distance_m,
        min_follow_speed_mps,
        max_follow_linear_mps,
        max_follow_angular_rps,
        k_distance,
        k_heading,
        min_away_angular_rps,
    )
    return assisted_linear, assisted_angular


def assist_wall_follow_with_state(
    linear_x: float,
    angular_z: float,
    left_points: Sequence[Point],
    right_points: Sequence[Point],
    front_points: Sequence[Point] | None = None,
    target_wall_distance_m: float = 0.80,
    enter_distance_m: float = 1.20,
    min_follow_speed_mps: float = 0.10,
    max_follow_linear_mps: float = 0.45,
    max_follow_angular_rps: float = 0.80,
    k_distance: float = 1.0,
    k_heading: float = 1.2,
    min_away_angular_rps: float = 0.25,
) -> tuple[float, float, bool]:
    if linear_x <= min_follow_speed_mps:
        return linear_x, angular_z, False

    wall = _nearest_wall(left_points, right_points, front_points or (), enter_distance_m)
    if wall is None:
        return linear_x, angular_z, False

    target_signed_distance = math.copysign(target_wall_distance_m, wall.closest_lateral_distance)
    distance_error = wall.closest_lateral_distance - target_signed_distance
    corrected_angular = _wall_follow_angular(
        angular_z,
        wall.closest_lateral_distance,
        wall.heading,
        target_wall_distance_m,
        k_distance,
        k_heading,
        min_away_angular_rps,
        distance_deadband_m=0.0,
        heading_deadband_rad=0.0,
        away_distance_margin_m=0.0,
    )

    return (
        min(linear_x, max_follow_linear_mps),
        _clamp(corrected_angular, -max_follow_angular_rps, max_follow_angular_rps),
        True,
    )


def scan_points_in_base(
    ranges: Sequence[float],
    angle_min: float,
    angle_increment: float,
    range_min: float,
    range_max: float,
    lidar_x_m: float,
    lidar_y_m: float,
    body_min_x_m: float,
    body_max_x_m: float,
    body_min_y_m: float,
    body_max_y_m: float,
    body_filter_margin_m: float,
    min_x_m: float = -0.20,
    max_x_m: float = 1.80,
) -> list[Point]:
    points = []
    for index, scan_range in enumerate(ranges):
        if not math.isfinite(scan_range) or scan_range < range_min or scan_range > range_max:
            continue
        angle = angle_min + index * angle_increment
        x = lidar_x_m + scan_range * math.cos(angle)
        y = lidar_y_m + scan_range * math.sin(angle)
        inside_body = (
            body_min_x_m - body_filter_margin_m <= x <= body_max_x_m + body_filter_margin_m
            and body_min_y_m - body_filter_margin_m <= y <= body_max_y_m + body_filter_margin_m
        )
        if inside_body or x < min_x_m or x > max_x_m:
            continue
        points.append((x, y))
    return points


def split_wall_points(points: Sequence[Point], min_lateral_m: float = 0.45) -> tuple[list[Point], list[Point]]:
    left = [(x, y) for x, y in points if y >= min_lateral_m]
    right = [(x, y) for x, y in points if y <= -min_lateral_m]
    return left, right


def _nearest_wall(
    left_points: Sequence[Point],
    right_points: Sequence[Point],
    front_points: Sequence[Point],
    enter_distance_m: float,
    preferred_side: int = 0,
    side_switch_margin_m: float = 0.0,
) -> WallLine | None:
    candidate_groups = [
        left_points,
        right_points,
        [point for point in front_points if point[0] > 0.0],
    ]
    candidates = [
        wall
        for wall in (_fit_wall_line(points) for points in candidate_groups)
        if wall is not None
        and abs(wall.closest_lateral_distance) <= enter_distance_m
        and _can_follow_wall_line(wall)
    ]
    if not candidates:
        return None
    preferred = [
        wall
        for wall in candidates
        if preferred_side and _wall_side(wall) == preferred_side
        and abs(wall.closest_lateral_distance) <= enter_distance_m + side_switch_margin_m
    ]
    if preferred:
        return min(preferred, key=lambda wall: abs(wall.closest_lateral_distance))
    return min(candidates, key=lambda wall: abs(wall.closest_lateral_distance))


def _fit_wall_line(points: Sequence[Point]) -> WallLine | None:
    if len(points) < 2:
        return None
    mean_x = sum(x for x, _ in points) / len(points)
    mean_y = sum(y for _, y in points) / len(points)
    centered = [(x - mean_x, y - mean_y) for x, y in points]
    cov_xx = sum(x * x for x, _ in centered)
    cov_xy = sum(x * y for x, y in centered)
    cov_yy = sum(y * y for _, y in centered)
    heading = 0.5 * math.atan2(2.0 * cov_xy, cov_xx - cov_yy)
    if math.cos(heading) < 0.0:
        heading = _normalize_angle(heading + math.pi)
    normal_x = -math.sin(heading)
    normal_y = math.cos(heading)
    signed_distance = mean_x * normal_x + mean_y * normal_y
    closest_lateral_distance = _closest_lateral_distance(points, signed_distance)
    projections = [x * math.cos(heading) + y * math.sin(heading) for x, y in points]
    return WallLine(
        heading,
        signed_distance,
        closest_lateral_distance,
        max(projections) - min(projections),
    )


def _closest_lateral_distance(points: Sequence[Point], signed_distance: float) -> float:
    if signed_distance >= 0.0:
        same_side = [y for _, y in points if y > 0.0]
        return min(same_side) if same_side else signed_distance
    same_side = [y for _, y in points if y < 0.0]
    return max(same_side) if same_side else signed_distance


def _can_follow_wall_line(wall: WallLine) -> bool:
    return wall.length >= 0.35 and abs(wall.heading) <= 1.35


def _wall_follow_angular(
    angular_z: float,
    wall_distance: float,
    wall_heading: float,
    target_wall_distance_m: float,
    k_distance: float,
    k_heading: float,
    min_away_angular_rps: float,
    distance_deadband_m: float,
    heading_deadband_rad: float,
    away_distance_margin_m: float,
) -> float:
    if abs(wall_heading) < heading_deadband_rad:
        wall_heading = 0.0

    distance_error = 0.0
    minimum_comfort_distance = max(
        0.0,
        target_wall_distance_m - away_distance_margin_m,
    )
    if abs(wall_distance) < minimum_comfort_distance:
        comfort_signed_distance = math.copysign(minimum_comfort_distance, wall_distance)
        distance_error = wall_distance - comfort_signed_distance
        if abs(distance_error) < distance_deadband_m:
            distance_error = 0.0

    distance_weight = max(0.0, 1.0 - abs(wall_heading) / 0.70)
    corrected_angular = angular_z + k_heading * wall_heading + k_distance * distance_error * distance_weight
    if abs(wall_distance) < minimum_comfort_distance:
        away_direction = -math.copysign(1.0, wall_distance)
        if corrected_angular * away_direction < min_away_angular_rps:
            corrected_angular = away_direction * min_away_angular_rps
    return corrected_angular


def _lookahead_wall_follow_angular(
    linear_x: float,
    wall_distance: float,
    wall_signed_distance: float,
    wall_heading: float,
    target_wall_distance_m: float,
    k_distance: float,
    k_heading: float,
    min_away_angular_rps: float,
    distance_deadband_m: float,
    away_distance_margin_m: float,
    lookahead_m: float,
) -> float:
    if lookahead_m <= 0.0:
        raise ValueError("lookahead_m must be positive")

    side = 1.0 if wall_distance >= 0.0 else -1.0
    normal_x = -math.sin(wall_heading)
    normal_y = math.cos(wall_heading)
    tangent_x = math.cos(wall_heading)
    tangent_y = math.sin(wall_heading)

    target_path_distance = wall_signed_distance - side * target_wall_distance_m
    target_x = tangent_x * lookahead_m + normal_x * target_path_distance
    target_y = tangent_y * lookahead_m + normal_y * target_path_distance
    lookahead_squared = max(target_x * target_x + target_y * target_y, 1e-6)
    pure_pursuit_angular = linear_x * (2.0 * target_y / lookahead_squared)

    minimum_comfort_distance = max(0.0, target_wall_distance_m - away_distance_margin_m)
    distance_bias = 0.0
    if abs(wall_distance) < minimum_comfort_distance:
        comfort_signed_distance = math.copysign(minimum_comfort_distance, wall_distance)
        distance_error = wall_distance - comfort_signed_distance
        if abs(distance_error) >= distance_deadband_m:
            distance_bias = k_distance * distance_error

    corrected_angular = k_heading * pure_pursuit_angular + distance_bias
    if abs(wall_distance) < minimum_comfort_distance:
        away_direction = -side
        if corrected_angular * away_direction < min_away_angular_rps:
            corrected_angular = away_direction * min_away_angular_rps
    return corrected_angular


def _wall_side(wall: WallLine) -> int:
    return 1 if wall.closest_lateral_distance >= 0.0 else -1


def _limit_step(value: float, previous: float, max_step: float) -> float:
    if max_step <= 0.0:
        return value
    return previous + _clamp(value - previous, -max_step, max_step)


def _normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
