import math
from collections.abc import Sequence


def limit_forward_speed(
    requested_speed: float,
    ranges: Sequence[float],
    stop_distance_m: float,
    slow_distance_m: float,
) -> float:
    if slow_distance_m <= stop_distance_m:
        raise ValueError("slow_distance_m must be greater than stop_distance_m")

    if requested_speed <= 0.0:
        return requested_speed

    valid_ranges = [value for value in ranges if math.isfinite(value) and value > 0.0]
    if not valid_ranges:
        return 0.0

    nearest = min(valid_ranges)
    if nearest <= stop_distance_m:
        return 0.0
    if nearest >= slow_distance_m:
        return requested_speed

    scale = (nearest - stop_distance_m) / (slow_distance_m - stop_distance_m)
    return requested_speed * scale


def limit_forward_speed_with_scan_state(
    requested_speed: float,
    ranges: Sequence[float],
    scan_is_fresh: bool,
    stop_distance_m: float,
    slow_distance_m: float,
) -> float:
    if not scan_is_fresh:
        return limit_forward_speed(requested_speed, [], stop_distance_m, slow_distance_m)
    if requested_speed > 0.0 and not ranges:
        return requested_speed
    return limit_forward_speed(requested_speed, ranges, stop_distance_m, slow_distance_m)


def limit_forward_speed_for_arc(
    requested_speed: float,
    angular_speed: float,
    ranges: Sequence[float],
    scan_is_fresh: bool,
    stop_distance_m: float,
    slow_distance_m: float,
    arc_angular_threshold_rps: float,
) -> float:
    if requested_speed <= 0.0 or abs(angular_speed) < arc_angular_threshold_rps:
        return limit_forward_speed_with_scan_state(
            requested_speed,
            ranges,
            scan_is_fresh,
            stop_distance_m,
            slow_distance_m,
        )
    if not scan_is_fresh:
        return 0.0
    valid_ranges = [value for value in ranges if math.isfinite(value) and value > 0.0]
    if valid_ranges and min(valid_ranges) <= stop_distance_m:
        return 0.0
    return requested_speed


def limit_forward_speed_for_wall_follow(
    requested_speed: float,
    angular_speed: float,
    front_ranges: Sequence[float],
    body_ranges: Sequence[float],
    scan_is_fresh: bool,
    stop_distance_m: float,
    slow_distance_m: float,
    arc_angular_threshold_rps: float,
    min_body_clearance_m: float,
    slow_body_clearance_m: float,
    min_wall_follow_linear_mps: float,
) -> float:
    if slow_body_clearance_m <= min_body_clearance_m:
        raise ValueError("slow_body_clearance_m must be greater than min_body_clearance_m")
    if not scan_is_fresh:
        return 0.0
    arc_limited_speed = limit_forward_speed_for_arc(
        requested_speed,
        max(abs(angular_speed), arc_angular_threshold_rps),
        front_ranges,
        scan_is_fresh,
        stop_distance_m,
        slow_distance_m,
        arc_angular_threshold_rps,
    )
    if arc_limited_speed <= 0.0:
        return arc_limited_speed

    valid_body_ranges = [value for value in body_ranges if math.isfinite(value) and value > 0.0]
    if not valid_body_ranges:
        return arc_limited_speed

    nearest_body = min(valid_body_ranges)
    if nearest_body <= min_body_clearance_m:
        return 0.0
    if nearest_body >= slow_body_clearance_m:
        return arc_limited_speed

    scale = (nearest_body - min_body_clearance_m) / (
        slow_body_clearance_m - min_body_clearance_m
    )
    limited_speed = arc_limited_speed * scale
    return min(arc_limited_speed, max(min_wall_follow_linear_mps, limited_speed))


def limit_turn_speed_for_forward_arc(
    requested_speed: float,
    forward_speed: float,
    ranges: Sequence[float],
    scan_is_fresh: bool,
    stop_distance_m: float,
    slow_distance_m: float,
    arc_angular_threshold_rps: float,
) -> float:
    if forward_speed <= 0.0 or abs(requested_speed) < arc_angular_threshold_rps:
        return limit_speed_with_scan_state(
            requested_speed,
            ranges,
            scan_is_fresh,
            stop_distance_m,
            slow_distance_m,
        )
    if not scan_is_fresh:
        return 0.0
    valid_ranges = [value for value in ranges if math.isfinite(value) and value > 0.0]
    if valid_ranges and min(valid_ranges) <= stop_distance_m:
        return 0.0
    return requested_speed


def limit_speed_with_scan_state(
    requested_speed: float,
    ranges: Sequence[float],
    scan_is_fresh: bool,
    stop_distance_m: float,
    slow_distance_m: float,
) -> float:
    if slow_distance_m <= stop_distance_m:
        raise ValueError("slow_distance_m must be greater than stop_distance_m")
    if requested_speed == 0.0:
        return requested_speed
    if not scan_is_fresh:
        return 0.0
    valid_ranges = [value for value in ranges if math.isfinite(value) and value > 0.0]
    if not valid_ranges:
        return requested_speed

    nearest = min(valid_ranges)
    if nearest <= stop_distance_m:
        return 0.0
    if nearest >= slow_distance_m:
        return requested_speed

    scale = (nearest - stop_distance_m) / (slow_distance_m - stop_distance_m)
    return requested_speed * scale


def front_sector_ranges(
    ranges: Sequence[float],
    angle_min: float,
    angle_increment: float,
    half_width_rad: float,
) -> list[float]:
    return sector_ranges(ranges, angle_min, angle_increment, 0.0, half_width_rad)


def sector_ranges(
    ranges: Sequence[float],
    angle_min: float,
    angle_increment: float,
    center_angle_rad: float,
    half_width_rad: float,
) -> list[float]:
    return [
        value
        for index, value in enumerate(ranges)
        if abs(_normalize_angle(angle_min + index * angle_increment - center_angle_rad))
        <= half_width_rad
    ]


def safety_ranges_in_sector(
    ranges: Sequence[float],
    angle_min: float,
    angle_increment: float,
    center_angle_rad: float,
    half_width_rad: float,
    range_min: float,
    range_max: float,
    body_filter_distance_m: float,
) -> list[float]:
    return [
        value
        for value in sector_ranges(
            ranges,
            angle_min,
            angle_increment,
            center_angle_rad,
            half_width_rad,
        )
        if range_min <= value <= range_max and value > body_filter_distance_m
    ]


def safety_clearances_in_sector(
    ranges: Sequence[float],
    angle_min: float,
    angle_increment: float,
    center_angle_rad: float,
    half_width_rad: float,
    range_min: float,
    range_max: float,
    lidar_x_m: float,
    lidar_y_m: float,
    body_min_x_m: float,
    body_max_x_m: float,
    body_min_y_m: float,
    body_max_y_m: float,
    body_filter_margin_m: float,
) -> list[float]:
    clearances = []
    for index, scan_range in enumerate(ranges):
        angle = angle_min + index * angle_increment
        if abs(_normalize_angle(angle - center_angle_rad)) > half_width_rad:
            continue
        if not math.isfinite(scan_range) or scan_range < range_min or scan_range > range_max:
            continue

        body_exit = _ray_box_exit_distance(
            lidar_x_m,
            lidar_y_m,
            math.cos(angle),
            math.sin(angle),
            body_min_x_m,
            body_max_x_m,
            body_min_y_m,
            body_max_y_m,
        )
        clearance = scan_range - body_exit
        if clearance > body_filter_margin_m:
            clearances.append(clearance)
    return clearances


def safety_clearances_in_forward_corridor(
    ranges: Sequence[float],
    angle_min: float,
    angle_increment: float,
    range_min: float,
    range_max: float,
    lidar_x_m: float,
    lidar_y_m: float,
    body_max_x_m: float,
    body_min_y_m: float,
    body_max_y_m: float,
    corridor_half_width_m: float | None = None,
) -> list[float]:
    min_y = body_min_y_m
    max_y = body_max_y_m
    if corridor_half_width_m is not None:
        min_y = -corridor_half_width_m
        max_y = corridor_half_width_m
    clearances = []
    for index, scan_range in enumerate(ranges):
        if not math.isfinite(scan_range) or scan_range < range_min or scan_range > range_max:
            continue
        angle = angle_min + index * angle_increment
        x = lidar_x_m + scan_range * math.cos(angle)
        y = lidar_y_m + scan_range * math.sin(angle)
        if x > body_max_x_m and min_y <= y <= max_y:
            clearances.append(x - body_max_x_m)
    return clearances


def safety_ranges(
    ranges: Sequence[float],
    angle_min: float,
    angle_increment: float,
    half_width_rad: float,
    range_min: float,
    range_max: float,
    body_filter_distance_m: float,
) -> list[float]:
    return safety_ranges_in_sector(
        ranges,
        angle_min,
        angle_increment,
        0.0,
        half_width_rad,
        range_min,
        range_max,
        body_filter_distance_m,
    )


def _normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _ray_box_exit_distance(
    origin_x: float,
    origin_y: float,
    direction_x: float,
    direction_y: float,
    min_x: float,
    max_x: float,
    min_y: float,
    max_y: float,
) -> float:
    distances = []
    if direction_x > 0.0:
        distances.append((max_x - origin_x) / direction_x)
    elif direction_x < 0.0:
        distances.append((min_x - origin_x) / direction_x)

    if direction_y > 0.0:
        distances.append((max_y - origin_y) / direction_y)
    elif direction_y < 0.0:
        distances.append((min_y - origin_y) / direction_y)

    positive_distances = [distance for distance in distances if distance >= 0.0]
    return min(positive_distances) if positive_distances else 0.0
