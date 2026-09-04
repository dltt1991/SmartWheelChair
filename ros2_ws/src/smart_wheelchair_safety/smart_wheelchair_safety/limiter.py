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


def front_sector_ranges(
    ranges: Sequence[float],
    angle_min: float,
    angle_increment: float,
    half_width_rad: float,
) -> list[float]:
    return [
        value
        for index, value in enumerate(ranges)
        if abs(_normalize_angle(angle_min + index * angle_increment)) <= half_width_rad
    ]


def safety_ranges(
    ranges: Sequence[float],
    angle_min: float,
    angle_increment: float,
    half_width_rad: float,
    range_min: float,
    range_max: float,
    body_filter_distance_m: float,
) -> list[float]:
    return [
        value
        for value in front_sector_ranges(ranges, angle_min, angle_increment, half_width_rad)
        if range_min <= value <= range_max and value > body_filter_distance_m
    ]


def _normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))
