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
        return requested_speed

    nearest = min(valid_ranges)
    if nearest <= stop_distance_m:
        return 0.0
    if nearest >= slow_distance_m:
        return requested_speed

    scale = (nearest - stop_distance_m) / (slow_distance_m - stop_distance_m)
    return requested_speed * scale
