import base64
import math


def joystick_to_velocity(
    x,
    y,
    max_forward_linear,
    max_angular,
    deadzone=0.08,
    max_reverse_linear=None,
):
    x = _clamp(float(x), -1.0, 1.0)
    y = _clamp(float(y), -1.0, 1.0)
    if math.hypot(x, y) < deadzone:
        return 0.0, 0.0
    reverse_limit = max_forward_linear if max_reverse_linear is None else max_reverse_linear
    linear_limit = max_forward_linear if y >= 0.0 else reverse_limit
    return y * linear_limit, -x * max_angular


def _clamp(value, lower, upper):
    return min(max(value, lower), upper)


def camera_frame_payload(width, height, encoding, data):
    if encoding.lower() not in ("rgb8", "r8g8b8"):
        return {"width": 0, "height": 0, "encoding": "unsupported", "data": []}

    limit = min(len(data), width * height * 3)

    return {
        "width": width,
        "height": height,
        "encoding": "rgb8",
        "data": base64.b64encode(bytes(data[:limit])).decode("ascii"),
    }


def reverse_wheel_paths(
    linear_x_mps,
    angular_z_rps,
    wheel_width_m=0.72,
    max_length_m=2.0,
    step_m=0.1,
):
    if linear_x_mps >= -0.01:
        return {"left": [], "right": []}

    curvature = angular_z_rps / linear_x_mps if abs(linear_x_mps) > 1e-6 else 0.0
    return {
        "left": _reverse_wheel_path(curvature, wheel_width_m / 2.0, max_length_m, step_m),
        "right": _reverse_wheel_path(curvature, -wheel_width_m / 2.0, max_length_m, step_m),
    }


def _reverse_wheel_path(curvature, side_offset_m, max_length_m, step_m):
    points = []
    steps = max(1, int(max_length_m / step_m))
    for step in range(steps + 1):
        distance = min(step * step_m, max_length_m)
        s = -distance
        if abs(curvature) < 1e-6:
            x = s
            y = 0.0
            theta = 0.0
        else:
            theta = curvature * s
            x = math.sin(theta) / curvature
            y = (1.0 - math.cos(theta)) / curvature
        points.append((x - math.sin(theta) * side_offset_m, y + math.cos(theta) * side_offset_m))
    return points
