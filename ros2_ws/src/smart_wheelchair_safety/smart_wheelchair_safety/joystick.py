import math


def joystick_to_velocity(x, y, max_linear, max_angular, deadzone=0.08):
    x = _clamp(float(x), -1.0, 1.0)
    y = _clamp(float(y), -1.0, 1.0)
    if math.hypot(x, y) < deadzone:
        return 0.0, 0.0
    return y * max_linear, -x * max_angular


def _clamp(value, lower, upper):
    return min(max(value, lower), upper)
