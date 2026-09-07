import math
import re
from xml.sax.saxutils import escape


def predict_trajectory(
    linear_x_mps,
    angular_z_rps,
    start_x_m=0.0,
    start_y_m=0.0,
    start_yaw_rad=0.0,
    seconds=3.0,
    step=0.1,
    z_m=0.03,
):
    if abs(linear_x_mps) < 1e-4 and abs(angular_z_rps) < 1e-4:
        return []

    points = []
    steps = max(1, int(round(seconds / step)))
    for index in range(steps + 1):
        t = min(index * step, seconds)
        if abs(angular_z_rps) < 1e-6:
            local_x = linear_x_mps * t
            local_y = 0.0
        else:
            theta = angular_z_rps * t
            radius = linear_x_mps / angular_z_rps
            local_x = radius * math.sin(theta)
            local_y = radius * (1.0 - math.cos(theta))

        cos_yaw = math.cos(start_yaw_rad)
        sin_yaw = math.sin(start_yaw_rad)
        world_x = start_x_m + local_x * cos_yaw - local_y * sin_yaw
        world_y = start_y_m + local_x * sin_yaw + local_y * cos_yaw
        points.append((world_x, world_y, z_m))
    return points


def preview_paths(
    linear_x_mps,
    angular_z_rps,
    start_x_m=0.0,
    start_y_m=0.0,
    start_yaw_rad=0.0,
    seconds=3.0,
    step=0.1,
    z_m=0.03,
    rear_axle_x_m=-0.33,
    wheel_separation_m=0.72,
):
    if abs(linear_x_mps) < 1e-4 and abs(angular_z_rps) < 1e-4:
        return {"rear_axle": [], "left_wheel": [], "right_wheel": []}

    paths = {"rear_axle": [], "left_wheel": [], "right_wheel": []}
    offsets = {
        "rear_axle": (rear_axle_x_m, 0.0),
        "left_wheel": (rear_axle_x_m, wheel_separation_m / 2.0),
        "right_wheel": (rear_axle_x_m, -wheel_separation_m / 2.0),
    }
    steps = max(1, int(round(seconds / step)))
    for index in range(steps + 1):
        t = min(index * step, seconds)
        if abs(angular_z_rps) < 1e-6:
            base_x = linear_x_mps * t
            base_y = 0.0
            heading = 0.0
        else:
            heading = angular_z_rps * t
            radius = linear_x_mps / angular_z_rps
            base_x = radius * math.sin(heading)
            base_y = radius * (1.0 - math.cos(heading))

        world_yaw = start_yaw_rad + heading
        cos_start = math.cos(start_yaw_rad)
        sin_start = math.sin(start_yaw_rad)
        cos_world = math.cos(world_yaw)
        sin_world = math.sin(world_yaw)
        base_world_x = start_x_m + base_x * cos_start - base_y * sin_start
        base_world_y = start_y_m + base_x * sin_start + base_y * cos_start
        for name, (offset_x, offset_y) in offsets.items():
            x = base_world_x + offset_x * cos_world - offset_y * sin_world
            y = base_world_y + offset_x * sin_world + offset_y * cos_world
            paths[name].append((x, y, z_m))
    return paths


def gazebo_marker_text(
    points,
    red=0.0,
    green=0.85,
    blue=1.0,
    namespace="trajectory_preview",
    marker_id=1,
):
    point_text = " ".join(
        f"point {{x: {x:.3f} y: {y:.3f} z: {z:.3f}}}" for x, y, z in points
    )
    return (
        f'action: ADD_MODIFY ns: "{namespace}" id: {marker_id} '
        "type: LINE_STRIP visibility: ALL scale {x: 0.080 y: 0.080 z: 0.080} "
        f"material {{diffuse {{r: {red:.3f} g: {green:.3f} b: {blue:.3f} a: 1.000}} "
        f"ambient {{r: {red:.3f} g: {green:.3f} b: {blue:.3f} a: 1.000}}}} "
        f"{point_text}"
    )


def delete_marker_text(namespace="trajectory_preview", marker_id=1):
    return f'action: DELETE_MARKER ns: "{namespace}" id: {marker_id}'


def trajectory_model_sdf(paths, model_name="trajectory_preview_model", radius_m=0.035):
    colors = {
        "rear_axle": (0.0, 0.85, 1.0),
        "left_wheel": (1.0, 0.45, 0.0),
        "right_wheel": (0.60, 0.35, 1.0),
    }
    visuals = []
    for path_name in ("rear_axle", "left_wheel", "right_wheel"):
        points = paths.get(path_name, [])
        red, green, blue = colors[path_name]
        for index, (start, end) in enumerate(zip(points, points[1:])):
            x1, y1, z1 = start
            x2, y2, z2 = end
            dx = x2 - x1
            dy = y2 - y1
            dz = z2 - z1
            length = math.sqrt(dx * dx + dy * dy + dz * dz)
            if length < 1e-4:
                continue
            x = (x1 + x2) / 2.0
            y = (y1 + y2) / 2.0
            z = (z1 + z2) / 2.0
            yaw = math.atan2(dy, dx)
            pitch = math.atan2(math.sqrt(dx * dx + dy * dy), dz)
            visuals.append(
                f"""
      <visual name="{escape(path_name)}_{index}">
        <pose>{x:.3f} {y:.3f} {z:.3f} 0 {pitch:.6f} {yaw:.6f}</pose>
        <geometry>
          <cylinder>
            <radius>{radius_m:.3f}</radius>
            <length>{length:.3f}</length>
          </cylinder>
        </geometry>
        <material>
          <ambient>{red:.3f} {green:.3f} {blue:.3f} 1</ambient>
          <diffuse>{red:.3f} {green:.3f} {blue:.3f} 1</diffuse>
          <emissive>{red:.3f} {green:.3f} {blue:.3f} 1</emissive>
        </material>
        <cast_shadows>false</cast_shadows>
      </visual>"""
            )

    return f"""<sdf version="1.9">
  <model name="{escape(model_name)}">
    <static>true</static>
    <link name="trajectory_link">{''.join(visuals)}
    </link>
  </model>
</sdf>"""


def trajectory_visual_config_text(
    path_name,
    segment_index,
    start,
    end,
    parent_name="smart_wheelchair::base_link",
    radius_m=0.035,
):
    colors = {
        "rear_axle": (0.0, 0.85, 1.0),
        "left_wheel": (1.0, 0.45, 0.0),
        "right_wheel": (0.60, 0.35, 1.0),
    }
    red, green, blue = colors[path_name]
    x1, y1, z1 = start
    x2, y2, z2 = end
    dx = x2 - x1
    dy = y2 - y1
    dz = z2 - z1
    length = max(0.001, math.sqrt(dx * dx + dy * dy + dz * dz))
    x = (x1 + x2) / 2.0
    y = (y1 + y2) / 2.0
    z = (z1 + z2) / 2.0
    yaw = math.atan2(dy, dx)
    pitch = math.atan2(math.sqrt(dx * dx + dy * dy), dz)
    sin_pitch = math.sin(pitch / 2.0)
    cos_pitch = math.cos(pitch / 2.0)
    sin_yaw = math.sin(yaw / 2.0)
    cos_yaw = math.cos(yaw / 2.0)
    name = f"{parent_name}::trajectory_{path_name}_{segment_index}"
    return (
        f'name: "{name}" parent_name: "{parent_name}" type: VISUAL '
        "cast_shadows: false visible: true transparency: 0 "
        f"pose {{position {{x: {x:.3f} y: {y:.3f} z: {z:.3f}}} "
        f"orientation {{x: {-sin_pitch * sin_yaw:.9f} y: {sin_pitch * cos_yaw:.9f} "
        f"z: {cos_pitch * sin_yaw:.9f} w: {cos_pitch * cos_yaw:.9f}}}}} "
        f"geometry {{type: CYLINDER cylinder {{radius: {radius_m:.3f} length: {length:.3f}}}}} "
        f"material {{ambient {{r: {red:.3f} g: {green:.3f} b: {blue:.3f} a: 1}} "
        f"diffuse {{r: {red:.3f} g: {green:.3f} b: {blue:.3f} a: 1}} "
        f"emissive {{r: {red:.3f} g: {green:.3f} b: {blue:.3f} a: 1}}}}"
    )


def hide_trajectory_visual_config_text(
    path_name,
    segment_index,
    parent_name="smart_wheelchair::base_link",
):
    name = f"{parent_name}::trajectory_{path_name}_{segment_index}"
    return (
        f'name: "{name}" parent_name: "{parent_name}" type: VISUAL '
        "visible: false transparency: 1 "
        "pose {position {x: 0 y: 0 z: -10} orientation {w: 1}} "
        "geometry {type: CYLINDER cylinder {radius: 0.001 length: 0.001}}"
    )


def gazebo_model_pose_from_text(text, model_name):
    pose_blocks = re.finditer(r"pose\s*\{(?P<body>.*?)(?=\npose\s*\{|\Z)", text, re.S)
    for match in pose_blocks:
        body = match.group("body")
        name_match = re.search(r'name:\s*"([^"]+)"', body)
        if name_match is None or name_match.group(1) != model_name:
            continue
        return (
            _number_field(body, "x"),
            _number_field(body, "y"),
            _number_field(body, "z"),
            _number_field(body, "x", "orientation"),
            _number_field(body, "y", "orientation"),
            _number_field(body, "z", "orientation"),
            _number_field(body, "w", "orientation", default=1.0),
        )
    return None


def _number_field(text, field, section=None, default=0.0):
    source = text
    if section is not None:
        section_match = re.search(rf"{section}\s*\{{(?P<body>.*?)\}}", text, re.S)
        if section_match is None:
            return default
        source = section_match.group("body")
    match = re.search(rf"\b{field}:\s*([-+0-9.eE]+)", source)
    return default if match is None else float(match.group(1))


def yaw_from_quaternion(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)
