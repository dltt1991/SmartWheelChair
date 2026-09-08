import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory("smart_wheelchair_gazebo")
    ros_gz_sim_share = get_package_share_directory("ros_gz_sim")
    world = LaunchConfiguration("world")
    models = os.path.join(pkg_share, "models")
    plugins = os.path.join(os.path.dirname(os.path.dirname(pkg_share)), "lib")
    gui_config = os.path.join(pkg_share, "config", "top_down_gui.config")
    gui = LaunchConfiguration("gui")
    unified = LaunchConfiguration("unified_control")
    nav_config = os.path.join(pkg_share, "config", "unified_control.yaml")

    gz_server = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ros_gz_sim_share, "launch", "gz_sim.launch.py")
        ),
        launch_arguments={"gz_args": ["-r -s ", world]}.items(),
        condition=UnlessCondition(gui),
    )

    gz_gui = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ros_gz_sim_share, "launch", "gz_sim.launch.py")
        ),
        launch_arguments={"gz_args": ["-r ", world, " --gui-config ", gui_config]}.items(),
        condition=IfCondition(gui),
    )

    bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
            "/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist",
            "/cmd_vel_raw@geometry_msgs/msg/Twist]gz.msgs.Twist",
            "/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry",
            "/scan_left@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan",
            "/scan_right@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan",
            "/camera/rear/image@sensor_msgs/msg/Image[gz.msgs.Image",
            "/camera/rear/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo",
        ],
        output="screen",
    )

    common_safety_geometry = [
        {"scan_timeout_s": 2.0},
        {"body_min_x_m": -0.58},
        {"body_max_x_m": 0.64},
        {"body_min_y_m": -0.40},
        {"body_max_y_m": 0.40},
        {"body_filter_margin_m": 0.02},
    ]

    wall_follow_assist = Node(
        package="smart_wheelchair_safety",
        executable="wall_follow_assist_node",
        condition=UnlessCondition(unified),
        parameters=[
            *common_safety_geometry,
            {"target_wall_distance_m": 0.70},
            {"wall_follow_enter_distance_m": 1.20},
            {"min_follow_speed_mps": 0.10},
            {"max_follow_linear_mps": 0.60},
            {"max_follow_angular_rps": 0.30},
            {"wall_follow_hold_s": 0.60},
            {"wall_filter_alpha": 0.10},
            {"side_switch_margin_m": 0.30},
            {"distance_deadband_m": 0.20},
            {"heading_deadband_rad": 0.08},
            {"away_distance_margin_m": 0.30},
            {"max_angular_step_rps": 0.04},
            {"angular_deadband_rps": 0.08},
            {"lookahead_m": 1.20},
            {"k_distance": 0.25},
            {"k_heading": 1.0},
            {"min_away_angular_rps": 0.12},
        ],
        output="screen",
    )

    safety_filter = Node(
        package="smart_wheelchair_safety",
        executable="safety_filter_node",
        condition=UnlessCondition(unified),
        parameters=[
            *common_safety_geometry,
            {"input_topic": "cmd_vel_assisted"},
            {"stop_distance_m": 0.10},
            {"slow_distance_m": 0.90},
            {"body_sector_half_angle_rad": 3.14159265},
            {"arc_front_corridor_half_width_m": 0.40},
            {"wall_follow_min_body_clearance_m": 0.05},
            {"wall_follow_slow_body_clearance_m": 0.35},
            {"wall_follow_min_linear_mps": 0.12},
        ],
        output="screen",
    )

    web_joystick = Node(
        package="smart_wheelchair_safety",
        executable="web_joystick_node",
        parameters=[
            {"http_port": 8090},
            {"command_timeout_s": 1.0},
            {"max_forward_linear_mps": 1.6666667},
            {"max_reverse_linear_mps": 0.8333333},
            {"max_angular_rps": 1.4},
        ],
        output="screen",
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("unified_control", default_value="true"),
            DeclareLaunchArgument("world", default_value=os.path.join(pkg_share, "worlds", "m6_room.sdf")),
            DeclareLaunchArgument(
                "gui",
                default_value="false",
                description="Start Gazebo with GUI instead of headless server mode.",
            ),
            SetEnvironmentVariable("GZ_SIM_RESOURCE_PATH", models),
            SetEnvironmentVariable("GZ_SIM_SYSTEM_PLUGIN_PATH", plugins),
            gz_server,
            gz_gui,
            bridge,
            wall_follow_assist,
            safety_filter,
            Node(package="nav2_controller", executable="controller_server",
                 name="controller_server", parameters=[nav_config],
                 remappings=[("cmd_vel", "cmd_vel_planned")],
                 condition=IfCondition(unified), output="screen"),
            Node(package="nav2_lifecycle_manager", executable="lifecycle_manager",
                 name="lifecycle_manager_local", parameters=[nav_config],
                 condition=IfCondition(unified), output="screen"),
            Node(package="smart_wheelchair_safety", executable="unified_control_node",
                 parameters=[{"use_sim_time": True}],
                 condition=IfCondition(unified), output="screen"),
            web_joystick,
        ]
    )
