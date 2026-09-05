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
    world = os.path.join(pkg_share, "worlds", "m6_room.sdf")
    models = os.path.join(pkg_share, "models")
    gui = LaunchConfiguration("gui")

    gz_server = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ros_gz_sim_share, "launch", "gz_sim.launch.py")
        ),
        launch_arguments={"gz_args": f"-r -s {world}"}.items(),
        condition=UnlessCondition(gui),
    )

    gz_gui = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ros_gz_sim_share, "launch", "gz_sim.launch.py")
        ),
        launch_arguments={"gz_args": f"-r {world}"}.items(),
        condition=IfCondition(gui),
    )

    bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            "/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist",
            "/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry",
            "/scan_left@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan",
            "/scan_right@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan",
            "/camera/rear/image@sensor_msgs/msg/Image[gz.msgs.Image",
            "/camera/rear/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo",
        ],
        output="screen",
    )

    safety_filter = Node(
        package="smart_wheelchair_safety",
        executable="safety_filter_node",
        parameters=[
            {"stop_distance_m": 0.10},
            {"slow_distance_m": 0.90},
            {"scan_timeout_s": 2.0},
            {"body_min_x_m": -0.58},
            {"body_max_x_m": 0.64},
            {"body_min_y_m": -0.40},
            {"body_max_y_m": 0.40},
            {"body_filter_margin_m": 0.02},
        ],
        output="screen",
    )

    web_joystick = Node(
        package="smart_wheelchair_safety",
        executable="web_joystick_node",
        parameters=[
            {"http_port": 8090},
            {"max_forward_linear_mps": 1.6666667},
            {"max_reverse_linear_mps": 0.8333333},
            {"max_angular_rps": 1.4},
        ],
        output="screen",
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "gui",
                default_value="false",
                description="Start Gazebo with GUI instead of headless server mode.",
            ),
            SetEnvironmentVariable("GZ_SIM_RESOURCE_PATH", models),
            gz_server,
            gz_gui,
            bridge,
            safety_filter,
            web_joystick,
        ]
    )
