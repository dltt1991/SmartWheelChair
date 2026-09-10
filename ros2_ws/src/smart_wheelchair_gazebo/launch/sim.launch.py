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
            Node(package="nav2_controller", executable="controller_server",
                 name="controller_server", parameters=[nav_config],
                 remappings=[("cmd_vel", "cmd_vel_planned")], output="screen"),
            Node(package="nav2_lifecycle_manager", executable="lifecycle_manager",
                 name="lifecycle_manager_local", parameters=[nav_config],
                 output="screen"),
            Node(package="smart_wheelchair_safety", executable="unified_control_node",
                 parameters=[{"use_sim_time": True}],
                 output="screen"),
            web_joystick,
        ]
    )
