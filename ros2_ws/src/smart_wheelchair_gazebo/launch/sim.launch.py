import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory("smart_wheelchair_gazebo")
    ros_gz_sim_share = get_package_share_directory("ros_gz_sim")
    world = os.path.join(pkg_share, "worlds", "m6_room.sdf")
    models = os.path.join(pkg_share, "models")

    gz_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ros_gz_sim_share, "launch", "gz_sim.launch.py")
        ),
        launch_arguments={"gz_args": f"-r -s {world}"}.items(),
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
            {"stop_distance_m": 0.45},
            {"slow_distance_m": 1.20},
        ],
        output="screen",
    )

    return LaunchDescription(
        [
            SetEnvironmentVariable("GZ_SIM_RESOURCE_PATH", models),
            gz_launch,
            bridge,
            safety_filter,
        ]
    )
