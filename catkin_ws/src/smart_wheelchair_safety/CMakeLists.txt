cmake_minimum_required(VERSION 3.0.2)
project(smart_wheelchair_safety)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs nav_msgs rospy sensor_msgs std_msgs tf2_ros
)

catkin_python_setup()
catkin_package(CATKIN_DEPENDS geometry_msgs nav_msgs rospy sensor_msgs std_msgs tf2_ros)

catkin_install_python(PROGRAMS
  nodes/local_path_follower_node
  nodes/web_joystick_node
  nodes/unified_control_node
  nodes/odom_frame_relay
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

if(CATKIN_ENABLE_TESTING)
  find_package(rostest REQUIRED)
  catkin_add_nosetests(test/test_joystick.py)
  catkin_add_nosetests(test/test_unified_geometry.py)
  catkin_add_nosetests(test/test_odom_frame_relay.py)
  add_rostest(test/web_and_follower.test)
  add_rostest(test/unified_control.test)
  add_rostest(test/safety_regression.test)
endif()
