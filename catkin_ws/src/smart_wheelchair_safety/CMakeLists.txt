cmake_minimum_required(VERSION 3.0.2)
project(smart_wheelchair_safety)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs nav_msgs rospy sensor_msgs std_msgs tf2_ros
)

catkin_python_setup()
catkin_package(CATKIN_DEPENDS geometry_msgs nav_msgs rospy sensor_msgs std_msgs tf2_ros)

if(CATKIN_ENABLE_TESTING)
  find_package(rostest REQUIRED)
  catkin_add_nosetests(test/test_joystick.py)
  catkin_add_nosetests(test/test_unified_geometry.py)
endif()
