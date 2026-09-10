cmake_minimum_required(VERSION 3.0.2)
project(smart_wheelchair_gazebo)

find_package(catkin REQUIRED COMPONENTS smart_wheelchair_safety)

catkin_package(CATKIN_DEPENDS smart_wheelchair_safety)

install(DIRECTORY launch worlds models config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
