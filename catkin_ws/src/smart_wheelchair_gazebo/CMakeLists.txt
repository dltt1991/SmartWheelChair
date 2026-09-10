cmake_minimum_required(VERSION 3.0.2)
project(smart_wheelchair_gazebo)

find_package(catkin REQUIRED COMPONENTS gazebo_ros geometry_msgs roscpp)
find_package(gazebo REQUIRED)

catkin_package(LIBRARIES SmartWheelChairTrajectoryPreview SmartWheelChairGuiOnlyVisual
  CATKIN_DEPENDS gazebo_ros geometry_msgs roscpp)

include_directories(${catkin_INCLUDE_DIRS} ${GAZEBO_INCLUDE_DIRS})
link_directories(${GAZEBO_LIBRARY_DIRS})
add_library(SmartWheelChairTrajectoryPreview SHARED src/trajectory_preview_plugin.cc)
target_compile_features(SmartWheelChairTrajectoryPreview PRIVATE cxx_std_14)
target_link_libraries(SmartWheelChairTrajectoryPreview
  ${catkin_LIBRARIES} ${GAZEBO_LIBRARIES})
add_library(SmartWheelChairGuiOnlyVisual SHARED src/gui_only_visual_plugin.cc)
target_compile_features(SmartWheelChairGuiOnlyVisual PRIVATE cxx_std_14)
target_link_libraries(SmartWheelChairGuiOnlyVisual ${GAZEBO_LIBRARIES})

if(CATKIN_ENABLE_TESTING)
  catkin_add_nosetests(test)
  add_executable(trajectory_preview_render_check test/trajectory_preview_render_check.cc)
  target_compile_features(trajectory_preview_render_check PRIVATE cxx_std_14)
  target_link_libraries(trajectory_preview_render_check ${catkin_LIBRARIES} ${GAZEBO_LIBRARIES})
endif()

install(TARGETS SmartWheelChairTrajectoryPreview SmartWheelChairGuiOnlyVisual
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION})

install(DIRECTORY launch worlds models config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
