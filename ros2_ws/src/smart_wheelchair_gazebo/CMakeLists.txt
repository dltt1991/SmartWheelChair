cmake_minimum_required(VERSION 3.8)
project(smart_wheelchair_gazebo)

find_package(ament_cmake REQUIRED)
find_package(gz-cmake3 REQUIRED)
find_package(gz-math7 REQUIRED)
find_package(gz-msgs10 REQUIRED)
find_package(gz-plugin2 REQUIRED COMPONENTS register)
find_package(gz-sim8 REQUIRED)
find_package(gz-transport13 REQUIRED)

add_library(SmartWheelChairTrajectoryPreview SHARED
  src/trajectory_preview_system.cc
)
target_link_libraries(SmartWheelChairTrajectoryPreview
  gz-math7::gz-math7
  gz-msgs10::gz-msgs10
  gz-plugin2::register
  gz-sim8::gz-sim8
  gz-transport13::gz-transport13
)

install(TARGETS SmartWheelChairTrajectoryPreview
  LIBRARY DESTINATION lib
)

install(DIRECTORY launch worlds models config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
