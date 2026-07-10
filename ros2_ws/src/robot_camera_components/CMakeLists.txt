cmake_minimum_required(VERSION 3.8)
project(robot_camera_components)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(sensor_msgs REQUIRED)

add_library(image_rate_gate_component SHARED
  src/image_rate_gate_node.cpp
)

target_compile_features(image_rate_gate_component PUBLIC cxx_std_17)

ament_target_dependencies(image_rate_gate_component
  rclcpp
  rclcpp_components
  sensor_msgs
)

rclcpp_components_register_nodes(
  image_rate_gate_component
  "robot_camera_components::ImageRateGateNode"
)

install(
  TARGETS image_rate_gate_component
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

ament_package()
