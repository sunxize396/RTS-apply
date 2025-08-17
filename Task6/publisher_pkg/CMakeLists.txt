cmake_minimum_required(VERSION 3.0.2)
project(publisher_pkg)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  message_generation
)

add_message_files(FILES trebleints.msg)
generate_messages(DEPENDENCIES std_msgs)

catkin_package(
  CATKIN_DEPENDS message_runtime roscpp std_msgs
)

include_directories(${catkin_INCLUDE_DIRS})

add_executable(publisher_node src/publisher_node.cpp)
target_link_libraries(publisher_node ${catkin_LIBRARIES})
add_dependencies(publisher_node ${PROJECT_NAME}_generate_messages_cpp)
