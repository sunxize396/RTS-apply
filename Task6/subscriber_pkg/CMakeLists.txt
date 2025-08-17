cmake_minimum_required(VERSION 3.0.2)
project(subscriber_pkg)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  publisher_pkg  
)

catkin_package(
  CATKIN_DEPENDS roscpp publisher_pkg
)

include_directories(${catkin_INCLUDE_DIRS})

add_executable(subscriber_node src/subscriber_node.cpp)
target_link_libraries(subscriber_node ${catkin_LIBRARIES})
