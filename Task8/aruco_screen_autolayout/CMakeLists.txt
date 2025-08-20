cmake_minimum_required(VERSION 3.0.2)
project(aruco_screen_pkg)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  cv_bridge
  image_transport
)

catkin_package()

# 安装 Python 节点（自动布局版）
catkin_install_python(PROGRAMS
  scripts/aruco_screen_node_autolayout.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# 安装 launch 与 data
install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

install(DIRECTORY data/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/data
  FILES_MATCHING PATTERN "*.*"
)

