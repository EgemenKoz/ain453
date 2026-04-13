cmake_minimum_required(VERSION 2.8.3)
project(assignment3)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  geometry_msgs
  cv_bridge
  duckietown_msgs
)

catkin_package()

catkin_install_python(
  PROGRAMS
    src/assignment3.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
