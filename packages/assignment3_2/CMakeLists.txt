cmake_minimum_required(VERSION 2.8.3)
project(assignment3_2)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  nav_msgs
  duckietown_msgs
)

catkin_package()

catkin_install_python(
  PROGRAMS
    src/assignment3_2.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
