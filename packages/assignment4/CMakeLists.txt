cmake_minimum_required(VERSION 2.8.3)
project(assignment4)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  geometry_msgs
  nav_msgs
  visualization_msgs
  duckietown_msgs
)

catkin_package()

catkin_install_python(
  PROGRAMS
    src/assignment4.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
