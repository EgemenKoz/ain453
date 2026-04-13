#!/usr/bin/env python3
"""
assignment3.py – Entry point for the A* pathfinding + ARTag navigation node.

All implementation lives in the sibling modules:
  config.py                – shared constants (graph, ArUco, navigation params)
  astar.py                 – A* algorithm (both Euclidean and Manhattan)
  calibration.py           – camera intrinsic loading
  aruco_detector.py        – ArUco detection + pose estimation
  navigation_controller.py – proportional visual-servo controller (Twist)
  navigation_node.py       – main DTROS node and state machine
"""

import os
import sys

# Ensure sibling modules in this directory are importable when run via rosrun.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import rospy  # noqa: E402
from navigation_node import PathNavigationNode  # noqa: E402

if __name__ == "__main__":
    node = PathNavigationNode(node_name="path_navigation_node")
    rospy.spin()
