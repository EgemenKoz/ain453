#!/usr/bin/env python3
"""
assignment2.py – Entry point for the ArUco localisation node.

This file only bootstraps ROS and starts ArUcoLocalizationNode.
All implementation lives in the sibling modules:
  config.py            – shared constants
  calibration.py       – camera intrinsic loading
  odometry.py          – differential-drive wheel odometry
  aruco_detector.py    – ArUco detection + pose estimation
  pose_estimator.py    – camera-to-world pose transform
  visualizer.py        – combined camera + map visualisation
  localization_node.py – main DTROS node
"""

import os
import sys

# Ensure sibling modules in this directory are importable when run via rosrun
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import rospy  # noqa: E402  (must come after sys.path patch)
from localization_node import ArUcoLocalizationNode  # noqa: E402

if __name__ == "__main__":
    node = ArUcoLocalizationNode(node_name="aruco_localization_node")
    rospy.spin()
