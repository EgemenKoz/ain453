#!/usr/bin/env python3
"""
assignment4.py – Entry point for the A* + DWA navigation node.

All implementation lives in sibling modules:
  config_loader.py   – YAML → typed dataclasses
  astar_grid.py      – 8-connected grid A* (Task 1)
  costmap.py         – circular obstacle inflation (Task 3)
  dwa.py             – DWA local planner (Tasks 2 + 3)
  pose_source.py     – /odometry_node/odometry → world pose
  viz.py             – Renderer used by both offline and live demo
  assignment4_node.py – DTROS node tying it all together (Task 4 viz publish)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import rospy  # noqa: E402
from assignment4_node import Assignment4Node  # noqa: E402

if __name__ == "__main__":
    node = Assignment4Node(node_name="assignment4_node")
    rospy.spin()
