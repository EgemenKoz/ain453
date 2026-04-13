"""
config.py – Shared constants for the A* pathfinding + ARTag navigation package.

Edit the sections below to match your lab setup before running.
"""

import cv2

# ── Heuristic ────────────────────────────────────────────────────────────────
# Controls which heuristic the A* algorithm uses.
# Options: "euclidean" | "manhattan"
HEURISTIC = "euclidean"

# ── Graph: Node coordinates (x, y) ──────────────────────────────────────────
# 4×4 grid maze. N0 (start) = (0,0), N15 (goal) = (3,3).
# x increases right, y increases up.
#
#   y=3:  N12  N13  N14  N15(GOAL)
#   y=2:  N8   N9   N10  N11
#   y=1:  N4   N5   N6   N7
#   y=0:  N0   N1   N2   N3
#         x=0  x=1  x=2  x=3
NODE_COORDS = {
     0: (0, 0),   1: (1, 0),   2: (2, 0),   3: (3, 0),
     4: (0, 1),   5: (1, 1),   6: (2, 1),   7: (3, 1),
     8: (0, 2),   9: (1, 2),  10: (2, 2),  11: (3, 2),
    12: (0, 3),  13: (1, 3),  14: (2, 3),  15: (3, 3),
}

START_NODE = 0
GOAL_NODE  = 15

# ── Graph: Edges with costs ──────────────────────────────────────────────────
# Each tuple (u, v, cost) represents a bidirectional edge.
# Edges not listed here are walls and must not be crossed.
EDGES = [
    ( 0,  1, 1.5),
    ( 0,  4, 2.0),
    ( 1,  2, 1.0),
    ( 1,  5, 2.0),
    ( 2,  3, 1.0),
    ( 2,  6, 1.5),
    ( 4,  8, 1.5),
    ( 5,  6, 1.0),
    ( 5,  9, 2.0),
    ( 6,  7, 0.5),
    ( 6, 10, 4.0),
    ( 7, 11, 1.5),
    ( 8,  9, 1.5),
    ( 8, 12, 2.0),
    (10, 11, 1.0),
    (10, 14, 1.5),
    (12, 13, 1.5),
    (13, 14, 2.0),
    (14, 15, 1.0),
]

# ── ArUco ────────────────────────────────────────────────────────────────────
# ArUco dictionary: must match the printed tags in the lab.
# ARTag ID i corresponds to node Ni (e.g. tag 0 → N0, tag 15 → N15).
ARUCO_DICT_TYPE  = cv2.aruco.DICT_APRILTAG_36h11
MARKER_SIZE_M    = 0.0675   # physical side length of the printed tags (metres)

# ── Navigation ───────────────────────────────────────────────────────────────
# Distance (metres) from the ARTag at which a node is considered "reached".
PROXIMITY_THRESHOLD_M = 0.30

# Maximum forward speed (m/s) sent to cmd_vel.
LINEAR_SPEED = 0.20

# Proportional gain for heading correction (rad/s per rad of error).
ANGULAR_GAIN = 1.5

# When |heading_error| exceeds this value (rad), rotate in place before
# moving forward.  Helps the robot orient toward the next tag first.
ALIGN_THRESHOLD_RAD = 0.25

# Forward speed is scaled by distance; this factor sets the ramp:
#   speed = min(LINEAR_SPEED, forward_distance * SLOWDOWN_FACTOR)
# Slowdown kicks in when forward_distance < LINEAR_SPEED / SLOWDOWN_FACTOR.
# With the values below: 0.20 / 0.40 = 0.50 m  → robot starts slowing at ~0.5 m.
# At PROXIMITY_THRESHOLD_M (0.30 m): speed = 0.30 * 0.40 = 0.12 m/s (gradual stop).
SLOWDOWN_FACTOR = 0.40

# Angular speed (rad/s) when rotating to search for a lost tag.
SEARCH_ANGULAR_SPEED = 0.30

# Number of consecutive frames without a target tag detection before the
# robot switches back to SEARCHING state.
TAG_LOST_PATIENCE_FRAMES = 5

# How often the control timer fires (Hz).  Camera callbacks drive the actual
# control; this timer only acts as a watchdog to stop the robot when no image
# has been received recently.
WATCHDOG_HZ = 10
