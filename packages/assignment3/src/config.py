"""
config.py – Shared constants for the A* pathfinding + ARTag navigation package.

Edit the sections below to match your lab setup before running.
"""

import math
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
ARUCO_DICT_TYPE  = cv2.aruco.DICT_5X5_100
MARKER_SIZE_M    = 0.0675   # physical side length of the printed tags (metres)

# ── Navigation ───────────────────────────────────────────────────────────────
# Distance (metres) from the ARTag at which a node is considered "reached".
# Uses tvec[2] (forward/depth component only) so camera height is excluded.
# With 38-40 cm node spacing, 15 cm forward distance = robot is right on the tag.
PROXIMITY_THRESHOLD_M = 0.07

# Maximum forward speed (m/s) sent to cmd_vel.
LINEAR_SPEED = 0.26

# Proportional gain for heading correction (rad/s per rad of error).
ANGULAR_GAIN = 0.8

# When |heading_error| exceeds this value (rad), rotate in place before
# moving forward.  Helps the robot orient toward the next tag first.
ALIGN_THRESHOLD_RAD = 0.3

# Forward speed is scaled by distance; this factor sets the ramp:
#   speed = min(LINEAR_SPEED, forward_distance * SLOWDOWN_FACTOR)
SLOWDOWN_FACTOR = 0.80

# Camera lateral bias (metres). If the robot consistently drifts LEFT during
# approach, set a small positive value (e.g. 0.02). If it drifts RIGHT, use
# a small negative value (e.g. -0.02). Set 0.0 to disable.
CAMERA_LATERAL_BIAS = 0.0

# Angular speed (rad/s) when rotating to search for a lost tag.
# Must be slow enough that a floor tag stays in camera FOV for at least 2-3 frames.
# At 1-2 Hz camera: 1.0 rad/s → 57°/frame → tag disappears between frames.
# At 0.3 rad/s → ~17°/frame → tag stays visible long enough to detect.
SEARCH_ANGULAR_SPEED = 2.0

# Pulse-search: rotate for SEARCH_PULSE_ROTATE_SEC, then stop for
# SEARCH_PULSE_PAUSE_SEC so the camera gets a clear still frame.
# Set SEARCH_PULSE_PAUSE_SEC = 0.0 to disable.
SEARCH_PULSE_ROTATE_SEC = 0.7
SEARCH_PULSE_PAUSE_SEC  = 0.4

# Number of consecutive frames without a target tag detection before the
# robot switches back to SEARCHING state or enters BLIND_FORWARD.
TAG_LOST_PATIENCE_FRAMES = 15

# How often the control timer fires (Hz).  Camera callbacks drive the actual
# control; this timer only acts as a watchdog to stop the robot when no image
# has been received recently.
WATCHDOG_HZ = 10

# ── Blind forward (tag lost while close) ─────────────────────────────────────
# When the robot is APPROACHING and the tag disappears for
# TAG_LOST_PATIENCE_FRAMES while the last known distance was below
# BLIND_APPROACH_DIST, drive forward blindly for this many seconds before
# declaring the node as reached.
BLIND_FORWARD_SEC   = 0.20
BLIND_FORWARD_SPEED = 0.18

# If tag was lost while APPROACHING and last known distance (tvec[2]) was below
# this threshold, assume the tag went under the camera → enter blind forward.
# Floor tags disappear from view at ~20-25 cm forward distance.
BLIND_APPROACH_DIST = 0.30

# ── Heading / orientation ────────────────────────────────────────────────────
# Initial heading of the robot in the grid frame (radians).
# 0 = facing +x (right), π/2 = facing +y (up), etc.
# Robot starts at N0 facing +y (toward N4/N12 direction in the lab).
INITIAL_HEADING_RAD = math.pi / 2
