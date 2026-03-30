"""
config.py – Shared constants for the ArUco localisation package.

Edit TAG_POSES to match your actual lab / map layout before running.
"""

import math
import cv2

# ── ArUco ────────────────────────────────────────────────────────────────────

ARUCO_DICT_TYPE = cv2.aruco.DICT_6X6_250
MARKER_SIZE_M = 0.0675  # physical side length of the printed tags (metres)

# Known tag poses on the map: tag_id → (x_m, y_m, yaw_rad) in world frame.
# Update these to match your actual lab / map layout.
TAG_POSES = {
    0: (0.50, 0.00, 0.0),
    1: (1.00, 0.00, 0.0),
    2: (1.50, 0.00, 0.0),
    3: (0.50, 1.00, math.pi),
    4: (1.00, 1.00, math.pi),
    5: (1.50, 1.00, math.pi),
}

# ── Duckiebot differential-drive parameters ──────────────────────────────────

WHEEL_RADIUS_M = 0.0318     # DB21 wheel radius (metres)
WHEEL_BASELINE_M = 0.1      # distance between wheel contact points (metres)
TICKS_PER_REV = 135         # encoder ticks per full wheel revolution (DB21)

# ── Top-down map visualisation ───────────────────────────────────────────────

MAP_WIDTH_PX = 800          # full map canvas width  (pixels)
MAP_HEIGHT_PX = 600         # full map canvas height (pixels)
MAP_SCALE = 300             # pixels per metre
MAP_ORIGIN_X = 50           # pixel x-offset for world x = 0
MAP_ORIGIN_Y = 500          # pixel y-offset for world y = 0

VIS_PANEL_W = 400           # each sub-panel width in the published image
VIS_PANEL_H = 300           # each sub-panel height

PUBLISH_RATE_HZ = 30


def world_to_pixel(x: float, y: float) -> tuple:
    """Convert world coordinates (metres) to map canvas pixel coordinates."""
    px = int(MAP_ORIGIN_X + x * MAP_SCALE)
    py = int(MAP_ORIGIN_Y - y * MAP_SCALE)
    return px, py
