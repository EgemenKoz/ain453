"""
visualizer.py – Builds and publishes the combined camera + map visualisation.

Published image layout (stacked vertically):
  ┌──────────────────────┐
  │  Camera feed         │  (with ArUco overlays drawn by aruco_detector)
  ├──────────────────────┤
  │  Top-down map        │  (tag positions + robot dot + heading arrow)
  └──────────────────────┘

Colour coding:
  Green  (0, 200,   0) – pose from ArUco measurement
  Orange (0, 100, 255) – pose from wheel-odometry fallback
"""

import math

import cv2
import numpy as np
import rospy
from sensor_msgs.msg import CompressedImage

from config import (
    TAG_POSES,
    MAP_WIDTH_PX, MAP_HEIGHT_PX,
    VIS_PANEL_W, VIS_PANEL_H,
    world_to_pixel,
)

_COLOUR_ARUCO = (0, 200, 0)
_COLOUR_ODOM = (0, 100, 255)


class Visualizer:
    def __init__(self, publisher: rospy.Publisher):
        self._pub = publisher
        self.last_frame = None

    def publish(
        self,
        camera_img,
        robot_x: float,
        robot_y: float,
        robot_theta: float,
        pose_source: str,
    ) -> None:
        """Compose the two panels and publish as a CompressedImage."""
        cam_panel = self._camera_panel(camera_img)
        map_panel = self._map_panel(robot_x, robot_y, robot_theta, pose_source)

        vis = np.vstack([cam_panel, map_panel])
        self.last_frame = vis

        msg = CompressedImage()
        msg.header.stamp = rospy.Time.now()
        msg.format = "jpeg"
        ok, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            msg.data = buf.tobytes()
            self._pub.publish(msg)

    # ── Panel builders ────────────────────────────────────────────────────────

    def _camera_panel(self, img) -> np.ndarray:
        if img is not None:
            return cv2.resize(img, (VIS_PANEL_W, VIS_PANEL_H))
        panel = np.zeros((VIS_PANEL_H, VIS_PANEL_W, 3), dtype=np.uint8)
        cv2.putText(
            panel, "Waiting for camera...", (20, VIS_PANEL_H // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2,
        )
        return panel

    def _map_panel(
        self, rx: float, ry: float, theta: float, source: str
    ) -> np.ndarray:
        panel = np.full((VIS_PANEL_H, VIS_PANEL_W, 3), 235, dtype=np.uint8)

        # Draw a simple grid (every 0.5 m)
        for gx in range(0, int(MAP_WIDTH_PX // (0.5 * 300)) + 1):
            world_x = gx * 0.5
            px, _ = world_to_pixel(world_x, 0.0)
            px_s = int(px * VIS_PANEL_W / MAP_WIDTH_PX)
            cv2.line(panel, (px_s, 0), (px_s, VIS_PANEL_H), (210, 210, 210), 1)
        for gy in range(0, int(MAP_HEIGHT_PX // (0.5 * 300)) + 1):
            world_y = gy * 0.5
            _, py = world_to_pixel(0.0, world_y)
            py_s = int(py * VIS_PANEL_H / MAP_HEIGHT_PX)
            cv2.line(panel, (0, py_s), (VIS_PANEL_W, py_s), (210, 210, 210), 1)

        # Draw known tags
        for tid, (tx, ty, tyaw) in TAG_POSES.items():
            px, py = world_to_pixel(tx, ty)
            px_s = int(px * VIS_PANEL_W / MAP_WIDTH_PX)
            py_s = int(py * VIS_PANEL_H / MAP_HEIGHT_PX)
            cv2.rectangle(
                panel, (px_s - 7, py_s - 7), (px_s + 7, py_s + 7),
                (30, 30, 30), -1,
            )
            cv2.putText(
                panel, str(tid), (px_s + 9, py_s + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (30, 30, 30), 1,
            )

        # Draw robot
        rpx, rpy = world_to_pixel(rx, ry)
        rpx_s = int(rpx * VIS_PANEL_W / MAP_WIDTH_PX)
        rpy_s = int(rpy * VIS_PANEL_H / MAP_HEIGHT_PX)

        colour = _COLOUR_ARUCO if source == "aruco" else _COLOUR_ODOM
        label = "ArUco" if source == "aruco" else "Odom"

        cv2.circle(panel, (rpx_s, rpy_s), 8, colour, -1)
        cv2.circle(panel, (rpx_s, rpy_s), 9, (0, 0, 0), 1)  # outline

        arrow_len = 22
        ax = int(rpx_s + arrow_len * math.cos(theta))
        ay = int(rpy_s - arrow_len * math.sin(theta))
        cv2.arrowedLine(panel, (rpx_s, rpy_s), (ax, ay), colour, 2, tipLength=0.35)

        # Pose text
        cv2.putText(
            panel,
            f"{label}  x={rx:.2f} y={ry:.2f} yaw={math.degrees(theta):.0f}°",
            (5, VIS_PANEL_H - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38, colour, 1,
        )

        # Legend
        cv2.circle(panel, (10, 12), 5, _COLOUR_ARUCO, -1)
        cv2.putText(panel, "ArUco", (20, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, 0, 0), 1)
        cv2.circle(panel, (78, 12), 5, _COLOUR_ODOM, -1)
        cv2.putText(panel, "Odom", (88, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, 0, 0), 1)

        return panel
