"""
pose_estimator.py – Compute world-frame robot pose from an ArUco detection.

Coordinate conventions
----------------------
Marker frame   : x-right, y-up, z-pointing out of the tag face
Camera frame   : x-right, y-down, z-forward (into the scene)
Robot/world    : standard ROS 2-D map frame (x-forward, y-left, z-up)

The Duckiebot camera is mounted looking forward, so the camera +Z axis aligns
roughly with the robot's +X axis.
"""

import math

import cv2
import numpy as np

from config import TAG_POSES


class PoseEstimator:
    """Static helper – converts a tag detection to a world-frame robot pose."""

    @staticmethod
    def from_tag(tag_id: int, rvec: np.ndarray, tvec: np.ndarray) -> tuple:
        """
        Return the estimated world-frame pose of the robot.

        Parameters
        ----------
        tag_id : int   – ID of the detected marker (must be in TAG_POSES)
        rvec   : (3,)  – Rodrigues rotation vector (camera ← marker)
        tvec   : (3,)  – translation vector in camera frame (metres)

        Returns
        -------
        (x, y, theta) in world frame (metres, metres, radians)
        """
        tag_x, tag_y, tag_yaw = TAG_POSES[tag_id]

        # Rotation matrix  R_cm : rotates vectors from marker frame → camera frame
        R_cm, _ = cv2.Rodrigues(rvec)

        # Camera centre expressed in marker frame:  p = -R_cm^T · t
        cam_in_marker = (-R_cm.T @ tvec).flatten()

        # Heading of the camera relative to the tag.
        # The Duckiebot's forward direction is camera +Z.
        # In the marker frame the camera looks along R_cm[:, 2] (third column of R_cm).
        # We extract the in-plane (x-z of marker frame) yaw.
        yaw_cam_to_marker = math.atan2(-R_cm[2, 0], R_cm[2, 2])

        # Rotate from marker frame to world frame using the known tag yaw
        cos_ty = math.cos(tag_yaw)
        sin_ty = math.sin(tag_yaw)

        # cam_in_marker[0] = x in marker frame (right)
        # cam_in_marker[2] = z in marker frame (toward tag face → away from robot)
        wx = tag_x + cos_ty * cam_in_marker[2] - sin_ty * cam_in_marker[0]
        wy = tag_y + sin_ty * cam_in_marker[2] + cos_ty * cam_in_marker[0]

        # Robot faces the tag → add π to the tag's own yaw, then apply camera offset
        world_yaw = tag_yaw + math.pi + yaw_cam_to_marker

        return wx, wy, world_yaw

    @staticmethod
    def tag_from_robot(
        robot_x: float, robot_y: float, robot_theta: float,
        rvec: np.ndarray, tvec: np.ndarray,
    ) -> tuple:
        """
        Estimate the world-frame pose of a tag from the robot's current pose.

        Inverse of from_tag(): used to register a newly seen tag into TAG_POSES.

        Returns
        -------
        (tag_x, tag_y, tag_yaw) in world frame
        """
        R_cm, _ = cv2.Rodrigues(rvec)
        cam_in_marker = (-R_cm.T @ tvec).flatten()
        yaw_cam_to_marker = math.atan2(-R_cm[2, 0], R_cm[2, 2])

        tag_yaw = robot_theta - math.pi - yaw_cam_to_marker
        cos_ty = math.cos(tag_yaw)
        sin_ty = math.sin(tag_yaw)

        tag_x = robot_x - cos_ty * cam_in_marker[2] + sin_ty * cam_in_marker[0]
        tag_y = robot_y - sin_ty * cam_in_marker[2] - cos_ty * cam_in_marker[0]

        return tag_x, tag_y, tag_yaw
