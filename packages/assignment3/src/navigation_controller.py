"""
navigation_controller.py – Proportional visual-servo velocity controller.

Given the camera-frame translation vector (tvec) of the target ARTag,
produces a geometry_msgs/Twist command to drive toward it.

Camera frame convention (OpenCV / Duckiebot):
  x = right,  y = down,  z = forward into the scene

ROS Twist convention:
  linear.x  > 0  →  move forward
  angular.z > 0  →  turn left  (CCW, positive yaw)

Behaviour
---------
1. ALIGN phase: when |heading_error| > ALIGN_THRESHOLD_RAD, the robot
   rotates in place (linear.x = 0) to face the tag before advancing.
2. APPROACH phase: robot moves forward while correcting heading.
   Forward speed is scaled by distance to slow down as the tag gets closer.
3. SEARCH: robot rotates in place to scan for a lost tag.
4. STOP: all velocities zeroed.
"""

import math
from typing import Tuple

import numpy as np
from geometry_msgs.msg import Twist

from config import (
    ALIGN_THRESHOLD_RAD,
    ANGULAR_GAIN,
    LINEAR_SPEED,
    SEARCH_ANGULAR_SPEED,
    SLOWDOWN_FACTOR,
)


class NavigationController:
    """Stateless velocity controller – all methods are pure functions of input."""

    # ── Public API ────────────────────────────────────────────────────────────

    def compute_cmd_vel(self, tvec: np.ndarray) -> Twist:
        """
        Compute a Twist command to drive the robot toward the detected tag.

        Parameters
        ----------
        tvec : (3,) array – translation from camera to tag in camera frame
               tvec[0] = lateral (positive = tag is to the right)
               tvec[2] = forward distance to tag

        Returns
        -------
        geometry_msgs/Twist
        """
        forward = float(tvec[2])         # depth (z)
        lateral = float(tvec[0])         # horizontal offset (x)

        # Heading error: positive means tag is to the right
        #   → angular.z should be negative (turn right = CW in ROS)
        heading_error = math.atan2(lateral, max(forward, 0.01))
        angular_z = -ANGULAR_GAIN * heading_error

        cmd = Twist()
        cmd.angular.z = angular_z

        if abs(heading_error) > ALIGN_THRESHOLD_RAD:
            # Rotate in place first – do not advance until roughly aligned.
            cmd.linear.x = 0.0
        else:
            # Scale speed by distance (slow down when close).
            speed = min(LINEAR_SPEED, forward * SLOWDOWN_FACTOR)
            # Reduce forward speed proportionally to residual heading error.
            speed *= math.cos(heading_error)
            cmd.linear.x = max(speed, 0.0)

        return cmd

    def search_cmd(self) -> Twist:
        """
        Rotate in place to scan for a lost or not-yet-visible ARTag.

        Returns
        -------
        geometry_msgs/Twist  (linear.x = 0, angular.z = SEARCH_ANGULAR_SPEED)
        """
        cmd = Twist()
        cmd.angular.z = SEARCH_ANGULAR_SPEED
        return cmd

    @staticmethod
    def stop_cmd() -> Twist:
        """Return an all-zero Twist (full stop)."""
        return Twist()

    # ── Diagnostics ───────────────────────────────────────────────────────────

    @staticmethod
    def heading_error_rad(tvec: np.ndarray) -> float:
        """Return the raw heading error (rad) for a given tvec."""
        forward = float(tvec[2])
        lateral = float(tvec[0])
        return math.atan2(lateral, max(forward, 0.01))

    @staticmethod
    def distance_m(tvec: np.ndarray) -> float:
        """Return the Euclidean distance (m) from the camera to the tag."""
        return float(np.linalg.norm(tvec))
