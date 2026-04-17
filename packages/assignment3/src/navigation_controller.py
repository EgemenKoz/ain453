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
from duckietown_msgs.msg import Twist2DStamped

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

    def compute_cmd_vel(self, tvec: np.ndarray) -> Twist2DStamped:
        """
        Compute a Twist2DStamped command to drive the robot toward the detected tag.

        Parameters
        ----------
        tvec : (3,) array – translation from camera to tag in camera frame
               tvec[0] = lateral (positive = tag is to the right)
               tvec[2] = forward distance to tag

        Returns
        -------
        duckietown_msgs/Twist2DStamped
        """
        forward = float(tvec[2])         # depth (z)
        lateral = float(tvec[0])         # horizontal offset (x)

        # Heading error: positive means tag is to the right
        #   → omega should be negative (turn right = CW in ROS)
        heading_error = math.atan2(lateral, max(forward, 0.01))
        omega = -ANGULAR_GAIN * heading_error

        cmd = Twist2DStamped()
        cmd.omega = omega

        if abs(heading_error) > ALIGN_THRESHOLD_RAD:
            # Rotate in place first – do not advance until roughly aligned.
            cmd.v = 0.0
        else:
            # Scale speed by distance (slow down when close).
            speed = min(LINEAR_SPEED, forward * SLOWDOWN_FACTOR)
            # Reduce forward speed proportionally to residual heading error.
            speed *= math.cos(heading_error)
            cmd.v = max(speed, 0.0)

        return cmd

    def search_cmd(self, direction: float = 1.0) -> Twist2DStamped:
        """
        Rotate in place to scan for a lost or not-yet-visible ARTag.

        Parameters
        ----------
        direction : +1.0 for CCW (left), -1.0 for CW (right)

        Returns
        -------
        duckietown_msgs/Twist2DStamped  (v = 0, omega = ±SEARCH_ANGULAR_SPEED)
        """
        cmd = Twist2DStamped()
        cmd.omega = SEARCH_ANGULAR_SPEED * direction
        return cmd

    @staticmethod
    def stop_cmd() -> Twist2DStamped:
        """Return an all-zero Twist2DStamped (full stop)."""
        return Twist2DStamped()

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
