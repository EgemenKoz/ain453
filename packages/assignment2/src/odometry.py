"""
odometry.py – Differential-drive wheel odometry.

Computes incremental (d_center, d_theta) from left/right encoder tick counts
and exposes reset_to_current() so that an ArUco correction can restart the
dead-reckoning chain without accumulated error.
"""

import math
from typing import Optional

from config import WHEEL_RADIUS_M, WHEEL_BASELINE_M, TICKS_PER_REV


class WheelOdometry:
    def __init__(self):
        self._left_ticks: Optional[int] = None
        self._right_ticks: Optional[int] = None
        self._left_prev: Optional[int] = None
        self._right_prev: Optional[int] = None

    # ── Encoder callbacks ─────────────────────────────────────────────────────

    def update_left(self, ticks: int) -> None:
        if self._left_prev is None:
            self._left_prev = ticks
        self._left_ticks = ticks

    def update_right(self, ticks: int) -> None:
        if self._right_prev is None:
            self._right_prev = ticks
        self._right_ticks = ticks

    # ── Odometry step ─────────────────────────────────────────────────────────

    def compute_delta(self) -> tuple:
        """
        Compute incremental motion since the last call.

        Returns
        -------
        (d_center, d_theta) : floats
            d_center – distance travelled by robot centre (metres, signed)
            d_theta  – change in heading (radians, positive = counter-clockwise)
        Returns (0.0, 0.0) when encoder data is not yet available.
        """
        if None in (
            self._left_ticks, self._right_ticks,
            self._left_prev, self._right_prev,
        ):
            return 0.0, 0.0

        dl = (
            (self._left_ticks - self._left_prev)
            / TICKS_PER_REV
            * (2.0 * math.pi * WHEEL_RADIUS_M)
        )
        dr = (
            (self._right_ticks - self._right_prev)
            / TICKS_PER_REV
            * (2.0 * math.pi * WHEEL_RADIUS_M)
        )

        self._left_prev = self._left_ticks
        self._right_prev = self._right_ticks

        d_center = (dl + dr) / 2.0
        d_theta = (dr - dl) / WHEEL_BASELINE_M
        return d_center, d_theta

    def reset_to_current(self) -> None:
        """
        Set previous ticks equal to current ticks.
        Call this after an ArUco pose correction so that the next odometry step
        starts accumulating from the corrected pose, not the old one.
        """
        self._left_prev = self._left_ticks
        self._right_prev = self._right_ticks
