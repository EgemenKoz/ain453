"""
pose_source.py – World-frame pose for the Duckiebot from wheel encoders.

The /<vehicle>/odometry_node/odometry topic is unreliable on stock Duckiebot
images (the node is sometimes not running and the topic only updates while
the robot is moving). The wheel encoder tick topics are always available,
so we integrate differential-drive odometry ourselves.

Convention
----------
Robot is physically placed at the configured start point ``A`` facing
``start.theta`` *before the node starts*. The first encoder tick we see
from each wheel becomes the zero reference; everything afterwards is a
delta integrated through the standard unicycle update.

World-frame update on each new tick (either wheel):

    Δs_left  = (ticks_left  − ticks_left_prev)  / N · 2π · R
    Δs_right = (ticks_right − ticks_right_prev) / N · 2π · R
    Δs       = (Δs_left + Δs_right) / 2
    Δθ       = (Δs_left − Δs_right) / L   # see _fuse_locked: yaw inverted on this hardware
    x  += Δs · cos(θ + Δθ/2)
    y  += Δs · sin(θ + Δθ/2)
    θ  += Δθ

where N = ticks_per_rev, R = wheel_radius_m, L = wheel_base_m.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import Optional, Tuple

import rospy
from duckietown_msgs.msg import WheelEncoderStamped

State = Tuple[float, float, float]


@dataclass
class _WheelState:
    last_ticks: Optional[int] = None  # cumulative tick count at last callback
    distance: float = 0.0             # cumulative distance since first tick

    def update(self, ticks: int, ticks_per_rev: int, wheel_radius: float) -> float:
        """Return Δdistance for *this* tick callback (0 on the very first one)."""
        if self.last_ticks is None:
            self.last_ticks = ticks
            return 0.0
        d_ticks = ticks - self.last_ticks
        self.last_ticks = ticks
        d_dist = d_ticks / ticks_per_rev * 2.0 * math.pi * wheel_radius
        self.distance += d_dist
        return d_dist


class PoseSource:
    """Thread-safe wheel-encoder odometry, anchored at the configured start pose."""

    def __init__(
        self,
        vehicle_name: str,
        start: State,
        wheel_radius_m: float,
        wheel_base_m: float,
        ticks_per_rev: int,
        left_sign: float = 1.0,
        right_sign: float = 1.0,
        log_throttle_s: float = 0.0,
    ):
        self._wheel_radius = wheel_radius_m
        self._wheel_base = wheel_base_m
        self._ticks_per_rev_default = ticks_per_rev
        self._left_sign = float(left_sign)
        self._right_sign = float(right_sign)
        self._log_throttle_s = float(log_throttle_s)
        self._last_log_t: float = 0.0

        self._lock = threading.Lock()

        # Pose state (in world frame).
        self._x, self._y, self._theta = start

        # Per-wheel state.
        self._left = _WheelState()
        self._right = _WheelState()

        # Bookkeeping for pose updates: distances captured at the last fused update.
        self._left_at_last_fuse: float = 0.0
        self._right_at_last_fuse: float = 0.0
        self._has_left = False
        self._has_right = False

        prefix = f"/{vehicle_name}" if vehicle_name else ""
        self._left_topic = f"{prefix}/left_wheel_encoder_node/tick"
        self._right_topic = f"{prefix}/right_wheel_encoder_node/tick"
        self._sub_left = rospy.Subscriber(
            self._left_topic, WheelEncoderStamped, self._left_cb,
            queue_size=10, buff_size=2 ** 16,
        )
        self._sub_right = rospy.Subscriber(
            self._right_topic, WheelEncoderStamped, self._right_cb,
            queue_size=10, buff_size=2 ** 16,
        )
        rospy.loginfo("[Pose] Subscribed to %s", self._left_topic)
        rospy.loginfo("[Pose] Subscribed to %s", self._right_topic)
        rospy.loginfo(
            "[Pose] Anchored start=(%.3f, %.3f, %.1f°)  "
            "R=%.4f m  L=%.4f m  N=%d ticks/rev",
            start[0], start[1], math.degrees(start[2]),
            wheel_radius_m, wheel_base_m, ticks_per_rev,
        )

    # ── public API ──────────────────────────────────────────────────────────

    def is_ready(self) -> bool:
        with self._lock:
            return self._has_left and self._has_right

    def get(self) -> Optional[State]:
        """Return latest pose. Triggers a fuse so each call reflects the
        most recent encoder data from *both* wheels — this eliminates the
        per-callback wobble we used to get when only one wheel had updated.
        """
        with self._lock:
            if not (self._has_left and self._has_right):
                return None
            self._fuse_locked()
            return (self._x, self._y, self._theta)

    @property
    def topics(self) -> Tuple[str, str]:
        return self._left_topic, self._right_topic

    # ── internals ───────────────────────────────────────────────────────────

    def _left_cb(self, msg: WheelEncoderStamped) -> None:
        with self._lock:
            n = self._ticks_per_rev(msg)
            d = self._left.update(int(msg.data), n, self._wheel_radius)
            self._left.distance += (self._left_sign - 1.0) * d  # apply sign
            self._has_left = True
            # Fusion is deferred until get() so we always combine both
            # wheels' freshest data.

    def _right_cb(self, msg: WheelEncoderStamped) -> None:
        with self._lock:
            n = self._ticks_per_rev(msg)
            d = self._right.update(int(msg.data), n, self._wheel_radius)
            self._right.distance += (self._right_sign - 1.0) * d
            self._has_right = True

    def _ticks_per_rev(self, msg: WheelEncoderStamped) -> int:
        n = int(getattr(msg, "resolution", 0)) or 0
        return n if n > 0 else self._ticks_per_rev_default

    def _fuse_locked(self) -> None:
        """Apply the differential-drive update from the latest wheel distances.

        Caller must hold ``self._lock``.
        """
        if not (self._has_left and self._has_right):
            return
        d_left = self._left.distance - self._left_at_last_fuse
        d_right = self._right.distance - self._right_at_last_fuse
        self._left_at_last_fuse = self._left.distance
        self._right_at_last_fuse = self._right.distance

        d_s = 0.5 * (d_left + d_right)
        # left − right (not the textbook right − left): on this Duckiebot the
        # wheel encoders are assigned such that the standard convention yields
        # an inverted yaw, so a physical left turn would integrate as a right
        # turn. Forward/back motion (d_s) is unaffected by this choice.
        d_th = (d_left - d_right) / self._wheel_base
        # Mid-point integration for better accuracy on curves.
        th_mid = self._theta + 0.5 * d_th
        self._x += d_s * math.cos(th_mid)
        self._y += d_s * math.sin(th_mid)
        self._theta += d_th
        # normalise to [-pi, pi]
        while self._theta > math.pi:  self._theta -= 2.0 * math.pi
        while self._theta < -math.pi: self._theta += 2.0 * math.pi

        # Throttled debug log (helps verify integration direction in the lab).
        if self._log_throttle_s > 0.0:
            now = rospy.Time.now().to_sec()
            if now - self._last_log_t >= self._log_throttle_s:
                self._last_log_t = now
                rospy.loginfo(
                    "[Pose] dL=%+.4f dR=%+.4f → dS=%+.4f dθ=%+.3f rad   "
                    "pose=(%.3f, %.3f, %.1f°)",
                    d_left, d_right, d_s, d_th,
                    self._x, self._y, math.degrees(self._theta),
                )
