"""
pose_source.py – World-frame pose for the Duckiebot, from /odometry_node/odometry.

The Duckiebot stack already publishes integrated wheel odometry as a
nav_msgs/Odometry on ``/<robot>/odometry_node/odometry``; we subscribe to
that and convert the *odometry-frame* pose into the assignment's world
frame using a single rigid transform that is locked in on the first
message.

Convention
----------
At startup we assume the robot is physically placed at the configured
start point ``A`` facing ``start.theta``. The first odometry message we
receive defines the odometry origin. From that point on::

    world_pose = T_world←odom(odom_pose)

where ``T_world←odom`` is the rigid transform that maps the initial
odom_pose to ``(start.x, start.y, start.theta)``.

If the odometry service is unavailable, ``get()`` returns ``None`` until
the first message arrives.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import Optional, Tuple

import rospy
from nav_msgs.msg import Odometry

State = Tuple[float, float, float]


def _yaw_from_quat(qx: float, qy: float, qz: float, qw: float) -> float:
    siny = 2.0 * (qw * qz + qx * qy)
    cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny, cosy)


@dataclass
class _OdomFrame:
    x: float
    y: float
    yaw: float


class PoseSource:
    """Thread-safe source of world-frame pose, anchored on first odom message."""

    def __init__(self, vehicle_name: str, start: State):
        self._start = start  # (x_world, y_world, yaw_world) at robot's start
        self._lock = threading.Lock()
        self._latest: Optional[_OdomFrame] = None
        self._origin: Optional[_OdomFrame] = None  # set on first message

        prefix = f"/{vehicle_name}" if vehicle_name else ""
        self._topic = f"{prefix}/odometry_node/odometry"
        self._sub = rospy.Subscriber(
            self._topic, Odometry, self._cb, queue_size=1, buff_size=2 ** 16,
        )
        rospy.loginfo("[Pose] Subscribed to %s", self._topic)

    # ── public API ──────────────────────────────────────────────────────────

    def is_ready(self) -> bool:
        with self._lock:
            return self._origin is not None and self._latest is not None

    def get(self) -> Optional[State]:
        """Return current world pose (x, y, yaw), or None if no odom yet."""
        with self._lock:
            if self._origin is None or self._latest is None:
                return None
            return self._world_pose(self._origin, self._latest, self._start)

    def reset_origin(self) -> None:
        """Re-anchor: treat the next odom message as the origin again."""
        with self._lock:
            self._origin = None
            self._latest = None

    @property
    def topic(self) -> str:
        return self._topic

    # ── internals ───────────────────────────────────────────────────────────

    def _cb(self, msg: Odometry) -> None:
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        frame = _OdomFrame(x=p.x, y=p.y, yaw=_yaw_from_quat(q.x, q.y, q.z, q.w))
        with self._lock:
            self._latest = frame
            if self._origin is None:
                self._origin = frame
                rospy.loginfo(
                    "[Pose] Anchored origin: odom=(%.3f, %.3f, %.1f°) "
                    "→ world=(%.3f, %.3f, %.1f°)",
                    frame.x, frame.y, math.degrees(frame.yaw),
                    self._start[0], self._start[1], math.degrees(self._start[2]),
                )

    @staticmethod
    def _world_pose(origin: _OdomFrame, latest: _OdomFrame, start: State) -> State:
        # Displacement in odom frame
        dx_o = latest.x - origin.x
        dy_o = latest.y - origin.y

        # Express displacement in robot's *initial* body frame
        c0 = math.cos(-origin.yaw)
        s0 = math.sin(-origin.yaw)
        dx_b = dx_o * c0 - dy_o * s0
        dy_b = dx_o * s0 + dy_o * c0

        # Rotate body-frame displacement into world (the body frame is rotated
        # by start.theta relative to world)
        cs = math.cos(start[2])
        ss = math.sin(start[2])
        x_w = start[0] + dx_b * cs - dy_b * ss
        y_w = start[1] + dx_b * ss + dy_b * cs

        # Yaw is the same delta, added to the world-start yaw
        dyaw = latest.yaw - origin.yaw
        # normalise to [-pi, pi]
        while dyaw > math.pi: dyaw -= 2 * math.pi
        while dyaw < -math.pi: dyaw += 2 * math.pi
        yaw_w = start[2] + dyaw
        while yaw_w > math.pi: yaw_w -= 2 * math.pi
        while yaw_w < -math.pi: yaw_w += 2 * math.pi
        return (x_w, y_w, yaw_w)
