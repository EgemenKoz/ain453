"""
tof_detector.py – Front-center ToF → world-frame obstacle (Task 5 bonus).

The Duckiebot front-center Time-of-Flight sensor (sensor_msgs/Range) gives
a forward range measurement along the robot's heading. When that range
drops below ``bonus.detect_distance_m`` and the reading is valid, we
project the hit point into world coordinates using the current pose:

    hit_x = x + (offset + range) · cos(θ)
    hit_y = y + (offset + range) · sin(θ)

where ``offset`` is the ToF mounting offset forward of the robot centre.
The hit point itself is the *front face* of the obstacle along the beam,
so we push the centre estimate one detected-radius further forward.

The detector exposes ``latest_detection(pose)`` so the main node can poll
on its own timer without callback coupling. If ``sticky`` is true, the
first valid detection is frozen and re-used forever (no oscillation).

The topic-handling pattern is taken from assignment1's ToF subscriber.
"""

from __future__ import annotations

import math
import threading
from typing import Optional, Tuple

import rospy
from sensor_msgs.msg import Range

from config_loader import Config
from costmap import CircleObstacle

State = Tuple[float, float, float]


class TofObstacleDetector:
    """Subscribes to the front-center ToF and projects hits into the world."""

    def __init__(self, cfg: Config, vehicle_name: str):
        self.cfg = cfg
        self._lock = threading.Lock()
        self._latest_range: Optional[float] = None
        self._latest_valid: bool = False
        self._frozen: Optional[CircleObstacle] = None

        topic = cfg.bonus.tof_topic.strip()
        if not topic:
            prefix = f"/{vehicle_name}" if vehicle_name else ""
            topic = f"{prefix}/front_center_tof_driver_node/range"
        self.topic = topic
        rospy.loginfo("[A4-bonus] Subscribing to ToF topic: %s", topic)
        self._sub = rospy.Subscriber(topic, Range, self._cb, queue_size=1)

    # ── ROS callback ───────────────────────────────────────────────────────

    def _cb(self, msg: Range) -> None:
        r = msg.range
        valid = (not math.isnan(r)) and (msg.min_range <= r <= msg.max_range)
        with self._lock:
            self._latest_range = r if valid else None
            self._latest_valid = valid

    # ── public API ─────────────────────────────────────────────────────────

    @property
    def frozen(self) -> Optional[CircleObstacle]:
        with self._lock:
            return self._frozen

    def latest_detection(self, pose: State) -> Optional[CircleObstacle]:
        """Return an obstacle estimate if the last ToF reading triggers, else None.

        Once a valid detection has been frozen (``bonus.sticky``), the same
        object is returned on every subsequent call.
        """
        with self._lock:
            if self._frozen is not None:
                return self._frozen
            r = self._latest_range
            valid = self._latest_valid

        if not valid or r is None:
            return None
        if r > self.cfg.bonus.detect_distance_m:
            return None

        det = self._project(pose, r)
        if self.cfg.bonus.sticky:
            with self._lock:
                # Re-check inside the lock so we don't race two detections.
                if self._frozen is None:
                    self._frozen = det
                    rospy.loginfo(
                        "[A4-bonus] Obstacle detected at (%.2f, %.2f)  "
                        "range=%.3f m → frozen",
                        det.cx, det.cy, r,
                    )
                return self._frozen
        return det

    # ── helpers ────────────────────────────────────────────────────────────

    def _project(self, pose: State, r: float) -> CircleObstacle:
        x, y, th = pose
        radius = self.cfg.bonus.detected_radius_m
        # Push the centre half a body further than the hit point so the
        # circle straddles the obstacle rather than sitting on its near face.
        d = self.cfg.bonus.tof_offset_m + r + radius
        cx = x + d * math.cos(th)
        cy = y + d * math.sin(th)
        inflation = self.cfg.robot.inflated_radius_m
        return CircleObstacle(cx=cx, cy=cy, radius=radius, inflation=inflation)
