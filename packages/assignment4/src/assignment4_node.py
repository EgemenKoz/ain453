"""
assignment4_node.py – Main DTROS node.

Pipeline at startup
-------------------
1. Load YAML.
2. Run A* once on the empty grid → list of dense waypoints.
3. Subscribe to /<robot>/odometry_node/odometry through PoseSource.
4. Wait until pose is available, then start the control loop.

Control loop  (cfg.control.rate_hz, default 10 Hz)
---------------------------------------------------
* Read current world pose from PoseSource.
* If within cfg.goal.tol_m of B → publish stop, transition to GOAL_REACHED.
* Otherwise call DWAPlanner.step → get the chosen (v, ω) and all rollouts.
* Publish chosen (v, ω) on /<robot>/car_cmd_switch_node/cmd.
* Cache the latest snapshot for the visualiser.

Visualiser  (cfg.viz.rate_hz, default 5 Hz)
-------------------------------------------
* Render the cached snapshot via viz.Renderer (matplotlib + Agg backend).
* Convert the figure canvas to JPEG via cv2.
* Publish a sensor_msgs/CompressedImage on
      /<robot>/assignment4/viz/compressed
  Watch live from a laptop with rqt_image_view.

State machine
-------------
INIT          – waiting for the first odometry message.
RUNNING       – control loop active, robot driving toward B.
GOAL_REACHED  – within tolerance, publishing zero velocity.
ABORT         – DWA could not find any feasible move (every rollout
                rejected). The robot is stopped and the user is asked to
                intervene; the node continues publishing for the visualiser.
"""

from __future__ import annotations

# Use a non-interactive backend BEFORE importing pyplot anywhere in the tree.
import matplotlib
matplotlib.use("Agg")

import math
import os
import sys
import threading
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge  # noqa: F401  (kept for downstream consumers)
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import Twist2DStamped
from sensor_msgs.msg import CompressedImage

# Local imports — sys.path is set in assignment4.py so these resolve.
from astar_grid import path_length, plan as astar_plan
from config_loader import Config, default_path, load
from costmap import CircleObstacle
from dwa import DWAPlanner, DWAResult, Rollout
from pose_source import PoseSource
from viz import Renderer, Snapshot

State = Tuple[float, float, float]
Point = Tuple[float, float]

# State machine
_INIT = "INIT"
_RUNNING = "RUNNING"
_GOAL_REACHED = "GOAL_REACHED"
_ABORT = "ABORT"


class Assignment4Node(DTROS):

    def __init__(self, node_name: str):
        super().__init__(node_name=node_name, node_type=NodeType.GENERIC)

        # ── Config ──────────────────────────────────────────────────────────
        cfg_path = rospy.get_param("~config_path", str(default_path()))
        self.cfg: Config = load(cfg_path)
        rospy.loginfo("[A4] Loaded config from %s", cfg_path)

        self._vehicle = (
            rospy.get_param("~vehicle_name", "")
            or os.environ.get("VEHICLE_NAME", "")
        )
        prefix = f"/{self._vehicle}" if self._vehicle else ""

        # ── Plan ────────────────────────────────────────────────────────────
        self._waypoints: List[Point] = astar_plan(self.cfg)
        rospy.loginfo(
            "[A4] A*: %d dense waypoints, length %.3f m",
            len(self._waypoints), path_length(self._waypoints),
        )
        self._obstacle = CircleObstacle.from_config(self.cfg)
        rospy.loginfo(
            "[A4] Obstacle at (%.2f, %.2f)  r=%.3f m  inflated=%.3f m",
            self._obstacle.cx, self._obstacle.cy,
            self._obstacle.radius, self._obstacle.inflated_radius,
        )

        # ── Components ──────────────────────────────────────────────────────
        self._planner = DWAPlanner(self.cfg, self._obstacle, self._waypoints)
        self._pose_src = PoseSource(
            self._vehicle,
            (self.cfg.start.x, self.cfg.start.y, self.cfg.start.theta),
            wheel_radius_m=self.cfg.robot.wheel_radius_m,
            wheel_base_m=self.cfg.robot.wheel_base_m,
            ticks_per_rev=self.cfg.robot.ticks_per_rev,
            left_sign=self.cfg.robot.left_sign,
            right_sign=self.cfg.robot.right_sign,
            log_throttle_s=self.cfg.robot.pose_log_throttle_s,
        )
        self._renderer = Renderer(self.cfg, self._waypoints, self._obstacle)
        self._renderer.fig.suptitle(
            "Assignment 4 — A* + DWA local planner", fontsize=11,
        )

        # ── ROS topics ──────────────────────────────────────────────────────
        self._cmd_pub = rospy.Publisher(
            f"{prefix}/car_cmd_switch_node/cmd", Twist2DStamped, queue_size=1,
        )
        self._viz_pub = rospy.Publisher(
            f"{prefix}/assignment4/viz/compressed", CompressedImage, queue_size=1,
        )

        # ── Mutable state ───────────────────────────────────────────────────
        self._state = _INIT
        self._snap_lock = threading.Lock()
        self._snap: Optional[Snapshot] = None
        self._trace: List[Point] = []
        self._step_idx: int = 0

        # ── Timers ──────────────────────────────────────────────────────────
        rospy.Timer(
            rospy.Duration(1.0 / self.cfg.control.rate_hz),
            self._control_cb,
        )
        rospy.Timer(
            rospy.Duration(1.0 / self.cfg.viz.rate_hz),
            self._viz_cb,
        )
        rospy.on_shutdown(self._on_shutdown)

        rospy.loginfo(
            "[A4] Topics:  cmd=%s  viz=%s",
            self._cmd_pub.resolved_name, self._viz_pub.resolved_name,
        )
        rospy.loginfo("[A4] Waiting for first encoder ticks from both wheels …")

    # ── Control ─────────────────────────────────────────────────────────────

    def _control_cb(self, _event) -> None:
        if self._state == _GOAL_REACHED or self._state == _ABORT:
            self._cmd_pub.publish(self._stop_cmd())
            return

        pose = self._pose_src.get()
        if pose is None:
            # Don't move until we have an anchored pose.
            self._cmd_pub.publish(self._stop_cmd())
            return

        if self._state == _INIT:
            self._state = _RUNNING
            self._trace = [(pose[0], pose[1])]
            rospy.loginfo(
                "[A4] First pose received: (%.3f, %.3f, %.1f°). Running.",
                pose[0], pose[1], math.degrees(pose[2]),
            )

        x, y, _ = pose
        d_goal = math.hypot(x - self.cfg.goal.x, y - self.cfg.goal.y)
        if d_goal <= self.cfg.goal.tol_m:
            self._state = _GOAL_REACHED
            self._cmd_pub.publish(self._stop_cmd())
            rospy.loginfo("[A4] *** Goal reached at d=%.3f m ***", d_goal)
            self._cache_snapshot(pose, [], None, info="*** goal reached ***")
            return

        # Run DWA on the latest pose.
        result: DWAResult = self._planner.step(pose)
        chosen: Optional[Rollout] = result.chosen

        if chosen is None or chosen.rejected:
            self._state = _ABORT
            self._cmd_pub.publish(self._stop_cmd())
            rospy.logwarn(
                "[A4] No feasible DWA move (every rollout rejected). Stopping.",
            )
            self._cache_snapshot(pose, list(result.rollouts), chosen,
                                 info="ABORT: no feasible move")
            return

        # Publish command, clamped to control limits as a safety net.
        v = max(0.0, min(self.cfg.control.linear_max, float(chosen.v)))
        w = max(-self.cfg.control.angular_max,
                min(self.cfg.control.angular_max, float(chosen.w)))
        cmd = Twist2DStamped()
        cmd.header.stamp = rospy.Time.now()
        cmd.v = v
        cmd.omega = w
        self._cmd_pub.publish(cmd)

        # Update trace + cache snapshot.
        self._trace.append((x, y))
        self._step_idx += 1
        self._cache_snapshot(
            pose, list(result.rollouts), chosen,
            info=f"d_goal={d_goal:.2f} m",
        )

        rospy.loginfo_throttle(
            1.0,
            "[A4] pose=(%.2f, %.2f, %.0f°)  cmd  v=%.2f  ω=%+.2f  d_goal=%.2f",
            x, y, math.degrees(pose[2]), v, w, d_goal,
        )

    # ── Visualisation ───────────────────────────────────────────────────────

    def _viz_cb(self, _event) -> None:
        with self._snap_lock:
            snap = self._snap
        if snap is None:
            return

        try:
            self._renderer.update(snap)
            self._renderer.fig.canvas.draw()
            buf = self._renderer.fig.canvas.buffer_rgba()
            img_rgba = np.asarray(buf, dtype=np.uint8)
            img_bgr = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2BGR)
            ok, jpg = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ok:
                rospy.logwarn_throttle(5.0, "[A4] viz: JPEG encode failed")
                return
            msg = CompressedImage()
            msg.header.stamp = rospy.Time.now()
            msg.format = "jpeg"
            msg.data = jpg.tobytes()
            self._viz_pub.publish(msg)
        except Exception as exc:  # noqa: BLE001
            rospy.logwarn_throttle(
                5.0, "[A4] viz render failed: %s", exc,
            )

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _cache_snapshot(
        self,
        pose: State,
        rollouts: List[Rollout],
        chosen: Optional[Rollout],
        info: str = "",
    ) -> None:
        snap = Snapshot(
            state=pose,
            rollouts=rollouts,
            chosen=chosen,
            trace=list(self._trace),
            step=self._step_idx,
            info=f"state={self._state}  {info}".strip(),
        )
        with self._snap_lock:
            self._snap = snap

    @staticmethod
    def _stop_cmd() -> Twist2DStamped:
        cmd = Twist2DStamped()
        cmd.header.stamp = rospy.Time.now()
        cmd.v = 0.0
        cmd.omega = 0.0
        return cmd

    def _on_shutdown(self) -> None:
        try:
            self._cmd_pub.publish(self._stop_cmd())
            rospy.loginfo("[A4] Shutdown: stop published.")
        except Exception:
            pass
