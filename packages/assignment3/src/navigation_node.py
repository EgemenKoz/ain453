"""
navigation_node.py – Main DTROS node for discrete-action grid navigation.

Navigation strategy
-------------------
A* computes the optimal node path.  action_planner converts that path into
an ordered sequence of discrete actions:

    TURN_LEFT  – rotate 90° CCW  (timed, open-loop)
    TURN_RIGHT – rotate 90° CW   (timed, open-loop)
    FORWARD    – drive one grid cell toward the next tag (visual servo)

State machine
-------------
IDLE           An action just finished; fetch and start the next one.
TURNING_LEFT   Rotating CCW for TURN_90_DURATION_S seconds.
TURNING_RIGHT  Rotating CW  for TURN_90_DURATION_S seconds.
SEARCHING      Spinning slowly to find the target ARTag (FORWARD action).
APPROACHING    Visual-servoing toward the detected target tag.
GOAL_REACHED   At N_GOAL; zero velocity published.

Localisation
------------
Position is confirmed when the ARTag whose ID equals the target node ID is
detected within PROXIMITY_THRESHOLD_M.  Turns are purely time-based (no tag
needed); FORWARD actions wait for tag confirmation before advancing.
"""

import os

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from duckietown.dtros import DTROS, NodeType
from geometry_msgs.msg import Twist
from sensor_msgs.msg import CameraInfo, CompressedImage

import astar
import action_planner
from aruco_detector import ArucoDetector
from calibration import CameraCalibration
from config import (
    GOAL_NODE,
    PROXIMITY_THRESHOLD_M,
    START_NODE,
    TAG_LOST_PATIENCE_FRAMES,
    TURN_90_DURATION_S,
    WATCHDOG_HZ,
)
from navigation_controller import NavigationController

# ── State constants ───────────────────────────────────────────────────────────
_IDLE          = "IDLE"
_TURNING_LEFT  = "TURNING_LEFT"
_TURNING_RIGHT = "TURNING_RIGHT"
_SEARCHING     = "SEARCHING"
_APPROACHING   = "APPROACHING"
_GOAL_REACHED  = "GOAL_REACHED"


class PathNavigationNode(DTROS):
    """
    Autonomous grid-navigation node: A* planning + discrete actions
    + ARTag-based localisation.
    """

    def __init__(self, node_name: str):
        super().__init__(node_name=node_name, node_type=NodeType.GENERIC)

        self._vehicle = os.environ.get("VEHICLE_NAME", "")
        self._bridge = CvBridge()

        # ── A* planning ───────────────────────────────────────────────────────
        rospy.loginfo("[Nav] Running A* (start=N%d, goal=N%d) …", START_NODE, GOAL_NODE)
        self._path, self._path_cost = astar.run()
        astar.print_result(self._path, self._path_cost)

        # ── Action sequence ───────────────────────────────────────────────────
        # Each entry: (action_type, path_step_idx)
        #   action_type    ∈ {FORWARD, TURN_LEFT, TURN_RIGHT}
        #   path_step_idx  = index in self._path of the destination node
        self._actions = action_planner.compute_actions(self._path)
        self._action_idx = 0   # index of the action currently being executed

        rospy.loginfo(
            "[Nav] Action plan (%d steps): %s",
            len(self._actions),
            action_planner.format_actions(self._path, self._actions),
        )

        # ── State ─────────────────────────────────────────────────────────────
        self._state: str = _IDLE
        self._turn_end_time = None        # rospy.Time when current turn finishes
        self._tag_lost_frames: int = 0

        # ── Sub-components ────────────────────────────────────────────────────
        self._calib = CameraCalibration(self._vehicle)
        self._detector = ArucoDetector()
        self._controller = NavigationController()
        self._new_K = None   # cached optimal camera matrix

        # ── ROS topics ────────────────────────────────────────────────────────
        prefix = f"/{self._vehicle}" if self._vehicle else ""

        rospy.Subscriber(
            f"{prefix}/camera_node/image/compressed",
            CompressedImage,
            self._image_cb,
            queue_size=1,
            buff_size=2 ** 24,
        )
        rospy.Subscriber(
            f"{prefix}/camera_node/camera_info",
            CameraInfo,
            self._calib.update_from_camera_info,
            queue_size=1,
        )

        self._cmd_pub = rospy.Publisher(
            f"{prefix}/cmd_vel",
            Twist,
            queue_size=1,
        )

        # Watchdog: stop if no camera frame arrives for >1 s.
        self._last_image_stamp = rospy.Time.now()
        rospy.Timer(
            rospy.Duration(1.0 / WATCHDOG_HZ),
            self._watchdog_cb,
        )

        rospy.loginfo("[Nav] Node ready.  Starting at N%d, heading toward N%d.",
                      START_NODE, GOAL_NODE)

    # ── Convenience properties ────────────────────────────────────────────────

    @property
    def _done(self) -> bool:
        """True when every action in the plan has been executed."""
        return self._action_idx >= len(self._actions)

    @property
    def _current_action(self):
        """(action_type, path_step_idx) for the active action, or None."""
        if self._done:
            return None
        return self._actions[self._action_idx]

    @property
    def _target_node(self) -> int:
        """Node ID the robot is currently navigating toward."""
        _, step_idx = self._current_action
        return self._path[step_idx]

    # ── Main image callback ───────────────────────────────────────────────────

    def _image_cb(self, msg: CompressedImage) -> None:
        self._last_image_stamp = rospy.Time.now()

        # ── Goal already reached ──────────────────────────────────────────────
        if self._state == _GOAL_REACHED:
            self._cmd_pub.publish(self._controller.stop_cmd())
            return

        # ── IDLE: pick and start the next action ──────────────────────────────
        if self._state == _IDLE:
            self._start_next_action()
            if self._state == _GOAL_REACHED:
                return   # plan was empty or just finished

        # ── TURNING: time-based open-loop rotation ────────────────────────────
        if self._state in (_TURNING_LEFT, _TURNING_RIGHT):
            if rospy.Time.now() >= self._turn_end_time:
                rospy.loginfo("[Nav] Turn complete → IDLE")
                self._cmd_pub.publish(self._controller.stop_cmd())
                self._state = _IDLE
            elif self._state == _TURNING_LEFT:
                self._cmd_pub.publish(self._controller.turn_left_cmd())
            else:
                self._cmd_pub.publish(self._controller.turn_right_cmd())
            return   # no image processing needed during a turn

        # ── SEARCHING / APPROACHING: need camera image ────────────────────────
        np_arr = np.frombuffer(msg.data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            rospy.logwarn_throttle(5.0, "[Nav] Failed to decode image.")
            return

        if not self._calib.is_ready:
            # No calibration yet; keep searching so the robot isn't frozen.
            self._cmd_pub.publish(self._controller.search_cmd())
            return

        h, w = img.shape[:2]
        if self._new_K is None:
            self._new_K, _ = cv2.getOptimalNewCameraMatrix(
                self._calib.K, self._calib.D, (w, h), alpha=1, newImgSize=(w, h)
            )
        undistorted = cv2.undistort(img, self._calib.K, self._calib.D, None, self._new_K)
        detections = self._detector.detect(undistorted, self._new_K)

        target_id = self._target_node

        if target_id in detections:
            _, tvec = detections[target_id]
            dist = self._controller.distance_m(tvec)
            self._tag_lost_frames = 0

            rospy.loginfo_throttle(
                1.0,
                "[Nav] Target N%d  dist=%.3f m  state=%s",
                target_id, dist, self._state,
            )

            if dist < PROXIMITY_THRESHOLD_M:
                # Arrived at node — confirm position and advance plan
                self._on_node_reached(target_id)
            else:
                self._state = _APPROACHING
                self._cmd_pub.publish(self._controller.compute_cmd_vel(tvec))

        else:
            # Target tag not visible this frame
            self._tag_lost_frames += 1

            if self._tag_lost_frames >= TAG_LOST_PATIENCE_FRAMES:
                if self._state != _SEARCHING:
                    rospy.logwarn(
                        "[Nav] N%d lost for %d frames → SEARCHING",
                        target_id, self._tag_lost_frames,
                    )
                self._state = _SEARCHING

            if self._state == _SEARCHING:
                self._cmd_pub.publish(self._controller.search_cmd())
            else:
                self._cmd_pub.publish(self._controller.stop_cmd())

    # ── Action execution ──────────────────────────────────────────────────────

    def _start_next_action(self) -> None:
        """
        Fetch the next action from the plan and begin executing it.

        TURN actions: set turn state + deadline; advance action_idx immediately
                      (turn completion is time-based, not tag-based).
        FORWARD actions: set SEARCHING state; action_idx advances in
                         _on_node_reached() when the tag is confirmed.
        """
        if self._done:
            # No more actions — the goal should have been confirmed already,
            # but guard here just in case.
            rospy.loginfo("[Nav] Action plan exhausted → GOAL_REACHED")
            self._state = _GOAL_REACHED
            self._cmd_pub.publish(self._controller.stop_cmd())
            return

        action_type, step_idx = self._current_action
        target_node = self._path[step_idx]

        rospy.loginfo(
            "[Nav] Action [%d/%d]: %s toward N%d",
            self._action_idx + 1, len(self._actions),
            action_type, target_node,
        )

        if action_type == action_planner.TURN_LEFT:
            self._state = _TURNING_LEFT
            self._turn_end_time = rospy.Time.now() + rospy.Duration(TURN_90_DURATION_S)
            self._action_idx += 1   # advance now; completion is timer-based

        elif action_type == action_planner.TURN_RIGHT:
            self._state = _TURNING_RIGHT
            self._turn_end_time = rospy.Time.now() + rospy.Duration(TURN_90_DURATION_S)
            self._action_idx += 1   # advance now; completion is timer-based

        elif action_type == action_planner.FORWARD:
            self._state = _SEARCHING
            self._tag_lost_frames = 0
            # action_idx is advanced in _on_node_reached()

    def _on_node_reached(self, node_id: int) -> None:
        """
        Called when the robot is within PROXIMITY_THRESHOLD_M of the target tag.

        Confirms the current grid position, publishes a stop, then either
        transitions to GOAL_REACHED (if at the goal) or back to IDLE to
        fetch the next action.
        """
        rospy.loginfo("[Nav] Confirmed position: N%d", node_id)
        self._cmd_pub.publish(self._controller.stop_cmd())

        # Advance past the current FORWARD action
        self._action_idx += 1

        if node_id == GOAL_NODE:
            self._state = _GOAL_REACHED
            rospy.loginfo("[Nav] *** Goal Reached ***")
            print("Goal Reached")
            print(f"Path      : {astar.format_path(self._path)}")
            print(f"Total cost: {self._path_cost:.4f}")
        else:
            self._state = _IDLE
            rospy.loginfo(
                "[Nav] N%d reached → IDLE  (%d actions remaining)",
                node_id, len(self._actions) - self._action_idx,
            )

    # ── Watchdog ──────────────────────────────────────────────────────────────

    def _watchdog_cb(self, _event) -> None:
        """Stop the robot if no camera image has arrived for >1 second."""
        if self._state == _GOAL_REACHED:
            return
        elapsed = (rospy.Time.now() - self._last_image_stamp).to_sec()
        if elapsed > 1.0:
            rospy.logwarn_throttle(
                2.0,
                "[Nav] No image for %.1f s → stop",
                elapsed,
            )
            self._cmd_pub.publish(self._controller.stop_cmd())
