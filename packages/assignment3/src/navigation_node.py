"""
navigation_node.py – Main DTROS node for A* pathfinding + ARTag navigation.

Start-up sequence
-----------------
1. Run A* to compute the shortest path from N0 to N15.
2. Print the path sequence and total cost to the terminal.
3. Begin autonomous navigation along the computed path.

State machine
-------------
SEARCHING   The robot rotates in place to scan for the target ARTag.
            Entry: on node arrival OR after TAG_LOST_PATIENCE_FRAMES
                   consecutive frames without a detection.

APPROACHING The target ARTag is visible; the proportional controller
            drives the robot toward it.
            Entry: target tag detected while SEARCHING or APPROACHING.

GOAL_REACHED Robot has reached N15.  Publishes zero velocity and prints
             "Goal Reached".

Localisation
------------
Position is determined solely by ARTag detections.  The node tracks which
path waypoint the robot is heading toward (self._target_idx) and waits for
the corresponding ARTag (ID == node ID) to be detected within
PROXIMITY_THRESHOLD_M before advancing to the next waypoint.
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
from aruco_detector import ArucoDetector
from calibration import CameraCalibration
from config import (
    GOAL_NODE,
    PROXIMITY_THRESHOLD_M,
    START_NODE,
    TAG_LOST_PATIENCE_FRAMES,
    WATCHDOG_HZ,
)
from navigation_controller import NavigationController

# Robot states
_SEARCHING   = "SEARCHING"
_APPROACHING = "APPROACHING"
_GOAL_REACHED = "GOAL_REACHED"


class PathNavigationNode(DTROS):
    """
    Autonomous navigation node: A* planning + ARTag-based localisation.
    """

    def __init__(self, node_name: str):
        super().__init__(node_name=node_name, node_type=NodeType.GENERIC)

        self._vehicle = os.environ.get("VEHICLE_NAME", "")
        self._bridge = CvBridge()

        # ── A* planning ───────────────────────────────────────────────────────
        rospy.loginfo("[Nav] Running A* (start=N%d, goal=N%d) …", START_NODE, GOAL_NODE)
        self._path, self._path_cost = astar.run()
        astar.print_result(self._path, self._path_cost)
        rospy.loginfo("[Nav] Path: %s", astar.format_path(self._path))

        # _target_idx points to the *next* waypoint the robot is heading to.
        # The robot is assumed to start exactly at path[0] = N0.
        self._target_idx: int = 1          # heading toward path[1]
        self._state: str = _SEARCHING
        self._tag_lost_frames: int = 0

        # ── Sub-components ────────────────────────────────────────────────────
        self._calib = CameraCalibration(self._vehicle)
        self._detector = ArucoDetector()
        self._controller = NavigationController()

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

        # Watchdog: stop the robot if no image arrives for >1 s.
        self._last_image_stamp = rospy.Time.now()
        rospy.Timer(
            rospy.Duration(1.0 / WATCHDOG_HZ),
            self._watchdog_cb,
        )

        rospy.loginfo(
            "[Nav] Node ready. Heading toward N%d.  State: %s",
            self._path[self._target_idx],
            self._state,
        )

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def _target_node(self) -> int:
        """The ID of the node the robot is currently heading toward."""
        return self._path[self._target_idx]

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _image_cb(self, msg: CompressedImage) -> None:
        self._last_image_stamp = rospy.Time.now()

        if self._state == _GOAL_REACHED:
            self._cmd_pub.publish(self._controller.stop_cmd())
            return

        # ── Decode + undistort ───────────────────────────────────────────────
        np_arr = np.frombuffer(msg.data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            rospy.logwarn_throttle(5.0, "[Nav] Failed to decode image.")
            return

        if not self._calib.is_ready:
            # Cannot detect tags without calibration; keep searching.
            self._cmd_pub.publish(self._controller.search_cmd())
            return

        h, w = img.shape[:2]
        new_K, _ = cv2.getOptimalNewCameraMatrix(
            self._calib.K, self._calib.D, (w, h), alpha=1, newImgSize=(w, h)
        )
        undistorted = cv2.undistort(img, self._calib.K, self._calib.D, None, new_K)

        # ── ARTag detection ──────────────────────────────────────────────────
        detections = self._detector.detect(undistorted, new_K)

        # ── Control decision ─────────────────────────────────────────────────
        target_id = self._target_node

        if target_id in detections:
            _, tvec = detections[target_id]
            dist = self._controller.distance_m(tvec)
            self._tag_lost_frames = 0

            rospy.loginfo_throttle(
                1.0,
                "[Nav] Target N%d detected  dist=%.3f m  state=%s",
                target_id, dist, self._state,
            )

            if dist < PROXIMITY_THRESHOLD_M:
                self._on_node_reached(target_id)
            else:
                self._state = _APPROACHING
                self._cmd_pub.publish(self._controller.compute_cmd_vel(tvec))
        else:
            # Target tag not visible
            self._tag_lost_frames += 1

            if self._tag_lost_frames >= TAG_LOST_PATIENCE_FRAMES:
                if self._state != _SEARCHING:
                    rospy.logwarn(
                        "[Nav] Target N%d lost for %d frames. Switching to SEARCHING.",
                        target_id, self._tag_lost_frames,
                    )
                self._state = _SEARCHING

            # Stop immediately; rotate only when in SEARCHING state.
            if self._state == _SEARCHING:
                self._cmd_pub.publish(self._controller.search_cmd())
            else:
                self._cmd_pub.publish(self._controller.stop_cmd())

    def _on_node_reached(self, node_id: int) -> None:
        """Handle arrival at a waypoint node."""
        rospy.loginfo("[Nav] Reached N%d.", node_id)
        self._cmd_pub.publish(self._controller.stop_cmd())

        if node_id == GOAL_NODE:
            self._state = _GOAL_REACHED
            rospy.loginfo("[Nav] *** Goal Reached ***")
            print("Goal Reached")
            print(f"Path      : {astar.format_path(self._path)}")
            print(f"Total cost: {self._path_cost:.4f}")
        else:
            self._target_idx += 1
            self._state = _SEARCHING
            self._tag_lost_frames = 0
            rospy.loginfo(
                "[Nav] Next target: N%d.  State: SEARCHING",
                self._target_node,
            )

    def _watchdog_cb(self, _event) -> None:
        """Stop the robot if no camera image has arrived for >1 second."""
        if self._state == _GOAL_REACHED:
            return
        elapsed = (rospy.Time.now() - self._last_image_stamp).to_sec()
        if elapsed > 1.0:
            rospy.logwarn_throttle(
                2.0,
                "[Nav] No camera image for %.1f s. Publishing stop.",
                elapsed,
            )
            self._cmd_pub.publish(self._controller.stop_cmd())
