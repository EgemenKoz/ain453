"""
navigation_node.py – Main DTROS node for A* pathfinding + ARTag navigation.

Start-up sequence
-----------------
1. Run A* to compute the shortest path from N0 to N15.
2. Print the path sequence and total cost to the terminal.
3. Begin autonomous navigation along the computed path.

State machine
-------------
SEARCHING      The robot rotates in place to scan for the target ARTag.
               Uses heading tracking to decide turn direction.

APPROACHING    The target ARTag is visible; the proportional controller
               drives the robot toward it.

BLIND_FORWARD  Tag disappeared while APPROACHING and last distance was
               small.  Robot drives straight forward briefly, then
               declares the node reached.

GOAL_REACHED   Robot has reached N15.  Publishes zero velocity.

Heading tracking
----------------
The robot's heading (orientation in the grid frame) is estimated from the
graph: each time the robot travels from node A to node B, the heading is
set to atan2(By-Ay, Bx-Ax).  This allows the node to compute how much
the robot must rotate to face the next target.
"""

import math
import os

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import Twist2DStamped
from sensor_msgs.msg import CameraInfo, CompressedImage

import astar
from aruco_detector import ArucoDetector
from calibration import CameraCalibration
from config import (
    BLIND_APPROACH_DIST,
    BLIND_FORWARD_SEC,
    BLIND_FORWARD_SPEED,
    GOAL_NODE,
    INITIAL_HEADING_RAD,
    NODE_COORDS,
    PROXIMITY_THRESHOLD_M,
    SEARCH_ANGULAR_SPEED,
    START_NODE,
    TAG_LOST_PATIENCE_FRAMES,
    WATCHDOG_HZ,
)
from navigation_controller import NavigationController

# Robot states
_SEARCHING     = "SEARCHING"
_APPROACHING   = "APPROACHING"
_BLIND_FORWARD = "BLIND_FORWARD"
_GOAL_REACHED  = "GOAL_REACHED"


def _normalize_angle(a: float) -> float:
    """Normalize angle to [-pi, pi]."""
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def _direction_between(from_node: int, to_node: int) -> float:
    """Return the heading (radians) from from_node to to_node in the grid frame."""
    fx, fy = NODE_COORDS[from_node]
    tx, ty = NODE_COORDS[to_node]
    return math.atan2(ty - fy, tx - fx)


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
        self._target_idx: int = 1          # heading toward path[1]
        self._state: str = _SEARCHING
        self._tag_lost_frames: int = 0
        self._last_tvec = None              # last seen tvec of target tag
        self._last_dist: float = 999.0      # last known distance to target tag
        self._blind_start = None            # rospy.Time when BLIND_FORWARD began

        # ── Heading tracking ──────────────────────────────────────────────────
        # Heading = robot's orientation in grid frame (radians).
        # 0 = facing +x (right), π/2 = facing +y (up)
        self._heading: float = INITIAL_HEADING_RAD

        # Pre-compute the required turn for each path segment
        self._log_planned_turns()

        # ── Sub-components ────────────────────────────────────────────────────
        self._calib = CameraCalibration(self._vehicle)
        self._detector = ArucoDetector()
        self._controller = NavigationController()

        # Cached optimal camera matrix (computed once on first calibrated frame).
        self._new_K = None

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

        # Duckiebot uses duckietown_msgs/Twist2DStamped on car_cmd_switch_node/cmd.
        self._cmd_pub = rospy.Publisher(
            f"{prefix}/car_cmd_switch_node/cmd",
            Twist2DStamped,
            queue_size=1,
        )

        # Watchdog: stop the robot if no image arrives for >1 s.
        self._last_image_stamp = rospy.Time.now()
        rospy.Timer(
            rospy.Duration(1.0 / WATCHDOG_HZ),
            self._watchdog_cb,
        )

        rospy.loginfo(
            "[Nav] Node ready. Heading=%.0f°  Target=N%d  State=%s",
            math.degrees(self._heading),
            self._path[self._target_idx],
            self._state,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _log_planned_turns(self) -> None:
        """Log the planned heading changes at each waypoint for debugging."""
        heading = self._heading
        for i in range(1, len(self._path)):
            prev = self._path[i - 1]
            curr = self._path[i]
            new_heading = _direction_between(prev, curr)
            turn = _normalize_angle(new_heading - heading)
            direction_str = "LEFT" if turn > 0 else "RIGHT" if turn < 0 else "STRAIGHT"
            rospy.loginfo(
                "[Nav] Plan: N%d→N%d  heading %.0f°→%.0f°  turn %.0f° %s",
                prev, curr,
                math.degrees(heading), math.degrees(new_heading),
                math.degrees(abs(turn)), direction_str,
            )
            heading = new_heading

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def _target_node(self) -> int:
        """The ID of the node the robot is currently heading toward."""
        if self._target_idx >= len(self._path):
            raise IndexError(
                f"_target_idx={self._target_idx} is out of range for path "
                f"of length {len(self._path)}. This is a bug."
            )
        return self._path[self._target_idx]

    @property
    def _search_direction(self) -> float:
        """
        Decide which way to rotate when searching.

        Uses heading tracking: computes the angle from the robot's current
        heading to the direction of the next target node.

        Returns +1.0 (CCW/left) or -1.0 (CW/right).
        """
        # If we saw the tag recently, turn toward where it was.
        if self._last_tvec is not None:
            lateral = float(self._last_tvec[0])
            if abs(lateral) > 0.005:
                return 1.0 if lateral < 0 else -1.0

        # Use heading tracking: which direction to face the next node?
        prev_node = self._path[self._target_idx - 1]
        tgt_node = self._target_node
        target_heading = _direction_between(prev_node, tgt_node)
        turn = _normalize_angle(target_heading - self._heading)

        rospy.loginfo_throttle(
            2.0,
            "[Nav] Search direction: heading=%.0f°  target_heading=%.0f°  turn=%.0f°  dir=%s",
            math.degrees(self._heading),
            math.degrees(target_heading),
            math.degrees(turn),
            "LEFT" if turn > 0 else "RIGHT",
        )

        return 1.0 if turn >= 0 else -1.0

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _image_cb(self, msg: CompressedImage) -> None:
        self._last_image_stamp = rospy.Time.now()

        if self._state == _GOAL_REACHED:
            self._cmd_pub.publish(self._controller.stop_cmd())
            return

        # ── BLIND_FORWARD: drive straight, ignore detections ─────────────────
        if self._state == _BLIND_FORWARD:
            elapsed = (rospy.Time.now() - self._blind_start).to_sec()
            if elapsed >= BLIND_FORWARD_SEC:
                rospy.loginfo(
                    "[Nav] Blind forward finished (%.1f s). Declaring node reached.",
                    elapsed,
                )
                self._on_node_reached(self._target_node)
            else:
                cmd = Twist2DStamped()
                cmd.v = BLIND_FORWARD_SPEED
                cmd.omega = 0.0
                self._cmd_pub.publish(cmd)
            return

        # ── Decode + undistort ───────────────────────────────────────────────
        np_arr = np.frombuffer(msg.data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            rospy.logwarn_throttle(5.0, "[Nav] Failed to decode image.")
            return

        if not self._calib.is_ready:
            self._cmd_pub.publish(self._controller.search_cmd(self._search_direction))
            return

        h, w = img.shape[:2]
        if self._new_K is None:
            self._new_K, _ = cv2.getOptimalNewCameraMatrix(
                self._calib.K, self._calib.D, (w, h), alpha=1, newImgSize=(w, h)
            )
        new_K = self._new_K
        undistorted = cv2.undistort(img, self._calib.K, self._calib.D, None, new_K)

        # ── ARTag detection ──────────────────────────────────────────────────
        detections = self._detector.detect(undistorted, new_K)

        # ── Control decision ─────────────────────────────────────────────────
        target_id = self._target_node

        if target_id in detections:
            _, tvec = detections[target_id]
            dist = self._controller.distance_m(tvec)
            self._tag_lost_frames = 0
            self._last_tvec = tvec.copy()
            self._last_dist = dist

            rospy.loginfo_throttle(
                1.0,
                "[Nav] Target N%d detected  dist=%.3f m  state=%s",
                target_id, dist, self._state,
            )

            if dist < PROXIMITY_THRESHOLD_M:
                self._on_node_reached(target_id)
            else:
                self._state = _APPROACHING
                cmd = self._controller.compute_cmd_vel(tvec)
                rospy.loginfo_throttle(
                    1.0,
                    "[Nav] CMD  v=%.3f  omega=%.3f",
                    cmd.v, cmd.omega,
                )
                self._cmd_pub.publish(cmd)
        else:
            # Target tag not visible
            self._tag_lost_frames += 1

            # Wait for patience frames before taking action
            if self._tag_lost_frames >= TAG_LOST_PATIENCE_FRAMES:
                if self._state == _APPROACHING and self._last_dist < BLIND_APPROACH_DIST:
                    # Tag disappeared while we were close → likely went under camera.
                    rospy.loginfo(
                        "[Nav] Tag N%d lost during APPROACHING (last dist=%.3f m). "
                        "BLIND_FORWARD for %.1f s.",
                        target_id, self._last_dist, BLIND_FORWARD_SEC,
                    )
                    self._state = _BLIND_FORWARD
                    self._blind_start = rospy.Time.now()
                    cmd = Twist2DStamped()
                    cmd.v = BLIND_FORWARD_SPEED
                    cmd.omega = 0.0
                    self._cmd_pub.publish(cmd)
                    return
                else:
                    if self._state != _SEARCHING:
                        rospy.logwarn(
                            "[Nav] Target N%d lost for %d frames (dist=%.3f m). "
                            "Switching to SEARCHING.",
                            target_id, self._tag_lost_frames, self._last_dist,
                        )
                    self._state = _SEARCHING

            # While waiting for patience or in SEARCHING, act accordingly
            if self._state == _SEARCHING:
                self._cmd_pub.publish(self._controller.search_cmd(self._search_direction))
            elif self._state == _APPROACHING:
                # Still within patience window – keep driving toward last known position
                if self._last_tvec is not None:
                    # YENİ EKLENEN KISIM: 
                    # Hedef anlık kaybolduğunda agresif açıyla dönmeye devam etme.
                    # İleri gitme hızını (v) koru ama dönüşü sıfırla ki dümdüz ilerlesin.
                    cmd = self._controller.compute_cmd_vel(self._last_tvec)
                    cmd.omega = 0.0  
                    self._cmd_pub.publish(cmd)
                else:
                    self._cmd_pub.publish(self._controller.stop_cmd())
            else:
                self._cmd_pub.publish(self._controller.stop_cmd())

    def _on_node_reached(self, node_id: int) -> None:
        """Handle arrival at a waypoint node."""
        rospy.loginfo("[Nav] ✓ Reached N%d.", node_id)
        self._cmd_pub.publish(self._controller.stop_cmd())

        # Update heading: direction of travel from previous node to this one
        prev_node = self._path[self._target_idx - 1]
        self._heading = _direction_between(prev_node, node_id)
        rospy.loginfo(
            "[Nav] Heading updated to %.0f° (from N%d→N%d)",
            math.degrees(self._heading), prev_node, node_id,
        )

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
            self._last_tvec = None
            self._last_dist = 999.0

            # Log what turn is needed for the next target
            next_node = self._target_node
            target_heading = _direction_between(node_id, next_node)
            turn = _normalize_angle(target_heading - self._heading)
            direction_str = "LEFT" if turn > 0 else "RIGHT" if turn < 0 else "STRAIGHT"

            rospy.loginfo(
                "[Nav] Next: N%d  need to turn %.0f° %s  then search.",
                next_node, math.degrees(abs(turn)), direction_str,
            )

    def _watchdog_cb(self, _event) -> None:
        """Stop the robot if no camera image has arrived for >1 second."""
        if self._state in (_GOAL_REACHED, _BLIND_FORWARD):
            return
        elapsed = (rospy.Time.now() - self._last_image_stamp).to_sec()
        if elapsed > 1.0:
            rospy.logwarn_throttle(
                2.0,
                "[Nav] No camera image for %.1f s. Publishing stop.",
                elapsed,
            )
            self._cmd_pub.publish(self._controller.stop_cmd())
