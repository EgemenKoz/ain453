"""
localization_node.py – Main DTROS node for ArUco-based localisation.

Wires together calibration, odometry, ArUco detection, pose estimation,
and visualisation into a single ROS node.
"""

import math
import os

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import WheelEncoderStamped
from sensor_msgs.msg import CameraInfo, CompressedImage

from aruco_detector import ArucoDetector
from calibration import CameraCalibration
from config import PUBLISH_RATE_HZ, TAG_POSES
from odometry import WheelOdometry
from pose_estimator import PoseEstimator
from visualizer import Visualizer


class ArUcoLocalizationNode(DTROS):
    def __init__(self, node_name: str):
        super().__init__(node_name=node_name, node_type=NodeType.GENERIC)

        self._vehicle = os.environ.get("VEHICLE_NAME", "")
        self._bridge = CvBridge()

        # Sub-components
        self._calib = CameraCalibration(self._vehicle)
        self._odom = WheelOdometry()
        self._detector = ArucoDetector()

        # World-frame robot pose
        self._x: float = 0.0
        self._y: float = 0.0
        self._theta: float = 0.0
        self._pose_source: str = "odometry"

        # Latest annotated camera frame for visualisation
        self._camera_img = None

        # ROS topic prefix
        prefix = f"/{self._vehicle}" if self._vehicle else ""

        # ── Subscribers ───────────────────────────────────────────────────────
        rospy.Subscriber(
            f"/mouse/camera_node/image/compressed",
            CompressedImage,
            self._image_cb,
            queue_size=1,
            buff_size=2 ** 24,
        )
        rospy.Subscriber(
            f"/mouse/camera_node/camera_info",
            CameraInfo,
            self._calib.update_from_camera_info,
            queue_size=1,
        )
        rospy.Subscriber(
            f"/mouse/left_wheel_encoder_node/tick",
            WheelEncoderStamped,
            lambda msg: self._odom.update_left(msg.data),
            queue_size=1,
        )
        rospy.Subscriber(
            f"/mouse/right_wheel_encoder_node/tick",
            WheelEncoderStamped,
            lambda msg: self._odom.update_right(msg.data),
            queue_size=1,
        )

        # ── Publisher + visualisation timer ──────────────────────────────────
        vis_pub = rospy.Publisher(
            f"/mouse/assignment2/visualization/compressed",
            CompressedImage,
            queue_size=1,
        )
        self._vis = Visualizer(vis_pub)

        rospy.Timer(
            rospy.Duration(1.0 / PUBLISH_RATE_HZ),
            self._vis_timer_cb,
        )

        rospy.loginfo("[ArUcoLoc] Node ready. Waiting for camera feed...")

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _image_cb(self, msg: CompressedImage) -> None:
        # Decode compressed JPEG
        # rospy.loginfo_throttle(2.0, "[ArUcoLoc] Image callback triggered.")  # EKLE

        np_arr = np.frombuffer(msg.data, np.uint8)
        # rospy.loginfo_throttle(2.0, "[ArUcoLoc] np_arr len=%d", len(np_arr))
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            rospy.logwarn_throttle(2.0, "[ArUcoLoc] imdecode returned None!")
            return
        # rospy.loginfo_throttle(2.0, "[ArUcoLoc] img shape=%s", str(img.shape))
        if img is None:
            rospy.logwarn_throttle(5.0, "[ArUcoLoc] Failed to decode image.")
            return

        # Always integrate odometry (even without camera calibration)
        d_center, d_theta = self._odom.compute_delta()
        self._x += d_center * math.cos(self._theta + d_theta / 2.0)
        self._y += d_center * math.sin(self._theta + d_theta / 2.0)
        self._theta += d_theta

        if not self._calib.is_ready:
            self._camera_img = img
            return

        # Undistort
        h, w = img.shape[:2]
        new_K, _ = cv2.getOptimalNewCameraMatrix(
            self._calib.K, self._calib.D, (w, h), alpha=1, newImgSize=(w, h)
        )
        undistorted = cv2.undistort(img, self._calib.K, self._calib.D, None, new_K)

        # ArUco detection + annotation (draws on undistorted in-place)
        tag_id, rvec, tvec = self._detector.detect_and_annotate(undistorted, new_K)

        if tag_id is not None:
            if tag_id not in TAG_POSES:
                # First time seeing this tag: register its world pose from odometry
                tx, ty, tyaw = PoseEstimator.tag_from_robot(
                    self._x, self._y, self._theta, rvec, tvec
                )
                TAG_POSES[tag_id] = (tx, ty, tyaw)
                rospy.loginfo(
                    "[ArUcoLoc] Registered tag %d at (%.2f, %.2f, %.1f°)",
                    tag_id, tx, ty, math.degrees(tyaw),
                )
            else:
                # Subsequent sightings: correct pose from known tag
                self._x, self._y, self._theta = PoseEstimator.from_tag(tag_id, rvec, tvec)
                self._pose_source = "aruco"
                self._odom.reset_to_current()
                rospy.loginfo_throttle(2.0, "[ArUcoLoc] ID %d  x=%.2f y=%.2f yaw=%.0f°",
                                       tag_id, self._x, self._y, math.degrees(self._theta))
        else:
            self._pose_source = "odometry"

        self._camera_img = undistorted

    def _vis_timer_cb(self, _event) -> None:
        self._vis.publish(
            self._camera_img,
            self._x,
            self._y,
            self._theta,
            self._pose_source,
        )

        # Local display window (works when a desktop/display is available)
        # vis_img = self._vis.last_frame
        # if vis_img is not None:
        #     cv2.imshow("ArUco Localization", vis_img)
        #     cv2.waitKey(1)
