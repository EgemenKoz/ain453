#!/usr/bin/env python3

"""
Assignment 2 – ArUco-Based Localization on the Duckiebot
=========================================================
1. Detects ArUco markers from the camera feed.
2. Estimates robot pose relative to each detected tag.
3. Falls back to wheel-odometry when no tag is visible.
4. Provides a live visualization (camera + top-down map).
"""

import os
import math
import yaml
import numpy as np

import rospy
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage, CameraInfo
from duckietown_msgs.msg import WheelEncoderStamped
from geometry_msgs.msg import TransformStamped
import tf.transformations as tft

import cv2
from cv_bridge import CvBridge

# ── Configuration ────────────────────────────────────────────────────────────

ARUCO_DICT_TYPE = cv2.aruco.DICT_4X4_50
MARKER_SIZE_M = 0.065  # physical side length of the ArUco tags (metres)

# Known tag poses on the map  (tag_id → (x, y, yaw_rad) in world frame)
# Update these according to your actual lab / map layout.
TAG_POSES = {
    0: (0.50, 0.00, 0.0),
    1: (1.00, 0.00, 0.0),
    2: (1.50, 0.00, 0.0),
    3: (0.50, 1.00, math.pi),
    4: (1.00, 1.00, math.pi),
    5: (1.50, 1.00, math.pi),
}

# Duckiebot wheel parameters (metres)
WHEEL_RADIUS = 0.0318
WHEEL_BASELINE = 0.1

# Visualization
MAP_WIDTH = 800
MAP_HEIGHT = 600
MAP_SCALE = 300  # pixels per metre
MAP_ORIGIN_X = 50  # pixel offset for world x=0
MAP_ORIGIN_Y = 500  # pixel offset for world y=0

PUBLISH_RATE_HZ = 30


# ── Helper: world → pixel ────────────────────────────────────────────────────

def world_to_pixel(x, y):
    px = int(MAP_ORIGIN_X + x * MAP_SCALE)
    py = int(MAP_ORIGIN_Y - y * MAP_SCALE)
    return (px, py)


# ── Main Node ────────────────────────────────────────────────────────────────

class ArUcoLocalizationNode(DTROS):
    def __init__(self, node_name):
        super(ArUcoLocalizationNode, self).__init__(
            node_name=node_name,
            node_type=NodeType.GENERIC,
        )

        self._vehicle = os.environ.get("VEHICLE_NAME", "")
        self._bridge = CvBridge()

        # Camera intrinsics (will be populated from CameraInfo or calibration)
        self._K = None
        self._D = None
        self._camera_info_received = False

        # ArUco detector
        self._aruco_dict = cv2.aruco.Dictionary_get(ARUCO_DICT_TYPE)
        self._aruco_params = cv2.aruco.DetectorParameters_create()

        # ── Robot pose state (world frame) ──
        self._x = 0.0
        self._y = 0.0
        self._theta = 0.0
        self._pose_source = "odometry"  # "aruco" or "odometry"

        # ── Wheel-odometry state ──
        self._left_ticks = None
        self._right_ticks = None
        self._left_ticks_prev = None
        self._right_ticks_prev = None
        self._ticks_per_rev = 135  # DB21 encoder resolution

        # ── Visualisation images ──
        self._latest_camera_img = None

        # ── Load calibration ──
        self._load_camera_calibration()

        # ── Topics ──
        vn = self._vehicle
        prefix = f"/{vn}" if vn else ""

        # Camera (compressed)
        self._img_sub = rospy.Subscriber(
            f"{prefix}/camera_node/image/compressed",
            CompressedImage,
            self._image_cb,
            queue_size=1,
            buff_size=2**24,
        )

        # Camera info (backup for intrinsics)
        self._ci_sub = rospy.Subscriber(
            f"{prefix}/camera_node/camera_info",
            CameraInfo,
            self._camera_info_cb,
            queue_size=1,
        )

        # Wheel encoders
        self._left_enc_sub = rospy.Subscriber(
            f"{prefix}/left_wheel_encoder_node/tick",
            WheelEncoderStamped,
            self._left_enc_cb,
            queue_size=1,
        )
        self._right_enc_sub = rospy.Subscriber(
            f"{prefix}/right_wheel_encoder_node/tick",
            WheelEncoderStamped,
            self._right_enc_cb,
            queue_size=1,
        )

        # Visualisation publisher (compressed, viewable via rqt_image_view)
        self._vis_pub = rospy.Publisher(
            f"{prefix}/assignment2/visualization/compressed",
            CompressedImage,
            queue_size=1,
        )

        # Timer for publishing visualization
        self._vis_timer = rospy.Timer(
            rospy.Duration(1.0 / PUBLISH_RATE_HZ),
            self._publish_visualization,
        )

        rospy.loginfo("[ArUcoLoc] Node initialised. Waiting for camera feed...")

    # ── Camera calibration ───────────────────────────────────────────────────

    def _load_camera_calibration(self):
        """Try to load intrinsic calibration from the Duckiebot filesystem."""
        vn = self._vehicle
        calib_file = f"/data/config/calibrations/camera_intrinsic/{vn}.yaml"
        if not os.path.isfile(calib_file):
            calib_file = "/data/config/calibrations/camera_intrinsic/default.yaml"

        if os.path.isfile(calib_file):
            try:
                with open(calib_file, "r") as f:
                    calib = yaml.safe_load(f)
                K = np.array(calib["camera_matrix"]["data"]).reshape(3, 3)
                D = np.array(calib["distortion_coefficients"]["data"])
                self._K = K
                self._D = D
                self._camera_info_received = True
                rospy.loginfo("[ArUcoLoc] Loaded intrinsics from %s", calib_file)
            except Exception as e:
                rospy.logwarn("[ArUcoLoc] Failed to parse calibration: %s", e)
        else:
            rospy.logwarn("[ArUcoLoc] No calibration file found; waiting for CameraInfo.")

    def _camera_info_cb(self, msg):
        if self._camera_info_received:
            return
        self._K = np.array(msg.K).reshape(3, 3)
        self._D = np.array(msg.D)
        self._camera_info_received = True
        rospy.loginfo("[ArUcoLoc] Received intrinsics from CameraInfo topic.")

    # ── Wheel encoder callbacks ──────────────────────────────────────────────

    def _left_enc_cb(self, msg):
        if self._left_ticks_prev is None:
            self._left_ticks_prev = msg.data
        self._left_ticks = msg.data

    def _right_enc_cb(self, msg):
        if self._right_ticks_prev is None:
            self._right_ticks_prev = msg.data
        self._right_ticks = msg.data

    def _update_odometry(self):
        """Differential-drive odometry from wheel encoder ticks."""
        if (
            self._left_ticks is None
            or self._right_ticks is None
            or self._left_ticks_prev is None
            or self._right_ticks_prev is None
        ):
            return

        dl = (self._left_ticks - self._left_ticks_prev) / self._ticks_per_rev * (2 * math.pi * WHEEL_RADIUS)
        dr = (self._right_ticks - self._right_ticks_prev) / self._ticks_per_rev * (2 * math.pi * WHEEL_RADIUS)

        self._left_ticks_prev = self._left_ticks
        self._right_ticks_prev = self._right_ticks

        d_center = (dl + dr) / 2.0
        d_theta = (dr - dl) / WHEEL_BASELINE

        self._x += d_center * math.cos(self._theta + d_theta / 2.0)
        self._y += d_center * math.sin(self._theta + d_theta / 2.0)
        self._theta += d_theta

    # ── Image callback (main pipeline) ───────────────────────────────────────

    def _image_cb(self, msg):
        # Decode compressed image
        np_arr = np.frombuffer(msg.data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            return

        # Always update odometry
        self._update_odometry()

        if not self._camera_info_received:
            self._latest_camera_img = img
            return

        # Undistort
        h, w = img.shape[:2]
        new_K, roi = cv2.getOptimalNewCameraMatrix(self._K, self._D, (w, h), 1, (w, h))
        undistorted = cv2.undistort(img, self._K, self._D, None, new_K)

        # Detect ArUco markers
        corners, ids, _ = cv2.aruco.detectMarkers(
            undistorted, self._aruco_dict, parameters=self._aruco_params
        )

        if ids is not None and len(ids) > 0:
            # Draw detected markers
            cv2.aruco.drawDetectedMarkers(undistorted, corners, ids)

            # Pose estimation for each marker
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, MARKER_SIZE_M, new_K, np.zeros(5)
            )

            best_tag_id = None
            best_distance = float("inf")
            best_rvec = None
            best_tvec = None

            for i, tag_id in enumerate(ids.flatten()):
                rvec = rvecs[i][0]
                tvec = tvecs[i][0]

                # Draw axis on image
                cv2.aruco.drawAxis(undistorted, new_K, np.zeros(5), rvec, tvec, MARKER_SIZE_M * 0.5)

                rospy.loginfo_throttle(
                    1.0,
                    "[ArUcoLoc] Tag %d  t=[%.3f, %.3f, %.3f]",
                    tag_id, tvec[0], tvec[1], tvec[2],
                )

                # Choose closest tag for pose update
                dist = np.linalg.norm(tvec)
                if tag_id in TAG_POSES and dist < best_distance:
                    best_distance = dist
                    best_tag_id = tag_id
                    best_rvec = rvec
                    best_tvec = tvec

            # Update world pose from the best (closest) visible known tag
            if best_tag_id is not None:
                self._update_pose_from_tag(best_tag_id, best_rvec, best_tvec)
                self._pose_source = "aruco"
        else:
            self._pose_source = "odometry"

        self._latest_camera_img = undistorted

    # ── Pose from ArUco ──────────────────────────────────────────────────────

    def _update_pose_from_tag(self, tag_id, rvec, tvec):
        """Compute world-frame robot pose from a detected ArUco tag."""
        tag_x, tag_y, tag_yaw = TAG_POSES[tag_id]

        # Rotation matrix: camera ← marker
        R_cm, _ = cv2.Rodrigues(rvec)

        # Camera position in marker frame:  p_marker = -R_cm^T * t
        cam_in_marker = -R_cm.T @ tvec

        # Robot heading relative to marker (extract yaw from rotation)
        # The camera looks along +Z on the Duckiebot, so yaw in the
        # camera frame corresponds to rotation around the Y axis of the marker.
        yaw_cam_to_marker = math.atan2(-R_cm[2, 0], R_cm[2, 2])

        # Transform to world frame using the known tag pose
        cos_ty = math.cos(tag_yaw)
        sin_ty = math.sin(tag_yaw)

        # Marker frame: x-right, z-forward for the tag face
        # cam_in_marker: x-right, y-down, z-forward (marker frame)
        wx = tag_x + cos_ty * cam_in_marker[2] - sin_ty * cam_in_marker[0]
        wy = tag_y + sin_ty * cam_in_marker[2] + cos_ty * cam_in_marker[0]

        world_yaw = tag_yaw + math.pi + yaw_cam_to_marker  # facing toward the tag

        self._x = wx
        self._y = wy
        self._theta = world_yaw

        # Reset encoder baselines so odometry starts fresh from this correction
        self._left_ticks_prev = self._left_ticks
        self._right_ticks_prev = self._right_ticks

    # ── Visualisation ────────────────────────────────────────────────────────

    def _publish_visualization(self, _event):
        """Combine camera image and top-down map into a single visualisation."""
        # ── Camera panel ──
        if self._latest_camera_img is not None:
            cam_panel = cv2.resize(self._latest_camera_img, (400, 300))
        else:
            cam_panel = np.zeros((300, 400, 3), dtype=np.uint8)
            cv2.putText(
                cam_panel, "Waiting for camera...", (30, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
            )

        # ── Map panel ──
        map_panel = np.ones((300, 400, 3), dtype=np.uint8) * 240  # light grey bg

        # Draw known tags on map
        for tid, (tx, ty, _tyaw) in TAG_POSES.items():
            px, py = world_to_pixel(tx, ty)
            # Rescale to fit the small panel
            px_s = int(px * 400 / MAP_WIDTH)
            py_s = int(py * 300 / MAP_HEIGHT)
            cv2.rectangle(map_panel, (px_s - 6, py_s - 6), (px_s + 6, py_s + 6), (0, 0, 0), -1)
            cv2.putText(
                map_panel, str(tid), (px_s + 8, py_s + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1,
            )

        # Draw robot position
        rpx, rpy = world_to_pixel(self._x, self._y)
        rpx_s = int(rpx * 400 / MAP_WIDTH)
        rpy_s = int(rpy * 300 / MAP_HEIGHT)

        if self._pose_source == "aruco":
            colour = (0, 200, 0)   # green = ArUco-based
            label = "ArUco"
        else:
            colour = (0, 100, 255)  # orange = odometry fallback
            label = "Odom"

        cv2.circle(map_panel, (rpx_s, rpy_s), 8, colour, -1)

        # Draw heading arrow
        arrow_len = 20
        ax = int(rpx_s + arrow_len * math.cos(self._theta))
        ay = int(rpy_s - arrow_len * math.sin(self._theta))
        cv2.arrowedLine(map_panel, (rpx_s, rpy_s), (ax, ay), colour, 2, tipLength=0.4)

        # Pose text
        cv2.putText(
            map_panel,
            f"{label}  x={self._x:.2f} y={self._y:.2f} th={math.degrees(self._theta):.0f}deg",
            (5, 290),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1,
        )

        # Legend
        cv2.circle(map_panel, (10, 15), 5, (0, 200, 0), -1)
        cv2.putText(map_panel, "ArUco", (20, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
        cv2.circle(map_panel, (80, 15), 5, (0, 100, 255), -1)
        cv2.putText(map_panel, "Odom", (90, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)

        # ── Stack vertically ──
        vis = np.vstack([cam_panel, map_panel])  # 600×400

        # Publish as compressed image
        out_msg = CompressedImage()
        out_msg.header.stamp = rospy.Time.now()
        out_msg.format = "jpeg"
        out_msg.data = np.array(cv2.imencode(".jpg", vis)[1]).tobytes()
        self._vis_pub.publish(out_msg)


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    node = ArUcoLocalizationNode(node_name="aruco_localization_node")
    rospy.spin()
