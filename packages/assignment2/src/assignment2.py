#!/usr/bin/env python3

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

ARUCO_DICT_TYPE = cv2.aruco.DICT_4X4_50
MARKER_SIZE_M = 0.065

# tag_id -> (x [m], y [m], yaw [rad]) in world frame
TAG_POSES = {
    0: (0.50, 0.00, 0.0),
    1: (1.00, 0.00, 0.0),
    2: (1.50, 0.00, 0.0),
    3: (0.50, 1.00, math.pi),
    4: (1.00, 1.00, math.pi),
    5: (1.50, 1.00, math.pi),
}

WHEEL_RADIUS = 0.0318
WHEEL_BASELINE = 0.1

MAP_WIDTH = 800
MAP_HEIGHT = 600
MAP_SCALE = 300
MAP_ORIGIN_X = 50
MAP_ORIGIN_Y = 500

PUBLISH_RATE_HZ = 30
ODOM_UPDATE_HZ = 50
TAG_LOG_PERIOD = 1.0

def world_to_pixel(x, y):
    px = int(MAP_ORIGIN_X + x * MAP_SCALE)
    py = int(MAP_ORIGIN_Y - y * MAP_SCALE)
    return px, py


def wrap_to_pi(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


class ArUcoLocalizationNode(DTROS):
    def __init__(self, node_name):
        super(ArUcoLocalizationNode, self).__init__(
            node_name=node_name,
            node_type=NodeType.GENERIC,
        )

        self._vehicle = os.environ.get("VEHICLE_NAME", "")
        self._prefix = f"/{self._vehicle}" if self._vehicle else ""

        self._K = None
        self._D = None
        self._camera_info_received = False

        self._aruco_dict = self._create_aruco_dict()
        self._aruco_params = self._create_aruco_params()
        self._aruco_detector = self._create_aruco_detector()

        self._x = 0.0
        self._y = 0.0
        self._theta = 0.0
        self._pose_source = "odometry"
        self._last_tag_seen_s = 0.0

        self._left_ticks = None
        self._right_ticks = None
        self._left_ticks_prev = None
        self._right_ticks_prev = None
        self._ticks_per_rev = 135

        self._latest_camera_img = None
        self._last_detected_ids = []
        self._last_tag_log_s = {}
        self._marker_size_m = rospy.get_param("~marker_size_m", MARKER_SIZE_M)
        self._camera_offset_x_m = rospy.get_param("~camera_offset_x_m", 0.0)
        self._camera_offset_y_m = rospy.get_param("~camera_offset_y_m", 0.0)
        self._tag_timeout_s = rospy.get_param("~tag_timeout_s", 0.5)
        self._tag_poses = self._load_tag_poses()

        self._load_camera_calibration()

        self._img_sub = rospy.Subscriber(
            f"{self._prefix}/camera_node/image/compressed",
            CompressedImage,
            self._image_cb,
            queue_size=1,
            buff_size=2**24,
        )

        self._ci_sub = rospy.Subscriber(
            f"{self._prefix}/camera_node/camera_info",
            CameraInfo,
            self._camera_info_cb,
            queue_size=1,
        )

        self._left_enc_sub = rospy.Subscriber(
            f"{self._prefix}/left_wheel_encoder_node/tick",
            WheelEncoderStamped,
            self._left_enc_cb,
            queue_size=1,
        )
        self._right_enc_sub = rospy.Subscriber(
            f"{self._prefix}/right_wheel_encoder_node/tick",
            WheelEncoderStamped,
            self._right_enc_cb,
            queue_size=1,
        )

        self._vis_pub = rospy.Publisher(
            f"{self._prefix}/assignment2/visualization/compressed",
            CompressedImage,
            queue_size=1,
        )
        self._tag_pose_pub = rospy.Publisher(
            f"{self._prefix}/assignment2/tag_pose",
            TransformStamped,
            queue_size=10,
        )

        self._vis_timer = rospy.Timer(
            rospy.Duration(1.0 / PUBLISH_RATE_HZ),
            self._publish_visualization,
        )
        self._odom_timer = rospy.Timer(
            rospy.Duration(1.0 / ODOM_UPDATE_HZ),
            self._odometry_timer_cb,
        )

        rospy.loginfo("[ArUcoLoc] Node initialised. Waiting for camera feed...")

    def _create_aruco_dict(self):
        if hasattr(cv2.aruco, "getPredefinedDictionary"):
            return cv2.aruco.getPredefinedDictionary(ARUCO_DICT_TYPE)
        return cv2.aruco.Dictionary_get(ARUCO_DICT_TYPE)

    def _create_aruco_params(self):
        if hasattr(cv2.aruco, "DetectorParameters"):
            return cv2.aruco.DetectorParameters()
        return cv2.aruco.DetectorParameters_create()

    def _create_aruco_detector(self):
        if hasattr(cv2.aruco, "ArucoDetector"):
            return cv2.aruco.ArucoDetector(self._aruco_dict, self._aruco_params)
        return None

    def _load_camera_calibration(self):
        calib_file = f"/data/config/calibrations/camera_intrinsic/{self._vehicle}.yaml"
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

    def _load_tag_poses(self):
        tag_poses_param = rospy.get_param("~tag_poses", None)
        if tag_poses_param is None:
            return dict(TAG_POSES)

        try:
            parsed = {}
            for key, value in tag_poses_param.items():
                if len(value) != 3:
                    raise ValueError(f"tag {key} should have [x, y, yaw]")
                parsed[int(key)] = (float(value[0]), float(value[1]), float(value[2]))
            if not parsed:
                raise ValueError("empty tag map")
            rospy.loginfo("[ArUcoLoc] Loaded %d tag poses from ~tag_poses.", len(parsed))
            return parsed
        except Exception as e:
            rospy.logwarn("[ArUcoLoc] Invalid ~tag_poses, using defaults: %s", e)
            return dict(TAG_POSES)

    def _camera_info_cb(self, msg):
        if self._camera_info_received:
            return
        self._K = np.array(msg.K).reshape(3, 3)
        self._D = np.array(msg.D)
        self._camera_info_received = True
        rospy.loginfo("[ArUcoLoc] Received intrinsics from CameraInfo topic.")

    def _left_enc_cb(self, msg):
        resolution = getattr(msg, "resolution", 0)
        if resolution > 0:
            self._ticks_per_rev = resolution
        if self._left_ticks_prev is None:
            self._left_ticks_prev = msg.data
        self._left_ticks = msg.data

    def _right_enc_cb(self, msg):
        resolution = getattr(msg, "resolution", 0)
        if resolution > 0:
            self._ticks_per_rev = resolution
        if self._right_ticks_prev is None:
            self._right_ticks_prev = msg.data
        self._right_ticks = msg.data

    def _update_odometry(self):
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
        self._theta = wrap_to_pi(self._theta + d_theta)

    def _odometry_timer_cb(self, _event):
        self._update_odometry()
        if rospy.get_time() - self._last_tag_seen_s > self._tag_timeout_s:
            self._pose_source = "odometry"

    def _detect_markers(self, image):
        if self._aruco_detector is not None:
            return self._aruco_detector.detectMarkers(image)
        return cv2.aruco.detectMarkers(
            image, self._aruco_dict, parameters=self._aruco_params
        )

    def _draw_axis(self, image, camera_matrix, dist_coeffs, rvec, tvec):
        if hasattr(cv2.aruco, "drawAxis"):
            cv2.aruco.drawAxis(
                image, camera_matrix, dist_coeffs, rvec, tvec, self._marker_size_m * 0.5
            )
            return
        cv2.drawFrameAxes(
            image, camera_matrix, dist_coeffs, rvec, tvec, self._marker_size_m * 0.5
        )

    def _publish_tag_pose(self, header, tag_id, rvec, tvec):
        transform = TransformStamped()
        transform.header.stamp = header.stamp
        transform.header.frame_id = header.frame_id or "camera"
        transform.child_frame_id = f"aruco_{tag_id}"
        transform.transform.translation.x = float(tvec[0])
        transform.transform.translation.y = float(tvec[1])
        transform.transform.translation.z = float(tvec[2])

        rot_mat, _ = cv2.Rodrigues(rvec)
        tf_mat = np.eye(4)
        tf_mat[:3, :3] = rot_mat
        qx, qy, qz, qw = tft.quaternion_from_matrix(tf_mat)
        transform.transform.rotation.x = qx
        transform.transform.rotation.y = qy
        transform.transform.rotation.z = qz
        transform.transform.rotation.w = qw

        self._tag_pose_pub.publish(transform)

    def _maybe_log_tag_pose(self, tag_id, rvec, tvec):
        now_s = rospy.get_time()
        last_s = self._last_tag_log_s.get(tag_id, 0.0)
        if now_s - last_s < TAG_LOG_PERIOD:
            return
        self._last_tag_log_s[tag_id] = now_s
        rospy.loginfo(
            "[ArUcoLoc] id=%d t=[%.3f %.3f %.3f] r=[%.3f %.3f %.3f]",
            tag_id,
            tvec[0],
            tvec[1],
            tvec[2],
            rvec[0],
            rvec[1],
            rvec[2],
        )

    def _image_cb(self, msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            return

        if not self._camera_info_received:
            self._latest_camera_img = img
            return

        undistorted = cv2.undistort(img, self._K, self._D, None, self._K)
        rectified_d = np.zeros_like(self._D)

        corners, ids, _ = self._detect_markers(undistorted)

        if ids is not None and len(ids) > 0:
            self._last_detected_ids = [int(x) for x in ids.flatten()]
            cv2.aruco.drawDetectedMarkers(undistorted, corners, ids)

            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, self._marker_size_m, self._K, rectified_d
            )

            best_tag_id = None
            best_distance = float("inf")
            best_rvec = None
            best_tvec = None

            for i, tag_id in enumerate(ids.flatten()):
                tag_id_int = int(tag_id)
                rvec = rvecs[i][0]
                tvec = tvecs[i][0]

                self._draw_axis(undistorted, self._K, rectified_d, rvec, tvec)
                self._publish_tag_pose(msg.header, tag_id_int, rvec, tvec)
                self._maybe_log_tag_pose(tag_id_int, rvec, tvec)

                dist = np.linalg.norm(tvec)
                if tag_id_int in self._tag_poses and dist < best_distance:
                    best_distance = dist
                    best_tag_id = tag_id_int
                    best_rvec = rvec
                    best_tvec = tvec

            if best_tag_id is not None:
                self._update_pose_from_tag(best_tag_id, best_rvec, best_tvec)
                self._pose_source = "aruco"
                self._last_tag_seen_s = rospy.get_time()
            else:
                # Tags are visible but none are mapped for global localization.
                self._pose_source = "odometry"
        else:
            self._last_detected_ids = []
            self._pose_source = "odometry"

        self._latest_camera_img = undistorted

    def _update_pose_from_tag(self, tag_id, rvec, tvec):
        tag_x, tag_y, tag_yaw = self._tag_poses[tag_id]

        R_cm, _ = cv2.Rodrigues(rvec)

        cam_in_marker = -R_cm.T @ tvec

        yaw_cam_to_marker = math.atan2(-R_cm[2, 0], R_cm[2, 2])

        cos_ty = math.cos(tag_yaw)
        sin_ty = math.sin(tag_yaw)

        cam_wx = tag_x + cos_ty * cam_in_marker[0] - sin_ty * cam_in_marker[2]
        cam_wy = tag_y + sin_ty * cam_in_marker[0] + cos_ty * cam_in_marker[2]

        cam_world_yaw = wrap_to_pi(tag_yaw + math.pi + yaw_cam_to_marker)

        cos_c = math.cos(cam_world_yaw)
        sin_c = math.sin(cam_world_yaw)

        self._x = cam_wx - (cos_c * self._camera_offset_x_m - sin_c * self._camera_offset_y_m)
        self._y = cam_wy - (sin_c * self._camera_offset_x_m + cos_c * self._camera_offset_y_m)
        self._theta = cam_world_yaw

        self._left_ticks_prev = self._left_ticks
        self._right_ticks_prev = self._right_ticks

    def _publish_visualization(self, _event):
        if self._latest_camera_img is not None:
            cam_panel = cv2.resize(self._latest_camera_img, (400, 300))
        else:
            cam_panel = np.zeros((300, 400, 3), dtype=np.uint8)
            cv2.putText(
                cam_panel, "Waiting for camera...", (30, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
            )

        if self._pose_source == "aruco":
            status_text = "POSE: ArUco"
            status_colour = (0, 200, 0)
        else:
            status_text = "POSE: Fallback (Odom)"
            status_colour = (0, 100, 255)
        cv2.putText(
            cam_panel,
            status_text,
            (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            status_colour,
            2,
        )

        if self._last_detected_ids:
            ids_text = "Tags: " + ",".join(str(tid) for tid in self._last_detected_ids)
        else:
            ids_text = "Tags: none"
        cv2.putText(
            cam_panel,
            ids_text,
            (8, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
        )

        map_panel = np.ones((300, 400, 3), dtype=np.uint8) * 240

        for tid, (tx, ty, _tyaw) in self._tag_poses.items():
            px, py = world_to_pixel(tx, ty)
            px_s = int(px * 400 / MAP_WIDTH)
            py_s = int(py * 300 / MAP_HEIGHT)
            cv2.rectangle(map_panel, (px_s - 6, py_s - 6), (px_s + 6, py_s + 6), (0, 0, 0), -1)
            cv2.putText(
                map_panel, str(tid), (px_s + 8, py_s + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1,
            )

        rpx, rpy = world_to_pixel(self._x, self._y)
        rpx_s = int(rpx * 400 / MAP_WIDTH)
        rpy_s = int(rpy * 300 / MAP_HEIGHT)

        if self._pose_source == "aruco":
            colour = (0, 200, 0)
            label = "ArUco"
        else:
            colour = (0, 100, 255)
            label = "Odom"

        cv2.circle(map_panel, (rpx_s, rpy_s), 8, colour, -1)

        arrow_len = 20
        ax = int(rpx_s + arrow_len * math.cos(self._theta))
        ay = int(rpy_s - arrow_len * math.sin(self._theta))
        cv2.arrowedLine(map_panel, (rpx_s, rpy_s), (ax, ay), colour, 2, tipLength=0.4)

        cv2.putText(
            map_panel,
            f"{label}  x={self._x:.2f} y={self._y:.2f} th={math.degrees(self._theta):.0f}deg",
            (5, 290),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1,
        )

        cv2.circle(map_panel, (10, 15), 5, (0, 200, 0), -1)
        cv2.putText(map_panel, "ArUco", (20, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
        cv2.circle(map_panel, (80, 15), 5, (0, 100, 255), -1)
        cv2.putText(map_panel, "Odom", (90, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)

        vis = np.vstack([cam_panel, map_panel])

        out_msg = CompressedImage()
        out_msg.header.stamp = rospy.Time.now()
        out_msg.format = "jpeg"
        out_msg.data = np.array(cv2.imencode(".jpg", vis)[1]).tobytes()
        self._vis_pub.publish(out_msg)


if __name__ == "__main__":
    node = ArUcoLocalizationNode(node_name="aruco_localization_node")
    rospy.spin()
