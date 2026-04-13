"""
calibration.py – Camera intrinsic calibration loader.

Tries (in order):
  1. /data/config/calibrations/camera_intrinsic/<vehicle>.yaml
  2. /data/config/calibrations/camera_intrinsic/default.yaml
  3. CameraInfo topic  (via update_from_camera_info callback)
"""

import os

import numpy as np
import rospy
import yaml
from sensor_msgs.msg import CameraInfo


class CameraCalibration:
    def __init__(self, vehicle_name: str):
        self._K: np.ndarray | None = None
        self._D: np.ndarray | None = None
        self._ready = False
        self._load_from_file(vehicle_name)

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def K(self) -> np.ndarray:
        """3×3 camera matrix."""
        return self._K

    @property
    def D(self) -> np.ndarray:
        """Distortion coefficient vector."""
        return self._D

    def update_from_camera_info(self, msg: CameraInfo) -> None:
        """ROS subscriber callback – populates K/D from a CameraInfo message."""
        if self._ready:
            return
        self._K = np.array(msg.K, dtype=np.float64).reshape(3, 3)
        self._D = np.array(msg.D, dtype=np.float64)
        self._ready = True
        rospy.loginfo("[Calibration] Received intrinsics from /camera_info topic.")

    # ── Private ───────────────────────────────────────────────────────────────

    def _load_from_file(self, vehicle_name: str) -> None:
        candidates = [
            f"/data/config/calibrations/camera_intrinsic/{vehicle_name}.yaml",
            "/data/config/calibrations/camera_intrinsic/default.yaml",
        ]
        for path in candidates:
            if not os.path.isfile(path):
                continue
            try:
                with open(path, "r") as fh:
                    calib = yaml.safe_load(fh)
                self._K = np.array(
                    calib["camera_matrix"]["data"], dtype=np.float64
                ).reshape(3, 3)
                self._D = np.array(
                    calib["distortion_coefficients"]["data"], dtype=np.float64
                )
                self._ready = True
                rospy.loginfo("[Calibration] Loaded intrinsics from %s", path)
                return
            except Exception as exc:
                rospy.logwarn("[Calibration] Could not parse %s: %s", path, exc)

        rospy.logwarn(
            "[Calibration] No calibration file found; "
            "waiting for /camera_info topic."
        )
