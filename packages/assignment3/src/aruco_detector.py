"""
aruco_detector.py – ArUco / APrilTag marker detection and pose estimation.

Key difference from the A2 version: detect() returns ALL detected markers as a
dict {tag_id: (rvec, tvec)} so the navigation node can look up a specific tag
by ID rather than receiving only the closest one.

Handles OpenCV API differences between older (<4.7) and newer (>=4.7) versions:
  - Dictionary / DetectorParameters creation
  - detectMarkers (old API) vs ArucoDetector class (new API)
  - drawAxis (deprecated) vs drawFrameAxes
  - estimatePoseSingleMarkers (removed in 4.9) vs solvePnP per-marker
"""

from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import rospy

from config import ARUCO_DICT_TYPE, MARKER_SIZE_M

# 3-D corners of a flat marker in marker frame (z = 0).
# Order: top-left, top-right, bottom-right, bottom-left.
_MARKER_OBJ_PTS = np.array(
    [
        [-MARKER_SIZE_M / 2,  MARKER_SIZE_M / 2, 0.0],
        [ MARKER_SIZE_M / 2,  MARKER_SIZE_M / 2, 0.0],
        [ MARKER_SIZE_M / 2, -MARKER_SIZE_M / 2, 0.0],
        [-MARKER_SIZE_M / 2, -MARKER_SIZE_M / 2, 0.0],
    ],
    dtype=np.float32,
)


def _make_detector():
    """Build dictionary + parameters, handling OpenCV < 4.7 and >= 4.7."""
    try:
        # OpenCV >= 4.7
        aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_TYPE)
        params = cv2.aruco.DetectorParameters()
        params.minMarkerPerimeterRate = 0.05
        params.errorCorrectionRate = 0.5
        params.polygonalApproxAccuracyRate = 0.03
        detector = cv2.aruco.ArucoDetector(aruco_dict, params)
        return aruco_dict, params, detector, True
    except AttributeError:
        # OpenCV < 4.7
        aruco_dict = cv2.aruco.Dictionary_get(ARUCO_DICT_TYPE)
        params = cv2.aruco.DetectorParameters_create()
        params.minMarkerPerimeterRate = 0.05
        params.errorCorrectionRate = 0.5
        params.polygonalApproxAccuracyRate = 0.03
        return aruco_dict, params, None, False


def _draw_axes(
    img: np.ndarray,
    K: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    length: float,
) -> None:
    """Draw coordinate axes; compatible with OpenCV 4.6 and 4.7+."""
    D_zeros = np.zeros((4, 1), dtype=np.float64)
    try:
        cv2.drawFrameAxes(img, K, D_zeros, rvec, tvec, length)
    except AttributeError:
        cv2.aruco.drawAxis(img, K, D_zeros, rvec, tvec, length)


def _estimate_pose_single(
    corner: np.ndarray,
    K: np.ndarray,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Estimate pose of one marker via solvePnP.

    Parameters
    ----------
    corner : (1, 4, 2) array – corners from detectMarkers
    K      : (3, 3) camera matrix

    Returns
    -------
    (rvec, tvec) as (3,) arrays, or (None, None) on failure.
    """
    img_pts = corner[0].astype(np.float32)   # (4, 2)
    D_zeros = np.zeros((4, 1), dtype=np.float64)
    success, rvec, tvec = cv2.solvePnP(
        _MARKER_OBJ_PTS, img_pts, K, D_zeros,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if not success:
        return None, None
    return rvec.flatten(), tvec.flatten()


class ArucoDetector:
    """Detects ArUco markers and returns all their camera-frame poses."""

    def __init__(self):
        self._dict, self._params, self._detector, self._new_api = _make_detector()
        api_label = "new (>=4.7)" if self._new_api else "legacy (<4.7)"
        rospy.loginfo("[ArUco] Initialised with OpenCV %s API.", api_label)

    def detect(
        self,
        img: np.ndarray,
        K: np.ndarray,
    ) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """
        Detect all ArUco markers, draw overlays on *img* (in-place), and
        return every marker's pose.

        Parameters
        ----------
        img : undistorted BGR image (modified in-place with annotations)
        K   : optimal camera matrix used for undistortion

        Returns
        -------
        dict mapping  tag_id (int) -> (rvec, tvec)
            rvec : (3,) Rodrigues rotation vector in camera frame
            tvec : (3,) translation vector in camera frame (metres)
                    x = right,  y = down,  z = forward into the scene
        Returns an empty dict when no markers are found.
        """
        if self._new_api:
            corners, ids, _ = self._detector.detectMarkers(img)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(
                img, self._dict, parameters=self._params
            )

        if ids is None or len(ids) == 0:
            return {}

        cv2.aruco.drawDetectedMarkers(img, corners, ids)

        detections: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

        for i, tag_id in enumerate(ids.flatten()):
            rvec, tvec = _estimate_pose_single(corners[i], K)
            if rvec is None:
                continue

            _draw_axes(img, K, rvec, tvec, MARKER_SIZE_M * 0.5)

            dist = float(np.linalg.norm(tvec))
            rospy.loginfo_throttle(
                1.0,
                "[ArUco] Tag %d  tvec=[%.3f, %.3f, %.3f]  dist=%.3f m",
                tag_id, tvec[0], tvec[1], tvec[2], dist,
            )

            detections[int(tag_id)] = (rvec, tvec)

        return detections
