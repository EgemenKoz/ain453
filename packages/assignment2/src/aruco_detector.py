"""
aruco_detector.py – ArUco marker detection, pose estimation, and annotation.

Handles OpenCV API differences between older (<4.7) and newer (>=4.7) versions:
  - Dictionary / DetectorParameters creation
  - detectMarkers (old API vs ArucoDetector class)
  - drawAxis (deprecated) vs drawFrameAxes (replacement)
  - estimatePoseSingleMarkers (removed in 4.9) vs solvePnP per-marker

All drawing is done in-place on the supplied BGR image.
"""

import numpy as np
import cv2
import rospy

from config import ARUCO_DICT_TYPE, MARKER_SIZE_M

# 3-D corners of a flat marker (marker frame, z=0)
#   top-left, top-right, bottom-right, bottom-left
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
    """Return (dict, params, detector_or_None, use_new_api)."""
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


def _draw_axes(img: np.ndarray, K: np.ndarray, rvec, tvec, length: float) -> None:
    """Draw coordinate axes; bridges OpenCV 4.6- and 4.7+ APIs."""
    D_zeros = np.zeros((4, 1), dtype=np.float64)
    try:
        cv2.drawFrameAxes(img, K, D_zeros, rvec, tvec, length)
    except AttributeError:
        cv2.aruco.drawAxis(img, K, D_zeros, rvec, tvec, length)


def _estimate_pose_single(corner: np.ndarray, K: np.ndarray):
    """
    Estimate the pose of one marker using solvePnP.

    Parameters
    ----------
    corner : (1, 4, 2) array – marker corners from detectMarkers
    K      : (3, 3) camera matrix

    Returns
    -------
    rvec, tvec : (3,) arrays
    """
    img_pts = corner[0].astype(np.float32)  # (4, 2)
    D_zeros = np.zeros((4, 1), dtype=np.float64)
    success, rvec, tvec = cv2.solvePnP(
        _MARKER_OBJ_PTS, img_pts, K, D_zeros,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if not success:
        return None, None
    return rvec.flatten(), tvec.flatten()


class ArucoDetector:
    def __init__(self):
        self._dict, self._params, self._detector, self._new_api = _make_detector()
        api_ver = "new (>=4.7)" if self._new_api else "legacy (<4.7)"
        rospy.loginfo("[ArUco] Initialised with OpenCV %s API.", api_ver)

    def detect_and_annotate(
        self,
        img: np.ndarray,
        new_K: np.ndarray,
    ):
        """
        Detect ArUco markers, draw overlays on `img` (in-place), and return
        the pose of the closest known tag.

        Parameters
        ----------
        img   : undistorted BGR image
        new_K : optimal camera matrix used for undistortion

        Returns
        -------
        (tag_id, rvec, tvec)  – closest known tag, or (None, None, None)
        """
        if self._new_api:
            corners, ids, _ = self._detector.detectMarkers(img)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(
                img, self._dict, parameters=self._params
            )

        if ids is None or len(ids) == 0:
            return None, None, None

        # Draw marker outlines and IDs
        cv2.aruco.drawDetectedMarkers(img, corners, ids)

        best_tag_id = None
        best_dist = float("inf")
        best_rvec = None
        best_tvec = None

        for i, tag_id in enumerate(ids.flatten()):
            rvec, tvec = _estimate_pose_single(corners[i], new_K)
            if rvec is None:
                continue

            _draw_axes(img, new_K, rvec, tvec, MARKER_SIZE_M * 0.5)

            rospy.loginfo_throttle(
                1.0,
                "[ArUco] Tag %d  t=[%.3f, %.3f, %.3f]  dist=%.3f m",
                tag_id, tvec[0], tvec[1], tvec[2], float(np.linalg.norm(tvec)),
            )

            dist = float(np.linalg.norm(tvec))
            if dist < best_dist:
                best_dist = dist
                best_tag_id = tag_id
                best_rvec = rvec
                best_tvec = tvec

        return best_tag_id, best_rvec, best_tvec
