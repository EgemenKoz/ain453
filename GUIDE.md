# Assignment 2 – ArUco Localisation: Run & Test Guide

## Project structure

```
packages/assignment2/
├── package.xml
├── CMakeLists.txt
└── src/
    ├── assignment2.py       ← entry point (rosrun target)
    ├── config.py            ← all constants (edit TAG_POSES here)
    ├── calibration.py       ← camera intrinsic loader
    ├── odometry.py          ← wheel-odometry dead reckoning
    ├── aruco_detector.py    ← ArUco detection + pose estimation
    ├── pose_estimator.py    ← camera→world pose transform
    ├── visualizer.py        ← combined camera + map publisher
    ├── localization_node.py ← main DTROS node
    └── generate_markers.py  ← utility: print ArUco tags
```

---

## 1. Before you begin

### 1.1 Set correct TAG_POSES

Open [packages/assignment2/src/config.py](packages/assignment2/src/config.py) and update `TAG_POSES` to match your physical lab map:

```python
TAG_POSES = {
    0: (x_metres, y_metres, yaw_radians),
    ...
}
```

- `x`, `y`: position of the tag centre on the map (metres)
- `yaw`: direction the tag face points in the world frame (radians)
  - `0.0` = tag faces +X (toward x-increasing direction)
  - `math.pi` = tag faces -X (toward x-decreasing direction)

### 1.2 Set correct marker size

```python
MARKER_SIZE_M = 0.065  # physical side length of printed tag (metres)
```

Measure your printed tags and update this value.

### 1.3 Print ArUco markers

```bash
# Run locally (needs opencv-contrib-python)
python3 packages/assignment2/src/generate_markers.py
# Output: markers/aruco_4x4_id{0..5}.png
# Print at correct physical size (default 6.5 cm side)
```

---

## 2. Build & deploy

### 2.1 Build the Docker image

```bash
# From the repository root (where Dockerfile lives)
dts devel build -f --arch arm64v8   # on your laptop (cross-compile for DB21)
# or
dts devel build -f                  # directly on the Duckiebot (native build)
```

### 2.2 Run on the Duckiebot

```bash
dts devel run -H <ROBOT_HOSTNAME> --launcher assignment2
```

Replace `<ROBOT_HOSTNAME>` with your Duckiebot's hostname (e.g. `csc229db`).

The launcher executes:
```bash
dt-exec rosrun assignment2 assignment2.py
```

---

## 3. Viewing the live visualisation

### Option A – rqt_image_view (recommended)

```bash
# On your laptop, with ROS_MASTER_URI set to the Duckiebot
export ROS_MASTER_URI=http://<ROBOT_HOSTNAME>.local:11311
export ROS_IP=<YOUR_LAPTOP_IP>

rosrun rqt_image_view rqt_image_view
```

Subscribe to:
```
/<ROBOT_NAME>/assignment2/visualization/compressed
```

The image shows:
- **Top half**: camera feed with detected ArUco markers outlined and axes drawn
- **Bottom half**: top-down map with robot position
  - **Green dot** = pose from ArUco measurement (high accuracy)
  - **Orange dot** = pose from wheel-odometry fallback (accumulates drift)

### Option B – rostopic + ImageView

```bash
rosrun image_view image_view \
  image:=/<ROBOT_NAME>/assignment2/visualization/compressed \
  _image_transport:=compressed
```

---

## 4. Monitoring topics

```bash
# Confirm topics are active
rostopic list | grep assignment2
rostopic hz /<ROBOT_NAME>/assignment2/visualization/compressed

# Raw encoder ticks
rostopic echo /<ROBOT_NAME>/left_wheel_encoder_node/tick
rostopic echo /<ROBOT_NAME>/right_wheel_encoder_node/tick

# Camera feed
rostopic hz /<ROBOT_NAME>/camera_node/image/compressed
```

---

## 5. Camera calibration

The node looks for calibration in this order:

1. `/data/config/calibrations/camera_intrinsic/<VEHICLE_NAME>.yaml`
2. `/data/config/calibrations/camera_intrinsic/default.yaml`
3. The `/camera_info` ROS topic (auto-published by the camera node)

If no file is found, detection is paused until a CameraInfo message arrives.

To verify calibration was loaded:
```bash
rostopic echo /<ROBOT_NAME>/rosout | grep Calibration
```

---

## 6. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| "Waiting for camera…" forever | Camera node not running | `dts start_gui_tools` → check `rostopic list` |
| No ArUco detections | Wrong DICT / tag not printed correctly | Confirm `ARUCO_DICT_TYPE = DICT_4X4_50`; re-print markers |
| Pose jumps wildly when tag visible | Wrong `TAG_POSES` or `MARKER_SIZE_M` | Measure physical tags; update `config.py` |
| Orange dot drifts fast | Normal for odometry | Bring tag back into view to reset |
| `ImportError: cv2.aruco` | OpenCV without contrib | `pip3 install opencv-contrib-python-headless` |
| `ModuleNotFoundError: calibration` | sys.path not set | Should be auto-fixed by entry-point; check `assignment2.py` top lines |
| Node exits immediately | DTROS exception at startup | Run `rosrun assignment2 assignment2.py` and read stderr for the traceback |

---

## 7. Quick test without a Duckiebot (local)

You can smoke-test the detector and pose estimator on a webcam:

```bash
python3 - <<'EOF'
import cv2, sys, os
sys.path.insert(0, "packages/assignment2/src")
from aruco_detector import ArucoDetector
import numpy as np

# Approximate webcam intrinsics (replace with real calibration)
K = np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1]], dtype=float)

det = ArucoDetector()
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    tid, rvec, tvec = det.detect_and_annotate(frame, K)
    if tid is not None:
        print(f"Tag {tid}: t={tvec}")
    cv2.imshow("ArUco", frame)
    if cv2.waitKey(1) == 27:
        break
cap.release()
cv2.destroyAllWindows()
EOF
```

Hold a printed ArUco tag in front of the webcam; you should see detected corners and axes drawn live.
