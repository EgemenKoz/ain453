# Assignment 3 – Code Analysis

## Package Structure

| File | Role |
|---|---|
| `assignment3.py` | Entry point — spins up `PathNavigationNode` |
| `astar.py` | A* algorithm (OPEN heap, CLOSED set, parent tracking) |
| `config.py` | All shared constants: graph, ArUco params, nav tuning |
| `calibration.py` | Camera intrinsics loader (YAML file → CameraInfo topic fallback) |
| `aruco_detector.py` | ArUco detection + per-marker solvePnP pose estimation |
| `navigation_controller.py` | Proportional visual-servo controller (produces `Twist2DStamped`) |
| `navigation_node.py` | Main DTROS node — A* planning + 4-state navigation FSM |

---

## What the Code Does

### 1. A* Planning (`astar.py` + `config.py`)
On start-up the node runs A* from `N0` to `N15` using the graph defined in `config.py`.

- **OPEN list**: `heapq` min-heap of `_Entry(f, h, node)` — tuple comparison gives the required tie-break rule (equal `f` → lower `h` wins).
- **CLOSED set**: skips already-expanded nodes.
- **Heuristic**: Euclidean distance by default (configurable to Manhattan via `HEURISTIC` in `config.py`).
- **Output** (printed to `stdout`):
  ```
  Heuristic : euclidean
  Path      : N0 → N1 → …
  Total cost: X.XXXX
  ```

### 2. Navigation FSM (`navigation_node.py`)

```
SEARCHING ──(tag visible)──► APPROACHING ──(dist < threshold)──► [next node / GOAL_REACHED]
    ▲                              │
    │    (lost too long, far)      │ (lost too long, was close)
    └──────────────────────────────┤
                               BLIND_FORWARD ──(timer done)──► [node reached]
```

- **SEARCHING**: rotates in place; direction decided by heading tracking (which way does the robot need to face to reach the next node?).
- **APPROACHING**: proportional controller (`tvec[0]` → `omega`, `tvec[2]` → `v`). Align-first: if `|heading_error| > 0.6 rad`, holds `v=0` until roughly pointed at the tag.
- **BLIND_FORWARD**: when the tag disappears while the robot is already very close (`tvec[2] < 0.30 m`), drives straight at low speed for 0.8 s and then declares the node reached. Handles the case where floor tags slide under the camera.
- **GOAL_REACHED**: publishes zero velocity; prints `"Goal Reached"` to terminal.

### 3. Heading Tracking
After each node is reached, the robot's heading is updated to `atan2(Δy, Δx)` between the previous and current node. This is used purely to decide the initial rotation direction when SEARCHING for the next tag. The actual alignment is done visually via the ARTag pose.

### 4. Localization
- Continuous: every camera frame the robot detects all visible ArUco markers (`ArucoDetector.detect`).
- A node is considered **reached** only when the **target tag** is detected AND `tvec[2] < PROXIMITY_THRESHOLD_M` (0.15 m).
- If the target tag is lost for `TAG_LOST_PATIENCE_FRAMES` (15 frames) the robot falls back to SEARCHING.

---

## Map & Graph Verification

### Node Coordinates (matches A3.md exactly)
```
y=3:  N12(0,3) N13(1,3) N14(2,3) N15(3,3) ← GOAL
y=2:  N8 (0,2) N9 (1,2) N10(2,2) N11(3,2)
y=1:  N4 (0,1) N5 (1,1) N6 (2,1) N7 (3,1)
y=0:  N0 (0,0) N1 (1,0) N2 (2,0) N3 (3,0) ← START
      x=0      x=1      x=2      x=3
```
`+x = right`, `+y = up`. All 19 edges and their costs are verified against A3.md — **no discrepancies**.

Walls correctly absent: `N3↔N7`, `N4↔N5`, `N9↔N10`, `N9↔N13`, `N11↔N15`.

---

## Bug Report

### BUG 1 — Heading Convention Mismatch (config.py line 114)

**The problem**: The user's stated convention is *theta = 0 means looking up (+y)*. The code uses the standard math `atan2` convention where 0 = facing right (+x) and π/2 = facing up (+y).

```python
# config.py line 114
INITIAL_HEADING_RAD = math.pi / 2   # ← encodes "up" in atan2 convention
```

The code is internally **consistent** with its own convention (all `_direction_between` calls use `atan2(dy, dx)`, and the initial value of π/2 correctly represents +y). Functionally the robot behaviour is correct.

However, if any external grader or display system expects the heading to be 0 when facing up, the stored value would be wrong.

**Fix (if theta=0=up convention is required)**: Define a wrapper that maps "navigation bearing" (0=up, increases CW) to the internal atan2 heading, or simply change all angle representations to use `math.pi/2 - atan2(dy, dx)` and set `INITIAL_HEADING_RAD = 0`.

### BUG 2 — Planned Turn Commands Not Printed to stdout (navigation_node.py lines 165–180, 377–385)

**The problem**: The per-segment "turn LEFT/RIGHT/STRAIGHT" commands are computed correctly but are sent to `rospy.loginfo` (ROS log), not `print()` (terminal stdout).

```python
# navigation_node.py line 176 — goes to ROS log only
rospy.loginfo(
    "[Nav] Plan: N%d→N%d  heading %.0f°→%.0f°  turn %.0f° %s",
    prev, curr, ..., direction_str,
)
```

The assignment requires the robot to output discrete navigation commands ("go straight", "turn left", etc.) that are readable in the terminal alongside the path and cost.

**Fix**: Add `print()` calls for the planned turn sequence at start-up and for each runtime turn decision:

```python
# In _log_planned_turns, after the rospy.loginfo:
print(f"  N{prev}→N{curr}: turn {math.degrees(abs(turn)):.0f}° {direction_str} then go straight")

# In _on_node_reached, after computing direction_str:
print(f"[Nav] N{next_node}: turn {math.degrees(abs(turn)):.0f}° {direction_str} then go straight")
```

### BUG 3 — Left-over Turkish Comment (navigation_node.py line 339)

**Minor**: An inline comment in Turkish was left in the code, inconsistent with the English-only codebase and potentially confusing.

```python
# YENİ EKLENEN KISIM: 
# Hedef anlık kaybolduğunda agresif açıyla dönmeye devam etme.
```

**Fix**: Replace with an English comment or remove it.

### BUG 4 — `dist` Uses `abs(tvec[2])` Instead of `tvec[2]` (navigation_node.py line 282)

```python
dist = float(abs(tvec[2]))
```

`tvec[2]` is the forward/depth component. It should always be positive when the tag is in front of the camera. The `abs()` silently hides the case where the tag is somehow behind the camera (negative `tvec[2]`), which would be detected as "very close" even though the robot is facing the wrong way.

**Fix**:
```python
dist = float(tvec[2])
if dist < 0:
    rospy.logwarn_throttle(1.0, "[Nav] Negative tvec[2] for tag %d — tag behind camera?", target_id)
    dist = 999.0  # treat as not reached
```

### BUG 5 — Watchdog Does Not Guard BLIND_FORWARD Timeout (navigation_node.py lines 388–398)

```python
def _watchdog_cb(self, _event) -> None:
    if self._state in (_GOAL_REACHED, _BLIND_FORWARD):
        return   # ← watchdog is disabled during blind forward
```

If the camera feed dies **while in BLIND_FORWARD**, the watchdog does nothing, but the blind forward timeout is checked only inside `_image_cb` (line 239). With no images arriving, `_image_cb` is never called, so `_blind_start` is never checked, and the robot drives forward indefinitely at `BLIND_FORWARD_SPEED`.

**Fix**: Check the BLIND_FORWARD timer inside the watchdog as well:

```python
def _watchdog_cb(self, _event) -> None:
    if self._state == _GOAL_REACHED:
        return
    if self._state == _BLIND_FORWARD:
        elapsed = (rospy.Time.now() - self._blind_start).to_sec()
        if elapsed >= BLIND_FORWARD_SEC:
            self._on_node_reached(self._target_node)
        return
    # ... existing image-timeout check
```

---

## Summary Table

| # | Severity | File | Issue |
|---|---|---|---|
| 1 | Medium | `config.py:114` | Heading convention: code stores π/2 for "up"; user expects 0 for "up" |
| 2 | Medium | `navigation_node.py:165-180, 377-385` | Turn commands (LEFT/RIGHT/STRAIGHT) go to rospy.loginfo, not printed to stdout |
| 3 | Low | `navigation_node.py:339` | Turkish comment left in English codebase |
| 4 | Low | `navigation_node.py:282` | `abs(tvec[2])` hides negative depth silently |
| 5 | Medium | `navigation_node.py:388` | Watchdog skips BLIND_FORWARD — robot could drive forever if camera dies |

---

## A* Path (Expected)

Running A* with Euclidean heuristic on this graph, the lowest-cost path from N0 to N15 is:

**N0 → N1 → N2 → N6 → N7 → N11 → N10 → N14 → N15**  
Total cost: **1.5 + 1.0 + 1.5 + 0.5 + 1.5 + 1.0 + 1.5 + 1.0 = 9.5**

Initial heading = π/2 (facing +y = up). Planned turns:
- N0 → N1: heading π/2 → 0 (right) → **turn RIGHT 90°**
- N1 → N2: heading stays 0 → **go STRAIGHT**
- N2 → N6: heading 0 → π/2 (up) → **turn LEFT 90°**
- N6 → N7: heading π/2 → 0 (right) → **turn RIGHT 90°**
- N7 → N11: heading 0 → π/2 (up) → **turn LEFT 90°**
- N11 → N10: heading π/2 → π (left) → **turn LEFT 90°**
- N10 → N14: heading π → π/2 (up) → **turn RIGHT 90°**
- N14 → N15: heading π/2 → 0 (right) → **turn RIGHT 90°**
