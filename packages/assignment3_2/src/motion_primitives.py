"""
motion_primitives.py – Precise, heavily-logged movement building blocks.

Every public function:
  1. Logs INTENT  ("I am about to turn 90° right")
  2. Executes the motion step by step
  3. Logs RESULT  ("Turned for 1.05 s, estimated 91.2°")

Two execution modes:
  TIME_BASED  – uses calibrated duration (default, always works)
  ODOM_BASED  – subscribes to /odometry, logs both, stops on target delta
                (set USE_ODOMETRY = True once odometry is confirmed working)

Calibration constants live at the top – tune these first.
"""

import math
import time
import rospy
from duckietown_msgs.msg import Twist2DStamped
from nav_msgs.msg import Odometry

# ─── Calibration ─────────────────────────────────────────────────────────────
# Measure these on your robot and update until motion is accurate.

# Forward speed used in move_forward_cm()  [m/s]
# Calibrated: robot travels 44 cm when commanded 40 cm at 0.20 m/s → actual = 0.22 m/s
MOVE_SPEED = 0.22

# Angular speed used in turn_degrees()  [rad/s]
# Calibrated: 1.283 rad/s → 92.2° without ramp (best known value)
# Positive = CCW (left), robot frame
TURN_SPEED = 1.283

# Ramp-down durations [s] — speed linearly drops from full → 0.
# Kept short for forward motion: long ramp causes arc because motors diverge at low speed.
MOVE_RAMP_DOWN_SEC = 0.15
# No ramp for turns: ramp causes unpredictable overshoot due to nonlinear motor behavior at low omega.
TURN_RAMP_DOWN_SEC = 0.0

# Set True only after confirming /odometry topic publishes correctly.
USE_ODOMETRY = False

# Odometry topic – adjust VEHICLE_NAME if needed (empty = no prefix).
VEHICLE_NAME = ""   # e.g. "db21m"

# How often the motion loop ticks [Hz].
LOOP_HZ = 20

# Safety: any single primitive is killed after this many seconds.
MAX_PRIMITIVE_SEC = 15.0
# ─────────────────────────────────────────────────────────────────────────────


def _cmd_topic() -> str:
    prefix = f"/{VEHICLE_NAME}" if VEHICLE_NAME else ""
    return f"{prefix}/car_cmd_switch_node/cmd"


def _odom_topic() -> str:
    prefix = f"/{VEHICLE_NAME}" if VEHICLE_NAME else ""
    return f"{prefix}/odometry_node/odometry"


def _make_cmd(v: float, omega: float) -> Twist2DStamped:
    cmd = Twist2DStamped()
    cmd.v = v
    cmd.omega = omega
    return cmd


def _stop_cmd() -> Twist2DStamped:
    return _make_cmd(0.0, 0.0)


# ─── Odometry helper ─────────────────────────────────────────────────────────

class _OdomTracker:
    """Subscribes to odometry and tracks position/yaw delta since reset()."""

    def __init__(self):
        self._x = None
        self._y = None
        self._yaw = None
        self._x0 = None
        self._y0 = None
        self._yaw0 = None
        self._sub = None

    def start(self):
        self._sub = rospy.Subscriber(
            _odom_topic(), Odometry, self._cb, queue_size=1
        )
        rospy.loginfo("[Motion/Odom] Subscribed to %s", _odom_topic())

    def stop(self):
        if self._sub:
            self._sub.unregister()
            self._sub = None

    def _cb(self, msg: Odometry):
        self._x = msg.pose.pose.position.x
        self._y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        # yaw from quaternion
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self._yaw = math.atan2(siny, cosy)

    def reset(self):
        self._x0 = self._x
        self._y0 = self._y
        self._yaw0 = self._yaw
        rospy.loginfo(
            "[Motion/Odom] Reference set  x0=%.4f  y0=%.4f  yaw0=%.2f°",
            self._x0 if self._x0 else float("nan"),
            self._y0 if self._y0 else float("nan"),
            math.degrees(self._yaw0) if self._yaw0 else float("nan"),
        )

    @property
    def is_ready(self) -> bool:
        return self._x0 is not None

    @property
    def distance_m(self) -> float:
        if not self.is_ready or self._x is None:
            return 0.0
        return math.hypot(self._x - self._x0, self._y - self._y0)

    @property
    def yaw_delta_rad(self) -> float:
        if not self.is_ready or self._yaw is None:
            return 0.0
        raw = self._yaw - self._yaw0
        # Normalize to [-pi, pi]
        while raw > math.pi:
            raw -= 2 * math.pi
        while raw < -math.pi:
            raw += 2 * math.pi
        return raw

    def status_str(self) -> str:
        return (
            f"dist={self.distance_m*100:.1f}cm  "
            f"yaw_delta={math.degrees(self.yaw_delta_rad):.1f}°"
        )


# Module-level tracker (shared across calls)
_odom = _OdomTracker()


# ─── Public API ───────────────────────────────────────────────────────────────

def init(vehicle_name: str = ""):
    """
    Call once at node startup.
    Sets the vehicle name and starts odometry tracking if USE_ODOMETRY is True.
    """
    global VEHICLE_NAME
    VEHICLE_NAME = vehicle_name
    rospy.loginfo("=" * 60)
    rospy.loginfo("[Motion] Initialising motion primitives")
    rospy.loginfo("[Motion]   vehicle      : '%s'", VEHICLE_NAME or "(no prefix)")
    rospy.loginfo("[Motion]   cmd topic    : %s", _cmd_topic())
    rospy.loginfo("[Motion]   odom topic   : %s", _odom_topic())
    rospy.loginfo("[Motion]   MOVE_SPEED   : %.3f m/s", MOVE_SPEED)
    rospy.loginfo("[Motion]   TURN_SPEED   : %.3f rad/s", TURN_SPEED)
    rospy.loginfo("[Motion]   USE_ODOMETRY : %s", USE_ODOMETRY)
    rospy.loginfo("=" * 60)

    if USE_ODOMETRY:
        _odom.start()
        rospy.loginfo("[Motion] Waiting up to 3 s for first odometry message …")
        deadline = rospy.Time.now() + rospy.Duration(3.0)
        rate = rospy.Rate(10)
        while rospy.Time.now() < deadline and not rospy.is_shutdown():
            if _odom._x is not None:
                rospy.loginfo("[Motion] Odometry ready.")
                break
            rate.sleep()
        else:
            if _odom._x is None:
                rospy.logwarn(
                    "[Motion] No odometry received in 3 s! "
                    "Check topic '%s'. Falling back to time-based control.",
                    _odom_topic(),
                )


def move_forward_cm(
    distance_cm: float,
    publisher: rospy.Publisher,
) -> None:
    """
    Drive the robot straight forward by `distance_cm` centimetres.

    Steps
    -----
    1. Log intent.
    2. Compute expected duration from MOVE_SPEED.
    3. Publish forward velocity in a loop until duration elapses
       (or odometry target reached if USE_ODOMETRY).
    4. Publish stop.
    5. Log actual result.
    """
    distance_m = distance_cm / 100.0
    # Total duration accounts for ramp-down: distance = v*(t_full) + v*ramp/2
    # → t_total = distance/v + ramp/2
    ramp = min(MOVE_RAMP_DOWN_SEC, distance_m / MOVE_SPEED * 0.5)  # cap at 50% of motion
    expected_sec = distance_m / MOVE_SPEED + ramp / 2.0

    rospy.loginfo("-" * 50)
    rospy.loginfo("[Motion] INTENT: move forward %.1f cm (%.3f m)", distance_cm, distance_m)
    rospy.loginfo(
        "[Motion]   speed=%.3f m/s  expected_duration=%.3f s  ramp=%.2f s",
        MOVE_SPEED, expected_sec, ramp,
    )

    if USE_ODOMETRY and _odom._x is not None:
        _odom.reset()
        rospy.loginfo("[Motion]   mode=ODOMETRY_BASED  target=%.3f m", distance_m)
    else:
        rospy.loginfo("[Motion]   mode=TIME_BASED  duration=%.3f s", expected_sec)

    rate = rospy.Rate(LOOP_HZ)
    t_start = rospy.Time.now()
    ramp_start = expected_sec - ramp

    step = 0
    while not rospy.is_shutdown():
        elapsed = (rospy.Time.now() - t_start).to_sec()

        if elapsed > MAX_PRIMITIVE_SEC:
            rospy.logerr(
                "[Motion] SAFETY TIMEOUT after %.1f s — stopping immediately!", elapsed
            )
            break

        if USE_ODOMETRY and _odom.is_ready:
            travelled_m = _odom.distance_m
            remaining_m = distance_m - travelled_m
            rospy.loginfo_throttle(
                0.5,
                "[Motion] MOVING  elapsed=%.2f s  odom: %s  remaining=%.3f m",
                elapsed, _odom.status_str(), remaining_m,
            )
            if travelled_m >= distance_m:
                rospy.loginfo(
                    "[Motion] Odometry target reached: %.3f m >= %.3f m",
                    travelled_m, distance_m,
                )
                break
        else:
            if elapsed >= expected_sec:
                break
            remaining = expected_sec - elapsed
            if step % LOOP_HZ == 0:
                rospy.loginfo(
                    "[Motion] MOVING  elapsed=%.2f s  remaining=%.2f s",
                    elapsed, remaining,
                )

        # Ramp-down: linearly reduce speed in the final `ramp` seconds
        if elapsed >= ramp_start and ramp > 0:
            scale = max(0.0, (expected_sec - elapsed) / ramp)
            v = MOVE_SPEED * scale
        else:
            v = MOVE_SPEED

        publisher.publish(_make_cmd(v, 0.0))
        step += 1
        rate.sleep()

    publisher.publish(_stop_cmd())
    total_elapsed = (rospy.Time.now() - t_start).to_sec()

    time_based_estimate_cm = (total_elapsed - ramp / 2.0) * MOVE_SPEED * 100.0
    rospy.loginfo("[Motion] RESULT: move_forward_cm")
    rospy.loginfo("  target   : %.1f cm", distance_cm)
    rospy.loginfo("  duration : %.3f s  (expected %.3f s)", total_elapsed, expected_sec)
    rospy.loginfo("  time-est : %.1f cm", time_based_estimate_cm)
    if USE_ODOMETRY and _odom.is_ready:
        rospy.loginfo("  odom-est : %.1f cm", _odom.distance_m * 100.0)
    rospy.loginfo("-" * 50)


def turn_degrees(
    angle_deg: float,
    publisher: rospy.Publisher,
) -> None:
    """
    Rotate the robot by `angle_deg` degrees.

    Sign convention (ROS standard):
        positive angle → CCW / turn LEFT
        negative angle → CW  / turn RIGHT

    Example:
        turn_degrees(-90, pub)   →  turn 90° to the RIGHT
        turn_degrees( 90, pub)   →  turn 90° to the LEFT
    """
    angle_rad = math.radians(angle_deg)
    direction_str = "LEFT (CCW)" if angle_deg >= 0 else "RIGHT (CW)"
    sign = 1.0 if angle_deg >= 0 else -1.0
    omega_full = sign * TURN_SPEED
    # Total duration accounts for ramp-down: angle = w*(t_full) + w*ramp/2
    ramp = min(TURN_RAMP_DOWN_SEC, abs(angle_rad) / TURN_SPEED * 0.5)
    expected_sec = abs(angle_rad) / TURN_SPEED + ramp / 2.0

    rospy.loginfo("-" * 50)
    rospy.loginfo(
        "[Motion] INTENT: turn %.1f°  direction=%s",
        abs(angle_deg), direction_str,
    )
    rospy.loginfo(
        "[Motion]   omega=%.3f rad/s  expected_duration=%.3f s  ramp=%.2f s",
        omega_full, expected_sec, ramp,
    )

    if USE_ODOMETRY and _odom._yaw is not None:
        _odom.reset()
        rospy.loginfo(
            "[Motion]   mode=ODOMETRY_BASED  target=%.3f rad (%.1f°)",
            abs(angle_rad), abs(angle_deg),
        )
    else:
        rospy.loginfo("[Motion]   mode=TIME_BASED  duration=%.3f s", expected_sec)

    rate = rospy.Rate(LOOP_HZ)
    t_start = rospy.Time.now()
    ramp_start = expected_sec - ramp
    step = 0

    while not rospy.is_shutdown():
        elapsed = (rospy.Time.now() - t_start).to_sec()

        if elapsed > MAX_PRIMITIVE_SEC:
            rospy.logerr(
                "[Motion] SAFETY TIMEOUT after %.1f s — stopping immediately!", elapsed
            )
            break

        if USE_ODOMETRY and _odom.is_ready:
            turned_rad = abs(_odom.yaw_delta_rad)
            remaining_rad = abs(angle_rad) - turned_rad
            rospy.loginfo_throttle(
                0.5,
                "[Motion] TURNING  elapsed=%.2f s  odom: %s  remaining=%.1f°",
                elapsed, _odom.status_str(), math.degrees(remaining_rad),
            )
            if turned_rad >= abs(angle_rad):
                rospy.loginfo(
                    "[Motion] Odometry target reached: %.1f° >= %.1f°",
                    math.degrees(turned_rad), abs(angle_deg),
                )
                break
        else:
            if elapsed >= expected_sec:
                break
            remaining = expected_sec - elapsed
            if step % LOOP_HZ == 0:
                rospy.loginfo(
                    "[Motion] TURNING  elapsed=%.2f s  remaining=%.2f s  "
                    "time-est=%.1f°",
                    elapsed, remaining,
                    elapsed * TURN_SPEED * (180.0 / math.pi),
                )

        # Ramp-down: linearly reduce omega in the final `ramp` seconds
        if elapsed >= ramp_start and ramp > 0:
            scale = max(0.0, (expected_sec - elapsed) / ramp)
            omega = omega_full * scale
        else:
            omega = omega_full

        publisher.publish(_make_cmd(0.0, omega))
        step += 1
        rate.sleep()

    publisher.publish(_stop_cmd())
    total_elapsed = (rospy.Time.now() - t_start).to_sec()

    time_based_estimate_deg = (total_elapsed - ramp / 2.0) * TURN_SPEED * (180.0 / math.pi)
    rospy.loginfo("[Motion] RESULT: turn_degrees")
    rospy.loginfo("  target   : %.1f° %s", abs(angle_deg), direction_str)
    rospy.loginfo("  duration : %.3f s  (expected %.3f s)", total_elapsed, expected_sec)
    rospy.loginfo("  time-est : %.1f°", time_based_estimate_deg)
    if USE_ODOMETRY and _odom.is_ready:
        rospy.loginfo("  odom-est : %.1f°", math.degrees(abs(_odom.yaw_delta_rad)))
    rospy.loginfo("-" * 50)


def pause(seconds: float, publisher: rospy.Publisher) -> None:
    """Stop and wait for `seconds`. Useful between primitives."""
    rospy.loginfo("[Motion] PAUSE %.2f s …", seconds)
    publisher.publish(_stop_cmd())
    rospy.sleep(seconds)
    rospy.loginfo("[Motion] PAUSE done.")
