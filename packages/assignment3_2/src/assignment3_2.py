#!/usr/bin/env python3
"""
assignment3_2.py – Debug node for precise Duckiebot motion.

Run with:
    rosrun assignment3_2 assignment3_2.py

What this node does
-------------------
1. Starts up and logs all configuration.
2. Runs the DEMO_SEQUENCE defined at the bottom of this file.
3. Each step is logged before and after execution.

Edit DEMO_SEQUENCE to match the moves you actually need.
Edit VEHICLE_NAME to match your robot.
Edit motion_primitives.py constants to calibrate speed/turn_rate.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import rospy
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import Twist2DStamped

import motion_primitives as mp


# ─── Robot identity ───────────────────────────────────────────────────────────
# Set this to your robot's name, e.g. "db21m".
# Leave empty "" if topics have no vehicle prefix.
VEHICLE_NAME = os.environ.get("VEHICLE_NAME", "")


# ─── Demo sequence ────────────────────────────────────────────────────────────
# Each entry is a (function, args) tuple.
# The node will execute them one by one with a short pause between each.
#
# Available functions:
#   mp.move_forward_cm(distance_cm, publisher)
#   mp.turn_degrees(angle_deg, publisher)   – positive=LEFT, negative=RIGHT
#   mp.pause(seconds, publisher)
#
# Add, remove, or reorder steps freely.

def build_demo_sequence(pub):
    """
    Returns a list of (label, callable) pairs.
    Each callable takes no arguments (publisher is already bound via lambda).
    """
    return [
        # Step 1 – verify the robot can go straight 40 cm
        ("Move forward 40 cm",
         lambda: mp.move_forward_cm(40.0, pub)),

        # Step 2 – short pause so we can see what happened
        ("Pause 3 s",
         lambda: mp.pause(3.0, pub)),

        # Step 3 – turn 90° to the RIGHT (negative = CW)
        ("Turn 90° RIGHT",
         lambda: mp.turn_degrees(-90.0, pub)),

        # # Step 4 – pause again
        ("Pause 3 s",
         lambda: mp.pause(3.0, pub)),

        # # Step 5 – move another 40 cm straight ahead (new direction)
        ("Move forward 40 cm",
         lambda: mp.move_forward_cm(40.0, pub)),

        # # Step 6 – turn 90° to the LEFT (+90)
        ("Turn 90° LEFT",
         lambda: mp.turn_degrees(90.0, pub)),

        # ── Add your own steps below ──────────────────────────────────────────
        ("Turn 180° (U-turn)",
         lambda: mp.turn_degrees(180.0, pub)),
        #
        ("Move forward 20 cm",
         lambda: mp.move_forward_cm(20.0, pub)),
    ]


# ─── Node ─────────────────────────────────────────────────────────────────────

class MotionDebugNode(DTROS):

    def __init__(self, node_name: str):
        super().__init__(node_name=node_name, node_type=NodeType.GENERIC)

        # ── Publisher ─────────────────────────────────────────────────────────
        prefix = f"/{VEHICLE_NAME}" if VEHICLE_NAME else ""
        cmd_topic = f"{prefix}/car_cmd_switch_node/cmd"
        self._pub = rospy.Publisher(cmd_topic, Twist2DStamped, queue_size=1)

        rospy.loginfo("[Node] Publishing commands to: %s", cmd_topic)

        # ── Initialise motion library ──────────────────────────────────────────
        mp.init(vehicle_name=VEHICLE_NAME)

        # Give ROS time to set up connections before moving.
        rospy.loginfo("[Node] Waiting 2 s for connections to establish …")
        rospy.sleep(2.0)

        # ── Run the demo sequence ──────────────────────────────────────────────
        self._run_sequence()

    def _run_sequence(self):
        sequence = build_demo_sequence(self._pub)
        total = len(sequence)

        rospy.loginfo("=" * 60)
        rospy.loginfo("[Node] Starting demo sequence  (%d steps)", total)
        rospy.loginfo("=" * 60)

        for idx, (label, action) in enumerate(sequence, start=1):
            if rospy.is_shutdown():
                rospy.logwarn("[Node] Shutdown requested – aborting sequence.")
                break

            rospy.loginfo("")
            rospy.loginfo(">>> STEP %d / %d : %s", idx, total, label)
            action()
            rospy.loginfo("<<< STEP %d done", idx)

        rospy.loginfo("")
        rospy.loginfo("=" * 60)
        rospy.loginfo("[Node] Demo sequence complete.")
        rospy.loginfo("=" * 60)

        # Ensure robot is stopped at the end.
        stop = Twist2DStamped()
        stop.v = 0.0
        stop.omega = 0.0
        self._pub.publish(stop)
        rospy.loginfo("[Node] Final stop published.")


if __name__ == "__main__":
    node = MotionDebugNode(node_name="motion_debug_node")
    rospy.spin()
