#!/usr/bin/env python3

import os
import math

import rospy
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import WheelsCmdStamped
from sensor_msgs.msg import Range

# Stop exactly 10 cm before an obstacle
THRESHOLD_M = 0.10

# Wheel speed for straight-line driving (range: -1.0 to 1.0)
FORWARD_SPEED = 0.3

# How often wheel commands are published (Hz)
PUBLISH_RATE_HZ = 10


class NavigationNode(DTROS):
    """Drive in a straight line and stop 10 cm before an obstacle detected by
    the front-center Time-of-Flight sensor."""

    def __init__(self, node_name):
        super(NavigationNode, self).__init__(
            node_name=node_name,
            node_type=NodeType.GENERIC,
        )

        vehicle_name = os.environ.get("VEHICLE_NAME", "")
        if vehicle_name:
            tof_topic = f"/{vehicle_name}/front_center_tof_driver_node/range"
            wheels_topic = f"/{vehicle_name}/wheels_driver_node/wheels_cmd"
        else:
            tof_topic = "/front_center_tof_driver_node/range"
            wheels_topic = "/wheels_driver_node/wheels_cmd"

        rospy.loginfo("Subscribing to ToF topic:  %s", tof_topic)
        rospy.loginfo("Publishing to wheels topic: %s", wheels_topic)

        self._stop_flag = False

        # Publisher: wheel velocity commands
        self._wheels_pub = rospy.Publisher(
            wheels_topic, WheelsCmdStamped, queue_size=1
        )

        # Subscriber: front-center ToF distance readings
        self._tof_sub = rospy.Subscriber(tof_topic, Range, self._tof_callback)

        # Timer: publish wheel commands at a fixed rate
        self._control_timer = rospy.Timer(
            rospy.Duration(1.0 / PUBLISH_RATE_HZ),
            self._control_loop,
        )

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _tof_callback(self, msg: Range):
        """Receive a Range message and trigger a stop when close enough."""
        distance = msg.range

        # sensor_msgs/Range uses NaN or values outside [min_range, max_range]
        # to indicate an invalid / out-of-range reading — skip those.
        if math.isnan(distance) or not (msg.min_range <= distance <= msg.max_range):
            rospy.logwarn_throttle(
                5.0, "ToF reading out of valid range (%.3f m) — skipping.", distance
            )
            return

        rospy.loginfo_throttle(1.0, "ToF distance: %.3f m", distance)

        if not self._stop_flag and distance <= THRESHOLD_M:
            rospy.loginfo(
                "Obstacle detected at %.3f m (threshold %.2f m). Stopping robot.",
                distance,
                THRESHOLD_M,
            )
            self._stop_flag = True

    def _control_loop(self, _event):
        """Publish wheel velocity commands at a fixed rate."""
        cmd = WheelsCmdStamped()
        cmd.header.stamp = rospy.Time.now()

        if self._stop_flag:
            cmd.vel_left = 0.0
            cmd.vel_right = 0.0
        else:
            # Equal speeds on both wheels → straight-line motion
            cmd.vel_left = FORWARD_SPEED
            cmd.vel_right = FORWARD_SPEED

        self._wheels_pub.publish(cmd)


if __name__ == "__main__":
    node = NavigationNode(node_name="navigation_node")
    rospy.spin()
