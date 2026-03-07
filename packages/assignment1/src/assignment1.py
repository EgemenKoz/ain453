#!/usr/bin/env python3

import os

import rospy
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import Range
from std_msgs.msg import String


# Assignment constants (edit these directly if needed)
THRESHOLD_M = 0.1
STOP_TOPIC = "chatter"
STOP_MESSAGE = "Stop the robot at 10cm away from the block."

# If VEHICLE_NAME exists, we use /<vehicle>/front_center_tof_driver_node/range.
# Otherwise this fallback topic is used.
TOF_TOPIC_FALLBACK = "/front_center_tof_driver_node/range"


class DistanceStopperNode(DTROS):
    def __init__(self, node_name):
        super(DistanceStopperNode, self).__init__(
            node_name=node_name,
            node_type=NodeType.GENERIC,
        )

        vehicle_name = os.environ.get("VEHICLE_NAME", "")
        if vehicle_name:
            self._tof_topic = f"/{vehicle_name}/front_center_tof_driver_node/range"
        else:
            self._tof_topic = TOF_TOPIC_FALLBACK

        self._publisher = rospy.Publisher(STOP_TOPIC, String, queue_size=10)
        self._subscriber = rospy.Subscriber(self._tof_topic, Range, self._tof_callback)
        self._stop_msg = String(data=STOP_MESSAGE)

    def _tof_callback(self, msg):
        distance = msg.range
        rospy.loginfo("ToF distance: %.3f m", distance)

        if distance <= THRESHOLD_M:
            rospy.loginfo(
                "Distance %.3f m <= %.3f m. Publishing stop command.",
                distance,
                THRESHOLD_M,
            )
            self._publisher.publish(self._stop_msg)


if __name__ == '__main__':
    node = DistanceStopperNode(node_name='distance_stopper_node')
    rospy.spin()
