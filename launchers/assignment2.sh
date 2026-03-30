#!/bin/bash

source /environment.sh

# initialize launch file
dt-launchfile-init

# launch assignment2 node
dt-exec rosrun assignment2 assignment2.py

# wait for app to end
dt-launchfile-join
