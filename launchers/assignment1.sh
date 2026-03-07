#!/bin/bash

source /environment.sh

# initialize launch file
dt-launchfile-init

# launch assignment node
dt-exec rosrun assignment1 assignment1.py

# wait for app to end
dt-launchfile-join
