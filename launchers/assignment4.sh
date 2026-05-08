#!/bin/bash

source /environment.sh

# initialize launch file
dt-launchfile-init

# launch assignment4 controller + visualizer
dt-exec roslaunch assignment4 assignment4.launch

# wait for app to end
dt-launchfile-join
