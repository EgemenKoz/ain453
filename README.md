## How to Run

1. Make sure the Duckiebot is reachable on the network.

```bash
dts fleet discover
ping ROBOT_NAME.local
dts devel build -f -H ROBOT_NAME
dts devel run -H ROBOT_NAME -L my-launcher
```

``` bash
export ROS_MASTER_URI=http://mouse.local:11311
export ROS_IP=$(hostname -I | awk '{print $1}')
rqt_image_view
```