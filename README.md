## How to Run

1. Make sure the Duckiebot is reachable on the network.

```bash
dts fleet discover
ping ROBOT_NAME.local
dts devel build -f -H ROBOT_NAME
dts devel run -H ROBOT_NAME -L my-launcher
```