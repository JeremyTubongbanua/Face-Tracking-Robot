# Face Tracking Robot

## YouTube Demo

<https://www.youtube.com/shorts/XLtfobL6sic?feature=share>

## BOM

- Luxonis Oak-D Lite AI Robotics Camera
- Raspberry Pi 4 8 GB
- [stls](./stls) printed with Polymaker PLA Matte Pastel Candy
- Raspberry Pi case
- 2 SG90 Servos
- Some M-to-F and M-to-M jumper wires
- Some 3x and 2x Wago Connectors

Some other things:

- Travel router
- Power banks to power router and Raspberry Pi
- USB-C cables for power-delivery

## Robot

See [object.py](./object.py) for the main robot code.

All servo control, PID control, and object tracking code are in this file.

This python file exposes `0.0.0.0:5000/tune` and `0.0.0.0:5000/feed` endpoints for PID tuning and object tracking preview respectively.
