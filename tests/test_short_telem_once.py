#!/usr/bin/env python3
from test_wheels_uart import Esp32WheelClient

client = Esp32WheelClient("/dev/ttyTHS1", 115200)

try:
    print("YAW =", client.get_yaw_telemetry())
    print("WS1 =", client.get_wheel_speeds_1())
    print("WS2 =", client.get_wheel_speeds_2())
finally:
    client.close()
