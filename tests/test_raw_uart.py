#!/usr/bin/env python3
import time
from test_wheels_uart import Esp32WheelClient, CMD_GET_DRIVE_TELEMETRY

client = Esp32WheelClient("/dev/ttyTHS1", 115200)

try:
    client.ser.reset_input_buffer()
    seq = client.send_packet(CMD_GET_DRIVE_TELEMETRY, b"")
    print("sent seq =", seq)

    time.sleep(0.30)

    n = client.ser.in_waiting
    print("bytes waiting =", n)

    if n > 0:
        raw = client.ser.read(n)
        print("raw hex =", raw.hex())
    else:
        print("no raw bytes")
finally:
    client.close()
