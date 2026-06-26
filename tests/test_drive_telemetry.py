#!/usr/bin/env python3
import time
from test_wheels_uart import Esp32WheelClient

PORT = "/dev/ttyTHS1"
BAUD = 115200


def send_all_seq(client, fl, rl, fr, rr):
    ok = True
    ok &= client.set_wheel_speed("fl", fl)
    time.sleep(0.03)
    ok &= client.set_wheel_speed("rl", rl)
    time.sleep(0.03)
    ok &= client.set_wheel_speed("fr", fr)
    time.sleep(0.03)
    ok &= client.set_wheel_speed("rr", rr)
    return ok


def print_telem(client):
    ws1 = client.get_wheel_speeds_1()
    ws2 = client.get_wheel_speeds_2()
    yaw = client.get_yaw_telemetry()

    print("WS1 =", ws1, "WS2 =", ws2, "YAW =", yaw)


def run_case(client, name, fl, rl, fr, rr, run_s=1.2):
    print()
    print(f"{name}: FL={fl:+.1f} RL={rl:+.1f} FR={fr:+.1f} RR={rr:+.1f}")

    ok = send_all_seq(client, fl, rl, fr, rr)
    print("SET ACK:", ok)

    time.sleep(0.40)

    t0 = time.time()
    while time.time() - t0 < run_s:
        print_telem(client)
        time.sleep(0.25)

    ok = client.stop_all()
    print("STOP ACK:", ok)

    time.sleep(0.35)
    print("AFTER STOP:")
    print_telem(client)


def main():
    client = Esp32WheelClient(PORT, BAUD)
    try:
        run_case(client, "FORWARD",  +900.0, +900.0, +900.0, +900.0, 1.0)
        run_case(client, "ROTATE",   -900.0, -900.0, +900.0, +900.0, 1.0)
        run_case(client, "STRAFE",  +1500.0, -1500.0, -1500.0, +1500.0, 1.0)
    finally:
        try:
            client.stop_all()
        except Exception:
            pass
        client.close()


if __name__ == "__main__":
    main()
