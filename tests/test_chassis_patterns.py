#!/usr/bin/env python3
import time
from test_wheels_uart import Esp32WheelClient

PORT = "/dev/ttyTHS1"
BAUD = 115200
SPEED = 1500.0
RUN_S = 2.5
PAUSE_S = 1.0

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

def run_pattern(client, name, fl, rl, fr, rr):
    print(f"\n{name}: FL={fl:+.1f} RL={rl:+.1f} FR={fr:+.1f} RR={rr:+.1f}")
    ok = send_all_seq(client, fl, rl, fr, rr)
    print("  ACK:", ok)
    if not ok:
      return False
    time.sleep(RUN_S)
    ok = client.stop_all()
    print("  STOP ACK:", ok)
    time.sleep(PAUSE_S)
    return ok

def main():
    client = Esp32WheelClient(PORT, BAUD)

    try:
        patterns = [
	    ("STRAFE_LEFT_CANDIDATE", -1000, +1200, +1000, -1100),
            ("STRAFE_RIGHT_CANDIDATE", +1000, -1200, -1000, +1100),

        ]

        for p in patterns:
            ok = run_pattern(client, *p)
            if not ok:
                break

    finally:
        try:
            client.stop_all()
        except Exception:
            pass
        client.close()

if __name__ == "__main__":
    main()
