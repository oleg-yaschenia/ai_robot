#!/usr/bin/env python3
import sys
import time
import termios
import tty
import select

from test_wheels_uart import Esp32WheelClient

PORT = "/dev/ttyTHS1"
BAUD = 115200

MAX_WHEEL_TARGET = 1800.0

VX_SPEED = 900.0
VY_SPEED = 1500.0
WZ_SPEED = 900.0

SEND_PERIOD = 0.20
TELEM_PERIOD = 0.30


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def mecanum_mix(vx, vy, wz):
    fl = +vx + vy - wz
    rl = +vx - vy - wz
    fr = +vx - vy + wz
    rr = +vx + vy + wz

    fl = clamp(fl, -MAX_WHEEL_TARGET, MAX_WHEEL_TARGET)
    rl = clamp(rl, -MAX_WHEEL_TARGET, MAX_WHEEL_TARGET)
    fr = clamp(fr, -MAX_WHEEL_TARGET, MAX_WHEEL_TARGET)
    rr = clamp(rr, -MAX_WHEEL_TARGET, MAX_WHEEL_TARGET)

    return fl, rl, fr, rr


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


def read_key_nonblocking():
    dr, _, _ = select.select([sys.stdin], [], [], 0)
    if dr:
        return sys.stdin.read(1)
    return None


def print_help():
    print()
    print("Teleop controls:")
    print("  w - forward")
    print("  s - backward")
    print("  a - strafe left")
    print("  d - strafe right")
    print("  q - rotate left")
    print("  e - rotate right")
    print("  x - stop")
    print("  z - reduce speeds")
    print("  c - increase speeds")
    print("  h - help")
    print("  Ctrl+C - exit")
    print()
    print(f"Current speeds: VX={VX_SPEED:.0f} VY={VY_SPEED:.0f} WZ={WZ_SPEED:.0f}")
    print()


def format_telem(telem):
    if telem is None:
        return "TELEM=None"

    return (
        f"TELEM FL={telem['fl']:+.0f} "
        f"RL={telem['rl']:+.0f} "
        f"FR={telem['fr']:+.0f} "
        f"RR={telem['rr']:+.0f} "
        f"YAW={telem['yaw']:+.2f}"
    )


def main():
    global VX_SPEED, VY_SPEED, WZ_SPEED

    client = Esp32WheelClient(PORT, BAUD)
    old_settings = termios.tcgetattr(sys.stdin)

    vx_cmd = 0.0
    vy_cmd = 0.0
    wz_cmd = 0.0

    last_send = 0.0
    last_telem = 0.0
    last_sent = None
    last_telem_str = "TELEM=not requested yet"

    try:
        tty.setcbreak(sys.stdin.fileno())
        print_help()

        while True:
            key = read_key_nonblocking()

            if key is not None:
                if key == "w":
                    vx_cmd = +VX_SPEED
                    vy_cmd = 0.0
                    wz_cmd = 0.0
                elif key == "s":
                    vx_cmd = -VX_SPEED
                    vy_cmd = 0.0
                    wz_cmd = 0.0
                elif key == "a":
                    vx_cmd = 0.0
                    vy_cmd = -VY_SPEED
                    wz_cmd = 0.0
                elif key == "d":
                    vx_cmd = 0.0
                    vy_cmd = +VY_SPEED
                    wz_cmd = 0.0
                elif key == "q":
                    vx_cmd = 0.0
                    vy_cmd = 0.0
                    wz_cmd = -WZ_SPEED
                elif key == "e":
                    vx_cmd = 0.0
                    vy_cmd = 0.0
                    wz_cmd = +WZ_SPEED
                elif key == "x":
                    vx_cmd = 0.0
                    vy_cmd = 0.0
                    wz_cmd = 0.0
                    print("\nCMD STOP")
                elif key == "z":
                    VX_SPEED = max(200.0, VX_SPEED - 100.0)
                    VY_SPEED = max(300.0, VY_SPEED - 100.0)
                    WZ_SPEED = max(200.0, WZ_SPEED - 100.0)
                    print(f"\nSpeeds: VX={VX_SPEED:.0f} VY={VY_SPEED:.0f} WZ={WZ_SPEED:.0f}")
                elif key == "c":
                    VX_SPEED = min(1800.0, VX_SPEED + 100.0)
                    VY_SPEED = min(1800.0, VY_SPEED + 100.0)
                    WZ_SPEED = min(1800.0, WZ_SPEED + 100.0)
                    print(f"\nSpeeds: VX={VX_SPEED:.0f} VY={VY_SPEED:.0f} WZ={WZ_SPEED:.0f}")
                elif key == "h":
                    print_help()

            now = time.time()

            if now - last_send >= SEND_PERIOD:
                last_send = now

                fl, rl, fr, rr = mecanum_mix(vx_cmd, vy_cmd, wz_cmd)
                current_cmd = (int(fl), int(rl), int(fr), int(rr))

                if current_cmd != last_sent:
                    ok = send_all_seq(client, fl, rl, fr, rr)
                    last_sent = current_cmd

                    print(
                        f"\rCMD vx={vx_cmd:+.0f} vy={vy_cmd:+.0f} wz={wz_cmd:+.0f} | "
                        f"FL={fl:+.0f} RL={rl:+.0f} FR={fr:+.0f} RR={rr:+.0f} | "
                        f"ACK={ok} | {last_telem_str}      ",
                        end="",
                        flush=True,
                    )

            if now - last_telem >= TELEM_PERIOD:
                last_telem = now
                telem = client.get_drive_telemetry()
                last_telem_str = format_telem(telem)

                fl, rl, fr, rr = mecanum_mix(vx_cmd, vy_cmd, wz_cmd)
                print(
                    f"\rCMD vx={vx_cmd:+.0f} vy={vy_cmd:+.0f} wz={wz_cmd:+.0f} | "
                    f"FL={fl:+.0f} RL={rl:+.0f} FR={fr:+.0f} RR={rr:+.0f} | "
                    f"{last_telem_str}      ",
                    end="",
                    flush=True,
                )

    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        try:
            client.stop_all()
        except Exception:
            pass
        client.close()
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)


if __name__ == "__main__":
    main()
