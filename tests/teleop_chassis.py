#!/usr/bin/env python3
import sys
import time
import termios
import tty
import select

from robot_drive import RobotDrive

VX_SPEED = 900.0
VY_SPEED = 1500.0
WZ_SPEED = 900.0

SEND_PERIOD = 0.20


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


def main():
    global VX_SPEED, VY_SPEED, WZ_SPEED

    drive = RobotDrive(port="/dev/ttyTHS1", baud=115200, inter_cmd_delay=0.03)
    old_settings = termios.tcgetattr(sys.stdin)

    vx_cmd = 0.0
    vy_cmd = 0.0
    wz_cmd = 0.0

    last_send = 0.0
    last_sent = None

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
                    drive.stop()
                    last_sent = None
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
                current_cmd = (int(vx_cmd), int(vy_cmd), int(wz_cmd))

                if current_cmd != last_sent:
                    ok = drive.set_velocity(vx_cmd, vy_cmd, wz_cmd)
                    last_sent = current_cmd

                    last = drive.get_last_command()
                    print(
                        f"\rCMD vx={last['vx']:+.0f} vy={last['vy']:+.0f} wz={last['wz']:+.0f} | "
                        f"FL={last['fl']:+.0f} RL={last['rl']:+.0f} FR={last['fr']:+.0f} RR={last['rr']:+.0f} | "
                        f"ACK={ok}      ",
                        end="",
                        flush=True,
                    )

    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        try:
            drive.stop()
        except Exception:
            pass
        drive.close()
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)


if __name__ == "__main__":
    main()
