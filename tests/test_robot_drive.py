#!/usr/bin/env python3
import time
from robot_drive import RobotDrive


def run_case(drive, name, vx, vy, wz, run_s=1.0, pause_s=0.6):
    print()
    print(f"{name}: vx={vx:+.1f} vy={vy:+.1f} wz={wz:+.1f}")

    ok = drive.set_velocity(vx, vy, wz)
    print("SET ACK:", ok)
    print("LAST CMD:", drive.get_last_command())

    time.sleep(run_s)

    ok = drive.stop()
    print("STOP ACK:", ok)

    time.sleep(pause_s)


def main():
    drive = RobotDrive(port="/dev/ttyTHS1", baud=115200, inter_cmd_delay=0.03)
    try:
        run_case(drive, "FORWARD",  +900.0,   0.0,   0.0, 1.0, 0.8)
        run_case(drive, "BACKWARD", -900.0,   0.0,   0.0, 1.0, 0.8)
        run_case(drive, "ROTATE",     0.0,   0.0, +900.0, 0.8, 0.8)
        run_case(drive, "STRAFE",     0.0, +1500.0, 0.0, 0.8, 0.8)
    finally:
        try:
            drive.stop()
        except Exception:
            pass
        drive.close()


if __name__ == "__main__":
    main()
