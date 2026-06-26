#!/usr/bin/env python3
import time
from robot_esp32_bridge.esp32_protocol import Esp32Protocol

MAX_WHEEL_TARGET = 1800.0


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


class RobotDrive:
    def __init__(self, port="/dev/ttyTHS1", baud=115200, inter_cmd_delay=0.03):
        self.proto = Esp32Protocol(port=port, baudrate=baud)
        self.inter_cmd_delay = inter_cmd_delay

        self.last_vx = 0.0
        self.last_vy = 0.0
        self.last_wz = 0.0

        self.last_fl = 0.0
        self.last_rl = 0.0
        self.last_fr = 0.0
        self.last_rr = 0.0

    def close(self):
        self.proto.close()

    def stop(self):
        self.last_vx = 0.0
        self.last_vy = 0.0
        self.last_wz = 0.0
        self.last_fl = 0.0
        self.last_rl = 0.0
        self.last_fr = 0.0
        self.last_rr = 0.0
        return self.proto.wheels_stop()

    def mecanum_mix(self, vx, vy, wz):
        fl = +vx + vy - wz
        rl = +vx - vy - wz
        fr = +vx - vy + wz
        rr = +vx + vy + wz

        fl = clamp(fl, -MAX_WHEEL_TARGET, MAX_WHEEL_TARGET)
        rl = clamp(rl, -MAX_WHEEL_TARGET, MAX_WHEEL_TARGET)
        fr = clamp(fr, -MAX_WHEEL_TARGET, MAX_WHEEL_TARGET)
        rr = clamp(rr, -MAX_WHEEL_TARGET, MAX_WHEEL_TARGET)

        return fl, rl, fr, rr

    def _send_wheels(self, fl, rl, fr, rr):
        ok = True
        ok &= self.proto.set_wheel_speed(0, fl)   # FL
        time.sleep(self.inter_cmd_delay)
        ok &= self.proto.set_wheel_speed(1, rl)   # RL
        time.sleep(self.inter_cmd_delay)
        ok &= self.proto.set_wheel_speed(2, fr)   # FR
        time.sleep(self.inter_cmd_delay)
        ok &= self.proto.set_wheel_speed(3, rr)   # RR
        return ok

    def set_velocity(self, vx, vy, wz):
        fl, rl, fr, rr = self.mecanum_mix(vx, vy, wz)
        ok = self._send_wheels(fl, rl, fr, rr)

        self.last_vx = vx
        self.last_vy = vy
        self.last_wz = wz

        self.last_fl = fl
        self.last_rl = rl
        self.last_fr = fr
        self.last_rr = rr

        return ok

    def get_last_command(self):
        return {
            "vx": self.last_vx,
            "vy": self.last_vy,
            "wz": self.last_wz,
            "fl": self.last_fl,
            "rl": self.last_rl,
            "fr": self.last_fr,
            "rr": self.last_rr,
        }
