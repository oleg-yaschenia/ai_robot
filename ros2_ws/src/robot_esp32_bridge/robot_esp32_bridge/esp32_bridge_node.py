#!/usr/bin/env python3

from __future__ import annotations

import time
from typing import Callable

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import Int16MultiArray, UInt8MultiArray

from .esp32_protocol import Esp32Protocol


class Esp32BridgeNode(Node):
    """The only ROS 2 process allowed to own the Jetson <-> ESP32 UART."""

    def __init__(self) -> None:
        super().__init__("esp32_bridge_node")

        port = str(self.declare_parameter("port", "/dev/ttyTHS1").value)
        baudrate = int(self.declare_parameter("baudrate", 115200).value)

        self.enable_drive = bool(self.declare_parameter("enable_drive", True).value)
        self.enable_head = bool(self.declare_parameter("enable_head", True).value)
        self.enable_neck = bool(self.declare_parameter("enable_neck", True).value)

        self.k_vx = float(self.declare_parameter("k_vx", 900.0).value)
        self.k_vy = float(self.declare_parameter("k_vy", 1500.0).value)
        self.k_wz = float(self.declare_parameter("k_wz", 900.0).value)
        self.max_wheel_target = float(
            self.declare_parameter("max_wheel_target", 1800.0).value
        )
        self.cmd_timeout_sec = float(
            self.declare_parameter("cmd_timeout_sec", 0.5).value
        )
        self.inter_cmd_delay_sec = float(
            self.declare_parameter("inter_cmd_delay_sec", 0.03).value
        )
        self.use_all_wheels_command = bool(
            self.declare_parameter("use_all_wheels_command", False).value
        )

        self.proto = Esp32Protocol(port=port, baudrate=baudrate)

        self.last_mode: int | None = None
        self.last_brightness: int | None = None
        self.last_level: int | None = None
        self.last_style: int | None = None

        self.last_cmd_time = self.get_clock().now()
        self.wheels_stopped = True

        self.head_sub = None
        self.neck_sub = None
        self.cmd_vel_sub = None
        self.watchdog_timer = None

        if self.enable_head:
            self.head_sub = self.create_subscription(
                UInt8MultiArray,
                "/robot/head/cmd",
                self.on_head_cmd,
                10,
            )

        if self.enable_neck:
            self.neck_sub = self.create_subscription(
                Int16MultiArray,
                "/robot/neck/cmd",
                self.on_neck_cmd,
                10,
            )

        if self.enable_drive:
            self.cmd_vel_sub = self.create_subscription(
                Twist,
                "/cmd_vel",
                self.on_cmd_vel,
                10,
            )
            self.watchdog_timer = self.create_timer(0.1, self.on_drive_watchdog)

        self.get_logger().info(
            "ESP32 transport started: "
            f"port={port}, baudrate={baudrate}, "
            f"drive={self.enable_drive}, head={self.enable_head}, "
            f"neck={self.enable_neck}"
        )

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    def _mecanum_mix(
        self,
        vx: float,
        vy: float,
        wz: float,
    ) -> tuple[float, float, float, float]:
        fl = vx + vy - wz
        rl = vx - vy - wz
        fr = vx - vy + wz
        rr = vx + vy + wz

        limit = self.max_wheel_target
        return (
            self._clamp(fl, -limit, limit),
            self._clamp(rl, -limit, limit),
            self._clamp(fr, -limit, limit),
            self._clamp(rr, -limit, limit),
        )

    def _run_command(self, label: str, command: Callable[[], bool]) -> bool:
        try:
            ok = bool(command())
        except Exception as exc:
            self.get_logger().error(f"{label} failed: {exc}")
            return False

        if not ok:
            self.get_logger().warning(f"{label}: ESP32 ACK timeout")
        return ok

    def _send_wheels(
        self,
        fl: float,
        rl: float,
        fr: float,
        rr: float,
    ) -> bool:
        if self.use_all_wheels_command:
            return self._run_command(
                "set_all_wheel_speeds",
                lambda: self.proto.set_all_wheel_speeds(fl, rl, fr, rr),
            )

        commands = (
            (0, fl, "FL"),
            (1, rl, "RL"),
            (2, fr, "FR"),
            (3, rr, "RR"),
        )
        all_ok = True

        for index, (wheel_id, speed, name) in enumerate(commands):
            ok = self._run_command(
                f"set_wheel_speed {name}",
                lambda wheel_id=wheel_id, speed=speed: self.proto.set_wheel_speed(
                    wheel_id, speed
                ),
            )
            all_ok = all_ok and ok
            if index < len(commands) - 1 and self.inter_cmd_delay_sec > 0.0:
                time.sleep(self.inter_cmd_delay_sec)

        return all_ok

    def _stop_wheels(self, reason: str) -> bool:
        if self.wheels_stopped:
            return True

        ok = self._run_command("wheels_stop", self.proto.wheels_stop)
        if ok:
            self.wheels_stopped = True
            self.get_logger().info(f"Wheels stopped: {reason}")
        return ok

    def on_cmd_vel(self, msg: Twist) -> None:
        vx = float(msg.linear.x) * self.k_vx
        vy = float(msg.linear.y) * self.k_vy
        wz = float(msg.angular.z) * self.k_wz
        fl, rl, fr, rr = self._mecanum_mix(vx, vy, wz)

        ok = self._send_wheels(fl, rl, fr, rr)
        if not ok:
            self.get_logger().error("Wheel command failed; requesting emergency stop")
            self.wheels_stopped = False
            self._stop_wheels("wheel command failure")
            return

        self.last_cmd_time = self.get_clock().now()
        self.wheels_stopped = all(
            abs(value) < 1.0e-6 for value in (fl, rl, fr, rr)
        )

        self.get_logger().debug(
            f"cmd_vel -> vx={vx:.1f} vy={vy:.1f} wz={wz:.1f} | "
            f"FL={fl:.1f} RL={rl:.1f} FR={fr:.1f} RR={rr:.1f}"
        )

    def on_drive_watchdog(self) -> None:
        if self.wheels_stopped:
            return

        elapsed = (
            self.get_clock().now() - self.last_cmd_time
        ).nanoseconds / 1.0e9
        if elapsed > self.cmd_timeout_sec:
            self._stop_wheels(f"cmd_vel timeout ({elapsed:.2f} s)")

    def on_head_cmd(self, msg: UInt8MultiArray) -> None:
        data = list(msg.data)
        if len(data) < 4:
            self.get_logger().warning(
                "Head command must contain [mode, brightness, level, style]"
            )
            return

        mode, brightness, level, style = (int(value) for value in data[:4])

        if mode != self.last_mode and self._run_command(
            "set_head_mode",
            lambda: self.proto.set_head_mode(mode),
        ):
            self.last_mode = mode

        if brightness != self.last_brightness and self._run_command(
            "set_head_brightness",
            lambda: self.proto.set_head_brightness(brightness),
        ):
            self.last_brightness = brightness

        if style != self.last_style and self._run_command(
            "set_head_speaking_style",
            lambda: self.proto.set_head_speaking_style(style),
        ):
            self.last_style = style

        if level != self.last_level and self._run_command(
            "set_head_level",
            lambda: self.proto.set_head_level(level),
        ):
            self.last_level = level

    def on_neck_cmd(self, msg: Int16MultiArray) -> None:
        data = list(msg.data)
        if len(data) < 3:
            self.get_logger().warning(
                "Neck command must contain [yaw, lift, pitch_offset]"
            )
            return

        yaw, lift, pitch_offset = (int(value) for value in data[:3])

        if yaw == -1 and lift == -1 and pitch_offset == -1:
            self._run_command("neck_home", self.proto.neck_home)
            return

        self._run_command(
            "neck_set_pose",
            lambda: self.proto.neck_set_pose(yaw, lift, pitch_offset),
        )

    def destroy_node(self) -> None:
        if self.enable_drive:
            try:
                self.wheels_stopped = False
                self._stop_wheels("bridge shutdown")
            except Exception as exc:
                self.get_logger().error(f"Failed to stop wheels on shutdown: {exc}")

        try:
            self.proto.close()
        except Exception as exc:
            self.get_logger().error(f"Failed to close ESP32 serial port: {exc}")

        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = Esp32BridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
