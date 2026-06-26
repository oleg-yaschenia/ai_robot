#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

from robot_esp32_bridge.robot_drive import RobotDrive


class DriveBridgeNode(Node):
    def __init__(self):
        super().__init__("drive_bridge_node")

        self.declare_parameter("port", "/dev/ttyTHS1")
        self.declare_parameter("baud", 115200)
        self.declare_parameter("k_vx", 900.0)
        self.declare_parameter("k_vy", 1500.0)
        self.declare_parameter("k_wz", 900.0)
        self.declare_parameter("cmd_timeout_sec", 0.5)

        port = self.get_parameter("port").value
        baud = int(self.get_parameter("baud").value)

        self.k_vx = float(self.get_parameter("k_vx").value)
        self.k_vy = float(self.get_parameter("k_vy").value)
        self.k_wz = float(self.get_parameter("k_wz").value)
        self.cmd_timeout_sec = float(self.get_parameter("cmd_timeout_sec").value)

        self.drive = RobotDrive(port=port, baud=baud, inter_cmd_delay=0.03)
        self.last_cmd_time = self.get_clock().now()

        self.sub = self.create_subscription(
            Twist,
            "/cmd_vel",
            self.cmd_vel_callback,
            10,
        )

        self.timer = self.create_timer(0.1, self.watchdog_timer_cb)

        self.get_logger().info(
            f"drive_bridge_node started: port={port}, baud={baud}, "
            f"k_vx={self.k_vx}, k_vy={self.k_vy}, k_wz={self.k_wz}"
        )

    def cmd_vel_callback(self, msg: Twist):
        vx = float(msg.linear.x) * self.k_vx
        vy = float(msg.linear.y) * self.k_vy
        wz = float(msg.angular.z) * self.k_wz

        ok = self.drive.set_velocity(vx, vy, wz)
        self.last_cmd_time = self.get_clock().now()

        last = self.drive.get_last_command()
        self.get_logger().debug(
            f"cmd_vel -> vx={vx:.1f} vy={vy:.1f} wz={wz:.1f} | "
            f"FL={last['fl']:.1f} RL={last['rl']:.1f} "
            f"FR={last['fr']:.1f} RR={last['rr']:.1f} | ACK={ok}"
        )

    def watchdog_timer_cb(self):
        dt = (self.get_clock().now() - self.last_cmd_time).nanoseconds / 1e9
        if dt > self.cmd_timeout_sec:
            self.drive.stop()

    def destroy_node(self):
        try:
            self.drive.stop()
        except Exception:
            pass
        try:
            self.drive.close()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = DriveBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
