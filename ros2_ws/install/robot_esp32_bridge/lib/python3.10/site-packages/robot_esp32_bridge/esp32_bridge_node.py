import rclpy
from rclpy.node import Node
from std_msgs.msg import UInt8MultiArray

from .esp32_protocol import Esp32Protocol


class Esp32BridgeNode(Node):
    def __init__(self):
        super().__init__("esp32_bridge_node")

        port = self.declare_parameter("port", "/dev/ttyTHS1").value
        baudrate = self.declare_parameter("baudrate", 115200).value

        self.proto = Esp32Protocol(port=port, baudrate=baudrate)

        self.last_mode = None
        self.last_brightness = None
        self.last_level = None
        self.last_style = None

        self.sub = self.create_subscription(
            UInt8MultiArray,
            "/robot/head/cmd",
            self.on_head_cmd,
            10,
        )

        self.get_logger().info(f"ESP32 bridge started on {port} @ {baudrate}")

    def on_head_cmd(self, msg: UInt8MultiArray) -> None:
        data = list(msg.data)

        if len(data) < 4:
            self.get_logger().warn("head cmd must contain [mode, brightness, level, style]")
            return

        mode, brightness, level, style = data[:4]

        try:
            if mode != self.last_mode:
                self.proto.set_head_mode(mode)
                self.last_mode = mode

            if brightness != self.last_brightness:
                self.proto.set_head_brightness(brightness)
                self.last_brightness = brightness

            if style != self.last_style:
                self.proto.set_head_speaking_style(style)
                self.last_style = style

            if level != self.last_level:
                self.proto.set_head_level(level)
                self.last_level = level

        except Exception as e:
            self.get_logger().error(f"Failed to send command to ESP32: {e}")

    def destroy_node(self):
        try:
            self.proto.close()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = Esp32BridgeNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
