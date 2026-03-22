import rclpy
from rclpy.node import Node
from std_msgs.msg import Int16MultiArray

from .esp32_protocol import Esp32Protocol


class NeckBridgeNode(Node):
    def __init__(self):
        super().__init__("neck_bridge_node")

        self.declare_parameter("port", "/dev/ttyTHS1")
        self.declare_parameter("baudrate", 115200)

        port = self.get_parameter("port").value
        baudrate = self.get_parameter("baudrate").value

        self.proto = Esp32Protocol(port=port, baudrate=baudrate)

        self.sub = self.create_subscription(
            Int16MultiArray,
            "/robot/neck/cmd",
            self.cmd_callback,
            10,
        )

        self.get_logger().info(f"Neck bridge ready on {port} @ {baudrate}")

    def cmd_callback(self, msg: Int16MultiArray):
        data = list(msg.data)

        if len(data) < 3:
            self.get_logger().warning("Expected 3 values: [yaw, lift, pitch_offset]")
            return

        yaw, lift, pitch_offset = int(data[0]), int(data[1]), int(data[2])

        # special home command
        if yaw == -1 and lift == -1 and pitch_offset == -1:
            ok = self.proto.neck_home()
            self.get_logger().info(f"neck_home -> {ok}")
            return

        ok = self.proto.neck_set_pose(yaw, lift, pitch_offset)
        self.get_logger().info(
            f"neck_set_pose(yaw={yaw}, lift={lift}, pitch_offset={pitch_offset}) -> {ok}"
        )

    def destroy_node(self):
        try:
            self.proto.close()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = NeckBridgeNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
