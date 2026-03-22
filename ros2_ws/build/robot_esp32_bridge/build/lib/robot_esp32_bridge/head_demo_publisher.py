import sys
import rclpy
from rclpy.node import Node
from std_msgs.msg import UInt8MultiArray


class HeadDemoPublisher(Node):
    def __init__(self, mode: int, brightness: int, level: int, style: int):
        super().__init__("head_demo_publisher")
        self.pub = self.create_publisher(UInt8MultiArray, "/robot/head/cmd", 10)
        self.msg = UInt8MultiArray()
        self.msg.data = [mode, brightness, level, style]

        self.timer = self.create_timer(0.2, self.publish_once)
        self.sent_count = 0

    def publish_once(self):
        self.pub.publish(self.msg)
        self.sent_count += 1
        if self.sent_count >= 5:
            self.get_logger().info(f"Published head cmd: {list(self.msg.data)}")
            rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)

    argv = sys.argv
    mode = int(argv[1]) if len(argv) > 1 else 1
    brightness = int(argv[2]) if len(argv) > 2 else 64
    level = int(argv[3]) if len(argv) > 3 else 100
    style = int(argv[4]) if len(argv) > 4 else 0

    node = HeadDemoPublisher(mode, brightness, level, style)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
