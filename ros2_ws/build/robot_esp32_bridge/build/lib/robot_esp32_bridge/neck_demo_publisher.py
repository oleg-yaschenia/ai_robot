import rclpy
from rclpy.node import Node
from std_msgs.msg import Int16MultiArray


class NeckDemoPublisher(Node):
    def __init__(self):
        super().__init__("neck_demo_publisher")

        self.pub = self.create_publisher(Int16MultiArray, "/robot/neck/cmd", 10)
        self.timer = self.create_timer(3.5, self.on_timer)
        self.step = 0

        self.get_logger().info("Neck demo publisher started")

    def publish_pose(self, yaw: int, lift: int, pitch_offset: int):
        msg = Int16MultiArray()
        msg.data = [yaw, lift, pitch_offset]
        self.pub.publish(msg)
        self.get_logger().info(f"Published: {msg.data}")

    def on_timer(self):
        if self.step == 0:
            self.publish_pose(-1, -1, -1)  # home
        elif self.step == 1:
            self.publish_pose(385, 80, 0)
        elif self.step == 2:
            self.publish_pose(385, 160, 0)
        elif self.step == 3:
            self.publish_pose(385, 160, 80)
        elif self.step == 4:
            self.publish_pose(500, 0, 0)
        elif self.step == 5:
            self.publish_pose(300, 0, 0)
        else:
            self.publish_pose(-1, -1, -1)
            self.step = -1

        self.step += 1


def main(args=None):
    rclpy.init(args=args)
    node = NeckDemoPublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
