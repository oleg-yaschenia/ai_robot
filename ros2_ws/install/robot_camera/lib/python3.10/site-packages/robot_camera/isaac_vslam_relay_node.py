#!/usr/bin/env python3
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge


class IsaacVslamRelayNode(Node):
    def __init__(self):
        super().__init__("isaac_vslam_relay_node")

        self.bridge = CvBridge()

        self.left_img_pub = self.create_publisher(Image, "/visual_slam/image_0", 10)
        self.right_img_pub = self.create_publisher(Image, "/visual_slam/image_1", 10)
        self.left_info_pub = self.create_publisher(CameraInfo, "/visual_slam/camera_info_0", 10)
        self.right_info_pub = self.create_publisher(CameraInfo, "/visual_slam/camera_info_1", 10)

        self.create_subscription(Image, "/camera/left/image_rect", self.left_image_cb, 10)
        self.create_subscription(Image, "/camera/right/image_rect", self.right_image_cb, 10)
        self.create_subscription(CameraInfo, "/camera/left/camera_info", self.left_info_cb, 10)
        self.create_subscription(CameraInfo, "/camera/right/camera_info", self.right_info_cb, 10)

        self.get_logger().info("isaac_vslam_relay_node started")

    def left_image_cb(self, msg: Image):
        cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        out = self.bridge.cv2_to_imgmsg(gray, encoding="mono8")
        out.header = msg.header
        self.left_img_pub.publish(out)

    def right_image_cb(self, msg: Image):
        cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        out = self.bridge.cv2_to_imgmsg(gray, encoding="mono8")
        out.header = msg.header
        self.right_img_pub.publish(out)

    def left_info_cb(self, msg: CameraInfo):
        self.left_info_pub.publish(msg)

    def right_info_cb(self, msg: CameraInfo):
        self.right_info_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = IsaacVslamRelayNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
