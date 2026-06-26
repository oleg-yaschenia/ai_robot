#!/usr/bin/env python3
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import yaml


def gstreamer_pipeline(sensor_id: int, width: int, height: int, fps: int) -> str:
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width=(int){width}, height=(int){height}, "
        f"format=(string)NV12, framerate=(fraction){fps}/1 ! "
        f"nvvidconv ! video/x-raw, format=(string)BGRx ! "
        f"videoconvert ! video/x-raw, format=(string)BGR ! "
        f"appsink drop=true max-buffers=1 sync=false"
    )


def load_camera_info_from_yaml(path: str, frame_id: str, stamp):
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    msg = CameraInfo()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id

    msg.width = int(data["image_width"])
    msg.height = int(data["image_height"])
    msg.distortion_model = data["distortion_model"]

    msg.d = [float(x) for x in data["distortion_coefficients"]["data"]]
    msg.k = [float(x) for x in data["camera_matrix"]["data"]]
    msg.r = [float(x) for x in data["rectification_matrix"]["data"]]
    msg.p = [float(x) for x in data["projection_matrix"]["data"]]

    return msg
    
    
class StereoCameraNode(Node):
    def __init__(self):
        super().__init__("stereo_camera_node")

        self.declare_parameter("left_sensor_id", 0)
        self.declare_parameter("right_sensor_id", 1)
        self.declare_parameter("width", 1280)
        self.declare_parameter("height", 720)
        self.declare_parameter("fps", 30)
        self.declare_parameter("left_frame_id", "camera_left_optical_frame")
        self.declare_parameter("right_frame_id", "camera_right_optical_frame")
        self.declare_parameter("left_camera_info_yaml", "/home/warxen/ai_robot/calib/stereo/left.yaml")
        self.declare_parameter("right_camera_info_yaml", "/home/warxen/ai_robot/calib/stereo/right.yaml")

        self.left_sensor_id = int(self.get_parameter("left_sensor_id").value)
        self.right_sensor_id = int(self.get_parameter("right_sensor_id").value)
        self.width = int(self.get_parameter("width").value)
        self.height = int(self.get_parameter("height").value)
        self.fps = int(self.get_parameter("fps").value)
        self.left_frame_id = str(self.get_parameter("left_frame_id").value)
        self.right_frame_id = str(self.get_parameter("right_frame_id").value)
        self.left_camera_info_yaml = str(self.get_parameter("left_camera_info_yaml").value)
        self.right_camera_info_yaml = str(self.get_parameter("right_camera_info_yaml").value)

        self.bridge = CvBridge()

        self.left_pub = self.create_publisher(Image, "/camera/left/image_raw", 10)
        self.right_pub = self.create_publisher(Image, "/camera/right/image_raw", 10)
        self.left_info_pub = self.create_publisher(CameraInfo, "/camera/left/camera_info", 10)
        self.right_info_pub = self.create_publisher(CameraInfo, "/camera/right/camera_info", 10)

        left_pipeline = gstreamer_pipeline(self.left_sensor_id, self.width, self.height, self.fps)
        right_pipeline = gstreamer_pipeline(self.right_sensor_id, self.width, self.height, self.fps)

        self.left_cap = cv2.VideoCapture(left_pipeline, cv2.CAP_GSTREAMER)
        self.right_cap = cv2.VideoCapture(right_pipeline, cv2.CAP_GSTREAMER)

        if not self.left_cap.isOpened():
            raise RuntimeError("Failed to open left camera")
        if not self.right_cap.isOpened():
            raise RuntimeError("Failed to open right camera")

        self.timer = self.create_timer(1.0 / float(self.fps), self.timer_cb)

        self.get_logger().info(
            f"stereo_camera_node started: left={self.left_sensor_id}, "
            f"right={self.right_sensor_id}, {self.width}x{self.height}@{self.fps}"
        )


    def timer_cb(self):
        ok_l, frame_l = self.left_cap.read()
        ok_r, frame_r = self.right_cap.read()

        if not ok_l or frame_l is None:
            self.get_logger().warning("Left camera frame read failed")
            return
        if not ok_r or frame_r is None:
            self.get_logger().warning("Right camera frame read failed")
            return

        stamp = self.get_clock().now().to_msg()

        left_msg = self.bridge.cv2_to_imgmsg(frame_l, encoding="bgr8")
        left_msg.header.stamp = stamp
        left_msg.header.frame_id = self.left_frame_id

        right_msg = self.bridge.cv2_to_imgmsg(frame_r, encoding="bgr8")
        right_msg.header.stamp = stamp
        right_msg.header.frame_id = self.right_frame_id

        self.left_pub.publish(left_msg)
        self.right_pub.publish(right_msg)
        self.left_info_pub.publish(
            load_camera_info_from_yaml(self.left_camera_info_yaml, self.left_frame_id, stamp)
        )
        self.right_info_pub.publish(
            load_camera_info_from_yaml(self.right_camera_info_yaml, self.right_frame_id, stamp)
        )


    def destroy_node(self):
        try:
            if self.left_cap is not None:
                self.left_cap.release()
            if self.right_cap is not None:
                self.right_cap.release()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = StereoCameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
