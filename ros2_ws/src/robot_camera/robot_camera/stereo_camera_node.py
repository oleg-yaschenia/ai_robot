#!/usr/bin/env python3

import cv2
import yaml
import rclpy

from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image


def gstreamer_pipeline(
    sensor_id: int,
    sensor_mode: int,
    sensor_width: int,
    sensor_height: int,
    output_width: int,
    output_height: int,
    fps: int,
    interpolation_method: int,
    ee_mode: int,
    ee_strength: float,
) -> str:
    return (
        f"nvarguscamerasrc "
        f"sensor-id={sensor_id} "
        f"sensor-mode={sensor_mode} "
        f"ee-mode={ee_mode} "
        f"ee-strength={ee_strength:.3f} ! "
        f"video/x-raw(memory:NVMM), "
        f"width=(int){sensor_width}, "
        f"height=(int){sensor_height}, "
        f"format=(string)NV12, "
        f"framerate=(fraction){fps}/1 ! "
        f"nvvidconv interpolation-method={interpolation_method} ! "
        f"video/x-raw, "
        f"width=(int){output_width}, "
        f"height=(int){output_height}, "
        f"format=(string)BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=(string)BGR ! "
        f"appsink drop=true max-buffers=1 sync=false"
    )


def load_camera_info_from_yaml(
    path: str,
    frame_id: str,
    stamp,
) -> CameraInfo:
    with open(path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    message = CameraInfo()
    message.header.stamp = stamp
    message.header.frame_id = frame_id
    message.width = int(data["image_width"])
    message.height = int(data["image_height"])
    message.distortion_model = data["distortion_model"]
    message.d = [
        float(value)
        for value in data["distortion_coefficients"]["data"]
    ]
    message.k = [
        float(value)
        for value in data["camera_matrix"]["data"]
    ]
    message.r = [
        float(value)
        for value in data["rectification_matrix"]["data"]
    ]
    message.p = [
        float(value)
        for value in data["projection_matrix"]["data"]
    ]

    return message


class StereoCameraNode(Node):
    def __init__(self):
        super().__init__("stereo_camera_node")

        self.declare_parameter("left_sensor_id", 0)
        self.declare_parameter("right_sensor_id", 1)

        self.declare_parameter("sensor_mode", 4)
        self.declare_parameter("sensor_width", 1280)
        self.declare_parameter("sensor_height", 720)

        self.declare_parameter("width", 1280)
        self.declare_parameter("height", 720)
        self.declare_parameter("fps", 10)

        self.declare_parameter("interpolation_method", 5)
        self.declare_parameter("ee_mode", 1)
        self.declare_parameter("ee_strength", -1.0)

        self.declare_parameter("publish_camera_info", True)

        self.declare_parameter(
            "left_frame_id",
            "camera_left_optical_frame",
        )
        self.declare_parameter(
            "right_frame_id",
            "camera_right_optical_frame",
        )
        self.declare_parameter(
            "left_camera_info_yaml",
            "/home/warxen/ai_robot/calib/stereo/left.yaml",
        )
        self.declare_parameter(
            "right_camera_info_yaml",
            "/home/warxen/ai_robot/calib/stereo/right.yaml",
        )

        self.left_sensor_id = int(
            self.get_parameter("left_sensor_id").value
        )
        self.right_sensor_id = int(
            self.get_parameter("right_sensor_id").value
        )

        self.sensor_mode = int(
            self.get_parameter("sensor_mode").value
        )
        self.sensor_width = int(
            self.get_parameter("sensor_width").value
        )
        self.sensor_height = int(
            self.get_parameter("sensor_height").value
        )

        self.width = int(self.get_parameter("width").value)
        self.height = int(self.get_parameter("height").value)
        self.fps = int(self.get_parameter("fps").value)

        self.interpolation_method = int(
            self.get_parameter("interpolation_method").value
        )
        self.ee_mode = int(
            self.get_parameter("ee_mode").value
        )
        self.ee_strength = float(
            self.get_parameter("ee_strength").value
        )

        self.publish_camera_info = bool(
            self.get_parameter("publish_camera_info").value
        )

        self.left_frame_id = str(
            self.get_parameter("left_frame_id").value
        )
        self.right_frame_id = str(
            self.get_parameter("right_frame_id").value
        )
        self.left_camera_info_yaml = str(
            self.get_parameter("left_camera_info_yaml").value
        )
        self.right_camera_info_yaml = str(
            self.get_parameter("right_camera_info_yaml").value
        )

        self.bridge = CvBridge()

        self.left_pub = self.create_publisher(
            Image,
            "/camera/left/image_raw",
            10,
        )
        self.right_pub = self.create_publisher(
            Image,
            "/camera/right/image_raw",
            10,
        )
        self.left_info_pub = self.create_publisher(
            CameraInfo,
            "/camera/left/camera_info",
            10,
        )
        self.right_info_pub = self.create_publisher(
            CameraInfo,
            "/camera/right/camera_info",
            10,
        )

        left_pipeline = gstreamer_pipeline(
            self.left_sensor_id,
            self.sensor_mode,
            self.sensor_width,
            self.sensor_height,
            self.width,
            self.height,
            self.fps,
            self.interpolation_method,
            self.ee_mode,
            self.ee_strength,
        )

        right_pipeline = gstreamer_pipeline(
            self.right_sensor_id,
            self.sensor_mode,
            self.sensor_width,
            self.sensor_height,
            self.width,
            self.height,
            self.fps,
            self.interpolation_method,
            self.ee_mode,
            self.ee_strength,
        )

        self.left_cap = cv2.VideoCapture(
            left_pipeline,
            cv2.CAP_GSTREAMER,
        )
        self.right_cap = cv2.VideoCapture(
            right_pipeline,
            cv2.CAP_GSTREAMER,
        )

        if not self.left_cap.isOpened():
            raise RuntimeError("Failed to open left camera")

        if not self.right_cap.isOpened():
            self.left_cap.release()
            raise RuntimeError("Failed to open right camera")

        self.timer = self.create_timer(
            1.0 / float(self.fps),
            self.timer_cb,
        )

        self.get_logger().info(
            "stereo_camera_node started: "
            f"left={self.left_sensor_id}, "
            f"right={self.right_sensor_id}, "
            f"sensor_mode={self.sensor_mode}, "
            f"sensor={self.sensor_width}x{self.sensor_height}, "
            f"output={self.width}x{self.height}@{self.fps}, "
            f"interpolation={self.interpolation_method}, "
            f"ee={self.ee_mode}/{self.ee_strength:.3f}, "
            f"camera_info={self.publish_camera_info}"
        )

    def timer_cb(self):
        ok_left_grab = self.left_cap.grab()
        ok_right_grab = self.right_cap.grab()

        if not ok_left_grab:
            self.get_logger().warning("Left camera grab failed")
            return

        if not ok_right_grab:
            self.get_logger().warning("Right camera grab failed")
            return

        ok_left, left_frame = self.left_cap.retrieve()
        ok_right, right_frame = self.right_cap.retrieve()

        if not ok_left or left_frame is None:
            self.get_logger().warning("Left camera retrieve failed")
            return

        if not ok_right or right_frame is None:
            self.get_logger().warning("Right camera retrieve failed")
            return

        stamp = self.get_clock().now().to_msg()

        left_message = self.bridge.cv2_to_imgmsg(
            left_frame,
            encoding="bgr8",
        )
        left_message.header.stamp = stamp
        left_message.header.frame_id = self.left_frame_id

        right_message = self.bridge.cv2_to_imgmsg(
            right_frame,
            encoding="bgr8",
        )
        right_message.header.stamp = stamp
        right_message.header.frame_id = self.right_frame_id

        self.left_pub.publish(left_message)
        self.right_pub.publish(right_message)

        if self.publish_camera_info:
            self.left_info_pub.publish(
                load_camera_info_from_yaml(
                    self.left_camera_info_yaml,
                    self.left_frame_id,
                    stamp,
                )
            )
            self.right_info_pub.publish(
                load_camera_info_from_yaml(
                    self.right_camera_info_yaml,
                    self.right_frame_id,
                    stamp,
                )
            )

    def destroy_node(self):
        try:
            if self.left_cap is not None:
                self.left_cap.release()

            if self.right_cap is not None:
                self.right_cap.release()
        finally:
            super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = StereoCameraNode()

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
