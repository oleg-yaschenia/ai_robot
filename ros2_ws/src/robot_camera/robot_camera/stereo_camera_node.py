#!/usr/bin/env python3

import time

import cv2
import yaml
import rclpy

from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
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
) -> CameraInfo:
    """Load immutable calibration data once during node startup."""
    with open(path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    message = CameraInfo()
    message.header.frame_id = frame_id
    message.width = int(data["image_width"])
    message.height = int(data["image_height"])
    message.distortion_model = str(data["distortion_model"])
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
        self.declare_parameter("diagnostics_period_sec", 5.0)

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
        self.diagnostics_period_sec = max(
            1.0,
            float(
                self.get_parameter("diagnostics_period_sec").value
            ),
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

        # Old video frames are useless for live perception. Never build a
        # reliable queue of multi-megabyte Image messages.
        image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        info_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.left_pub = self.create_publisher(
            Image,
            "/camera/left/image_raw",
            image_qos,
        )
        self.right_pub = self.create_publisher(
            Image,
            "/camera/right/image_raw",
            image_qos,
        )
        self.left_info_pub = self.create_publisher(
            CameraInfo,
            "/camera/left/camera_info",
            info_qos,
        )
        self.right_info_pub = self.create_publisher(
            CameraInfo,
            "/camera/right/camera_info",
            info_qos,
        )

        self.left_camera_info = None
        self.right_camera_info = None
        if self.publish_camera_info:
            self.left_camera_info = load_camera_info_from_yaml(
                self.left_camera_info_yaml,
                self.left_frame_id,
            )
            self.right_camera_info = load_camera_info_from_yaml(
                self.right_camera_info_yaml,
                self.right_frame_id,
            )
            self._validate_camera_info(
                self.left_camera_info,
                "left",
            )
            self._validate_camera_info(
                self.right_camera_info,
                "right",
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

        self._diag_reset(time.perf_counter())
        self._last_callback_start = None
        self._expected_period_sec = 1.0 / float(self.fps)

        self.timer = self.create_timer(
            self._expected_period_sec,
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
            f"camera_info={self.publish_camera_info}, "
            "image_qos=BEST_EFFORT/depth1"
        )

    def _validate_camera_info(
        self,
        message: CameraInfo,
        side: str,
    ) -> None:
        if (
            int(message.width) != self.width
            or int(message.height) != self.height
        ):
            self.get_logger().warning(
                f"{side} calibration is "
                f"{message.width}x{message.height}, but output is "
                f"{self.width}x{self.height}"
            )

    def _diag_reset(self, now: float) -> None:
        self._diag_started_at = now
        self._diag_frames = 0
        self._diag_failures = 0
        self._diag_missed_periods = 0
        self._diag_total_sum_ms = 0.0
        self._diag_total_max_ms = 0.0
        self._diag_stage_sum_ms = [0.0] * 7
        self._diag_stage_max_ms = [0.0] * 7

    def _diag_record(
        self,
        callback_start: float,
        stage_times,
        success: bool,
    ) -> None:
        now = time.perf_counter()

        if self._last_callback_start is not None:
            callback_gap = callback_start - self._last_callback_start
            if callback_gap > self._expected_period_sec * 1.5:
                missed = max(
                    1,
                    int(round(callback_gap / self._expected_period_sec)) - 1,
                )
                self._diag_missed_periods += missed
        self._last_callback_start = callback_start

        if success:
            self._diag_frames += 1
        else:
            self._diag_failures += 1

        stage_ms = [
            (stage_times[index + 1] - stage_times[index]) * 1000.0
            for index in range(len(stage_times) - 1)
        ]
        total_ms = (stage_times[-1] - stage_times[0]) * 1000.0

        self._diag_total_sum_ms += total_ms
        self._diag_total_max_ms = max(
            self._diag_total_max_ms,
            total_ms,
        )

        for index, value in enumerate(stage_ms):
            self._diag_stage_sum_ms[index] += value
            self._diag_stage_max_ms[index] = max(
                self._diag_stage_max_ms[index],
                value,
            )

        elapsed = now - self._diag_started_at
        if elapsed < self.diagnostics_period_sec:
            return

        attempts = self._diag_frames + self._diag_failures
        divisor = max(1, attempts)
        average_total = self._diag_total_sum_ms / divisor
        average_stages = [
            value / divisor
            for value in self._diag_stage_sum_ms
        ]
        output_hz = self._diag_frames / max(elapsed, 1e-6)

        names = (
            "grabL",
            "grabR",
            "retrieveL",
            "retrieveR",
            "convert",
            "publishImage",
            "publishInfo",
        )
        stage_text = " ".join(
            f"{name}={average_stages[index]:.1f}/"
            f"{self._diag_stage_max_ms[index]:.1f}ms"
            for index, name in enumerate(names)
        )

        self.get_logger().info(
            "camera_perf "
            f"out={output_hz:.2f}Hz "
            f"frames={self._diag_frames} "
            f"fail={self._diag_failures} "
            f"missed={self._diag_missed_periods} "
            f"total={average_total:.1f}/"
            f"{self._diag_total_max_ms:.1f}ms "
            f"{stage_text}"
        )

        self._diag_reset(now)

    def timer_cb(self):
        callback_start = time.perf_counter()
        stage_times = [callback_start]

        ok_left_grab = self.left_cap.grab()
        stage_times.append(time.perf_counter())

        ok_right_grab = self.right_cap.grab()
        stage_times.append(time.perf_counter())

        if not ok_left_grab:
            self.get_logger().warning("Left camera grab failed")
            while len(stage_times) < 8:
                stage_times.append(stage_times[-1])
            self._diag_record(callback_start, stage_times, False)
            return

        if not ok_right_grab:
            self.get_logger().warning("Right camera grab failed")
            while len(stage_times) < 8:
                stage_times.append(stage_times[-1])
            self._diag_record(callback_start, stage_times, False)
            return

        ok_left, left_frame = self.left_cap.retrieve()
        stage_times.append(time.perf_counter())

        ok_right, right_frame = self.right_cap.retrieve()
        stage_times.append(time.perf_counter())

        if not ok_left or left_frame is None:
            self.get_logger().warning("Left camera retrieve failed")
            while len(stage_times) < 8:
                stage_times.append(stage_times[-1])
            self._diag_record(callback_start, stage_times, False)
            return

        if not ok_right or right_frame is None:
            self.get_logger().warning("Right camera retrieve failed")
            while len(stage_times) < 8:
                stage_times.append(stage_times[-1])
            self._diag_record(callback_start, stage_times, False)
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
        stage_times.append(time.perf_counter())

        self.left_pub.publish(left_message)
        self.right_pub.publish(right_message)
        stage_times.append(time.perf_counter())

        if self.publish_camera_info:
            self.left_camera_info.header.stamp = stamp
            self.right_camera_info.header.stamp = stamp
            self.left_info_pub.publish(self.left_camera_info)
            self.right_info_pub.publish(self.right_camera_info)
        stage_times.append(time.perf_counter())

        self._diag_record(callback_start, stage_times, True)

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
