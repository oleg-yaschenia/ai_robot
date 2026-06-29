#!/usr/bin/env python3
import math
import statistics
from collections import deque

import numpy as np
import rclpy
from rclpy.node import Node
from stereo_msgs.msg import DisparityImage


class DisparityDepthCheck(Node):
    def __init__(self):
        super().__init__("disparity_depth_check")
        self.declare_parameter("topic", "/disparity")
        self.declare_parameter("roi_fraction", 0.12)
        self.declare_parameter("min_depth_m", 0.15)
        self.declare_parameter("max_depth_m", 10.0)

        topic = str(self.get_parameter("topic").value)
        self.roi_fraction = float(self.get_parameter("roi_fraction").value)
        self.min_depth_m = float(self.get_parameter("min_depth_m").value)
        self.max_depth_m = float(self.get_parameter("max_depth_m").value)

        self.last_stamp = None
        self.periods = deque(maxlen=30)

        self.create_subscription(
            DisparityImage,
            topic,
            self.callback,
            10,
        )

        self.get_logger().info(
            f"Listening to {topic}. Aim the image centre at a flat object."
        )

    @staticmethod
    def image_to_array(msg: DisparityImage) -> np.ndarray:
        image = msg.image

        if image.encoding not in ("32FC1", "32FC"):
            raise ValueError(
                f"Expected 32FC1 disparity image, got {image.encoding}"
            )

        if image.step % 4 != 0:
            raise ValueError(f"Invalid image step: {image.step}")

        floats_per_row = image.step // 4
        raw = np.frombuffer(image.data, dtype=np.float32)
        raw = raw.reshape((image.height, floats_per_row))
        return raw[:, : image.width]

    def callback(self, msg: DisparityImage):
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.last_stamp is not None and stamp > self.last_stamp:
            self.periods.append(stamp - self.last_stamp)
        self.last_stamp = stamp

        try:
            disparity = self.image_to_array(msg)
        except Exception as exc:
            self.get_logger().error(str(exc))
            return

        height, width = disparity.shape
        half_w = max(3, int(width * self.roi_fraction / 2.0))
        half_h = max(3, int(height * self.roi_fraction / 2.0))
        cx = width // 2
        cy = height // 2

        x1 = max(0, cx - half_w)
        x2 = min(width, cx + half_w)
        y1 = max(0, cy - half_h)
        y2 = min(height, cy + half_h)

        roi = disparity[y1:y2, x1:x2]

        min_disp = float(msg.min_disparity)
        valid = np.isfinite(roi) & (roi > max(min_disp, 0.0))

        total = roi.size
        valid_count = int(valid.sum())
        valid_ratio = valid_count / total if total else 0.0

        baseline_m = float(msg.t)
        focal_px = float(msg.f)

        if valid_count == 0 or baseline_m <= 0.0 or focal_px <= 0.0:
            print(
                f"f={focal_px:.3f}px baseline={baseline_m:.5f}m "
                f"disp=[{msg.min_disparity:.3f}, {msg.max_disparity:.3f}] "
                f"valid={valid_ratio:.1%} depth=INVALID"
            )
            return

        disparities = roi[valid]
        depths = focal_px * baseline_m / disparities
        depths = depths[
            np.isfinite(depths)
            & (depths >= self.min_depth_m)
            & (depths <= self.max_depth_m)
        ]

        if depths.size == 0:
            print(
                f"f={focal_px:.3f}px baseline={baseline_m:.5f}m "
                f"valid={valid_ratio:.1%} depth=OUT_OF_RANGE"
            )
            return

        median_depth = float(np.median(depths))
        p10 = float(np.percentile(depths, 10))
        p90 = float(np.percentile(depths, 90))
        median_disp = float(np.median(disparities))

        hz = 0.0
        if self.periods:
            mean_period = statistics.mean(self.periods)
            hz = 1.0 / mean_period if mean_period > 0 else 0.0

        print(
            f"rate={hz:5.2f}Hz "
            f"f={focal_px:8.3f}px "
            f"baseline={baseline_m:.5f}m "
            f"disp_med={median_disp:7.3f}px "
            f"valid={valid_ratio:6.1%} "
            f"depth_med={median_depth:5.2f}m "
            f"depth_p10-p90={p10:5.2f}..{p90:5.2f}m"
        )


def main():
    rclpy.init()
    node = DisparityDepthCheck()
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
