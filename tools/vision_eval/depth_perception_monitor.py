#!/usr/bin/env python3
import json

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class DepthPerceptionMonitor(Node):
    def __init__(self):
        super().__init__("depth_perception_monitor")
        self.create_subscription(
            String,
            "/perception/state_json",
            self.callback,
            10,
        )
        self.get_logger().info(
            "Monitoring YOLO + stereo depth. Press Ctrl+C to stop."
        )

    def callback(self, msg):
        try:
            state = json.loads(msg.data)
        except Exception as exc:
            print(f"JSON error: {exc}")
            return

        detector = state.get("detector", {})
        spatial = state.get("spatial_estimation", {})
        target = state.get("primary_target", {})
        nearest = state.get("nearest_hint", {})

        print(
            f"latency={detector.get('latency_ms')}ms | "
            f"depth_frame={spatial.get('depth_frame_available')} "
            f"age={spatial.get('depth_age_ms')}ms | "
            f"metric={spatial.get('metric_detection_count')} | "
            f"target={target.get('class_name')}#"
            f"{target.get('track_id')} "
            f"distance={target.get('distance_m')}m | "
            f"nearest={nearest.get('class_name')}#"
            f"{nearest.get('track_id')} "
            f"distance={nearest.get('distance_m')}m"
        )

        for detection in state.get("detections", []):
            print(
                f"  {detection.get('class_name')}#"
                f"{detection.get('track_id')} "
                f"conf={detection.get('confidence')} "
                f"side={detection.get('side')} "
                f"distance={detection.get('distance_m')}m "
                f"valid={detection.get('distance_valid')} "
                f"quality={detection.get('depth_confidence')} "
                f"status={detection.get('depth_status')} "
                f"ratio={detection.get('depth_valid_ratio')} "
                f"samples={detection.get('depth_samples')} "
                f"p10-p90={detection.get('depth_p10_m')}.."
                f"{detection.get('depth_p90_m')}m"
            )
        print("-" * 120)


def main():
    rclpy.init()
    node = DepthPerceptionMonitor()
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
