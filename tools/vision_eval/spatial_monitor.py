#!/usr/bin/env python3
import json

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class SpatialMonitor(Node):
    def __init__(self):
        super().__init__("spatial_monitor")
        self.create_subscription(
            String,
            "/perception/state_json",
            self.callback,
            10,
        )
        self.get_logger().info(
            "Monitoring spatial perception. Press Ctrl+C to stop."
        )

    def callback(self, msg):
        try:
            state = json.loads(msg.data)
        except Exception as exc:
            print(f"JSON error: {exc}")
            return

        detector = state.get("detector", {})
        target = state.get("primary_target", {})
        nearest = state.get("nearest_hint", {})

        print(
            f"latency={detector.get('latency_ms')} ms | "
            f"target={target.get('class_name')}#"
            f"{target.get('track_id')} "
            f"{target.get('side')}/"
            f"{target.get('proximity_hint')} | "
            f"nearest_hint={nearest.get('class_name')}#"
            f"{nearest.get('track_id')}"
        )

        for detection in state.get("detections", []):
            spatial = detection.get("spatial", {})
            print(
                f"  {detection.get('class_name')}#"
                f"{detection.get('track_id')} "
                f"conf={detection.get('confidence')} "
                f"side={detection.get('side')} "
                f"vertical={detection.get('vertical_region')} "
                f"proximity={detection.get('proximity_hint')} "
                f"offset_x={spatial.get('offset_x_norm')} "
                f"area={spatial.get('area_ratio')} "
                f"miss={detection.get('missed_frames')}"
            )
        print("-" * 100)


def main():
    rclpy.init()
    node = SpatialMonitor()
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
