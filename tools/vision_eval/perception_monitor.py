#!/usr/bin/env python3
import json

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class PerceptionMonitor(Node):
    def __init__(self):
        super().__init__("perception_monitor")
        self.create_subscription(
            String,
            "/perception/state_json",
            self.callback,
            10,
        )
        self.get_logger().info(
            "Monitoring /perception/state_json. Press Ctrl+C to stop."
        )

    def callback(self, msg: String):
        try:
            state = json.loads(msg.data)
        except Exception as exc:
            print(f"JSON ERROR: {exc}")
            return

        detector = state.get("detector", {})
        counts = state.get("counts", {})
        detections = state.get("detections", [])

        print(
            f"latency={detector.get('latency_ms')} ms | "
            f"tracks={detector.get('active_tracks')} | "
            f"counts={counts}"
        )

        for det in detections:
            print(
                "  "
                f"{det.get('class_name')}#"
                f"{det.get('track_id')} "
                f"conf={det.get('confidence')} "
                f"hits={det.get('hits')} "
                f"miss={det.get('missed_frames')} "
f"bbox={det.get('bbox_xyxy')}"
            )

        print("-" * 80)


def main():
    rclpy.init()
    node = PerceptionMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
