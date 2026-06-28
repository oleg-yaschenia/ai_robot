#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class VoiceLedBridgeNode(Node):
    def __init__(self):
        super().__init__("voice_led_bridge_node")

        self.declare_parameter("voice_state_topic", "/voice/state")
        self.declare_parameter("head_events_topic", "/robot/head/events")
        self.declare_parameter("debug_topic", "/voice_led_bridge/debug")

        self.voice_state_topic = str(self.get_parameter("voice_state_topic").value)
        self.head_events_topic = str(self.get_parameter("head_events_topic").value)
        self.debug_topic = str(self.get_parameter("debug_topic").value)

        self.events_pub = self.create_publisher(String, self.head_events_topic, 10)
        self.debug_pub = self.create_publisher(String, self.debug_topic, 10)

        self.state_sub = self.create_subscription(
            String, self.voice_state_topic, self.state_cb, 10
        )

        self.last_state = None

        self.get_logger().info(
            f"voice_led_bridge_node started: voice_state_topic={self.voice_state_topic}, "
            f"head_events_topic={self.head_events_topic}"
        )

    def publish_event(self, text: str):
        msg = String()
        msg.data = text
        self.events_pub.publish(msg)

        dbg = String()
        dbg.data = text
        self.debug_pub.publish(dbg)

        self.get_logger().info(f"head_event -> {text}")

    def stop_event_for_state(self, state: str):
        if state == "LISTENING":
            return "listening_stopped"
        if state == "THINKING":
            return "thinking_stopped"
        if state == "SPEAKING":
            return "speaking_finished"
        return None

    def start_event_for_state(self, state: str):
        if state == "LISTENING":
            return "listening_started"
        if state == "THINKING":
            return "thinking_started"
        if state == "SPEAKING":
            return "speaking_started"
        return None

    def state_cb(self, msg: String):
        state = (msg.data or "").strip().upper()
        if not state:
            return

        if state not in ("IDLE", "LISTENING", "THINKING", "SPEAKING"):
            self.get_logger().warning(f"unknown voice state: {state}")
            return

        if state == self.last_state:
            return

        if self.last_state is not None:
            stop_evt = self.stop_event_for_state(self.last_state)
            if stop_evt:
                self.publish_event(stop_evt)

        start_evt = self.start_event_for_state(state)
        if start_evt:
            self.publish_event(start_evt)

        self.last_state = state


def main(args=None):
    rclpy.init(args=args)
    node = VoiceLedBridgeNode()
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
