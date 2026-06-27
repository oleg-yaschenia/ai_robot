#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class VoiceLedBridgeNode(Node):
    def __init__(self):
        super().__init__("voice_led_bridge_node")

        self.declare_parameter("voice_state_topic", "/voice/state")
        self.declare_parameter("head_mode_topic", "/head_mode")
        self.declare_parameter("debug_topic", "/voice_led_bridge/debug")

        self.declare_parameter("idle_mode", "IDLE")
        self.declare_parameter("listening_mode", "LISTENING")
        self.declare_parameter("thinking_mode", "THINKING")
        self.declare_parameter("speaking_mode", "SPEAKING")

        self.voice_state_topic = str(self.get_parameter("voice_state_topic").value)
        self.head_mode_topic = str(self.get_parameter("head_mode_topic").value)
        self.debug_topic = str(self.get_parameter("debug_topic").value)

        self.idle_mode = str(self.get_parameter("idle_mode").value)
        self.listening_mode = str(self.get_parameter("listening_mode").value)
        self.thinking_mode = str(self.get_parameter("thinking_mode").value)
        self.speaking_mode = str(self.get_parameter("speaking_mode").value)

        self.mode_pub = self.create_publisher(String, self.head_mode_topic, 10)
        self.debug_pub = self.create_publisher(String, self.debug_topic, 10)

        self.state_sub = self.create_subscription(
            String, self.voice_state_topic, self.state_cb, 10
        )

        self.last_sent_mode = None

        self.get_logger().info(
            f"voice_led_bridge_node started: voice_state_topic={self.voice_state_topic}, "
            f"head_mode_topic={self.head_mode_topic}"
        )

    def publish_debug(self, text: str):
        msg = String()
        msg.data = text
        self.debug_pub.publish(msg)

    def map_state_to_mode(self, state: str):
        state = state.strip().upper()

        if state == "IDLE":
            return self.idle_mode
        if state == "LISTENING":
            return self.listening_mode
        if state == "THINKING":
            return self.thinking_mode
        if state == "SPEAKING":
            return self.speaking_mode

        return None

    def state_cb(self, msg: String):
        state = (msg.data or "").strip()
        if not state:
            return

        mode = self.map_state_to_mode(state)
        if mode is None:
            self.get_logger().warning(f"unknown voice state: {state}")
            return

        if mode == self.last_sent_mode:
            return

        out = String()
        out.data = mode
        self.mode_pub.publish(out)
        self.last_sent_mode = mode

        debug_text = f"voice_state={state} -> head_mode={mode}"
        self.publish_debug(debug_text)
        self.get_logger().info(debug_text)


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
