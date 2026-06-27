#!/usr/bin/env python3
from enum import Enum

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class VoiceState(str, Enum):
    IDLE = "IDLE"
    LISTENING = "LISTENING"
    THINKING = "THINKING"
    SPEAKING = "SPEAKING"


class VoiceManagerNode(Node):
    def __init__(self):
        super().__init__("voice_manager_node")

        self.declare_parameter("start_topic", "/voice/start")
        self.declare_parameter("state_topic", "/voice/state")
        self.declare_parameter("asr_listen_topic", "/voice_asr/listen")
        self.declare_parameter("asr_status_topic", "/voice_asr/status")
        self.declare_parameter("asr_transcript_topic", "/voice_asr/transcript")
        self.declare_parameter("answer_topic", "/vision_assistant/answer")
        self.declare_parameter("tts_status_topic", "/voice_tts/status")

        self.start_topic = str(self.get_parameter("start_topic").value)
        self.state_topic = str(self.get_parameter("state_topic").value)
        self.asr_listen_topic = str(self.get_parameter("asr_listen_topic").value)
        self.asr_status_topic = str(self.get_parameter("asr_status_topic").value)
        self.asr_transcript_topic = str(self.get_parameter("asr_transcript_topic").value)
        self.answer_topic = str(self.get_parameter("answer_topic").value)
        self.tts_status_topic = str(self.get_parameter("tts_status_topic").value)

        self.state = VoiceState.IDLE
        self.last_transcript = ""
        self.last_answer = ""

        self.state_pub = self.create_publisher(String, self.state_topic, 10)
        self.asr_listen_pub = self.create_publisher(String, self.asr_listen_topic, 10)

        self.start_sub = self.create_subscription(
            String, self.start_topic, self.start_cb, 10
        )
        self.asr_status_sub = self.create_subscription(
            String, self.asr_status_topic, self.asr_status_cb, 10
        )
        self.asr_transcript_sub = self.create_subscription(
            String, self.asr_transcript_topic, self.asr_transcript_cb, 10
        )
        self.answer_sub = self.create_subscription(
            String, self.answer_topic, self.answer_cb, 10
        )
        self.tts_status_sub = self.create_subscription(
            String, self.tts_status_topic, self.tts_status_cb, 10
        )

        self.set_state(VoiceState.IDLE)
        self.get_logger().info("voice_manager_node started")

    def set_state(self, new_state: VoiceState):
        self.state = new_state
        msg = String()
        msg.data = str(new_state.value)
        self.state_pub.publish(msg)
        self.get_logger().info(f"VOICE STATE -> {new_state.value}")

    def start_cb(self, msg: String):
        if self.state != VoiceState.IDLE:
            self.get_logger().info(f"start ignored, current state={self.state.value}")
            return

        self.last_transcript = ""
        self.last_answer = ""
        self.set_state(VoiceState.LISTENING)

        listen_msg = String()
        listen_msg.data = "listen"
        self.asr_listen_pub.publish(listen_msg)

    def asr_status_cb(self, msg: String):
        text = (msg.data or "").strip()

        if self.state == VoiceState.LISTENING:
            if "empty" in text.lower() or "invalid" in text.lower():
                self.get_logger().info("ASR returned empty/invalid transcript")
                self.set_state(VoiceState.IDLE)

    def asr_transcript_cb(self, msg: String):
        text = (msg.data or "").strip()
        if not text:
            return

        self.last_transcript = text

        if self.state == VoiceState.LISTENING:
            self.set_state(VoiceState.THINKING)

    def answer_cb(self, msg: String):
        text = (msg.data or "").strip()
        if not text:
            return

        self.last_answer = text

        if self.state == VoiceState.THINKING:
            # Ждём, что TTS начнёт говорить и сам переведёт нас в SPEAKING через /voice_tts/status
            self.get_logger().info(f"answer received: {text}")

    def tts_status_cb(self, msg: String):
        text = (msg.data or "").strip().lower()

        if text == "speaking_start":
            self.set_state(VoiceState.SPEAKING)
            return

        if text == "speaking_done":
            self.set_state(VoiceState.IDLE)
            return


def main(args=None):
    rclpy.init(args=args)
    node = VoiceManagerNode()
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
