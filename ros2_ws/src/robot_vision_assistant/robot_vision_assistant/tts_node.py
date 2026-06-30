#!/usr/bin/env python3
import os
import queue
import subprocess
import tempfile
import threading
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class TtsNode(Node):
    def __init__(self):
        super().__init__("tts_node")
        project_root = Path(
            os.environ.get(
                "AI_ROBOT_ROOT",
                str(Path.home() / "ai_robot"),
            )
        ).expanduser()

        self.declare_parameter("answer_topic", "/vision_assistant/answer")
        self.declare_parameter("status_topic", "/voice_tts/status")
        self.declare_parameter("enabled", True)
        self.declare_parameter(
            "piper_bin",
            str(project_root / "data" / "tts" / "piper" / "piper" / "piper"),
        )
        self.declare_parameter(
            "model_path",
            str(project_root / "data" / "tts" / "piper" / "ru_RU-ruslan-medium.onnx"),
        )
        self.declare_parameter("audio_player", "aplay")
        self.declare_parameter("audio_player_args", ["-q"])
        self.declare_parameter("tmp_dir", "/tmp/robot_tts")

        self.answer_topic = str(self.get_parameter("answer_topic").value)
        self.status_topic = str(self.get_parameter("status_topic").value)
        self.enabled = bool(self.get_parameter("enabled").value)
        self.piper_bin = str(self.get_parameter("piper_bin").value)
        self.model_path = str(self.get_parameter("model_path").value)
        self.audio_player = str(self.get_parameter("audio_player").value)
        self.audio_player_args = list(self.get_parameter("audio_player_args").value)
        self.tmp_dir = Path(str(self.get_parameter("tmp_dir").value))
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

        self.q = queue.Queue()
        self.worker = threading.Thread(target=self.worker_loop, daemon=True)
        self.worker.start()

        self.sub = self.create_subscription(
            String, self.answer_topic, self.answer_cb, 10
        )

        self.status_pub = self.create_publisher(String, self.status_topic, 10)

        self.get_logger().info(
            f"tts_node started: topic={self.answer_topic}, status_topic={self.status_topic}, enabled={self.enabled}"
        )

    def publish_status(self, text: str):
        msg = String()
        msg.data = text
        self.status_pub.publish(msg)

    def answer_cb(self, msg: String):
        if not self.enabled:
            return

        text = (msg.data or "").strip()
        if not text:
            return

        self.q.put(text)

    def worker_loop(self):
        while True:
            text = self.q.get()
            try:
                self.speak(text)
            except Exception as e:
                self.get_logger().warning(f"TTS failed: {e}")
                self.publish_status("speaking_done")

    def speak(self, text: str):
        if not os.path.exists(self.piper_bin):
            raise RuntimeError(f"piper binary not found: {self.piper_bin}")
        if not os.path.exists(self.model_path):
            raise RuntimeError(f"piper model not found: {self.model_path}")

        safe_text = text.replace("\n", " ").strip()
        if not safe_text:
            return

        with tempfile.NamedTemporaryFile(
            suffix=".wav", dir=self.tmp_dir, delete=False
        ) as f:
            wav_path = f.name

        self.publish_status("speaking_start")

        try:
            subprocess.run(
                [
                    self.piper_bin,
                    "--model", self.model_path,
                    "--output_file", wav_path,
                ],
                input=safe_text.encode("utf-8"),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )

            subprocess.run(
                [self.audio_player, *self.audio_player_args, wav_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )

        finally:
            self.publish_status("speaking_done")
            try:
                os.remove(wav_path)
            except Exception:
                pass


def main(args=None):
    rclpy.init(args=args)
    node = TtsNode()
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
