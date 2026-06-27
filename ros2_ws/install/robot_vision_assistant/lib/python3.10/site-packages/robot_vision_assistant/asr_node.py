#!/usr/bin/env python3
import os
import re
import subprocess
import tempfile
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class AsrNode(Node):
    def __init__(self):
        super().__init__("asr_node")

        self.declare_parameter("listen_topic", "/voice_asr/listen")
        self.declare_parameter("query_topic", "/vision_assistant/query")
        self.declare_parameter("transcript_topic", "/voice_asr/transcript")
        self.declare_parameter("status_topic", "/voice_asr/status")

        self.declare_parameter("record_device", "hw:1,0")
        self.declare_parameter("record_seconds", 3)
        self.declare_parameter("record_sample_rate", 48000)
        self.declare_parameter("record_channels", 2)
        self.declare_parameter("record_format", "S32_LE")

        self.declare_parameter(
            "whisper_cli",
            "/home/warxen/ai_robot/tools/whisper.cpp/build/bin/whisper-cli",
        )
        self.declare_parameter(
            "model_path",
            "/home/warxen/ai_robot/tools/whisper.cpp/models/ggml-tiny.bin",
        )
        self.declare_parameter("language", "ru")
        self.declare_parameter("threads", 4)
        self.declare_parameter("processors", 1)
        self.declare_parameter("tmp_dir", "/tmp/robot_asr")

        self.listen_topic = str(self.get_parameter("listen_topic").value)
        self.query_topic = str(self.get_parameter("query_topic").value)
        self.transcript_topic = str(self.get_parameter("transcript_topic").value)
        self.status_topic = str(self.get_parameter("status_topic").value)

        self.record_device = str(self.get_parameter("record_device").value)
        self.record_seconds = int(self.get_parameter("record_seconds").value)
        self.record_sample_rate = int(self.get_parameter("record_sample_rate").value)
        self.record_channels = int(self.get_parameter("record_channels").value)
        self.record_format = str(self.get_parameter("record_format").value)

        self.whisper_cli = str(self.get_parameter("whisper_cli").value)
        self.model_path = str(self.get_parameter("model_path").value)
        self.language = str(self.get_parameter("language").value)
        self.threads = int(self.get_parameter("threads").value)
        self.processors = int(self.get_parameter("processors").value)

        self.tmp_dir = Path(str(self.get_parameter("tmp_dir").value))
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

        self.query_pub = self.create_publisher(String, self.query_topic, 10)
        self.transcript_pub = self.create_publisher(String, self.transcript_topic, 10)
        self.status_pub = self.create_publisher(String, self.status_topic, 10)

        self.listen_sub = self.create_subscription(
            String, self.listen_topic, self.listen_cb, 10
        )

        self.publish_status(
            f"asr_node started: device={self.record_device}, model={self.model_path}"
        )
        self.get_logger().info(
            f"asr_node started: device={self.record_device}, model={self.model_path}"
        )

    def publish_status(self, text: str):
        msg = String()
        msg.data = text
        self.status_pub.publish(msg)

    def listen_cb(self, msg: String):
        self.publish_status("ASR listen requested")

        raw_wav_path = None
        wav_path = None
        try:
            raw_wav_path = self.record_audio()
            wav_path = self.prepare_audio_for_asr(raw_wav_path)
            text = self.transcribe_audio(wav_path)

            bad_tokens = {
                "[музыка]",
                "[смех]",
                "[аплодисменты]",
                "[music]",
                "[laughter]",
                "[applause]",
            }

            if not text or text.strip().lower() in bad_tokens:
                self.publish_status("ASR: empty or invalid transcript")
                return

            transcript_msg = String()
            transcript_msg.data = text
            self.transcript_pub.publish(transcript_msg)

            query_msg = String()
            query_msg.data = text
            self.query_pub.publish(query_msg)

            self.publish_status(f"ASR transcript published: {text}")
            self.get_logger().info(f"ASR transcript: {text}")

        except Exception as e:
            self.publish_status(f"ASR failed: {e}")
            self.get_logger().warning(f"ASR failed: {e}")
        finally:
            for path in [raw_wav_path, wav_path]:
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                    except Exception:
                        pass

    def record_audio(self) -> str:
        with tempfile.NamedTemporaryFile(
            suffix=".wav", dir=self.tmp_dir, delete=False
        ) as f:
            wav_path = f.name

        cmd = [
            "arecord",
            "-D", self.record_device,
            "-c", str(self.record_channels),
            "-r", str(self.record_sample_rate),
            "-f", self.record_format,
            "-d", str(self.record_seconds),
            wav_path,
        ]

        self.publish_status("ASR recording started")
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.publish_status(f"ASR recording finished: {wav_path}")
        return wav_path

    def prepare_audio_for_asr(self, raw_wav_path: str) -> str:
        with tempfile.NamedTemporaryFile(
            suffix=".wav", dir=self.tmp_dir, delete=False
        ) as f:
            mono_wav_path = f.name

        cmd = [
            "ffmpeg",
            "-y",
            "-i", raw_wav_path,
            "-ac", "1",
            "-ar", "16000",
            "-c:a", "pcm_s16le",
            mono_wav_path,
        ]

        self.publish_status("ASR audio conversion started")
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.publish_status(f"ASR audio conversion finished: {mono_wav_path}")
        return mono_wav_path

    def transcribe_audio(self, wav_path: str) -> str:
        if not os.path.exists(self.whisper_cli):
            raise RuntimeError(f"whisper-cli not found: {self.whisper_cli}")
        if not os.path.exists(self.model_path):
            raise RuntimeError(f"model not found: {self.model_path}")

        cmd = [
            self.whisper_cli,
            "-m", self.model_path,
            "-f", wav_path,
            "-l", self.language,
            "-t", str(self.threads),
            "-p", str(self.processors),
        ]

        self.publish_status("ASR whisper.cpp transcription started")
        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.publish_status("ASR whisper.cpp transcription finished")

        text = self.extract_text(result.stdout)
        self.get_logger().info(f"ASR raw stdout:\n{result.stdout}")
        return text

    def extract_text(self, stdout_text: str) -> str:
        lines = stdout_text.splitlines()
        text_parts = []

        # Пример строки:
        # [00:00:00.000 --> 00:00:01.500]   Привет
        ts_pattern = re.compile(r"^\[[0-9:\.\-\->\s]+\]\s*(.*)$")

        for line in lines:
            line = line.strip()
            if not line:
                continue

            m = ts_pattern.match(line)
            if m:
                part = m.group(1).strip()
                if part:
                    text_parts.append(part)

        if text_parts:
            return " ".join(text_parts).strip()

        # fallback: если формат другой, берём последние осмысленные строки
        fallback = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.startswith("whisper_") or line.startswith("system_info:") or line.startswith("main:"):
                continue
            fallback.append(line)

        return " ".join(fallback).strip()


def main(args=None):
    rclpy.init(args=args)
    node = AsrNode()
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
