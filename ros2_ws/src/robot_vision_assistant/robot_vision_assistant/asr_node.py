#!/usr/bin/env python3
import os
import re
import subprocess
import tempfile
import time
import wave
from collections import deque
from pathlib import Path

import numpy as np
import rclpy
import webrtcvad
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
        self.declare_parameter("record_sample_rate", 48000)
        self.declare_parameter("record_channels", 2)
        self.declare_parameter("record_format", "S32_LE")

        self.declare_parameter(
            "whisper_cli",
            "/home/warxen/ai_robot/tools/whisper.cpp/build/bin/whisper-cli",
        )
        self.declare_parameter(
            "model_path",
            "/home/warxen/ai_robot/tools/whisper.cpp/models/ggml-base.bin",
        )
        self.declare_parameter("language", "ru")
        self.declare_parameter("threads", 4)
        self.declare_parameter("processors", 1)
        self.declare_parameter("tmp_dir", "/tmp/robot_asr")

        # VAD config
        self.declare_parameter("vad_mode", 2)                 # 0..3
        self.declare_parameter("frame_ms", 30)                # 10, 20, 30
        self.declare_parameter("pre_roll_ms", 300)            # audio before trigger
        self.declare_parameter("speech_timeout_sec", 5.0)     # wait for speech start
        self.declare_parameter("max_utterance_sec", 8.0)      # hard limit
        self.declare_parameter("start_window_ms", 240)        # start detection window
        self.declare_parameter("start_trigger_ratio", 0.60)   # voiced ratio to start
        self.declare_parameter("end_window_ms", 900)          # end detection window
        self.declare_parameter("end_trigger_ratio", 0.90)     # unvoiced ratio to stop
        self.declare_parameter("min_utterance_ms", 400)       # ignore too-short utterances

        self.listen_topic = str(self.get_parameter("listen_topic").value)
        self.query_topic = str(self.get_parameter("query_topic").value)
        self.transcript_topic = str(self.get_parameter("transcript_topic").value)
        self.status_topic = str(self.get_parameter("status_topic").value)

        self.record_device = str(self.get_parameter("record_device").value)
        self.record_sample_rate = int(self.get_parameter("record_sample_rate").value)
        self.record_channels = int(self.get_parameter("record_channels").value)
        self.record_format = str(self.get_parameter("record_format").value)

        self.whisper_cli = str(self.get_parameter("whisper_cli").value)
        self.model_path = str(self.get_parameter("model_path").value)
        self.language = str(self.get_parameter("language").value)
        self.threads = int(self.get_parameter("threads").value)
        self.processors = int(self.get_parameter("processors").value)

        self.vad_mode = int(self.get_parameter("vad_mode").value)
        self.frame_ms = int(self.get_parameter("frame_ms").value)
        self.pre_roll_ms = int(self.get_parameter("pre_roll_ms").value)
        self.speech_timeout_sec = float(self.get_parameter("speech_timeout_sec").value)
        self.max_utterance_sec = float(self.get_parameter("max_utterance_sec").value)
        self.start_window_ms = int(self.get_parameter("start_window_ms").value)
        self.start_trigger_ratio = float(self.get_parameter("start_trigger_ratio").value)
        self.end_window_ms = int(self.get_parameter("end_window_ms").value)
        self.end_trigger_ratio = float(self.get_parameter("end_trigger_ratio").value)
        self.min_utterance_ms = int(self.get_parameter("min_utterance_ms").value)

        self.tmp_dir = Path(str(self.get_parameter("tmp_dir").value))
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

        self.query_pub = self.create_publisher(String, self.query_topic, 10)
        self.transcript_pub = self.create_publisher(String, self.transcript_topic, 10)
        self.status_pub = self.create_publisher(String, self.status_topic, 10)

        self.listen_sub = self.create_subscription(
            String, self.listen_topic, self.listen_cb, 10
        )

        self.busy = False
        self.vad = webrtcvad.Vad(self.vad_mode)

        self.publish_status(
            f"asr_node started: device={self.record_device}, sr={self.record_sample_rate}, "
            f"channels={self.record_channels}, format={self.record_format}, "
            f"vad_mode={self.vad_mode}, frame_ms={self.frame_ms}"
        )
        self.get_logger().info(
            f"asr_node started: device={self.record_device}, sr={self.record_sample_rate}, "
            f"channels={self.record_channels}, format={self.record_format}, "
            f"vad_mode={self.vad_mode}, frame_ms={self.frame_ms}"
        )

    def publish_status(self, text: str):
        msg = String()
        msg.data = text
        self.status_pub.publish(msg)

    def listen_cb(self, msg: String):
        if self.busy:
            self.publish_status("ASR busy, listen ignored")
            return

        self.busy = True
        wav_path = None

        try:
            self.publish_status("ASR listen requested")
            wav_path = self.record_until_silence()

            if wav_path is None:
                self.publish_status("ASR: no speech detected")
                return

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
            self.busy = False
            if wav_path and os.path.exists(wav_path):
                try:
                    os.remove(wav_path)
                except Exception:
                    pass

    def record_until_silence(self):
        if self.frame_ms not in (10, 20, 30):
            raise RuntimeError("frame_ms must be 10, 20 or 30")

        # 48k stereo s32le raw chunk size
        samples_per_ch_48k = int(self.record_sample_rate * self.frame_ms / 1000)
        bytes_per_chunk = samples_per_ch_48k * self.record_channels * 4  # int32

        pre_roll_frames = max(1, self.pre_roll_ms // self.frame_ms)
        start_window_frames = max(1, self.start_window_ms // self.frame_ms)
        end_window_frames = max(1, self.end_window_ms // self.frame_ms)
        min_utterance_frames = max(1, self.min_utterance_ms // self.frame_ms)
        max_utterance_frames = max(1, int(self.max_utterance_sec * 1000 / self.frame_ms))
        speech_timeout_frames = max(1, int(self.speech_timeout_sec * 1000 / self.frame_ms))

        pre_roll = deque(maxlen=pre_roll_frames)
        start_ring = deque(maxlen=start_window_frames)
        end_ring = deque(maxlen=end_window_frames)

        collected_pcm16_frames = []

        cmd = [
            "arecord",
            "-D", self.record_device,
            "-c", str(self.record_channels),
            "-r", str(self.record_sample_rate),
            "-f", self.record_format,
            "-t", "raw",
        ]

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )

        self.publish_status("ASR recording started")

        triggered = False
        frames_since_start = 0
        waited_frames = 0

        try:
            while True:
                raw_chunk = self.read_exact(proc.stdout, bytes_per_chunk)
                if not raw_chunk:
                    break

                frame16 = self.raw48k_stereo_s32_to_16k_mono_s16(raw_chunk)
                if frame16 is None:
                    continue

                is_speech = self.vad.is_speech(frame16, 16000)

                if not triggered:
                    pre_roll.append(frame16)
                    start_ring.append(is_speech)
                    waited_frames += 1

                    voiced_ratio = sum(start_ring) / len(start_ring)

                    if len(start_ring) == start_window_frames and voiced_ratio >= self.start_trigger_ratio:
                        triggered = True
                        self.publish_status("ASR speech detected")
                        collected_pcm16_frames.extend(list(pre_roll))
                        end_ring.clear()
                        frames_since_start = 0
                        continue

                    if waited_frames >= speech_timeout_frames:
                        self.publish_status("ASR speech timeout")
                        return None

                else:
                    collected_pcm16_frames.append(frame16)
                    end_ring.append(is_speech)
                    frames_since_start += 1

                    if frames_since_start >= max_utterance_frames:
                        self.publish_status("ASR utterance stopped by max duration")
                        break

                    if frames_since_start >= min_utterance_frames and len(end_ring) == end_window_frames:
                        unvoiced_ratio = (len(end_ring) - sum(end_ring)) / len(end_ring)
                        if unvoiced_ratio >= self.end_trigger_ratio:
                            self.publish_status("ASR utterance stopped by silence")
                            break

        finally:
            try:
                proc.terminate()
            except Exception:
                pass
            try:
                proc.wait(timeout=1.0)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

        if not triggered or not collected_pcm16_frames:
            return None

        wav_path = self.save_pcm16_frames_to_wav(collected_pcm16_frames)
        self.publish_status(f"ASR recording finished: {wav_path}")
        return wav_path

    def read_exact(self, stream, nbytes: int) -> bytes:
        data = bytearray()
        while len(data) < nbytes:
            part = stream.read(nbytes - len(data))
            if not part:
                break
            data.extend(part)
        return bytes(data)

    def raw48k_stereo_s32_to_16k_mono_s16(self, raw_chunk: bytes):
        try:
            data = np.frombuffer(raw_chunk, dtype=np.int32)
            if data.size == 0:
                return None

            data = data.reshape(-1, self.record_channels)
            mono32 = data.mean(axis=1)

            # convert to int16 scale
            mono16 = np.clip(mono32 / 65536.0, -32768, 32767).astype(np.int16)

            # 48k -> 16k simple decimation by 3
            mono16_16k = mono16[::3]

            return mono16_16k.tobytes()
        except Exception:
            return None

    def save_pcm16_frames_to_wav(self, frames):
        with tempfile.NamedTemporaryFile(
            suffix=".wav", dir=self.tmp_dir, delete=False
        ) as f:
            wav_path = f.name

        with wave.open(wav_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(b"".join(frames))

        return wav_path

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
            result = " ".join(text_parts).strip()
        else:
            fallback = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("whisper_") or line.startswith("system_info:") or line.startswith("main:"):
                    continue
                fallback.append(line)
            result = " ".join(fallback).strip()

        bad_tokens = {
            "[музыка]",
            "[смех]",
            "[аплодисменты]",
            "[music]",
            "[laughter]",
            "[applause]",
        }

        if result.lower() in bad_tokens:
            return ""

        return result


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
