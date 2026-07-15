#!/usr/bin/env python3
import math
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
from std_msgs.msg import Bool, String


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

        # Far-field audio front-end. The ICS-43434 path has no useful ALSA
        # capture gain, so sensitivity is handled before VAD and Whisper.
        self.declare_parameter("channel_strategy", "best_snr")
        self.declare_parameter("input_gain_db", 18.0)
        self.declare_parameter("limiter_peak_dbfs", -6.0)
        self.declare_parameter("start_energy_dbfs", -47.0)
        self.declare_parameter("noise_calibration_ms", 600)
        self.declare_parameter("start_snr_margin_db", 4.0)
        self.declare_parameter("end_snr_margin_db", 2.0)
        self.declare_parameter("speech_confirm_ms", 240)
        self.declare_parameter("end_grace_ms", 1200)
        self.declare_parameter("debug_keep_wav", False)

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

        self.channel_strategy = str(
            self.get_parameter("channel_strategy").value
        ).strip().lower()
        self.input_gain_db = float(self.get_parameter("input_gain_db").value)
        self.limiter_peak_dbfs = float(
            self.get_parameter("limiter_peak_dbfs").value
        )
        self.start_energy_dbfs = float(
            self.get_parameter("start_energy_dbfs").value
        )
        self.noise_calibration_ms = int(
            self.get_parameter("noise_calibration_ms").value
        )
        self.start_snr_margin_db = float(
            self.get_parameter("start_snr_margin_db").value
        )
        self.end_snr_margin_db = float(
            self.get_parameter("end_snr_margin_db").value
        )
        self.speech_confirm_ms = int(
            self.get_parameter("speech_confirm_ms").value
        )
        self.end_grace_ms = int(
            self.get_parameter("end_grace_ms").value
        )
        self.debug_keep_wav = bool(
            self.get_parameter("debug_keep_wav").value
        )

        self.tmp_dir = Path(str(self.get_parameter("tmp_dir").value))
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

        self.query_pub = self.create_publisher(String, self.query_topic, 10)
        self.transcript_pub = self.create_publisher(String, self.transcript_topic, 10)
        self.status_pub = self.create_publisher(String, self.status_topic, 10)

        self.listen_sub = self.create_subscription(
            Bool, self.listen_topic, self.listen_cb, 10
        )

        self.busy = False
        self.vads = [
            webrtcvad.Vad(self.vad_mode)
            for _ in range(max(1, self.record_channels))
        ]

        self.publish_status(
            f"asr_node started: device={self.record_device}, sr={self.record_sample_rate}, "
            f"channels={self.record_channels}, format={self.record_format}, "
            f"vad_mode={self.vad_mode}, frame_ms={self.frame_ms}, "
            f"channel_strategy={self.channel_strategy}, "
            f"input_gain_db={self.input_gain_db:.1f}, "
            f"start_energy_dbfs={self.start_energy_dbfs:.1f}, "
            f"noise_calibration_ms={self.noise_calibration_ms}, "
            f"speech_confirm_ms={self.speech_confirm_ms}"
        )
        self.get_logger().info(
            f"asr_node started: device={self.record_device}, sr={self.record_sample_rate}, "
            f"channels={self.record_channels}, format={self.record_format}, "
            f"vad_mode={self.vad_mode}, frame_ms={self.frame_ms}, "
            f"channel_strategy={self.channel_strategy}, "
            f"input_gain_db={self.input_gain_db:.1f}, "
            f"start_energy_dbfs={self.start_energy_dbfs:.1f}, "
            f"noise_calibration_ms={self.noise_calibration_ms}, "
            f"speech_confirm_ms={self.speech_confirm_ms}"
        )

    def publish_status(self, text: str):
        msg = String()
        msg.data = text
        self.status_pub.publish(msg)

    def listen_cb(self, msg: Bool):
        if not msg.data:
            self.publish_status("ASR listen false ignored")
            return

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
            keep_wav = self.debug_keep_wav
            try:
                keep_wav = bool(self.get_parameter("debug_keep_wav").value)
            except Exception:
                pass

            if (
                wav_path
                and os.path.exists(wav_path)
                and not keep_wav
            ):
                try:
                    os.remove(wav_path)
                except Exception:
                    pass

    def record_until_silence(self):
        if self.frame_ms not in (10, 20, 30):
            raise RuntimeError("frame_ms must be 10, 20 or 30")
        if self.record_channels < 1:
            raise RuntimeError("record_channels must be at least 1")

        samples_per_ch_48k = int(
            self.record_sample_rate * self.frame_ms / 1000
        )
        bytes_per_chunk = samples_per_ch_48k * self.record_channels * 4

        pre_roll_frames = max(1, self.pre_roll_ms // self.frame_ms)
        start_window_frames = max(1, self.start_window_ms // self.frame_ms)
        end_window_frames = max(1, self.end_window_ms // self.frame_ms)
        min_utterance_frames = max(1, self.min_utterance_ms // self.frame_ms)
        max_utterance_frames = max(
            1, int(self.max_utterance_sec * 1000 / self.frame_ms)
        )
        speech_timeout_frames = max(
            1, int(self.speech_timeout_sec * 1000 / self.frame_ms)
        )
        calibration_frames = max(
            1, self.noise_calibration_ms // self.frame_ms
        )
        confirm_frames = max(
            1, self.speech_confirm_ms // self.frame_ms
        )
        end_grace_frames = max(
            min_utterance_frames,
            self.end_grace_ms // self.frame_ms,
        )

        # Each buffered item is (channel_frames, vad_flags, rms_dbfs).
        pre_roll = deque(maxlen=pre_roll_frames)
        start_ring = deque(maxlen=start_window_frames)
        start_channel_rings = [
            deque(maxlen=start_window_frames)
            for _ in range(self.record_channels)
        ]
        end_ring = deque(maxlen=end_window_frames)

        noise_rms_by_channel = [
            [] for _ in range(self.record_channels)
        ]
        start_thresholds = [
            self.start_energy_dbfs for _ in range(self.record_channels)
        ]
        end_thresholds = [
            self.start_energy_dbfs - 3.0
            for _ in range(self.record_channels)
        ]

        collected_by_channel = [
            [] for _ in range(self.record_channels)
        ]
        voiced_counts = [0 for _ in range(self.record_channels)]
        voiced_rms_by_channel = [
            [] for _ in range(self.record_channels)
        ]
        all_rms_by_channel = [
            [] for _ in range(self.record_channels)
        ]

        def reset_collection():
            for channel_index in range(self.record_channels):
                collected_by_channel[channel_index].clear()
                voiced_counts[channel_index] = 0
                voiced_rms_by_channel[channel_index].clear()
                all_rms_by_channel[channel_index].clear()

        def collect(item):
            channel_frames, vad_flags, rms_values = item
            for channel_index in range(self.record_channels):
                collected_by_channel[channel_index].append(
                    channel_frames[channel_index]
                )
                all_rms_by_channel[channel_index].append(
                    rms_values[channel_index]
                )
                if vad_flags[channel_index]:
                    voiced_counts[channel_index] += 1
                    voiced_rms_by_channel[channel_index].append(
                        rms_values[channel_index]
                    )

        def activity_flags(vad_flags, rms_values, thresholds):
            return [
                bool(vad_flags[channel_index])
                and rms_values[channel_index] >= thresholds[channel_index]
                for channel_index in range(self.record_channels)
            ]

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
        confirmed_activity_frames = 0
        waited_frames = 0
        calibration_count = 0

        try:
            while True:
                raw_chunk = self.read_exact(proc.stdout, bytes_per_chunk)
                if not raw_chunk:
                    break

                channel_frames = (
                    self.raw48k_stereo_s32_to_16k_channels_s16(raw_chunk)
                )
                if not channel_frames:
                    continue

                vad_flags = []
                rms_values = []
                for channel_index, frame16 in enumerate(channel_frames):
                    rms_dbfs = self.pcm16_rms_dbfs(frame16)
                    rms_values.append(rms_dbfs)
                    vad_flags.append(
                        self.vads[channel_index].is_speech(frame16, 16000)
                    )

                item = (channel_frames, vad_flags, rms_values)
                pre_roll.append(item)

                # Ignore the ALSA/I2S startup transient and estimate the
                # current room noise before allowing a speech trigger.
                if calibration_count < calibration_frames:
                    for channel_index in range(self.record_channels):
                        noise_rms_by_channel[channel_index].append(
                            rms_values[channel_index]
                        )
                    calibration_count += 1
                    if calibration_count == calibration_frames:
                        noise_floors = []
                        for channel_index in range(self.record_channels):
                            values = noise_rms_by_channel[channel_index]
                            noise_floor = (
                                float(np.percentile(values, 30))
                                if values
                                else -120.0
                            )
                            noise_floors.append(noise_floor)
                            start_thresholds[channel_index] = max(
                                self.start_energy_dbfs,
                                noise_floor + self.start_snr_margin_db,
                            )
                            end_thresholds[channel_index] = max(
                                self.start_energy_dbfs - 3.0,
                                noise_floor + self.end_snr_margin_db,
                            )
                        self.publish_status(
                            "ASR noise calibrated: floor="
                            + ",".join(
                                f"CH{index}:{value:.1f}"
                                for index, value in enumerate(noise_floors)
                            )
                            + " dBFS, start="
                            + ",".join(
                                f"CH{index}:{value:.1f}"
                                for index, value in enumerate(start_thresholds)
                            )
                            + " dBFS"
                        )
                        start_ring.clear()
                        for ring in start_channel_rings:
                            ring.clear()
                    continue

                start_flags = activity_flags(
                    vad_flags, rms_values, start_thresholds
                )
                end_flags = activity_flags(
                    vad_flags, rms_values, end_thresholds
                )
                start_speech = any(start_flags)
                end_speech = any(end_flags)

                if not triggered:
                    start_ring.append(start_speech)
                    for channel_index in range(self.record_channels):
                        start_channel_rings[channel_index].append(
                            start_flags[channel_index]
                        )
                    waited_frames += 1

                    voiced_ratio = sum(start_ring) / len(start_ring)

                    if (
                        len(start_ring) == start_window_frames
                        and voiced_ratio >= self.start_trigger_ratio
                    ):
                        triggered = True
                        reset_collection()
                        for buffered_item in pre_roll:
                            collect(buffered_item)

                        confirmed_activity_frames = 0
                        for buffered_item in pre_roll:
                            _, buffered_vad, buffered_rms = buffered_item
                            buffered_flags = activity_flags(
                                buffered_vad,
                                buffered_rms,
                                start_thresholds,
                            )
                            if any(buffered_flags):
                                confirmed_activity_frames += 1

                        active_channels = [
                            str(index)
                            for index, ring in enumerate(start_channel_rings)
                            if any(ring)
                        ]
                        self.publish_status(
                            "ASR speech candidate: channels="
                            + ",".join(active_channels)
                        )
                        end_ring.clear()
                        frames_since_start = 0
                        continue

                    if waited_frames >= speech_timeout_frames:
                        self.publish_status("ASR speech timeout")
                        return None

                else:
                    collect(item)
                    if start_speech:
                        confirmed_activity_frames += 1
                    end_ring.append(end_speech)
                    frames_since_start += 1

                    if (
                        confirmed_activity_frames == confirm_frames
                    ):
                        self.publish_status(
                            "ASR speech detected: "
                            f"active_frames={confirmed_activity_frames}"
                        )

                    if frames_since_start >= max_utterance_frames:
                        if confirmed_activity_frames < confirm_frames:
                            self.publish_status(
                                "ASR false trigger rejected at max duration"
                            )
                            return None
                        self.publish_status(
                            "ASR utterance stopped by max duration"
                        )
                        break

                    if (
                        frames_since_start >= end_grace_frames
                        and len(end_ring) == end_window_frames
                    ):
                        unvoiced_ratio = (
                            len(end_ring) - sum(end_ring)
                        ) / len(end_ring)
                        if unvoiced_ratio >= self.end_trigger_ratio:
                            if confirmed_activity_frames < confirm_frames:
                                self.publish_status(
                                    "ASR false trigger rejected: "
                                    f"active_frames={confirmed_activity_frames}"
                                )
                                triggered = False
                                frames_since_start = 0
                                confirmed_activity_frames = 0
                                reset_collection()
                                start_ring.clear()
                                for ring in start_channel_rings:
                                    ring.clear()
                                end_ring.clear()
                                continue
                            self.publish_status(
                                "ASR utterance stopped by silence"
                            )
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

        if not triggered:
            return None
        if confirmed_activity_frames < confirm_frames:
            self.publish_status(
                "ASR candidate rejected: insufficient confirmed speech"
            )
            return None

        selected_channel = self.select_best_channel(
            voiced_counts,
            voiced_rms_by_channel,
            all_rms_by_channel,
        )
        selected_frames = collected_by_channel[selected_channel]
        if not selected_frames:
            return None

        voiced_values = voiced_rms_by_channel[selected_channel]
        voiced_mean = (
            float(np.mean(voiced_values))
            if voiced_values
            else -120.0
        )
        p90 = (
            float(np.percentile(all_rms_by_channel[selected_channel], 90))
            if all_rms_by_channel[selected_channel]
            else -120.0
        )
        self.publish_status(
            f"ASR selected channel CH{selected_channel}: "
            f"voiced_frames={voiced_counts[selected_channel]}, "
            f"voiced_mean={voiced_mean:.1f} dBFS, "
            f"p90={p90:.1f} dBFS"
        )

        wav_path = self.save_pcm16_frames_to_wav(selected_frames)
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

    def raw48k_stereo_s32_to_16k_channels_s16(self, raw_chunk: bytes):
        try:
            data = np.frombuffer(raw_chunk, dtype=np.int32)
            if data.size == 0:
                return None

            data = data.reshape(-1, self.record_channels)

            # 48 kHz -> 16 kHz using a 3-sample boxcar before decimation.
            # This is still lightweight, but avoids the worst aliasing of
            # taking every third sample directly.
            usable = (data.shape[0] // 3) * 3
            if usable <= 0:
                return None
            reduced = data[:usable].reshape(
                -1, 3, self.record_channels
            ).mean(axis=1)

            base_gain = 10.0 ** (self.input_gain_db / 20.0)
            limiter_linear = 10.0 ** (self.limiter_peak_dbfs / 20.0)
            limiter_peak = 32767.0 * limiter_linear

            result = []
            for channel_index in range(self.record_channels):
                samples = reduced[:, channel_index].astype(np.float64)

                # Map S32 full scale to S16 full scale.
                samples /= 65536.0

                peak = float(np.max(np.abs(samples)))
                applied_gain = base_gain
                if peak > 0.0:
                    applied_gain = min(
                        base_gain,
                        limiter_peak / peak,
                    )

                samples *= applied_gain
                pcm16 = np.clip(
                    np.rint(samples), -32768, 32767
                ).astype(np.int16)
                result.append(pcm16.tobytes())

            return result
        except Exception as exc:
            self.get_logger().debug(f"audio conversion failed: {exc}")
            return None

    @staticmethod
    def pcm16_rms_dbfs(frame16: bytes) -> float:
        samples = np.frombuffer(frame16, dtype=np.int16).astype(np.float64)
        if samples.size == 0:
            return -120.0
        rms = math.sqrt(float(np.mean(samples * samples)))
        if rms <= 0.0:
            return -120.0
        return 20.0 * math.log10(rms / 32767.0)

    def select_best_channel(
        self,
        voiced_counts,
        voiced_rms_by_channel,
        all_rms_by_channel,
    ) -> int:
        if self.record_channels == 1:
            return 0
        if self.channel_strategy in {"left", "ch0", "0"}:
            return 0
        if self.channel_strategy in {"right", "ch1", "1"}:
            return min(1, self.record_channels - 1)
        if self.channel_strategy not in {"best_snr", "best_vad"}:
            self.get_logger().warning(
                f"Unknown channel_strategy={self.channel_strategy}; "
                "using best_snr"
            )

        best_index = 0
        best_score = None
        for channel_index in range(self.record_channels):
            all_values = all_rms_by_channel[channel_index]
            voiced_values = voiced_rms_by_channel[channel_index]

            noise_floor = (
                float(np.percentile(all_values, 25))
                if all_values
                else -120.0
            )
            speech_level = (
                float(np.percentile(voiced_values, 60))
                if voiced_values
                else -120.0
            )
            estimated_snr = speech_level - noise_floor
            count = voiced_counts[channel_index]

            # Prefer a channel with a sustained speech-like signal rather
            # than a channel that merely contains a few loud impulses.
            score = (
                count >= max(1, self.speech_confirm_ms // self.frame_ms),
                estimated_snr,
                count,
                speech_level,
            )
            if best_score is None or score > best_score:
                best_score = score
                best_index = channel_index

        return best_index

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
