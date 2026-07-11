import os
from pathlib import Path
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    project_root = Path(
        os.environ.get(
            "AI_ROBOT_ROOT",
            str(Path.home() / "ai_robot"),
        )
    ).expanduser()
    return LaunchDescription([
        Node(
            package="robot_vision_assistant",
            executable="asr_node",
            name="asr_node",
            output="screen",
            parameters=[
                {"listen_topic": "/voice_asr/listen"},
                {"query_topic": "/vision_assistant/query"},
                {"transcript_topic": "/voice_asr/transcript"},
                {"status_topic": "/voice_asr/status"},

                {"record_device": "hw:1,0"},
                {"record_sample_rate": 48000},
                {"record_channels": 2},
                {"record_format": "S32_LE"},

                {"whisper_cli": str(project_root / "tools" / "whisper.cpp" / "build" / "bin" / "whisper-cli")},
                {"model_path": str(project_root / "tools" / "whisper.cpp" / "models" / "ggml-base.bin")},
                {"language": "ru"},
                {"threads": 4},
                {"processors": 1},
                {"tmp_dir": "/tmp/robot_asr"},

                {"channel_strategy": "best_snr"},
                {"input_gain_db": 18.0},
                {"limiter_peak_dbfs": -6.0},
                {"start_energy_dbfs": -47.0},
                {"vad_mode": 0},
                {"frame_ms": 30},
                {"pre_roll_ms": 900},
                {"speech_timeout_sec": 10.0},
                {"max_utterance_sec": 8.0},
                {"start_window_ms": 300},
                {"start_trigger_ratio": 0.30},
                {"end_window_ms": 1200},
                {"end_trigger_ratio": 0.85},
                {"min_utterance_ms": 500},
                {"noise_calibration_ms": 600},
                {"start_snr_margin_db": 4.0},
                {"end_snr_margin_db": 2.0},
                {"speech_confirm_ms": 240},
                {"end_grace_ms": 1200},
                {"debug_keep_wav": True},
            ],
        )
    ])
