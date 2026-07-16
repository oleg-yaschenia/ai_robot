import os
from pathlib import Path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    project_root = Path(
        os.environ.get(
            "AI_ROBOT_ROOT",
            str(Path.home() / "ai_robot"),
        )
    ).expanduser()
    default_whisper_cli = str(project_root / "tools" / "whisper.cpp" / "build" / "bin" / "whisper-cli")
    default_model_path = str(project_root / "tools" / "whisper.cpp" / "models" / "ggml-base.bin")

    return LaunchDescription([
        DeclareLaunchArgument("whisper_cli", default_value=default_whisper_cli),
        DeclareLaunchArgument("model_path", default_value=default_model_path),
        DeclareLaunchArgument("start_energy_dbfs", default_value="-47.0"),
        DeclareLaunchArgument("start_snr_margin_db", default_value="4.0"),
        DeclareLaunchArgument("speech_confirm_ms", default_value="240"),
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

                {"whisper_cli": LaunchConfiguration("whisper_cli")},
                {"model_path": LaunchConfiguration("model_path")},
                {"language": "ru"},
                {"threads": 4},
                {"processors": 1},
                {"tmp_dir": "/tmp/robot_asr"},

                {"channel_strategy": "best_snr"},
                {"input_gain_db": 18.0},
                {"limiter_peak_dbfs": -6.0},
                {"start_energy_dbfs": LaunchConfiguration("start_energy_dbfs")},
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
                {"start_snr_margin_db": LaunchConfiguration("start_snr_margin_db")},
                {"end_snr_margin_db": 2.0},
                {"speech_confirm_ms": LaunchConfiguration("speech_confirm_ms")},
                {"end_grace_ms": 1200},
                {"debug_keep_wav": False},
            ],
        )
    ])
