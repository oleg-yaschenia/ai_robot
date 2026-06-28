from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package="robot_vision_assistant",
            executable="yolo_perception_node",
            name="yolo_perception_node",
            output="screen",
            parameters=[
                {"image_topic": "/camera/left/image_raw"},
                {"model_path": "yolo11s.pt"},
                {"imgsz": 960},
                {"conf_threshold": 0.5},
                {"analysis_period_sec": 0.7},
                {"max_det": 10},
            ],
        ),
        Node(
            package="robot_vision_assistant",
            executable="vision_assistant_node",
            name="vision_assistant_node",
            output="screen",
            parameters=[
                {"image_topic": "/camera/left/image_raw"},
                {"mode": "local_only"},
                {"allow_cloud": False},
                {"allow_realtime": False},
                {"snapshots_dir": "/home/warxen/ai_robot/data/vision_assistant/snapshots"},
                {"db_path": "/home/warxen/ai_robot/data/vision_assistant/assistant_memory.sqlite"},
            ],
        ),
        Node(
            package="robot_vision_assistant",
            executable="tts_node",
            name="tts_node",
            output="screen",
            parameters=[
                {"answer_topic": "/vision_assistant/answer"},
                {"status_topic": "/voice_tts/status"},
                {"enabled": True},
                {"piper_bin": "/home/warxen/ai_robot/data/tts/piper/piper/piper"},
                {"model_path": "/home/warxen/ai_robot/data/tts/piper/ru_RU-ruslan-medium.onnx"},
                {"audio_player": "aplay"},
                {"audio_player_args": ["-q"]},
                {"tmp_dir": "/tmp/robot_tts"},
            ],
        ),
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

                {"whisper_cli": "/home/warxen/ai_robot/tools/whisper.cpp/build/bin/whisper-cli"},
                {"model_path": "/home/warxen/ai_robot/tools/whisper.cpp/models/ggml-base.bin"},
                {"language": "ru"},
                {"threads": 4},
                {"processors": 1},
                {"tmp_dir": "/tmp/robot_asr"},

                {"vad_mode": 2},
                {"frame_ms": 30},
                {"pre_roll_ms": 450},
                {"speech_timeout_sec": 8.0},
                {"max_utterance_sec": 8.0},
                {"start_window_ms": 240},
                {"start_trigger_ratio": 0.60},
                {"end_window_ms": 900},
                {"end_trigger_ratio": 0.90},
                {"min_utterance_ms": 400},
            ],
        ),
        Node(
            package="robot_vision_assistant",
            executable="voice_manager_node",
            name="voice_manager_node",
            output="screen",
            parameters=[
                {"start_topic": "/voice/start"},
                {"state_topic": "/voice/state"},
                {"asr_listen_topic": "/voice_asr/listen"},
                {"asr_status_topic": "/voice_asr/status"},
                {"asr_transcript_topic": "/voice_asr/transcript"},
                {"answer_topic": "/vision_assistant/answer"},
                {"tts_status_topic": "/voice_tts/status"},
            ],
        ),
    ])
