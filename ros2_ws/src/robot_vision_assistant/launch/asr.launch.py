from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
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
                {"record_seconds": 5},
                {"record_sample_rate": 48000},
                {"record_channels": 2},
                {"record_format": "S32_LE"},
                {"whisper_cli": "/home/warxen/ai_robot/tools/whisper.cpp/build/bin/whisper-cli"},
                {"model_path": "/home/warxen/ai_robot/tools/whisper.cpp/models/ggml-base.bin"},
                {"language": "ru"},
                {"threads": 4},
                {"processors": 1},
                {"tmp_dir": "/tmp/robot_asr"},
            ],
        )
    ])
