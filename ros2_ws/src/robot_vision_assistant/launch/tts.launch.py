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
            executable="tts_node",
            name="tts_node",
            output="screen",
            parameters=[
                {"answer_topic": "/vision_assistant/answer"},
                {"enabled": True},
                {"piper_bin": str(project_root / "data" / "tts" / "piper" / "piper" / "piper")},
                {"model_path": str(project_root / "data" / "tts" / "piper" / "ru_RU-ruslan-medium.onnx")},
                {"audio_player": "aplay"},
                {"audio_player_args": ["-q"]},
                {"tmp_dir": "/tmp/robot_tts"},
                {"status_topic": "/voice_tts/status"},
            ],
        )
    ])
