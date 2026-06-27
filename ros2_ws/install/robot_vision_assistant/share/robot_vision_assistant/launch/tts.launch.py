from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package="robot_vision_assistant",
            executable="tts_node",
            name="tts_node",
            output="screen",
            parameters=[
                {"answer_topic": "/vision_assistant/answer"},
                {"enabled": True},
                {"piper_bin": "/home/warxen/ai_robot/data/tts/piper/piper/piper"},
                {"model_path": "/home/warxen/ai_robot/data/tts/piper/ru_RU-ruslan-medium.onnx"},
                {"audio_player": "aplay"},
                {"audio_player_args": ["-q"]},
                {"tmp_dir": "/tmp/robot_tts"},
                {"status_topic": "/voice_tts/status"},
            ],
        )
    ])
