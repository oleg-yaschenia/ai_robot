from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
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
        )
    ])
