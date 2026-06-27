from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package="robot_vision_assistant",
            executable="voice_led_bridge_node",
            name="voice_led_bridge_node",
            output="screen",
            parameters=[
                {"voice_state_topic": "/voice/state"},

                # ВАЖНО:
                # здесь должен быть ваш реальный topic, который уже принимает режимы головы
                {"head_mode_topic": "/head_mode"},

                {"debug_topic": "/voice_led_bridge/debug"},

                # если ваш head LED bridge уже понимает такие строки, оставляем как есть
                {"idle_mode": "IDLE"},
                {"listening_mode": "LISTENING"},
                {"thinking_mode": "THINKING"},
                {"speaking_mode": "SPEAKING"},
            ],
        )
    ])
