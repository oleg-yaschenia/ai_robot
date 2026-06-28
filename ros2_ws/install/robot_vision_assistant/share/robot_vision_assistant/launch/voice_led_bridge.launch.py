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
                {"head_events_topic": "/robot/head/events"},
                {"debug_topic": "/voice_led_bridge/debug"},
            ],
        )
    ])
