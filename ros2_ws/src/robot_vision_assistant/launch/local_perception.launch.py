from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package="robot_vision_assistant",
            executable="local_perception_node",
            name="local_perception_node",
            output="screen",
            parameters=[
                {"image_topic": "/camera/left/image_raw"},
                {"analysis_period_sec": 0.5},
                {"presence_hold_sec": 2.0},
                {"motion_threshold_low": 4.0},
                {"motion_threshold_high": 12.0},
            ],
        )
    ])
