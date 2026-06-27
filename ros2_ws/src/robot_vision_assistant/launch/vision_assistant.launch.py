from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
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
        )
    ])
