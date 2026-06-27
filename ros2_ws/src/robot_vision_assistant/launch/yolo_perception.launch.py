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
        )
    ])
