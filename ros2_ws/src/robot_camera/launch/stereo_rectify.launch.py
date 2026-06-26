from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package="image_proc",
            executable="rectify_node",
            name="left_rectify",
            namespace="camera/left",
            output="screen",
            remappings=[
                ("image", "/camera/left/image_raw"),
                ("camera_info", "/camera/left/camera_info"),
            ],
        ),
        Node(
            package="image_proc",
            executable="rectify_node",
            name="right_rectify",
            namespace="camera/right",
            output="screen",
            remappings=[
                ("image", "/camera/right/image_raw"),
                ("camera_info", "/camera/right/camera_info"),
            ],
        ),
    ])
