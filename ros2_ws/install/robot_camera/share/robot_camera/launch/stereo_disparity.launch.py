from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package="stereo_image_proc",
            executable="disparity_node",
            name="stereo_disparity",
            output="screen",
            remappings=[
                ("left/image_rect", "/camera/left/image_rect"),
                ("left/camera_info", "/camera/left/camera_info"),
                ("right/image_rect", "/camera/right/image_rect"),
                ("right/camera_info", "/camera/right/camera_info"),
            ],
        )
    ])
