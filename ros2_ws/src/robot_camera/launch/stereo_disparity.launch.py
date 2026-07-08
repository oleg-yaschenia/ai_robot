from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package="stereo_image_proc",
            executable="disparity_node",
            name="stereo_disparity",
            output="screen",
            parameters=[{
                "stereo_algorithm": 0,
                "min_disparity": 0,
                "disparity_range": 128,
                "correlation_window_size": 15,
                "texture_threshold": 10,
                "uniqueness_ratio": 15.0,
                "speckle_size": 100,
                "speckle_range": 4,
                "disp12_max_diff": 0,
                "full_dp": False,
                "approximate_sync": False,
                "queue_size": 5,
            }],
            remappings=[
                (
                    "left/image_rect",
                    "/camera/left/image_rect",
                ),
                (
                    "left/camera_info",
                    "/camera/left/camera_info",
                ),
                (
                    "right/image_rect",
                    "/camera/right/image_rect",
                ),
                (
                    "right/camera_info",
                    "/camera/right/camera_info",
                ),
            ],
        ),
    ])
