from launch import LaunchDescription
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    common_extra_arguments = [
        {"use_intra_process_comms": True},
    ]

    left_rectify = ComposableNode(
        package="image_proc",
        plugin="image_proc::RectifyNode",
        name="left_rectify",
        namespace="camera/left",
        parameters=[
            {"queue_size": 5},
            {"interpolation": 1},
        ],
        remappings=[
            ("image", "/camera/left/image_raw"),
            ("camera_info", "/camera/left/camera_info"),
        ],
        extra_arguments=common_extra_arguments,
    )

    right_rectify = ComposableNode(
        package="image_proc",
        plugin="image_proc::RectifyNode",
        name="right_rectify",
        namespace="camera/right",
        parameters=[
            {"queue_size": 5},
            {"interpolation": 1},
        ],
        remappings=[
            ("image", "/camera/right/image_raw"),
            ("camera_info", "/camera/right/camera_info"),
        ],
        extra_arguments=common_extra_arguments,
    )

    right_image_yolo_gate = ComposableNode(
        package="robot_camera_components",
        plugin="robot_camera_components::ImageRateGateNode",
        name="right_image_yolo_gate",
        namespace="",
        parameters=[
            {"rate_hz": 4.0},
            {"diagnostics_period_sec": 5.0},
        ],
        remappings=[
            ("image_in", "/camera/right/image_rect"),
            ("image_out", "/camera/right/image_rect_yolo"),
        ],
        extra_arguments=common_extra_arguments,
    )

    stereo_disparity = ComposableNode(
        package="stereo_image_proc",
        plugin="stereo_image_proc::DisparityNode",
        name="stereo_disparity",
        namespace="",
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
            ("left/image_rect", "/camera/left/image_rect"),
            ("left/camera_info", "/camera/left/camera_info"),
            ("right/image_rect", "/camera/right/image_rect"),
            ("right/camera_info", "/camera/right/camera_info"),
            ("disparity", "/disparity"),
        ],
        extra_arguments=common_extra_arguments,
    )

    container = ComposableNodeContainer(
        name="stereo_processing_container",
        namespace="",
        package="rclcpp_components",
        executable="component_container_mt",
        output="screen",
        composable_node_descriptions=[
            left_rectify,
            right_rectify,
            right_image_yolo_gate,
            stereo_disparity,
        ],
    )

    return LaunchDescription([container])
