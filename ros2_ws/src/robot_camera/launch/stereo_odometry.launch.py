from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package="rtabmap_odom",
            executable="stereo_odometry",
            name="stereo_odometry",
            output="screen",
			parameters=[{
				"frame_id": "camera_left_optical_frame",
				"odom_frame_id": "odom_vo",
				"publish_tf": True,
				"approx_sync": True,
				"approx_sync_max_interval": 0.01,
				"subscribe_odom_info": False,
				"queue_size": 20,
				"qos_image": 0,
				"qos_camera_info": 0,
				"Vis/MinInliers": "8",
				"Stereo/MaxDisparity": "256",
				"Odom/Strategy": "0",
			}],
            remappings=[
                ("left/image_rect", "/camera/left/image_rect"),
                ("left/camera_info", "/camera/left/camera_info"),
                ("right/image_rect", "/camera/right/image_rect"),
                ("right/camera_info", "/camera/right/camera_info"),
                ("odom", "/visual_odom"),
            ],
        )
    ])
