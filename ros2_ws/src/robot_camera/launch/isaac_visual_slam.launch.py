from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package="robot_camera",
            executable="isaac_vslam_relay_node",
            name="isaac_vslam_relay_node",
            output="screen",
        ),
        Node(
            package="isaac_ros_visual_slam",
            executable="visual_slam_node",
            name="visual_slam_node",
            output="screen",
            parameters=[{
                "num_cameras": 2,
                "enable_imu_fusion": False,
                "publish_map_to_odom_tf": True,
                "publish_odom_to_base_tf": True,
                "base_frame": "base_link",
                "odom_frame": "odom_vo",
                "map_frame": "map_vo",
                "enable_slam_visualization": False,
            }],
        ),
    ])
