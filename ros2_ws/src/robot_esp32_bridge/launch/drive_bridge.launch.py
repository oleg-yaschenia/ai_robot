from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package="robot_esp32_bridge",
            executable="drive_bridge_node",
            name="drive_bridge_node",
            output="screen",
            parameters=[
                {"port": "/dev/ttyTHS1"},
                {"baud": 115200},
                {"k_vx": 900.0},
                {"k_vy": 1500.0},
                {"k_wz": 900.0},
                {"cmd_timeout_sec": 0.5},
            ],
        )
    ])
