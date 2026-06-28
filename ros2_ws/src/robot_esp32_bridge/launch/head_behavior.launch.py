"""Head state logic only.

Serial transport is intentionally not started here. The unified
esp32_bridge_node is started once by robot_base.launch.py.
"""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package="robot_esp32_bridge",
            executable="head_state_manager",
            name="head_state_manager",
            output="screen",
        ),
    ])
