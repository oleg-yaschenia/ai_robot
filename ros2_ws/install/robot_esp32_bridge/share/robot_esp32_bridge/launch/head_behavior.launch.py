from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    enable_serial_bridge = LaunchConfiguration("enable_serial_bridge")
    serial_port = LaunchConfiguration("serial_port")
    baud_rate = LaunchConfiguration("baud_rate")

    return LaunchDescription([
        DeclareLaunchArgument(
            "enable_serial_bridge",
            default_value="true",
            description="Start esp32_bridge_node for head LEDs"
        ),
        DeclareLaunchArgument(
            "serial_port",
            default_value="/dev/ttyTHS1",
        ),
        DeclareLaunchArgument(
            "baud_rate",
            default_value="115200",
        ),

        Node(
            package="robot_esp32_bridge",
            executable="head_state_manager",
            name="head_state_manager",
            output="screen",
        ),

        Node(
            package="robot_esp32_bridge",
            executable="esp32_bridge_node",
            name="esp32_bridge_node",
            output="screen",
            condition=IfCondition(enable_serial_bridge),
            parameters=[
                {"port": serial_port},
                {"baud": baud_rate},
            ],
        ),
    ])
