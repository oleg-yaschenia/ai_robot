from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    serial_port = LaunchConfiguration("serial_port")
    baud_rate = LaunchConfiguration("baud_rate")

    transport = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("robot_esp32_bridge"),
                "launch",
                "esp32_bridge.launch.py",
            ])
        ),
        launch_arguments={
            "serial_port": serial_port,
            "baud_rate": baud_rate,
            "enable_drive": "false",
            "enable_head": "true",
            "enable_neck": "false",
        }.items(),
    )

    behavior = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("robot_esp32_bridge"),
                "launch",
                "head_behavior.launch.py",
            ])
        )
    )

    return LaunchDescription([
        DeclareLaunchArgument("serial_port", default_value="/dev/ttyTHS1"),
        DeclareLaunchArgument("baud_rate", default_value="115200"),
        transport,
        behavior,
    ])
