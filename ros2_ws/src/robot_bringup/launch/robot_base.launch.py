from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    UnsetEnvironmentVariable,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    enable_esp32 = LaunchConfiguration("enable_esp32")
    serial_port = LaunchConfiguration("serial_port")
    baud_rate = LaunchConfiguration("baud_rate")

    esp32_transport = IncludeLaunchDescription(
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
            "enable_drive": "true",
            "enable_head": "true",
            "enable_neck": "true",
        }.items(),
        condition=IfCondition(enable_esp32),
    )

    stereo_camera_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("robot_camera"),
                "launch",
                "stereo_camera.launch.py",
            ])
        )
    )

    stereo_rectify_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("robot_camera"),
                "launch",
                "stereo_processing_composed.launch.py",
            ])
        )
    )

    stereo_disparity_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("robot_camera"),
                "launch",
                "stereo_disparity.launch.py",
            ])
        )
    )

    return LaunchDescription([
        DeclareLaunchArgument("enable_esp32", default_value="true"),
        DeclareLaunchArgument("serial_port", default_value="/dev/ttyTHS1"),
        DeclareLaunchArgument("baud_rate", default_value="115200"),
        UnsetEnvironmentVariable(name="DISPLAY"),
        UnsetEnvironmentVariable(name="WAYLAND_DISPLAY"),
        UnsetEnvironmentVariable(name="XAUTHORITY"),
        esp32_transport,
        stereo_camera_launch,
        stereo_rectify_launch,
    ])
