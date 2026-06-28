from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    EnvironmentVariable,
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    enable_head = LaunchConfiguration("enable_head")
    enable_esp32 = LaunchConfiguration("enable_esp32")
    serial_port = LaunchConfiguration("serial_port")
    baud_rate = LaunchConfiguration("baud_rate")
    project_root = LaunchConfiguration("project_root")

    robot_base = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("robot_bringup"),
                "launch",
                "robot_base.launch.py",
            ])
        ),
        launch_arguments={
            "enable_esp32": enable_esp32,
            "serial_port": serial_port,
            "baud_rate": baud_rate,
        }.items(),
    )

    local_assistant = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("robot_vision_assistant"),
                "launch",
                "local_assistant.launch.py",
            ])
        ),
        launch_arguments={"project_root": project_root}.items(),
    )

    voice_led_bridge = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("robot_vision_assistant"),
                "launch",
                "voice_led_bridge.launch.py",
            ])
        ),
        condition=IfCondition(enable_head),
    )

    head_behavior = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("robot_esp32_bridge"),
                "launch",
                "head_behavior.launch.py",
            ])
        ),
        condition=IfCondition(enable_head),
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "enable_head",
            default_value="true",
            description="Enable head behavior and voice LED bridge",
        ),
        DeclareLaunchArgument(
            "enable_esp32",
            default_value="true",
            description="Start the single ESP32 UART owner",
        ),
        DeclareLaunchArgument("serial_port", default_value="/dev/ttyTHS1"),
        DeclareLaunchArgument("baud_rate", default_value="115200"),
        DeclareLaunchArgument(
            "project_root",
            default_value=EnvironmentVariable(
                "AI_ROBOT_ROOT",
                default_value="/home/warxen/ai_robot",
            ),
            description="Repository root; may also be set with AI_ROBOT_ROOT",
        ),
        robot_base,
        local_assistant,
        voice_led_bridge,
        head_behavior,
    ])
