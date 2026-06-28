from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    enable_head = LaunchConfiguration("enable_head")
    enable_head_serial = LaunchConfiguration("enable_head_serial")

    robot_base = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("robot_bringup"),
                "launch",
                "robot_base.launch.py",
            ])
        )
    )

    local_assistant = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("robot_vision_assistant"),
                "launch",
                "local_assistant.launch.py",
            ])
        )
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
        launch_arguments={
            "enable_serial_bridge": enable_head_serial,
            "serial_port": "/dev/ttyTHS1",
            "baud_rate": "115200",
        }.items(),
        condition=IfCondition(enable_head),
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "enable_head",
            default_value="true",
            description="Enable head behavior and voice LED bridge"
        ),
        DeclareLaunchArgument(
            "enable_head_serial",
            default_value="true",
            description="Enable esp32 head serial bridge"
        ),

        robot_base,
        local_assistant,
        voice_led_bridge,
        head_behavior,
    ])
