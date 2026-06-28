from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    serial_port = LaunchConfiguration("serial_port")
    baud_rate = LaunchConfiguration("baud_rate")
    enable_drive = LaunchConfiguration("enable_drive")
    enable_head = LaunchConfiguration("enable_head")
    enable_neck = LaunchConfiguration("enable_neck")
    k_vx = LaunchConfiguration("k_vx")
    k_vy = LaunchConfiguration("k_vy")
    k_wz = LaunchConfiguration("k_wz")
    max_wheel_target = LaunchConfiguration("max_wheel_target")
    cmd_timeout_sec = LaunchConfiguration("cmd_timeout_sec")
    inter_cmd_delay_sec = LaunchConfiguration("inter_cmd_delay_sec")
    use_all_wheels_command = LaunchConfiguration("use_all_wheels_command")

    return LaunchDescription([
        DeclareLaunchArgument("serial_port", default_value="/dev/ttyTHS1"),
        DeclareLaunchArgument("baud_rate", default_value="115200"),
        DeclareLaunchArgument("enable_drive", default_value="true"),
        DeclareLaunchArgument("enable_head", default_value="true"),
        DeclareLaunchArgument("enable_neck", default_value="true"),
        DeclareLaunchArgument("k_vx", default_value="900.0"),
        DeclareLaunchArgument("k_vy", default_value="1500.0"),
        DeclareLaunchArgument("k_wz", default_value="900.0"),
        DeclareLaunchArgument("max_wheel_target", default_value="1800.0"),
        DeclareLaunchArgument("cmd_timeout_sec", default_value="0.5"),
        DeclareLaunchArgument("inter_cmd_delay_sec", default_value="0.03"),
        DeclareLaunchArgument("use_all_wheels_command", default_value="false"),
        Node(
            package="robot_esp32_bridge",
            executable="esp32_bridge_node",
            name="esp32_bridge_node",
            output="screen",
            parameters=[{
                "port": serial_port,
                "baudrate": ParameterValue(baud_rate, value_type=int),
                "enable_drive": ParameterValue(enable_drive, value_type=bool),
                "enable_head": ParameterValue(enable_head, value_type=bool),
                "enable_neck": ParameterValue(enable_neck, value_type=bool),
                "k_vx": ParameterValue(k_vx, value_type=float),
                "k_vy": ParameterValue(k_vy, value_type=float),
                "k_wz": ParameterValue(k_wz, value_type=float),
                "max_wheel_target": ParameterValue(
                    max_wheel_target, value_type=float
                ),
                "cmd_timeout_sec": ParameterValue(
                    cmd_timeout_sec, value_type=float
                ),
                "inter_cmd_delay_sec": ParameterValue(
                    inter_cmd_delay_sec, value_type=float
                ),
                "use_all_wheels_command": ParameterValue(
                    use_all_wheels_command, value_type=bool
                ),
            }],
        ),
    ])
