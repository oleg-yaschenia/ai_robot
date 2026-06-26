from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, UnsetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    drive_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("robot_esp32_bridge"),
                "launch",
                "drive_bridge.launch.py",
            ])
        )
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
                "stereo_rectify.launch.py",
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
        UnsetEnvironmentVariable(name="DISPLAY"),
        UnsetEnvironmentVariable(name="WAYLAND_DISPLAY"),
        UnsetEnvironmentVariable(name="XAUTHORITY"),

        drive_launch,
        stereo_camera_launch,
        stereo_rectify_launch,
        stereo_disparity_launch,
    ])
