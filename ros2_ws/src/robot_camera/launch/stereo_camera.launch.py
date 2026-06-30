import os
from pathlib import Path
from launch import LaunchDescription
from launch.actions import UnsetEnvironmentVariable
from launch_ros.actions import Node


def generate_launch_description():
    project_root = Path(
        os.environ.get(
            "AI_ROBOT_ROOT",
            str(Path.home() / "ai_robot"),
        )
    ).expanduser()
    return LaunchDescription([
        UnsetEnvironmentVariable(name="DISPLAY"),
        UnsetEnvironmentVariable(name="WAYLAND_DISPLAY"),
        UnsetEnvironmentVariable(name="XAUTHORITY"),
        Node(
            package="robot_camera",
            executable="stereo_camera_node",
            name="stereo_camera_node",
            output="screen",
            parameters=[
                {"left_sensor_id": 0},
                {"right_sensor_id": 1},
                {"width": 1280},
                {"height": 720},
                {"fps": 10},
                {"left_frame_id": "camera_left_optical_frame"},
                {"right_frame_id": "camera_right_optical_frame"},
                {"left_camera_info_yaml": str(project_root / "calib" / "stereo" / "left.yaml")},
                {"right_camera_info_yaml": str(project_root / "calib" / "stereo" / "right.yaml")},
            ],
        )
    ])
