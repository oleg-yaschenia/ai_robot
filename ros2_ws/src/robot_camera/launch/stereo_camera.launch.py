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

    calibration_dir = (
        project_root
        / "calib"
        / "stereo_1024x768"
    )

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
                # Physical camera order.
                {"left_sensor_id": 1},
                {"right_sensor_id": 0},

                # Full-FOV IMX219 sensor mode.
                {"sensor_mode": 3},
                {"sensor_width": 1640},
                {"sensor_height": 1232},

                # Continuous hardware-scaled ROS output.
                {"width": 1024},
                {"height": 768},
                {"fps": 10},

                {"interpolation_method": 5},

                # Conservative ISP sharpening.
                {"ee_mode": 1},
                {"ee_strength": -1.0},

                {"publish_camera_info": True},

                {
                    "left_frame_id":
                    "camera_left_optical_frame"
                },
                {
                    "right_frame_id":
                    "camera_right_optical_frame"
                },

                {
                    "left_camera_info_yaml":
                    str(calibration_dir / "left.yaml")
                },
                {
                    "right_camera_info_yaml":
                    str(calibration_dir / "right.yaml")
                },
            ],
        ),
    ])
