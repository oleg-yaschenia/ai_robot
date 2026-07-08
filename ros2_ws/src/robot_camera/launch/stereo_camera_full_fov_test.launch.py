from launch import LaunchDescription
from launch.actions import UnsetEnvironmentVariable
from launch_ros.actions import Node


def generate_launch_description():
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
                {"left_sensor_id": 1},
                {"right_sensor_id": 0},

                {"sensor_mode": 3},
                {"sensor_width": 1640},
                {"sensor_height": 1232},

                {"width": 1024},
                {"height": 768},
                {"fps": 10},

                {"interpolation_method": 5},
                {"ee_mode": 1},
                {"ee_strength": -1.0},

                # Старые YAML относятся к 1280x720.
                {"publish_camera_info": True},
                {
                    "left_camera_info_yaml":
                    "/home/warxen/ai_robot/calib/stereo_1024x768/left.yaml"
                },
                {
                    "right_camera_info_yaml":
                    "/home/warxen/ai_robot/calib/stereo_1024x768/right.yaml"
                },

                {
                    "left_frame_id":
                    "camera_left_optical_frame"
                },
                {
                    "right_frame_id":
                    "camera_right_optical_frame"
                },
            ],
        ),
    ])
