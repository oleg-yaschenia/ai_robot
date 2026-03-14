from launch import LaunchDescription
from launch.actions import ExecuteProcess


def generate_launch_description():
    return LaunchDescription([
        ExecuteProcess(
            cmd=['bash', '/home/warxen/ai_robot/scripts/camera/stereo_capture_test.sh'],
            output='screen'
        ),
        ExecuteProcess(
            cmd=['bash', '/home/warxen/ai_robot/scripts/audio/mic_record_test.sh'],
            output='screen'
        ),
    ])
