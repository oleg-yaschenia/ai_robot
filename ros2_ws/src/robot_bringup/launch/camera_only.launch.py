from launch import LaunchDescription
from launch.actions import ExecuteProcess


def generate_launch_description():
    return LaunchDescription([
        ExecuteProcess(
            cmd=['bash', '/home/warxen/ai_robot/scripts/camera/save_one_stereo_pair.sh'],
            output='screen'
        ),
    ])
