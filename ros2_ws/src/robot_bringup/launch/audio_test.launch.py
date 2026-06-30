from pathlib import Path
import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess


def generate_launch_description():
    project_root = Path(
        os.environ.get(
            "AI_ROBOT_ROOT",
            str(Path.home() / "ai_robot"),
        )
    ).expanduser()
    return LaunchDescription([
        ExecuteProcess(
            cmd=["bash", str(project_root / "scripts" / "audio" / "mic_record_test.sh")],
            output='screen'
        ),
    ])
