from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import (
    EnvironmentVariable,
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node


def generate_launch_description():
    project_root = LaunchConfiguration("project_root")
    yolo_model_path = LaunchConfiguration("yolo_model_path")
    snapshots_dir = LaunchConfiguration("snapshots_dir")
    db_path = LaunchConfiguration("db_path")
    piper_bin = LaunchConfiguration("piper_bin")
    piper_model = LaunchConfiguration("piper_model")
    whisper_cli = LaunchConfiguration("whisper_cli")
    whisper_model = LaunchConfiguration("whisper_model")

    return LaunchDescription([
        DeclareLaunchArgument(
            "project_root",
            default_value=EnvironmentVariable(
                "AI_ROBOT_ROOT",
                default_value=PathJoinSubstitution([
                    EnvironmentVariable("HOME"),
                    "ai_robot",
                ]),
            ),
            description=(
                "Repository root; may also be set with AI_ROBOT_ROOT"
            ),
        ),
        DeclareLaunchArgument(
            "yolo_model_path",
            default_value=PathJoinSubstitution([
                project_root,
                "ros2_ws",
                "yolo11l.engine",
            ]),
        ),
        DeclareLaunchArgument(
            "snapshots_dir",
            default_value=PathJoinSubstitution([
                project_root,
                "data",
                "vision_assistant",
                "snapshots",
            ]),
        ),
        DeclareLaunchArgument(
            "db_path",
            default_value=PathJoinSubstitution([
                project_root,
                "data",
                "vision_assistant",
                "assistant_memory.sqlite",
            ]),
        ),
        DeclareLaunchArgument(
            "piper_bin",
            default_value=PathJoinSubstitution([
                project_root,
                "data",
                "tts",
                "piper",
                "piper",
                "piper",
            ]),
        ),
        DeclareLaunchArgument(
            "piper_model",
            default_value=PathJoinSubstitution([
                project_root,
                "data",
                "tts",
                "piper",
                "ru_RU-ruslan-medium.onnx",
            ]),
        ),
        DeclareLaunchArgument(
            "whisper_cli",
            default_value=PathJoinSubstitution([
                project_root,
                "tools",
                "whisper.cpp",
                "build",
                "bin",
                "whisper-cli",
            ]),
        ),
        DeclareLaunchArgument(
            "whisper_model",
            default_value=PathJoinSubstitution([
                project_root,
                "tools",
                "whisper.cpp",
                "models",
                "ggml-base.bin",
            ]),
        ),

        Node(
            package="robot_vision_assistant",
            executable="yolo_perception_node",
            name="yolo_perception_node",
            output="screen",
            parameters=[
                {"image_topic": "/camera/right/image_rect"},
                {"model_path": yolo_model_path},
                {"device": "0"},
                {"imgsz": 640},
                {"inference_conf_threshold": 0.10},
                {"iou_threshold": 0.45},
                {"analysis_period_sec": 0.25},
                {"max_det": 100},
                {"person_conf_threshold": 0.35},
                {"pet_conf_threshold": 0.25},
                {"cup_conf_threshold": 0.20},
                {"chair_conf_threshold": 0.45},
                {"default_conf_threshold": 0.35},
                {"tv_conf_threshold": 0.55},
                {"remote_conf_threshold": 0.55},
                {"cell_phone_conf_threshold": 0.40},
                {"mouse_conf_threshold": 0.40},
                {"keyboard_conf_threshold": 0.30},
                {"laptop_conf_threshold": 0.30},
                {"track_iou_threshold": 0.15},
                {"track_center_distance_factor": 2.5},
                {"track_center_distance_min_px": 80.0},
                {"velocity_alpha": 0.65},
                {"duplicate_iou_threshold": 0.45},
                {"duplicate_containment_threshold": 0.75},
                {"default_confirm_hits": 3},
                {"motion_extra_hits": 2},
                {"max_missed_frames": 4},
                {"noisy_max_missed_frames": 2},
                {"immediate_conf_threshold": 0.85},
                {"motion_threshold": 0.035},
                {"motion_diff_threshold": 22},
                {"motion_conf_boost": 0.10},
                {"horizontal_left_boundary": 0.38},
                {"horizontal_right_boundary": 0.62},
                {"vertical_upper_boundary": 0.33},
                {"vertical_lower_boundary": 0.67},
                {"depth_enabled": True},
                {"depth_topic": "/disparity"},
                {"depth_max_age_sec": 0.35},
                {"depth_min_m": 0.35},
                {"depth_max_m": 8.0},
                {"depth_roi_scale_x": 0.50},
                {"depth_roi_scale_y": 0.50},
                {"depth_min_samples": 40},
                {"depth_min_valid_ratio": 0.08},
                {"depth_max_relative_spread": 0.35},
                {"depth_max_absolute_spread_m": 0.35},
                {"depth_saturation_margin_px": 1.0},
            ],
        ),

        Node(
            package="robot_vision_assistant",
            executable="scene_interpreter_node",
            name="scene_interpreter_node",
            output="screen",
            parameters=[
                {"input_topic": "/perception/state_json"},
                {"output_topic": "/scene/interpreted_json"},
                {"summary_topic": "/scene/interpreted_summary"},
                {"change_iou_threshold": 0.35},
            ],
        ),

        Node(
            package="robot_vision_assistant",
            executable="vision_assistant_node",
            name="vision_assistant_node",
            output="screen",
            parameters=[
                {"image_topic": "/camera/right/image_rect"},
                {"mode": "local_only"},
                {"allow_cloud": False},
                {"allow_realtime": False},
                {"snapshots_dir": snapshots_dir},
                {"db_path": db_path},
            ],
        ),

        Node(
            package="robot_vision_assistant",
            executable="tts_node",
            name="tts_node",
            output="screen",
            parameters=[
                {"answer_topic": "/vision_assistant/answer"},
                {"status_topic": "/voice_tts/status"},
                {"enabled": True},
                {"piper_bin": piper_bin},
                {"model_path": piper_model},
                {"audio_player": "aplay"},
                {"audio_player_args": ["-q"]},
                {"tmp_dir": "/tmp/robot_tts"},
            ],
        ),

        Node(
            package="robot_vision_assistant",
            executable="asr_node",
            name="asr_node",
            output="screen",
            parameters=[
                {"listen_topic": "/voice_asr/listen"},
                {"query_topic": "/vision_assistant/query"},
                {"transcript_topic": "/voice_asr/transcript"},
                {"status_topic": "/voice_asr/status"},
                {"record_device": "hw:1,0"},
                {"record_sample_rate": 48000},
                {"record_channels": 2},
                {"record_format": "S32_LE"},
                {"whisper_cli": whisper_cli},
                {"model_path": whisper_model},
                {"language": "ru"},
                {"threads": 4},
                {"processors": 1},
                {"tmp_dir": "/tmp/robot_asr"},
                {"vad_mode": 2},
                {"frame_ms": 30},
                {"pre_roll_ms": 450},
                {"speech_timeout_sec": 8.0},
                {"max_utterance_sec": 8.0},
                {"start_window_ms": 240},
                {"start_trigger_ratio": 0.60},
                {"end_window_ms": 900},
                {"end_trigger_ratio": 0.90},
                {"min_utterance_ms": 400},
            ],
        ),

        Node(
            package="robot_vision_assistant",
            executable="voice_manager_node",
            name="voice_manager_node",
            output="screen",
            parameters=[
                {"start_topic": "/voice/start"},
                {"state_topic": "/voice/state"},
                {"asr_listen_topic": "/voice_asr/listen"},
                {"asr_status_topic": "/voice_asr/status"},
                {"asr_transcript_topic": "/voice_asr/transcript"},
                {"answer_topic": "/vision_assistant/answer"},
                {"tts_status_topic": "/voice_tts/status"},
            ],
        ),
    ])
