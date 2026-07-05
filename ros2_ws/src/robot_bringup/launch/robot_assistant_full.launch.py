from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    EnvironmentVariable,
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    enable_head = LaunchConfiguration("enable_head")
    enable_esp32 = LaunchConfiguration("enable_esp32")
    serial_port = LaunchConfiguration("serial_port")
    baud_rate = LaunchConfiguration("baud_rate")
    project_root = LaunchConfiguration("project_root")
    yolo_model_path = LaunchConfiguration("yolo_model_path")
    yolo_fallback_model_path = LaunchConfiguration(
        "yolo_fallback_model_path"
    )
    yolo_runtime = LaunchConfiguration("yolo_runtime")
    scene_input_topic = LaunchConfiguration("scene_input_topic")
    enable_qwen_vl_shadow = LaunchConfiguration("enable_qwen_vl_shadow")
    qwen_server_url = LaunchConfiguration("qwen_server_url")
    qwen_query_topic = LaunchConfiguration("qwen_query_topic")
    qwen_warmup_text = LaunchConfiguration("qwen_warmup_text")
    qwen_warmup_visual = LaunchConfiguration("qwen_warmup_visual")
    qwen_active_output = LaunchConfiguration("qwen_active_output")
    qwen_answer_topic = LaunchConfiguration("qwen_answer_topic")
    legacy_answer_topic = LaunchConfiguration("legacy_answer_topic")
    enable_response_orchestrator = LaunchConfiguration(
        "enable_response_orchestrator"
    )
    response_mode = LaunchConfiguration("response_mode")
    response_qwen_timeout_sec = LaunchConfiguration(
        "response_qwen_timeout_sec"
    )
    response_legacy_timeout_sec = LaunchConfiguration(
        "response_legacy_timeout_sec"
    )
    legacy_record_interactions = LaunchConfiguration(
        "legacy_record_interactions"
    )
    tts_enabled = LaunchConfiguration("tts_enabled")
    asr_channel_strategy = LaunchConfiguration("asr_channel_strategy")
    asr_input_gain_db = LaunchConfiguration("asr_input_gain_db")
    asr_limiter_peak_dbfs = LaunchConfiguration("asr_limiter_peak_dbfs")
    asr_start_energy_dbfs = LaunchConfiguration("asr_start_energy_dbfs")
    asr_vad_mode = LaunchConfiguration("asr_vad_mode")
    asr_start_trigger_ratio = LaunchConfiguration(
        "asr_start_trigger_ratio"
    )
    asr_noise_calibration_ms = LaunchConfiguration(
        "asr_noise_calibration_ms"
    )
    asr_start_snr_margin_db = LaunchConfiguration(
        "asr_start_snr_margin_db"
    )
    asr_end_snr_margin_db = LaunchConfiguration(
        "asr_end_snr_margin_db"
    )
    asr_speech_confirm_ms = LaunchConfiguration(
        "asr_speech_confirm_ms"
    )
    asr_end_grace_ms = LaunchConfiguration("asr_end_grace_ms")
    asr_debug_keep_wav = LaunchConfiguration("asr_debug_keep_wav")

    robot_base = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("robot_bringup"),
                "launch",
                "robot_base.launch.py",
            ])
        ),
        launch_arguments={
            "enable_esp32": enable_esp32,
            "serial_port": serial_port,
            "baud_rate": baud_rate,
        }.items(),
    )

    local_assistant = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("robot_vision_assistant"),
                "launch",
                "local_assistant.launch.py",
            ])
        ),
        launch_arguments={
            "project_root": project_root,
            "yolo_model_path": yolo_model_path,
            "yolo_fallback_model_path": yolo_fallback_model_path,
            "yolo_runtime": yolo_runtime,
            "scene_input_topic": scene_input_topic,
            "enable_qwen_vl_shadow": enable_qwen_vl_shadow,
            "qwen_server_url": qwen_server_url,
            "qwen_query_topic": qwen_query_topic,
            "qwen_warmup_text": qwen_warmup_text,
            "qwen_warmup_visual": qwen_warmup_visual,
            "qwen_active_output": qwen_active_output,
            "qwen_answer_topic": qwen_answer_topic,
            "legacy_answer_topic": legacy_answer_topic,
            "enable_response_orchestrator": enable_response_orchestrator,
            "response_mode": response_mode,
            "response_qwen_timeout_sec": response_qwen_timeout_sec,
            "response_legacy_timeout_sec": response_legacy_timeout_sec,
            "legacy_record_interactions": legacy_record_interactions,
            "tts_enabled": tts_enabled,
            "asr_channel_strategy": asr_channel_strategy,
            "asr_input_gain_db": asr_input_gain_db,
            "asr_limiter_peak_dbfs": asr_limiter_peak_dbfs,
            "asr_start_energy_dbfs": asr_start_energy_dbfs,
            "asr_vad_mode": asr_vad_mode,
            "asr_start_trigger_ratio": asr_start_trigger_ratio,
            "asr_noise_calibration_ms": asr_noise_calibration_ms,
            "asr_start_snr_margin_db": asr_start_snr_margin_db,
            "asr_end_snr_margin_db": asr_end_snr_margin_db,
            "asr_speech_confirm_ms": asr_speech_confirm_ms,
            "asr_end_grace_ms": asr_end_grace_ms,
            "asr_debug_keep_wav": asr_debug_keep_wav,
        }.items(),
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
        condition=IfCondition(enable_head),
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "enable_head",
            default_value="true",
            description="Enable head behavior and voice LED bridge",
        ),
        DeclareLaunchArgument(
            "enable_esp32",
            default_value="true",
            description="Start the single ESP32 UART owner",
        ),
        DeclareLaunchArgument("serial_port", default_value="/dev/ttyTHS1"),
        DeclareLaunchArgument("baud_rate", default_value="115200"),
        DeclareLaunchArgument(
            "project_root",
            default_value=EnvironmentVariable(
                "AI_ROBOT_ROOT",
                default_value=PathJoinSubstitution([
                    EnvironmentVariable("HOME"),
                    "ai_robot",
                ]),
            ),
            description="Repository root; may also be set with AI_ROBOT_ROOT",
        ),
        DeclareLaunchArgument(
            "yolo_model_path",
            default_value=PathJoinSubstitution([
                project_root,
                "ros2_ws",
                "yolo11l.engine",
            ]),
            description="Preferred YOLO model path.",
        ),
        DeclareLaunchArgument(
            "yolo_fallback_model_path",
            default_value=PathJoinSubstitution([
                project_root,
                "ros2_ws",
                "yolo11l.pt",
            ]),
            description=(
                "Fallback YOLO model used when the preferred model "
                "cannot be loaded or warmed up."
            ),
        ),
        DeclareLaunchArgument(
            "yolo_runtime",
            default_value="tensorrt",
            description=(
                "YOLO execution backend: ultralytics or direct tensorrt. "
                "Direct TensorRT is the production default; use ultralytics for rollback."
            ),
        ),
        DeclareLaunchArgument(
            "scene_input_topic",
            default_value="/perception/entities_json",
            description=(
                "Scene interpreter input. Use "
                "/perception/state_json for legacy rollback."
            ),
        ),
        DeclareLaunchArgument(
            "enable_qwen_vl_shadow",
            default_value="false",
            description=(
                "Start the shadow-only persistent Qwen3-VL runtime node."
            ),
        ),
        DeclareLaunchArgument(
            "qwen_server_url",
            default_value="http://127.0.0.1:8080",
        ),
        DeclareLaunchArgument(
            "qwen_query_topic",
            default_value="/qwen_vl/query_json",
            description=(
                "Input topic for Qwen shadow requests. Use "
                "/vision_assistant/query to observe the real assistant path "
                "without changing active answers or TTS."
            ),
        ),
        DeclareLaunchArgument(
            "qwen_warmup_text",
            default_value="true",
        ),
        DeclareLaunchArgument(
            "qwen_warmup_visual",
            default_value="true",
        ),
        DeclareLaunchArgument(
            "enable_response_orchestrator",
            default_value="false",
            description=(
                "Enable AI-first Response Orchestrator v1. Qwen generates "
                "normal spoken responses from text, image or verified scene "
                "context; legacy is fallback only."
            ),
        ),
        DeclareLaunchArgument(
            "response_mode",
            default_value="hybrid",
        ),
        DeclareLaunchArgument(
            "response_qwen_timeout_sec",
            default_value="20.0",
        ),
        DeclareLaunchArgument(
            "response_legacy_timeout_sec",
            default_value="3.0",
        ),
        DeclareLaunchArgument(
            "qwen_active_output",
            default_value="false",
            description=(
                "Allow Qwen to publish the selected assistant answer "
                "topic. Physical action execution remains unavailable."
            ),
        ),
        DeclareLaunchArgument(
            "qwen_answer_topic",
            default_value="/qwen_vl/shadow_answer",
        ),
        DeclareLaunchArgument(
            "legacy_answer_topic",
            default_value="/vision_assistant/answer",
        ),
        DeclareLaunchArgument(
            "legacy_record_interactions",
            default_value="true",
        ),
        DeclareLaunchArgument(
            "tts_enabled",
            default_value="true",
        ),
        DeclareLaunchArgument(
            "asr_channel_strategy",
            default_value="best_snr",
        ),
        DeclareLaunchArgument(
            "asr_input_gain_db",
            default_value="18.0",
        ),
        DeclareLaunchArgument(
            "asr_limiter_peak_dbfs",
            default_value="-6.0",
        ),
        DeclareLaunchArgument(
            "asr_start_energy_dbfs",
            default_value="-47.0",
        ),
        DeclareLaunchArgument(
            "asr_vad_mode",
            default_value="0",
        ),
        DeclareLaunchArgument(
            "asr_start_trigger_ratio",
            default_value="0.30",
        ),
        DeclareLaunchArgument(
            "asr_noise_calibration_ms",
            default_value="600",
        ),
        DeclareLaunchArgument(
            "asr_start_snr_margin_db",
            default_value="4.0",
        ),
        DeclareLaunchArgument(
            "asr_end_snr_margin_db",
            default_value="2.0",
        ),
        DeclareLaunchArgument(
            "asr_speech_confirm_ms",
            default_value="240",
        ),
        DeclareLaunchArgument(
            "asr_end_grace_ms",
            default_value="1200",
        ),
        DeclareLaunchArgument(
            "asr_debug_keep_wav",
            default_value="false",
        ),
        robot_base,
        local_assistant,
        voice_led_bridge,
        head_behavior,
    ])
