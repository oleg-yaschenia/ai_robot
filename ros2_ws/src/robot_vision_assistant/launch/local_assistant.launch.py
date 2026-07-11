from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import (
    EnvironmentVariable,
    LaunchConfiguration,
    PathJoinSubstitution,
    PythonExpression,
)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    project_root = LaunchConfiguration("project_root")
    yolo_model_path = LaunchConfiguration("yolo_model_path")
    yolo_fallback_model_path = LaunchConfiguration(
        "yolo_fallback_model_path"
    )
    yolo_runtime = LaunchConfiguration("yolo_runtime")
    scene_input_topic = LaunchConfiguration("scene_input_topic")
    snapshots_dir = LaunchConfiguration("snapshots_dir")
    db_path = LaunchConfiguration("db_path")
    piper_bin = LaunchConfiguration("piper_bin")
    piper_model = LaunchConfiguration("piper_model")
    whisper_cli = LaunchConfiguration("whisper_cli")
    whisper_model = LaunchConfiguration("whisper_model")
    enable_qwen_vl_shadow = LaunchConfiguration("enable_qwen_vl_shadow")
    enable_local_semantic_resolver = LaunchConfiguration(
        "enable_local_semantic_resolver"
    )
    semantic_resolver_timeout_sec = LaunchConfiguration(
        "semantic_resolver_timeout_sec"
    )
    semantic_resolver_max_tokens = LaunchConfiguration(
        "semantic_resolver_max_tokens"
    )
    qwen_server_url = LaunchConfiguration("qwen_server_url")
    qwen_query_topic = LaunchConfiguration("qwen_query_topic")
    qwen_image_topic = LaunchConfiguration("qwen_image_topic")
    qwen_warmup_text = LaunchConfiguration("qwen_warmup_text")
    qwen_warmup_visual = LaunchConfiguration("qwen_warmup_visual")
    qwen_active_output = LaunchConfiguration("qwen_active_output")
    qwen_answer_topic = LaunchConfiguration("qwen_answer_topic")
    legacy_query_topic = LaunchConfiguration("legacy_query_topic")
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
    response_qwen_query_topic = LaunchConfiguration(
        "response_qwen_query_topic"
    )
    response_legacy_query_topic = LaunchConfiguration(
        "response_legacy_query_topic"
    )
    response_qwen_candidate_topic = LaunchConfiguration(
        "response_qwen_candidate_topic"
    )
    response_legacy_candidate_topic = LaunchConfiguration(
        "response_legacy_candidate_topic"
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

    enabled_test = "'.lower() in ('true','1','yes','on')"
    qwen_enabled_condition = PythonExpression([
        "'",
        enable_qwen_vl_shadow,
        enabled_test,
        " or '",
        enable_response_orchestrator,
        enabled_test,
    ])
    qwen_runtime_query_topic = PythonExpression([
        "'",
        response_qwen_query_topic,
        "' if '",
        enable_response_orchestrator,
        enabled_test,
        " else '",
        qwen_query_topic,
        "'",
    ])
    qwen_runtime_answer_topic = PythonExpression([
        "'/qwen_vl/orchestrator_answer' if '",
        enable_response_orchestrator,
        enabled_test,
        " else '",
        qwen_answer_topic,
        "'",
    ])
    qwen_runtime_active_output = PythonExpression([
        "'false' if '",
        enable_response_orchestrator,
        enabled_test,
        " else '",
        qwen_active_output,
        "'",
    ])
    legacy_runtime_query_topic = PythonExpression([
        "'",
        response_legacy_query_topic,
        "' if '",
        enable_response_orchestrator,
        enabled_test,
        " else '",
        legacy_query_topic,
        "'",
    ])
    legacy_runtime_answer_topic = PythonExpression([
        "'/vision_assistant/rule_answer' if '",
        enable_response_orchestrator,
        enabled_test,
        " else '",
        legacy_answer_topic,
        "'",
    ])
    legacy_runtime_record_interactions = PythonExpression([
        "'false' if '",
        enable_response_orchestrator,
        enabled_test,
        " else '",
        legacy_record_interactions,
        "'",
    ])
    yolo_ultralytics_condition = IfCondition(PythonExpression([
        "'",
        yolo_runtime,
        "'.lower() == 'ultralytics'",
    ]))
    yolo_tensorrt_condition = IfCondition(PythonExpression([
        "'",
        yolo_runtime,
        "'.lower() == 'tensorrt'",
    ]))

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
            "yolo_fallback_model_path",
            default_value=PathJoinSubstitution([
                project_root,
                "ros2_ws",
                "yolo11l.pt",
            ]),
        ),
        DeclareLaunchArgument(
            "yolo_runtime",
            default_value="tensorrt",
            description=(
                "YOLO execution backend: tensorrt for the production "
                "low-memory runtime or ultralytics for rollback. Both "
                "publish the same perception contract."
            ),
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
            "enable_local_semantic_resolver",
            default_value="false",
            description=(
                "Enable Local Semantic Resolver v1 in shadow mode. "
                "It processes only UNKNOWN request plans, uses the "
                "existing llama-server, and cannot execute actions."
            ),
        ),
        DeclareLaunchArgument(
            "semantic_resolver_timeout_sec",
            default_value="30.0",
        ),
        DeclareLaunchArgument(
            "semantic_resolver_max_tokens",
            default_value="96",
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
                "/vision_assistant/query to observe real user queries."
            ),
        ),
        DeclareLaunchArgument(
            "qwen_image_topic",
            default_value="/camera/right/image_rect",
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
                "Enable Response Orchestrator v1. It becomes the only "
                "publisher of /vision_assistant/answer; normal responses "
                "are generated by Qwen using text, image or verified scene "
                "context. Legacy remains timeout/error fallback only."
            ),
        ),
        DeclareLaunchArgument(
            "response_mode",
            default_value="hybrid",
            description="Response policy: legacy, hybrid or qwen.",
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
            "response_qwen_query_topic",
            default_value="/assistant/response/qwen_request_json",
        ),
        DeclareLaunchArgument(
            "response_legacy_query_topic",
            default_value="/assistant/response/legacy_request_json",
        ),
        DeclareLaunchArgument(
            "response_qwen_candidate_topic",
            default_value="/assistant/response/qwen_candidate_json",
        ),
        DeclareLaunchArgument(
            "response_legacy_candidate_topic",
            default_value="/assistant/response/legacy_candidate_json",
        ),
        DeclareLaunchArgument(
            "qwen_active_output",
            default_value="false",
            description=(
                "When true, Qwen is allowed to publish the selected "
                "assistant answer topic. This does not grant any physical "
                "action or executor access."
            ),
        ),
        DeclareLaunchArgument(
            "qwen_answer_topic",
            default_value="/qwen_vl/shadow_answer",
            description=(
                "Qwen full-answer output. Set to "
                "/vision_assistant/answer only for controlled active tests."
            ),
        ),
        DeclareLaunchArgument(
            "legacy_query_topic",
            default_value="/vision_assistant/query",
        ),
        DeclareLaunchArgument(
            "legacy_answer_topic",
            default_value="/vision_assistant/answer",
            description=(
                "Rule-based assistant output. For Qwen active tests set "
                "this to /vision_assistant/rule_answer to avoid duplicates."
            ),
        ),
        DeclareLaunchArgument(
            "legacy_record_interactions",
            default_value="true",
            description=(
                "Record rule-based answers in the legacy memory store. "
                "Set false while Qwen owns the active answer topic."
            ),
        ),
        DeclareLaunchArgument(
            "tts_enabled",
            default_value="true",
            description="Enable or disable local TTS output.",
        ),
        DeclareLaunchArgument(
            "asr_channel_strategy",
            default_value="best_snr",
            description=(
                "ASR stereo channel selection: best_snr, left or right."
            ),
        ),
        DeclareLaunchArgument(
            "asr_input_gain_db",
            default_value="18.0",
            description="Software gain applied before VAD and Whisper.",
        ),
        DeclareLaunchArgument(
            "asr_limiter_peak_dbfs",
            default_value="-6.0",
            description="Per-frame peak limiter target after ASR gain.",
        ),
        DeclareLaunchArgument(
            "asr_start_energy_dbfs",
            default_value="-47.0",
            description=(
                "Minimum gained frame RMS used to start far-field VAD."
            ),
        ),
        DeclareLaunchArgument(
            "asr_vad_mode",
            default_value="0",
            description=(
                "WebRTC VAD aggressiveness. 0 is most sensitive; "
                "3 is most aggressive."
            ),
        ),
        DeclareLaunchArgument(
            "asr_start_trigger_ratio",
            default_value="0.30",
            description=(
                "Required voiced-frame ratio in the ASR start window."
            ),
        ),

        DeclareLaunchArgument(
            "asr_noise_calibration_ms",
            default_value="600",
            description="Startup/noise calibration before ASR triggering.",
        ),
        DeclareLaunchArgument(
            "asr_start_snr_margin_db",
            default_value="4.0",
            description="Required RMS margin above calibrated noise to start.",
        ),
        DeclareLaunchArgument(
            "asr_end_snr_margin_db",
            default_value="2.0",
            description="RMS margin above noise used while speech is active.",
        ),
        DeclareLaunchArgument(
            "asr_speech_confirm_ms",
            default_value="240",
            description="Minimum confirmed speech activity before accepting audio.",
        ),
        DeclareLaunchArgument(
            "asr_end_grace_ms",
            default_value="1200",
            description="Minimum post-trigger time before silence may stop capture.",
        ),
        DeclareLaunchArgument(
            "asr_debug_keep_wav",
            default_value="false",
            description="Preserve recorded ASR WAV files for diagnostics.",
        ),

        Node(
            package="robot_vision_assistant",
            executable="yolo_perception_node",
            name="yolo_perception_node",
            output="screen",
            condition=yolo_ultralytics_condition,
            parameters=[
                {"image_topic": "/camera/right/image_rect"},
                {"model_path": yolo_model_path},
                {"fallback_model_path": yolo_fallback_model_path},
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
            executable="yolo_tensorrt_node",
            name="yolo_perception_node",
            output="screen",
            condition=yolo_tensorrt_condition,
            parameters=[
                {"image_topic": "/camera/right/image_rect_yolo"},
                {"depth_topic": "/disparity"},
                {"model_path": yolo_model_path},
                {"fallback_model_path": yolo_model_path},
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
            executable="perception_entity_adapter_node",
            name="perception_entity_adapter_node",
            output="screen",
            parameters=[
                {"input_topic": "/perception/state_json"},
                {"output_topic": "/perception/entities_json"},
            ],
        ),

        Node(
            package="robot_vision_assistant",
            executable="scene_interpreter_node",
            name="scene_interpreter_node",
            output="screen",
            parameters=[
                {"input_topic": scene_input_topic},
                {"output_topic": "/scene/interpreted_json"},
                {"summary_topic": "/scene/interpreted_summary"},
                {"change_iou_threshold": 0.35},
            ],
        ),

        Node(
            package="robot_vision_assistant",
            executable="assistant_router_node",
            name="assistant_router_node",
            output="screen",
            parameters=[
                {"query_topic": "/vision_assistant/query"},
                {"scene_topic": "/scene/interpreted_json"},
                {"decision_topic": "/assistant/router/decision_json"},
                {"max_scene_age_sec": 2.0},
            ],
        ),

        Node(
            package="robot_vision_assistant",
            executable="assistant_core_shadow_node",
            name="assistant_core_shadow_node",
            output="screen",
            parameters=[
                {"query_topic": "/vision_assistant/query"},
                {"plan_topic": "/assistant/core/request_plan_json"},
                {"clarification_topic": "/assistant/clarification/request_json"},
                {"max_clarification_attempts": 2},
            ],
        ),

        Node(
            package="robot_vision_assistant",
            executable="local_semantic_resolver_node",
            name="local_semantic_resolver_node",
            output="screen",
            condition=IfCondition(
                enable_local_semantic_resolver
            ),
            parameters=[
                {
                    "input_topic":
                    "/assistant/core/request_plan_json"
                },
                {
                    "output_topic":
                    "/assistant/semantic/request_plan_json"
                },
                {
                    "status_topic":
                    "/assistant/semantic/status_json"
                },
                {"server_url": qwen_server_url},
                {
                    "request_timeout_sec": ParameterValue(
                        semantic_resolver_timeout_sec,
                        value_type=float,
                    )
                },
                {
                    "max_tokens": ParameterValue(
                        semantic_resolver_max_tokens,
                        value_type=int,
                    )
                },
                {"queue_size": 2},
                {"dedupe_history": 256},
            ],
        ),

        Node(
            package="robot_vision_assistant",
            executable="response_orchestrator_node",
            name="response_orchestrator_node",
            output="screen",
            condition=IfCondition(enable_response_orchestrator),
            parameters=[
                {"response_mode": response_mode},
                {"query_topic": "/vision_assistant/query"},
                {"qwen_query_topic": response_qwen_query_topic},
                {"legacy_query_topic": response_legacy_query_topic},
                {"qwen_candidate_topic": response_qwen_candidate_topic},
                {"legacy_candidate_topic": response_legacy_candidate_topic},
                {"answer_topic": "/vision_assistant/answer"},
                {"trace_topic": "/assistant/response/trace_json"},
                {"status_topic": "/assistant/response/status_json"},
                {"action_request_topic": "/assistant/action/request_json"},
                {
                    "qwen_timeout_sec": ParameterValue(
                        response_qwen_timeout_sec, value_type=float
                    )
                },
                {
                    "legacy_timeout_sec": ParameterValue(
                        response_legacy_timeout_sec, value_type=float
                    )
                },
                {"visual_session_ttl_sec": 30.0},
            ],
        ),

        Node(
            package="robot_vision_assistant",
            executable="qwen_vl_runtime_node",
            name="qwen_vl_runtime_node",
            output="screen",
            condition=IfCondition(qwen_enabled_condition),
            parameters=[
                {"server_url": qwen_server_url},
                {"image_topic": qwen_image_topic},
                {"scene_topic": "/scene/interpreted_json"},
                {"query_topic": qwen_runtime_query_topic},
                {"answer_topic": qwen_runtime_answer_topic},
                {"candidate_topic": response_qwen_candidate_topic},
                {"sentence_topic": "/qwen_vl/shadow_sentence"},
                {"status_topic": "/qwen_vl/status_json"},
                {"metrics_topic": "/qwen_vl/metrics_json"},
                {"warmup_text": ParameterValue(qwen_warmup_text, value_type=bool)},
                {"warmup_visual": ParameterValue(qwen_warmup_visual, value_type=bool)},
                {
                    "active_output": ParameterValue(
                        qwen_runtime_active_output, value_type=bool
                    )
                },
                {"visual_session_ttl_sec": 30.0},
                {"max_visual_turns": 4},
                {"jpeg_quality": 90},
                {"default_max_tokens": 220},
            ],
        ),

        Node(
            package="robot_vision_assistant",
            executable="vision_assistant_node",
            name="vision_assistant_node",
            output="screen",
            parameters=[
                {"image_topic": "/vision_assistant/unused_image"},
                {"query_topic": legacy_runtime_query_topic},
                {"answer_topic": legacy_runtime_answer_topic},
                {"candidate_topic": response_legacy_candidate_topic},
                {"status_topic": "/vision_assistant/status"},
                {
                    "record_interactions": ParameterValue(
                        legacy_runtime_record_interactions,
                        value_type=bool,
                    )
                },
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
                {"enabled": ParameterValue(tts_enabled, value_type=bool)},
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
                {
                    "channel_strategy": ParameterValue(
                        asr_channel_strategy, value_type=str
                    )
                },
                {
                    "input_gain_db": ParameterValue(
                        asr_input_gain_db, value_type=float
                    )
                },
                {
                    "limiter_peak_dbfs": ParameterValue(
                        asr_limiter_peak_dbfs, value_type=float
                    )
                },
                {
                    "start_energy_dbfs": ParameterValue(
                        asr_start_energy_dbfs, value_type=float
                    )
                },
                {
                    "vad_mode": ParameterValue(
                        asr_vad_mode, value_type=int
                    )
                },
                {"frame_ms": 30},
                {"pre_roll_ms": 900},
                {"speech_timeout_sec": 10.0},
                {"max_utterance_sec": 8.0},
                {"start_window_ms": 300},
                {
                    "start_trigger_ratio": ParameterValue(
                        asr_start_trigger_ratio, value_type=float
                    )
                },
                {"end_window_ms": 1200},
                {"end_trigger_ratio": 0.85},
                {"min_utterance_ms": 500},
                {
                    "noise_calibration_ms": ParameterValue(
                        asr_noise_calibration_ms, value_type=int
                    )
                },
                {
                    "start_snr_margin_db": ParameterValue(
                        asr_start_snr_margin_db, value_type=float
                    )
                },
                {
                    "end_snr_margin_db": ParameterValue(
                        asr_end_snr_margin_db, value_type=float
                    )
                },
                {
                    "speech_confirm_ms": ParameterValue(
                        asr_speech_confirm_ms, value_type=int
                    )
                },
                {
                    "end_grace_ms": ParameterValue(
                        asr_end_grace_ms, value_type=int
                    )
                },
                {
                    "debug_keep_wav": ParameterValue(
                        asr_debug_keep_wav, value_type=bool
                    )
                },
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
