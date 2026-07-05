from robot_vision_assistant.assistant_response_contract import (
    DEFAULT_CAPABILITIES,
    DEFAULT_RESPONSE_SYSTEM_PROMPT,
    render_response_context,
)
from robot_vision_assistant.response_orchestrator import (
    build_qwen_request,
    classify_response_route,
    fallback_source,
)


def test_general_text_uses_ai_text_mode():
    route = classify_response_route("Что такое одометрия?")
    assert route.route == "text"
    assert route.qwen_mode == "text"


def test_visual_overview_uses_new_image_session():
    route = classify_response_route("Что ты видишь?")
    assert route.route == "visual"
    assert route.qwen_mode == "image"
    assert route.new_visual_session is True


def test_count_uses_scene_facts_but_ai_phrasing():
    route = classify_response_route("Сколько человек ты видишь?")
    assert route.route == "scene_fact"
    assert route.qwen_mode == "scene"


def test_distance_uses_scene_facts_but_ai_phrasing():
    route = classify_response_route(
        "Как далеко находится ближайший человек?"
    )
    assert route.route == "scene_fact"
    assert route.qwen_mode == "scene"


def test_visual_followup_reuses_active_frame():
    route = classify_response_route(
        "А что находится справа?",
        visual_session_active=True,
    )
    assert route.route == "visual_followup"
    assert route.qwen_mode == "image"
    assert route.new_visual_session is False


def test_action_is_ai_phrased_but_not_executable():
    route = classify_response_route("Поверни направо")
    request = build_qwen_request(
        request_id="r1",
        query="Поверни направо",
        route=route,
    )
    assert route.is_action is True
    assert request["mode"] == "text"
    action = request["response_context"]["action_result"]
    assert action["state"] == "unavailable"
    assert action["execution_allowed"] is False


def test_emergency_uses_deterministic_safety_path():
    route = classify_response_route("Стоп")
    assert route.is_emergency is True
    assert route.route == "emergency"


def test_capabilities_are_provider_neutral_context():
    route = classify_response_route("Что ты умеешь?")
    request = build_qwen_request(
        request_id="r2",
        query="Что ты умеешь?",
        route=route,
        capabilities=DEFAULT_CAPABILITIES,
    )
    context = request["response_context"]
    assert context["capabilities"] == DEFAULT_CAPABILITIES
    assert context["provider"] == "local_qwen"
    assert context["rules"]["hardware_commands_forbidden"] is True


def test_hybrid_falls_back_to_legacy_only_on_ai_failure():
    assert fallback_source("hybrid", True) == "legacy"
    assert fallback_source("hybrid", False) == "deterministic_error"


def test_qwen_mode_never_silently_uses_legacy():
    assert fallback_source("qwen", True) == "deterministic_error"


def test_shared_prompt_forbids_false_action_claims_and_hardware_commands():
    assert "action_result.state" in DEFAULT_RESPONSE_SYSTEM_PROMPT
    assert "cmd_vel" in DEFAULT_RESPONSE_SYSTEM_PROMPT
    assert "UART" in DEFAULT_RESPONSE_SYSTEM_PROMPT


def test_context_renderer_is_explicit_and_machine_readable():
    rendered = render_response_context({"request_id": "r3"})
    assert rendered.startswith("ROBOT_RESPONSE_CONTEXT=")
    assert '"request_id":"r3"' in rendered
