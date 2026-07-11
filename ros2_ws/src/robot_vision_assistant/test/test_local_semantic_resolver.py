from robot_vision_assistant.assistant_request_contract import (
    build_request_plan,
    validate_request_plan,
)
from robot_vision_assistant.local_semantic_resolver import (
    LocalSemanticResolver,
    apply_semantic_resolution,
    deterministic_pre_resolve,
    normalize_candidate,
)


def _unknown_plan(query: str = "неизвестный запрос"):
    return build_request_plan(
        query=query,
        normalized_query=query,
        resolution={
            "status": "UNKNOWN",
            "source": "known_intent_resolver",
            "confidence": 1.0,
            "reason": "no_strict_full_match",
            "candidates": [],
            "missing_arguments": [],
        },
        plan={
            "request_type": "unknown",
            "intent": None,
            "arguments": {},
            "steps": [],
            "needs_scene": False,
            "needs_confirmation": False,
            "needs_clarification": False,
            "clarification_question": None,
            "risk_level": "none",
        },
    )


def test_general_chat_guard_blocks_hypothetical_action():
    result = deterministic_pre_resolve(
        "Если я скажу «поверни направо», что ты сделаешь?"
    )
    assert result is not None
    assert result["status"] == "MATCHED"
    assert result["intent"] == "general_chat"


def test_approach_user_guard_requires_distance():
    result = deterministic_pre_resolve("Подъедь ко мне")
    assert result is not None
    assert result["status"] == "AMBIGUOUS"
    assert result["intent"] == "base_move"
    assert result["arguments"] == {"target": "user"}
    assert result["missing_arguments"] == ["distance_m"]


def test_base_turn_keeps_only_direction():
    result = normalize_candidate(
        {
            "intent": "base_turn",
            "direction": "right",
            "state": "on",
            "object": "platform",
            "target": "right",
            "distance_m": 4,
        }
    )
    assert result["status"] == "MATCHED"
    assert result["arguments"] == {"direction": "right"}


def test_base_move_validates_required_slots():
    result = normalize_candidate(
        {
            "intent": "base_move",
            "direction": "forward",
            "state": "none",
            "object": "диван",
            "target": "диван",
            "distance_m": 1,
        }
    )
    assert result["status"] == "MATCHED"
    assert result["arguments"] == {
        "direction": "forward",
        "target": "диван",
        "distance_m": 1.0,
    }


def test_base_move_missing_distance_is_ambiguous():
    result = normalize_candidate(
        {
            "intent": "base_move",
            "direction": "none",
            "state": "none",
            "object": "",
            "target": "user",
            "distance_m": None,
        }
    )
    assert result["status"] == "AMBIGUOUS"
    assert result["missing_arguments"] == ["distance_m"]


def test_scene_path_safety_discards_invented_slots():
    result = normalize_candidate(
        {
            "intent": "scene_path_safety",
            "direction": "forward",
            "state": "on",
            "object": "стул",
            "target": "стена",
            "distance_m": 1.5,
        }
    )
    assert result["status"] == "MATCHED"
    assert result["arguments"] == {}


def test_unknown_capability_stays_unknown():
    result = normalize_candidate(
        {
            "intent": "none",
            "direction": "forward",
            "state": "none",
            "object": "robot",
            "target": "",
            "distance_m": None,
        }
    )
    assert result["status"] == "UNKNOWN"
    assert result["intent"] is None
    assert result["arguments"] == {}


def test_resolver_uses_injected_candidate_provider():
    def provider(_query):
        return (
            {
                "intent": "head_turn",
                "direction": "left",
                "state": "none",
                "object": "",
                "target": "",
                "distance_m": None,
            },
            {"elapsed_sec": 0.5},
        )

    resolver = LocalSemanticResolver(provider)
    result, metrics = resolver.resolve(
        "Посмотри в левую сторону"
    )
    assert result["status"] == "MATCHED"
    assert result["intent"] == "head_turn"
    assert result["arguments"] == {"direction": "left"}
    assert metrics["model_called"] is True


def test_apply_matched_semantic_plan_preserves_request_id():
    base = _unknown_plan(
        "Осмотрись и оцени, безопасно ли здесь проехать"
    )
    request_id = base["request_id"]

    result = apply_semantic_resolution(
        base,
        {
            "status": "MATCHED",
            "intent": "scene_path_safety",
            "arguments": {},
            "missing_arguments": [],
            "source": "qwen_semantic_candidate",
        },
    )

    assert result["request_id"] == request_id
    assert result["resolution"]["status"] == "MATCHED"
    assert result["plan"]["request_type"] == "observation"
    assert result["plan"]["needs_scene"] is True
    assert result["shadow_mode"] is True
    assert result["execution_allowed"] is False
    assert result["semantic_fallback"]["required"] is False
    assert validate_request_plan(result) == []


def test_apply_ambiguous_semantic_plan_builds_clarification():
    base = _unknown_plan("Подъедь ко мне")

    result = apply_semantic_resolution(
        base,
        {
            "status": "AMBIGUOUS",
            "intent": "base_move",
            "arguments": {"target": "user"},
            "missing_arguments": ["distance_m"],
            "source": "deterministic_guard",
        },
    )

    assert result["resolution"]["status"] == "AMBIGUOUS"
    assert result["resolution"]["candidates"] == ["base_move"]
    assert result["plan"]["request_type"] == "clarification"
    assert result["plan"]["needs_clarification"] is True
    assert "расстоянии" in result["plan"]["clarification_question"]
    assert result["execution_allowed"] is False
    assert validate_request_plan(result) == []


def test_apply_unknown_stops_fallback_loop():
    base = _unknown_plan("Сделай сальто")

    result = apply_semantic_resolution(
        base,
        {
            "status": "UNKNOWN",
            "intent": None,
            "arguments": {},
            "missing_arguments": [],
            "source": "qwen_semantic_candidate",
        },
    )

    assert result["resolution"]["status"] == "UNKNOWN"
    assert result["semantic_fallback"]["required"] is False
    assert result["semantic_fallback"]["attempted"] is True
    assert result["semantic_fallback"]["resolved"] is False
    assert result["execution_allowed"] is False
    assert validate_request_plan(result) == []
