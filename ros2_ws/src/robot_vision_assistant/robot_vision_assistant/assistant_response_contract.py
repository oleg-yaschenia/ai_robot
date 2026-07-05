#!/usr/bin/env python3
"""Shared response-generation contract for local and future cloud AI providers.

The contract deliberately separates verified robot state and action results from
natural-language generation. A model may phrase a response, but it may not
invent sensor values, claim execution without executor confirmation, or emit
hardware commands.
"""

from __future__ import annotations

import json
from typing import Any, Dict


SCHEMA_NAME = "robot_assistant_response_context"
SCHEMA_VERSION = 1

DEFAULT_RESPONSE_SYSTEM_PROMPT = (
    "Ты голосовой интерфейс домашнего робота. Отвечай по-русски, естественно, "
    "кратко и по существу. Используй только факты и результаты, переданные в "
    "контексте запроса. Значения Scene Interpreter и датчиков являются "
    "единственным источником для количества, расстояния, положения и "
    "безопасности. Никогда не выдумывай измерения. Никогда не утверждай, что "
    "физическое действие началось или завершилось, если action_result.state "
    "не равен accepted, running или completed. При blocked, unavailable или "
    "rejected спокойно объясни, что действие не выполняется. Не создавай "
    "cmd_vel, UART-пакеты, PWM, траектории или команды оборудованию. Верни "
    "только текст, который робот должен произнести, без JSON и служебных "
    "комментариев."
)

DEFAULT_CAPABILITIES: Dict[str, Any] = {
    "available": [
        "answer_general_questions",
        "describe_current_image",
        "describe_structured_scene",
        "count_detected_entities",
        "report_measured_distance",
        "report_spatial_relations",
        "speak_response",
    ],
    "planned_but_not_executable": [
        "turn_head",
        "turn_base",
        "move_base",
        "navigate_to_person",
        "control_headlights",
        "manipulate_object",
    ],
}


def build_response_context(
    *,
    request_id: str,
    query: str,
    route: str,
    capabilities: Dict[str, Any] | None = None,
    action_result: Dict[str, Any] | None = None,
    provider: str = "local_qwen",
) -> Dict[str, Any]:
    """Build the provider-neutral context passed to response generators."""
    return {
        "schema": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "request_id": request_id,
        "query": query,
        "route": route,
        "provider": provider,
        "capabilities": capabilities or DEFAULT_CAPABILITIES,
        "action_result": action_result or {
            "state": "not_requested",
            "execution_allowed": False,
        },
        "rules": {
            "sensor_values_must_be_supplied": True,
            "claim_action_only_after_executor_confirmation": True,
            "hardware_commands_forbidden": True,
        },
    }


def render_response_context(context: Dict[str, Any] | None) -> str:
    """Render a compact, explicit context block for any AI provider."""
    payload = context if isinstance(context, dict) else {}
    return (
        "ROBOT_RESPONSE_CONTEXT="
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )
