#!/usr/bin/env python3
"""Pure routing helpers for Response Orchestrator v1.

The final spoken response is AI-generated for normal text, visual and
structured-scene requests. Scene Interpreter remains a verified fact source,
not the conversational endpoint. Physical actions remain blocked until a
Policy Gate and Executor report an accepted/running/completed state.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict

from robot_vision_assistant.assistant_response_contract import (
    build_response_context,
)
from robot_vision_assistant.assistant_router import classify_route


@dataclass(frozen=True)
class ResponseRoute:
    route: str
    qwen_mode: str
    reason: str
    new_visual_session: bool = False
    is_action: bool = False
    is_emergency: bool = False


EMERGENCY_PATTERN = re.compile(
    r"^(?:стоп|стой|остановись|аварийная остановка)$",
    flags=re.IGNORECASE,
)

STRUCTURED_SCENE_PATTERNS = (
    r"\bсколько\s+(?:человек|людей|персон|объектов|предметов|"
    r"стульев|бутылок|чашек)\b",
    r"\b(?:как|насколько)\s+далеко\b",
    r"\b(?:на\s+каком\s+)?расстоянии\b",
    r"\b(?:какое|каков[ао]?)\s+расстояние\b",
    r"\bрасстояни\w*\s+до\b",
    r"\bдистанц\w*\b",
    r"\bближайш\w*\s+(?:человек|объект|предмет)\b",
    r"\b(?:где\s+находится|где\s+виден)\b",
    r"\b(?:слева|справа)\s+от\b",
    r"\b(?:безопасно|можно)\s+ли\s+(?:здесь\s+)?(?:ехать|проехать)\b",
    r"\bпрепятств",
)

VISUAL_OVERVIEW_PATTERNS = (
    r"\bчто\s+(?:ты\s+)?(?:сейчас\s+)?видишь\b",
    r"\b(?:подробно\s+)?опиши(?:\s+мне)?\s+"
    r"(?:текущий\s+|этот\s+)?(?:кадр|изображение|сцену)\b",
    r"\bчто\s+(?:ты\s+)?(?:сейчас\s+)?можешь\s+видеть\b",
    r"\bопиши\s+(?:эту\s+|текущую\s+)?"
    r"(?:сцену|обстановку|изображение|кадр|комнату)\b",
    r"\b(?:что|кто)\s+(?:сейчас\s+)?"
    r"(?:есть|находится)?\s*в\s+кадре\b",
    r"\bчто\s+(?:сейчас\s+)?перед\s+тобой\b",
    r"\b(?:осмотрись|оглянись|посмотри\s+вокруг)\b",
    r"\bчто\s+(?:сейчас\s+)?(?:находится\s+)?"
    r"(?:слева|справа)\b",
    r"\bчто\s+написано\b",
    r"\bпрочитай\b",
    r"\bкакого\s+цвета\b",
    r"\bкак\s+выглядит\b",
    r"\bчто\s+(?:он|она|они)\s+держит\b",
)

VISUAL_FOLLOWUP_PATTERNS = (
    r"\b(?:кто|что)\s+(?:находится\s+)?(?:слева|справа)\b",
    r"\b(?:а\s+)?(?:слева|справа)\b",
    r"\bчто\s+(?:он|она|они)\s+держит\b",
    r"\bчто\s+(?:находится\s+)?рядом\s+"
    r"(?:с\s+ним|с\s+ней|с\s+этим|с\s+человеком)\b",
)


def normalize_query(query: str) -> str:
    value = (query or "").strip().lower().replace("ё", "е")
    return re.sub(r"\s+", " ", value)


def classify_response_route(
    query: str,
    *,
    visual_session_active: bool = False,
) -> ResponseRoute:
    normalized = normalize_query(query)
    if not normalized:
        raise ValueError("query must not be empty")

    if EMERGENCY_PATTERN.fullmatch(normalized):
        return ResponseRoute(
            route="emergency",
            qwen_mode="text",
            reason="deterministic_emergency_path",
            is_action=True,
            is_emergency=True,
        )

    if any(re.search(pattern, normalized) for pattern in STRUCTURED_SCENE_PATTERNS):
        return ResponseRoute(
            route="scene_fact",
            qwen_mode="scene",
            reason="verified_scene_fact_required",
        )

    if (
        visual_session_active
        and any(re.search(pattern, normalized) for pattern in VISUAL_FOLLOWUP_PATTERNS)
    ):
        return ResponseRoute(
            route="visual_followup",
            qwen_mode="image",
            reason="reuse_active_visual_session",
            new_visual_session=False,
        )

    if any(re.search(pattern, normalized) for pattern in VISUAL_OVERVIEW_PATTERNS):
        return ResponseRoute(
            route="visual",
            qwen_mode="image",
            reason="image_reasoning_requested",
            new_visual_session=True,
        )

    router_result = classify_route(query)
    if router_result["mode"] in {"action", "action+report"}:
        return ResponseRoute(
            route="action",
            qwen_mode="text",
            reason=router_result["reason"],
            is_action=True,
        )

    return ResponseRoute(
        route="text",
        qwen_mode="text",
        reason="general_conversation",
    )


def build_action_result(route: ResponseRoute) -> Dict[str, Any]:
    if route.is_emergency:
        return {
            "state": "safety_request_received",
            "execution_allowed": False,
            "reason": "executor_not_connected",
            "instruction": (
                "Сформулируй, что запрос остановки принят системой, но не "
                "утверждай, что движение было остановлено без подтверждения "
                "исполнительного контура."
            ),
        }
    if route.is_action:
        return {
            "state": "unavailable",
            "execution_allowed": False,
            "reason": "policy_gate_and_executor_not_connected",
            "instruction": (
                "Естественно сообщи, что команда понята, но физическое "
                "выполнение пока не подключено. Не говори «выполняю»."
            ),
        }
    return {
        "state": "not_requested",
        "execution_allowed": False,
    }


def build_qwen_request(
    *,
    request_id: str,
    query: str,
    route: ResponseRoute,
    capabilities: Dict[str, Any] | None = None,
    memory_context: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    context = build_response_context(
        request_id=request_id,
        query=query,
        route=route.route,
        capabilities=capabilities,
        action_result=build_action_result(route),
        memory_context=memory_context,
    )
    return {
        "request_id": request_id,
        "query": query,
        "mode": route.qwen_mode,
        "new_session": route.new_visual_session,
        "response_context": context,
    }


def build_legacy_request(
    *,
    request_id: str,
    query: str,
    route: ResponseRoute,
) -> Dict[str, Any]:
    return {
        "request_id": request_id,
        "query": query,
        "route": route.route,
    }


def fallback_source(response_mode: str, has_legacy_candidate: bool) -> str:
    mode = response_mode.strip().lower()
    if mode == "legacy":
        return "legacy" if has_legacy_candidate else "deterministic_error"
    if mode == "hybrid" and has_legacy_candidate:
        return "legacy"
    return "deterministic_error"
