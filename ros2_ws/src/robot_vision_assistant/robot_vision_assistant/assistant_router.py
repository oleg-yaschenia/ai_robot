#!/usr/bin/env python3
"""Deterministic Assistant Router v1.

This module has no ROS dependencies. It classifies a user query into:
chat, visual_chat, action, or action+report.

The first deployment is shadow-only. Decisions are telemetry and never execute
robot actions.
"""

from __future__ import annotations

import re
import time
import uuid
from typing import Any, Dict, Iterable, List, Optional, Tuple


SCHEMA_NAME = "robot_assistant_route"
SCHEMA_VERSION = 1
ROUTE_MODES = ("chat", "visual_chat", "action", "action+report")

ACTION_PATTERNS: Tuple[Tuple[str, str], ...] = (
    ("stop", r"\b(остановись|останови|стой|стоп)\b"),
    ("move", r"\b(едь|поезжай|двигайся|подъедь|отъедь|проедь)\b"),
    ("turn", r"\b(поверни|развернись|разверни)\b"),
    ("follow", r"\b(следуй|иди\s+за|езжай\s+за)\b"),
    ("look", r"\b(посмотри|поверни\s+голову|наклони\s+голову)\b"),
    ("manipulate", r"\b(возьми|подними|принеси|положи|отдай)\b"),
    ("switch", r"\b(включи|выключи)\b"),
)

REPORT_PATTERNS: Tuple[Tuple[str, str], ...] = (
    ("and_tell", r"\bи\s+(скажи|сообщи|доложи|опиши)\b"),
    (
        "then_tell",
        r"\b(а\s+потом|потом|после\s+этого)\s+"
        r"(скажи|сообщи|доложи|опиши)\b",
    ),
    ("report_result", r"\b(сообщи|скажи|доложи)\s+результат\b"),
    ("report_back", r"\bотчитайся\b"),
)

VISUAL_DIRECT_PATTERNS: Tuple[Tuple[str, str], ...] = (
    ("what_seen", r"\bчто\s+(ты\s+)?видишь\b"),
    ("describe_scene", r"\bопиши\s+(сцену|обстановку|что\s+вокруг)\b"),
    ("in_frame", r"\b(что|кто)\s+в\s+кадре\b"),
    ("camera_view", r"\b(перед\s+камерой|перед\s+тобой)\b"),
    ("who_seen", r"\bкого\s+(ты\s+)?видишь\b"),
    ("object_list", r"\bкакие\s+(предметы|объекты)\b"),
    ("person_present", r"\bесть\s+ли\s+человек\b"),
    ("scene_change", r"\bчто\s+изменилось\b"),
)

VISUAL_OPERATORS = (
    "где",
    "сколько",
    "слева",
    "справа",
    "рядом",
    "возле",
    "около",
    "ближе",
    "дальше",
)

VISUAL_ENTITIES = (
    "человек",
    "люд",
    "кот",
    "кошк",
    "собак",
    "стул",
    "кресл",
    "бутыл",
    "чаш",
    "круж",
    "телефон",
    "ноутбук",
    "предмет",
    "объект",
)


def normalize_query(query: str) -> str:
    normalized = (query or "").strip().lower().replace("ё", "е")
    return re.sub(r"\s+", " ", normalized)


def _matched_patterns(
    text: str,
    patterns: Iterable[Tuple[str, str]],
) -> List[str]:
    return [
        name
        for name, pattern in patterns
        if re.search(pattern, text, flags=re.IGNORECASE)
    ]


def classify_route(query: str) -> Dict[str, Any]:
    normalized = normalize_query(query)
    if not normalized:
        raise ValueError("query must not be empty")

    action_matches = _matched_patterns(normalized, ACTION_PATTERNS)
    report_matches = _matched_patterns(normalized, REPORT_PATTERNS)
    direct_visual_matches = _matched_patterns(
        normalized,
        VISUAL_DIRECT_PATTERNS,
    )
    operator_matches = [
        item for item in VISUAL_OPERATORS if item in normalized
    ]
    entity_matches = [
        item for item in VISUAL_ENTITIES if item in normalized
    ]

    if action_matches and report_matches:
        mode = "action+report"
        confidence = 0.97
        reason = "action_and_report_markers"
        matched_rules = (
            [f"action:{item}" for item in action_matches]
            + [f"report:{item}" for item in report_matches]
        )
    elif action_matches:
        mode = "action"
        confidence = 0.93
        reason = "action_markers"
        matched_rules = [f"action:{item}" for item in action_matches]
    elif direct_visual_matches:
        mode = "visual_chat"
        confidence = 0.95
        reason = "direct_visual_question"
        matched_rules = [
            f"visual:{item}" for item in direct_visual_matches
        ]
    elif operator_matches and entity_matches:
        mode = "visual_chat"
        confidence = 0.88
        reason = "visual_operator_and_entity"
        matched_rules = (
            [f"operator:{item}" for item in operator_matches]
            + [f"entity:{item}" for item in entity_matches]
        )
    else:
        mode = "chat"
        confidence = 0.70
        reason = "default_chat"
        matched_rules = ["default:chat"]

    return {
        "mode": mode,
        "confidence": confidence,
        "reason": reason,
        "matched_rules": matched_rules,
    }


def build_route_decision(
    query: str,
    *,
    scene_context: Optional[Dict[str, Any]] = None,
    source_topic: str = "/vision_assistant/query",
) -> Dict[str, Any]:
    route = classify_route(query)
    scene = scene_context if isinstance(scene_context, dict) else {}

    handlers = {
        "chat": "future_chat_engine",
        "visual_chat": "vision_assistant",
        "action": "future_action_planner",
        "action+report": "future_action_planner",
    }

    return {
        "schema": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "timestamp": time.time(),
        "request_id": str(uuid.uuid4()),
        "source_topic": source_topic,
        "query": query.strip(),
        "normalized_query": normalize_query(query),
        "mode": route["mode"],
        "confidence": route["confidence"],
        "reason": route["reason"],
        "matched_rules": route["matched_rules"],
        "suggested_handler": handlers[route["mode"]],
        "shadow_mode": True,
        "execution_allowed": False,
        "scene_context": {
            "available": bool(scene.get("available", False)),
            "age_sec": scene.get("age_sec"),
            "entity_count": int(scene.get("entity_count", 0) or 0),
            "counts": (
                scene.get("counts")
                if isinstance(scene.get("counts"), dict)
                else {}
            ),
        },
    }


def validate_route_decision(message: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    if not isinstance(message, dict):
        return ["message must be a dictionary"]

    if message.get("schema") != SCHEMA_NAME:
        errors.append(f"schema must be {SCHEMA_NAME!r}")
    if message.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}")
    if message.get("mode") not in ROUTE_MODES:
        errors.append("mode is invalid")
    if message.get("shadow_mode") is not True:
        errors.append("shadow_mode must be true")
    if message.get("execution_allowed") is not False:
        errors.append("execution_allowed must be false")
    if not isinstance(message.get("request_id"), str):
        errors.append("request_id must be a string")
    if not isinstance(message.get("matched_rules"), list):
        errors.append("matched_rules must be a list")
    if not isinstance(message.get("scene_context"), dict):
        errors.append("scene_context must be a dictionary")

    return errors
