#!/usr/bin/env python3
"""Strict known-intent resolver for Assistant Core v2.

The resolver only returns MATCHED when the complete request is recognized and
all mandatory arguments are present. It never guesses. Unknown requests are
reserved for a future semantic AI resolver.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Pattern, Tuple

from robot_vision_assistant.assistant_request_contract import (
    build_request_plan,
)


def normalize_query(query: str) -> str:
    value = (query or "").strip().lower().replace("ё", "е")
    value = re.sub(r"[.!?…]+$", "", value).strip()
    return re.sub(r"\s+", " ", value)


OBJECT_ALIASES = {
    "человек": "person",
    "людей": "person",
    "люди": "person",
    "стул": "chair",
    "стульев": "chair",
    "бутылка": "bottle",
    "бутылку": "bottle",
    "бутылок": "bottle",
    "чашка": "cup",
    "чашку": "cup",
    "чашек": "cup",
    "телефон": "cell_phone",
    "телефона": "cell_phone",
    "кот": "cat",
    "кошку": "cat",
    "собака": "dog",
    "собаку": "dog",
}

DIRECTION_ALIASES = {
    "вправо": "right",
    "направо": "right",
    "влево": "left",
    "налево": "left",
}


@dataclass(frozen=True)
class KnownIntentSpec:
    intent: str
    request_type: str
    patterns: Tuple[Pattern[str], ...]
    needs_scene: bool = False
    needs_confirmation: bool = False
    risk_level: str = "none"


def _compile(*patterns: str) -> Tuple[Pattern[str], ...]:
    return tuple(re.compile(item, re.IGNORECASE) for item in patterns)


KNOWN_INTENTS: Tuple[KnownIntentSpec, ...] = (
    KnownIntentSpec(
        intent="emergency_stop",
        request_type="action",
        patterns=_compile(
            r"^(стоп|остановись|аварийная остановка)$",
        ),
        needs_confirmation=False,
        risk_level="low",
    ),
    KnownIntentSpec(
        intent="scene_overview",
        request_type="observation",
        patterns=_compile(
            r"^(что ты видишь|что перед тобой|опиши сцену|"
            r"кто в кадре|что в кадре)$",
        ),
        needs_scene=True,
    ),
    KnownIntentSpec(
        intent="person_presence",
        request_type="observation",
        patterns=_compile(
            r"^(есть ли (?:в кадре )?человек|видно ли человека)$",
        ),
        needs_scene=True,
    ),
    KnownIntentSpec(
        intent="object_count",
        request_type="observation",
        patterns=_compile(
            r"^сколько (?P<object>людей|стульев|бутылок|чашек)"
            r"(?: ты видишь| в кадре)?$",
        ),
        needs_scene=True,
    ),
    KnownIntentSpec(
        intent="object_location",
        request_type="observation",
        patterns=_compile(
            r"^где (?:находится )?"
            r"(?P<object>человек|стул|бутылка|чашка|телефон|кот|собака)$",
        ),
        needs_scene=True,
    ),
    KnownIntentSpec(
        intent="head_turn",
        request_type="action",
        patterns=_compile(
            r"^(?:пожалуйста[, ]+)?"
            r"(?:поверни голову|посмотри) "
            r"(?P<direction>вправо|направо|влево|налево)$",
        ),
        needs_confirmation=False,
        risk_level="low",
    ),
    KnownIntentSpec(
        intent="base_turn",
        request_type="action",
        patterns=_compile(
            r"^(?:пожалуйста[, ]+)?"
            r"(?:повернись|развернись) "
            r"(?P<direction>вправо|направо|влево|налево)$",
        ),
        needs_confirmation=True,
        risk_level="medium",
    ),
    KnownIntentSpec(
        intent="headlight_set",
        request_type="action",
        patterns=_compile(
            r"^(?P<state>включи|выключи) "
            r"(?P<device>свет|подсветку|фары)$",
        ),
        needs_confirmation=False,
        risk_level="low",
    ),
    KnownIntentSpec(
        intent="take_photo",
        request_type="action",
        patterns=_compile(
            r"^(сделай фотографию|сделай фото|сделай снимок|"
            r"сфотографируй)$",
        ),
        needs_confirmation=False,
        risk_level="low",
    ),
    KnownIntentSpec(
        intent="greeting",
        request_type="chat",
        patterns=_compile(
            r"^(привет|здравствуй|добрый день|как дела)$",
        ),
    ),
    KnownIntentSpec(
        intent="capabilities",
        request_type="chat",
        patterns=_compile(
            r"^(что ты умеешь|расскажи что ты умеешь)$",
        ),
    ),
)


AMBIGUOUS_PATTERNS: Tuple[
    Tuple[Pattern[str], List[str], List[str], str],
    ...,
] = (
    (
        re.compile(r"^(поверни|повернись|развернись)$"),
        ["head_turn", "base_turn"],
        ["target", "direction"],
        "turn_target_and_direction_missing",
    ),
    (
        re.compile(r"^(посмотри|посмотри туда|поверни голову)$"),
        ["head_turn"],
        ["direction"],
        "head_direction_missing",
    ),
    (
        re.compile(
            r"^(подъедь|подъедь ближе|едь вперед|езжай вперед|поезжай)$"
        ),
        ["base_move"],
        ["target", "distance"],
        "move_target_or_distance_missing",
    ),
    (
        re.compile(r"^(возьми это|возьми предмет|подними это)$"),
        ["manipulator_pick"],
        ["object_reference"],
        "object_reference_missing",
    ),
    (
        re.compile(r"^(включи|выключи)$"),
        ["headlight_set"],
        ["device"],
        "device_missing",
    ),
)


def _normalize_arguments(
    groups: Dict[str, Optional[str]],
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}

    for key, raw_value in groups.items():
        if raw_value is None:
            continue

        value = raw_value.strip().lower()
        if key == "object":
            result[key] = OBJECT_ALIASES.get(value, value)
        elif key == "direction":
            result[key] = DIRECTION_ALIASES.get(value, value)
        elif key == "state":
            result[key] = (
                "on" if value == "включи" else "off"
            )
        elif key == "device":
            result[key] = "headlight"
        else:
            result[key] = value

    return result


def _matched_plan(
    *,
    query: str,
    normalized: str,
    spec: KnownIntentSpec,
    arguments: Dict[str, Any],
) -> Dict[str, Any]:
    resolution = {
        "status": "MATCHED",
        "source": "known_intent_resolver",
        "confidence": 1.0,
        "reason": "strict_full_match",
        "candidates": [spec.intent],
        "missing_arguments": [],
    }
    plan = {
        "request_type": spec.request_type,
        "intent": spec.intent,
        "arguments": arguments,
        "steps": [
            {
                "type": spec.request_type,
                "intent": spec.intent,
                "arguments": arguments,
            }
        ],
        "needs_scene": spec.needs_scene,
        "needs_confirmation": spec.needs_confirmation,
        "needs_clarification": False,
        "clarification_question": None,
        "risk_level": spec.risk_level,
    }
    return build_request_plan(
        query=query,
        normalized_query=normalized,
        resolution=resolution,
        plan=plan,
    )


def _ambiguous_plan(
    *,
    query: str,
    normalized: str,
    candidates: List[str],
    missing_arguments: List[str],
    reason: str,
    clarification_question: str,
) -> Dict[str, Any]:
    resolution = {
        "status": "AMBIGUOUS",
        "source": "known_intent_resolver",
        "confidence": 1.0,
        "reason": reason,
        "candidates": candidates,
        "missing_arguments": missing_arguments,
    }
    plan = {
        "request_type": "clarification",
        "intent": None,
        "arguments": {},
        "steps": [],
        "needs_scene": False,
        "needs_confirmation": False,
        "needs_clarification": True,
        "clarification_question": clarification_question,
        "risk_level": "none",
    }
    return build_request_plan(
        query=query,
        normalized_query=normalized,
        resolution=resolution,
        plan=plan,
    )


def _unknown_plan(query: str, normalized: str) -> Dict[str, Any]:
    resolution = {
        "status": "UNKNOWN",
        "source": "known_intent_resolver",
        "confidence": 1.0,
        "reason": "no_strict_full_match",
        "candidates": [],
        "missing_arguments": [],
    }
    plan = {
        "request_type": "unknown",
        "intent": None,
        "arguments": {},
        "steps": [],
        "needs_scene": False,
        "needs_confirmation": False,
        "needs_clarification": False,
        "clarification_question": None,
        "risk_level": "none",
    }
    return build_request_plan(
        query=query,
        normalized_query=normalized,
        resolution=resolution,
        plan=plan,
    )


def clarification_question_for(
    candidates: List[str],
    missing_arguments: List[str],
) -> str:
    missing = set(missing_arguments)
    candidate_set = set(candidates)

    if {"head_turn", "base_turn"} <= candidate_set:
        return (
            "Повернуть голову или платформу, и в какую сторону?"
        )
    if "direction" in missing:
        return "В какую сторону?"
    if {"target", "distance"} & missing:
        return (
            "К какому объекту подъехать и на какое расстояние?"
        )
    if "object_reference" in missing:
        return "Какой именно объект?"
    if "device" in missing:
        return "Что именно включить или выключить?"
    return "Уточните, пожалуйста, что именно нужно сделать."


class KnownIntentResolver:
    def resolve(self, query: str) -> Dict[str, Any]:
        original = (query or "").strip()
        normalized = normalize_query(original)
        if not normalized:
            return _unknown_plan(original, normalized)

        for spec in KNOWN_INTENTS:
            for pattern in spec.patterns:
                match = pattern.fullmatch(normalized)
                if match is None:
                    continue
                arguments = _normalize_arguments(
                    match.groupdict()
                )
                return _matched_plan(
                    query=original,
                    normalized=normalized,
                    spec=spec,
                    arguments=arguments,
                )

        for pattern, candidates, missing, reason in AMBIGUOUS_PATTERNS:
            if pattern.fullmatch(normalized):
                return _ambiguous_plan(
                    query=original,
                    normalized=normalized,
                    candidates=candidates,
                    missing_arguments=missing,
                    reason=reason,
                    clarification_question=(
                        clarification_question_for(
                            candidates,
                            missing,
                        )
                    ),
                )

        return _unknown_plan(original, normalized)
