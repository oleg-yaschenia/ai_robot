#!/usr/bin/env python3
"""Local Semantic Resolver v1 core.

The model extracts only semantic intent and candidate slots.
Deterministic code owns capability allowlists, required arguments,
status, risk, confirmation, shadow mode and execution denial.
"""

from __future__ import annotations

import copy
import json
import re
import time
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

from robot_vision_assistant.assistant_request_contract import (
    validate_request_plan,
)


MODEL_INTENTS: Tuple[str, ...] = (
    "scene_path_safety",
    "scene_overview",
    "person_presence",
    "object_count",
    "object_location",
    "head_turn",
    "base_turn",
    "base_move",
    "headlight_set",
    "take_photo",
    "general_chat",
    "none",
)

MODEL_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": list(MODEL_INTENTS)},
        "direction": {
            "type": "string",
            "enum": ["left", "right", "forward", "backward", "none"],
        },
        "state": {"type": "string", "enum": ["on", "off", "none"]},
        "object": {"type": "string"},
        "target": {"type": "string"},
        "distance_m": {
            "anyOf": [{"type": "number"}, {"type": "null"}],
        },
    },
    "required": [
        "intent",
        "direction",
        "state",
        "object",
        "target",
        "distance_m",
    ],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """Ты извлекаешь смысл запроса для домашнего робота.
Не отвечай пользователю и не решай, разрешено ли выполнять действие.
Верни только JSON по schema.

Все intent из списка являются известными семантическими возможностями:
- scene_path_safety: оценить по текущей сцене возможность безопасного проезда;
- scene_overview: описать текущую сцену;
- person_presence: проверить наличие человека;
- object_count: посчитать объекты;
- object_location: определить положение объекта;
- head_turn: повернуть только голову;
- base_turn: повернуть мобильную платформу;
- base_move: переместить платформу к цели или на расстояние;
- headlight_set: включить или выключить свет;
- take_photo: сделать фотографию;
- general_chat: разговорный или информационный запрос;
- none: запрос не соответствует ни одному известному intent.

Правила:
1. Классифицируй смысл, а не готовность Executor.
2. Не используй none, если запрос соответствует известному intent.
3. Не придумывай отсутствующие аргументы.
4. Неиспользуемые поля:
   direction/state=none, object/target="", distance_m=null.
5. Для отсутствующих физических навыков, например сальто, полет или
   приготовление еды, используй intent=none.
6. Для base_move:
   target — цель движения;
   distance_m — только явно названное расстояние;
   direction=forward — только если направление явно задано.
"""


@dataclass(frozen=True)
class SemanticIntentSpec:
    intent: str
    request_type: str
    allowed_arguments: Tuple[str, ...] = ()
    required_arguments: Tuple[str, ...] = ()
    needs_scene: bool = False
    needs_confirmation: bool = False
    risk_level: str = "none"


INTENT_CATALOG: Dict[str, SemanticIntentSpec] = {
    "scene_path_safety": SemanticIntentSpec(
        "scene_path_safety", "observation", needs_scene=True
    ),
    "scene_overview": SemanticIntentSpec(
        "scene_overview", "observation", needs_scene=True
    ),
    "person_presence": SemanticIntentSpec(
        "person_presence", "observation", needs_scene=True
    ),
    "object_count": SemanticIntentSpec(
        "object_count",
        "observation",
        allowed_arguments=("object",),
        required_arguments=("object",),
        needs_scene=True,
    ),
    "object_location": SemanticIntentSpec(
        "object_location",
        "observation",
        allowed_arguments=("object",),
        required_arguments=("object",),
        needs_scene=True,
    ),
    "head_turn": SemanticIntentSpec(
        "head_turn",
        "action",
        allowed_arguments=("direction",),
        required_arguments=("direction",),
        risk_level="low",
    ),
    "base_turn": SemanticIntentSpec(
        "base_turn",
        "action",
        allowed_arguments=("direction",),
        required_arguments=("direction",),
        needs_confirmation=True,
        risk_level="medium",
    ),
    "base_move": SemanticIntentSpec(
        "base_move",
        "action",
        allowed_arguments=("direction", "target", "distance_m"),
        required_arguments=("target", "distance_m"),
        needs_confirmation=True,
        risk_level="medium",
    ),
    "headlight_set": SemanticIntentSpec(
        "headlight_set",
        "action",
        allowed_arguments=("state",),
        required_arguments=("state",),
        risk_level="low",
    ),
    "take_photo": SemanticIntentSpec(
        "take_photo", "action", risk_level="low"
    ),
    "general_chat": SemanticIntentSpec(
        "general_chat", "chat"
    ),
}


GENERAL_CHAT_PATTERNS: Tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, flags=re.IGNORECASE)
    for pattern in (
        r"^(?:робот[, ]+)?(?:объясни|расскажи|поясни)\b",
        r"^(?:что такое|что означает|как работает|как устроен|почему)\b",
        r"^как(?:\s+правильно)?\s+[а-я]+(?:ть|ти|чь)\b",
        r"^(?:ты\s+)?умеешь\b",
        r"^можно ли тебя попросить\b",
        r"^нужно ли говорить\b",
        r"^я не прошу тебя\b",
        r"\bесли я скажу\b",
        r"\bчто (?:ты )?сделаешь,?\s+если\b",
        r"\bчто (?:будет|произойдет),?\s+если\b",
        r"\bничего не делай\b",
        r"\bне выполняй\b",
        r"\bпросто расскажи\b",
    )
)

APPROACH_USER_PATTERN = re.compile(
    r"^(?:подъедь|подъезжай|приблизься)\s+(?:ко мне|к мне)$",
    flags=re.IGNORECASE,
)


def normalize_query(query: str) -> str:
    value = (query or "").strip().lower().replace("ё", "е")
    value = re.sub(r"[.!?…]+$", "", value).strip()
    return re.sub(r"\s+", " ", value)


def deterministic_pre_resolve(query: str) -> Optional[Dict[str, Any]]:
    normalized = normalize_query(query)

    if APPROACH_USER_PATTERN.fullmatch(normalized):
        return {
            "status": "AMBIGUOUS",
            "intent": "base_move",
            "arguments": {"target": "user"},
            "missing_arguments": ["distance_m"],
            "source": "deterministic_guard",
        }

    if any(pattern.search(normalized) for pattern in GENERAL_CHAT_PATTERNS):
        return {
            "status": "MATCHED",
            "intent": "general_chat",
            "arguments": {},
            "missing_arguments": [],
            "source": "deterministic_guard",
        }

    return None


def _clean_slot(name: str, value: Any) -> Any:
    if name in {"direction", "state"} and value == "none":
        return None
    if name in {"object", "target"} and value in {"", "none", None}:
        return None
    if name == "distance_m":
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return float(value)
        return None
    return value


def normalize_candidate(candidate: Dict[str, Any]) -> Dict[str, Any]:
    spec = INTENT_CATALOG.get(str(candidate.get("intent")))

    if spec is None:
        return {
            "status": "UNKNOWN",
            "intent": None,
            "arguments": {},
            "missing_arguments": [],
            "source": "qwen_semantic_candidate",
        }

    raw_arguments = {
        "direction": candidate.get("direction"),
        "state": candidate.get("state"),
        "object": candidate.get("object"),
        "target": candidate.get("target"),
        "distance_m": candidate.get("distance_m"),
    }

    arguments: Dict[str, Any] = {}
    for name in spec.allowed_arguments:
        value = _clean_slot(name, raw_arguments.get(name))
        if value is not None:
            arguments[name] = value

    missing_arguments = [
        name
        for name in spec.required_arguments
        if arguments.get(name) is None
    ]

    return {
        "status": "AMBIGUOUS" if missing_arguments else "MATCHED",
        "intent": spec.intent,
        "arguments": arguments,
        "missing_arguments": missing_arguments,
        "source": "qwen_semantic_candidate",
    }


def clarification_question(
    intent: Optional[str],
    missing_arguments: Tuple[str, ...] | list[str],
) -> str:
    missing = set(missing_arguments)

    if intent == "base_move":
        if {"target", "distance_m"} <= missing:
            return (
                "К какой цели подъехать и на каком расстоянии "
                "от неё остановиться?"
            )
        if "target" in missing:
            return "К какой цели подъехать?"
        if "distance_m" in missing:
            return "На каком расстоянии от цели остановиться?"

    if "direction" in missing:
        return "В какую сторону?"
    if "state" in missing:
        return "Включить или выключить?"
    if "object" in missing:
        return "Какой именно объект?"

    return "Уточните, пожалуйста, недостающие параметры команды."


def apply_semantic_resolution(
    base_request_plan: Dict[str, Any],
    semantic_result: Dict[str, Any],
) -> Dict[str, Any]:
    if not isinstance(base_request_plan, dict):
        raise ValueError("base_request_plan must be a dictionary")

    base_status = base_request_plan.get("resolution", {}).get("status")
    if base_status != "UNKNOWN":
        raise ValueError(
            "semantic resolution may only replace an UNKNOWN plan"
        )

    result = copy.deepcopy(base_request_plan)
    status = str(semantic_result.get("status", "UNKNOWN"))
    intent = semantic_result.get("intent")
    arguments = dict(semantic_result.get("arguments", {}))
    missing_arguments = list(
        semantic_result.get("missing_arguments", [])
    )
    source = str(
        semantic_result.get(
            "source",
            "local_semantic_resolver",
        )
    )

    spec = INTENT_CATALOG.get(str(intent)) if intent else None

    if status == "MATCHED" and spec is not None:
        result["resolution"] = {
            "status": "MATCHED",
            "source": source,
            "reason": "semantic_candidate_validated",
            "candidates": [spec.intent],
            "missing_arguments": [],
        }
        result["plan"] = {
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

    elif status == "AMBIGUOUS" and spec is not None:
        result["resolution"] = {
            "status": "AMBIGUOUS",
            "source": source,
            "reason": "semantic_required_arguments_missing",
            "candidates": [spec.intent],
            "missing_arguments": missing_arguments,
        }
        result["plan"] = {
            "request_type": "clarification",
            "intent": None,
            "arguments": arguments,
            "steps": [],
            "needs_scene": False,
            "needs_confirmation": False,
            "needs_clarification": True,
            "clarification_question": clarification_question(
                spec.intent,
                missing_arguments,
            ),
            "risk_level": "none",
        }

    else:
        result["resolution"] = {
            "status": "UNKNOWN",
            "source": source,
            "reason": "semantic_resolver_unresolved",
            "candidates": [],
            "missing_arguments": [],
        }
        result["plan"] = {
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

    result["semantic_fallback"] = {
        "required": False,
        "attempted": True,
        "resolved": result["resolution"]["status"] != "UNKNOWN",
        "reason": (
            None
            if result["resolution"]["status"] != "UNKNOWN"
            else "local_semantic_resolver_did_not_match"
        ),
    }
    result["shadow_mode"] = True
    result["execution_allowed"] = False

    errors = validate_request_plan(result)
    if errors:
        raise ValueError("; ".join(errors))

    return result


class LlamaSemanticClient:
    def __init__(
        self,
        server_url: str = "http://127.0.0.1:8080",
        model_id: str = "",
        timeout_sec: float = 30.0,
        max_tokens: int = 96,
    ) -> None:
        self.server_url = server_url.rstrip("/")
        self.model_id = model_id.strip()
        self.timeout_sec = float(timeout_sec)
        self.max_tokens = int(max_tokens)

    def _json_request(
        self,
        path: str,
        *,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        data = None
        method = "GET"
        headers = {"Accept": "application/json"}

        if payload is not None:
            data = json.dumps(
                payload,
                ensure_ascii=False,
            ).encode("utf-8")
            method = "POST"
            headers["Content-Type"] = "application/json"

        request = urllib.request.Request(
            f"{self.server_url}{path}",
            data=data,
            headers=headers,
            method=method,
        )
        with urllib.request.urlopen(
            request,
            timeout=self.timeout_sec,
        ) as response:
            return json.loads(
                response.read().decode("utf-8")
            )

    def resolve_model_id(self) -> str:
        if self.model_id:
            return self.model_id

        response = self._json_request("/v1/models")
        data = response.get("data")
        if isinstance(data, list) and data and isinstance(data[0], dict):
            model_id = data[0].get("id")
            if isinstance(model_id, str) and model_id:
                self.model_id = model_id
                return model_id

        raise RuntimeError(
            "model id not found in llama-server /v1/models"
        )

    def classify(
        self,
        query: str,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        payload = {
            "model": self.resolve_model_id(),
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "cache_prompt": True,
            "chat_template_kwargs": {"enable_thinking": False},
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "semantic_slots",
                    "strict": True,
                    "schema": MODEL_SCHEMA,
                },
            },
        }

        started = time.perf_counter()
        envelope = self._json_request(
            "/v1/chat/completions",
            payload=payload,
        )
        elapsed = time.perf_counter() - started

        choice = envelope["choices"][0]
        if choice.get("finish_reason") != "stop":
            raise RuntimeError(
                "semantic response did not finish cleanly: "
                f"{choice.get('finish_reason')!r}"
            )

        candidate = json.loads(choice["message"]["content"])
        return candidate, {
            "elapsed_sec": elapsed,
            "finish_reason": choice.get("finish_reason"),
            "timings": envelope.get("timings", {}),
            "usage": envelope.get("usage", {}),
        }


class LocalSemanticResolver:
    def __init__(
        self,
        candidate_provider: Optional[
            Callable[
                [str],
                Tuple[Dict[str, Any], Dict[str, Any]],
            ]
        ] = None,
    ) -> None:
        self.candidate_provider = candidate_provider

    def resolve(
        self,
        query: str,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        deterministic = deterministic_pre_resolve(query)
        if deterministic is not None:
            return deterministic, {
                "source": "deterministic_guard",
                "elapsed_sec": 0.0,
                "model_called": False,
            }

        if self.candidate_provider is None:
            raise RuntimeError(
                "candidate_provider is required for model resolution"
            )

        candidate, metrics = self.candidate_provider(query)
        normalized = normalize_candidate(candidate)
        return normalized, {
            **metrics,
            "source": "qwen_semantic_candidate",
            "model_called": True,
            "raw_candidate": candidate,
        }
