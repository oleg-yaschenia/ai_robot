#!/usr/bin/env python3
"""Request-plan contract for Assistant Core v2 shadow processing."""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List


SCHEMA_NAME = "robot_assistant_request_plan"
SCHEMA_VERSION = 1

RESOLUTION_STATUSES = (
    "MATCHED",
    "AMBIGUOUS",
    "UNKNOWN",
    "REJECTED",
)

REQUEST_TYPES = (
    "chat",
    "observation",
    "action",
    "compound",
    "clarification",
    "unknown",
)


def new_request_id() -> str:
    return str(uuid.uuid4())


def build_request_plan(
    *,
    query: str,
    normalized_query: str,
    resolution: Dict[str, Any],
    plan: Dict[str, Any],
    source_topic: str = "/vision_assistant/query",
) -> Dict[str, Any]:
    status = resolution.get("status", "UNKNOWN")

    message = {
        "schema": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "timestamp": time.time(),
        "request_id": new_request_id(),
        "source_topic": source_topic,
        "query": query,
        "normalized_query": normalized_query,
        "resolution": resolution,
        "plan": plan,
        "semantic_fallback": {
            "required": status == "UNKNOWN",
            "reason": (
                "known_intent_resolver_did_not_match"
                if status == "UNKNOWN"
                else None
            ),
        },
        "shadow_mode": True,
        "execution_allowed": False,
    }

    errors = validate_request_plan(message)
    if errors:
        raise ValueError("; ".join(errors))
    return message


def validate_request_plan(message: Dict[str, Any]) -> List[str]:
    errors: List[str] = []

    if not isinstance(message, dict):
        return ["message must be a dictionary"]

    if message.get("schema") != SCHEMA_NAME:
        errors.append(f"schema must be {SCHEMA_NAME!r}")
    if message.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}")

    resolution = message.get("resolution")
    if not isinstance(resolution, dict):
        errors.append("resolution must be a dictionary")
    else:
        if resolution.get("status") not in RESOLUTION_STATUSES:
            errors.append("resolution.status is invalid")
        if not isinstance(resolution.get("candidates", []), list):
            errors.append("resolution.candidates must be a list")
        if not isinstance(
            resolution.get("missing_arguments", []),
            list,
        ):
            errors.append(
                "resolution.missing_arguments must be a list"
            )

    plan = message.get("plan")
    if not isinstance(plan, dict):
        errors.append("plan must be a dictionary")
    else:
        if plan.get("request_type") not in REQUEST_TYPES:
            errors.append("plan.request_type is invalid")
        if not isinstance(plan.get("arguments", {}), dict):
            errors.append("plan.arguments must be a dictionary")
        if not isinstance(plan.get("steps", []), list):
            errors.append("plan.steps must be a list")
        if not isinstance(plan.get("needs_scene"), bool):
            errors.append("plan.needs_scene must be a boolean")
        if not isinstance(plan.get("needs_confirmation"), bool):
            errors.append(
                "plan.needs_confirmation must be a boolean"
            )
        if not isinstance(plan.get("needs_clarification"), bool):
            errors.append(
                "plan.needs_clarification must be a boolean"
            )

    fallback = message.get("semantic_fallback")
    if not isinstance(fallback, dict):
        errors.append("semantic_fallback must be a dictionary")
    elif not isinstance(fallback.get("required"), bool):
        errors.append(
            "semantic_fallback.required must be a boolean"
        )

    if message.get("shadow_mode") is not True:
        errors.append("shadow_mode must be true")
    if message.get("execution_allowed") is not False:
        errors.append("execution_allowed must be false")

    return errors
