#!/usr/bin/env python3
"""Clarification state for Assistant Core v2 shadow mode."""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, Optional


class ClarificationManager:
    def __init__(self, max_attempts: int = 2) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        self.max_attempts = max_attempts
        self.pending: Optional[Dict[str, Any]] = None

    def open_from_plan(
        self,
        request_plan: Dict[str, Any],
    ) -> Dict[str, Any]:
        plan = request_plan.get("plan", {})
        resolution = request_plan.get("resolution", {})

        if resolution.get("status") != "AMBIGUOUS":
            raise ValueError(
                "clarification can only open for AMBIGUOUS plans"
            )

        self.pending = {
            "clarification_id": str(uuid.uuid4()),
            "parent_request_id": request_plan["request_id"],
            "original_query": request_plan["query"],
            "question": plan.get("clarification_question"),
            "candidates": resolution.get("candidates", []),
            "missing_arguments": resolution.get(
                "missing_arguments",
                [],
            ),
            "attempt": 1,
            "max_attempts": self.max_attempts,
            "created_at": time.time(),
            "shadow_mode": True,
            "execution_allowed": False,
        }
        return dict(self.pending)

    def next_attempt(
        self,
        question: Optional[str] = None,
    ) -> Dict[str, Any]:
        if self.pending is None:
            raise RuntimeError("no clarification is pending")

        current_attempt = int(self.pending["attempt"])
        if current_attempt >= self.max_attempts:
            return {
                **self.pending,
                "status": "ABORTED",
                "question": (
                    "Я не смог однозначно понять задачу. "
                    "Сформулируйте её, пожалуйста, иначе."
                ),
            }

        self.pending["attempt"] = current_attempt + 1
        if question:
            self.pending["question"] = question
        return {
            **self.pending,
            "status": "PENDING",
        }

    def clear(self) -> None:
        self.pending = None
