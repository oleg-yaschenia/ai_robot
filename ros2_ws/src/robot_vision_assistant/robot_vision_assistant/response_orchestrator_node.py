#!/usr/bin/env python3
"""Response Orchestrator v1.

Normal text, visual and structured-scene requests are phrased by the local AI
provider. Scene Interpreter supplies verified facts; it is not the final
speaker. Legacy output is retained only as a timeout/error rollback path.
Physical action requests are published as non-executable telemetry and are
phrased by AI as unavailable until Policy Gate and Executor are connected.
"""

from __future__ import annotations

import json
import time
import uuid
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from robot_vision_assistant.assistant_response_contract import (
    DEFAULT_CAPABILITIES,
)
from robot_vision_assistant.response_memory_bridge import (
    ResponseMemoryBridge,
)
from robot_vision_assistant.response_orchestrator import (
    ResponseRoute,
    build_legacy_request,
    build_qwen_request,
    classify_response_route,
    fallback_source,
)


@dataclass
class PendingResponse:
    request_id: str
    query: str
    route: ResponseRoute
    started_monotonic: float
    qwen_deadline: float
    legacy_deadline: float
    legacy_answer: Optional[str] = None
    qwen_failed: bool = False
    memory_conversation_id: Optional[str] = None
    memory_user_message_id: Optional[int] = None
    memory_context: Optional[Dict[str, Any]] = None


class ResponseOrchestratorNode(Node):
    def __init__(self) -> None:
        super().__init__("response_orchestrator_node")

        self.declare_parameter("response_mode", "hybrid")
        self.declare_parameter("query_topic", "/vision_assistant/query")
        self.declare_parameter(
            "qwen_query_topic", "/assistant/response/qwen_request_json"
        )
        self.declare_parameter(
            "legacy_query_topic", "/assistant/response/legacy_request_json"
        )
        self.declare_parameter(
            "qwen_candidate_topic", "/assistant/response/qwen_candidate_json"
        )
        self.declare_parameter(
            "legacy_candidate_topic", "/assistant/response/legacy_candidate_json"
        )
        self.declare_parameter("answer_topic", "/vision_assistant/answer")
        self.declare_parameter(
            "trace_topic", "/assistant/response/trace_json"
        )
        self.declare_parameter(
            "status_topic", "/assistant/response/status_json"
        )
        self.declare_parameter(
            "action_request_topic", "/assistant/action/request_json"
        )
        self.declare_parameter("qwen_timeout_sec", 20.0)
        self.declare_parameter("legacy_timeout_sec", 3.0)
        self.declare_parameter("visual_session_ttl_sec", 30.0)
        self.declare_parameter("max_pending_queries", 4)
        self.declare_parameter("memory_enabled", False)
        self.declare_parameter("memory_db_path", "")
        self.declare_parameter(
            "identity_topic",
            "/assistant/identity/current_speaker_json",
        )
        self.declare_parameter("identity_ttl_sec", 15.0)
        self.declare_parameter(
            "memory_inactivity_timeout_sec",
            900.0,
        )
        self.declare_parameter("memory_recent_messages", 8)
        self.declare_parameter("memory_personal_facts", 4)
        self.declare_parameter("memory_max_context_chars", 2400)
        self.declare_parameter("memory_worker_enabled", False)
        self.declare_parameter("memory_worker_queue_size", 128)
        self.declare_parameter(
            "default_speaker_entity_id",
            "person:unknown",
        )
        self.declare_parameter(
            "capabilities_json",
            json.dumps(DEFAULT_CAPABILITIES, ensure_ascii=False),
        )

        self.response_mode = str(
            self.get_parameter("response_mode").value
        ).strip().lower()
        if self.response_mode not in {"legacy", "hybrid", "qwen"}:
            raise RuntimeError(
                "response_mode must be legacy, hybrid or qwen"
            )

        self.query_topic = str(self.get_parameter("query_topic").value)
        self.qwen_query_topic = str(
            self.get_parameter("qwen_query_topic").value
        )
        self.legacy_query_topic = str(
            self.get_parameter("legacy_query_topic").value
        )
        self.qwen_candidate_topic = str(
            self.get_parameter("qwen_candidate_topic").value
        )
        self.legacy_candidate_topic = str(
            self.get_parameter("legacy_candidate_topic").value
        )
        self.answer_topic = str(self.get_parameter("answer_topic").value)
        self.trace_topic = str(self.get_parameter("trace_topic").value)
        self.status_topic = str(self.get_parameter("status_topic").value)
        self.action_request_topic = str(
            self.get_parameter("action_request_topic").value
        )
        self.qwen_timeout_sec = float(
            self.get_parameter("qwen_timeout_sec").value
        )
        self.legacy_timeout_sec = float(
            self.get_parameter("legacy_timeout_sec").value
        )
        self.visual_session_ttl_sec = float(
            self.get_parameter("visual_session_ttl_sec").value
        )
        self.max_pending_queries = int(
            self.get_parameter("max_pending_queries").value
        )
        self.memory_enabled = bool(
            self.get_parameter("memory_enabled").value
        )
        self.identity_topic = str(
            self.get_parameter("identity_topic").value
        )
        self.memory_bridge: Optional[ResponseMemoryBridge] = None
        if self.memory_enabled:
            memory_db_path = str(
                self.get_parameter("memory_db_path").value
            ).strip()
            if not memory_db_path:
                raise RuntimeError(
                    "memory_db_path is required when memory_enabled"
                )
            self.memory_bridge = ResponseMemoryBridge(
                memory_db_path,
                default_speaker_entity_id=str(
                    self.get_parameter(
                        "default_speaker_entity_id"
                    ).value
                ),
                identity_ttl_sec=float(
                    self.get_parameter("identity_ttl_sec").value
                ),
                inactivity_timeout_sec=float(
                    self.get_parameter(
                        "memory_inactivity_timeout_sec"
                    ).value
                ),
                recent_message_limit=int(
                    self.get_parameter(
                        "memory_recent_messages"
                    ).value
                ),
                personal_fact_limit=int(
                    self.get_parameter(
                        "memory_personal_facts"
                    ).value
                ),
                max_context_chars=int(
                    self.get_parameter(
                        "memory_max_context_chars"
                    ).value
                ),
                memory_worker_enabled=bool(
                    self.get_parameter(
                        "memory_worker_enabled"
                    ).value
                ),
                memory_worker_queue_size=int(
                    self.get_parameter(
                        "memory_worker_queue_size"
                    ).value
                ),
            )

        try:
            capabilities = json.loads(
                str(self.get_parameter("capabilities_json").value)
            )
            if not isinstance(capabilities, dict):
                raise ValueError("capabilities_json must be an object")
            self.capabilities: Dict[str, Any] = capabilities
        except Exception as exc:
            raise RuntimeError(f"invalid capabilities_json: {exc}") from exc

        self._pending: Optional[PendingResponse] = None
        self._queue: Deque[str] = deque()
        self._last_visual_completed_monotonic: Optional[float] = None
        self._completed_ids: Deque[str] = deque(maxlen=32)

        self.answer_pub = self.create_publisher(String, self.answer_topic, 10)
        self.qwen_query_pub = self.create_publisher(
            String, self.qwen_query_topic, 10
        )
        self.legacy_query_pub = self.create_publisher(
            String, self.legacy_query_topic, 10
        )
        self.trace_pub = self.create_publisher(String, self.trace_topic, 10)
        self.status_pub = self.create_publisher(String, self.status_topic, 10)
        self.action_request_pub = self.create_publisher(
            String, self.action_request_topic, 10
        )

        self.query_sub = self.create_subscription(
            String, self.query_topic, self.query_cb, 10
        )
        self.qwen_candidate_sub = self.create_subscription(
            String, self.qwen_candidate_topic, self.qwen_candidate_cb, 10
        )
        self.legacy_candidate_sub = self.create_subscription(
            String, self.legacy_candidate_topic, self.legacy_candidate_cb, 10
        )
        self.identity_sub = None
        if self.memory_bridge is not None:
            self.identity_sub = self.create_subscription(
                String,
                self.identity_topic,
                self.identity_cb,
                10,
            )
        self.timer = self.create_timer(0.1, self.timer_cb)

        self._publish_status(
            "started",
            {
                "response_mode": self.response_mode,
                "answer_topic": self.answer_topic,
                "qwen_query_topic": self.qwen_query_topic,
                "legacy_query_topic": self.legacy_query_topic,
                "execution_allowed": False,
                "memory_enabled": self.memory_enabled,
                "memory_worker_enabled": bool(
                    self.memory_bridge is not None
                    and self.memory_bridge.memory_worker is not None
                ),
            },
        )
        self.get_logger().info(
            "response_orchestrator_node started: "
            f"mode={self.response_mode}, answer={self.answer_topic}"
        )

    def identity_cb(self, msg: String) -> None:
        if self.memory_bridge is None:
            return
        try:
            payload = json.loads(msg.data)
            if not self.memory_bridge.update_identity(payload):
                self.get_logger().warning(
                    "ignored invalid speaker identity payload"
                )
        except Exception as exc:
            self.get_logger().warning(
                f"invalid speaker identity payload: {exc}"
            )

    def query_cb(self, msg: String) -> None:
        query = (msg.data or "").strip()
        if not query:
            return
        if self._pending is not None:
            if len(self._queue) >= self.max_pending_queries:
                self._publish_status("busy", {"reason": "queue_full"})
                return
            self._queue.append(query)
            self._publish_status("queued", {"queue_size": len(self._queue)})
            return
        self._start_query(query)

    def _visual_session_active(self) -> bool:
        if self._last_visual_completed_monotonic is None:
            return False
        return (
            time.monotonic() - self._last_visual_completed_monotonic
            <= self.visual_session_ttl_sec
        )

    def _start_query(self, query: str) -> None:
        request_id = f"resp-{uuid.uuid4()}"
        route = classify_response_route(
            query,
            visual_session_active=self._visual_session_active(),
        )
        memory_record = None
        if self.memory_bridge is not None:
            try:
                memory_record = self.memory_bridge.before_request(
                    query,
                    metadata={
                        "request_id": request_id,
                        "route": route.route,
                    },
                )
            except Exception as exc:
                self.get_logger().error(
                    f"memory before_request failed: {exc}"
                )
                self._publish_status(
                    "memory_error",
                    {
                        "stage": "before_request",
                        "error": repr(exc),
                    },
                )

        now = time.monotonic()
        self._pending = PendingResponse(
            request_id=request_id,
            query=query,
            route=route,
            started_monotonic=now,
            qwen_deadline=now + self.qwen_timeout_sec,
            legacy_deadline=now + self.legacy_timeout_sec,
            memory_conversation_id=(
                memory_record.conversation_id
                if memory_record is not None
                else None
            ),
            memory_user_message_id=(
                memory_record.user_message_id
                if memory_record is not None
                else None
            ),
            memory_context=(
                memory_record.memory_context
                if memory_record is not None
                else None
            ),
        )

        self._publish_trace(
            "request_started",
            {
                "request_id": request_id,
                "query": query,
                "route": route.route,
                "qwen_mode": route.qwen_mode,
                "reason": route.reason,
                "response_mode": self.response_mode,
                "execution_allowed": False,
            },
        )

        if route.is_action:
            self._publish_json(
                self.action_request_pub,
                {
                    "schema": "robot_action_request",
                    "schema_version": 1,
                    "request_id": request_id,
                    "query": query,
                    "route": route.route,
                    "emergency": route.is_emergency,
                    "execution_allowed": False,
                    "status": "PENDING_POLICY_GATE",
                },
            )

        if route.is_emergency:
            self._finalize(
                answer="Запрос остановки принят системой.",
                source="deterministic_safety",
                details={"execution_confirmed": False},
            )
            return

        if self.response_mode == "legacy":
            self._send_legacy_request(request_id, query, route)
            return

        self._send_qwen_request(request_id, query, route)
        if (
            self.response_mode == "hybrid"
            and not route.is_action
            and route.route != "text"
        ):
            self._send_legacy_request(request_id, query, route)

    def _send_qwen_request(
        self,
        request_id: str,
        query: str,
        route: ResponseRoute,
    ) -> None:
        payload = build_qwen_request(
            request_id=request_id,
            query=query,
            route=route,
            capabilities=self.capabilities,
            memory_context=(
                self._pending.memory_context
                if self._pending is not None
                and self._pending.request_id == request_id
                else None
            ),
        )
        self._publish_json(self.qwen_query_pub, payload)

    def _send_legacy_request(
        self,
        request_id: str,
        query: str,
        route: ResponseRoute,
    ) -> None:
        self._publish_json(
            self.legacy_query_pub,
            build_legacy_request(
                request_id=request_id,
                query=query,
                route=route,
            ),
        )

    def qwen_candidate_cb(self, msg: String) -> None:
        candidate = self._parse_candidate(msg.data, "qwen")
        if candidate is None or not self._candidate_matches(candidate):
            return
        if bool(candidate.get("success", False)):
            answer = str(candidate.get("answer", "")).strip()
            if answer:
                if self._pending and self._pending.route.qwen_mode == "image":
                    self._last_visual_completed_monotonic = time.monotonic()
                self._finalize(
                    answer=answer,
                    source="qwen_local",
                    details={
                        "provider_mode": candidate.get("mode"),
                        "frame_id": candidate.get("frame_id"),
                    },
                )
                return
        if self._pending is not None:
            self._pending.qwen_failed = True
            self._try_fallback("qwen_failed")

    def legacy_candidate_cb(self, msg: String) -> None:
        candidate = self._parse_candidate(msg.data, "legacy")
        if candidate is None or not self._candidate_matches(candidate):
            return
        answer = str(candidate.get("answer", "")).strip()
        if not answer:
            return
        if self.response_mode == "legacy":
            self._finalize(answer=answer, source="legacy")
            return
        if self._pending is not None:
            self._pending.legacy_answer = answer
            if self._pending.qwen_failed:
                self._try_fallback("legacy_after_qwen_failure")

    def timer_cb(self) -> None:
        pending = self._pending
        if pending is None:
            return
        now = time.monotonic()
        if self.response_mode == "legacy" and now >= pending.legacy_deadline:
            self._finalize(
                answer="Не удалось сформировать ответ.",
                source="deterministic_error",
                details={"reason": "legacy_timeout"},
            )
            return
        if self.response_mode in {"hybrid", "qwen"} and now >= pending.qwen_deadline:
            self._try_fallback("qwen_timeout")

    def _try_fallback(self, reason: str) -> None:
        pending = self._pending
        if pending is None:
            return
        source = fallback_source(
            self.response_mode,
            bool(pending.legacy_answer),
        )
        if source == "legacy" and pending.legacy_answer:
            self._finalize(
                answer=pending.legacy_answer,
                source="legacy_fallback",
                details={"reason": reason},
            )
            return
        action_text = (
            "Я понял команду, но физическое выполнение пока не подключено."
            if pending.route.is_action
            else "Не удалось получить ответ от локальной модели."
        )
        self._finalize(
            answer=action_text,
            source="deterministic_error",
            details={"reason": reason},
        )

    def _candidate_matches(self, candidate: Dict[str, Any]) -> bool:
        request_id = str(candidate.get("request_id", ""))
        if request_id in self._completed_ids:
            return False
        return self._pending is not None and request_id == self._pending.request_id

    def _parse_candidate(
        self,
        raw: str,
        expected_source: str,
    ) -> Optional[Dict[str, Any]]:
        try:
            candidate = json.loads(raw)
            if not isinstance(candidate, dict):
                raise ValueError("candidate must be an object")
            return candidate
        except Exception as exc:
            self.get_logger().warning(
                f"invalid {expected_source} candidate: {exc}"
            )
            return None

    def _finalize(
        self,
        *,
        answer: str,
        source: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        pending = self._pending
        if pending is None:
            return
        request_id = pending.request_id
        elapsed = time.monotonic() - pending.started_monotonic
        memory_assistant_message_id = None
        if (
            self.memory_bridge is not None
            and pending.memory_conversation_id is not None
            and pending.memory_user_message_id is not None
        ):
            try:
                memory_assistant_message_id = (
                    self.memory_bridge.after_response(
                        conversation_id=(
                            pending.memory_conversation_id
                        ),
                        reply_to_message_id=(
                            pending.memory_user_message_id
                        ),
                        answer=answer,
                        provider=source,
                        provider_message_id=request_id,
                        metadata={
                            "route": pending.route.route,
                            **(details or {}),
                        },
                    )
                )
            except Exception as exc:
                self.get_logger().error(
                    f"memory after_response failed: {exc}"
                )
                self._publish_status(
                    "memory_error",
                    {
                        "stage": "after_response",
                        "error": repr(exc),
                    },
                )
        self._publish_string(self.answer_pub, answer)
        self._publish_trace(
            "response_selected",
            {
                "request_id": request_id,
                "query": pending.query,
                "route": pending.route.route,
                "source": source,
                "answer": answer,
                "elapsed_sec": round(elapsed, 4),
                "execution_allowed": False,
                "memory_conversation_id": (
                    pending.memory_conversation_id
                ),
                "memory_user_message_id": (
                    pending.memory_user_message_id
                ),
                "memory_assistant_message_id": (
                    memory_assistant_message_id
                ),
                **(details or {}),
            },
        )
        self._completed_ids.append(request_id)
        self._pending = None
        if self._queue:
            next_query = self._queue.popleft()
            self._start_query(next_query)

    def _publish_trace(self, event: str, payload: Dict[str, Any]) -> None:
        self._publish_json(
            self.trace_pub,
            {"event": event, "timestamp": time.time(), **payload},
        )

    def _publish_status(self, state: str, payload: Dict[str, Any]) -> None:
        self._publish_json(
            self.status_pub,
            {"state": state, "timestamp": time.time(), **payload},
        )

    @staticmethod
    def _publish_string(publisher, text: str) -> None:
        msg = String()
        msg.data = text
        publisher.publish(msg)

    @staticmethod
    def _publish_json(publisher, payload: Dict[str, Any]) -> None:
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        publisher.publish(msg)

    def destroy_node(self) -> None:
        if self.memory_bridge is not None:
            try:
                self.memory_bridge.close()
            except Exception:
                pass
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ResponseOrchestratorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
