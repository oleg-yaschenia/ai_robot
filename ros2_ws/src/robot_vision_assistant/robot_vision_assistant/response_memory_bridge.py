from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from robot_vision_assistant.conversation_session_manager import (
    ConversationSessionManager,
    SessionPolicy,
)
from robot_vision_assistant.conversation_summary_worker import (
    ConversationSummaryWorker,
)
from robot_vision_assistant.deterministic_memory_worker import (
    DeterministicMemoryWorker,
)
from robot_vision_assistant.memory_store import MemoryStore


@dataclass(frozen=True)
class MemoryRequestRecord:
    conversation_id: str
    user_message_id: int
    speaker_entity_id: str
    memory_context: Dict[str, Any]


@dataclass
class SpeakerIdentity:
    entity_id: str
    name: str = ""
    role: str = ""
    confidence: Optional[float] = None
    source: str = ""
    status: str = "unknown"
    updated_monotonic: float = 0.0


class ResponseMemoryBridge:
    """Provider-neutral bridge between dialogue routing and local memory."""

    def __init__(
        self,
        db_path: str,
        *,
        default_speaker_entity_id: str = "person:unknown",
        identity_ttl_sec: float = 15.0,
        inactivity_timeout_sec: float = 15.0 * 60.0,
        recent_message_limit: int = 10,
        personal_fact_limit: int = 6,
        max_context_chars: int = 4200,
        memory_worker_enabled: bool = False,
        memory_worker_queue_size: int = 128,
        conversation_summary_enabled: bool = False,
        conversation_summary_queue_size: int = 64,
        conversation_summary_min_messages: int = 8,
        monotonic_clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not db_path.strip():
            raise ValueError("memory db_path must not be empty")

        self._monotonic_clock = monotonic_clock
        self.identity_ttl_sec = max(0.0, float(identity_ttl_sec))
        self.max_context_chars = max(1200, int(max_context_chars))
        self.default_speaker_entity_id = (
            default_speaker_entity_id.strip() or "person:unknown"
        )

        self.store = MemoryStore(db_path)
        self.manager = ConversationSessionManager(
            self.store,
            policy=SessionPolicy(
                inactivity_timeout_sec=max(
                    1.0,
                    float(inactivity_timeout_sec),
                ),
                recent_message_limit=max(
                    2,
                    int(recent_message_limit),
                ),
                personal_fact_limit=max(
                    0,
                    int(personal_fact_limit),
                ),
            ),
            monotonic_clock=monotonic_clock,
        )
        self.memory_worker: Optional[
            DeterministicMemoryWorker
        ] = None
        if memory_worker_enabled:
            self.memory_worker = DeterministicMemoryWorker(
                self.store,
                queue_size=memory_worker_queue_size,
            )
        self.conversation_summary_worker: Optional[
            ConversationSummaryWorker
        ] = None
        if conversation_summary_enabled:
            self.conversation_summary_worker = ConversationSummaryWorker(
                self.store,
                queue_size=conversation_summary_queue_size,
                min_messages=conversation_summary_min_messages,
            )
        self._identity = SpeakerIdentity(
            entity_id=self.default_speaker_entity_id,
            name="Неизвестный собеседник",
            status="unknown",
            updated_monotonic=0.0,
        )

    @staticmethod
    def _clean_text(value: Any, limit: int) -> str:
        text = " ".join(str(value or "").split())
        if len(text) <= limit:
            return text
        return text[: max(0, limit - 1)].rstrip() + "…"

    def update_identity(self, payload: Dict[str, Any]) -> bool:
        if not isinstance(payload, dict):
            return False

        entity_id = str(payload.get("entity_id", "")).strip()
        status = str(payload.get("status", "unknown")).strip().lower()
        if not entity_id:
            return False
        if status not in {"confirmed", "probable", "unknown"}:
            return False

        confidence_value = payload.get("confidence")
        confidence: Optional[float] = None
        if isinstance(confidence_value, (int, float)):
            confidence = max(0.0, min(float(confidence_value), 1.0))

        self._identity = SpeakerIdentity(
            entity_id=entity_id,
            name=self._clean_text(payload.get("name"), 80),
            role=self._clean_text(payload.get("role"), 80),
            confidence=confidence,
            source=self._clean_text(payload.get("source"), 80),
            status=status,
            updated_monotonic=self._monotonic_clock(),
        )
        return True

    def _current_identity(self) -> SpeakerIdentity:
        identity = self._identity
        if identity.status == "unknown":
            return SpeakerIdentity(
                entity_id=self.default_speaker_entity_id,
                name="Неизвестный собеседник",
                status="unknown",
            )

        age_sec = (
            self._monotonic_clock()
            - identity.updated_monotonic
        )
        if age_sec < 0.0 or age_sec > self.identity_ttl_sec:
            return SpeakerIdentity(
                entity_id=self.default_speaker_entity_id,
                name="Неизвестный собеседник",
                status="unknown",
            )
        return identity

    def before_request(
        self,
        query: str,
        *,
        provider: str = "user_input",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MemoryRequestRecord:
        identity = self._current_identity()
        turn = self.manager.record_user_message(
            speaker_entity_id=identity.entity_id,
            name=identity.name,
            role=identity.role,
            content=query,
            identity_confidence=identity.confidence,
            identity_source=identity.source,
            provider=provider,
            metadata={
                "identity_status": identity.status,
                **(metadata or {}),
            },
        )
        context = self.manager.build_context(
            current_speaker_entity_id=identity.entity_id,
            conversation_id=turn.conversation_id,
        )
        if self.memory_worker is not None:
            self.memory_worker.submit(
                conversation_id=turn.conversation_id,
                message_id=turn.message_id,
                speaker_entity_id=identity.entity_id,
                identity_status=identity.status,
                text=query,
            )
        return MemoryRequestRecord(
            conversation_id=turn.conversation_id,
            user_message_id=turn.message_id,
            speaker_entity_id=identity.entity_id,
            memory_context=self._compact_context(context),
        )

    def after_response(
        self,
        *,
        conversation_id: str,
        reply_to_message_id: int,
        answer: str,
        provider: str,
        provider_message_id: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        turn = self.manager.record_assistant_message(
            conversation_id=conversation_id,
            content=answer,
            provider=provider,
            provider_message_id=provider_message_id,
            reply_to_message_id=reply_to_message_id,
            metadata=metadata,
        )
        if self.conversation_summary_worker is not None:
            self.conversation_summary_worker.submit(conversation_id)
        return turn.message_id

    @staticmethod
    def _compact_participant(item: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "entity_id": item.get("entity_id"),
            "name": item.get("name") or "",
            "role": item.get("role") or "",
            "identity_confidence": item.get(
                "identity_confidence"
            ),
        }

    def _compact_context(
        self,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        conversation = context.get("conversation")
        if not isinstance(conversation, dict):
            conversation = {}

        current_speaker = context.get("current_speaker")
        if not isinstance(current_speaker, dict):
            current_speaker = {}

        participants_source = context.get("participants")
        participants: List[Dict[str, Any]] = []
        if isinstance(participants_source, list):
            participants = [
                self._compact_participant(item)
                for item in participants_source[:6]
                if isinstance(item, dict)
            ]

        messages_source = context.get("recent_messages")
        messages: List[Dict[str, Any]] = []
        if isinstance(messages_source, list):
            for item in messages_source[-10:]:
                if not isinstance(item, dict):
                    continue
                messages.append(
                    {
                        "message_id": item.get("message_id"),
                        "role": item.get("role"),
                        "speaker_entity_id": item.get(
                            "speaker_entity_id"
                        ),
                        "speaker_name": self._clean_text(
                            item.get("speaker_name"),
                            80,
                        ),
                        "content": self._clean_text(
                            item.get("content"),
                            320,
                        ),
                        "provider": item.get("provider") or "",
                        "reply_to_message_id": item.get(
                            "reply_to_message_id"
                        ),
                        "topic_thread_id": item.get(
                            "topic_thread_id"
                        )
                        or "",
                    }
                )

        facts_source = context.get("current_speaker_facts")
        facts: List[Dict[str, Any]] = []
        if isinstance(facts_source, list):
            for item in facts_source[:6]:
                if not isinstance(item, dict):
                    continue
                facts.append(
                    {
                        "predicate": item.get("predicate"),
                        "value": item.get("value"),
                        "fact_type": item.get("fact_type"),
                        "confidence": item.get("confidence"),
                        "importance": item.get("importance"),
                        "confirmed_by_user": bool(
                            item.get("confirmed_by_user")
                        ),
                    }
                )

        current_summary_source = context.get(
            "current_conversation_summary"
        )
        current_summary_compact: Optional[Dict[str, Any]] = None
        if isinstance(current_summary_source, dict):
            topics = current_summary_source.get("open_topics")
            current_summary_compact = {
                "summary": self._clean_text(
                    current_summary_source.get("summary"),
                    700,
                ),
                "open_topics": (
                    [
                        self._clean_text(topic, 180)
                        for topic in topics[:4]
                    ]
                    if isinstance(topics, list)
                    else []
                ),
                "covered_until_message_id": (
                    current_summary_source.get(
                        "covered_until_message_id"
                    )
                ),
            }

        previous = context.get(
            "current_speaker_previous_conversation"
        )
        previous_compact: Optional[Dict[str, Any]] = None
        if isinstance(previous, dict):
            topics = previous.get("open_topics")
            previous_compact = {
                "last_activity_at": previous.get(
                    "last_activity_at"
                ),
                "summary": self._clean_text(
                    previous.get("summary"),
                    700,
                ),
                "open_topics": (
                    [
                        self._clean_text(topic, 180)
                        for topic in topics[:4]
                    ]
                    if isinstance(topics, list)
                    else []
                ),
            }

        compact: Dict[str, Any] = {
            "schema": "robot_memory_context",
            "schema_version": 1,
            "conversation": {
                "conversation_id": conversation.get(
                    "conversation_id"
                ),
                "active_topic": conversation.get(
                    "active_topic"
                )
                or "",
            },
            "current_speaker": {
                "entity_id": current_speaker.get("id"),
                "name": current_speaker.get("name") or "",
                "role": current_speaker.get("role") or "",
            },
            "participants": participants,
            "recent_messages": messages,
            "current_conversation_summary": (
                current_summary_compact
            ),
            "previous_conversation_with_current_speaker": (
                previous_compact
            ),
            "current_speaker_facts": facts,
        }

        def serialized_length() -> int:
            return len(
                json.dumps(
                    compact,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )

        while (
            serialized_length() > self.max_context_chars
            and len(compact["recent_messages"]) > 3
        ):
            compact["recent_messages"].pop(0)

        if serialized_length() > self.max_context_chars:
            compact["current_speaker_facts"] = (
                compact["current_speaker_facts"][:3]
            )

        if serialized_length() > self.max_context_chars:
            current_summary_value = compact[
                "current_conversation_summary"
            ]
            if isinstance(current_summary_value, dict):
                current_summary_value["summary"] = self._clean_text(
                    current_summary_value.get("summary"),
                    260,
                )
                current_summary_value["open_topics"] = (
                    current_summary_value.get("open_topics") or []
                )[:2]

        if serialized_length() > self.max_context_chars:
            previous_value = compact[
                "previous_conversation_with_current_speaker"
            ]
            if isinstance(previous_value, dict):
                previous_value["summary"] = self._clean_text(
                    previous_value.get("summary"),
                    280,
                )
                previous_value["open_topics"] = (
                    previous_value.get("open_topics") or []
                )[:2]

        if serialized_length() > self.max_context_chars:
            compact["participants"] = compact["participants"][:4]

        return compact

    def flush_memory_worker(
        self,
        timeout_sec: float = 5.0,
    ) -> bool:
        if self.memory_worker is None:
            return True
        return self.memory_worker.flush(timeout_sec=timeout_sec)

    def memory_worker_stats(self) -> Dict[str, int]:
        if self.memory_worker is None:
            return {}
        return self.memory_worker.stats()

    def flush_conversation_summary_worker(
        self,
        timeout_sec: float = 5.0,
    ) -> bool:
        if self.conversation_summary_worker is None:
            return True
        return self.conversation_summary_worker.flush(
            timeout_sec=timeout_sec
        )

    def conversation_summary_worker_stats(self) -> Dict[str, int]:
        if self.conversation_summary_worker is None:
            return {}
        return self.conversation_summary_worker.stats()

    def close(self) -> None:
        if self.memory_worker is not None:
            self.memory_worker.close(flush=True)
        if self.conversation_summary_worker is not None:
            self.conversation_summary_worker.close(flush=True)
        self.store.close()
