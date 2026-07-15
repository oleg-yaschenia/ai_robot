from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from robot_vision_assistant.memory_store import MemoryStore


@dataclass(frozen=True)
class SessionPolicy:
    inactivity_timeout_sec: float = 15.0 * 60.0
    recent_message_limit: int = 12
    personal_fact_limit: int = 8


@dataclass(frozen=True)
class RecordedTurn:
    conversation_id: str
    message_id: int
    speaker_entity_id: str
    reply_to_message_id: Optional[int]
    topic_thread_id: str


class ConversationSessionManager:
    """Maintain one shared multi-party dialogue across speaker changes."""

    ROBOT_ENTITY_ID = "robot:self"

    def __init__(
        self,
        store: MemoryStore,
        *,
        policy: Optional[SessionPolicy] = None,
        monotonic_clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], datetime] = (
            lambda: datetime.now(timezone.utc)
        ),
    ) -> None:
        self.store = store
        self.policy = policy or SessionPolicy()
        self._monotonic_clock = monotonic_clock
        self._wall_clock = wall_clock
        self._active_conversation_id: Optional[str] = None
        self._last_activity_monotonic: Optional[float] = None
        self._current_speaker_entity_id: Optional[str] = None

        self.store.upsert_entity(
            self.ROBOT_ENTITY_ID,
            "robot",
            name="Робот",
            role="assistant",
            privacy_scope="system_internal",
        )

    @staticmethod
    def _parse_sqlite_timestamp(value: Any) -> Optional[datetime]:
        if not value:
            return None
        try:
            parsed = datetime.fromisoformat(str(value).strip())
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    def _database_conversation_is_recent(
        self,
        conversation: Dict[str, Any],
    ) -> bool:
        timestamp = self._parse_sqlite_timestamp(
            conversation.get("last_activity_at")
        )
        if timestamp is None:
            return False
        age_sec = (
            self._wall_clock().astimezone(timezone.utc) - timestamp
        ).total_seconds()
        return 0.0 <= age_sec <= self.policy.inactivity_timeout_sec

    def _runtime_session_is_recent(self) -> bool:
        if (
            self._active_conversation_id is None
            or self._last_activity_monotonic is None
        ):
            return False
        age_sec = (
            self._monotonic_clock()
            - self._last_activity_monotonic
        )
        return 0.0 <= age_sec <= self.policy.inactivity_timeout_sec

    def _touch(self) -> None:
        self._last_activity_monotonic = self._monotonic_clock()

    def ensure_conversation(
        self,
        *,
        title: str = "",
        location: str = "",
        force_new: bool = False,
    ) -> str:
        if force_new and self._active_conversation_id is not None:
            self.store.close_conversation(
                self._active_conversation_id
            )
            self._active_conversation_id = None
            self._last_activity_monotonic = None

        if self._runtime_session_is_recent():
            assert self._active_conversation_id is not None
            return self._active_conversation_id

        if self._active_conversation_id is not None:
            self.store.close_conversation(
                self._active_conversation_id
            )
            self._active_conversation_id = None
            self._last_activity_monotonic = None

        existing = self.store.get_active_conversation()
        if (
            not force_new
            and existing is not None
            and self._database_conversation_is_recent(existing)
        ):
            conversation_id = str(existing["id"])
        else:
            if existing is not None:
                self.store.close_conversation(str(existing["id"]))
            conversation_id = self.store.start_conversation(
                title=title,
                location=location,
            )

        self.store.add_participant(
            conversation_id,
            self.ROBOT_ENTITY_ID,
            identity_confidence=1.0,
            identity_source="system",
        )
        self._active_conversation_id = conversation_id
        self._touch()
        return conversation_id

    def observe_speaker(
        self,
        speaker_entity_id: str,
        *,
        name: str = "",
        role: str = "",
        identity_confidence: Optional[float] = None,
        identity_source: str = "",
        entity_type: str = "person",
        conversation_id: Optional[str] = None,
    ) -> str:
        if not speaker_entity_id.strip():
            raise ValueError("speaker_entity_id must not be empty")

        self.store.upsert_entity(
            speaker_entity_id,
            entity_type,
            name=name,
            role=role,
            privacy_scope="person_private",
        )
        active_id = conversation_id or self.ensure_conversation()
        self.store.add_participant(
            active_id,
            speaker_entity_id,
            identity_confidence=identity_confidence,
            identity_source=identity_source,
        )
        self._active_conversation_id = active_id
        self._current_speaker_entity_id = speaker_entity_id
        self._touch()
        return active_id

    def _latest_message(
        self,
        conversation_id: str,
    ) -> Optional[Dict[str, Any]]:
        messages = self.store.get_recent_messages(
            conversation_id,
            limit=1,
        )
        return messages[-1] if messages else None

    def _resolve_topic_thread(
        self,
        conversation_id: str,
        *,
        explicit_topic_thread_id: str,
        reply_to_message_id: Optional[int],
    ) -> str:
        if explicit_topic_thread_id:
            return explicit_topic_thread_id

        if reply_to_message_id is not None:
            replied = self.store.get_message(reply_to_message_id)
            if replied is not None:
                inherited = str(
                    replied.get("topic_thread_id") or ""
                )
                if inherited:
                    return inherited

        conversation = self.store.get_conversation(conversation_id)
        if conversation is None:
            return ""
        return str(conversation.get("active_topic") or "")

    def record_user_message(
        self,
        *,
        speaker_entity_id: str,
        content: str,
        name: str = "",
        role: str = "",
        identity_confidence: Optional[float] = None,
        identity_source: str = "",
        provider: str = "local_input",
        provider_message_id: str = "",
        reply_to_message_id: Optional[int] = None,
        topic_thread_id: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> RecordedTurn:
        conversation_id = self.observe_speaker(
            speaker_entity_id,
            name=name,
            role=role,
            identity_confidence=identity_confidence,
            identity_source=identity_source,
        )

        latest = self._latest_message(conversation_id)
        resolved_reply = reply_to_message_id
        if resolved_reply is None and latest is not None:
            resolved_reply = int(latest["id"])

        resolved_topic = self._resolve_topic_thread(
            conversation_id,
            explicit_topic_thread_id=topic_thread_id,
            reply_to_message_id=resolved_reply,
        )
        if resolved_topic:
            self.store.update_conversation_topic(
                conversation_id,
                resolved_topic,
            )

        message_id = self.store.add_message(
            conversation_id,
            role="user",
            speaker_entity_id=speaker_entity_id,
            content=content,
            provider=provider,
            provider_message_id=provider_message_id,
            reply_to_message_id=resolved_reply,
            topic_thread_id=resolved_topic,
            identity_confidence=identity_confidence,
            metadata=metadata,
        )
        self._current_speaker_entity_id = speaker_entity_id
        self._touch()
        return RecordedTurn(
            conversation_id=conversation_id,
            message_id=message_id,
            speaker_entity_id=speaker_entity_id,
            reply_to_message_id=resolved_reply,
            topic_thread_id=resolved_topic,
        )

    def record_assistant_message(
        self,
        *,
        conversation_id: str,
        content: str,
        provider: str,
        reply_to_message_id: Optional[int] = None,
        provider_message_id: str = "",
        topic_thread_id: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> RecordedTurn:
        latest = self._latest_message(conversation_id)
        resolved_reply = reply_to_message_id
        if resolved_reply is None and latest is not None:
            resolved_reply = int(latest["id"])

        resolved_topic = self._resolve_topic_thread(
            conversation_id,
            explicit_topic_thread_id=topic_thread_id,
            reply_to_message_id=resolved_reply,
        )

        message_id = self.store.add_message(
            conversation_id,
            role="assistant",
            speaker_entity_id=self.ROBOT_ENTITY_ID,
            content=content,
            provider=provider,
            provider_message_id=provider_message_id,
            reply_to_message_id=resolved_reply,
            topic_thread_id=resolved_topic,
            metadata=metadata,
        )
        self._active_conversation_id = conversation_id
        self._touch()
        return RecordedTurn(
            conversation_id=conversation_id,
            message_id=message_id,
            speaker_entity_id=self.ROBOT_ENTITY_ID,
            reply_to_message_id=resolved_reply,
            topic_thread_id=resolved_topic,
        )

    @staticmethod
    def _decode_fact_value(raw: Any) -> Any:
        try:
            return json.loads(str(raw))
        except (TypeError, ValueError, json.JSONDecodeError):
            return raw

    def build_context(
        self,
        *,
        current_speaker_entity_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        active_id = (
            conversation_id
            or self._active_conversation_id
            or self.ensure_conversation()
        )
        speaker_id = (
            current_speaker_entity_id
            or self._current_speaker_entity_id
        )

        participants = self.store.get_conversation_participants(
            active_id
        )
        name_by_id = {
            str(item["entity_id"]): str(item.get("name") or "")
            for item in participants
        }

        recent_messages: List[Dict[str, Any]] = []
        for message in self.store.get_recent_messages(
            active_id,
            limit=self.policy.recent_message_limit,
        ):
            speaker_entity_id = message.get("speaker_entity_id")
            recent_messages.append(
                {
                    "message_id": message["id"],
                    "role": message["role"],
                    "speaker_entity_id": speaker_entity_id,
                    "speaker_name": (
                        name_by_id.get(str(speaker_entity_id), "")
                        if speaker_entity_id
                        else ""
                    ),
                    "content": message["content"],
                    "provider": message["provider"],
                    "reply_to_message_id": (
                        message["reply_to_message_id"]
                    ),
                    "topic_thread_id": message["topic_thread_id"],
                    "timestamp": message["ts"],
                }
            )

        current_speaker = (
            self.store.get_entity(speaker_id)
            if speaker_id
            else None
        )
        previous_conversation = (
            self.store.get_previous_conversation_for_entity(
                speaker_id,
                exclude_conversation_id=active_id,
            )
            if speaker_id
            else None
        )

        personal_facts: List[Dict[str, Any]] = []
        if speaker_id:
            for fact in self.store.get_active_facts(
                subject_entity_id=speaker_id,
                limit=self.policy.personal_fact_limit,
            ):
                scope = str(fact.get("privacy_scope") or "")
                if scope == "system_internal":
                    continue
                personal_facts.append(
                    {
                        "predicate": fact["predicate"],
                        "value": self._decode_fact_value(
                            fact["value_json"]
                        ),
                        "fact_type": fact["fact_type"],
                        "confidence": fact["confidence"],
                        "importance": fact["importance"],
                        "privacy_scope": scope,
                        "confirmed_by_user": bool(
                            fact["confirmed_by_user"]
                        ),
                    }
                )

        current_summary_raw = self.store.get_conversation_summary(
            active_id
        )
        current_summary: Optional[Dict[str, Any]] = None
        if current_summary_raw is not None:
            try:
                current_open_topics = json.loads(
                    current_summary_raw.get("open_topics_json")
                    or "[]"
                )
            except (TypeError, ValueError, json.JSONDecodeError):
                current_open_topics = []
            current_summary = {
                "summary": current_summary_raw.get("summary") or "",
                "open_topics": current_open_topics,
                "covered_until_message_id": current_summary_raw.get(
                    "covered_until_message_id"
                ),
                "updated_at": current_summary_raw.get("updated_at"),
            }

        previous_context: Optional[Dict[str, Any]] = None
        if previous_conversation is not None:
            try:
                open_topics = json.loads(
                    previous_conversation.get(
                        "open_topics_json"
                    )
                    or "[]"
                )
            except (TypeError, ValueError, json.JSONDecodeError):
                open_topics = []
            previous_context = {
                "conversation_id": previous_conversation["id"],
                "last_activity_at": previous_conversation[
                    "last_activity_at"
                ],
                "summary": previous_conversation.get("summary") or "",
                "open_topics": open_topics,
            }

        conversation = self.store.get_conversation(active_id) or {}
        return {
            "schema": "multi_party_conversation_context",
            "schema_version": 1,
            "conversation": {
                "conversation_id": active_id,
                "started_at": conversation.get("started_at"),
                "last_activity_at": conversation.get(
                    "last_activity_at"
                ),
                "active_topic": conversation.get("active_topic") or "",
            },
            "participants": participants,
            "current_speaker": current_speaker,
            "recent_messages": recent_messages,
            "current_conversation_summary": current_summary,
            "current_speaker_previous_conversation": previous_context,
            "current_speaker_facts": personal_facts,
        }

    def end_conversation(self) -> None:
        if self._active_conversation_id is not None:
            self.store.close_conversation(
                self._active_conversation_id
            )
        self._active_conversation_id = None
        self._last_activity_monotonic = None
        self._current_speaker_entity_id = None
