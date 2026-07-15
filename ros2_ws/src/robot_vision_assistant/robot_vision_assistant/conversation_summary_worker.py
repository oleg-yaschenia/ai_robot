from __future__ import annotations

import queue
import re
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from robot_vision_assistant.memory_store import MemoryStore


@dataclass(frozen=True)
class ConversationSummaryWorkItem:
    conversation_id: str


@dataclass(frozen=True)
class _FlushBarrier:
    event: threading.Event


class _Stop:
    pass


_STOP = _Stop()
QueueEntry = Union[ConversationSummaryWorkItem, _FlushBarrier, _Stop]


class DeterministicConversationSummarizer:
    """Build a lightweight extractive summary without calling an LLM."""

    MAX_MESSAGE_CHARS = 180
    MAX_SUMMARY_CHARS = 900
    MAX_OPEN_TOPIC_CHARS = 180

    _OPEN_TOPIC_RE = re.compile(
        r"(?:\?|\b(?:давай|надо|нужно|сделай|сделать|проверь|проверить|"
        r"исправь|исправить|реализуй|реализовать|продолжим|следующий\s+шаг)\b)",
        re.IGNORECASE,
    )
    _IGNORE_TOPIC_RE = re.compile(
        r"^\s*(?:запомни|помни|сохрани\s+.*памяти|ты\s+должен\s+был)\b",
        re.IGNORECASE,
    )

    @staticmethod
    def _clean(value: Any, limit: int) -> str:
        text = " ".join(str(value or "").split())
        text = text.strip()
        if len(text) <= limit:
            return text
        return text[: max(0, limit - 1)].rstrip() + "…"

    @staticmethod
    def _topic_key(value: str) -> str:
        lowered = value.casefold().replace("ё", "е")
        lowered = re.sub(r"[^\wа-я]+", " ", lowered, flags=re.IGNORECASE)
        return " ".join(lowered.split())

    def build_summary(
        self,
        *,
        previous_summary: str = "",
        messages: List[Dict[str, Any]],
    ) -> str:
        useful: List[str] = []
        for message in messages[-10:]:
            role = str(message.get("role") or "")
            if role not in {"user", "assistant"}:
                continue
            content = self._clean(
                message.get("content") or "",
                self.MAX_MESSAGE_CHARS,
            )
            if not content:
                continue
            speaker = "Пользователь" if role == "user" else "Робот"
            speaker_id = str(message.get("speaker_entity_id") or "")
            if speaker_id == "person:oleg":
                speaker = "Олег"
            elif speaker_id == "person:unknown":
                speaker = "Неизвестный"
            useful.append(f"{speaker}: {content}")

        if not useful:
            return self._clean(previous_summary, self.MAX_SUMMARY_CHARS)

        recent = "; ".join(useful)
        recent_segment = f"Последние реплики: {recent}"

        previous_clean = self._clean(
            previous_summary,
            self.MAX_SUMMARY_CHARS,
        ).strip()
        if not previous_clean:
            return self._clean(
                recent_segment,
                self.MAX_SUMMARY_CHARS,
            )

        new_segment = f"Новые реплики: {recent}"
        separator = " | "
        available_for_previous = (
            self.MAX_SUMMARY_CHARS
            - len(separator)
            - len(new_segment)
        )

        if available_for_previous < 60:
            return self._clean(
                new_segment,
                self.MAX_SUMMARY_CHARS,
            )

        previous_compact = self._clean(
            previous_clean,
            available_for_previous,
        )
        return (
            f"{previous_compact}{separator}{new_segment}"
        )

    def extract_open_topics(
        self,
        messages: List[Dict[str, Any]],
        *,
        max_topics: int = 5,
    ) -> List[str]:
        topics_reversed: List[str] = []
        seen = set()
        for message in reversed(messages):
            if str(message.get("role") or "") != "user":
                continue
            content = self._clean(
                message.get("content") or "",
                self.MAX_OPEN_TOPIC_CHARS,
            )
            if not content:
                continue
            if self._IGNORE_TOPIC_RE.search(content):
                continue
            if not self._OPEN_TOPIC_RE.search(content):
                continue
            key = self._topic_key(content)
            if not key or key in seen:
                continue
            seen.add(key)
            topics_reversed.append(content)
            if len(topics_reversed) >= max_topics:
                break
        return list(reversed(topics_reversed))


class ConversationSummaryWorker:
    """Background updater for conversation_summaries/open topics."""

    def __init__(
        self,
        store: MemoryStore,
        *,
        queue_size: int = 64,
        min_messages: int = 8,
        message_limit: int = 80,
        max_open_topics: int = 5,
        summarizer: Optional[DeterministicConversationSummarizer] = None,
    ) -> None:
        self.store = store
        self.min_messages = max(2, int(min_messages))
        self.message_limit = max(10, min(int(message_limit), 200))
        self.max_open_topics = max(1, min(int(max_open_topics), 10))
        self.summarizer = summarizer or DeterministicConversationSummarizer()
        self._queue: queue.Queue[QueueEntry] = queue.Queue(
            maxsize=max(4, int(queue_size))
        )
        self._stats_lock = threading.Lock()
        self._stats: Dict[str, int] = {
            "submitted": 0,
            "updated": 0,
            "ignored": 0,
            "queue_full": 0,
            "failed": 0,
        }
        self._closed = False
        self._thread = threading.Thread(
            target=self._run,
            name="conversation-summary-worker",
            daemon=True,
        )
        self._thread.start()

    def _increment(self, key: str) -> None:
        with self._stats_lock:
            self._stats[key] = self._stats.get(key, 0) + 1

    def stats(self) -> Dict[str, int]:
        with self._stats_lock:
            return dict(self._stats)

    def submit(self, conversation_id: str) -> bool:
        if self._closed or not str(conversation_id or "").strip():
            return False
        try:
            self._queue.put_nowait(
                ConversationSummaryWorkItem(
                    conversation_id=str(conversation_id),
                )
            )
        except queue.Full:
            self._increment("queue_full")
            return False
        self._increment("submitted")
        return True

    def _run(self) -> None:
        while True:
            entry = self._queue.get()
            try:
                if entry is _STOP:
                    return
                if isinstance(entry, _FlushBarrier):
                    entry.event.set()
                    continue
                self._process(entry.conversation_id)
            except Exception:
                self._increment("failed")
            finally:
                self._queue.task_done()

    def _process(self, conversation_id: str) -> None:
        message_count = self.store.get_conversation_message_count(
            conversation_id
        )
        if message_count < self.min_messages:
            self._increment("ignored")
            return

        existing = self.store.get_conversation_summary(conversation_id)
        covered_until = 0
        previous_summary = ""
        if existing is not None:
            covered_until = int(
                existing.get("covered_until_message_id") or 0
            )
            previous_summary = str(existing.get("summary") or "")

        new_messages = self.store.get_messages_after(
            conversation_id,
            after_message_id=covered_until,
            limit=self.message_limit,
        )
        if not new_messages:
            self._increment("ignored")
            return

        recent_messages = self.store.get_recent_messages(
            conversation_id,
            limit=self.message_limit,
        )
        if not recent_messages:
            self._increment("ignored")
            return

        summary = self.summarizer.build_summary(
            previous_summary=previous_summary,
            messages=new_messages,
        )
        open_topics = self.summarizer.extract_open_topics(
            recent_messages,
            max_topics=self.max_open_topics,
        )
        latest_id = max(int(item["id"]) for item in new_messages)
        self.store.upsert_conversation_summary(
            conversation_id,
            summary=summary,
            open_topics=open_topics,
            covered_until_message_id=latest_id,
        )
        self._increment("updated")

    def flush(self, timeout_sec: float = 5.0) -> bool:
        if self._closed:
            return True
        barrier = _FlushBarrier(event=threading.Event())
        try:
            self._queue.put(barrier, timeout=max(0.1, timeout_sec))
        except queue.Full:
            return False
        return barrier.event.wait(timeout=max(0.1, timeout_sec))

    def close(
        self,
        *,
        flush: bool = True,
        timeout_sec: float = 5.0,
    ) -> None:
        if self._closed:
            return
        if flush:
            self.flush(timeout_sec=timeout_sec)
        self._closed = True
        try:
            self._queue.put(_STOP, timeout=max(0.1, timeout_sec))
        except queue.Full:
            return
        self._thread.join(timeout=max(0.1, timeout_sec))
