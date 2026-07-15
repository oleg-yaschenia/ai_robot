from __future__ import annotations

import queue
import re
import threading
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

from robot_vision_assistant.memory_store import MemoryStore


@dataclass(frozen=True)
class MemoryCandidate:
    predicate: str
    value: Dict[str, Any]
    fact_type: str
    confidence: float
    importance: float
    source_type: str
    privacy_scope: str
    confirmed_by_user: bool
    dedupe_key: str


@dataclass(frozen=True)
class MemoryWorkItem:
    conversation_id: str
    message_id: int
    speaker_entity_id: str
    identity_status: str
    text: str


@dataclass(frozen=True)
class _FlushBarrier:
    event: threading.Event


class _Stop:
    pass


_STOP = _Stop()
QueueEntry = Union[MemoryWorkItem, _FlushBarrier, _Stop]


class DeterministicMemoryExtractor:
    """Extract only explicit user-approved facts and corrections.

    This deliberately avoids free-form inference. It will not promote ordinary
    dialogue or model guesses into long-term memory.
    """

    MAX_INPUT_CHARS = 1200
    MAX_VALUE_CHARS = 700

    _DO_NOT_REMEMBER_RE = re.compile(
        r"\b(?:не\s+(?:надо\s+)?"
        r"(?:запоминай|запоминать|сохраняй|сохранять)"
        r"|забудь)\b",
        re.IGNORECASE,
    )

    _EXPLICIT_MEMORY_PATTERNS = (
        re.compile(
            r"^\s*(?:пожалуйста[,\s]+)?"
            r"(?:запомни|помни)"
            r"(?:\s+пожалуйста)?"
            r"(?:\s*[:,\-—]\s*(?:что\s+)?|\s+что\s+)"
            r"(?P<value>.+?)\s*$",
            re.IGNORECASE,
        ),
        re.compile(
            r"^\s*(?:пожалуйста[,\s]+)?"
            r"сохрани(?:\s+это)?\s+в\s+памяти"
            r"(?:\s*[:,\-—]\s*(?:что\s+)?|\s+что\s+)"
            r"(?P<value>.+?)\s*$",
            re.IGNORECASE,
        ),
    )

    _CORRECTION_PATTERNS = (
        re.compile(
            r"^\s*ты\s+должен\s+был\s+"
            r"(?P<value>.+?)\s*$",
            re.IGNORECASE,
        ),
        re.compile(
            r"^\s*ты\s+(?:сделал|ответил|понял|выполнил)"
            r"\s+неправильно\s*[:,\-—]?\s*"
            r"(?:нужно|надо|следовало)\s+"
            r"(?P<value>.+?)\s*$",
            re.IGNORECASE,
        ),
        re.compile(
            r"^\s*в\s+следующий\s+раз\s+"
            r"(?:ты\s+)?(?:должен|нужно|надо)\s+"
            r"(?P<value>.+?)\s*$",
            re.IGNORECASE,
        ),
    )

    @staticmethod
    def _clean_text(value: str) -> str:
        text = " ".join(value.strip().split())
        return text.strip(" \t\r\n.,;:!—-")

    @staticmethod
    def _dedupe_text(value: str) -> str:
        normalized = unicodedata.normalize("NFKC", value)
        normalized = normalized.casefold().replace("ё", "е")
        normalized = re.sub(r"[^\w]+", " ", normalized, flags=re.UNICODE)
        return " ".join(normalized.split())

    def extract(self, text: str) -> Optional[MemoryCandidate]:
        source = " ".join(str(text or "").split())
        if not source or len(source) > self.MAX_INPUT_CHARS:
            return None
        if self._DO_NOT_REMEMBER_RE.search(source):
            return None

        for pattern in self._EXPLICIT_MEMORY_PATTERNS:
            match = pattern.match(source)
            if match is None:
                continue
            value = self._clean_text(match.group("value"))
            if not self._valid_value(value):
                return None
            normalized = self._dedupe_text(value)
            return MemoryCandidate(
                predicate="explicit_memory",
                value={"statement": value},
                fact_type="explicit_fact",
                confidence=1.0,
                importance=0.90,
                source_type="explicit_user_statement",
                privacy_scope="person_private",
                confirmed_by_user=True,
                dedupe_key=f"explicit_fact:{normalized}",
            )

        for pattern in self._CORRECTION_PATTERNS:
            match = pattern.match(source)
            if match is None:
                continue
            value = self._clean_text(match.group("value"))
            if not self._valid_value(value):
                return None
            normalized = self._dedupe_text(value)
            return MemoryCandidate(
                predicate="behavior_correction",
                value={"instruction": value},
                fact_type="correction",
                confidence=1.0,
                importance=0.95,
                source_type="explicit_user_correction",
                privacy_scope="person_private",
                confirmed_by_user=True,
                dedupe_key=f"correction:{normalized}",
            )

        return None

    def _valid_value(self, value: str) -> bool:
        return (
            3 <= len(value) <= self.MAX_VALUE_CHARS
            and not value.endswith("?")
            and bool(self._dedupe_text(value))
        )


class DeterministicMemoryWorker:
    """Lightweight background worker for explicit long-term memory."""

    def __init__(
        self,
        store: MemoryStore,
        *,
        queue_size: int = 128,
        extractor: Optional[DeterministicMemoryExtractor] = None,
    ) -> None:
        self.store = store
        self.extractor = extractor or DeterministicMemoryExtractor()
        self._queue: queue.Queue[QueueEntry] = queue.Queue(
            maxsize=max(8, int(queue_size))
        )
        self._stats_lock = threading.Lock()
        self._stats: Dict[str, int] = {
            "submitted": 0,
            "stored": 0,
            "deduplicated": 0,
            "ignored": 0,
            "skipped_identity": 0,
            "queue_full": 0,
            "failed": 0,
        }
        self._closed = False
        self._thread = threading.Thread(
            target=self._run,
            name="deterministic-memory-worker",
            daemon=True,
        )
        self._thread.start()

    def _increment(self, key: str) -> None:
        with self._stats_lock:
            self._stats[key] = self._stats.get(key, 0) + 1

    def stats(self) -> Dict[str, int]:
        with self._stats_lock:
            return dict(self._stats)

    def submit(
        self,
        *,
        conversation_id: str,
        message_id: int,
        speaker_entity_id: str,
        identity_status: str,
        text: str,
    ) -> bool:
        if self._closed:
            return False
        item = MemoryWorkItem(
            conversation_id=conversation_id,
            message_id=int(message_id),
            speaker_entity_id=speaker_entity_id,
            identity_status=identity_status,
            text=text,
        )
        try:
            self._queue.put_nowait(item)
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
                self._process(entry)
            except Exception:
                self._increment("failed")
            finally:
                self._queue.task_done()

    def _process(self, item: MemoryWorkItem) -> None:
        if (
            item.identity_status != "confirmed"
            or not item.speaker_entity_id
            or item.speaker_entity_id == "person:unknown"
        ):
            self._increment("skipped_identity")
            return

        candidate = self.extractor.extract(item.text)
        if candidate is None:
            self._increment("ignored")
            return

        _, created = self.store.upsert_memory_fact_deduplicated(
            subject_entity_id=item.speaker_entity_id,
            predicate=candidate.predicate,
            value=candidate.value,
            fact_type=candidate.fact_type,
            confidence=candidate.confidence,
            importance=candidate.importance,
            source_type=candidate.source_type,
            source_message_id=item.message_id,
            privacy_scope=candidate.privacy_scope,
            confirmed_by_user=candidate.confirmed_by_user,
            dedupe_key=candidate.dedupe_key,
        )
        self._increment("stored" if created else "deduplicated")

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
