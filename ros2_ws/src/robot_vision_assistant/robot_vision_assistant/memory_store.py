from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional


class MemoryStore:
    """Local source of truth for robot conversations and long-term memory.

    The v3 schema is additive: legacy tables remain untouched so existing
    interaction and scene-event data are preserved.
    """

    SCHEMA_VERSION = 3

    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,
            timeout=30.0,
        )
        self.conn.row_factory = sqlite3.Row
        with self._lock:
            self.conn.execute("PRAGMA foreign_keys=ON")
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA synchronous=NORMAL")
        self._init_db()

    def _init_db(self) -> None:
        with self._lock:
            cur = self.conn.cursor()

            # Existing v1 tables. Keep them for backward compatibility.
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS profiles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    entity_type TEXT NOT NULL,
                    name TEXT NOT NULL,
                    role TEXT DEFAULT '',
                    notes TEXT DEFAULT '',
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts DATETIME DEFAULT CURRENT_TIMESTAMP,
                    mode TEXT NOT NULL,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    snapshot_path TEXT DEFAULT ''
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS scene_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts DATETIME DEFAULT CURRENT_TIMESTAMP,
                    event_type TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    snapshot_path TEXT DEFAULT ''
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS assistant_state (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )

            # Unified conversation and memory schema.
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS entities (
                    id TEXT PRIMARY KEY,
                    entity_type TEXT NOT NULL,
                    name TEXT NOT NULL DEFAULT '',
                    role TEXT NOT NULL DEFAULT '',
                    notes TEXT NOT NULL DEFAULT '',
                    privacy_scope TEXT NOT NULL DEFAULT 'public_household',
                    active INTEGER NOT NULL DEFAULT 1,
                    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    started_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    ended_at DATETIME,
                    last_activity_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    title TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'active',
                    location TEXT NOT NULL DEFAULT '',
                    active_topic TEXT NOT NULL DEFAULT '',
                    created_by TEXT NOT NULL DEFAULT 'robot'
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_participants (
                    conversation_id TEXT NOT NULL,
                    entity_id TEXT NOT NULL,
                    joined_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    left_at DATETIME,
                    identity_confidence REAL,
                    identity_source TEXT NOT NULL DEFAULT '',
                    PRIMARY KEY (conversation_id, entity_id),
                    FOREIGN KEY (conversation_id)
                        REFERENCES conversations(id) ON DELETE CASCADE,
                    FOREIGN KEY (entity_id)
                        REFERENCES entities(id) ON DELETE CASCADE
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    ts DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    speaker_entity_id TEXT,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    provider TEXT NOT NULL DEFAULT '',
                    provider_message_id TEXT NOT NULL DEFAULT '',
                    reply_to_message_id INTEGER,
                    topic_thread_id TEXT NOT NULL DEFAULT '',
                    identity_confidence REAL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    FOREIGN KEY (conversation_id)
                        REFERENCES conversations(id) ON DELETE CASCADE,
                    FOREIGN KEY (speaker_entity_id)
                        REFERENCES entities(id) ON DELETE SET NULL,
                    FOREIGN KEY (reply_to_message_id)
                        REFERENCES messages(id) ON DELETE SET NULL
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_summaries (
                    conversation_id TEXT PRIMARY KEY,
                    summary TEXT NOT NULL DEFAULT '',
                    open_topics_json TEXT NOT NULL DEFAULT '[]',
                    covered_until_message_id INTEGER,
                    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (conversation_id)
                        REFERENCES conversations(id) ON DELETE CASCADE,
                    FOREIGN KEY (covered_until_message_id)
                        REFERENCES messages(id) ON DELETE SET NULL
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    subject_entity_id TEXT,
                    predicate TEXT NOT NULL,
                    value_json TEXT NOT NULL,
                    dedupe_key TEXT NOT NULL DEFAULT '',
                    fact_type TEXT NOT NULL DEFAULT 'general',
                    confidence REAL NOT NULL DEFAULT 1.0,
                    importance REAL NOT NULL DEFAULT 0.5,
                    source_type TEXT NOT NULL DEFAULT 'conversation',
                    source_message_id INTEGER,
                    privacy_scope TEXT NOT NULL DEFAULT 'public_household',
                    confirmed_by_user INTEGER NOT NULL DEFAULT 0,
                    valid_from DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    valid_to DATETIME,
                    active INTEGER NOT NULL DEFAULT 1,
                    supersedes_fact_id INTEGER,
                    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (subject_entity_id)
                        REFERENCES entities(id) ON DELETE SET NULL,
                    FOREIGN KEY (source_message_id)
                        REFERENCES messages(id) ON DELETE SET NULL,
                    FOREIGN KEY (supersedes_fact_id)
                        REFERENCES memory_facts(id) ON DELETE SET NULL
                )
                """
            )
            memory_fact_columns = {
                str(row["name"])
                for row in cur.execute(
                    "PRAGMA table_info(memory_facts)"
                ).fetchall()
            }
            if "dedupe_key" not in memory_fact_columns:
                cur.execute(
                    """
                    ALTER TABLE memory_facts
                    ADD COLUMN dedupe_key TEXT NOT NULL DEFAULT ''
                    """
                )

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT,
                    ts DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    event_type TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    importance REAL NOT NULL DEFAULT 0.5,
                    source_message_id INTEGER,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    FOREIGN KEY (conversation_id)
                        REFERENCES conversations(id) ON DELETE SET NULL,
                    FOREIGN KEY (source_message_id)
                        REFERENCES messages(id) ON DELETE SET NULL
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS provider_sessions (
                    provider TEXT NOT NULL,
                    conversation_id TEXT NOT NULL,
                    provider_session_id TEXT NOT NULL,
                    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (provider, conversation_id),
                    FOREIGN KEY (conversation_id)
                        REFERENCES conversations(id) ON DELETE CASCADE
                )
                """
            )

            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_messages_conversation_ts
                ON messages(conversation_id, ts, id)
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_messages_speaker_ts
                ON messages(speaker_entity_id, ts, id)
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_participants_entity
                ON conversation_participants(entity_id, conversation_id)
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_conversations_activity
                ON conversations(status, last_activity_at)
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memory_facts_subject
                ON memory_facts(subject_entity_id, active, predicate)
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memory_facts_source
                ON memory_facts(source_message_id)
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memory_facts_dedupe
                ON memory_facts(
                    subject_entity_id,
                    fact_type,
                    predicate,
                    dedupe_key,
                    active
                )
                """
            )

            cur.execute(
                """
                INSERT INTO schema_meta(key, value)
                VALUES('schema_version', ?)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value
                """,
                (str(self.SCHEMA_VERSION),),
            )
            self.conn.commit()

    @staticmethod
    def _json_dumps(value: Any) -> str:
        return json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
        )

    @staticmethod
    def _row_to_dict(row: Optional[sqlite3.Row]) -> Optional[Dict[str, Any]]:
        return dict(row) if row is not None else None

    # ---------- legacy API ----------

    def set_state(self, key: str, value: str) -> None:
        with self._lock:
            self.conn.execute(
                """
                INSERT INTO assistant_state(key, value)
                VALUES(?, ?)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value
                """,
                (key, value),
            )
            self.conn.commit()

    def get_state(
        self,
        key: str,
        default: Optional[str] = None,
    ) -> Optional[str]:
        with self._lock:
            row = self.conn.execute(
                "SELECT value FROM assistant_state WHERE key=?",
                (key,),
            ).fetchone()
        return str(row["value"]) if row else default

    def add_interaction(
        self,
        mode: str,
        question: str,
        answer: str,
        snapshot_path: str = "",
    ) -> int:
        with self._lock:
            cur = self.conn.execute(
                """
                INSERT INTO interactions(
                    mode, question, answer, snapshot_path
                )
                VALUES(?, ?, ?, ?)
                """,
                (mode, question, answer, snapshot_path),
            )
            self.conn.commit()
            return int(cur.lastrowid)

    def add_scene_event(
        self,
        event_type: str,
        summary: str,
        snapshot_path: str = "",
    ) -> int:
        with self._lock:
            cur = self.conn.execute(
                """
                INSERT INTO scene_events(
                    event_type, summary, snapshot_path
                )
                VALUES(?, ?, ?)
                """,
                (event_type, summary, snapshot_path),
            )
            self.conn.commit()
            return int(cur.lastrowid)

    def add_profile(
        self,
        entity_type: str,
        name: str,
        role: str = "",
        notes: str = "",
    ) -> int:
        with self._lock:
            cur = self.conn.execute(
                """
                INSERT INTO profiles(entity_type, name, role, notes)
                VALUES(?, ?, ?, ?)
                """,
                (entity_type, name, role, notes),
            )
            self.conn.commit()
            return int(cur.lastrowid)

    # ---------- entities ----------

    def upsert_entity(
        self,
        entity_id: str,
        entity_type: str,
        *,
        name: str = "",
        role: str = "",
        notes: str = "",
        privacy_scope: str = "public_household",
    ) -> None:
        with self._lock:
            self.conn.execute(
                """
                INSERT INTO entities(
                    id,
                    entity_type,
                    name,
                    role,
                    notes,
                    privacy_scope
                )
                VALUES(?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    entity_type=excluded.entity_type,
                    name=CASE
                        WHEN excluded.name != '' THEN excluded.name
                        ELSE entities.name
                    END,
                    role=CASE
                        WHEN excluded.role != '' THEN excluded.role
                        ELSE entities.role
                    END,
                    notes=CASE
                        WHEN excluded.notes != '' THEN excluded.notes
                        ELSE entities.notes
                    END,
                    privacy_scope=excluded.privacy_scope,
                    active=1,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (
                    entity_id,
                    entity_type,
                    name,
                    role,
                    notes,
                    privacy_scope,
                ),
            )
            self.conn.commit()

    def get_entity(self, entity_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self.conn.execute(
                "SELECT * FROM entities WHERE id=?",
                (entity_id,),
            ).fetchone()
        return self._row_to_dict(row)

    # ---------- conversations ----------

    def start_conversation(
        self,
        *,
        conversation_id: Optional[str] = None,
        title: str = "",
        location: str = "",
        created_by: str = "robot",
    ) -> str:
        conversation_id = (
            conversation_id
            or f"conv-{uuid.uuid4()}"
        )
        with self._lock:
            self.conn.execute(
                """
                INSERT INTO conversations(
                    id, title, location, created_by
                )
                VALUES(?, ?, ?, ?)
                """,
                (
                    conversation_id,
                    title,
                    location,
                    created_by,
                ),
            )
            self.conn.commit()
        return conversation_id

    def get_active_conversation(
        self,
    ) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self.conn.execute(
                """
                SELECT *
                FROM conversations
                WHERE status='active'
                ORDER BY last_activity_at DESC, started_at DESC
                LIMIT 1
                """
            ).fetchone()
        return self._row_to_dict(row)

    def close_conversation(self, conversation_id: str) -> None:
        with self._lock:
            self.conn.execute(
                """
                UPDATE conversations
                SET
                    status='closed',
                    ended_at=CURRENT_TIMESTAMP,
                    last_activity_at=CURRENT_TIMESTAMP
                WHERE id=?
                """,
                (conversation_id,),
            )
            self.conn.commit()

    def add_participant(
        self,
        conversation_id: str,
        entity_id: str,
        *,
        identity_confidence: Optional[float] = None,
        identity_source: str = "",
    ) -> None:
        with self._lock:
            self.conn.execute(
                """
                INSERT INTO conversation_participants(
                    conversation_id,
                    entity_id,
                    identity_confidence,
                    identity_source
                )
                VALUES(?, ?, ?, ?)
                ON CONFLICT(conversation_id, entity_id) DO UPDATE SET
                    left_at=NULL,
                    identity_confidence=CASE
                        WHEN excluded.identity_confidence IS NOT NULL
                        THEN excluded.identity_confidence
                        ELSE conversation_participants.identity_confidence
                    END,
                    identity_source=CASE
                        WHEN excluded.identity_source != ''
                        THEN excluded.identity_source
                        ELSE conversation_participants.identity_source
                    END
                """,
                (
                    conversation_id,
                    entity_id,
                    identity_confidence,
                    identity_source,
                ),
            )
            self.conn.commit()

    def add_message(
        self,
        conversation_id: str,
        *,
        role: str,
        content: str,
        speaker_entity_id: Optional[str] = None,
        provider: str = "",
        provider_message_id: str = "",
        reply_to_message_id: Optional[int] = None,
        topic_thread_id: str = "",
        identity_confidence: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        if role not in {"user", "assistant", "system", "tool"}:
            raise ValueError(f"unsupported message role: {role}")
        if not content.strip():
            raise ValueError("message content must not be empty")

        with self._lock:
            cur = self.conn.execute(
                """
                INSERT INTO messages(
                    conversation_id,
                    speaker_entity_id,
                    role,
                    content,
                    provider,
                    provider_message_id,
                    reply_to_message_id,
                    topic_thread_id,
                    identity_confidence,
                    metadata_json
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    conversation_id,
                    speaker_entity_id,
                    role,
                    content,
                    provider,
                    provider_message_id,
                    reply_to_message_id,
                    topic_thread_id,
                    identity_confidence,
                    self._json_dumps(metadata or {}),
                ),
            )
            self.conn.execute(
                """
                UPDATE conversations
                SET last_activity_at=CURRENT_TIMESTAMP
                WHERE id=?
                """,
                (conversation_id,),
            )
            self.conn.commit()
            return int(cur.lastrowid)

    def get_recent_messages(
        self,
        conversation_id: str,
        *,
        limit: int = 12,
    ) -> List[Dict[str, Any]]:
        safe_limit = max(1, min(int(limit), 200))
        with self._lock:
            rows = self.conn.execute(
                """
                SELECT *
                FROM (
                    SELECT *
                    FROM messages
                    WHERE conversation_id=?
                    ORDER BY id DESC
                    LIMIT ?
                )
                ORDER BY id ASC
                """,
                (conversation_id, safe_limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_last_conversation_for_entity(
        self,
        entity_id: str,
    ) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self.conn.execute(
                """
                SELECT
                    c.*,
                    s.summary,
                    s.open_topics_json,
                    p.identity_confidence,
                    p.identity_source
                FROM conversation_participants AS p
                JOIN conversations AS c
                    ON c.id=p.conversation_id
                LEFT JOIN conversation_summaries AS s
                    ON s.conversation_id=c.id
                WHERE p.entity_id=?
                ORDER BY c.last_activity_at DESC, c.started_at DESC
                LIMIT 1
                """,
                (entity_id,),
            ).fetchone()
        return self._row_to_dict(row)

    def upsert_conversation_summary(
        self,
        conversation_id: str,
        *,
        summary: str,
        open_topics: Optional[List[str]] = None,
        covered_until_message_id: Optional[int] = None,
    ) -> None:
        with self._lock:
            self.conn.execute(
                """
                INSERT INTO conversation_summaries(
                    conversation_id,
                    summary,
                    open_topics_json,
                    covered_until_message_id
                )
                VALUES(?, ?, ?, ?)
                ON CONFLICT(conversation_id) DO UPDATE SET
                    summary=excluded.summary,
                    open_topics_json=excluded.open_topics_json,
                    covered_until_message_id=
                        excluded.covered_until_message_id,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (
                    conversation_id,
                    summary,
                    self._json_dumps(open_topics or []),
                    covered_until_message_id,
                ),
            )
            self.conn.commit()


    def get_conversation_summary(
        self,
        conversation_id: str,
    ) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self.conn.execute(
                """
                SELECT *
                FROM conversation_summaries
                WHERE conversation_id=?
                """,
                (conversation_id,),
            ).fetchone()
            return self._row_to_dict(row)

    def get_conversation_message_count(
        self,
        conversation_id: str,
    ) -> int:
        with self._lock:
            row = self.conn.execute(
                """
                SELECT COUNT(*) AS count
                FROM messages
                WHERE conversation_id=?
                """,
                (conversation_id,),
            ).fetchone()
            return int(row["count"]) if row else 0

    def get_messages_after(
        self,
        conversation_id: str,
        *,
        after_message_id: int = 0,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        safe_limit = max(1, min(int(limit), 500))
        with self._lock:
            rows = self.conn.execute(
                """
                SELECT *
                FROM messages
                WHERE conversation_id=? AND id>?
                ORDER BY id ASC
                LIMIT ?
                """,
                (conversation_id, int(after_message_id), safe_limit),
            ).fetchall()
            return [dict(row) for row in rows]

    def get_conversation(
        self,
        conversation_id: str,
    ) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self.conn.execute(
                "SELECT * FROM conversations WHERE id=?",
                (conversation_id,),
            ).fetchone()
        return self._row_to_dict(row)

    def get_message(
        self,
        message_id: int,
    ) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self.conn.execute(
                "SELECT * FROM messages WHERE id=?",
                (int(message_id),),
            ).fetchone()
        return self._row_to_dict(row)

    def update_conversation_topic(
        self,
        conversation_id: str,
        active_topic: str,
    ) -> None:
        with self._lock:
            self.conn.execute(
                """
                UPDATE conversations
                SET
                    active_topic=?,
                    last_activity_at=CURRENT_TIMESTAMP
                WHERE id=?
                """,
                (active_topic, conversation_id),
            )
            self.conn.commit()

    def get_conversation_participants(
        self,
        conversation_id: str,
    ) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self.conn.execute(
                """
                SELECT
                    p.conversation_id,
                    p.entity_id,
                    p.joined_at,
                    p.left_at,
                    p.identity_confidence,
                    p.identity_source,
                    e.entity_type,
                    e.name,
                    e.role,
                    e.privacy_scope
                FROM conversation_participants AS p
                JOIN entities AS e ON e.id=p.entity_id
                WHERE p.conversation_id=?
                ORDER BY p.joined_at ASC, p.entity_id ASC
                """,
                (conversation_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_previous_conversation_for_entity(
        self,
        entity_id: str,
        *,
        exclude_conversation_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        query = """
            SELECT
                c.*,
                s.summary,
                s.open_topics_json,
                p.identity_confidence,
                p.identity_source
            FROM conversation_participants AS p
            JOIN conversations AS c
                ON c.id=p.conversation_id
            LEFT JOIN conversation_summaries AS s
                ON s.conversation_id=c.id
            WHERE p.entity_id=?
        """
        params: List[Any] = [entity_id]
        if exclude_conversation_id is not None:
            query += " AND c.id != ?"
            params.append(exclude_conversation_id)
        query += """
            ORDER BY c.last_activity_at DESC, c.started_at DESC
            LIMIT 1
        """
        with self._lock:
            row = self.conn.execute(query, params).fetchone()
        return self._row_to_dict(row)

    # ---------- long-term facts ----------

    def add_memory_fact(
        self,
        *,
        predicate: str,
        value: Any,
        subject_entity_id: Optional[str] = None,
        dedupe_key: str = "",
        fact_type: str = "general",
        confidence: float = 1.0,
        importance: float = 0.5,
        source_type: str = "conversation",
        source_message_id: Optional[int] = None,
        privacy_scope: str = "public_household",
        confirmed_by_user: bool = False,
    ) -> int:
        with self._lock:
            cur = self.conn.execute(
                """
                INSERT INTO memory_facts(
                    subject_entity_id,
                    predicate,
                    value_json,
                    dedupe_key,
                    fact_type,
                    confidence,
                    importance,
                    source_type,
                    source_message_id,
                    privacy_scope,
                    confirmed_by_user
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    subject_entity_id,
                    predicate,
                    self._json_dumps(value),
                    dedupe_key.strip(),
                    fact_type,
                    max(0.0, min(float(confidence), 1.0)),
                    max(0.0, min(float(importance), 1.0)),
                    source_type,
                    source_message_id,
                    privacy_scope,
                    1 if confirmed_by_user else 0,
                ),
            )
            self.conn.commit()
            return int(cur.lastrowid)

    def upsert_memory_fact_deduplicated(
        self,
        *,
        predicate: str,
        value: Any,
        dedupe_key: str,
        subject_entity_id: Optional[str] = None,
        fact_type: str = "general",
        confidence: float = 1.0,
        importance: float = 0.5,
        source_type: str = "conversation",
        source_message_id: Optional[int] = None,
        privacy_scope: str = "public_household",
        confirmed_by_user: bool = False,
    ) -> tuple[int, bool]:
        normalized_key = dedupe_key.strip()
        if not normalized_key:
            raise ValueError("dedupe_key must not be empty")

        confidence_value = max(
            0.0,
            min(float(confidence), 1.0),
        )
        importance_value = max(
            0.0,
            min(float(importance), 1.0),
        )
        confirmed_value = 1 if confirmed_by_user else 0

        with self._lock:
            row = self.conn.execute(
                """
                SELECT id
                FROM memory_facts
                WHERE
                    active=1
                    AND subject_entity_id IS ?
                    AND predicate=?
                    AND fact_type=?
                    AND dedupe_key=?
                ORDER BY id ASC
                LIMIT 1
                """,
                (
                    subject_entity_id,
                    predicate,
                    fact_type,
                    normalized_key,
                ),
            ).fetchone()

            if row is not None:
                fact_id = int(row["id"])
                self.conn.execute(
                    """
                    UPDATE memory_facts
                    SET
                        confidence=MAX(confidence, ?),
                        importance=MAX(importance, ?),
                        confirmed_by_user=MAX(
                            confirmed_by_user,
                            ?
                        ),
                        updated_at=CURRENT_TIMESTAMP
                    WHERE id=?
                    """,
                    (
                        confidence_value,
                        importance_value,
                        confirmed_value,
                        fact_id,
                    ),
                )
                self.conn.commit()
                return fact_id, False

            cur = self.conn.execute(
                """
                INSERT INTO memory_facts(
                    subject_entity_id,
                    predicate,
                    value_json,
                    dedupe_key,
                    fact_type,
                    confidence,
                    importance,
                    source_type,
                    source_message_id,
                    privacy_scope,
                    confirmed_by_user
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    subject_entity_id,
                    predicate,
                    self._json_dumps(value),
                    normalized_key,
                    fact_type,
                    confidence_value,
                    importance_value,
                    source_type,
                    source_message_id,
                    privacy_scope,
                    confirmed_value,
                ),
            )
            self.conn.commit()
            return int(cur.lastrowid), True

    def get_active_facts(
        self,
        *,
        subject_entity_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        safe_limit = max(1, min(int(limit), 500))
        query = """
            SELECT *
            FROM memory_facts
            WHERE active=1
        """
        params: List[Any] = []
        if subject_entity_id is not None:
            query += " AND subject_entity_id=?"
            params.append(subject_entity_id)
        query += """
            ORDER BY
                confirmed_by_user DESC,
                importance DESC,
                updated_at DESC
            LIMIT ?
        """
        params.append(safe_limit)
        with self._lock:
            rows = self.conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    # ---------- provider continuity ----------

    def set_provider_session(
        self,
        provider: str,
        conversation_id: str,
        provider_session_id: str,
    ) -> None:
        with self._lock:
            self.conn.execute(
                """
                INSERT INTO provider_sessions(
                    provider,
                    conversation_id,
                    provider_session_id
                )
                VALUES(?, ?, ?)
                ON CONFLICT(provider, conversation_id) DO UPDATE SET
                    provider_session_id=excluded.provider_session_id,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (
                    provider,
                    conversation_id,
                    provider_session_id,
                ),
            )
            self.conn.commit()

    def get_provider_session(
        self,
        provider: str,
        conversation_id: str,
    ) -> Optional[str]:
        with self._lock:
            row = self.conn.execute(
                """
                SELECT provider_session_id
                FROM provider_sessions
                WHERE provider=? AND conversation_id=?
                """,
                (provider, conversation_id),
            ).fetchone()
        return str(row["provider_session_id"]) if row else None

    def close(self) -> None:
        with self._lock:
            self.conn.close()
