import sqlite3
from pathlib import Path
from typing import Optional


class MemoryStore:
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._init_db()

    def _init_db(self):
        cur = self.conn.cursor()

        cur.execute("""
        CREATE TABLE IF NOT EXISTS profiles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_type TEXT NOT NULL,          -- person, pet
            name TEXT NOT NULL,
            role TEXT DEFAULT '',               -- owner, family, guest, cat, dog
            notes TEXT DEFAULT '',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """)

        cur.execute("""
        CREATE TABLE IF NOT EXISTS interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts DATETIME DEFAULT CURRENT_TIMESTAMP,
            mode TEXT NOT NULL,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            snapshot_path TEXT DEFAULT ''
        )
        """)

        cur.execute("""
        CREATE TABLE IF NOT EXISTS scene_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts DATETIME DEFAULT CURRENT_TIMESTAMP,
            event_type TEXT NOT NULL,
            summary TEXT NOT NULL,
            snapshot_path TEXT DEFAULT ''
        )
        """)

        cur.execute("""
        CREATE TABLE IF NOT EXISTS assistant_state (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """)

        self.conn.commit()

    def set_state(self, key: str, value: str):
        cur = self.conn.cursor()
        cur.execute("""
        INSERT INTO assistant_state(key, value)
        VALUES(?, ?)
        ON CONFLICT(key) DO UPDATE SET value=excluded.value
        """, (key, value))
        self.conn.commit()

    def get_state(self, key: str, default: Optional[str] = None) -> Optional[str]:
        cur = self.conn.cursor()
        cur.execute("SELECT value FROM assistant_state WHERE key=?", (key,))
        row = cur.fetchone()
        return row[0] if row else default

    def add_interaction(self, mode: str, question: str, answer: str, snapshot_path: str = ""):
        cur = self.conn.cursor()
        cur.execute("""
        INSERT INTO interactions(mode, question, answer, snapshot_path)
        VALUES(?, ?, ?, ?)
        """, (mode, question, answer, snapshot_path))
        self.conn.commit()

    def add_scene_event(self, event_type: str, summary: str, snapshot_path: str = ""):
        cur = self.conn.cursor()
        cur.execute("""
        INSERT INTO scene_events(event_type, summary, snapshot_path)
        VALUES(?, ?, ?)
        """, (event_type, summary, snapshot_path))
        self.conn.commit()

    def add_profile(self, entity_type: str, name: str, role: str = "", notes: str = ""):
        cur = self.conn.cursor()
        cur.execute("""
        INSERT INTO profiles(entity_type, name, role, notes)
        VALUES(?, ?, ?, ?)
        """, (entity_type, name, role, notes))
        self.conn.commit()

    def close(self):
        self.conn.close()
