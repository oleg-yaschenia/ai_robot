import json
import sqlite3

from robot_vision_assistant.memory_store import MemoryStore


def test_schema_upgrade_preserves_legacy_tables(tmp_path):
    db_path = tmp_path / "memory.sqlite"

    connection = sqlite3.connect(db_path)
    connection.execute(
        """
        CREATE TABLE interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts DATETIME DEFAULT CURRENT_TIMESTAMP,
            mode TEXT NOT NULL,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            snapshot_path TEXT DEFAULT ''
        )
        """
    )
    connection.execute(
        """
        INSERT INTO interactions(mode, question, answer)
        VALUES('legacy', 'старый вопрос', 'старый ответ')
        """
    )
    connection.commit()
    connection.close()

    store = MemoryStore(str(db_path))
    try:
        row = store.conn.execute(
            "SELECT question, answer FROM interactions"
        ).fetchone()
        assert row["question"] == "старый вопрос"
        assert row["answer"] == "старый ответ"

        version = store.conn.execute(
            """
            SELECT value
            FROM schema_meta
            WHERE key='schema_version'
            """
        ).fetchone()
        assert version["value"] == "2"
    finally:
        store.close()


def test_multi_party_messages_remain_in_one_conversation(tmp_path):
    store = MemoryStore(str(tmp_path / "memory.sqlite"))
    try:
        store.upsert_entity(
            "person:oleg",
            "person",
            name="Олег",
            role="owner",
        )
        store.upsert_entity(
            "person:anna",
            "person",
            name="Анна",
            role="family",
        )

        conversation_id = store.start_conversation(
            title="Проверка температуры Jetson"
        )
        store.add_participant(
            conversation_id,
            "person:oleg",
            identity_confidence=0.97,
            identity_source="face+voice",
        )
        store.add_participant(
            conversation_id,
            "person:anna",
            identity_confidence=0.94,
            identity_source="voice",
        )

        first = store.add_message(
            conversation_id,
            role="user",
            speaker_entity_id="person:oleg",
            content="Какая температура Jetson?",
        )
        assistant = store.add_message(
            conversation_id,
            role="assistant",
            speaker_entity_id=None,
            content="Сейчас 62 градуса.",
            provider="qwen_local",
            reply_to_message_id=first,
        )
        followup = store.add_message(
            conversation_id,
            role="user",
            speaker_entity_id="person:anna",
            content="А это нормально?",
            reply_to_message_id=assistant,
            identity_confidence=0.94,
        )

        messages = store.get_recent_messages(
            conversation_id,
            limit=10,
        )
        assert [item["id"] for item in messages] == [
            first,
            assistant,
            followup,
        ]
        assert messages[0]["speaker_entity_id"] == "person:oleg"
        assert messages[2]["speaker_entity_id"] == "person:anna"
        assert messages[2]["conversation_id"] == conversation_id

        participants = store.conn.execute(
            """
            SELECT entity_id
            FROM conversation_participants
            WHERE conversation_id=?
            ORDER BY entity_id
            """,
            (conversation_id,),
        ).fetchall()
        assert [row["entity_id"] for row in participants] == [
            "person:anna",
            "person:oleg",
        ]
    finally:
        store.close()


def test_last_conversation_and_summary_are_person_specific(tmp_path):
    store = MemoryStore(str(tmp_path / "memory.sqlite"))
    try:
        store.upsert_entity(
            "person:oleg",
            "person",
            name="Олег",
        )
        conversation_id = store.start_conversation()
        store.add_participant(
            conversation_id,
            "person:oleg",
            identity_confidence=0.99,
            identity_source="confirmed",
        )
        message_id = store.add_message(
            conversation_id,
            role="user",
            speaker_entity_id="person:oleg",
            content="Продолжим настройку памяти робота.",
        )
        store.upsert_conversation_summary(
            conversation_id,
            summary="Обсуждали единую память робота.",
            open_topics=["подключить Memory Manager"],
            covered_until_message_id=message_id,
        )

        result = store.get_last_conversation_for_entity(
            "person:oleg"
        )
        assert result is not None
        assert result["id"] == conversation_id
        assert result["summary"] == (
            "Обсуждали единую память робота."
        )
        assert json.loads(result["open_topics_json"]) == [
            "подключить Memory Manager"
        ]
    finally:
        store.close()


def test_long_term_fact_keeps_source_and_privacy(tmp_path):
    store = MemoryStore(str(tmp_path / "memory.sqlite"))
    try:
        store.upsert_entity(
            "pet:jack",
            "pet",
            name="Джек",
            role="cat",
            privacy_scope="family",
        )
        conversation_id = store.start_conversation()
        message_id = store.add_message(
            conversation_id,
            role="user",
            content="Моего кота зовут Джек.",
        )

        fact_id = store.add_memory_fact(
            subject_entity_id="pet:jack",
            predicate="name",
            value="Джек",
            fact_type="identity",
            confidence=1.0,
            importance=0.95,
            source_type="explicit_user_statement",
            source_message_id=message_id,
            privacy_scope="family",
            confirmed_by_user=True,
        )

        facts = store.get_active_facts(
            subject_entity_id="pet:jack"
        )
        assert len(facts) == 1
        assert facts[0]["id"] == fact_id
        assert json.loads(facts[0]["value_json"]) == "Джек"
        assert facts[0]["confirmed_by_user"] == 1
        assert facts[0]["privacy_scope"] == "family"
    finally:
        store.close()


def test_legacy_add_interaction_still_works(tmp_path):
    store = MemoryStore(str(tmp_path / "memory.sqlite"))
    try:
        interaction_id = store.add_interaction(
            "local_only",
            "Что ты видишь?",
            "Я вижу человека.",
        )
        row = store.conn.execute(
            """
            SELECT mode, question, answer
            FROM interactions
            WHERE id=?
            """,
            (interaction_id,),
        ).fetchone()
        assert row["mode"] == "local_only"
        assert row["question"] == "Что ты видишь?"
        assert row["answer"] == "Я вижу человека."
    finally:
        store.close()
