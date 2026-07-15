from __future__ import annotations

import json

from robot_vision_assistant.deterministic_memory_worker import (
    DeterministicMemoryExtractor,
    DeterministicMemoryWorker,
)
from robot_vision_assistant.memory_store import MemoryStore
from robot_vision_assistant.response_memory_bridge import (
    ResponseMemoryBridge,
)


def test_extractor_accepts_only_explicit_memory():
    extractor = DeterministicMemoryExtractor()

    candidate = extractor.extract(
        "Запомни, что я предпочитаю короткие ответы."
    )

    assert candidate is not None
    assert candidate.fact_type == "explicit_fact"
    assert candidate.predicate == "explicit_memory"
    assert candidate.value == {
        "statement": "я предпочитаю короткие ответы"
    }
    assert candidate.confirmed_by_user is True


def test_extractor_accepts_explicit_behavior_correction():
    extractor = DeterministicMemoryExtractor()

    candidate = extractor.extract(
        "Ты должен был остановиться дальше от стола."
    )

    assert candidate is not None
    assert candidate.fact_type == "correction"
    assert candidate.predicate == "behavior_correction"
    assert candidate.value == {
        "instruction": "остановиться дальше от стола"
    }


def test_extractor_rejects_ordinary_negated_and_question_text():
    extractor = DeterministicMemoryExtractor()

    assert extractor.extract(
        "Я предпочитаю короткие ответы."
    ) is None
    assert extractor.extract(
        "Не запоминай, что я предпочитаю короткие ответы."
    ) is None
    assert extractor.extract(
        "Запомни, что я сейчас сказал?"
    ) is None


def test_worker_stores_confirmed_fact_and_deduplicates(tmp_path):
    store = MemoryStore(str(tmp_path / "memory.sqlite"))
    worker = DeterministicMemoryWorker(store)
    try:
        store.upsert_entity(
            "person:oleg",
            "person",
            name="Олег",
        )
        conversation_id = store.start_conversation()
        first_message = store.add_message(
            conversation_id,
            role="user",
            speaker_entity_id="person:oleg",
            content="Запомни, что мой любимый цвет — синий.",
        )
        second_message = store.add_message(
            conversation_id,
            role="user",
            speaker_entity_id="person:oleg",
            content="Запомни: мой любимый цвет синий!",
        )

        assert worker.submit(
            conversation_id=conversation_id,
            message_id=first_message,
            speaker_entity_id="person:oleg",
            identity_status="confirmed",
            text="Запомни, что мой любимый цвет — синий.",
        )
        assert worker.submit(
            conversation_id=conversation_id,
            message_id=second_message,
            speaker_entity_id="person:oleg",
            identity_status="confirmed",
            text="Запомни: мой любимый цвет синий!",
        )
        assert worker.flush(timeout_sec=3.0)

        facts = store.get_active_facts(
            subject_entity_id="person:oleg"
        )
        assert len(facts) == 1
        assert facts[0]["fact_type"] == "explicit_fact"
        assert facts[0]["confirmed_by_user"] == 1
        assert json.loads(facts[0]["value_json"]) == {
            "statement": "мой любимый цвет — синий"
        }

        stats = worker.stats()
        assert stats["stored"] == 1
        assert stats["deduplicated"] == 1
        assert stats["failed"] == 0
    finally:
        worker.close()
        store.close()


def test_worker_does_not_assign_fact_to_unknown_person(tmp_path):
    store = MemoryStore(str(tmp_path / "memory.sqlite"))
    worker = DeterministicMemoryWorker(store)
    try:
        conversation_id = store.start_conversation()
        message_id = store.add_message(
            conversation_id,
            role="user",
            content="Запомни, что мой любимый цвет — синий.",
        )

        assert worker.submit(
            conversation_id=conversation_id,
            message_id=message_id,
            speaker_entity_id="person:unknown",
            identity_status="unknown",
            text="Запомни, что мой любимый цвет — синий.",
        )
        assert worker.flush(timeout_sec=3.0)

        assert store.get_active_facts() == []
        assert worker.stats()["skipped_identity"] == 1
    finally:
        worker.close()
        store.close()


def test_bridge_exposes_fact_on_following_turn(tmp_path):
    bridge = ResponseMemoryBridge(
        str(tmp_path / "memory.sqlite"),
        memory_worker_enabled=True,
    )
    try:
        assert bridge.update_identity(
            {
                "entity_id": "person:oleg",
                "name": "Олег",
                "role": "owner",
                "confidence": 0.99,
                "source": "test",
                "status": "confirmed",
            }
        )

        bridge.before_request(
            "Запомни, что я предпочитаю короткие ответы."
        )
        assert bridge.flush_memory_worker(timeout_sec=3.0)

        followup = bridge.before_request(
            "Какие ответы я предпочитаю?"
        )

        assert followup.memory_context[
            "current_speaker_facts"
        ] == [
            {
                "predicate": "explicit_memory",
                "value": {
                    "statement": (
                        "я предпочитаю короткие ответы"
                    )
                },
                "fact_type": "explicit_fact",
                "confidence": 1.0,
                "importance": 0.9,
                "confirmed_by_user": True,
            }
        ]
    finally:
        bridge.close()


def test_worker_does_not_call_any_model(tmp_path):
    store = MemoryStore(str(tmp_path / "memory.sqlite"))
    worker = DeterministicMemoryWorker(store)
    try:
        assert not hasattr(worker, "model")
        assert not hasattr(worker, "provider")
        assert not hasattr(worker, "http_client")
    finally:
        worker.close()
        store.close()
