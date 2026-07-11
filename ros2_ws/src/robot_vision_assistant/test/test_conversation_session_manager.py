from __future__ import annotations

from datetime import datetime, timezone

from robot_vision_assistant.conversation_session_manager import (
    ConversationSessionManager,
    SessionPolicy,
)
from robot_vision_assistant.memory_store import MemoryStore


class FakeClock:
    def __init__(self) -> None:
        self.monotonic_value = 100.0
        self.wall_value = datetime.now(timezone.utc)

    def monotonic(self) -> float:
        return self.monotonic_value

    def wall(self) -> datetime:
        return self.wall_value

    def advance(self, seconds: float) -> None:
        self.monotonic_value += seconds


def make_manager(tmp_path, timeout_sec=900.0):
    store = MemoryStore(str(tmp_path / "memory.sqlite"))
    clock = FakeClock()
    manager = ConversationSessionManager(
        store,
        policy=SessionPolicy(
            inactivity_timeout_sec=timeout_sec,
            recent_message_limit=12,
            personal_fact_limit=8,
        ),
        monotonic_clock=clock.monotonic,
        wall_clock=clock.wall,
    )
    return store, manager, clock


def test_speaker_change_keeps_shared_conversation_and_reply_chain(
    tmp_path,
):
    store, manager, _ = make_manager(tmp_path)
    try:
        oleg = manager.record_user_message(
            speaker_entity_id="person:oleg",
            name="Олег",
            role="owner",
            content="Какая температура Jetson?",
            identity_confidence=0.98,
            identity_source="face+voice",
            topic_thread_id="jetson_temperature",
        )
        robot = manager.record_assistant_message(
            conversation_id=oleg.conversation_id,
            content="Сейчас 62 градуса.",
            provider="qwen_local",
            reply_to_message_id=oleg.message_id,
        )
        anna = manager.record_user_message(
            speaker_entity_id="person:anna",
            name="Анна",
            role="family",
            content="А это нормально?",
            identity_confidence=0.95,
            identity_source="voice",
        )

        assert anna.conversation_id == oleg.conversation_id
        assert anna.reply_to_message_id == robot.message_id
        assert anna.topic_thread_id == "jetson_temperature"

        messages = store.get_recent_messages(
            oleg.conversation_id,
            limit=10,
        )
        assert [message["speaker_entity_id"] for message in messages] == [
            "person:oleg",
            "robot:self",
            "person:anna",
        ]
    finally:
        store.close()


def test_personal_context_changes_without_replacing_shared_dialogue(
    tmp_path,
):
    store, manager, _ = make_manager(tmp_path)
    try:
        store.upsert_entity(
            "person:anna",
            "person",
            name="Анна",
            role="family",
        )
        old_conversation = store.start_conversation(
            title="Прошлый разговор с Анной"
        )
        store.add_participant(
            old_conversation,
            "person:anna",
            identity_confidence=0.99,
            identity_source="confirmed",
        )
        old_message = store.add_message(
            old_conversation,
            role="user",
            speaker_entity_id="person:anna",
            content="Вентилятор работает слишком громко.",
        )
        store.upsert_conversation_summary(
            old_conversation,
            summary="Анна спрашивала о шуме вентилятора.",
            open_topics=["проверить уровень шума"],
            covered_until_message_id=old_message,
        )
        store.close_conversation(old_conversation)

        oleg = manager.record_user_message(
            speaker_entity_id="person:oleg",
            name="Олег",
            role="owner",
            content="Какая температура Jetson?",
            topic_thread_id="jetson_temperature",
        )
        manager.record_assistant_message(
            conversation_id=oleg.conversation_id,
            content="Сейчас 62 градуса.",
            provider="openai",
            reply_to_message_id=oleg.message_id,
        )
        manager.record_user_message(
            speaker_entity_id="person:anna",
            name="Анна",
            role="family",
            content="А это нормально?",
        )

        context = manager.build_context(
            current_speaker_entity_id="person:anna"
        )

        assert [
            message["speaker_entity_id"]
            for message in context["recent_messages"]
        ] == [
            "person:oleg",
            "robot:self",
            "person:anna",
        ]
        previous = context[
            "current_speaker_previous_conversation"
        ]
        assert previous is not None
        assert previous["conversation_id"] == old_conversation
        assert previous["summary"] == (
            "Анна спрашивала о шуме вентилятора."
        )
        assert previous["open_topics"] == [
            "проверить уровень шума"
        ]
    finally:
        store.close()


def test_timeout_starts_new_conversation(tmp_path):
    store, manager, clock = make_manager(
        tmp_path,
        timeout_sec=30.0,
    )
    try:
        first = manager.record_user_message(
            speaker_entity_id="person:oleg",
            name="Олег",
            content="Первый разговор.",
        )
        clock.advance(31.0)
        second = manager.record_user_message(
            speaker_entity_id="person:oleg",
            name="Олег",
            content="Новый разговор.",
        )

        assert second.conversation_id != first.conversation_id
        old = store.get_conversation(first.conversation_id)
        assert old is not None
        assert old["status"] == "closed"
    finally:
        store.close()


def test_multiple_providers_share_one_local_conversation(tmp_path):
    store, manager, _ = make_manager(tmp_path)
    try:
        user = manager.record_user_message(
            speaker_entity_id="person:oleg",
            name="Олег",
            content="Продолжим настройку памяти.",
        )
        cloud = manager.record_assistant_message(
            conversation_id=user.conversation_id,
            content="Продолжим.",
            provider="openai",
            reply_to_message_id=user.message_id,
        )
        followup = manager.record_user_message(
            speaker_entity_id="person:oleg",
            name="Олег",
            content="Теперь ответь локально.",
        )
        local = manager.record_assistant_message(
            conversation_id=followup.conversation_id,
            content="Контекст сохранён.",
            provider="qwen_local",
            reply_to_message_id=followup.message_id,
        )

        assert cloud.conversation_id == local.conversation_id
        context = manager.build_context(
            current_speaker_entity_id="person:oleg"
        )
        providers = [
            message["provider"]
            for message in context["recent_messages"]
            if message["role"] == "assistant"
        ]
        assert providers == ["openai", "qwen_local"]
    finally:
        store.close()


def test_current_speaker_facts_are_separate_context_layer(tmp_path):
    store, manager, _ = make_manager(tmp_path)
    try:
        manager.record_user_message(
            speaker_entity_id="person:oleg",
            name="Олег",
            role="owner",
            content="Запомни, что я предпочитаю точные ответы.",
        )
        store.add_memory_fact(
            subject_entity_id="person:oleg",
            predicate="response_preference",
            value="точные технические ответы",
            fact_type="preference",
            confidence=1.0,
            importance=0.9,
            source_type="explicit_user_statement",
            privacy_scope="person_private",
            confirmed_by_user=True,
        )

        context = manager.build_context(
            current_speaker_entity_id="person:oleg"
        )

        assert context["current_speaker"]["name"] == "Олег"
        assert context["current_speaker_facts"] == [
            {
                "predicate": "response_preference",
                "value": "точные технические ответы",
                "fact_type": "preference",
                "confidence": 1.0,
                "importance": 0.9,
                "privacy_scope": "person_private",
                "confirmed_by_user": True,
            }
        ]
    finally:
        store.close()
