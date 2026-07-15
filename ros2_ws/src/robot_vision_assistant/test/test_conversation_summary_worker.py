import tempfile

from robot_vision_assistant.conversation_session_manager import (
    ConversationSessionManager,
    SessionPolicy,
)
from robot_vision_assistant.conversation_summary_worker import (
    ConversationSummaryWorker,
    DeterministicConversationSummarizer,
)
from robot_vision_assistant.memory_store import MemoryStore
from robot_vision_assistant.response_memory_bridge import (
    ResponseMemoryBridge,
)


def make_store():
    tmp = tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False)
    tmp.close()
    return MemoryStore(tmp.name)


def record_pair(manager, conversation_id, user_text, answer="ok"):
    user = manager.record_user_message(
        speaker_entity_id="person:oleg",
        name="Олег",
        role="owner",
        content=user_text,
        identity_confidence=1.0,
        identity_source="test",
    )
    assistant = manager.record_assistant_message(
        conversation_id=conversation_id,
        content=answer,
        provider="test",
        reply_to_message_id=user.message_id,
    )
    return user, assistant


def test_summarizer_preserves_new_messages_when_previous_summary_is_long():
    summarizer = DeterministicConversationSummarizer()
    previous = " ".join(
        ["старое-summary-содержимое"] * 120
    )
    summary = summarizer.build_summary(
        previous_summary=previous,
        messages=[
            {
                "id": 101,
                "role": "user",
                "speaker_entity_id": "person:oleg",
                "content": "Новая тема summary-новый-маркер-001",
            },
            {
                "id": 102,
                "role": "assistant",
                "speaker_entity_id": "robot:self",
                "content": "Принято summary-новый-маркер-001",
            },
        ],
    )

    assert "summary-новый-маркер-001" in summary
    assert len(summary) <= summarizer.MAX_SUMMARY_CHARS


def test_summarizer_extracts_open_topics_and_ignores_memory_commands():
    summarizer = DeterministicConversationSummarizer()
    topics = summarizer.extract_open_topics(
        [
            {"role": "user", "content": "Запомни, что цвет синий"},
            {"role": "user", "content": "Надо проверить камеру завтра"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "Что дальше делаем?"},
        ]
    )

    assert topics == [
        "Надо проверить камеру завтра",
        "Что дальше делаем?",
    ]


def test_summary_worker_incremental_update_does_not_duplicate_history():
    store = make_store()
    try:
        manager = ConversationSessionManager(
            store,
            policy=SessionPolicy(recent_message_limit=12),
        )
        conversation_id = manager.ensure_conversation()
        manager.observe_speaker(
            "person:oleg",
            name="Олег",
            role="owner",
            identity_confidence=1.0,
            identity_source="test",
            conversation_id=conversation_id,
        )

        record_pair(
            manager,
            conversation_id,
            "Надо проверить summary-инкремент-1",
            "Принято",
        )

        worker = ConversationSummaryWorker(
            store,
            min_messages=2,
        )
        try:
            assert worker.submit(conversation_id)
            assert worker.flush(timeout_sec=2.0)

            first = store.get_conversation_summary(conversation_id)
            assert first is not None
            assert "summary-инкремент-1" in first["summary"]

            record_pair(
                manager,
                conversation_id,
                "Следующий шаг summary-инкремент-2",
                "Продолжаю",
            )

            assert worker.submit(conversation_id)
            assert worker.flush(timeout_sec=2.0)
        finally:
            worker.close()

        second = store.get_conversation_summary(conversation_id)
        assert second is not None
        assert second["summary"].count("summary-инкремент-1") == 1
        assert "summary-инкремент-2" in second["summary"]
        assert int(second["covered_until_message_id"]) == 4
    finally:
        store.close()


def test_summary_worker_creates_summary_after_threshold():
    store = make_store()
    try:
        manager = ConversationSessionManager(
            store,
            policy=SessionPolicy(recent_message_limit=12),
        )
        conversation_id = manager.ensure_conversation()
        manager.observe_speaker(
            "person:oleg",
            name="Олег",
            role="owner",
            identity_confidence=1.0,
            identity_source="test",
            conversation_id=conversation_id,
        )

        record_pair(
            manager,
            conversation_id,
            "Надо проверить камеру завтра",
            "Проверю камеру",
        )
        record_pair(
            manager,
            conversation_id,
            "Что дальше делаем?",
            "Следующий шаг — тест",
        )

        worker = ConversationSummaryWorker(
            store,
            min_messages=4,
        )
        try:
            assert worker.submit(conversation_id)
            assert worker.flush(timeout_sec=2.0)
        finally:
            worker.close()

        summary = store.get_conversation_summary(conversation_id)
        assert summary is not None
        assert "Надо проверить камеру" in summary["summary"]
        assert int(summary["covered_until_message_id"]) == 4
        assert "Что дальше делаем?" in summary["open_topics_json"]
    finally:
        store.close()


def test_summary_worker_ignores_short_conversation():
    store = make_store()
    try:
        manager = ConversationSessionManager(store)
        conversation_id = manager.ensure_conversation()
        record_pair(manager, conversation_id, "Что дальше делаем?")

        worker = ConversationSummaryWorker(
            store,
            min_messages=6,
        )
        try:
            assert worker.submit(conversation_id)
            assert worker.flush(timeout_sec=2.0)
        finally:
            worker.close()

        assert store.get_conversation_summary(conversation_id) is None
        assert worker.stats()["ignored"] >= 1
    finally:
        store.close()


def test_response_memory_bridge_injects_current_conversation_summary():
    tmp = tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False)
    tmp.close()
    bridge = ResponseMemoryBridge(
        tmp.name,
        identity_ttl_sec=60.0,
        recent_message_limit=6,
        conversation_summary_enabled=True,
        conversation_summary_min_messages=4,
    )
    try:
        assert bridge.update_identity(
            {
                "entity_id": "person:oleg",
                "name": "Олег",
                "role": "owner",
                "confidence": 1.0,
                "source": "test",
                "status": "confirmed",
            }
        )

        first = bridge.before_request(
            "Надо проверить камеру завтра"
        )
        bridge.after_response(
            conversation_id=first.conversation_id,
            reply_to_message_id=first.user_message_id,
            answer="Проверю камеру",
            provider="test",
        )
        second = bridge.before_request("Что дальше делаем?")
        bridge.after_response(
            conversation_id=second.conversation_id,
            reply_to_message_id=second.user_message_id,
            answer="Следующий шаг — тест",
            provider="test",
        )
        assert bridge.flush_conversation_summary_worker(
            timeout_sec=2.0
        )

        third = bridge.before_request("Напомни открытые темы")
        current_summary = third.memory_context.get(
            "current_conversation_summary"
        )
        assert isinstance(current_summary, dict)
        assert "Надо проверить камеру" in current_summary["summary"]
        assert "Что дальше делаем?" in " ".join(
            current_summary["open_topics"]
        )
    finally:
        bridge.close()
