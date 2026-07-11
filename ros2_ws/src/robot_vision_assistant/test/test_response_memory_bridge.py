from __future__ import annotations

import json

from robot_vision_assistant.response_memory_bridge import (
    ResponseMemoryBridge,
)


class FakeClock:
    def __init__(self) -> None:
        self.value = 100.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def test_user_message_is_saved_before_provider_response(tmp_path):
    bridge = ResponseMemoryBridge(
        str(tmp_path / "memory.sqlite")
    )
    try:
        record = bridge.before_request("Что ты видишь?")

        messages = bridge.store.get_recent_messages(
            record.conversation_id,
            limit=10,
        )
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Что ты видишь?"
        assert record.memory_context["recent_messages"][-1][
            "content"
        ] == "Что ты видишь?"
    finally:
        bridge.close()


def test_provider_answer_is_saved_in_same_local_conversation(
    tmp_path,
):
    bridge = ResponseMemoryBridge(
        str(tmp_path / "memory.sqlite")
    )
    try:
        request = bridge.before_request("Продолжим?")
        assistant_id = bridge.after_response(
            conversation_id=request.conversation_id,
            reply_to_message_id=request.user_message_id,
            answer="Продолжим.",
            provider="openai",
        )

        messages = bridge.store.get_recent_messages(
            request.conversation_id,
            limit=10,
        )
        assert [item["role"] for item in messages] == [
            "user",
            "assistant",
        ]
        assert messages[-1]["id"] == assistant_id
        assert messages[-1]["provider"] == "openai"
        assert messages[-1]["reply_to_message_id"] == (
            request.user_message_id
        )
    finally:
        bridge.close()


def test_speaker_change_preserves_shared_recent_dialogue(tmp_path):
    bridge = ResponseMemoryBridge(
        str(tmp_path / "memory.sqlite")
    )
    try:
        assert bridge.update_identity(
            {
                "entity_id": "person:oleg",
                "name": "Олег",
                "role": "owner",
                "confidence": 0.98,
                "source": "face+voice",
                "status": "confirmed",
            }
        )
        first = bridge.before_request(
            "Какая температура Jetson?"
        )
        bridge.after_response(
            conversation_id=first.conversation_id,
            reply_to_message_id=first.user_message_id,
            answer="Сейчас 62 градуса.",
            provider="qwen_local",
        )

        assert bridge.update_identity(
            {
                "entity_id": "person:anna",
                "name": "Анна",
                "role": "family",
                "confidence": 0.94,
                "source": "voice",
                "status": "confirmed",
            }
        )
        second = bridge.before_request("А это нормально?")

        assert second.conversation_id == first.conversation_id
        speakers = [
            item["speaker_entity_id"]
            for item in second.memory_context["recent_messages"]
        ]
        assert speakers == [
            "person:oleg",
            "robot:self",
            "person:anna",
        ]
    finally:
        bridge.close()


def test_expired_identity_falls_back_to_unknown_speaker(tmp_path):
    clock = FakeClock()
    bridge = ResponseMemoryBridge(
        str(tmp_path / "memory.sqlite"),
        identity_ttl_sec=5.0,
        monotonic_clock=clock,
    )
    try:
        bridge.update_identity(
            {
                "entity_id": "person:oleg",
                "name": "Олег",
                "confidence": 0.99,
                "source": "face",
                "status": "confirmed",
            }
        )
        clock.advance(6.0)
        record = bridge.before_request("Кто я?")

        assert record.speaker_entity_id == "person:unknown"
    finally:
        bridge.close()


def test_compact_context_respects_character_budget(tmp_path):
    bridge = ResponseMemoryBridge(
        str(tmp_path / "memory.sqlite"),
        max_context_chars=1600,
    )
    try:
        record = None
        for index in range(20):
            record = bridge.before_request(
                f"Сообщение {index}: " + ("длинный текст " * 40)
            )
            bridge.after_response(
                conversation_id=record.conversation_id,
                reply_to_message_id=record.user_message_id,
                answer="Короткий ответ.",
                provider="qwen_local",
            )

        assert record is not None
        serialized = json.dumps(
            record.memory_context,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        assert len(serialized) <= 1900
        assert len(record.memory_context["recent_messages"]) >= 3
    finally:
        bridge.close()
