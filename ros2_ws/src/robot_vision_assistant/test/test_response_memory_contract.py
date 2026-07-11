from robot_vision_assistant.assistant_response_contract import (
    build_response_context,
)
from robot_vision_assistant.response_orchestrator import (
    build_qwen_request,
    classify_response_route,
)


def test_response_context_carries_provider_neutral_memory():
    memory_context = {
        "schema": "robot_memory_context",
        "conversation": {"conversation_id": "conv-1"},
        "current_speaker": {
            "entity_id": "person:oleg",
            "name": "Олег",
        },
        "recent_messages": [
            {
                "role": "user",
                "speaker_entity_id": "person:oleg",
                "content": "Продолжим?",
            }
        ],
    }

    context = build_response_context(
        request_id="resp-1",
        query="Продолжим?",
        route="text",
        memory_context=memory_context,
    )

    assert context["memory_context"] == memory_context


def test_qwen_request_carries_same_local_memory_context():
    route = classify_response_route("Продолжим разговор?")
    memory_context = {
        "schema": "robot_memory_context",
        "conversation": {"conversation_id": "conv-2"},
        "recent_messages": [
            {
                "role": "assistant",
                "speaker_entity_id": "robot:self",
                "content": "На чём мы остановились?",
                "provider": "openai",
            },
            {
                "role": "user",
                "speaker_entity_id": "person:oleg",
                "content": "Продолжим разговор?",
            },
        ],
    }

    request = build_qwen_request(
        request_id="resp-2",
        query="Продолжим разговор?",
        route=route,
        memory_context=memory_context,
    )

    assert request["response_context"]["memory_context"] == (
        memory_context
    )
