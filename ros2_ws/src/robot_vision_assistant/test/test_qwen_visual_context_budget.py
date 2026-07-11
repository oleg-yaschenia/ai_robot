import json

import pytest

from robot_vision_assistant.qwen_vl_runtime_node import (
    HttpStatusError,
    PersistentLlamaClient,
    QwenVlRuntimeNode,
)


class _FakeResponse:
    status = 400

    def read(self):
        return (
            b'{"error":{"type":"exceed_context_size_error",'
            b'"n_prompt_tokens":4486,"n_ctx":4096}}'
        )


class _FakeConnection:
    def __init__(self):
        self.request_count = 0

    def request(self, *args, **kwargs):
        self.request_count += 1

    def getresponse(self):
        return _FakeResponse()

    def close(self):
        pass


def test_visual_response_context_omits_capabilities_and_raw_scene():
    context = {
        "request_id": "r1",
        "query": "Что ты сейчас видишь?",
        "route": "visual",
        "capabilities": {
            "available": ["x"] * 100,
            "planned_but_not_executable": ["y"] * 100,
        },
        "scene": {"objects": [{"label": "person"}] * 100},
        "action_result": {
            "state": "not_requested",
            "execution_allowed": False,
        },
        "rules": {
            "sensor_values_must_be_supplied": True,
            "claim_action_only_after_executor_confirmation": True,
            "hardware_commands_forbidden": True,
        },
    }

    rendered = QwenVlRuntimeNode._compact_visual_response_context(
        context
    )

    assert rendered.startswith("ROBOT_VISUAL_CONTEXT=")
    payload = json.loads(rendered.split("=", 1)[1])
    assert payload["request_id"] == "r1"
    assert payload["route"] == "visual"
    assert payload["action_result"]["execution_allowed"] is False
    assert payload["rules"]["hardware_commands_forbidden"] is True
    assert "capabilities" not in payload
    assert "scene" not in payload
    assert len(rendered) < 700


def test_stream_chat_does_not_retry_http_400():
    client = PersistentLlamaClient(
        "http://127.0.0.1:8080",
        timeout_sec=1.0,
    )
    connection = _FakeConnection()
    client._connection = connection

    with pytest.raises(HttpStatusError) as exc_info:
        client.stream_chat(
            {
                "model": "test",
                "messages": [],
                "max_tokens": 32,
            },
            lambda fragment, visible: None,
        )

    assert exc_info.value.status == 400
    assert connection.request_count == 1



def _make_runtime_without_ros_init():
    import threading
    import time

    node = object.__new__(QwenVlRuntimeNode)
    node._state_lock = threading.Lock()
    node.max_scene_age_sec = 2.0
    node._latest_scene_monotonic = time.monotonic()
    node._latest_scene = {
        "source_timestamp": 123.5,
        "counts": {"person": 1, "chair": 2},
        "salient_entities": [
            {
                "entity_id": "person_0",
                "class_name": "person",
                "confirmed": True,
                "confidence": 0.91,
                "position_text": "по центру",
                "distance_m": 1.2345,
                "distance_valid": True,
                "depth_confidence": "high",
            },
            {
                "entity_id": "chair_0",
                "class_name": "chair",
                "confirmed": True,
                "confidence": 0.84,
                "position_text": "слева",
                "distance_m": 2.5,
                "distance_valid": True,
            },
        ],
        "relations": [
            {
                "subject_id": "chair_0",
                "relation": "left_of",
                "object_id": "person_0",
            }
        ],
        "primary_person": {
            "entity_id": "person_0",
            "class_name": "person",
            "position_text": "по центру",
            "distance_m": 1.2345,
            "distance_valid": True,
        },
        "nearest_entity": {
            "entity_id": "person_0",
            "class_name": "person",
            "distance_m": 1.2345,
            "distance_valid": True,
        },
    }
    return node


def test_visual_scene_context_contains_fresh_yolo_facts():
    node = _make_runtime_without_ros_init()

    scene = node._compact_visual_scene_context()

    assert scene["source"] == "yolo_scene_interpreter"
    assert scene["available"] is True
    assert scene["counts"] == {"chair": 2, "person": 1}
    assert scene["entities"][0]["class_name"] == "person"
    assert (
        scene["entities"][0]["camera_distance_m"]
        == 1.234
    )
    assert (
        scene["entities"][0]["distance_reference"]
        == "camera_optical_center"
    )
    assert (
        scene["entities"][0]["distance_source"]
        == "unknown"
    )
    assert "distance_m" not in scene["entities"][0]
    assert scene["relations"][0]["relation"] == "left_of"
    assert len(json.dumps(scene, ensure_ascii=False)) < 2400


def test_visual_scene_context_marks_stale_data_unavailable():
    import time

    node = _make_runtime_without_ros_init()
    node._latest_scene_monotonic = time.monotonic() - 10.0

    scene = node._compact_visual_scene_context()

    assert scene["available"] is False
    assert scene["age_sec"] >= 9.0
