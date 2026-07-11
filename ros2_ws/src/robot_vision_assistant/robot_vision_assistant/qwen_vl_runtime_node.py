#!/usr/bin/env python3
"""Persistent Qwen3-VL runtime node for ROS 2.

The node supports both shadow and explicitly enabled active-answer output.
Even in active mode it only publishes assistant text; it has no motor, UART,
cmd_vel, action-server or executor interfaces. It provides:

- persistent HTTP connection to llama-server;
- text, scene-context and image request modes;
- frozen visual sessions that reuse one exact image payload;
- cache_prompt on follow-up questions for the same image;
- SSE streaming with time-to-first-token / first-sentence metrics;
- startup text and visual warmup.

Input topic accepts either plain text or a JSON object:

    {"query":"Что ты видишь?","mode":"image","new_session":true}
    {"query":"Что находится слева?","mode":"image"}
    {"query":"Сколько человек?","mode":"scene"}
    {"query":"Что такое одометрия?","mode":"text"}

Supported modes: auto, text, scene, image.
"""

from __future__ import annotations

import base64
import hashlib
import http.client
import json
import queue
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

from robot_vision_assistant.assistant_response_contract import (
    DEFAULT_RESPONSE_SYSTEM_PROMPT,
    render_response_context,
)


@dataclass
class RuntimeRequest:
    query: str
    mode: str = "auto"
    new_session: bool = False
    max_tokens: Optional[int] = None
    request_id: str = ""
    response_context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VisualSession:
    frame_id: str
    data_url: str
    created_monotonic: float
    last_used_monotonic: float
    frame_metadata: Dict[str, Any] = field(default_factory=dict)
    messages: List[Dict[str, Any]] = field(default_factory=list)
    turns: int = 0


class HttpStatusError(RuntimeError):
    """Non-retryable HTTP response from llama-server."""

    def __init__(self, status: int, body: str) -> None:
        self.status = int(status)
        self.body = body
        super().__init__(f"HTTP {self.status}: {self.body}")


class PersistentLlamaClient:
    """Single-threaded persistent HTTP client for llama-server."""

    def __init__(self, server_url: str, timeout_sec: float) -> None:
        parsed = urlparse(server_url)
        if parsed.scheme not in ("http", "https"):
            raise ValueError("server_url must use http or https")
        if not parsed.hostname:
            raise ValueError("server_url must contain a host")

        self.scheme = parsed.scheme
        self.host = parsed.hostname
        self.port = parsed.port or (443 if parsed.scheme == "https" else 80)
        self.base_path = parsed.path.rstrip("/")
        self.timeout_sec = timeout_sec
        self._connection: Optional[http.client.HTTPConnection] = None

    def close(self) -> None:
        if self._connection is not None:
            try:
                self._connection.close()
            except Exception:
                pass
            self._connection = None

    def _connect(self) -> http.client.HTTPConnection:
        if self._connection is None:
            connection_cls = (
                http.client.HTTPSConnection
                if self.scheme == "https"
                else http.client.HTTPConnection
            )
            self._connection = connection_cls(
                self.host,
                self.port,
                timeout=self.timeout_sec,
            )
        return self._connection

    def _path(self, path: str) -> str:
        return f"{self.base_path}{path}" if self.base_path else path

    def json_request(
        self,
        method: str,
        path: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        body = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"

        for attempt in range(2):
            try:
                connection = self._connect()
                connection.request(method, self._path(path), body=body, headers=headers)
                response = connection.getresponse()
                raw = response.read()
                if response.status >= 400:
                    raise HttpStatusError(
                        response.status,
                        raw.decode("utf-8", "replace"),
                    )
                return json.loads(raw.decode("utf-8"))
            except HttpStatusError:
                self.close()
                raise
            except Exception:
                self.close()
                if attempt == 1:
                    raise
        raise RuntimeError("unreachable")

    def stream_chat(
        self,
        payload: Dict[str, Any],
        on_fragment,
    ) -> Tuple[str, Dict[str, Any], float, Optional[float], Optional[float]]:
        request_payload = dict(payload)
        request_payload["stream"] = True
        body = json.dumps(request_payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }

        for attempt in range(2):
            started = time.perf_counter()
            pieces: List[str] = []
            final_timings: Dict[str, Any] = {}
            first_content_sec: Optional[float] = None
            first_sentence_sec: Optional[float] = None
            visible_buffer = ""

            try:
                connection = self._connect()
                connection.request(
                    "POST",
                    self._path("/v1/chat/completions"),
                    body=body,
                    headers=headers,
                )
                response = connection.getresponse()
                if response.status >= 400:
                    raw = response.read()
                    raise HttpStatusError(
                        response.status,
                        raw.decode("utf-8", "replace"),
                    )

                while True:
                    raw_line = response.readline()
                    if not raw_line:
                        break
                    line = raw_line.decode("utf-8", "replace").strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    if isinstance(chunk.get("timings"), dict):
                        final_timings = chunk["timings"]

                    delta = (
                        chunk.get("choices", [{}])[0]
                        .get("delta", {})
                        .get("content")
                    )
                    if not isinstance(delta, str) or not delta:
                        continue

                    pieces.append(delta)
                    visible = self._strip_think("".join(pieces))
                    if first_content_sec is None and visible:
                        first_content_sec = time.perf_counter() - started

                    if len(visible) > len(visible_buffer):
                        fragment = visible[len(visible_buffer):]
                        visible_buffer = visible
                        on_fragment(fragment, visible)

                    if (
                        first_sentence_sec is None
                        and re.search(r"[.!?](?:\s|$)", visible)
                    ):
                        first_sentence_sec = time.perf_counter() - started

                elapsed = time.perf_counter() - started
                answer = self._strip_think("".join(pieces))
                return (
                    answer,
                    final_timings,
                    elapsed,
                    first_content_sec,
                    first_sentence_sec,
                )
            except HttpStatusError:
                self.close()
                raise
            except Exception:
                self.close()
                if attempt == 1:
                    raise

        raise RuntimeError("unreachable")

    @staticmethod
    def _strip_think(text: str) -> str:
        cleaned = text.lstrip()
        if re.match(r"^<think>", cleaned, flags=re.IGNORECASE):
            match = re.search(
                r"</think>",
                cleaned,
                flags=re.IGNORECASE,
            )
            if match is None:
                return ""
            cleaned = cleaned[match.end():].lstrip()
        return cleaned.rstrip()


class QwenVlRuntimeNode(Node):
    def __init__(self) -> None:
        super().__init__("qwen_vl_runtime_node")

        self.declare_parameter("server_url", "http://127.0.0.1:8080")
        self.declare_parameter("model_id", "")
        self.declare_parameter("image_topic", "/camera/right/image_rect")
        self.declare_parameter("scene_topic", "/scene/interpreted_json")
        self.declare_parameter("query_topic", "/qwen_vl/query_json")
        self.declare_parameter("answer_topic", "/qwen_vl/shadow_answer")
        self.declare_parameter(
            "candidate_topic", "/qwen_vl/candidate_json"
        )
        self.declare_parameter("sentence_topic", "/qwen_vl/shadow_sentence")
        self.declare_parameter("status_topic", "/qwen_vl/status_json")
        self.declare_parameter("metrics_topic", "/qwen_vl/metrics_json")
        self.declare_parameter("active_output", False)
        self.declare_parameter("request_timeout_sec", 180.0)
        self.declare_parameter("visual_session_ttl_sec", 30.0)
        self.declare_parameter("max_visual_turns", 4)
        self.declare_parameter("jpeg_quality", 90)
        self.declare_parameter("default_max_tokens", 220)
        self.declare_parameter("warmup_text", True)
        self.declare_parameter("warmup_visual", True)
        self.declare_parameter("warmup_delay_sec", 2.0)
        self.declare_parameter("max_scene_age_sec", 2.0)
        self.declare_parameter(
            "system_prompt",
            DEFAULT_RESPONSE_SYSTEM_PROMPT,
        )

        self.server_url = str(self.get_parameter("server_url").value).rstrip("/")
        self.model_id = str(self.get_parameter("model_id").value).strip()
        self.image_topic = str(self.get_parameter("image_topic").value)
        self.scene_topic = str(self.get_parameter("scene_topic").value)
        self.query_topic = str(self.get_parameter("query_topic").value)
        self.answer_topic = str(self.get_parameter("answer_topic").value)
        self.candidate_topic = str(
            self.get_parameter("candidate_topic").value
        )
        self.sentence_topic = str(self.get_parameter("sentence_topic").value)
        self.status_topic = str(self.get_parameter("status_topic").value)
        self.metrics_topic = str(self.get_parameter("metrics_topic").value)
        self.active_output = bool(self.get_parameter("active_output").value)
        self.request_timeout_sec = float(
            self.get_parameter("request_timeout_sec").value
        )
        self.visual_session_ttl_sec = float(
            self.get_parameter("visual_session_ttl_sec").value
        )
        self.max_visual_turns = int(
            self.get_parameter("max_visual_turns").value
        )
        self.jpeg_quality = int(self.get_parameter("jpeg_quality").value)
        self.default_max_tokens = int(
            self.get_parameter("default_max_tokens").value
        )
        self.warmup_text = bool(self.get_parameter("warmup_text").value)
        self.warmup_visual = bool(self.get_parameter("warmup_visual").value)
        self.warmup_delay_sec = float(
            self.get_parameter("warmup_delay_sec").value
        )
        self.max_scene_age_sec = float(
            self.get_parameter("max_scene_age_sec").value
        )
        self.system_prompt = str(self.get_parameter("system_prompt").value)

        if (
            self.answer_topic == "/vision_assistant/answer"
            and not self.active_output
        ):
            raise RuntimeError(
                "Publishing to /vision_assistant/answer requires "
                "active_output=true"
            )

        self.bridge = CvBridge()
        self.client = PersistentLlamaClient(
            self.server_url,
            self.request_timeout_sec,
        )
        self._client_lock = threading.Lock()

        self._state_lock = threading.Lock()
        self._latest_image: Optional[Image] = None
        self._latest_image_monotonic: Optional[float] = None
        self._latest_scene: Dict[str, Any] = {}
        self._latest_scene_monotonic: Optional[float] = None
        self._visual_session: Optional[VisualSession] = None
        self._sentence_cursor = 0

        self._request_queue: "queue.Queue[Optional[RuntimeRequest]]" = queue.Queue(
            maxsize=8
        )
        self._stop_event = threading.Event()
        self._worker = threading.Thread(
            target=self._worker_loop,
            name="qwen-vl-runtime-worker",
            daemon=True,
        )
        self._worker.start()

        self.answer_pub = self.create_publisher(String, self.answer_topic, 10)
        self.candidate_pub = self.create_publisher(
            String, self.candidate_topic, 10
        )
        self.sentence_pub = self.create_publisher(String, self.sentence_topic, 10)
        self.status_pub = self.create_publisher(String, self.status_topic, 10)
        self.metrics_pub = self.create_publisher(String, self.metrics_topic, 10)

        self.image_sub = self.create_subscription(
            Image,
            self.image_topic,
            self.image_cb,
            1,
        )
        self.scene_sub = self.create_subscription(
            String,
            self.scene_topic,
            self.scene_cb,
            10,
        )
        self.query_sub = self.create_subscription(
            String,
            self.query_topic,
            self.query_cb,
            10,
        )

        self._warmup_timer = self.create_timer(
            self.warmup_delay_sec,
            self._schedule_warmup_once,
        )
        self._warmup_scheduled = False

        output_mode = "active_answer" if self.active_output else "shadow"
        self._publish_status(
            "started",
            {
                "shadow_mode": not self.active_output,
                "active_output": self.active_output,
                "output_mode": output_mode,
                "server_url": self.server_url,
                "image_topic": self.image_topic,
                "query_topic": self.query_topic,
                "answer_topic": self.answer_topic,
                "candidate_topic": self.candidate_topic,
                "sentence_topic": self.sentence_topic,
            },
        )
        self.get_logger().info(
            "qwen_vl_runtime_node started: "
            f"mode={output_mode}, server={self.server_url}, "
            f"query={self.query_topic}, answer={self.answer_topic}"
        )

    # ---------- ROS callbacks ----------

    def image_cb(self, msg: Image) -> None:
        received_monotonic = time.monotonic()
        with self._state_lock:
            self._latest_image = msg
            self._latest_image_monotonic = received_monotonic

    def scene_cb(self, msg: String) -> None:
        try:
            parsed = json.loads(msg.data)
            if not isinstance(parsed, dict):
                raise ValueError("scene payload must be a JSON object")
            with self._state_lock:
                self._latest_scene = parsed
                self._latest_scene_monotonic = time.monotonic()
        except Exception as exc:
            self.get_logger().warning(f"failed to parse scene JSON: {exc}")

    def query_cb(self, msg: String) -> None:
        try:
            request = self._parse_request(msg.data)
        except Exception as exc:
            self._publish_status("invalid_request", {"error": str(exc)})
            return

        try:
            self._request_queue.put_nowait(request)
            self._publish_status(
                "queued",
                {
                    "request_id": request.request_id,
                    "mode": request.mode,
                    "queue_size": self._request_queue.qsize(),
                },
            )
        except queue.Full:
            self._publish_status(
                "busy",
                {"request_id": request.request_id, "error": "queue_full"},
            )

    # ---------- request parsing / routing ----------

    def _parse_request(self, raw: str) -> RuntimeRequest:
        text = (raw or "").strip()
        if not text:
            raise ValueError("empty query")

        payload: Dict[str, Any]
        if text.startswith("{"):
            parsed = json.loads(text)
            if not isinstance(parsed, dict):
                raise ValueError("request JSON must be an object")
            payload = parsed
        else:
            payload = {"query": text, "mode": "auto"}

        query = str(payload.get("query", "")).strip()
        if not query:
            raise ValueError("query is required")

        mode = str(payload.get("mode", "auto")).strip().lower()
        if mode not in {"auto", "text", "scene", "image"}:
            raise ValueError(f"unsupported mode: {mode}")

        request_id = str(payload.get("request_id", "")).strip()
        if not request_id:
            request_id = f"req-{time.time_ns()}"

        max_tokens_raw = payload.get("max_tokens")
        max_tokens = None
        if max_tokens_raw is not None:
            max_tokens = max(32, min(1024, int(max_tokens_raw)))

        return RuntimeRequest(
            query=query,
            mode=mode,
            new_session=bool(payload.get("new_session", False)),
            max_tokens=max_tokens,
            request_id=request_id,
            response_context=(
                payload.get("response_context")
                if isinstance(payload.get("response_context"), dict)
                else {}
            ),
        )

    def _resolve_mode(self, request: RuntimeRequest) -> str:
        if request.mode != "auto":
            return request.mode

        q = request.query.lower().replace("ё", "е")
        image_markers = (
            "какого цвета",
            "что написано",
            "прочитай",
            "как выглядит",
            "что держит",
            "на изображении",
            "на фото",
            "на кадре",
            "опиши сцену",
            "что ты видишь",
            "что видишь",
        )
        if any(marker in q for marker in image_markers):
            return "image"

        # A deictic follow-up after an image question should stay in the same
        # visual conversation. This preserves the exact frame and enables the
        # validated multimodal prompt cache. Structured Scene Interpreter data
        # remains preferred for explicit counts, distances and safety queries.
        visual_followup_patterns = (
            r"\b(?:кто|что)\s+(?:находится\s+)?(?:слева|справа)\b",
            r"\b(?:а\s+)?(?:слева|справа)\b",
            r"\bчто\s+(?:он|она|они)\s+держит\b",
            r"\bчто\s+рядом\s+(?:с\s+ним|с\s+ней|с\s+этим)\b",
        )
        if (
            self._visual_session_is_active()
            and any(re.search(pattern, q) for pattern in visual_followup_patterns)
        ):
            return "image"

        # Counts, metric distance and safety questions must never fall back
        # to general text or monocular image guessing. Route them through the
        # structured Scene Interpreter even when its payload is stale; the
        # scene prompt will then report that fresh data is unavailable.
        structured_scene_patterns = (
            r"\bсколько\s+(?:человек|людей|персон|объектов|предметов)\b",
            r"\b(?:как|насколько)\s+далеко\b",
            r"\b(?:на\s+каком\s+)?расстоянии\b",
            r"\b(?:какое|каков[ао]?)\s+расстояние\b",
            r"\bрасстояни\w*\s+до\b",
            r"\bдистанц\w*\b",
            r"\bближайш\w*\s+(?:человек|объект|предмет)\b",
            r"\bпрепятств",
        )
        if any(re.search(pattern, q) for pattern in structured_scene_patterns):
            return "scene"

        scene_patterns = (
            r"\bгде\s+находится\b",
            r"\b(?:кто|что)\s+(?:ближе|дальше)\b",
            r"\b(?:слева|справа)\s+от\b",
            r"\b(?:перед|рядом\s+с)\s+(?:тобой|камерой|роботом)\b",
            r"\bесть\s+ли\s+(?:в\s+кадре\s+)?(?:человек|люди|объект)\b",
        )
        if any(re.search(pattern, q) for pattern in scene_patterns):
            if self._scene_is_fresh():
                return "scene"
            if self._visual_session_is_active():
                return "image"

        return "text"

    # ---------- worker ----------

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                request = self._request_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if request is None:
                break

            try:
                self._handle_request(request)
            except Exception as exc:
                self.get_logger().warning(f"Qwen request failed: {exc}")
                failure = {
                    "schema": "assistant_response_candidate",
                    "schema_version": 1,
                    "request_id": request.request_id,
                    "source": "qwen_local",
                    "success": False,
                    "answer": "",
                    "error": repr(exc),
                }
                self._publish_json(self.candidate_pub, failure)
                self._publish_status(
                    "failed",
                    {
                        "request_id": request.request_id,
                        "error": repr(exc),
                    },
                )

    def _handle_request(self, request: RuntimeRequest) -> None:
        mode = self._resolve_mode(request)
        model_id = self._ensure_model_id()
        max_tokens = request.max_tokens or self.default_max_tokens

        visual_metadata: Dict[str, Any] = {}
        if mode == "image":
            (
                messages,
                cache_prompt,
                frame_id,
                visual_metadata,
            ) = self._prepare_visual_messages(request)
        elif mode == "scene":
            messages = self._build_scene_messages(
                request.query, request.response_context
            )
            cache_prompt = False
            frame_id = None
        else:
            messages = self._build_text_messages(
                request.query, request.response_context
            )
            cache_prompt = False
            frame_id = None

        payload: Dict[str, Any] = {
            "model": model_id,
            "messages": messages,
            "temperature": 0.0,
            "seed": 42,
            "max_tokens": max_tokens,
            "cache_prompt": cache_prompt,
            "chat_template_kwargs": {"enable_thinking": False},
            "reasoning_format": "none",
        }

        guard_visual_overview = (
            mode == "image"
            and self._is_generic_visual_overview(request.query)
        )
        self._sentence_cursor = 0
        sentence_accumulator = ""

        def on_fragment(fragment: str, visible: str) -> None:
            nonlocal sentence_accumulator
            # Generic visual overviews are buffered until completion so that
            # unsupported human/body/clothing details cannot reach the future
            # TTS path before the deterministic answer guard has run.
            if guard_visual_overview:
                return
            sentence_accumulator += fragment
            ready, remainder = self._extract_complete_sentences(
                sentence_accumulator
            )
            sentence_accumulator = remainder
            for sentence in ready:
                self._publish_string(self.sentence_pub, sentence)

        self._publish_status(
            "running",
            {
                "request_id": request.request_id,
                "mode": mode,
                "cache_prompt": cache_prompt,
                "frame_id": frame_id,
                "active_output": self.active_output,
                **visual_metadata,
            },
        )

        started = time.perf_counter()
        with self._client_lock:
            answer, timings, elapsed, ttft, first_sentence = (
                self.client.stream_chat(
                    payload,
                    on_fragment,
                )
            )
        wall_elapsed = time.perf_counter() - started

        raw_answer = answer
        answer_guard_applied = False
        if guard_visual_overview:
            answer, answer_guard_applied = self._guard_visual_overview_answer(
                answer
            )
            guarded_sentences, guarded_remainder = (
                self._extract_complete_sentences(answer)
            )
            for sentence in guarded_sentences:
                self._publish_string(self.sentence_pub, sentence)
            if guarded_remainder.strip():
                self._publish_string(
                    self.sentence_pub,
                    guarded_remainder.strip(),
                )
        elif sentence_accumulator.strip():
            self._publish_string(self.sentence_pub, sentence_accumulator.strip())

        self._publish_string(self.answer_pub, answer)
        self._publish_json(
            self.candidate_pub,
            {
                "schema": "assistant_response_candidate",
                "schema_version": 1,
                "request_id": request.request_id,
                "source": "qwen_local",
                "success": True,
                "answer": answer,
                "mode": mode,
                "frame_id": frame_id,
                "cache_prompt": cache_prompt,
                "active_output": self.active_output,
            },
        )

        if mode == "image":
            self._commit_visual_turn(request.query, answer)

        metrics = {
            "request_id": request.request_id,
            "mode": mode,
            "cache_prompt": cache_prompt,
            "frame_id": frame_id,
            "elapsed_sec": round(elapsed, 4),
            "wall_elapsed_sec": round(wall_elapsed, 4),
            "time_to_first_content_sec": (
                round(ttft, 4) if ttft is not None else None
            ),
            "time_to_first_sentence_sec": (
                round(first_sentence, 4)
                if first_sentence is not None
                else None
            ),
            "prompt_tokens": timings.get("prompt_n"),
            "cached_prompt_tokens": timings.get("cache_n"),
            "prompt_tps": timings.get("prompt_per_second"),
            "generation_tokens": timings.get("predicted_n"),
            "generation_tps": timings.get("predicted_per_second"),
            "answer_chars": len(answer),
            "raw_answer_chars": len(raw_answer),
            "answer_guard": (
                "visual_overview_v1" if guard_visual_overview else None
            ),
            "answer_guard_applied": answer_guard_applied,
            "active_output": self.active_output,
            **visual_metadata,
        }
        self._publish_json(self.metrics_pub, metrics)
        self._publish_status("done", metrics)

    # ---------- message builders ----------

    def _build_text_messages(
        self,
        query: str,
        response_context: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        context_block = render_response_context(response_context)
        return [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": f"{context_block}\nQUESTION={query}",
            },
        ]

    def _build_scene_messages(
        self,
        query: str,
        response_context: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        scene = self._compact_scene_context()
        context_block = render_response_context(response_context)
        return [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": (
                    "Ответь естественно, но только по достоверным данным "
                    "Scene Interpreter. Не добавляй объекты, количества, "
                    "расстояния или положения, которых нет в JSON. Если "
                    "available=false или нужное поле отсутствует/null, прямо "
                    "скажи, что свежих данных недостаточно. Никогда не оценивай "
                    "расстояние по догадке.\n"
                    f"{context_block}\n"
                    f"SCENE_JSON={json.dumps(scene, ensure_ascii=False, separators=(',', ':'))}\n"
                    f"QUESTION={query}"
                ),
            },
        ]

    @staticmethod
    def _compact_visual_response_context(
        response_context: Optional[Dict[str, Any]],
    ) -> str:
        """Keep only safety-critical metadata for image requests."""
        source = (
            response_context
            if isinstance(response_context, dict)
            else {}
        )
        action_source = source.get("action_result")
        rules_source = source.get("rules")
        memory_source = source.get("memory_context")

        action_result: Dict[str, Any] = {}
        if isinstance(action_source, dict):
            for key in (
                "state",
                "execution_allowed",
                "reason",
            ):
                if key in action_source:
                    action_result[key] = action_source[key]

        rules: Dict[str, Any] = {}
        if isinstance(rules_source, dict):
            for key in (
                "sensor_values_must_be_supplied",
                "claim_action_only_after_executor_confirmation",
                "hardware_commands_forbidden",
            ):
                if key in rules_source:
                    rules[key] = rules_source[key]

        compact = {
            "schema": "robot_visual_response_context",
            "schema_version": 1,
            "request_id": source.get("request_id"),
            "route": source.get("route"),
            "action_result": action_result,
            "rules": rules,
            "memory_context": (
                memory_source
                if isinstance(memory_source, dict)
                else {}
            ),
        }
        return (
            "ROBOT_VISUAL_CONTEXT="
            + json.dumps(
                compact,
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )

    @staticmethod
    def _compact_visual_entity(
        entity: Any,
    ) -> Dict[str, Any]:
        if not isinstance(entity, dict):
            return {}

        compact: Dict[str, Any] = {}
        for key in (
            "entity_id",
            "track_id",
            "class_name",
            "label",
            "semantic_group",
            "confirmed",
            "position_text",
            "side",
            "distance_valid",
            "depth_confidence",
            "depth_status",
            "depth_source",
        ):
            value = entity.get(key)
            if value is not None:
                compact[key] = value

        confidence = entity.get("confidence")
        if isinstance(confidence, (int, float)):
            compact["confidence"] = round(float(confidence), 3)

        distance = entity.get("distance_m")
        if (
            bool(entity.get("distance_valid"))
            and isinstance(distance, (int, float))
        ):
            compact["camera_distance_m"] = round(
                float(distance),
                3,
            )
            compact["distance_reference"] = (
                "camera_optical_center"
            )
            compact["distance_source"] = str(
                entity.get("depth_source")
                or "unknown"
            )

        return compact

    def _compact_visual_scene_context(self) -> Dict[str, Any]:
        """Return bounded, fresh YOLO/Scene Interpreter facts."""
        with self._state_lock:
            scene = dict(self._latest_scene)
            received = self._latest_scene_monotonic

        age_sec: Optional[float] = None
        if received is not None:
            age_sec = max(0.0, time.monotonic() - received)

        available = bool(
            received is not None
            and age_sec is not None
            and age_sec <= self.max_scene_age_sec
        )

        counts: Dict[str, int] = {}
        raw_counts = scene.get("counts")
        if isinstance(raw_counts, dict):
            for label in sorted(raw_counts)[:16]:
                try:
                    value = int(raw_counts[label])
                except (TypeError, ValueError):
                    continue
                if value > 0:
                    counts[str(label)] = value

        raw_entities = scene.get("salient_entities")
        if not isinstance(raw_entities, list) or not raw_entities:
            raw_entities = []
            persons = scene.get("persons")
            objects = scene.get("objects")
            if isinstance(persons, list):
                raw_entities.extend(persons)
            if isinstance(objects, list):
                raw_entities.extend(objects)

        entities: List[Dict[str, Any]] = []
        seen_ids = set()
        for raw_entity in raw_entities:
            compact = self._compact_visual_entity(raw_entity)
            if not compact:
                continue
            entity_id = str(
                compact.get("entity_id")
                or compact.get("track_id")
                or f"entity_{len(entities)}"
            )
            if entity_id in seen_ids:
                continue
            seen_ids.add(entity_id)
            compact["entity_id"] = entity_id
            entities.append(compact)
            if len(entities) >= 6:
                break

        relations: List[Dict[str, Any]] = []
        raw_relations = scene.get("relations")
        if isinstance(raw_relations, list):
            for relation in raw_relations:
                if not isinstance(relation, dict):
                    continue
                item = {
                    key: relation.get(key)
                    for key in (
                        "subject_id",
                        "relation",
                        "object_id",
                    )
                    if relation.get(key) is not None
                }
                if len(item) == 3:
                    relations.append(item)
                if len(relations) >= 6:
                    break

        compact_scene: Dict[str, Any] = {
            "source": "yolo_scene_interpreter",
            "available": available,
            "age_sec": (
                round(age_sec, 3)
                if age_sec is not None
                else None
            ),
            "source_timestamp": scene.get("source_timestamp"),
            "counts": counts,
            "entities": entities,
            "relations": relations,
            "primary_person": self._compact_visual_entity(
                scene.get("primary_person")
            ),
            "nearest_entity": self._compact_visual_entity(
                scene.get("nearest_entity")
            ),
        }

        serialized = json.dumps(
            compact_scene,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        if len(serialized) > 2200:
            compact_scene["relations"] = []
            compact_scene["entities"] = entities[:4]
            serialized = json.dumps(
                compact_scene,
                ensure_ascii=False,
                separators=(",", ":"),
            )
        if len(serialized) > 2200:
            compact_scene["primary_person"] = {}
            compact_scene["nearest_entity"] = {}
            compact_scene["entities"] = entities[:3]

        return compact_scene

    def _prepare_visual_messages(
        self,
        request: RuntimeRequest,
    ) -> Tuple[List[Dict[str, Any]], bool, str, Dict[str, Any]]:
        now = time.monotonic()
        with self._state_lock:
            session = self._visual_session

        expired = (
            session is None
            or now - session.last_used_monotonic > self.visual_session_ttl_sec
            or session.turns >= self.max_visual_turns
        )
        start_new = request.new_session or expired

        if start_new:
            frame_id, data_url, frame_metadata = (
                self._capture_frame_data_url()
            )
            context_block = self._compact_visual_response_context(
                request.response_context
            )
            scene_at_capture = self._compact_visual_scene_context()
            messages: List[Dict[str, Any]] = [
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Ответь по текущему изображению и данным сенсоров. "
                                "Для внешнего вида и общего описания используй "
                                "изображение. Для классов, количества, положения и "
                                "метрических расстояний используй только "
                                "VERIFIED_SCENE_FACTS от YOLO/Scene Interpreter. "
                                "Поле camera_distance_m всегда означает расстояние "
                                "от оптического центра камеры до конкретного объекта, "
                                "а не расстояние между двумя объектами. Никогда не "
                                "выводи межобъектное расстояние из двух значений "
                                "camera_distance_m. "
                                "Если available=false или нужного поля нет, прямо "
                                "скажи, что свежих данных недостаточно. Не придумывай "
                                "сенсорные значения, поверхности, отношения или детали. "
                                "Составь 1–3 отдельных коротких предложения. При "
                                "общем описании человека разрешено только нейтрально "
                                "сказать, что в кадре виден человек; не описывай его "
                                "пол, возраст, одежду, тело, эмоции, позу или действия, "
                                "если пользователь прямо об этом не спросил.\n"
                                f"{context_block}\n"
                                "VERIFIED_SCENE_FACTS="
                                f"{json.dumps(scene_at_capture, ensure_ascii=False, separators=(',', ':'))}\n"
                                f"QUESTION={request.query}"
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url},
                        },
                    ],
                },
            ]
            new_session = VisualSession(
                frame_id=frame_id,
                data_url=data_url,
                created_monotonic=now,
                last_used_monotonic=now,
                frame_metadata=dict(frame_metadata),
                messages=messages,
                turns=0,
            )
            with self._state_lock:
                self._visual_session = new_session
            metadata = dict(frame_metadata)
            metadata["visual_session_age_sec"] = 0.0
            metadata["visual_session_turns"] = 0
            return list(messages), False, frame_id, metadata

        assert session is not None
        messages = list(session.messages)
        messages.append({"role": "user", "content": request.query})
        metadata = dict(session.frame_metadata)
        metadata["visual_session_age_sec"] = round(
            max(0.0, now - session.created_monotonic),
            4,
        )
        metadata["visual_session_turns"] = session.turns
        return messages, True, session.frame_id, metadata

    @staticmethod
    def _is_generic_visual_overview(query: str) -> bool:
        normalized = query.lower().replace("ё", "е")
        patterns = (
            r"\bчто\s+ты\s+видишь\b",
            r"\bчто\s+видишь\b",
            r"\bопиши\s+(?:эту\s+)?сцену\b",
            r"\bчто\s+(?:есть|находится)\s+в\s+кадре\b",
            r"\bчто\s+в\s+кадре\b",
            r"\bчто\s+перед\s+тобой\b",
        )
        return any(re.search(pattern, normalized) for pattern in patterns)

    def _guard_visual_overview_answer(
        self,
        answer: str,
    ) -> Tuple[str, bool]:
        """Remove unsupported human-detail claims from generic overviews.

        Prompt-only restrictions are not sufficient for a small local VLM.
        This guard is intentionally narrow: it runs only for generic scene
        overview questions. Explicit questions about appearance are left
        untouched and continue through the normal visual path.
        """
        original = answer.strip()
        if not original:
            return original, False

        parts = re.split(r"(?<=[.!?])\s+|\n+", original)
        human_reference = re.compile(
            r"\b(?:человек|мужчина|женщина|парень|девушка|"
            r"мальчик|девочка|он|она|его|ее|неё|него|нему|ней)\b",
            flags=re.IGNORECASE,
        )
        unsupported_human_detail = re.compile(
            r"\b(?:"
            r"гол\w*|обнажен\w*|бель[её]|трус\w*|"
            r"одет\w*|одежд\w*|футболк\w*|рубашк\w*|"
            r"штан\w*|брюк\w*|плать\w*|бород\w*|ус(?:ы|ами)?|"
            r"волос\w*|тел\w*|лиц\w*|губ\w*|рук\w*|ног\w*|"
            r"улыба\w*|груст\w*|сердит\w*|эмоц\w*|"
            r"сид\w*|леж\w*|наклон\w*|прижат\w*|держ\w*"
            r")\b",
            flags=re.IGNORECASE,
        )
        gendered_person = re.compile(
            r"\b(?:мужчина|женщина|парень|девушка|мальчик|девочка)\b",
            flags=re.IGNORECASE,
        )

        kept: List[str] = []
        has_safe_person_sentence = False
        for part in parts:
            sentence = part.strip()
            if not sentence:
                continue

            # Remove deictic human references from otherwise useful object
            # statements, e.g. "Слева от него стоит коробка".
            sentence = re.sub(
                r"\b(слева|справа)\s+от\s+"
                r"(?:него|нее|неё|человека|мужчины|женщины)\b",
                r"\1",
                sentence,
                flags=re.IGNORECASE,
            )

            references_human = bool(human_reference.search(sentence))
            if references_human and unsupported_human_detail.search(sentence):
                continue

            sentence = gendered_person.sub("человек", sentence)
            sentence = re.sub(r"\s+", " ", sentence).strip()
            if not sentence:
                continue

            if re.search(r"\bчеловек\b", sentence, flags=re.IGNORECASE):
                has_safe_person_sentence = True
            kept.append(sentence)

        scene = self._compact_scene_context()
        counts = scene.get("counts", {})
        person_count = 0
        if isinstance(counts, dict):
            try:
                person_count = int(counts.get("person", 0) or 0)
            except (TypeError, ValueError):
                person_count = 0

        if person_count > 0 and not has_safe_person_sentence:
            person_summary = (
                "В кадре виден человек."
                if person_count == 1
                else f"В кадре видно людей: {person_count}."
            )
            kept.insert(0, person_summary)

        if not kept:
            guarded = "Не могу уверенно описать сцену."
        else:
            guarded = " ".join(kept)

        return guarded, guarded != original

    def _commit_visual_turn(self, query: str, answer: str) -> None:
        now = time.monotonic()
        with self._state_lock:
            session = self._visual_session
            if session is None:
                return

            if session.turns == 0:
                # The first user message already contains the image and query.
                session.messages.append({"role": "assistant", "content": answer})
            else:
                session.messages.append({"role": "user", "content": query})
                session.messages.append({"role": "assistant", "content": answer})

            session.turns += 1
            session.last_used_monotonic = now

    # ---------- image / scene helpers ----------

    def _capture_frame_data_url(
        self,
    ) -> Tuple[str, str, Dict[str, Any]]:
        capture_monotonic = time.monotonic()
        with self._state_lock:
            msg = self._latest_image
            image_received_monotonic = self._latest_image_monotonic
            scene = dict(self._latest_scene)
            scene_received_monotonic = self._latest_scene_monotonic

        if msg is None:
            raise RuntimeError("no camera frame received")

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        ok, encoded = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
        )
        if not ok:
            raise RuntimeError("JPEG encoding failed")

        jpeg = encoded.tobytes()
        frame_id = hashlib.sha256(jpeg).hexdigest()[:16]
        data_url = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode(
            "ascii"
        )

        stamp = msg.header.stamp
        image_stamp_sec = int(stamp.sec)
        image_stamp_nanosec = int(stamp.nanosec)
        image_source_timestamp = (
            float(image_stamp_sec)
            + float(image_stamp_nanosec) / 1_000_000_000.0
        )

        image_receive_age_sec = None
        if image_received_monotonic is not None:
            image_receive_age_sec = round(
                max(
                    0.0,
                    capture_monotonic - image_received_monotonic,
                ),
                4,
            )

        scene_receive_age_sec = None
        if scene_received_monotonic is not None:
            scene_receive_age_sec = round(
                max(
                    0.0,
                    capture_monotonic - scene_received_monotonic,
                ),
                4,
            )

        counts = scene.get("counts", {})
        if not isinstance(counts, dict):
            counts = {}

        frame_metadata: Dict[str, Any] = {
            "image_stamp_sec": image_stamp_sec,
            "image_stamp_nanosec": image_stamp_nanosec,
            "image_source_timestamp": image_source_timestamp,
            "image_receive_age_at_capture_sec": image_receive_age_sec,
            "scene_timestamp_at_capture": scene.get("timestamp"),
            "scene_source_timestamp_at_capture": scene.get(
                "source_timestamp"
            ),
            "scene_receive_age_at_capture_sec": scene_receive_age_sec,
            "scene_counts_at_capture": counts,
            "scene_person_count_at_capture": counts.get("person"),
        }
        return frame_id, data_url, frame_metadata

    def _scene_is_fresh(self) -> bool:
        with self._state_lock:
            received = self._latest_scene_monotonic
        return (
            received is not None
            and time.monotonic() - received <= self.max_scene_age_sec
        )

    def _visual_session_is_active(self) -> bool:
        now = time.monotonic()
        with self._state_lock:
            session = self._visual_session
            if session is None:
                return False
            return (
                now - session.last_used_monotonic
                <= self.visual_session_ttl_sec
                and session.turns < self.max_visual_turns
            )

    def _compact_scene_context(self) -> Dict[str, Any]:
        with self._state_lock:
            scene = dict(self._latest_scene)
            received = self._latest_scene_monotonic

        age_sec = None
        if received is not None:
            age_sec = round(max(0.0, time.monotonic() - received), 3)

        return {
            "available": self._scene_is_fresh(),
            "age_sec": age_sec,
            "counts": scene.get("counts", {}),
            "primary_person": scene.get("primary_person"),
            "nearest_entity": scene.get("nearest_entity"),
            "persons": scene.get("persons", [])[:5],
            "objects": scene.get("objects", [])[:10],
            "relations": scene.get("relations", [])[:12],
            "person_context": scene.get("person_context", {}),
            "changes": scene.get("changes", {}),
        }

    # ---------- warmup ----------

    def _schedule_warmup_once(self) -> None:
        if self._warmup_scheduled:
            return
        self._warmup_scheduled = True
        try:
            self.destroy_timer(self._warmup_timer)
        except Exception:
            pass

        threading.Thread(
            target=self._run_warmup,
            name="qwen-vl-warmup",
            daemon=True,
        ).start()

    def _run_warmup(self) -> None:
        try:
            model_id = self._ensure_model_id()
            if self.warmup_text:
                payload = {
                    "model": model_id,
                    "messages": [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": "Ответь одним словом: готов."},
                    ],
                    "temperature": 0.0,
                    "seed": 42,
                    "max_tokens": 16,
                    "stream": False,
                    "cache_prompt": False,
                    "chat_template_kwargs": {"enable_thinking": False},
                    "reasoning_format": "none",
                }
                with self._client_lock:
                    self.client.json_request(
                        "POST", "/v1/chat/completions", payload
                    )
                self._publish_status("warmup_text_done", {})

            if self.warmup_visual:
                deadline = time.monotonic() + 20.0
                while time.monotonic() < deadline:
                    with self._state_lock:
                        has_frame = self._latest_image is not None
                    if has_frame:
                        break
                    time.sleep(0.2)

                with self._state_lock:
                    has_frame = self._latest_image is not None
                if has_frame:
                    _, data_url = self._capture_frame_data_url()
                    payload = {
                        "model": model_id,
                        "messages": [
                            {"role": "system", "content": self.system_prompt},
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": "Ответь: кадр получен."},
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": data_url},
                                    },
                                ],
                            },
                        ],
                        "temperature": 0.0,
                        "seed": 42,
                        "max_tokens": 16,
                        "stream": False,
                        "cache_prompt": False,
                        "chat_template_kwargs": {"enable_thinking": False},
                        "reasoning_format": "none",
                    }
                    with self._client_lock:
                        self.client.json_request(
                            "POST", "/v1/chat/completions", payload
                        )
                    self._publish_status("warmup_visual_done", {})
                else:
                    self._publish_status(
                        "warmup_visual_skipped", {"reason": "no_frame"}
                    )
        except Exception as exc:
            self.get_logger().warning(f"Qwen warmup failed: {exc}")
            self._publish_status("warmup_failed", {"error": repr(exc)})

    # ---------- model / streaming helpers ----------

    def _ensure_model_id(self) -> str:
        if self.model_id:
            return self.model_id
        with self._client_lock:
            models = self.client.json_request("GET", "/v1/models")
        data = models.get("data", [])
        if not data:
            raise RuntimeError("llama-server returned no models")
        self.model_id = str(data[0].get("id", "")).strip()
        if not self.model_id:
            raise RuntimeError("model id is empty")
        self._publish_status("model_detected", {"model_id": self.model_id})
        return self.model_id

    @staticmethod
    def _extract_complete_sentences(text: str) -> Tuple[List[str], str]:
        sentences: List[str] = []
        start = 0
        for match in re.finditer(r"[.!?](?=\s|$)", text):
            end = match.end()
            sentence = text[start:end].strip()
            if sentence:
                sentences.append(sentence)
            start = end
        return sentences, text[start:].lstrip()

    # ---------- publishers ----------

    @staticmethod
    def _publish_string(publisher, text: str) -> None:
        message = String()
        message.data = text
        publisher.publish(message)

    def _publish_json(self, publisher, payload: Dict[str, Any]) -> None:
        self._publish_string(
            publisher,
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        )

    def _publish_status(self, state: str, details: Dict[str, Any]) -> None:
        payload = {"state": state, **details}
        self._publish_json(self.status_pub, payload)

    def destroy_node(self) -> None:
        self._stop_event.set()
        try:
            self._request_queue.put_nowait(None)
        except queue.Full:
            pass
        self._worker.join(timeout=3.0)
        self.client.close()
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = QwenVlRuntimeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
