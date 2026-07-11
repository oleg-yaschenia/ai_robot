#!/usr/bin/env python3
"""Shadow ROS 2 node for Local Semantic Resolver v1.

Input:
    /assistant/core/request_plan_json

Only UNKNOWN request plans with semantic_fallback.required=true are processed.

Output:
    /assistant/semantic/request_plan_json
    /assistant/semantic/status_json

The node has no cmd_vel, UART, motor, servo, action-server or executor
interfaces. It always preserves shadow_mode=true and execution_allowed=false.
"""

from __future__ import annotations

import json
import queue
import threading
from collections import deque
from typing import Any, Deque, Dict, Optional, Set

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from robot_vision_assistant.assistant_request_contract import (
    validate_request_plan,
)
from robot_vision_assistant.local_semantic_resolver import (
    LlamaSemanticClient,
    LocalSemanticResolver,
    apply_semantic_resolution,
)


class LocalSemanticResolverNode(Node):
    def __init__(self) -> None:
        super().__init__("local_semantic_resolver_node")

        self.declare_parameter(
            "input_topic",
            "/assistant/core/request_plan_json",
        )
        self.declare_parameter(
            "output_topic",
            "/assistant/semantic/request_plan_json",
        )
        self.declare_parameter(
            "status_topic",
            "/assistant/semantic/status_json",
        )
        self.declare_parameter(
            "server_url",
            "http://127.0.0.1:8080",
        )
        self.declare_parameter("model_id", "")
        self.declare_parameter("request_timeout_sec", 30.0)
        self.declare_parameter("max_tokens", 96)
        self.declare_parameter("queue_size", 2)
        self.declare_parameter("dedupe_history", 256)

        self.input_topic = str(
            self.get_parameter("input_topic").value
        )
        self.output_topic = str(
            self.get_parameter("output_topic").value
        )
        self.status_topic = str(
            self.get_parameter("status_topic").value
        )
        server_url = str(
            self.get_parameter("server_url").value
        )
        model_id = str(
            self.get_parameter("model_id").value
        )
        timeout_sec = float(
            self.get_parameter("request_timeout_sec").value
        )
        max_tokens = int(
            self.get_parameter("max_tokens").value
        )
        queue_size = max(
            1,
            int(self.get_parameter("queue_size").value),
        )
        self._dedupe_history = max(
            1,
            int(self.get_parameter("dedupe_history").value),
        )

        self._client = LlamaSemanticClient(
            server_url=server_url,
            model_id=model_id,
            timeout_sec=timeout_sec,
            max_tokens=max_tokens,
        )
        self._resolver = LocalSemanticResolver(
            self._client.classify
        )

        self._request_queue: "queue.Queue[Optional[Dict[str, Any]]]" = (
            queue.Queue(maxsize=queue_size)
        )
        self._stop_event = threading.Event()
        self._seen_order: Deque[str] = deque()
        self._seen_ids: Set[str] = set()

        self.output_pub = self.create_publisher(
            String,
            self.output_topic,
            10,
        )
        self.status_pub = self.create_publisher(
            String,
            self.status_topic,
            10,
        )
        self.input_sub = self.create_subscription(
            String,
            self.input_topic,
            self.request_plan_cb,
            10,
        )

        self._worker = threading.Thread(
            target=self._worker_loop,
            name="local-semantic-resolver-worker",
            daemon=True,
        )
        self._worker.start()

        self.get_logger().info(
            "local_semantic_resolver_node started: "
            f"input={self.input_topic}, "
            f"output={self.output_topic}, "
            f"status={self.status_topic}, "
            f"queue_size={queue_size}, "
            "shadow_mode=true, execution_allowed=false"
        )

    @staticmethod
    def _json_message(payload: Dict[str, Any]) -> String:
        message = String()
        message.data = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return message

    def _publish_status(
        self,
        *,
        state: str,
        request_id: str = "",
        **extra: Any,
    ) -> None:
        payload: Dict[str, Any] = {
            "state": state,
            "request_id": request_id,
            "shadow_mode": True,
            "execution_allowed": False,
        }
        payload.update(extra)
        self.status_pub.publish(
            self._json_message(payload)
        )

    def _remember_request_id(self, request_id: str) -> bool:
        if request_id in self._seen_ids:
            return False

        while len(self._seen_order) >= self._dedupe_history:
            old = self._seen_order.popleft()
            self._seen_ids.discard(old)

        self._seen_order.append(request_id)
        self._seen_ids.add(request_id)
        return True

    def request_plan_cb(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self._publish_status(
                state="rejected",
                reason="invalid_json",
                detail=str(exc),
            )
            return

        if not isinstance(payload, dict):
            self._publish_status(
                state="rejected",
                reason="message_must_be_object",
            )
            return

        errors = validate_request_plan(payload)
        if errors:
            self._publish_status(
                state="rejected",
                request_id=str(payload.get("request_id", "")),
                reason="invalid_request_plan",
                errors=errors,
            )
            return

        request_id = str(payload.get("request_id", "")).strip()
        resolution_status = (
            payload.get("resolution", {}).get("status")
        )
        fallback_required = (
            payload.get("semantic_fallback", {}).get("required")
        )

        if (
            resolution_status != "UNKNOWN"
            or fallback_required is not True
        ):
            self._publish_status(
                state="ignored",
                request_id=request_id,
                reason="semantic_fallback_not_required",
                resolution_status=resolution_status,
            )
            return

        query = str(payload.get("query", "")).strip()
        if not request_id or not query:
            self._publish_status(
                state="rejected",
                request_id=request_id,
                reason="request_id_or_query_missing",
            )
            return

        if request_id in self._seen_ids:
            self._publish_status(
                state="ignored",
                request_id=request_id,
                reason="duplicate_request_id",
            )
            return

        try:
            self._request_queue.put_nowait(payload)
        except queue.Full:
            self._publish_status(
                state="rejected",
                request_id=request_id,
                reason="semantic_queue_full",
            )
            return

        self._remember_request_id(request_id)

        self._publish_status(
            state="queued",
            request_id=request_id,
            queue_depth=self._request_queue.qsize(),
        )

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                item = self._request_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if item is None:
                self._request_queue.task_done()
                break

            request_id = str(item.get("request_id", ""))
            query = str(item.get("query", ""))

            self._publish_status(
                state="resolving",
                request_id=request_id,
            )

            try:
                semantic_result, metrics = self._resolver.resolve(
                    query
                )
                final_plan = apply_semantic_resolution(
                    item,
                    semantic_result,
                )
                self.output_pub.publish(
                    self._json_message(final_plan)
                )

                self._publish_status(
                    state="completed",
                    request_id=request_id,
                    resolution_status=final_plan[
                        "resolution"
                    ]["status"],
                    intent=(
                        final_plan.get("plan", {}).get("intent")
                        or (
                            final_plan.get("resolution", {})
                            .get("candidates", [None])[0]
                            if final_plan.get("resolution", {}).get(
                                "candidates"
                            )
                            else None
                        )
                    ),
                    model_called=bool(
                        metrics.get("model_called", False)
                    ),
                    elapsed_sec=float(
                        metrics.get("elapsed_sec", 0.0)
                    ),
                )
            except Exception as exc:
                self.get_logger().error(
                    "semantic resolution failed for "
                    f"request_id={request_id}: "
                    f"{type(exc).__name__}: {exc}"
                )
                self._publish_status(
                    state="error",
                    request_id=request_id,
                    reason="semantic_resolution_failed",
                    error_type=type(exc).__name__,
                    detail=str(exc),
                )
            finally:
                self._request_queue.task_done()

    def destroy_node(self) -> bool:
        self._stop_event.set()
        try:
            self._request_queue.put_nowait(None)
        except queue.Full:
            pass

        if hasattr(self, "_worker"):
            self._worker.join(timeout=2.0)

        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LocalSemanticResolverNode()

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
