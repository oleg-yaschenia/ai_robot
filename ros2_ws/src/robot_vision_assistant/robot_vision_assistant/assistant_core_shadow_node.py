#!/usr/bin/env python3
"""Assistant Core v2 shadow node.

This stage performs strict known-intent resolution and emits clarification or
semantic-fallback decisions. It does not intercept the existing assistant path
and cannot execute actions.
"""

from __future__ import annotations

import json

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from robot_vision_assistant.clarification_manager import (
    ClarificationManager,
)
from robot_vision_assistant.known_intent_resolver import (
    KnownIntentResolver,
)


class AssistantCoreShadowNode(Node):
    def __init__(self) -> None:
        super().__init__("assistant_core_shadow_node")

        self.declare_parameter(
            "query_topic",
            "/vision_assistant/query",
        )
        self.declare_parameter(
            "plan_topic",
            "/assistant/core/request_plan_json",
        )
        self.declare_parameter(
            "clarification_topic",
            "/assistant/clarification/request_json",
        )
        self.declare_parameter("max_clarification_attempts", 2)

        self.query_topic = str(
            self.get_parameter("query_topic").value
        )
        self.plan_topic = str(
            self.get_parameter("plan_topic").value
        )
        self.clarification_topic = str(
            self.get_parameter("clarification_topic").value
        )

        self.resolver = KnownIntentResolver()
        self.clarification_manager = ClarificationManager(
            max_attempts=int(
                self.get_parameter(
                    "max_clarification_attempts"
                ).value
            )
        )

        self.plan_pub = self.create_publisher(
            String,
            self.plan_topic,
            10,
        )
        self.clarification_pub = self.create_publisher(
            String,
            self.clarification_topic,
            10,
        )
        self.query_sub = self.create_subscription(
            String,
            self.query_topic,
            self.query_cb,
            10,
        )

        self.get_logger().info(
            "assistant_core_shadow_node started: "
            f"query={self.query_topic}, "
            f"plan={self.plan_topic}, "
            f"clarification={self.clarification_topic}, "
            "shadow_mode=true, execution_allowed=false"
        )

    @staticmethod
    def _json_message(payload: dict) -> String:
        message = String()
        message.data = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return message

    def query_cb(self, msg: String) -> None:
        query = (msg.data or "").strip()
        if not query:
            return

        try:
            request_plan = self.resolver.resolve(query)
        except Exception as exc:
            self.get_logger().warning(
                f"known intent resolution failed: {exc}"
            )
            return

        self.plan_pub.publish(
            self._json_message(request_plan)
        )

        status = request_plan["resolution"]["status"]
        intent = request_plan["plan"].get("intent")

        if status == "AMBIGUOUS":
            clarification = (
                self.clarification_manager.open_from_plan(
                    request_plan
                )
            )
            self.clarification_pub.publish(
                self._json_message(clarification)
            )

        self.get_logger().info(
            "assistant core shadow decision: "
            f"status={status}, intent={intent}, "
            f"semantic_fallback="
            f"{request_plan['semantic_fallback']['required']}, "
            f"query={query!r}"
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = AssistantCoreShadowNode()

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
