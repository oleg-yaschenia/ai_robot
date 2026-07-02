#!/usr/bin/env python3
"""ROS shadow node for Assistant Router v1.

It observes user queries and publishes route decisions. It never republishes a
query and cannot execute actions.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from robot_vision_assistant.assistant_router import (
    build_route_decision,
    validate_route_decision,
)


class AssistantRouterNode(Node):
    def __init__(self) -> None:
        super().__init__("assistant_router_node")

        self.declare_parameter(
            "query_topic",
            "/vision_assistant/query",
        )
        self.declare_parameter(
            "scene_topic",
            "/scene/interpreted_json",
        )
        self.declare_parameter(
            "decision_topic",
            "/assistant/router/decision_json",
        )
        self.declare_parameter("max_scene_age_sec", 2.0)

        self.query_topic = str(
            self.get_parameter("query_topic").value
        )
        self.scene_topic = str(
            self.get_parameter("scene_topic").value
        )
        self.decision_topic = str(
            self.get_parameter("decision_topic").value
        )
        self.max_scene_age_sec = float(
            self.get_parameter("max_scene_age_sec").value
        )

        self.last_scene: Dict[str, Any] = {}
        self.last_scene_received_monotonic: Optional[float] = None

        self.decision_pub = self.create_publisher(
            String,
            self.decision_topic,
            10,
        )
        self.query_sub = self.create_subscription(
            String,
            self.query_topic,
            self.query_cb,
            10,
        )
        self.scene_sub = self.create_subscription(
            String,
            self.scene_topic,
            self.scene_cb,
            10,
        )

        self.get_logger().info(
            "assistant_router_node started in shadow mode: "
            f"query={self.query_topic}, "
            f"scene={self.scene_topic}, "
            f"decision={self.decision_topic}, "
            "execution_allowed=false"
        )

    def scene_cb(self, msg: String) -> None:
        try:
            parsed = json.loads(msg.data)
            if not isinstance(parsed, dict):
                raise ValueError("scene payload must be a JSON object")
            self.last_scene = parsed
            self.last_scene_received_monotonic = time.monotonic()
        except Exception as exc:
            self.get_logger().warning(
                f"failed to parse interpreted scene: {exc}"
            )

    def _scene_context(self) -> Dict[str, Any]:
        if self.last_scene_received_monotonic is None:
            return {
                "available": False,
                "age_sec": None,
                "entity_count": 0,
                "counts": {},
            }

        age_sec = max(
            0.0,
            time.monotonic() - self.last_scene_received_monotonic,
        )
        entities = (
            self.last_scene.get("persons", [])
            + self.last_scene.get("objects", [])
        )
        return {
            "available": age_sec <= self.max_scene_age_sec,
            "age_sec": round(age_sec, 3),
            "entity_count": len(entities),
            "counts": self.last_scene.get("counts", {}),
        }

    def query_cb(self, msg: String) -> None:
        query = (msg.data or "").strip()
        if not query:
            return

        try:
            decision = build_route_decision(
                query,
                scene_context=self._scene_context(),
                source_topic=self.query_topic,
            )
            errors = validate_route_decision(decision)
            if errors:
                raise ValueError("; ".join(errors))
        except Exception as exc:
            self.get_logger().warning(
                f"failed to build route decision: {exc}"
            )
            return

        output = String()
        output.data = json.dumps(
            decision,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        self.decision_pub.publish(output)

        self.get_logger().info(
            "shadow route: "
            f"mode={decision['mode']}, "
            f"confidence={decision['confidence']:.2f}, "
            f"reason={decision['reason']}, "
            f"query={query!r}"
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = AssistantRouterNode()

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
