#!/usr/bin/env python3
"""ROS adapter from legacy perception state to EntityArray v1."""

from __future__ import annotations

import json
from typing import Any, Dict

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from robot_vision_assistant.perception_contract import (
    build_entity_array,
    validate_entity_array,
)


class PerceptionEntityAdapterNode(Node):
    def __init__(self) -> None:
        super().__init__("perception_entity_adapter_node")

        self.declare_parameter(
            "input_topic",
            "/perception/state_json",
        )
        self.declare_parameter(
            "output_topic",
            "/perception/entities_json",
        )

        self.input_topic = str(
            self.get_parameter("input_topic").value
        )
        self.output_topic = str(
            self.get_parameter("output_topic").value
        )

        self.publisher = self.create_publisher(
            String,
            self.output_topic,
            10,
        )
        self.subscription = self.create_subscription(
            String,
            self.input_topic,
            self.on_state,
            10,
        )

        self.received_messages = 0
        self.published_messages = 0
        self.invalid_messages = 0

        self.get_logger().info(
            "perception_entity_adapter_node started: "
            f"input={self.input_topic}, "
            f"output={self.output_topic}, "
            "schema=robot_perception_entities/v1"
        )

    def on_state(self, msg: String) -> None:
        self.received_messages += 1

        try:
            state: Dict[str, Any] = json.loads(msg.data)
            entity_array = build_entity_array(
                state,
                source_topic=self.input_topic,
            )
            errors = validate_entity_array(entity_array)
            if errors:
                self.invalid_messages += 1
                self.get_logger().warning(
                    "EntityArray validation failed: "
                    + "; ".join(errors[:5])
                )
                return
        except Exception as exc:
            self.invalid_messages += 1
            self.get_logger().warning(
                f"Failed to convert perception state: {exc}"
            )
            return

        output = String()
        output.data = json.dumps(
            entity_array,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        self.publisher.publish(output)
        self.published_messages += 1


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PerceptionEntityAdapterNode()

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
