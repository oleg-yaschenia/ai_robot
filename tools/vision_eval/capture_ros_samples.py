#!/usr/bin/env python3
"""Capture synchronized stereo samples plus current perception metadata.

Run after sourcing ROS 2 and the workspace. The script is intentionally kept
outside a ROS package so it can be used as an evaluation utility.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String


def stamp_ns(msg: Image) -> int:
    return int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def safe_json_loads(value: Optional[str]) -> Optional[dict[str, Any]]:
    if not value:
        return None
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    except json.JSONDecodeError:
        return {"raw": value, "parse_error": True}


@dataclass
class LatestData:
    left: Optional[Image] = None
    right: Optional[Image] = None
    perception_raw: Optional[str] = None
    scene_raw: Optional[str] = None


class VisionSampleCollector(Node):
    def __init__(
        self,
        left_topic: str,
        right_topic: str,
        perception_topic: str,
        scene_topic: str,
    ) -> None:
        super().__init__("vision_eval_collector")
        self.bridge = CvBridge()
        self.data = LatestData()

        self.create_subscription(Image, left_topic, self._left_cb, 10)
        self.create_subscription(Image, right_topic, self._right_cb, 10)
        self.create_subscription(String, perception_topic, self._perception_cb, 10)
        self.create_subscription(String, scene_topic, self._scene_cb, 10)

        self.get_logger().info(
            "Vision evaluator subscribed: "
            f"left={left_topic}, right={right_topic}, "
            f"perception={perception_topic}, scene={scene_topic}"
        )

    def _left_cb(self, msg: Image) -> None:
        self.data.left = msg

    def _right_cb(self, msg: Image) -> None:
        self.data.right = msg

    def _perception_cb(self, msg: String) -> None:
        self.data.perception_raw = msg.data

    def _scene_cb(self, msg: String) -> None:
        self.data.scene_raw = msg.data

    def wait_for_initial_data(self, timeout_sec: float) -> None:
        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.data.left is not None and self.data.right is not None:
                return
        raise TimeoutError(
            "No stereo images received. Check /camera/left/image_raw and "
            "/camera/right/image_raw."
        )

    def wait_for_new_pair(self, last_left_stamp: int, timeout_sec: float) -> None:
        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            left = self.data.left
            right = self.data.right
            if left is None or right is None:
                continue
            if stamp_ns(left) == last_left_stamp:
                continue
            # The current camera node publishes left/right with the same stamp.
            if stamp_ns(left) == stamp_ns(right):
                return
        raise TimeoutError("Timed out waiting for a new synchronized stereo pair.")


def image_to_array(bridge: CvBridge, msg: Image):
    if msg.encoding == "mono8":
        return bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
    return bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")


def save_sample(
    collector: VisionSampleCollector,
    output_root: Path,
    scenario: str,
    expected_people: int,
    expected_objects: list[str],
    expected_pets: list[str],
    lighting: str,
    distance: str,
    notes: str,
    sequence_index: int,
) -> tuple[str, int]:
    left_msg = collector.data.left
    right_msg = collector.data.right
    if left_msg is None or right_msg is None:
        raise RuntimeError("Stereo messages are not available.")

    now = datetime.now(timezone.utc)
    sample_id = (
        now.strftime("%Y%m%dT%H%M%S.%fZ")
        + f"_{sequence_index:03d}_{uuid.uuid4().hex[:6]}"
    )

    left_dir = output_root / "images" / "left"
    right_dir = output_root / "images" / "right"
    left_dir.mkdir(parents=True, exist_ok=True)
    right_dir.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    left_path = left_dir / f"{sample_id}.png"
    right_path = right_dir / f"{sample_id}.png"

    left_frame = image_to_array(collector.bridge, left_msg)
    right_frame = image_to_array(collector.bridge, right_msg)

    if not cv2.imwrite(str(left_path), left_frame):
        raise RuntimeError(f"Failed to save {left_path}")
    if not cv2.imwrite(str(right_path), right_frame):
        raise RuntimeError(f"Failed to save {right_path}")

    left_stamp = stamp_ns(left_msg)
    right_stamp = stamp_ns(right_msg)
    perception = safe_json_loads(collector.data.perception_raw)
    scene = safe_json_loads(collector.data.scene_raw)

    record = {
        "schema_version": 1,
        "sample_id": sample_id,
        "captured_at_utc": now.isoformat(),
        "scenario": scenario,
        "ground_truth": {
            "expected_people": expected_people,
            "expected_objects": expected_objects,
            "expected_pets": expected_pets,
            "lighting": lighting,
            "distance": distance,
            "notes": notes,
        },
        "sensor": {
            "left_topic": "/camera/left/image_raw",
            "right_topic": "/camera/right/image_raw",
            "left_stamp_ns": left_stamp,
            "right_stamp_ns": right_stamp,
            "synchronized": left_stamp == right_stamp,
            "left_encoding": left_msg.encoding,
            "right_encoding": right_msg.encoding,
            "width": int(left_msg.width),
            "height": int(left_msg.height),
        },
        "files": {
            "left": str(left_path.relative_to(output_root)),
            "right": str(right_path.relative_to(output_root)),
        },
        "baseline": {
            "perception_state": perception,
            "scene_interpretation": scene,
        },
    }

    manifest_path = output_root / "manifest.jsonl"
    with manifest_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    return sample_id, left_stamp


def build_parser() -> argparse.ArgumentParser:
    default_root = Path(
        os.environ.get("AI_ROBOT_ROOT", str(Path.home() / "ai_robot"))
    ) / "data" / "vision_eval"

    parser = argparse.ArgumentParser(
        description="Capture stereo frames and current perception output for evaluation."
    )
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--people", type=int, default=0)
    parser.add_argument("--objects", default="", help="Comma-separated canonical names")
    parser.add_argument("--pets", default="", help="Comma-separated pet labels/names")
    parser.add_argument(
        "--lighting", choices=("dark", "normal", "bright", "mixed"), default="normal"
    )
    parser.add_argument(
        "--distance", choices=("near", "medium", "far", "mixed"), default="medium"
    )
    parser.add_argument("--notes", default="")
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--interval", type=float, default=0.8)
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--output", type=Path, default=default_root)
    parser.add_argument("--left-topic", default="/camera/left/image_raw")
    parser.add_argument("--right-topic", default="/camera/right/image_raw")
    parser.add_argument("--perception-topic", default="/perception/state_json")
    parser.add_argument("--scene-topic", default="/scene/interpreted_json")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.count < 1:
        raise SystemExit("--count must be at least 1")
    if args.people < 0:
        raise SystemExit("--people cannot be negative")

    rclpy.init()
    node = VisionSampleCollector(
        args.left_topic,
        args.right_topic,
        args.perception_topic,
        args.scene_topic,
    )

    try:
        node.wait_for_initial_data(args.timeout)
        last_stamp = -1
        for index in range(1, args.count + 1):
            if last_stamp >= 0:
                node.wait_for_new_pair(last_stamp, args.timeout)
            else:
                rclpy.spin_once(node, timeout_sec=0.2)

            sample_id, last_stamp = save_sample(
                collector=node,
                output_root=args.output.expanduser().resolve(),
                scenario=args.scenario,
                expected_people=args.people,
                expected_objects=parse_csv(args.objects),
                expected_pets=parse_csv(args.pets),
                lighting=args.lighting,
                distance=args.distance,
                notes=args.notes,
                sequence_index=index,
            )
            print(f"[{index}/{args.count}] saved {sample_id}")

            if index < args.count:
                end = time.monotonic() + max(0.0, args.interval)
                while rclpy.ok() and time.monotonic() < end:
                    rclpy.spin_once(node, timeout_sec=0.05)

        print(f"Dataset: {args.output.expanduser().resolve()}")
        return 0
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
