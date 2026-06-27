#!/usr/bin/env python3
import json
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from robot_vision_assistant.memory_store import MemoryStore


class VisionAssistantNode(Node):
    def __init__(self):
        super().__init__("vision_assistant_node")

        self.declare_parameter("image_topic", "/camera/left/image_raw")
        self.declare_parameter("mode", "local_only")
        self.declare_parameter("allow_cloud", False)
        self.declare_parameter("allow_realtime", False)
        self.declare_parameter("snapshots_dir", "/home/warxen/ai_robot/data/vision_assistant/snapshots")
        self.declare_parameter("db_path", "/home/warxen/ai_robot/data/vision_assistant/assistant_memory.sqlite")

        self.image_topic = str(self.get_parameter("image_topic").value)
        self.mode = str(self.get_parameter("mode").value)
        self.allow_cloud = bool(self.get_parameter("allow_cloud").value)
        self.allow_realtime = bool(self.get_parameter("allow_realtime").value)
        self.snapshots_dir = Path(str(self.get_parameter("snapshots_dir").value))
        self.db_path = str(self.get_parameter("db_path").value)

        self.snapshots_dir.mkdir(parents=True, exist_ok=True)

        self.bridge = CvBridge()
        self.memory = MemoryStore(self.db_path)

        self.last_frame_bgr = None
        self.last_scene_summary = "У меня пока нет данных о сцене."
        self.last_state = {}

        self.image_sub = self.create_subscription(
            Image, self.image_topic, self.image_cb, 10
        )

        self.scene_summary_sub = self.create_subscription(
            String, "/perception/scene_summary", self.scene_summary_cb, 10
        )

        self.state_sub = self.create_subscription(
            String, "/perception/state_json", self.state_json_cb, 10
        )

        self.query_sub = self.create_subscription(
            String, "/vision_assistant/query", self.query_cb, 10
        )

        self.mode_sub = self.create_subscription(
            String, "/vision_assistant/set_mode", self.mode_cb, 10
        )

        self.answer_pub = self.create_publisher(String, "/vision_assistant/answer", 10)
        self.status_pub = self.create_publisher(String, "/vision_assistant/status", 10)

        self.publish_status(f"vision_assistant started in mode={self.mode}, image_topic={self.image_topic}")
        self.get_logger().info(f"vision_assistant_node started: mode={self.mode}")

    def publish_status(self, text: str):
        msg = String()
        msg.data = text
        self.status_pub.publish(msg)

    def publish_answer(self, text: str):
        msg = String()
        msg.data = text
        self.answer_pub.publish(msg)

    def mode_cb(self, msg: String):
        new_mode = msg.data.strip()
        if new_mode not in ("local_only", "hybrid", "realtime"):
            self.publish_status(f"invalid mode: {new_mode}")
            return

        if new_mode == "realtime" and not self.allow_realtime:
            self.publish_status("realtime mode requested but allow_realtime=false")
            return

        self.mode = new_mode
        self.memory.set_state("mode", self.mode)
        self.publish_status(f"mode changed to {self.mode}")
        self.get_logger().info(f"mode changed to {self.mode}")

    def image_cb(self, msg: Image):
        try:
            if msg.encoding == "bgr8":
                frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            elif msg.encoding == "mono8":
                gray = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
                frame = self.bridge.cv2_to_imgmsg(gray, encoding="mono8")
                return
            else:
                frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

            self.last_frame_bgr = frame
        except Exception as e:
            self.get_logger().warning(f"image conversion failed: {e}")

    def scene_summary_cb(self, msg: String):
        self.last_scene_summary = msg.data.strip() if msg.data else "Нет данных сцены."

    def state_json_cb(self, msg: String):
        try:
            self.last_state = json.loads(msg.data)
        except Exception as e:
            self.get_logger().warning(f"failed to parse state_json: {e}")

    def query_cb(self, msg: String):
        question = msg.data.strip()
        if not question:
            return

        answer = self.build_answer_from_perception(question)

        self.publish_answer(answer)
        self.memory.add_interaction(self.mode, question, answer, "")

    def build_answer_from_perception(self, question: str) -> str:
        q = question.lower()

        scene_flags = self.last_state.get("scene_flags", {})
        counts = self.last_state.get("counts", {})
        objects = self.last_state.get("objects", [])
        persons = self.last_state.get("persons", [])

        person_present = bool(scene_flags.get("person_present", False))
        pet_present = bool(scene_flags.get("pet_present", False))

        person_count = int(counts.get("person", len(persons)))

        object_names = []
        for obj in objects[:8]:
            name = obj.get("class_name")
            if name:
                object_names.append(name)

        if "что ты видишь" in q or "что видишь" in q:
            return self.last_scene_summary

        if "есть ли человек" in q or "кто-то есть" in q:
            if person_present:
                return f"Да, я вижу человека. Количество: {person_count}."
            return "Нет, сейчас человека в кадре не видно."

        if "есть ли кот" in q:
            if counts.get("cat", 0) > 0:
                return "Да, я вижу кота."
            if pet_present:
                return "Я вижу питомца, но не уверен, что это кот."
            return "Нет, кота в кадре не видно."

        if "есть ли собака" in q:
            if counts.get("dog", 0) > 0:
                return "Да, я вижу собаку."
            if pet_present:
                return "Я вижу питомца, но не уверен, что это собака."
            return "Нет, собаки в кадре не видно."

        if "что находится" in q or "какие предметы" in q:
            if object_names:
                return "Я вижу такие объекты: " + ", ".join(object_names) + "."
            return "Сейчас я не вижу уверенно распознанных объектов."

        if "питом" in q:
            if pet_present:
                return "Да, в кадре есть питомец."
            return "Нет, питомца в кадре не видно."

        return f"По текущему локальному анализу: {self.last_scene_summary}"

    def destroy_node(self):
        try:
            self.memory.close()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = VisionAssistantNode()
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
