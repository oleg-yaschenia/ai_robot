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
        self.declare_parameter(
            "snapshots_dir",
            "/home/warxen/ai_robot/data/vision_assistant/snapshots",
        )
        self.declare_parameter(
            "db_path",
            "/home/warxen/ai_robot/data/vision_assistant/assistant_memory.sqlite",
        )

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
        self.prev_state = {}

        self.last_question = ""
        self.last_answer = ""
        self.last_focus_label = None
        self.last_focus_count = None
        self.last_intent = None

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

        self.publish_status(
            f"vision_assistant started in mode={self.mode}, image_topic={self.image_topic}"
        )
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
                import cv2
                frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            else:
                frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.last_frame_bgr = frame
        except Exception as e:
            self.get_logger().warning(f"image conversion failed: {e}")

    def scene_summary_cb(self, msg: String):
        self.last_scene_summary = msg.data.strip() if msg.data else "Нет данных сцены."

    def state_json_cb(self, msg: String):
        try:
            new_state = json.loads(msg.data)
            self.prev_state = self.last_state
            self.last_state = new_state
        except Exception as e:
            self.get_logger().warning(f"failed to parse state_json: {e}")

    def query_cb(self, msg: String):
        question = msg.data.strip()
        if not question:
            return

        answer = self.build_answer_from_perception(question)
        self.publish_answer(answer)

        self.last_question = question
        self.last_answer = answer

        self.memory.add_interaction(self.mode, question, answer, "")

    # ---------- helpers ----------

    def get_counts(self):
        return self.last_state.get("counts", {}) if isinstance(self.last_state, dict) else {}

    def get_objects(self):
        return self.last_state.get("objects", []) if isinstance(self.last_state, dict) else []

    def get_persons(self):
        return self.last_state.get("persons", []) if isinstance(self.last_state, dict) else []

    def get_scene_flags(self):
        return self.last_state.get("scene_flags", {}) if isinstance(self.last_state, dict) else {}

    def get_image_size(self):
        image_size = self.last_state.get("image_size", {}) if isinstance(self.last_state, dict) else {}
        w = int(image_size.get("width", 0) or 0)
        h = int(image_size.get("height", 0) or 0)
        return w, h

    def normalize_question(self, question: str) -> str:
        q = question.strip().lower()
        q = q.replace("ё", "е")
        return q

    def is_followup_question(self, q: str) -> bool:
        followups = [
            "а где",
            "а где именно",
            "где именно",
            "а сколько",
            "сколько их",
            "а еще",
            "еще",
            "подробнее",
            "что еще",
            "а что еще",
            "кто именно",
            "какой именно",
        ]
        return any(token in q for token in followups)

    def classify_intent(self, q: str) -> str:
        if self.is_followup_question(q):
            return "followup"

        if "что измен" in q:
            return "change"

        if "где" in q or "слева" in q or "справа" in q or "в центре" in q:
            return "location"

        if "сколько" in q:
            return "count"

        if "есть ли человек" in q or "кто-то есть" in q or "есть человек" in q:
            return "person_presence"

        if "есть ли кот" in q or "есть ли кошк" in q:
            return "cat_presence"

        if "есть ли собак" in q:
            return "dog_presence"

        if "питом" in q:
            return "pet_presence"

        if "какие предметы" in q or "что находится" in q or "что за предмет" in q:
            return "object_list"

        if "что ты видишь" in q or "что видишь" in q or "что происходит" in q:
            return "scene_overview"

        return "generic"

    def object_label_from_question(self, q: str):
        label_map = {
            "человек": "person",
            "люд": "person",
            "кот": "cat",
            "кошк": "cat",
            "собак": "dog",
            "телефон": "cell phone",
            "смартфон": "cell phone",
            "ноутбук": "laptop",
            "стул": "chair",
            "бутыл": "bottle",
            "чаш": "cup",
            "круж": "cup",
        }
        for token, label in label_map.items():
            if token in q:
                return label
        return None

    def ru_label(self, label: str, count: int = 1) -> str:
        mapping = {
            "person": "человек" if count == 1 else "человек",
            "cat": "кот" if count == 1 else "коты",
            "dog": "собака" if count == 1 else "собаки",
            "cell phone": "телефон" if count == 1 else "телефоны",
            "laptop": "ноутбук" if count == 1 else "ноутбуки",
            "chair": "стул" if count == 1 else "стулья",
            "bottle": "бутылка" if count == 1 else "бутылки",
            "cup": "чашка" if count == 1 else "чашки",
        }
        return mapping.get(label, label)

    def all_detected_labels(self):
        counts = self.get_counts()
        labels = []
        for label, cnt in counts.items():
            if cnt > 0:
                labels.append(label)
        return labels

    def get_entities_by_label(self, label: str):
        if label == "person":
            return self.get_persons()

        return [obj for obj in self.get_objects() if obj.get("class_name") == label]

    def choose_focus_label(self, q: str):
        explicit = self.object_label_from_question(q)
        if explicit:
            return explicit

        if self.last_focus_label:
            return self.last_focus_label

        labels = self.all_detected_labels()
        if labels:
            return labels[0]

        return None

    def center_to_text(self, cx: int, cy: int):
        w, h = self.get_image_size()
        if w <= 0 or h <= 0:
            return "в кадре"

        x_zone = cx / w
        y_zone = cy / h

        if x_zone < 0.33:
            x_text = "слева"
        elif x_zone > 0.66:
            x_text = "справа"
        else:
            x_text = "по центру"

        if y_zone < 0.33:
            y_text = "вверху"
        elif y_zone > 0.66:
            y_text = "внизу"
        else:
            y_text = "по вертикали ближе к центру"

        if x_text == "по центру":
            return y_text if y_text != "по вертикали ближе к центру" else "по центру"
        if y_text == "по вертикали ближе к центру":
            return x_text
        return f"{x_text} {y_text}"

    def size_to_distance_text(self, size_wh):
        try:
            bw, bh = size_wh
            area = bw * bh
            if area > 160000:
                return "довольно близко"
            if area > 60000:
                return "на среднем расстоянии"
            return "дальше от камеры"
        except Exception:
            return ""

    def build_scene_overview(self):
        persons = self.get_persons()
        objects = self.get_objects()
        counts = self.get_counts()

        person_count = len(persons)
        object_labels = []
        for obj in objects:
            name = obj.get("class_name")
            if name and name not in object_labels:
                object_labels.append(name)

        parts = []

        if person_count > 0:
            parts.append(f"Я вижу {person_count} человек.")
        else:
            parts.append("Сейчас людей в кадре нет.")

        if object_labels:
            ru_names = [self.ru_label(name) for name in object_labels[:5]]
            parts.append("Также вижу: " + ", ".join(ru_names) + ".")
        else:
            parts.append("Других уверенно распознанных предметов сейчас немного или нет.")

        return " ".join(parts)

    def answer_person_presence(self):
        person_present = bool(self.get_scene_flags().get("person_present", False))
        person_count = int(self.get_counts().get("person", len(self.get_persons())))

        self.last_focus_label = "person"
        self.last_focus_count = person_count
        self.last_intent = "person_presence"

        if person_present:
            return f"Да, я вижу человека. Количество: {person_count}."
        return "Нет, сейчас человека в кадре не видно."

    def answer_cat_presence(self):
        cat_count = int(self.get_counts().get("cat", 0))
        self.last_focus_label = "cat"
        self.last_focus_count = cat_count
        self.last_intent = "cat_presence"

        if cat_count > 0:
            return f"Да, я вижу кота. Количество: {cat_count}."
        return "Нет, кота в кадре не видно."

    def answer_dog_presence(self):
        dog_count = int(self.get_counts().get("dog", 0))
        self.last_focus_label = "dog"
        self.last_focus_count = dog_count
        self.last_intent = "dog_presence"

        if dog_count > 0:
            return f"Да, я вижу собаку. Количество: {dog_count}."
        return "Нет, собаки в кадре не видно."

    def answer_pet_presence(self):
        pet_present = bool(self.get_scene_flags().get("pet_present", False))
        self.last_focus_label = "cat" if self.get_counts().get("cat", 0) > 0 else "dog"
        self.last_intent = "pet_presence"

        if pet_present:
            cat_count = int(self.get_counts().get("cat", 0))
            dog_count = int(self.get_counts().get("dog", 0))
            if cat_count > 0 and dog_count > 0:
                return f"Да, в кадре есть питомцы: котов {cat_count}, собак {dog_count}."
            if cat_count > 0:
                return f"Да, я вижу питомца — похоже, это кот. Количество: {cat_count}."
            if dog_count > 0:
                return f"Да, я вижу питомца — похоже, это собака. Количество: {dog_count}."
            return "Да, в кадре есть питомец."
        return "Нет, питомца в кадре не видно."

    def answer_object_list(self):
        objects = self.get_objects()
        if not objects:
            return "Сейчас я не вижу уверенно распознанных предметов."

        seen = []
        for obj in objects:
            label = obj.get("class_name")
            if label and label not in seen:
                seen.append(label)

        self.last_focus_label = seen[0] if seen else None
        self.last_intent = "object_list"

        ru_names = [self.ru_label(label) for label in seen[:6]]
        return "Я вижу такие предметы: " + ", ".join(ru_names) + "."

    def answer_count(self, q: str):
        label = self.choose_focus_label(q)
        if not label:
            return "Пока не понял, что именно нужно посчитать."

        count = int(self.get_counts().get(label, 0))
        self.last_focus_label = label
        self.last_focus_count = count
        self.last_intent = "count"

        ru_name = self.ru_label(label, count)
        return f"Сейчас я вижу {count} объект(ов) типа {ru_name}."

    def answer_location(self, q: str):
        label = self.choose_focus_label(q)
        if not label:
            return "Я не понял, положение какого объекта вас интересует."

        entities = self.get_entities_by_label(label)
        if not entities:
            return f"Сейчас я не вижу объект типа {self.ru_label(label)}."

        target = entities[0]
        center_xy = target.get("center_xy", [0, 0])
        size_wh = target.get("size_wh", [0, 0])

        position_text = self.center_to_text(int(center_xy[0]), int(center_xy[1]))
        distance_text = self.size_to_distance_text(size_wh)

        self.last_focus_label = label
        self.last_focus_count = len(entities)
        self.last_intent = "location"

        ru_name = self.ru_label(label, len(entities))

        if len(entities) == 1:
            if distance_text:
                return f"{ru_name.capitalize()} находится {position_text}, {distance_text}."
            return f"{ru_name.capitalize()} находится {position_text}."

        return f"Объектов типа {ru_name} несколько. Первый видимый находится {position_text}."

    def answer_change(self):
        if not self.prev_state:
            return "У меня пока недостаточно предыдущих данных, чтобы уверенно сказать, что изменилось."

        prev_counts = self.prev_state.get("counts", {})
        curr_counts = self.get_counts()

        changes = []

        all_labels = sorted(set(prev_counts.keys()) | set(curr_counts.keys()))
        for label in all_labels:
            prev_val = int(prev_counts.get(label, 0))
            curr_val = int(curr_counts.get(label, 0))
            if prev_val != curr_val:
                if curr_val > prev_val:
                    changes.append(f"{self.ru_label(label)}: стало {curr_val}")
                else:
                    changes.append(f"{self.ru_label(label)}: стало {curr_val}")

        if changes:
            return "По сравнению с предыдущим состоянием изменилось следующее: " + "; ".join(changes) + "."

        if self.last_scene_summary:
            return "Существенных изменений по распознанным объектам не вижу."
        return "Я не вижу заметных изменений."

    def answer_followup(self, q: str):
        if ("где" in q or "где именно" in q) and self.last_focus_label:
            return self.answer_location(q)

        if ("сколько" in q or "сколько их" in q) and self.last_focus_label:
            return self.answer_count(q)

        if "еще" in q or "что еще" in q:
            return self.answer_object_list()

        if self.last_answer:
            return f"Уточню по текущей сцене: {self.last_answer}"

        return self.build_scene_overview()

    def build_answer_from_perception(self, question: str) -> str:
        q = self.normalize_question(question)
        intent = self.classify_intent(q)

        if intent == "scene_overview":
            answer = self.build_scene_overview()
            self.last_intent = intent
            return answer

        if intent == "person_presence":
            return self.answer_person_presence()

        if intent == "cat_presence":
            return self.answer_cat_presence()

        if intent == "dog_presence":
            return self.answer_dog_presence()

        if intent == "pet_presence":
            return self.answer_pet_presence()

        if intent == "object_list":
            return self.answer_object_list()

        if intent == "count":
            return self.answer_count(q)

        if intent == "location":
            return self.answer_location(q)

        if intent == "change":
            return self.answer_change()

        if intent == "followup":
            return self.answer_followup(q)

        # generic fallback
        answer = self.build_scene_overview()
        self.last_intent = "generic"
        return answer

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
