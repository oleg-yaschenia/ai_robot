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
        self.last_interpreted = {}
        self.last_raw_state = {}

        self.last_question = ""
        self.last_answer = ""
        self.last_focus_label = None
        self.last_focus_entity_id = None
        self.last_focus_kind = None

        self.image_sub = self.create_subscription(
            Image, self.image_topic, self.image_cb, 10
        )
        self.scene_summary_sub = self.create_subscription(
            String, "/scene/interpreted_summary", self.scene_summary_cb, 10
        )
        self.interpreted_sub = self.create_subscription(
            String, "/scene/interpreted_json", self.interpreted_cb, 10
        )
        self.raw_state_sub = self.create_subscription(
            String, "/perception/state_json", self.raw_state_cb, 10
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

    def interpreted_cb(self, msg: String):
        try:
            self.last_interpreted = json.loads(msg.data)
        except Exception as e:
            self.get_logger().warning(f"failed to parse interpreted_json: {e}")

    def raw_state_cb(self, msg: String):
        try:
            self.last_raw_state = json.loads(msg.data)
        except Exception as e:
            self.get_logger().warning(f"failed to parse raw perception state_json: {e}")

    def query_cb(self, msg: String):
        question = msg.data.strip()
        if not question:
            return

        answer = self.build_answer(question)
        self.publish_answer(answer)

        self.last_question = question
        self.last_answer = answer
        self.memory.add_interaction(self.mode, question, answer, "")

    # ---------- helpers ----------

    def qnorm(self, q: str) -> str:
        return q.strip().lower().replace("ё", "е")

    def counts(self):
        return self.last_interpreted.get("counts", self.last_raw_state.get("counts", {}))

    def persons(self):
        return self.last_interpreted.get("persons", [])

    def objects(self):
        return self.last_interpreted.get("objects", [])

    def salient(self):
        return self.last_interpreted.get("salient_entities", [])

    def changes(self):
        return self.last_interpreted.get("changes", {})

    def relations(self):
        return self.last_interpreted.get("relations", [])

    def primary_person(self):
        return self.last_interpreted.get("primary_person")

    def nearest_entity(self):
        return self.last_interpreted.get("nearest_entity")

    def person_context(self):
        return self.last_interpreted.get("person_context", {})

    def all_entities(self):
        return self.persons() + self.objects()

    def ru_label(self, label: str) -> str:
        return {
            "person": "человек",
            "cat": "кот",
            "dog": "собака",
            "cell phone": "телефон",
            "laptop": "ноутбук",
            "chair": "стул",
            "bottle": "бутылка",
            "cup": "чашка",
        }.get(label, label)

    def find_entity_by_id(self, entity_id: str):
        for ent in self.all_entities():
            if ent.get("entity_id") == entity_id:
                return ent
        return None

    def find_label_from_question(self, q: str):
        mapping = {
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
        for token, label in mapping.items():
            if token in q:
                return label
        return self.last_focus_label

    def entities_by_label(self, label: str):
        if not label:
            return []
        return [e for e in self.all_entities() if e.get("class_name") == label]

    def classify(self, q: str) -> str:
        if "что измен" in q:
            return "change"

        if "кто ближе" in q or "что ближе" in q or "кто ближе к камере" in q or "что ближе к камере" in q:
            return "nearest"

        if "рядом" in q or "возле" in q or "около" in q:
            return "nearby"

        if "слева" in q or "справа" in q:
            return "side_query"

        if "где" in q:
            return "location"

        if "сколько" in q:
            return "count"

        if "есть ли человек" in q or "есть человек" in q:
            return "person_presence"

        if "какие предметы" in q or "какие объекты" in q:
            return "object_list"

        if "что ты видишь" in q or "что видишь" in q or "что происходит" in q or "опиши сцену" in q:
            return "scene_overview"

        if "а где" in q or "где именно" in q or "а сколько" in q or "сколько их" in q or "а еще" in q or "у него" in q or "рядом с ним" in q:
            return "followup"

        return "scene_overview"

    # ---------- answer builders ----------

    def answer_scene_overview(self):
        summary = self.last_scene_summary or "У меня пока нет данных о сцене."
        if self.salient():
            self.last_focus_entity_id = self.salient()[0].get("entity_id")
            self.last_focus_label = self.salient()[0].get("class_name")
            self.last_focus_kind = "entity"
        return summary

    def answer_person_presence(self):
        count = int(self.counts().get("person", len(self.persons())))
        self.last_focus_label = "person"
        self.last_focus_kind = "person"
        if count > 0:
            primary = self.primary_person()
            if primary:
                return (
                    f"Да, я вижу человека. Количество: {count}. "
                    f"Главный человек находится {primary.get('position_text', 'в кадре')}."
                )
            return f"Да, я вижу человека. Количество: {count}."
        return "Нет, человека в кадре не видно."

    def answer_object_list(self):
        objs = self.objects()
        if not objs:
            return "Сейчас я не вижу уверенно распознанных предметов."

        names = []
        for obj in objs:
            label = obj.get("class_name")
            if label and label not in names:
                names.append(label)

        if names:
            self.last_focus_label = names[0]
            self.last_focus_kind = "object"

        return "Я вижу такие предметы: " + ", ".join(self.ru_label(x) for x in names[:6]) + "."

    def answer_count(self, q: str):
        label = self.find_label_from_question(q)
        if not label:
            return "Пока не понял, что именно нужно посчитать."

        count = int(self.counts().get(label, 0))
        self.last_focus_label = label
        self.last_focus_kind = "count"
        return f"Сейчас я вижу {count} объект(ов) типа {self.ru_label(label)}."

    def answer_location(self, q: str):
        label = self.find_label_from_question(q)
        if not label:
            return "Я не понял, положение какого объекта вас интересует."

        ents = self.entities_by_label(label)
        if not ents:
            return f"Сейчас я не вижу объект типа {self.ru_label(label)}."

        ent = ents[0]
        pos = ent.get("position_text", "в кадре")
        dist = ent.get("distance_hint", "")
        self.last_focus_label = label
        self.last_focus_entity_id = ent.get("entity_id")
        self.last_focus_kind = "entity"

        if dist:
            return f"{self.ru_label(label).capitalize()} находится {pos}, {dist}."
        return f"{self.ru_label(label).capitalize()} находится {pos}."

    def answer_change(self):
        ch = self.changes()
        if not ch or not ch.get("has_changes"):
            return "Существенных изменений по распознанным объектам я не вижу."

        pieces = []
        for item in ch.get("count_changes", [])[:4]:
            pieces.append(f"{self.ru_label(item['label'])}: {item['from']}→{item['to']}")

        if pieces:
            return "По сравнению с предыдущим состоянием изменилось следующее: " + ", ".join(pieces) + "."
        return "Я вижу изменения в сцене, но они небольшие."

    def answer_nearest(self, q: str):
        label = self.find_label_from_question(q)
        if label:
            ents = self.entities_by_label(label)
            if not ents:
                return f"Сейчас я не вижу объект типа {self.ru_label(label)}."
            ent = sorted(ents, key=lambda x: x.get("area", 0), reverse=True)[0]
        else:
            ent = self.nearest_entity()

        if not ent:
            return "Сейчас я не могу определить, что находится ближе всего к камере."

        self.last_focus_label = ent.get("class_name")
        self.last_focus_entity_id = ent.get("entity_id")
        self.last_focus_kind = "entity"

        return (
            f"Ближе всего к камере сейчас {self.ru_label(ent.get('class_name', 'объект'))}, "
            f"он находится {ent.get('position_text', 'в кадре')}."
        )

    def answer_side_query(self, q: str):
        side = "слева" if "слева" in q else "справа"
        label = self.find_label_from_question(q)

        ents = self.salient() or self.all_entities()
        if label:
            ents = [e for e in ents if e.get("class_name") == label]

        matched = [e for e in ents if side in e.get("position_text", "")]
        if not matched:
            if label:
                return f"Сейчас я не вижу объект типа {self.ru_label(label)} {side}."
            return f"Сейчас у меня нет уверенно распознанных объектов {side}."

        names = []
        for ent in matched[:4]:
            nm = self.ru_label(ent.get("class_name", "объект"))
            if nm not in names:
                names.append(nm)

        first = matched[0]
        self.last_focus_label = first.get("class_name")
        self.last_focus_entity_id = first.get("entity_id")
        self.last_focus_kind = "entity"

        return f"{side.capitalize()} я вижу: " + ", ".join(names) + "."

    def answer_nearby(self, q: str):
        # explicit person-related question
        if "человек" in q or "у него" in q or "рядом с ним" in q:
            ctx = self.person_context()
            nearby = ctx.get("nearby_objects", [])
            if not nearby:
                return "Рядом с человеком я не вижу уверенно распознанных предметов."

            names = []
            for obj in nearby:
                nm = self.ru_label(obj.get("class_name", "объект"))
                if nm not in names:
                    names.append(nm)

            first = nearby[0]
            self.last_focus_label = first.get("class_name")
            self.last_focus_entity_id = first.get("entity_id")
            self.last_focus_kind = "entity"

            return "Рядом с человеком видно: " + ", ".join(names[:5]) + "."

        label = self.find_label_from_question(q)
        if not label:
            return "Я не понял, рядом с чем именно нужно посмотреть."

        ents = self.entities_by_label(label)
        if not ents:
            return f"Сейчас я не вижу объект типа {self.ru_label(label)}."

        anchor = ents[0]
        anchor_id = anchor.get("entity_id")

        near_rel = [r for r in self.relations() if r.get("subject_id") == anchor_id and r.get("relation") == "near"]
        if not near_rel:
            return f"Рядом с {self.ru_label(label)} я не вижу заметных объектов."

        nearby_names = []
        for rel in near_rel:
            other = self.find_entity_by_id(rel.get("object_id"))
            if other:
                nm = self.ru_label(other.get("class_name", "объект"))
                if nm not in nearby_names:
                    nearby_names.append(nm)

        self.last_focus_label = label
        self.last_focus_entity_id = anchor_id
        self.last_focus_kind = "entity"

        if not nearby_names:
            return f"Рядом с {self.ru_label(label)} я не вижу заметных объектов."

        return f"Рядом с {self.ru_label(label)} находится: " + ", ".join(nearby_names[:5]) + "."

    def answer_followup(self, q: str):
        if "у него" in q or "рядом с ним" in q:
            return self.answer_nearby("рядом с человеком")

        if "где" in q:
            return self.answer_location(q)

        if "сколько" in q:
            return self.answer_count(q)

        if "еще" in q:
            return self.answer_object_list()

        return self.answer_scene_overview()

    def build_answer(self, question: str) -> str:
        q = self.qnorm(question)
        intent = self.classify(q)

        if intent == "scene_overview":
            return self.answer_scene_overview()
        if intent == "person_presence":
            return self.answer_person_presence()
        if intent == "object_list":
            return self.answer_object_list()
        if intent == "count":
            return self.answer_count(q)
        if intent == "location":
            return self.answer_location(q)
        if intent == "change":
            return self.answer_change()
        if intent == "nearest":
            return self.answer_nearest(q)
        if intent == "side_query":
            return self.answer_side_query(q)
        if intent == "nearby":
            return self.answer_nearby(q)
        if intent == "followup":
            return self.answer_followup(q)

        return self.answer_scene_overview()

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
