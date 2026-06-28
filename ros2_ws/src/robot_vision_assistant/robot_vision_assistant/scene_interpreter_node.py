#!/usr/bin/env python3
import json
import math
import time
from typing import Any, Dict, List, Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class SceneInterpreterNode(Node):
    def __init__(self):
        super().__init__("scene_interpreter_node")

        self.declare_parameter("input_topic", "/perception/state_json")
        self.declare_parameter("output_topic", "/scene/interpreted_json")
        self.declare_parameter("summary_topic", "/scene/interpreted_summary")

        self.input_topic = str(self.get_parameter("input_topic").value)
        self.output_topic = str(self.get_parameter("output_topic").value)
        self.summary_topic = str(self.get_parameter("summary_topic").value)

        self.sub = self.create_subscription(String, self.input_topic, self.on_state, 10)
        self.pub = self.create_publisher(String, self.output_topic, 10)
        self.summary_pub = self.create_publisher(String, self.summary_topic, 10)

        self.prev_state: Optional[Dict[str, Any]] = None

        self.get_logger().info(
            f"scene_interpreter_node started: input={self.input_topic}, output={self.output_topic}"
        )

    def on_state(self, msg: String):
        try:
            raw = json.loads(msg.data)
        except Exception as e:
            self.get_logger().warning(f"failed to parse perception state: {e}")
            return

        interpreted = self.interpret(raw, self.prev_state)
        self.prev_state = raw

        out = String()
        out.data = json.dumps(interpreted, ensure_ascii=False)
        self.pub.publish(out)

        sm = String()
        sm.data = interpreted.get("human_summary", "")
        self.summary_pub.publish(sm)

    def interpret(self, state: Dict[str, Any], prev_state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        persons = state.get("persons", [])
        objects = state.get("objects", [])
        counts = state.get("counts", {})
        scene_flags = state.get("scene_flags", {})
        image_size = state.get("image_size", {})

        width = int(image_size.get("width", 0) or 0)
        height = int(image_size.get("height", 0) or 0)

        enriched_persons = []
        for i, p in enumerate(persons):
            enriched_persons.append(self.enrich_entity(p, "person", f"person_{i}", width, height))

        enriched_objects = []
        for i, o in enumerate(objects):
            label = o.get("class_name", "object")
            enriched_objects.append(self.enrich_entity(o, label, f"{label}_{i}", width, height))

        all_entities = enriched_persons + enriched_objects
        salient_entities = sorted(
            all_entities, key=lambda x: x.get("importance_score", 0.0), reverse=True
        )[:6]

        relations = self.build_relations(salient_entities, width, height)
        primary_person = self.select_primary_person(enriched_persons)
        nearest_entity = self.select_nearest_entity(all_entities)

        person_context = self.build_person_context(primary_person, enriched_objects, width, height)
        changes = self.detect_changes(state, prev_state)

        human_summary = self.build_human_summary(
            enriched_persons=enriched_persons,
            enriched_objects=enriched_objects,
            salient_entities=salient_entities,
            primary_person=primary_person,
            nearest_entity=nearest_entity,
            person_context=person_context,
            changes=changes,
            counts=counts,
            scene_flags=scene_flags,
        )

        return {
            "timestamp": time.time(),
            "source_timestamp": state.get("timestamp"),
            "counts": counts,
            "scene_flags": scene_flags,
            "image_size": image_size,
            "persons": enriched_persons,
            "objects": enriched_objects,
            "salient_entities": salient_entities,
            "relations": relations,
            "primary_person": primary_person,
            "nearest_entity": nearest_entity,
            "person_context": person_context,
            "changes": changes,
            "human_summary": human_summary,
        }

    def enrich_entity(
        self,
        ent: Dict[str, Any],
        label: str,
        entity_id: str,
        width: int,
        height: int,
    ) -> Dict[str, Any]:
        out = dict(ent)

        bbox = ent.get("bbox_xyxy", [0, 0, 0, 0])
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        bw = max(0, x2 - x1)
        bh = max(0, y2 - y1)
        area = bw * bh

        out["entity_id"] = entity_id
        out["class_name"] = label
        out["center_xy"] = [cx, cy]
        out["size_wh"] = [bw, bh]
        out["area"] = area
        out["position_text"] = self.position_text(cx, cy, width, height)
        out["distance_hint"] = self.distance_hint(area, width, height)
        out["importance_score"] = self.importance_score(label, area, cx, width)

        return out

    def position_text(self, cx: int, cy: int, width: int, height: int) -> str:
        if width <= 0 or height <= 0:
            return "в кадре"

        xr = cx / width
        yr = cy / height

        if xr < 0.33:
            x_text = "слева"
        elif xr > 0.66:
            x_text = "справа"
        else:
            x_text = "по центру"

        if yr < 0.33:
            y_text = "вверху"
        elif yr > 0.66:
            y_text = "внизу"
        else:
            y_text = "по вертикали ближе к центру"

        if x_text == "по центру" and y_text == "по вертикали ближе к центру":
            return "по центру"
        if y_text == "по вертикали ближе к центру":
            return x_text
        if x_text == "по центру":
            return y_text
        return f"{x_text} {y_text}"

    def distance_hint(self, area: int, width: int, height: int) -> str:
        if width <= 0 or height <= 0:
            return ""

        frame_area = width * height
        ratio = area / max(frame_area, 1)

        if ratio > 0.18:
            return "очень близко"
        if ratio > 0.07:
            return "довольно близко"
        if ratio > 0.025:
            return "на среднем расстоянии"
        return "дальше от камеры"

    def importance_score(self, label: str, area: int, cx: int, width: int) -> float:
        label_bonus = {
            "person": 3.0,
            "cat": 2.7,
            "dog": 2.7,
            "cell phone": 2.1,
            "laptop": 1.9,
            "cup": 1.4,
            "bottle": 1.4,
            "chair": 0.9,
        }.get(label, 1.0)

        center_bonus = 1.0
        if width > 0:
            center_dist = abs((cx / width) - 0.5)
            center_bonus = 1.2 - min(center_dist, 0.5)

        return label_bonus * (1.0 + math.log(max(area, 1), 10) / 10.0) * center_bonus

    def build_relations(self, entities: List[Dict[str, Any]], width: int, height: int) -> List[Dict[str, Any]]:
        relations = []
        if width <= 0 or height <= 0:
            return relations

        x_thr = width * 0.18
        y_thr = height * 0.18
        near_thr = math.sqrt(width * width + height * height) * 0.22

        for i in range(len(entities)):
            a = entities[i]
            ax, ay = a.get("center_xy", [0, 0])

            for j in range(i + 1, len(entities)):
                b = entities[j]
                bx, by = b.get("center_xy", [0, 0])

                dx = bx - ax
                dy = by - ay
                dist = math.sqrt(dx * dx + dy * dy)

                if abs(dx) > x_thr:
                    if dx > 0:
                        relations.append({
                            "subject_id": a["entity_id"],
                            "relation": "left_of",
                            "object_id": b["entity_id"],
                        })
                        relations.append({
                            "subject_id": b["entity_id"],
                            "relation": "right_of",
                            "object_id": a["entity_id"],
                        })
                    else:
                        relations.append({
                            "subject_id": a["entity_id"],
                            "relation": "right_of",
                            "object_id": b["entity_id"],
                        })
                        relations.append({
                            "subject_id": b["entity_id"],
                            "relation": "left_of",
                            "object_id": a["entity_id"],
                        })

                if abs(dy) > y_thr:
                    if dy > 0:
                        relations.append({
                            "subject_id": a["entity_id"],
                            "relation": "above",
                            "object_id": b["entity_id"],
                        })
                        relations.append({
                            "subject_id": b["entity_id"],
                            "relation": "below",
                            "object_id": a["entity_id"],
                        })
                    else:
                        relations.append({
                            "subject_id": a["entity_id"],
                            "relation": "below",
                            "object_id": b["entity_id"],
                        })
                        relations.append({
                            "subject_id": b["entity_id"],
                            "relation": "above",
                            "object_id": a["entity_id"],
                        })

                if dist < near_thr:
                    relations.append({
                        "subject_id": a["entity_id"],
                        "relation": "near",
                        "object_id": b["entity_id"],
                    })
                    relations.append({
                        "subject_id": b["entity_id"],
                        "relation": "near",
                        "object_id": a["entity_id"],
                    })

        return relations

    def select_primary_person(self, persons: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not persons:
            return None
        return sorted(persons, key=lambda x: x.get("importance_score", 0.0), reverse=True)[0]

    def select_nearest_entity(self, entities: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not entities:
            return None
        return sorted(entities, key=lambda x: x.get("area", 0), reverse=True)[0]

    def build_person_context(
        self,
        primary_person: Optional[Dict[str, Any]],
        objects: List[Dict[str, Any]],
        width: int,
        height: int,
    ) -> Dict[str, Any]:
        if not primary_person or width <= 0 or height <= 0:
            return {
                "person_id": None,
                "nearby_objects": [],
            }

        px, py = primary_person.get("center_xy", [0, 0])
        near_thr = math.sqrt(width * width + height * height) * 0.22

        nearby_objects = []
        for obj in objects:
            ox, oy = obj.get("center_xy", [0, 0])
            dist = math.sqrt((ox - px) ** 2 + (oy - py) ** 2)
            if dist < near_thr:
                nearby_objects.append(obj)

        nearby_objects = sorted(
            nearby_objects, key=lambda x: x.get("importance_score", 0.0), reverse=True
        )

        return {
            "person_id": primary_person.get("entity_id"),
            "nearby_objects": nearby_objects[:5],
        }

    def detect_changes(self, state: Dict[str, Any], prev_state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not prev_state:
            return {
                "has_changes": False,
                "new_labels": [],
                "removed_labels": [],
                "count_changes": [],
            }

        prev_counts = prev_state.get("counts", {})
        curr_counts = state.get("counts", {})
        labels = sorted(set(prev_counts.keys()) | set(curr_counts.keys()))

        new_labels = []
        removed_labels = []
        count_changes = []

        for label in labels:
            pv = int(prev_counts.get(label, 0))
            cv = int(curr_counts.get(label, 0))

            if pv == 0 and cv > 0:
                new_labels.append(label)
            elif pv > 0 and cv == 0:
                removed_labels.append(label)

            if pv != cv:
                count_changes.append({
                    "label": label,
                    "from": pv,
                    "to": cv,
                })

        return {
            "has_changes": bool(new_labels or removed_labels or count_changes),
            "new_labels": new_labels,
            "removed_labels": removed_labels,
            "count_changes": count_changes,
        }

    def build_human_summary(
        self,
        enriched_persons: List[Dict[str, Any]],
        enriched_objects: List[Dict[str, Any]],
        salient_entities: List[Dict[str, Any]],
        primary_person: Optional[Dict[str, Any]],
        nearest_entity: Optional[Dict[str, Any]],
        person_context: Dict[str, Any],
        changes: Dict[str, Any],
        counts: Dict[str, Any],
        scene_flags: Dict[str, Any],
    ) -> str:
        parts = []

        person_count = int(counts.get("person", len(enriched_persons)))
        if person_count > 0:
            parts.append(f"В кадре {person_count} человек.")
        else:
            parts.append("Людей в кадре сейчас нет.")

        if primary_person:
            parts.append(
                f"Главный человек находится {primary_person.get('position_text', 'в кадре')}, "
                f"{primary_person.get('distance_hint', '')}."
            )

        if person_context.get("nearby_objects"):
            names = []
            for obj in person_context["nearby_objects"]:
                label = obj.get("class_name")
                if label and label not in names:
                    names.append(label)
            if names:
                parts.append("Рядом с человеком видно: " + ", ".join(names[:4]) + ".")

        if nearest_entity:
            parts.append(
                f"Ближе всего к камере сейчас {nearest_entity.get('class_name')} "
                f"({nearest_entity.get('position_text', 'в кадре')})."
            )

        non_person_salient = [e for e in salient_entities if e.get("class_name") != "person"]
        if non_person_salient:
            brief = []
            for ent in non_person_salient[:3]:
                brief.append(f"{ent.get('class_name')} ({ent.get('position_text')})")
            parts.append("Заметные объекты: " + ", ".join(brief) + ".")

        if changes.get("has_changes"):
            short = []
            for item in changes.get("count_changes", [])[:3]:
                short.append(f"{item['label']}: {item['from']}→{item['to']}")
            if short:
                parts.append("Изменения: " + ", ".join(short) + ".")

        return " ".join(p for p in parts if p.strip())


def main(args=None):
    rclpy.init(args=args)
    node = SceneInterpreterNode()
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
