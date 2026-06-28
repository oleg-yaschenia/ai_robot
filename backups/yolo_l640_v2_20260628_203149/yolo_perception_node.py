#!/usr/bin/env python3

import json
import time
from typing import Any, Dict, List, Tuple, Union

import cv2
import rclpy
import torch
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Int32, String
from ultralytics import YOLO


class YoloPerceptionNode(Node):
    """Fast YOLO perception with class thresholds and temporal tracking."""

    SEMANTIC_GROUPS = {
        "person": "person",
        "cat": "pet",
        "dog": "pet",
        "chair": "seating_furniture",
        "couch": "seating_furniture",
        "dining table": "table",
        "cell phone": "phone",
        "tv": "display",
        "laptop": "computer",
        "keyboard": "computer_accessory",
        "mouse": "computer_accessory",
        "bottle": "container",
        "cup": "container",
        "bowl": "container",
        "fork": "cutlery",
        "knife": "cutlery",
        "spoon": "cutlery",
    }

    def __init__(self) -> None:
        super().__init__("yolo_perception_node")

        self.declare_parameter("image_topic", "/camera/left/image_raw")
        self.declare_parameter("model_path", "yolo11l.pt")
        self.declare_parameter("device", "0")
        self.declare_parameter("imgsz", 640)

        # Minimum score returned by YOLO. Final acceptance uses per-class
        # thresholds below.
        self.declare_parameter("inference_conf_threshold", 0.05)
        self.declare_parameter("iou_threshold", 0.45)
        self.declare_parameter("analysis_period_sec", 0.25)
        self.declare_parameter("max_det", 100)

        self.declare_parameter("person_conf_threshold", 0.35)
        self.declare_parameter("pet_conf_threshold", 0.25)
        self.declare_parameter("cup_conf_threshold", 0.20)
        self.declare_parameter("chair_conf_threshold", 0.45)
        self.declare_parameter("default_conf_threshold", 0.25)

        self.declare_parameter("track_iou_threshold", 0.15)
        self.declare_parameter("track_center_distance_factor", 2.5)
        self.declare_parameter("track_center_distance_min_px", 80.0)
        self.declare_parameter("velocity_alpha", 0.65)
        self.declare_parameter("duplicate_iou_threshold", 0.45)
        self.declare_parameter("duplicate_containment_threshold", 0.75)
        self.declare_parameter("confirm_hits", 2)
        self.declare_parameter("max_missed_frames", 4)
        self.declare_parameter("immediate_conf_threshold", 0.75)

        self.image_topic = str(self.get_parameter("image_topic").value)
        self.model_path = str(self.get_parameter("model_path").value)
        self.device = self._parse_device(
            str(self.get_parameter("device").value)
        )
        self.imgsz = int(self.get_parameter("imgsz").value)

        self.inference_conf_threshold = float(
            self.get_parameter("inference_conf_threshold").value
        )
        self.iou_threshold = float(
            self.get_parameter("iou_threshold").value
        )
        self.analysis_period_sec = float(
            self.get_parameter("analysis_period_sec").value
        )
        self.max_det = int(self.get_parameter("max_det").value)

        self.person_conf_threshold = float(
            self.get_parameter("person_conf_threshold").value
        )
        self.pet_conf_threshold = float(
            self.get_parameter("pet_conf_threshold").value
        )
        self.cup_conf_threshold = float(
            self.get_parameter("cup_conf_threshold").value
        )
        self.chair_conf_threshold = float(
            self.get_parameter("chair_conf_threshold").value
        )
        self.default_conf_threshold = float(
            self.get_parameter("default_conf_threshold").value
        )

        self.track_iou_threshold = float(
            self.get_parameter("track_iou_threshold").value
        )
        self.track_center_distance_factor = float(
            self.get_parameter("track_center_distance_factor").value
        )
        self.track_center_distance_min_px = float(
            self.get_parameter("track_center_distance_min_px").value
        )
        self.velocity_alpha = float(
            self.get_parameter("velocity_alpha").value
        )
        self.duplicate_iou_threshold = float(
            self.get_parameter("duplicate_iou_threshold").value
        )
        self.duplicate_containment_threshold = float(
            self.get_parameter("duplicate_containment_threshold").value
        )
        self.confirm_hits = int(
            self.get_parameter("confirm_hits").value
        )
        self.max_missed_frames = int(
            self.get_parameter("max_missed_frames").value
        )
        self.immediate_conf_threshold = float(
            self.get_parameter("immediate_conf_threshold").value
        )

        if self.device != "cpu" and not torch.cuda.is_available():
            self.get_logger().warning(
                "CUDA is unavailable; falling back to CPU"
            )
            self.device = "cpu"

        self.bridge = CvBridge()
        self.last_frame_bgr = None
        self.last_result_state = None
        self._warmed_up = False

        self.model = YOLO(self.model_path)

        self.tracks: Dict[int, Dict[str, Any]] = {}
        self.next_track_id = 1

        self.image_sub = self.create_subscription(
            Image,
            self.image_topic,
            self.image_cb,
            qos_profile_sensor_data,
        )

        self.person_count_pub = self.create_publisher(
            Int32,
            "/perception/person_count",
            10,
        )
        self.scene_summary_pub = self.create_publisher(
            String,
            "/perception/scene_summary",
            10,
        )
        self.state_json_pub = self.create_publisher(
            String,
            "/perception/state_json",
            10,
        )
        self.objects_json_pub = self.create_publisher(
            String,
            "/perception/objects_json",
            10,
        )
        self.debug_pub = self.create_publisher(
            String,
            "/perception/debug",
            10,
        )
        self.candidates_pub = self.create_publisher(
            String,
            "/perception/candidates_json",
            10,
        )

        self.timer = self.create_timer(
            self.analysis_period_sec,
            self.timer_cb,
        )

        self.get_logger().info(
            "yolo_perception_node l640-v1 started: "
            f"model={self.model_path}, device={self.device}, "
            f"imgsz={self.imgsz}, period={self.analysis_period_sec}s, "
            f"nms_iou={self.iou_threshold}"
        )

    @staticmethod
    def _parse_device(value: str) -> Union[int, str]:
        value = value.strip().lower()
        if value in {"cpu", "mps"}:
            return value
        if value.isdigit():
            return int(value)
        return value

    @staticmethod
    def _class_name(names: Any, class_id: int) -> str:
        if isinstance(names, dict):
            return str(names.get(class_id, class_id))
        if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
            return str(names[class_id])
        return str(class_id)

    @staticmethod
    def _iou(
        box_a: List[int],
        box_b: List[int],
    ) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        iw = max(0, ix2 - ix1)
        ih = max(0, iy2 - iy1)
        intersection = iw * ih

        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - intersection

        if union <= 0:
            return 0.0
        return intersection / union

    @staticmethod
    def _ema_box(
        old_box: List[int],
        new_box: List[int],
        alpha: float = 0.65,
    ) -> List[int]:
        return [
            int(round((1.0 - alpha) * old + alpha * new))
            for old, new in zip(old_box, new_box)
        ]

    @staticmethod
    def _center_and_size(
        bbox: List[int],
    ) -> Tuple[List[int], List[int]]:
        x1, y1, x2, y2 = bbox
        return (
            [int((x1 + x2) / 2), int((y1 + y2) / 2)],
            [int(x2 - x1), int(y2 - y1)],
        )

    @staticmethod
    def _overlap_metrics(
        box_a: List[int],
        box_b: List[int],
    ) -> Tuple[float, float]:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - intersection

        iou = intersection / union if union > 0 else 0.0
        smaller_area = min(area_a, area_b)
        containment = (
            intersection / smaller_area if smaller_area > 0 else 0.0
        )
        return iou, containment

    @staticmethod
    def _distance(
        point_a: List[float],
        point_b: List[float],
    ) -> float:
        dx = float(point_a[0]) - float(point_b[0])
        dy = float(point_a[1]) - float(point_b[1])
        return (dx * dx + dy * dy) ** 0.5

    def _suppress_duplicates(
        self,
        detections: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], int]:
        kept: List[Dict[str, Any]] = []
        suppressed = 0

        for detection in sorted(
            detections,
            key=lambda item: item["confidence"],
            reverse=True,
        ):
            duplicate = False

            for existing in kept:
                if existing["class_name"] != detection["class_name"]:
                    continue

                overlap, containment = self._overlap_metrics(
                    existing["bbox_xyxy"],
                    detection["bbox_xyxy"],
                )
                if (
                    overlap >= self.duplicate_iou_threshold
                    or containment >= self.duplicate_containment_threshold
                ):
                    duplicate = True
                    break

            if duplicate:
                suppressed += 1
            else:
                kept.append(detection)

        return kept, suppressed

    def image_cb(self, msg: Image) -> None:
        try:
            if msg.encoding == "bgr8":
                frame = self.bridge.imgmsg_to_cv2(
                    msg,
                    desired_encoding="bgr8",
                )
            elif msg.encoding == "mono8":
                gray = self.bridge.imgmsg_to_cv2(
                    msg,
                    desired_encoding="mono8",
                )
                frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            else:
                frame = self.bridge.imgmsg_to_cv2(
                    msg,
                    desired_encoding="bgr8",
                )

            self.last_frame_bgr = frame
        except Exception as exc:
            self.get_logger().warning(
                f"image conversion failed: {exc}"
            )

    def _threshold_for(self, class_name: str) -> float:
        if class_name == "person":
            return self.person_conf_threshold
        if class_name in {"cat", "dog"}:
            return self.pet_conf_threshold
        if class_name == "cup":
            return self.cup_conf_threshold
        if class_name == "chair":
            return self.chair_conf_threshold
        return self.default_conf_threshold

    def _warm_up(self, frame) -> None:
        if self._warmed_up:
            return

        started = time.perf_counter()
        for _ in range(2):
            self.model.predict(
                source=frame,
                imgsz=self.imgsz,
                conf=self.inference_conf_threshold,
                iou=self.iou_threshold,
                max_det=self.max_det,
                device=self.device,
                verbose=False,
                stream=False,
            )

        if self.device != "cpu":
            torch.cuda.synchronize()

        self._warmed_up = True
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self.get_logger().info(
            f"YOLO warm-up completed in {elapsed_ms:.1f} ms"
        )

    def _update_tracks(
        self,
        detections: List[Dict[str, Any]],
    ) -> None:
        unmatched_track_ids = set(self.tracks.keys())
        unmatched_detection_indexes = set(range(len(detections)))
        possible_matches: List[Tuple[float, int, int]] = []

        for track_id, track in self.tracks.items():
            predicted_center = [
                float(track["center_xy"][0])
                + float(track["velocity_xy"][0])
                * (track["missed_frames"] + 1),
                float(track["center_xy"][1])
                + float(track["velocity_xy"][1])
                * (track["missed_frames"] + 1),
            ]

            for detection_index, detection in enumerate(detections):
                if track["class_name"] != detection["class_name"]:
                    continue

                overlap = self._iou(
                    track["bbox_xyxy"],
                    detection["bbox_xyxy"],
                )
                center_distance = self._distance(
                    predicted_center,
                    detection["center_xy"],
                )
                object_scale = max(
                    track["size_wh"][0],
                    track["size_wh"][1],
                    detection["size_wh"][0],
                    detection["size_wh"][1],
                    1,
                )
                distance_gate = max(
                    self.track_center_distance_min_px,
                    self.track_center_distance_factor * object_scale,
                )

                if (
                    overlap < self.track_iou_threshold
                    and center_distance > distance_gate
                ):
                    continue

                distance_score = max(
                    0.0,
                    1.0 - center_distance / distance_gate,
                )
                match_score = 2.0 * overlap + distance_score
                possible_matches.append(
                    (match_score, track_id, detection_index)
                )

        possible_matches.sort(reverse=True, key=lambda item: item[0])

        for _, track_id, detection_index in possible_matches:
            if track_id not in unmatched_track_ids:
                continue
            if detection_index not in unmatched_detection_indexes:
                continue

            track = self.tracks[track_id]
            detection = detections[detection_index]
            old_center = list(track["center_xy"])
            new_center = list(detection["center_xy"])
            measured_velocity = [
                new_center[0] - old_center[0],
                new_center[1] - old_center[1],
            ]

            track["velocity_xy"] = [
                round(
                    (1.0 - self.velocity_alpha)
                    * float(track["velocity_xy"][axis])
                    + self.velocity_alpha
                    * float(measured_velocity[axis]),
                    2,
                )
                for axis in (0, 1)
            ]
            track["bbox_xyxy"] = self._ema_box(
                track["bbox_xyxy"],
                detection["bbox_xyxy"],
                alpha=0.75,
            )
            center_xy, size_wh = self._center_and_size(
                track["bbox_xyxy"]
            )
            track["center_xy"] = center_xy
            track["size_wh"] = size_wh
            track["confidence"] = round(
                0.25 * float(track["confidence"])
                + 0.75 * float(detection["confidence"]),
                4,
            )
            track["hits"] += 1
            track["missed_frames"] = 0
            track["last_seen"] = time.time()

            if (
                track["hits"] >= self.confirm_hits
                or track["confidence"] >= self.immediate_conf_threshold
                or track["class_name"] == "person"
            ):
                track["confirmed"] = True

            unmatched_track_ids.remove(track_id)
            unmatched_detection_indexes.remove(detection_index)

        for detection_index in unmatched_detection_indexes:
            detection = detections[detection_index]
            track_id = self.next_track_id
            self.next_track_id += 1

            confirmed = (
                detection["class_name"] == "person"
                or detection["confidence"] >= self.immediate_conf_threshold
                or self.confirm_hits <= 1
            )

            self.tracks[track_id] = {
                "track_id": track_id,
                "class_name": detection["class_name"],
                "semantic_group": detection["semantic_group"],
                "confidence": detection["confidence"],
                "bbox_xyxy": detection["bbox_xyxy"],
                "center_xy": detection["center_xy"],
                "size_wh": detection["size_wh"],
                "velocity_xy": [0.0, 0.0],
                "hits": 1,
                "missed_frames": 0,
                "confirmed": confirmed,
                "first_seen": time.time(),
                "last_seen": time.time(),
            }

        for track_id in unmatched_track_ids:
            track = self.tracks[track_id]
            track["missed_frames"] += 1
            track["velocity_xy"] = [
                round(float(value) * 0.85, 2)
                for value in track["velocity_xy"]
            ]

        expired = [
            track_id
            for track_id, track in self.tracks.items()
            if track["missed_frames"] > self.max_missed_frames
        ]
        for track_id in expired:
            del self.tracks[track_id]

    def _stable_detections(self) -> List[Dict[str, Any]]:
        stable: List[Dict[str, Any]] = []

        for track in self.tracks.values():
            if not track["confirmed"]:
                continue

            stable.append({
                "track_id": track["track_id"],
                "class_name": track["class_name"],
                "semantic_group": track["semantic_group"],
                "confidence": round(float(track["confidence"]), 4),
                "bbox_xyxy": track["bbox_xyxy"],
                "center_xy": track["center_xy"],
                "size_wh": track["size_wh"],
                "velocity_xy": track["velocity_xy"],
                "hits": track["hits"],
                "missed_frames": track["missed_frames"],
                "temporally_confirmed": True,
            })

        stable.sort(
            key=lambda item: (
                item["missed_frames"],
                -item["confidence"],
            )
        )
        return stable

    def timer_cb(self) -> None:
        if self.last_frame_bgr is None:
            return

        frame = self.last_frame_bgr.copy()
        height, width = frame.shape[:2]

        try:
            self._warm_up(frame)
            started = time.perf_counter()

            result = self.model.predict(
                source=frame,
                imgsz=self.imgsz,
                conf=self.inference_conf_threshold,
                iou=self.iou_threshold,
                max_det=self.max_det,
                device=self.device,
                verbose=False,
                stream=False,
            )[0]

            if self.device != "cpu":
                torch.cuda.synchronize()

            total_latency_ms = (
                time.perf_counter() - started
            ) * 1000.0
        except Exception as exc:
            self.get_logger().warning(
                f"YOLO inference failed: {exc}"
            )
            return

        accepted_raw: List[Dict[str, Any]] = []
        candidates: List[Dict[str, Any]] = []

        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                x1, y1, x2, y2 = [
                    int(value)
                    for value in box.xyxy[0].tolist()
                ]
                class_name = self._class_name(
                    result.names,
                    class_id,
                )
                threshold = self._threshold_for(class_name)
                bbox = [x1, y1, x2, y2]
                center_xy, size_wh = self._center_and_size(bbox)

                candidate = {
                    "class_id": class_id,
                    "class_name": class_name,
                    "semantic_group": self.SEMANTIC_GROUPS.get(
                        class_name,
                        class_name,
                    ),
                    "confidence": round(confidence, 4),
                    "threshold": round(threshold, 4),
                    "bbox_xyxy": bbox,
                    "center_xy": center_xy,
                    "size_wh": size_wh,
                    "accepted": confidence >= threshold,
                }
                candidates.append(candidate)

                if confidence >= threshold:
                    accepted_raw.append({
                        key: value
                        for key, value in candidate.items()
                        if key not in {"threshold", "accepted", "class_id"}
                    })

        accepted_raw, suppressed_duplicates = self._suppress_duplicates(
            accepted_raw
        )
        self._update_tracks(accepted_raw)
        detections = self._stable_detections()

        persons: List[Dict[str, Any]] = []
        objects: List[Dict[str, Any]] = []
        counts: Dict[str, int] = {}

        for detection in detections:
            class_name = detection["class_name"]
            counts[class_name] = counts.get(class_name, 0) + 1

            if class_name == "person":
                persons.append({
                    **detection,
                    "identity": None,
                    "role_guess": None,
                })
            else:
                objects.append(detection)

        brightness = float(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean()
        )
        if brightness < 50:
            brightness_label = "сцена тёмная"
        elif brightness > 180:
            brightness_label = "сцена очень светлая"
        else:
            brightness_label = "освещение нормальное"

        person_count = len(persons)
        person_present = person_count > 0
        owner_present = False
        pet_present = any(
            obj["class_name"] in {"cat", "dog"}
            for obj in objects
        )

        top_objects = [
            obj["class_name"]
            for obj in objects[:8]
        ]

        if person_present and top_objects:
            scene_summary = (
                f"В кадре {person_count} человек(а), также видны: "
                f"{', '.join(top_objects)}. {brightness_label}."
            )
        elif person_present:
            scene_summary = (
                f"В кадре {person_count} человек(а). "
                f"{brightness_label}."
            )
        elif top_objects:
            scene_summary = (
                f"Людей нет, видны: {', '.join(top_objects)}. "
                f"{brightness_label}."
            )
        else:
            scene_summary = (
                "Людей и заметных объектов не обнаружено. "
                f"{brightness_label}."
            )

        state = {
            "timestamp": time.time(),
            "frame_id": "camera_left_optical_frame",
            "image_size": {
                "width": width,
                "height": height,
            },
            "persons": persons,
            "objects": objects,
            "detections": detections,
            "counts": counts,
            "scene_flags": {
                "person_present": person_present,
                "owner_present": owner_present,
                "pet_present": pet_present,
            },
            "scene_summary": scene_summary,
            "detector": {
                "model": self.model_path,
                "type": "ultralytics_yolo",
                "version": "l640-v2",
                "device": str(self.device),
                "imgsz": self.imgsz,
                "inference_conf_threshold": (
                    self.inference_conf_threshold
                ),
                "analysis_period_sec": self.analysis_period_sec,
                "latency_ms": round(total_latency_ms, 2),
                "speed": {
                    key: round(float(value), 3)
                    for key, value in result.speed.items()
                },
                "active_tracks": len(self.tracks),
                "raw_accepted_detections": len(accepted_raw),
                "suppressed_duplicates": suppressed_duplicates,
            },
        }

        self.last_result_state = state

        count_msg = Int32()
        count_msg.data = person_count
        self.person_count_pub.publish(count_msg)

        summary_msg = String()
        summary_msg.data = scene_summary
        self.scene_summary_pub.publish(summary_msg)

        state_msg = String()
        state_msg.data = json.dumps(
            state,
            ensure_ascii=False,
        )
        self.state_json_pub.publish(state_msg)

        objects_msg = String()
        objects_msg.data = json.dumps(
            objects,
            ensure_ascii=False,
        )
        self.objects_json_pub.publish(objects_msg)

        candidates_msg = String()
        candidates_msg.data = json.dumps(
            candidates,
            ensure_ascii=False,
        )
        self.candidates_pub.publish(candidates_msg)

        debug_msg = String()
        debug_msg.data = (
            f"latency_ms={total_latency_ms:.1f}, "
            f"person_count={person_count}, "
            f"objects={[obj['class_name'] for obj in objects[:12]]}, "
            f"counts={counts}, "
            f"tracks={len(self.tracks)}, "
            f"suppressed_duplicates={suppressed_duplicates}"
        )
        self.debug_pub.publish(debug_msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = YoloPerceptionNode()

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
