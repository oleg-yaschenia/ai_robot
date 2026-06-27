#!/usr/bin/env python3
import json
import time
from typing import List, Dict, Any

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String, Int32
from cv_bridge import CvBridge
from ultralytics import YOLO


class YoloPerceptionNode(Node):
    def __init__(self):
        super().__init__("yolo_perception_node")

        self.declare_parameter("image_topic", "/camera/left/image_raw")
        self.declare_parameter("model_path", "yolo11n.pt")
        self.declare_parameter("imgsz", 640)
        self.declare_parameter("conf_threshold", 0.35)
        self.declare_parameter("analysis_period_sec", 0.5)
        self.declare_parameter("max_det", 20)

        self.image_topic = str(self.get_parameter("image_topic").value)
        self.model_path = str(self.get_parameter("model_path").value)
        self.imgsz = int(self.get_parameter("imgsz").value)
        self.conf_threshold = float(self.get_parameter("conf_threshold").value)
        self.analysis_period_sec = float(self.get_parameter("analysis_period_sec").value)
        self.max_det = int(self.get_parameter("max_det").value)

        self.bridge = CvBridge()
        self.last_frame_bgr = None
        self.last_result_state = None
        
        self.allowed_classes = {
            "person",
            "cat",
            "dog",
            "cup",
            "bottle",
            "cell phone",
            "laptop",
            "chair",
        }

        self.model = YOLO(self.model_path)

        self.image_sub = self.create_subscription(
            Image, self.image_topic, self.image_cb, 10
        )

        self.person_count_pub = self.create_publisher(Int32, "/perception/person_count", 10)
        self.scene_summary_pub = self.create_publisher(String, "/perception/scene_summary", 10)
        self.state_json_pub = self.create_publisher(String, "/perception/state_json", 10)
        self.objects_json_pub = self.create_publisher(String, "/perception/objects_json", 10)
        self.debug_pub = self.create_publisher(String, "/perception/debug", 10)

        self.timer = self.create_timer(self.analysis_period_sec, self.timer_cb)

        self.get_logger().info(
            f"yolo_perception_node started: image_topic={self.image_topic}, "
            f"model_path={self.model_path}, imgsz={self.imgsz}, conf={self.conf_threshold}"
        )

    def image_cb(self, msg: Image):
        try:
            if msg.encoding == "bgr8":
                frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            elif msg.encoding == "mono8":
                gray = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
                frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            else:
                frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

            self.last_frame_bgr = frame
        except Exception as e:
            self.get_logger().warning(f"image conversion failed: {e}")

    def timer_cb(self):
        if self.last_frame_bgr is None:
            return

        frame = self.last_frame_bgr.copy()
        h, w = frame.shape[:2]

        try:
            result = self.model.predict(
                source=frame,
                imgsz=self.imgsz,
                conf=self.conf_threshold,
                max_det=self.max_det,
                verbose=False,
                stream=False,
            )[0]
        except Exception as e:
            self.get_logger().warning(f"YOLO inference failed: {e}")
            return

        names = result.names
        boxes = result.boxes

        detections: List[Dict[str, Any]] = []
        persons: List[Dict[str, Any]] = []
        objects: List[Dict[str, Any]] = []
        counts: Dict[str, int] = {}

        if boxes is not None:
            for box in boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                xyxy = box.xyxy[0].tolist()
                x1, y1, x2, y2 = [int(v) for v in xyxy]
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                bw = int(x2 - x1)
                bh = int(y2 - y1)

                class_name = str(names.get(cls_id, str(cls_id)))
                
                if class_name not in self.allowed_classes:
                    continue
                min_conf = 0.5 if class_name == "person" else 0.6
                if conf < min_conf:
                    continue

                det = {
                    "class_name": class_name,
                    "confidence": round(conf, 4),
                    "bbox_xyxy": [x1, y1, x2, y2],
                    "center_xy": [cx, cy],
                    "size_wh": [bw, bh],
                }
                detections.append(det)

                counts[class_name] = counts.get(class_name, 0) + 1

                if class_name == "person":
                    persons.append({
                        "track_id": None,
                        "confidence": round(conf, 4),
                        "bbox_xyxy": [x1, y1, x2, y2],
                        "center_xy": [cx, cy],
                        "size_wh": [bw, bh],
                        "identity": None,
                        "role_guess": None,
                    })
                else:
                    objects.append(det)

        brightness = float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean())
        if brightness < 50:
            brightness_label = "сцена тёмная"
        elif brightness > 180:
            brightness_label = "сцена очень светлая"
        else:
            brightness_label = "освещение нормальное"

        person_count = len(persons)
        person_present = person_count > 0
        owner_present = False
        pet_present = any(obj["class_name"] in ("cat", "dog") for obj in objects)

        object_names = [obj["class_name"] for obj in objects]
        top_objects = object_names[:5]

        if person_present and top_objects:
            scene_summary = (
                f"В кадре {person_count} человек(а), также видны: {', '.join(top_objects)}. "
                f"{brightness_label}."
            )
        elif person_present:
            scene_summary = f"В кадре {person_count} человек(а). {brightness_label}."
        elif top_objects:
            scene_summary = f"Людей нет, видны: {', '.join(top_objects)}. {brightness_label}."
        else:
            scene_summary = f"Людей и заметных объектов не обнаружено. {brightness_label}."

        state = {
            "timestamp": time.time(),
            "frame_id": "camera_left_optical_frame",
            "image_size": {
                "width": w,
                "height": h,
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
                "version": "v1",
            },
        }

        self.last_result_state = state

        msg_count = Int32()
        msg_count.data = person_count
        self.person_count_pub.publish(msg_count)

        msg_summary = String()
        msg_summary.data = scene_summary
        self.scene_summary_pub.publish(msg_summary)

        msg_state = String()
        msg_state.data = json.dumps(state, ensure_ascii=False)
        self.state_json_pub.publish(msg_state)

        msg_objects = String()
        msg_objects.data = json.dumps(objects, ensure_ascii=False)
        self.objects_json_pub.publish(msg_objects)

        msg_debug = String()
        msg_debug.data = (
            f"person_count={person_count}, "
            f"objects={[obj['class_name'] for obj in objects[:8]]}, "
            f"counts={counts}"
        )
        self.debug_pub.publish(msg_debug)


def main(args=None):
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
