#!/usr/bin/env python3
import json
import time
import statistics
from collections import deque

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String, Int32
from cv_bridge import CvBridge


class LocalPerceptionNode(Node):
    def __init__(self):
        super().__init__("local_perception_node")

        self.declare_parameter("image_topic", "/camera/left/image_raw")
        self.declare_parameter("analysis_period_sec", 0.5)
        self.declare_parameter("presence_hold_sec", 2.0)
        self.declare_parameter("motion_threshold_low", 4.0)
        self.declare_parameter("motion_threshold_high", 12.0)

        self.image_topic = str(self.get_parameter("image_topic").value)
        self.analysis_period_sec = float(self.get_parameter("analysis_period_sec").value)
        self.presence_hold_sec = float(self.get_parameter("presence_hold_sec").value)
        self.motion_threshold_low = float(self.get_parameter("motion_threshold_low").value)
        self.motion_threshold_high = float(self.get_parameter("motion_threshold_high").value)

        self.bridge = CvBridge()
        self.last_frame_bgr = None
        self.prev_gray = None
        self.last_person_seen_ts = 0.0
        self.person_history = deque(maxlen=7)

        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        self.image_sub = self.create_subscription(
            Image, self.image_topic, self.image_cb, 10
        )

        self.person_count_pub = self.create_publisher(Int32, "/perception/person_count", 10)
        self.scene_summary_pub = self.create_publisher(String, "/perception/scene_summary", 10)
        self.state_json_pub = self.create_publisher(String, "/perception/state_json", 10)
        self.debug_pub = self.create_publisher(String, "/perception/debug", 10)

        self.timer = self.create_timer(self.analysis_period_sec, self.timer_cb)

        self.get_logger().info(
            f"local_perception_node started: image_topic={self.image_topic}, "
            f"analysis_period_sec={self.analysis_period_sec}"
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

    def detect_people_hog(self, frame_bgr):
        resized = cv2.resize(frame_bgr, (640, 360))
        rects, weights = self.hog.detectMultiScale(
            resized,
            winStride=(8, 8),
            padding=(8, 8),
            scale=1.05
        )
        return rects, weights

    def timer_cb(self):
        if self.last_frame_bgr is None:
            return

        frame = self.last_frame_bgr.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        brightness = float(gray.mean())
        if brightness < 50:
            brightness_label = "сцена тёмная"
        elif brightness > 180:
            brightness_label = "сцена очень светлая"
        else:
            brightness_label = "освещение нормальное"

        motion_score = 0.0
        motion_label = "движение не оценено"
        if self.prev_gray is not None:
            diff = cv2.absdiff(self.prev_gray, gray)
            motion_score = float(diff.mean())

            if motion_score > self.motion_threshold_high:
                motion_label = "в сцене есть заметное изменение"
            elif motion_score > self.motion_threshold_low:
                motion_label = "в сцене есть небольшое изменение"
            else:
                motion_label = "сцена почти не изменилась"

        self.prev_gray = gray

        rects, weights = self.detect_people_hog(frame)
        raw_person_count = len(rects)

        now = time.time()
        if raw_person_count > 0:
            self.last_person_seen_ts = now

        held_person_present = (now - self.last_person_seen_ts) <= self.presence_hold_sec

        self.person_history.append(raw_person_count)

        if len(self.person_history) > 0:
            stable_person_count = int(round(statistics.median(self.person_history)))
        else:
            stable_person_count = raw_person_count

        # Для v1 домашнего ассистента важнее надёжно понять:
        # есть человек или нет. Поэтому удерживаем присутствие.
        if stable_person_count == 0 and held_person_present:
            stable_person_count = 1

        person_present = stable_person_count > 0

        if person_present:
            people_label = f"людей обнаружено: {stable_person_count}"
        else:
            people_label = "людей не обнаружено"

        summary = f"{brightness_label}; {people_label}; {motion_label}"

        person_msg = Int32()
        person_msg.data = int(stable_person_count)
        self.person_count_pub.publish(person_msg)

        summary_msg = String()
        summary_msg.data = summary
        self.scene_summary_pub.publish(summary_msg)

        state = {
            "raw_person_count": int(raw_person_count),
            "stable_person_count": int(stable_person_count),
            "person_present": bool(person_present),
            "brightness": brightness_label,
            "motion_label": motion_label,
            "motion_score": round(motion_score, 3),
            "person_hold_active": bool(held_person_present),
        }

        state_msg = String()
        state_msg.data = json.dumps(state, ensure_ascii=False)
        self.state_json_pub.publish(state_msg)

        debug_msg = String()
        debug_msg.data = (
            f"raw_person_count={raw_person_count}, "
            f"stable_person_count={stable_person_count}, "
            f"motion_score={motion_score:.3f}, "
            f"hold_active={held_person_present}"
        )
        self.debug_pub.publish(debug_msg)


def main(args=None):
    rclpy.init(args=args)
    node = LocalPerceptionNode()
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
