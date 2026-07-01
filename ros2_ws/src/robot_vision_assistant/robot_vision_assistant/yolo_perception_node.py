#!/usr/bin/env python3

import json
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import rclpy
import torch
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from sensor_msgs.msg import Image
from stereo_msgs.msg import DisparityImage
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

        self.declare_parameter("image_topic", "/camera/right/image_rect")
        self.declare_parameter("model_path", "yolo11l.pt")
        self.declare_parameter("fallback_model_path", "")
        self.declare_parameter("device", "0")
        self.declare_parameter("imgsz", 640)

        # Minimum score returned by YOLO. Final acceptance uses per-class
        # thresholds below.
        self.declare_parameter("inference_conf_threshold", 0.10)
        self.declare_parameter("iou_threshold", 0.45)
        self.declare_parameter("analysis_period_sec", 0.25)
        self.declare_parameter("max_det", 100)

        self.declare_parameter("person_conf_threshold", 0.35)
        self.declare_parameter("pet_conf_threshold", 0.25)
        self.declare_parameter("cup_conf_threshold", 0.20)
        self.declare_parameter("chair_conf_threshold", 0.45)
        self.declare_parameter("default_conf_threshold", 0.35)

        self.declare_parameter("tv_conf_threshold", 0.55)
        self.declare_parameter("remote_conf_threshold", 0.55)
        self.declare_parameter("cell_phone_conf_threshold", 0.40)
        self.declare_parameter("mouse_conf_threshold", 0.40)
        self.declare_parameter("keyboard_conf_threshold", 0.30)
        self.declare_parameter("laptop_conf_threshold", 0.30)

        self.declare_parameter("track_iou_threshold", 0.15)
        self.declare_parameter("track_center_distance_factor", 2.5)
        self.declare_parameter("track_center_distance_min_px", 80.0)
        self.declare_parameter("velocity_alpha", 0.65)
        self.declare_parameter("duplicate_iou_threshold", 0.45)
        self.declare_parameter("duplicate_containment_threshold", 0.75)

        self.declare_parameter("default_confirm_hits", 3)
        self.declare_parameter("motion_extra_hits", 2)
        self.declare_parameter("max_missed_frames", 4)
        self.declare_parameter("noisy_max_missed_frames", 2)
        self.declare_parameter("immediate_conf_threshold", 0.85)

        self.declare_parameter("motion_threshold", 0.035)
        self.declare_parameter("motion_diff_threshold", 22)
        self.declare_parameter("motion_conf_boost", 0.10)

        # Spatial metadata. These are relative image-space estimates,
        # not calibrated metric distance.
        self.declare_parameter("horizontal_left_boundary", 0.38)
        self.declare_parameter("horizontal_right_boundary", 0.62)
        self.declare_parameter("vertical_upper_boundary", 0.33)
        self.declare_parameter("vertical_lower_boundary", 0.67)

        # Metric depth from stereo disparity. YOLO must run on the same
        # rectified physical-left image used as the left input of the
        # disparity node.
        self.declare_parameter("depth_enabled", True)
        self.declare_parameter("depth_topic", "/disparity")
        self.declare_parameter("depth_max_age_sec", 0.35)
        self.declare_parameter("depth_min_m", 0.35)
        self.declare_parameter("depth_max_m", 8.0)
        self.declare_parameter("depth_roi_scale_x", 0.50)
        self.declare_parameter("depth_roi_scale_y", 0.50)
        self.declare_parameter("depth_min_samples", 40)
        self.declare_parameter("depth_min_valid_ratio", 0.08)
        self.declare_parameter("depth_max_relative_spread", 0.35)
        self.declare_parameter("depth_max_absolute_spread_m", 0.35)
        self.declare_parameter("depth_saturation_margin_px", 1.0)

        self.image_topic = str(self.get_parameter("image_topic").value)
        self.requested_model_path = str(
            self.get_parameter("model_path").value
        ).strip()
        self.fallback_model_path = str(
            self.get_parameter("fallback_model_path").value
        ).strip()
        self.model_path = self.requested_model_path
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
        self.tv_conf_threshold = float(
            self.get_parameter("tv_conf_threshold").value
        )
        self.remote_conf_threshold = float(
            self.get_parameter("remote_conf_threshold").value
        )
        self.cell_phone_conf_threshold = float(
            self.get_parameter("cell_phone_conf_threshold").value
        )
        self.mouse_conf_threshold = float(
            self.get_parameter("mouse_conf_threshold").value
        )
        self.keyboard_conf_threshold = float(
            self.get_parameter("keyboard_conf_threshold").value
        )
        self.laptop_conf_threshold = float(
            self.get_parameter("laptop_conf_threshold").value
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
        self.default_confirm_hits = int(
            self.get_parameter("default_confirm_hits").value
        )
        self.motion_extra_hits = int(
            self.get_parameter("motion_extra_hits").value
        )
        self.max_missed_frames = int(
            self.get_parameter("max_missed_frames").value
        )
        self.noisy_max_missed_frames = int(
            self.get_parameter("noisy_max_missed_frames").value
        )
        self.immediate_conf_threshold = float(
            self.get_parameter("immediate_conf_threshold").value
        )
        self.motion_threshold = float(
            self.get_parameter("motion_threshold").value
        )
        self.motion_diff_threshold = int(
            self.get_parameter("motion_diff_threshold").value
        )
        self.motion_conf_boost = float(
            self.get_parameter("motion_conf_boost").value
        )

        self.horizontal_left_boundary = float(
            self.get_parameter("horizontal_left_boundary").value
        )
        self.horizontal_right_boundary = float(
            self.get_parameter("horizontal_right_boundary").value
        )
        self.vertical_upper_boundary = float(
            self.get_parameter("vertical_upper_boundary").value
        )
        self.vertical_lower_boundary = float(
            self.get_parameter("vertical_lower_boundary").value
        )

        self.depth_enabled = bool(
            self.get_parameter("depth_enabled").value
        )
        self.depth_topic = str(
            self.get_parameter("depth_topic").value
        )
        self.depth_max_age_sec = float(
            self.get_parameter("depth_max_age_sec").value
        )
        self.depth_min_m = float(
            self.get_parameter("depth_min_m").value
        )
        self.depth_max_m = float(
            self.get_parameter("depth_max_m").value
        )
        self.depth_roi_scale_x = float(
            self.get_parameter("depth_roi_scale_x").value
        )
        self.depth_roi_scale_y = float(
            self.get_parameter("depth_roi_scale_y").value
        )
        self.depth_min_samples = int(
            self.get_parameter("depth_min_samples").value
        )
        self.depth_min_valid_ratio = float(
            self.get_parameter("depth_min_valid_ratio").value
        )
        self.depth_max_relative_spread = float(
            self.get_parameter("depth_max_relative_spread").value
        )
        self.depth_max_absolute_spread_m = float(
            self.get_parameter("depth_max_absolute_spread_m").value
        )
        self.depth_saturation_margin_px = float(
            self.get_parameter("depth_saturation_margin_px").value
        )

        if self.device != "cpu" and not torch.cuda.is_available():
            self.get_logger().warning(
                "CUDA is unavailable; falling back to CPU"
            )
            self.device = "cpu"

        self.bridge = CvBridge()
        self.last_frame_bgr = None
        self.last_frame_stamp_sec: Optional[float] = None
        self.last_frame_id = ""
        self.last_result_state = None
        self._warmed_up = False
        self.previous_motion_gray = None
        self.disparity_frames = deque(maxlen=6)

        self.active_model_path = ""
        self.inference_backend = "unknown"
        self.fallback_used = False
        self.fallback_reason: Optional[str] = None
        self.model = self._load_preferred_or_fallback()

        self.tracks: Dict[int, Dict[str, Any]] = {}
        self.next_track_id = 1

        self.image_sub = self.create_subscription(
            Image,
            self.image_topic,
            self.image_cb,
            qos_profile_sensor_data,
        )

        depth_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=2,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.depth_sub = self.create_subscription(
            DisparityImage,
            self.depth_topic,
            self.disparity_cb,
            depth_qos,
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
        self.target_pub = self.create_publisher(
            String,
            "/perception/target_json",
            10,
        )

        self.timer = self.create_timer(
            self.analysis_period_sec,
            self.timer_cb,
        )

        self.get_logger().info(
            "yolo_perception_node l640-v5-depth started: "
            f"requested_model={self.requested_model_path}, "
            f"active_model={self.active_model_path}, "
            f"backend={self.inference_backend}, "
            f"fallback_used={self.fallback_used}, "
            f"device={self.device}, imgsz={self.imgsz}, "
            f"period={self.analysis_period_sec}s, "
            f"nms_iou={self.iou_threshold}"
        )

    @staticmethod
    def _normalize_model_path(model_path: str) -> str:
        model_path = model_path.strip()
        if not model_path:
            return ""
        return str(Path(model_path).expanduser())

    @staticmethod
    def _backend_for_path(model_path: str) -> str:
        suffix = Path(model_path).suffix.lower()
        if suffix == ".engine":
            return "tensorrt"
        if suffix == ".pt":
            return "pytorch"
        return suffix.lstrip(".") or "unknown"

    @staticmethod
    def _model_error_text(exc: Exception) -> str:
        return f"{type(exc).__name__}: {exc}"[:500]

    def _activate_model(
        self,
        model_path: str,
        fallback_reason: Optional[str] = None,
    ):
        normalized_path = self._normalize_model_path(model_path)
        if not normalized_path:
            raise ValueError("model path is empty")
        if not Path(normalized_path).is_file():
            raise FileNotFoundError(normalized_path)

        model = YOLO(normalized_path, task="detect")
        self.model_path = normalized_path
        self.active_model_path = normalized_path
        self.inference_backend = self._backend_for_path(normalized_path)
        requested_path = self._normalize_model_path(
            self.requested_model_path
        )
        self.fallback_used = normalized_path != requested_path
        self.fallback_reason = fallback_reason
        self._warmed_up = False
        return model

    def _load_preferred_or_fallback(self):
        requested_path = self._normalize_model_path(
            self.requested_model_path
        )
        fallback_path = self._normalize_model_path(
            self.fallback_model_path
        )

        try:
            return self._activate_model(requested_path)
        except Exception as primary_exc:
            primary_error = self._model_error_text(primary_exc)

            if not fallback_path or fallback_path == requested_path:
                raise RuntimeError(
                    "Failed to load requested YOLO model "
                    f"{requested_path}: {primary_error}; "
                    "no distinct fallback configured"
                ) from primary_exc

            self.get_logger().warning(
                "Requested YOLO model could not be loaded; trying fallback: "
                f"requested={requested_path}, fallback={fallback_path}, "
                f"reason={primary_error}"
            )
            try:
                return self._activate_model(
                    fallback_path,
                    fallback_reason=primary_error,
                )
            except Exception as fallback_exc:
                fallback_error = self._model_error_text(fallback_exc)
                raise RuntimeError(
                    "Failed to load both YOLO models: "
                    f"requested={requested_path} ({primary_error}); "
                    f"fallback={fallback_path} ({fallback_error})"
                ) from fallback_exc

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

    @staticmethod
    def _stamp_to_sec(stamp) -> float:
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    @staticmethod
    def _disparity_array(msg: DisparityImage) -> np.ndarray:
        image = msg.image
        if image.encoding not in {"32FC1", "32FC"}:
            raise ValueError(
                f"expected 32FC1 disparity, got {image.encoding}"
            )
        if image.step % 4 != 0:
            raise ValueError(f"invalid disparity step: {image.step}")

        dtype = np.dtype(">f4" if image.is_bigendian else "<f4")
        floats_per_row = image.step // 4
        values = np.frombuffer(image.data, dtype=dtype)
        values = values.reshape((image.height, floats_per_row))
        return values[:, : image.width]

    def disparity_cb(self, msg: DisparityImage) -> None:
        if not self.depth_enabled:
            return

        try:
            disparity = self._disparity_array(msg).copy()
        except Exception as exc:
            self.get_logger().warning(
                f"disparity conversion failed: {exc}"
            )
            return

        self.disparity_frames.append({
            "stamp_sec": self._stamp_to_sec(msg.header.stamp),
            "frame_id": msg.header.frame_id,
            "array": disparity,
            "focal_px": float(msg.f),
            "baseline_m": abs(float(msg.t)),
            "min_disparity": float(msg.min_disparity),
            "max_disparity": float(msg.max_disparity),
        })

    def _nearest_disparity(
        self,
        image_stamp_sec: Optional[float],
    ) -> Optional[Dict[str, Any]]:
        if (
            not self.depth_enabled
            or image_stamp_sec is None
            or not self.disparity_frames
        ):
            return None

        best = min(
            self.disparity_frames,
            key=lambda item: abs(
                float(item["stamp_sec"]) - image_stamp_sec
            ),
        )
        age_sec = abs(float(best["stamp_sec"]) - image_stamp_sec)
        if age_sec > self.depth_max_age_sec:
            return None

        selected = dict(best)
        selected["age_ms"] = age_sec * 1000.0
        return selected

    @staticmethod
    def _depth_defaults(status: str) -> Dict[str, Any]:
        return {
            "distance_m": None,
            "distance_valid": False,
            "depth_source": "stereo_disparity",
            "depth_confidence": "none",
            "depth_status": status,
            "depth_samples": 0,
            "depth_valid_ratio": 0.0,
            "depth_p10_m": None,
            "depth_p90_m": None,
            "depth_age_ms": None,
        }

    def _with_metric_depth(
        self,
        detection: Dict[str, Any],
        depth_frame: Optional[Dict[str, Any]],
        image_width: int,
        image_height: int,
    ) -> Dict[str, Any]:
        enriched = dict(detection)
        depth = self._depth_defaults("no_matching_depth_frame")

        if int(detection.get("missed_frames", 0)) > 0:
            depth["depth_status"] = "track_not_visible_now"
        elif depth_frame is not None:
            disparity = depth_frame["array"]
            disp_height, disp_width = disparity.shape[:2]

            scale_x = float(disp_width) / max(float(image_width), 1.0)
            scale_y = float(disp_height) / max(float(image_height), 1.0)

            x1, y1, x2, y2 = detection["bbox_xyxy"]
            x1 = max(0, min(disp_width - 1, int(round(x1 * scale_x))))
            x2 = max(0, min(disp_width, int(round(x2 * scale_x))))
            y1 = max(0, min(disp_height - 1, int(round(y1 * scale_y))))
            y2 = max(0, min(disp_height, int(round(y2 * scale_y))))

            box_width = max(1, x2 - x1)
            box_height = max(1, y2 - y1)

            scale_roi_x = self.depth_roi_scale_x
            scale_roi_y = self.depth_roi_scale_y
            if box_width * box_height < 2500:
                scale_roi_x = max(scale_roi_x, 0.70)
                scale_roi_y = max(scale_roi_y, 0.70)

            roi_width = max(5, int(round(box_width * scale_roi_x)))
            roi_height = max(5, int(round(box_height * scale_roi_y)))
            center_x = int(round((x1 + x2) * 0.5))
            center_y = int(round((y1 + y2) * 0.5))

            rx1 = max(0, center_x - roi_width // 2)
            rx2 = min(disp_width, rx1 + roi_width)
            ry1 = max(0, center_y - roi_height // 2)
            ry2 = min(disp_height, ry1 + roi_height)

            roi = disparity[ry1:ry2, rx1:rx2]
            min_disp = float(depth_frame["min_disparity"])
            max_disp = float(depth_frame["max_disparity"])

            lower_bound = max(0.0, min_disp) + 0.05
            upper_bound = max_disp - self.depth_saturation_margin_px
            valid_mask = (
                np.isfinite(roi)
                & (roi > lower_bound)
                & (roi < upper_bound)
            )

            roi_pixels = int(roi.size)
            disparities = roi[valid_mask].astype(np.float32)
            initial_count = int(disparities.size)
            valid_ratio = (
                float(initial_count) / float(roi_pixels)
                if roi_pixels > 0
                else 0.0
            )

            depth.update({
                "depth_age_ms": round(
                    float(depth_frame.get("age_ms", 0.0)),
                    1,
                ),
                "depth_samples": initial_count,
                "depth_valid_ratio": round(valid_ratio, 4),
            })

            focal_px = float(depth_frame["focal_px"])
            baseline_m = float(depth_frame["baseline_m"])

            if focal_px <= 0.0 or baseline_m <= 0.0:
                depth["depth_status"] = "invalid_stereo_calibration"
            elif initial_count < self.depth_min_samples:
                depth["depth_status"] = "insufficient_samples"
            elif valid_ratio < self.depth_min_valid_ratio:
                depth["depth_status"] = "low_valid_ratio"
            else:
                depths = focal_px * baseline_m / disparities
                depths = depths[
                    np.isfinite(depths)
                    & (depths >= self.depth_min_m)
                    & (depths <= self.depth_max_m)
                ]

                if depths.size < self.depth_min_samples:
                    depth["depth_status"] = "samples_out_of_range"
                    depth["depth_samples"] = int(depths.size)
                else:
                    distance_m = float(np.median(depths))
                    p10_m = float(np.percentile(depths, 10))
                    p90_m = float(np.percentile(depths, 90))
                    spread_m = max(0.0, p90_m - p10_m)
                    relative_spread = spread_m / max(distance_m, 0.01)
                    max_spread = max(
                        self.depth_max_absolute_spread_m,
                        self.depth_max_relative_spread * distance_m,
                    )

                    if spread_m > max_spread:
                        depth["depth_status"] = "unstable_depth_spread"
                    else:
                        if (
                            valid_ratio >= 0.30
                            and relative_spread <= 0.15
                        ):
                            confidence = "high"
                        elif (
                            valid_ratio >= 0.15
                            and relative_spread <= 0.30
                        ):
                            confidence = "medium"
                        else:
                            confidence = "low"

                        depth.update({
                            "distance_m": round(distance_m, 3),
                            "distance_valid": True,
                            "depth_confidence": confidence,
                            "depth_status": "valid",
                            "depth_samples": int(depths.size),
                            "depth_p10_m": round(p10_m, 3),
                            "depth_p90_m": round(p90_m, 3),
                        })

        enriched.update(depth)
        spatial = dict(enriched.get("spatial", {}))
        spatial.update({
            "metric_distance_available": bool(
                depth["distance_valid"]
            ),
            "distance_m": depth["distance_m"],
            "depth_source": depth["depth_source"],
            "depth_confidence": depth["depth_confidence"],
            "depth_status": depth["depth_status"],
        })
        enriched["spatial"] = spatial
        return enriched

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
            self.last_frame_stamp_sec = self._stamp_to_sec(
                msg.header.stamp
            )
            self.last_frame_id = msg.header.frame_id
        except Exception as exc:
            self.get_logger().warning(
                f"image conversion failed: {exc}"
            )

    @staticmethod
    def _is_noisy_class(class_name: str) -> bool:
        return class_name in {"tv", "remote", "cell phone", "mouse"}

    def _threshold_for(self, class_name: str) -> float:
        if class_name == "person":
            return self.person_conf_threshold
        if class_name in {"cat", "dog"}:
            return self.pet_conf_threshold
        if class_name == "cup":
            return self.cup_conf_threshold
        if class_name == "chair":
            return self.chair_conf_threshold
        if class_name == "tv":
            return self.tv_conf_threshold
        if class_name == "remote":
            return self.remote_conf_threshold
        if class_name == "cell phone":
            return self.cell_phone_conf_threshold
        if class_name == "mouse":
            return self.mouse_conf_threshold
        if class_name == "keyboard":
            return self.keyboard_conf_threshold
        if class_name == "laptop":
            return self.laptop_conf_threshold
        return self.default_conf_threshold

    def _required_hits_for(
        self,
        class_name: str,
        scene_motion: bool,
    ) -> int:
        if class_name == "person":
            return 1
        if class_name in {"cat", "dog", "cup", "chair", "keyboard", "laptop"}:
            required = 2
        elif self._is_noisy_class(class_name):
            required = 4
        else:
            required = self.default_confirm_hits

        if scene_motion and class_name != "person":
            required += self.motion_extra_hits
        return required

    def _max_missed_for(self, class_name: str) -> int:
        if self._is_noisy_class(class_name):
            return self.noisy_max_missed_frames
        return self.max_missed_frames

    def _can_confirm_immediately(
        self,
        class_name: str,
        confidence: float,
    ) -> bool:
        if class_name == "person":
            return True
        if self._is_noisy_class(class_name):
            return False
        return confidence >= self.immediate_conf_threshold

    def _compute_motion_ratio(self, frame) -> float:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (320, 180), interpolation=cv2.INTER_AREA)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        if self.previous_motion_gray is None:
            self.previous_motion_gray = gray
            return 0.0

        difference = cv2.absdiff(gray, self.previous_motion_gray)
        self.previous_motion_gray = gray
        changed = difference >= self.motion_diff_threshold
        return float(changed.mean())

    @staticmethod
    def _proximity_thresholds(
        class_name: str,
    ) -> Tuple[str, float, float]:
        """Return proxy source and near/medium thresholds.

        The estimate is based only on bounding-box size. It is useful for
        relative robot behaviour, but it is not a metric distance.
        """
        small_objects = {
            "cup",
            "bottle",
            "bowl",
            "remote",
            "cell phone",
            "mouse",
            "book",
            "fork",
            "knife",
            "spoon",
        }
        medium_objects = {
            "laptop",
            "keyboard",
            "backpack",
            "handbag",
        }

        if class_name == "person":
            return "bbox_height_ratio", 0.70, 0.40
        if class_name in {"cat", "dog"}:
            return "bbox_height_ratio", 0.38, 0.18
        if class_name in {"chair", "couch", "dining table", "bed", "tv"}:
            return "bbox_area_ratio", 0.22, 0.07
        if class_name in small_objects:
            return "bbox_area_ratio", 0.035, 0.008
        if class_name in medium_objects:
            return "bbox_area_ratio", 0.10, 0.025
        return "bbox_area_ratio", 0.14, 0.035

    def _with_spatial(
        self,
        detection: Dict[str, Any],
        image_width: int,
        image_height: int,
    ) -> Dict[str, Any]:
        enriched = dict(detection)
        center_x, center_y = detection["center_xy"]
        box_width, box_height = detection["size_wh"]

        x_ratio = float(center_x) / max(float(image_width), 1.0)
        y_ratio = float(center_y) / max(float(image_height), 1.0)
        width_ratio = float(box_width) / max(float(image_width), 1.0)
        height_ratio = float(box_height) / max(float(image_height), 1.0)
        area_ratio = max(0.0, width_ratio * height_ratio)

        if x_ratio < self.horizontal_left_boundary:
            horizontal = "left"
        elif x_ratio > self.horizontal_right_boundary:
            horizontal = "right"
        else:
            horizontal = "center"

        if y_ratio < self.vertical_upper_boundary:
            vertical = "upper"
        elif y_ratio > self.vertical_lower_boundary:
            vertical = "lower"
        else:
            vertical = "middle"

        proxy_source, near_threshold, medium_threshold = (
            self._proximity_thresholds(detection["class_name"])
        )
        proxy_value = (
            height_ratio
            if proxy_source == "bbox_height_ratio"
            else area_ratio
        )

        if proxy_value >= near_threshold:
            proximity = "near"
            proximity_rank = 3
        elif proxy_value >= medium_threshold:
            proximity = "medium"
            proximity_rank = 2
        else:
            proximity = "far"
            proximity_rank = 1

        offset_x_norm = max(-1.0, min(1.0, 2.0 * x_ratio - 1.0))
        offset_y_norm = max(-1.0, min(1.0, 2.0 * y_ratio - 1.0))

        enriched["side"] = horizontal
        enriched["vertical_region"] = vertical
        enriched["proximity_hint"] = proximity
        enriched["spatial"] = {
            "horizontal": horizontal,
            "vertical": vertical,
            "offset_x_norm": round(offset_x_norm, 4),
            "offset_y_norm": round(offset_y_norm, 4),
            "center_x_ratio": round(x_ratio, 4),
            "center_y_ratio": round(y_ratio, 4),
            "width_ratio": round(width_ratio, 4),
            "height_ratio": round(height_ratio, 4),
            "area_ratio": round(area_ratio, 5),
            "proximity": proximity,
            "proximity_rank": proximity_rank,
            "proximity_source": proxy_source,
            "metric_distance_available": False,
        }
        return enriched

    @staticmethod
    def _target_priority(detection: Dict[str, Any]) -> float:
        class_name = detection["class_name"]
        semantic_group = detection.get("semantic_group", "")

        if class_name == "person":
            return 100.0
        if class_name in {"cat", "dog"}:
            return 80.0
        if semantic_group == "container":
            return 45.0
        if semantic_group in {"computer", "computer_accessory"}:
            return 35.0
        return 20.0

    def _select_primary_target(
        self,
        detections: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if not detections:
            return {}

        visible_now = [
            detection
            for detection in detections
            if int(detection.get("missed_frames", 0)) == 0
        ]
        candidates = visible_now or detections

        best = None
        best_score = float("-inf")

        for detection in candidates:
            spatial = detection.get("spatial", {})
            center_score = 1.0 - min(
                1.0,
                abs(float(spatial.get("offset_x_norm", 0.0))),
            )
            area_score = min(
                1.0,
                float(spatial.get("area_ratio", 0.0)) * 8.0,
            )
            confidence_score = float(detection.get("confidence", 0.0))

            score = (
                self._target_priority(detection)
                + 10.0 * center_score
                + 5.0 * confidence_score
                + 5.0 * area_score
            )

            if score > best_score:
                best_score = score
                best = detection

        if best is None:
            return {}

        return {
            "track_id": best.get("track_id"),
            "class_name": best.get("class_name"),
            "semantic_group": best.get("semantic_group"),
            "confidence": best.get("confidence"),
            "bbox_xyxy": best.get("bbox_xyxy"),
            "center_xy": best.get("center_xy"),
            "velocity_xy": best.get("velocity_xy"),
            "side": best.get("side"),
            "vertical_region": best.get("vertical_region"),
            "proximity_hint": best.get("proximity_hint"),
            "distance_m": best.get("distance_m"),
            "distance_valid": best.get("distance_valid", False),
            "depth_confidence": best.get("depth_confidence", "none"),
            "depth_status": best.get("depth_status"),
            "spatial": best.get("spatial"),
            "selection_score": round(best_score, 3),
            "selection_policy": (
                "person_then_pet_then_salient_object"
            ),
        }

    @staticmethod
    def _spatial_label(detection: Dict[str, Any]) -> str:
        horizontal_ru = {
            "left": "слева",
            "center": "по центру",
            "right": "справа",
        }
        proximity_ru = {
            "near": "визуально близко",
            "medium": "на среднем плане",
            "far": "на дальнем плане",
        }
        if detection.get("distance_valid"):
            distance_text = (
                f"примерно {float(detection['distance_m']):.1f} м"
            )
        else:
            distance_text = proximity_ru.get(
                detection.get("proximity_hint"),
                "",
            )

        return (
            f"{detection['class_name']} "
            f"{horizontal_ru.get(detection.get('side'), '')}, "
            f"{distance_text}"
        ).strip(" ,")

    def _run_warm_up(self, frame) -> None:
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

    def _warm_up(self, frame) -> None:
        if self._warmed_up:
            return

        started = time.perf_counter()
        try:
            self._run_warm_up(frame)
        except Exception as primary_exc:
            primary_error = self._model_error_text(primary_exc)
            fallback_path = self._normalize_model_path(
                self.fallback_model_path
            )
            active_path = self._normalize_model_path(
                self.active_model_path
            )

            if (
                self.fallback_used
                or not fallback_path
                or fallback_path == active_path
            ):
                raise RuntimeError(
                    "YOLO warm-up failed and no unused fallback is available: "
                    f"active={active_path}, reason={primary_error}"
                ) from primary_exc

            self.get_logger().warning(
                "YOLO warm-up failed; activating fallback model: "
                f"active={active_path}, fallback={fallback_path}, "
                f"reason={primary_error}"
            )
            try:
                self.model = self._activate_model(
                    fallback_path,
                    fallback_reason=primary_error,
                )
                self._run_warm_up(frame)
            except Exception as fallback_exc:
                fallback_error = self._model_error_text(fallback_exc)
                raise RuntimeError(
                    "YOLO warm-up failed for both requested and fallback models: "
                    f"requested_error={primary_error}; "
                    f"fallback_error={fallback_error}"
                ) from fallback_exc

        self._warmed_up = True
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self.get_logger().info(
            "YOLO warm-up completed: "
            f"backend={self.inference_backend}, "
            f"active_model={self.active_model_path}, "
            f"fallback_used={self.fallback_used}, "
            f"elapsed_ms={elapsed_ms:.1f}"
        )

    def _update_tracks(
        self,
        detections: List[Dict[str, Any]],
        scene_motion: bool,
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
            track["consecutive_hits"] += 1
            track["missed_frames"] = 0
            track["last_seen"] = time.time()
            track["required_hits"] = max(
                int(track["required_hits"]),
                self._required_hits_for(
                    track["class_name"],
                    scene_motion,
                ),
            )

            if (
                track["consecutive_hits"] >= track["required_hits"]
                or self._can_confirm_immediately(
                    track["class_name"],
                    float(track["confidence"]),
                )
            ):
                track["confirmed"] = True

            unmatched_track_ids.remove(track_id)
            unmatched_detection_indexes.remove(detection_index)

        for detection_index in unmatched_detection_indexes:
            detection = detections[detection_index]
            track_id = self.next_track_id
            self.next_track_id += 1

            required_hits = self._required_hits_for(
                detection["class_name"],
                scene_motion,
            )
            confirmed = self._can_confirm_immediately(
                detection["class_name"],
                float(detection["confidence"]),
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
                "consecutive_hits": 1,
                "required_hits": required_hits,
                "missed_frames": 0,
                "confirmed": confirmed,
                "first_seen": time.time(),
                "last_seen": time.time(),
            }

        for track_id in unmatched_track_ids:
            track = self.tracks[track_id]
            track["missed_frames"] += 1
            track["consecutive_hits"] = 0
            track["velocity_xy"] = [
                round(float(value) * 0.85, 2)
                for value in track["velocity_xy"]
            ]

        expired = [
            track_id
            for track_id, track in self.tracks.items()
            if track["missed_frames"]
            > self._max_missed_for(track["class_name"])
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
                "consecutive_hits": track["consecutive_hits"],
                "required_hits": track["required_hits"],
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
        frame_stamp_sec = self.last_frame_stamp_sec
        frame_id = self.last_frame_id
        depth_frame = self._nearest_disparity(frame_stamp_sec)
        height, width = frame.shape[:2]
        motion_ratio = self._compute_motion_ratio(frame)
        scene_motion = motion_ratio >= self.motion_threshold

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
                base_threshold = self._threshold_for(class_name)
                threshold = base_threshold
                if scene_motion and self._is_noisy_class(class_name):
                    threshold = min(
                        0.95,
                        threshold + self.motion_conf_boost,
                    )
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
                    "base_threshold": round(base_threshold, 4),
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
        self._update_tracks(accepted_raw, scene_motion)
        detections = [
            self._with_metric_depth(
                self._with_spatial(detection, width, height),
                depth_frame,
                width,
                height,
            )
            for detection in self._stable_detections()
        ]
        primary_target = self._select_primary_target(detections)

        visible_for_nearest = [
            detection
            for detection in detections
            if int(detection.get("missed_frames", 0)) == 0
        ]
        nearest_hint = {}
        metric_candidates = [
            detection
            for detection in visible_for_nearest
            if detection.get("distance_valid")
        ]
        if metric_candidates:
            nearest = min(
                metric_candidates,
                key=lambda detection: float(detection["distance_m"]),
            )
            nearest_hint = {
                "track_id": nearest.get("track_id"),
                "class_name": nearest.get("class_name"),
                "side": nearest.get("side"),
                "proximity_hint": nearest.get("proximity_hint"),
                "distance_m": nearest.get("distance_m"),
                "distance_valid": True,
                "depth_confidence": nearest.get("depth_confidence"),
                "spatial": nearest.get("spatial"),
                "estimated_only": False,
            }
        elif visible_for_nearest:
            nearest = max(
                visible_for_nearest,
                key=lambda detection: (
                    int(
                        detection.get("spatial", {}).get(
                            "proximity_rank",
                            0,
                        )
                    ),
                    float(
                        detection.get("spatial", {}).get(
                            "area_ratio",
                            0.0,
                        )
                    ),
                ),
            )
            nearest_hint = {
                "track_id": nearest.get("track_id"),
                "class_name": nearest.get("class_name"),
                "side": nearest.get("side"),
                "proximity_hint": nearest.get("proximity_hint"),
                "distance_m": None,
                "distance_valid": False,
                "spatial": nearest.get("spatial"),
                "estimated_only": True,
            }

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
        spatial_items = [
            self._spatial_label(detection)
            for detection in detections[:6]
        ]
        spatial_summary = (
            "; ".join(spatial_items)
            if spatial_items
            else "Подтверждённых объектов нет."
        )

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
            "frame_id": frame_id or "camera_right_optical_frame",
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
            "spatial_summary": spatial_summary,
            "primary_target": primary_target,
            "nearest_hint": nearest_hint,
            "spatial_estimation": {
                "metric_distance_available": any(
                    detection.get("distance_valid")
                    for detection in detections
                ),
                "metric_detection_count": sum(
                    1
                    for detection in detections
                    if detection.get("distance_valid")
                ),
                "distance_method": "stereo_disparity_inner_bbox_median",
                "depth_topic": self.depth_topic,
                "depth_frame_available": depth_frame is not None,
                "depth_age_ms": (
                    round(float(depth_frame.get("age_ms", 0.0)), 1)
                    if depth_frame is not None
                    else None
                ),
                "fallback_method": "class_aware_bbox_size_proxy",
                "note": (
                    "distance_m is metric stereo depth when valid; "
                    "near/medium/far remains a visual fallback"
                ),
            },
            "detector": {
                "model": self.active_model_path,
                "requested_model": self.requested_model_path,
                "active_model": self.active_model_path,
                "backend": self.inference_backend,
                "fallback_used": self.fallback_used,
                "fallback_reason": self.fallback_reason,
                "type": "ultralytics_yolo",
                "version": "l640-v5-depth",
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
                "motion_ratio": round(motion_ratio, 4),
                "scene_motion": scene_motion,
                "tentative_tracks": sum(
                    1
                    for track in self.tracks.values()
                    if not track["confirmed"]
                ),
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

        target_msg = String()
        target_msg.data = json.dumps(
            primary_target,
            ensure_ascii=False,
        )
        self.target_pub.publish(target_msg)

        debug_msg = String()
        debug_msg.data = (
            f"latency_ms={total_latency_ms:.1f}, "
            f"person_count={person_count}, "
            f"objects={[obj['class_name'] for obj in objects[:12]]}, "
            f"counts={counts}, "
            f"tracks={len(self.tracks)}, "
            f"tentative={sum(1 for track in self.tracks.values() if not track['confirmed'])}, "
            f"motion={motion_ratio:.3f}, "
            f"scene_motion={scene_motion}, "
            f"suppressed_duplicates={suppressed_duplicates}, "
            f"depth_valid={sum(1 for detection in detections if detection.get('distance_valid'))}/"
            f"{len(detections)}, "
            f"depth_age_ms={(round(float(depth_frame.get('age_ms', 0.0)), 1) if depth_frame is not None else 'none')}, "
            f"target={primary_target.get('class_name', 'none')}#"
            f"{primary_target.get('track_id', '-')}:"
            f"{primary_target.get('side', '-')}/"
            f"{primary_target.get('proximity_hint', '-')}/"
            f"{primary_target.get('distance_m', '-') }m"
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
