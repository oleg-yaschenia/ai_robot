#!/usr/bin/env python3
"""Model-independent perception entity contract v1.

This module contains no ROS dependencies. It converts the current perception
state into a stable entity array that downstream components can consume without
depending on a specific detector implementation.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional


SCHEMA_NAME = "robot_perception_entities"
SCHEMA_VERSION = 1

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


def _optional_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _numeric_list(
    value: Any,
    expected_length: int,
    *,
    integers: bool,
) -> Optional[List[Any]]:
    if not isinstance(value, (list, tuple)):
        return None
    if len(value) != expected_length:
        return None

    converted: List[Any] = []
    for item in value:
        try:
            converted.append(int(item) if integers else float(item))
        except (TypeError, ValueError):
            return None
    return converted


def _semantic_status(detection: Dict[str, Any]) -> str:
    explicit = str(detection.get("semantic_status", "")).strip().lower()
    if explicit in {"raw", "tentative", "confirmed", "rejected", "unknown"}:
        return explicit

    if "confirmed" in detection:
        return "confirmed" if bool(detection.get("confirmed")) else "tentative"

    # Current state_json persons/objects arrays contain accepted tracks.
    return "confirmed"


def _source_detector(state: Dict[str, Any]) -> Dict[str, Any]:
    detector = state.get("detector")
    if not isinstance(detector, dict):
        detector = {}

    return {
        "type": detector.get("type", "unknown"),
        "backend": detector.get("backend", "unknown"),
        "model": (
            detector.get("active_model")
            or detector.get("model")
            or "unknown"
        ),
        "version": detector.get("version"),
        "device": detector.get("device"),
    }


def _entity_from_detection(
    detection: Dict[str, Any],
    *,
    entity_type: str,
    index: int,
    detector: Dict[str, Any],
    frame_id: Optional[str],
) -> Dict[str, Any]:
    label = str(
        detection.get("label")
        or detection.get("class_name")
        or "unknown"
    )
    track_id = _optional_int(detection.get("track_id"))

    explicit_entity_id = str(detection.get("entity_id", "")).strip()
    if explicit_entity_id:
        entity_id = explicit_entity_id
    elif track_id is not None:
        entity_id = f"track_{track_id}"
    else:
        entity_id = f"{entity_type}_{index}"

    bbox = _numeric_list(
        detection.get("bbox_xyxy"),
        4,
        integers=True,
    )
    center = _numeric_list(
        detection.get("center_xy"),
        2,
        integers=True,
    )
    size = _numeric_list(
        detection.get("size_wh"),
        2,
        integers=True,
    )
    velocity = _numeric_list(
        detection.get("velocity_xy"),
        2,
        integers=False,
    )

    distance_m = _optional_float(detection.get("distance_m"))
    distance_valid = bool(
        detection.get("distance_valid", False)
        and distance_m is not None
    )

    spatial = detection.get("spatial")
    if not isinstance(spatial, dict):
        spatial = {}

    position_3d = detection.get("position_3d")
    if not isinstance(position_3d, dict):
        position_3d = None

    semantic_confidence = _optional_float(
        detection.get(
            "semantic_confidence",
            detection.get("confidence"),
        )
    )

    return {
        "entity_id": entity_id,
        "track_id": track_id,
        "entity_type": entity_type,
        "label": label,
        "semantic_group": (
            detection.get("semantic_group")
            or SEMANTIC_GROUPS.get(label, "other")
        ),
        "semantic_confidence": semantic_confidence,
        "semantic_status": _semantic_status(detection),
        "bbox": {
            "format": "xyxy",
            "coordinates": bbox,
        },
        "image_geometry": {
            "center_xy": center,
            "size_wh": size,
        },
        "distance": {
            "meters": distance_m if distance_valid else None,
            "valid": distance_valid,
            "confidence": detection.get("depth_confidence", "none"),
            "status": detection.get("depth_status", "unavailable"),
            "source": detection.get("depth_source"),
            "samples": _optional_int(detection.get("depth_samples")),
            "valid_ratio": _optional_float(
                detection.get("depth_valid_ratio")
            ),
        },
        "position_3d": position_3d,
        "spatial": {
            "horizontal": (
                spatial.get("horizontal")
                or detection.get("side")
            ),
            "vertical": spatial.get("vertical"),
            "proximity": (
                spatial.get("proximity")
                or detection.get("proximity_hint")
            ),
            "metric_distance_available": bool(
                spatial.get(
                    "metric_distance_available",
                    distance_valid,
                )
            ),
        },
        "tracking": {
            "hits": _optional_int(detection.get("hits")),
            "consecutive_hits": _optional_int(
                detection.get("consecutive_hits")
            ),
            "missed_frames": _optional_int(
                detection.get("missed_frames")
            ),
            "velocity_xy": velocity,
        },
        "source": {
            "detector_type": detector.get("type"),
            "detector_backend": detector.get("backend"),
            "detector_model": detector.get("model"),
            "detector_version": detector.get("version"),
            "frame_id": frame_id,
        },
    }


def build_entity_array(
    state: Dict[str, Any],
    *,
    source_topic: str = "/perception/state_json",
) -> Dict[str, Any]:
    """Convert a perception state dictionary into EntityArray v1."""
    if not isinstance(state, dict):
        raise TypeError("perception state must be a dictionary")

    persons = state.get("persons")
    objects = state.get("objects")
    if not isinstance(persons, list):
        persons = []
    if not isinstance(objects, list):
        objects = []

    frame_id_value = (
        state.get("frame_id")
        or state.get("image_frame_id")
        or state.get("camera_frame_id")
    )
    frame_id = (
        str(frame_id_value)
        if frame_id_value not in {None, ""}
        else None
    )

    detector = _source_detector(state)
    entities: List[Dict[str, Any]] = []

    for index, item in enumerate(persons):
        if isinstance(item, dict):
            entities.append(
                _entity_from_detection(
                    item,
                    entity_type="person",
                    index=index,
                    detector=detector,
                    frame_id=frame_id,
                )
            )

    for index, item in enumerate(objects):
        if isinstance(item, dict):
            entities.append(
                _entity_from_detection(
                    item,
                    entity_type="object",
                    index=index,
                    detector=detector,
                    frame_id=frame_id,
                )
            )

    counts: Dict[str, int] = {}
    for entity in entities:
        label = str(entity.get("label", "unknown"))
        counts[label] = counts.get(label, 0) + 1

    image_size = state.get("image_size")
    if not isinstance(image_size, dict):
        image_size = {}

    source_timestamp = state.get("timestamp")
    if source_timestamp is None:
        source_timestamp = state.get("source_timestamp")

    return {
        "schema": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "timestamp": time.time(),
        "source_timestamp": source_timestamp,
        "frame_id": frame_id,
        "image_size": {
            "width": _optional_int(image_size.get("width")),
            "height": _optional_int(image_size.get("height")),
        },
        "entity_count": len(entities),
        "counts": counts,
        "entities": entities,
        "source": {
            "topic": source_topic,
            "detector": detector,
        },
    }


def validate_entity_array(message: Dict[str, Any]) -> List[str]:
    """Return contract validation errors. Empty means valid."""
    errors: List[str] = []

    if not isinstance(message, dict):
        return ["message must be a dictionary"]

    if message.get("schema") != SCHEMA_NAME:
        errors.append(f"schema must be {SCHEMA_NAME!r}")
    if message.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}")
    if not isinstance(message.get("entities"), list):
        errors.append("entities must be a list")
        return errors
    if message.get("entity_count") != len(message["entities"]):
        errors.append("entity_count does not match entities length")

    seen_ids = set()
    for index, entity in enumerate(message["entities"]):
        prefix = f"entities[{index}]"
        if not isinstance(entity, dict):
            errors.append(f"{prefix} must be a dictionary")
            continue

        entity_id = entity.get("entity_id")
        if not isinstance(entity_id, str) or not entity_id:
            errors.append(
                f"{prefix}.entity_id must be a non-empty string"
            )
        elif entity_id in seen_ids:
            errors.append(
                f"{prefix}.entity_id is duplicated: {entity_id}"
            )
        else:
            seen_ids.add(entity_id)

        if entity.get("entity_type") not in {"person", "object"}:
            errors.append(
                f"{prefix}.entity_type must be person or object"
            )

        bbox = entity.get("bbox")
        if not isinstance(bbox, dict):
            errors.append(f"{prefix}.bbox must be a dictionary")
        elif bbox.get("format") != "xyxy":
            errors.append(f"{prefix}.bbox.format must be xyxy")

        distance = entity.get("distance")
        if not isinstance(distance, dict):
            errors.append(
                f"{prefix}.distance must be a dictionary"
            )
        elif distance.get("valid") and distance.get("meters") is None:
            errors.append(
                f"{prefix}.distance.meters is required "
                "when valid=true"
            )

    return errors
