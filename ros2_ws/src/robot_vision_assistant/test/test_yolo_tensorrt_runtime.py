import json
from pathlib import Path

import numpy as np

from robot_vision_assistant.yolo_tensorrt_runtime import (
    decode_yolo_output,
    letterbox_bgr,
    normalize_class_names,
    read_ultralytics_engine,
)


def test_read_ultralytics_engine_strips_metadata(tmp_path: Path):
    metadata = {
        "names": {"0": "person", "1": "chair"},
        "imgsz": [640, 640],
    }
    metadata_bytes = json.dumps(metadata).encode("utf-8")
    plan = b"native-tensorrt-plan"
    engine_path = tmp_path / "model.engine"
    engine_path.write_bytes(
        len(metadata_bytes).to_bytes(4, "little", signed=True)
        + metadata_bytes
        + plan
    )

    loaded_metadata, loaded_plan = read_ultralytics_engine(
        str(engine_path)
    )

    assert loaded_metadata == metadata
    assert loaded_plan == plan


def test_normalize_class_names_supports_dict_and_list():
    assert normalize_class_names({"0": "person", 1: "chair"}) == {
        0: "person",
        1: "chair",
    }
    assert normalize_class_names(["person", "chair"]) == {
        0: "person",
        1: "chair",
    }


def test_letterbox_bgr_returns_nchw_float_tensor():
    frame = np.zeros((360, 640, 3), dtype=np.uint8)

    tensor, ratio, pad = letterbox_bgr(frame, 640)

    assert tensor.shape == (1, 3, 640, 640)
    assert tensor.dtype == np.float32
    assert ratio == 1.0
    assert pad == (0.0, 140.0)
    assert 0.0 <= float(tensor.min()) <= float(tensor.max()) <= 1.0


def test_decode_yolo_output_runs_class_aware_nms_and_unletterboxes():
    # Ten anchors, two classes: [cx, cy, w, h, class0, class1].
    prediction = np.zeros((10, 6), dtype=np.float32)
    prediction[0] = [320, 320, 200, 300, 0.90, 0.05]
    prediction[1] = [325, 325, 205, 305, 0.80, 0.05]
    prediction[2] = [100, 200, 80, 100, 0.05, 0.75]

    detections = decode_yolo_output(
        prediction,
        original_shape=(360, 640),
        ratio=1.0,
        pad=(0.0, 140.0),
        confidence_threshold=0.10,
        iou_threshold=0.45,
        max_det=100,
    )

    assert len(detections) == 2
    assert detections[0]["class_id"] == 0
    assert detections[0]["confidence"] > 0.89
    assert detections[0]["bbox_xyxy"] == [220, 30, 420, 330]
    assert detections[1]["class_id"] == 1
    assert detections[1]["bbox_xyxy"] == [60, 10, 140, 110]
