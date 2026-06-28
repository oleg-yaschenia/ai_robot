#!/usr/bin/env python3
from pathlib import Path
import json
import statistics
import time

import torch
from ultralytics import YOLO

ROOT = Path.home() / "ai_robot"
DATASET = ROOT / "data/vision_eval"
MANIFEST = DATASET / "manifest.jsonl"
MODEL_PATH = ROOT / "ros2_ws/yolo11s.pt"

if not MODEL_PATH.exists():
    raise SystemExit(f"Model not found: {MODEL_PATH}")

if not MANIFEST.exists():
    raise SystemExit(f"Manifest not found: {MANIFEST}")

records = [
    json.loads(line)
    for line in MANIFEST.read_text(encoding="utf-8").splitlines()
    if line.strip()
]

if not records:
    raise SystemExit("Manifest is empty")

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available")

def left_image_from_record(record: dict) -> Path:
    relative = (
        record.get("files", {}).get("left")
        or record.get("images", {}).get("left")
        or record.get("left_image")
    )
    if not relative:
        raise KeyError(
            f"No left image path in record {record.get('sample_id', '<unknown>')}"
        )
    return DATASET / relative

model = YOLO(str(MODEL_PATH))

for imgsz in (640, 1280):
    print(f"\n=== IMGSZ {imgsz} ===")

    first_image = left_image_from_record(records[0])

    # Warm-up
    for _ in range(3):
        model.predict(
            source=str(first_image),
            imgsz=imgsz,
            conf=0.01,
            iou=0.7,
            max_det=300,
            device=0,
            verbose=False,
        )

    times_ms = []

    for index, record in enumerate(records, start=1):
        image_path = left_image_from_record(record)
        scenario = record.get("scenario", "unknown")

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        started = time.perf_counter()

        result = model.predict(
            source=str(image_path),
            imgsz=imgsz,
            conf=0.01,
            iou=0.7,
            max_det=300,
            device=0,
            verbose=False,
        )[0]

        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        times_ms.append(elapsed_ms)

        best = {}

        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls.item())
                class_name = result.names[class_id]
                confidence = float(box.conf.item())
                best[class_name] = max(best.get(class_name, 0.0), confidence)

        print(
            f"[{index:02d}/{len(records):02d}] "
            f"{scenario:30s} "
            f"person={best.get('person', 0.0):.3f} "
            f"cat={best.get('cat', 0.0):.3f} "
            f"cup={best.get('cup', 0.0):.3f} "
            f"chair={best.get('chair', 0.0):.3f}"
        )

    ordered = sorted(times_ms)
    p95_index = max(
        0,
        min(len(ordered) - 1, int(round(0.95 * len(ordered))) - 1),
    )

    print(f"Warm mean ms: {statistics.mean(times_ms):.1f}")
    print(f"Warm p95 ms: {ordered[p95_index]:.1f}")
    print(f"Warm max ms: {max(times_ms):.1f}")
