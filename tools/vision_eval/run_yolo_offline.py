#!/usr/bin/env python3
"""Run the current Ultralytics YOLO baseline on the exact saved evaluation images.

This avoids associating a captured image with a stale live /perception/state_json
message. Metrics are calculated from inference on each saved left PNG.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from ultralytics import YOLO

ALLOWED_CLASSES = {
    "person",
    "cat",
    "dog",
    "cup",
    "bottle",
    "cell phone",
    "laptop",
    "chair",
}


def safe_div(a: int, b: int) -> float:
    return a / b if b else 0.0


def load_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSON at line {line_number}: {exc}") from exc
    return rows


def find_default_model(project_root: Path) -> Path | None:
    candidates = [
        project_root / "models" / "yolo11s.pt",
        project_root / "models" / "yolo11n.pt",
        project_root / "ros2_ws" / "yolo11s.pt",
        project_root / "ros2_ws" / "yolo11n.pt",
        project_root / "yolo11s.pt",
        project_root / "yolo11n.pt",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def parse_args() -> argparse.Namespace:
    project_root = Path(os.environ.get("AI_ROBOT_ROOT", str(Path.home() / "ai_robot")))
    default_manifest = project_root / "data" / "vision_eval" / "manifest.jsonl"
    default_model = find_default_model(project_root)

    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", nargs="?", type=Path, default=default_manifest)
    parser.add_argument("--model", type=Path, default=default_model)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--max-det", type=int, default=20)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--exclude-scenario",
        action="append",
        default=[],
        help="Scenario to omit; may be passed multiple times",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = args.manifest.expanduser().resolve()
    if not manifest.exists():
        raise SystemExit(f"Manifest not found: {manifest}")
    if args.model is None:
        raise SystemExit(
            "YOLO model not found automatically. Pass --model /path/to/model.pt"
        )
    model_path = args.model.expanduser().resolve()
    if not model_path.exists():
        raise SystemExit(f"YOLO model not found: {model_path}")

    records = load_manifest(manifest)
    excluded = set(args.exclude_scenario)
    records = [r for r in records if str(r.get("scenario", "")) not in excluded]
    if not records:
        raise SystemExit("No records left after filtering")

    output = args.output or manifest.with_name("yolo_offline_baseline.csv")
    output = output.expanduser().resolve()

    print(f"Model: {model_path}")
    print(f"Manifest: {manifest}")
    print(f"Samples selected: {len(records)}")
    if excluded:
        print(f"Excluded scenarios: {sorted(excluded)}")

    model = YOLO(str(model_path))
    class_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "support": 0})
    person_exact = 0
    inference_times: list[float] = []
    csv_rows: list[dict[str, Any]] = []

    for index, record in enumerate(records, 1):
        scenario = str(record.get("scenario", ""))
        sample_id = str(record.get("sample_id", ""))
        ground_truth = record.get("ground_truth", {})
        expected_people = int(ground_truth.get("expected_people", 0))
        expected = set(ground_truth.get("expected_objects") or []) | set(
            ground_truth.get("expected_pets") or []
        )

        rel_image = Path(record.get("files", {}).get("left", ""))
        image_path = manifest.parent / rel_image
        if not image_path.exists():
            raise RuntimeError(f"Image not found for {sample_id}: {image_path}")

        started = time.perf_counter()
        result = model.predict(
            source=str(image_path),
            imgsz=args.imgsz,
            conf=args.conf,
            max_det=args.max_det,
            verbose=False,
            stream=False,
        )[0]
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        inference_times.append(elapsed_ms)

        detections: list[dict[str, Any]] = []
        detected_people = 0
        detected_objects: set[str] = set()
        names = result.names
        boxes = result.boxes

        if boxes is not None:
            for box in boxes:
                cls_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                class_name = str(names.get(cls_id, str(cls_id)))
                if class_name not in ALLOWED_CLASSES:
                    continue
                min_conf = 0.5 if class_name == "person" else 0.6
                if confidence < min_conf:
                    continue
                xyxy = [int(v) for v in box.xyxy[0].tolist()]
                detections.append(
                    {
                        "class_name": class_name,
                        "confidence": round(confidence, 4),
                        "bbox_xyxy": xyxy,
                    }
                )
                if class_name == "person":
                    detected_people += 1
                else:
                    detected_objects.add(class_name)

        if detected_people == expected_people:
            person_exact += 1

        for class_name in sorted(expected | detected_objects):
            stats = class_stats[class_name]
            if class_name in expected:
                stats["support"] += 1
            if class_name in expected and class_name in detected_objects:
                stats["tp"] += 1
            elif class_name in detected_objects and class_name not in expected:
                stats["fp"] += 1
            elif class_name in expected and class_name not in detected_objects:
                stats["fn"] += 1

        people_ok = detected_people == expected_people
        objects_ok = detected_objects == expected
        csv_rows.append(
            {
                "sample_id": sample_id,
                "scenario": scenario,
                "image": str(rel_image),
                "expected_people": expected_people,
                "detected_people": detected_people,
                "expected_objects": ",".join(sorted(expected)),
                "detected_objects": ",".join(sorted(detected_objects)),
                "people_ok": people_ok,
                "objects_ok": objects_ok,
                "sample_ok": people_ok and objects_ok,
                "inference_ms": f"{elapsed_ms:.2f}",
                "detections_json": json.dumps(detections, ensure_ascii=False),
            }
        )
        print(
            f"[{index:03d}/{len(records):03d}] {scenario}: "
            f"people {detected_people}/{expected_people}, "
            f"objects {sorted(detected_objects)}/{sorted(expected)}, "
            f"{elapsed_ms:.1f} ms"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    print("\nExact-image YOLO baseline")
    print(f"Samples: {len(records)}")
    print(
        "Person count exact accuracy: "
        f"{person_exact}/{len(records)} = {safe_div(person_exact, len(records)):.1%}"
    )
    print("\nObject presence metrics:")
    print(f"{'class':20} {'support':>7} {'precision':>10} {'recall':>8} {'tp':>4} {'fp':>4} {'fn':>4}")
    for class_name, stats in sorted(class_stats.items()):
        precision = safe_div(stats["tp"], stats["tp"] + stats["fp"])
        recall = safe_div(stats["tp"], stats["tp"] + stats["fn"])
        print(
            f"{class_name:20} {stats['support']:7d} {precision:10.1%} "
            f"{recall:8.1%} {stats['tp']:4d} {stats['fp']:4d} {stats['fn']:4d}"
        )
    mean_ms = sum(inference_times) / len(inference_times)
    sorted_ms = sorted(inference_times)
    p95_index = max(0, int(round(0.95 * len(sorted_ms) + 0.499999)) - 1)
    print(f"\nInference mean: {mean_ms:.1f} ms")
    print(f"Inference p95: {sorted_ms[p95_index]:.1f} ms")
    print(f"Per-sample CSV: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
