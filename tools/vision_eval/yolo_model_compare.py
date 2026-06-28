#!/usr/bin/env python3
from pathlib import Path
import argparse
import json
import statistics
import time

import torch
from ultralytics import YOLO


def get_left_image(record: dict, dataset_root: Path) -> Path:
    rel = (
        record.get("files", {}).get("left")
        or record.get("images", {}).get("left")
        or record.get("left_image")
    )
    if not rel:
        raise KeyError(f"No left image path in sample {record.get('sample_id', '<unknown>')}")
    return dataset_root / rel


def percentile(values, q):
    ordered = sorted(values)
    if not ordered:
        return 0.0
    idx = max(0, min(len(ordered) - 1, int(round(q * len(ordered))) - 1))
    return ordered[idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["yolo11s.pt", "yolo11m.pt"])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.01)
    args = parser.parse_args()

    root = Path.home() / "ai_robot"
    dataset = root / "data/vision_eval"
    manifest = dataset / "manifest.jsonl"

    records = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    if not records:
        raise SystemExit("Manifest is empty")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")

    print("CUDA:", torch.version.cuda)
    print("GPU:", torch.cuda.get_device_name(0))
    print("Samples:", len(records))
    print("imgsz:", args.imgsz)
    print("conf:", args.conf)

    for model_name in args.models:
        model_path = Path(model_name)
        if not model_path.is_absolute():
            local_candidate = root / "ros2_ws" / model_name
            model_path = local_candidate if local_candidate.exists() else Path(model_name)

        print(f"\n=== MODEL {model_name} ===")
        model = YOLO(str(model_path))

        first_image = get_left_image(records[0], dataset)
        for _ in range(3):
            model.predict(
                source=str(first_image),
                imgsz=args.imgsz,
                conf=args.conf,
                iou=0.7,
                max_det=300,
                device=0,
                verbose=False,
            )

        times_ms = []
        summary = {
            "person": [],
            "cat": [],
            "cup": [],
            "chair": [],
        }

        for index, record in enumerate(records, start=1):
            image_path = get_left_image(record, dataset)
            scenario = record.get("scenario", "unknown")

            started = time.perf_counter()
            result = model.predict(
                source=str(image_path),
                imgsz=args.imgsz,
                conf=args.conf,
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
                    cls_id = int(box.cls.item())
                    cls_name = result.names[cls_id]
                    score = float(box.conf.item())
                    best[cls_name] = max(best.get(cls_name, 0.0), score)

            for key in summary:
                summary[key].append(best.get(key, 0.0))

            print(
                f"[{index:02d}/{len(records):02d}] "
                f"{scenario:30s} "
                f"person={best.get('person', 0.0):.3f} "
                f"cat={best.get('cat', 0.0):.3f} "
                f"cup={best.get('cup', 0.0):.3f} "
                f"chair={best.get('chair', 0.0):.3f} "
                f"time={elapsed_ms:.1f}ms"
            )

        print("\nSummary")
        print(f"Warm mean ms: {statistics.mean(times_ms):.1f}")
        print(f"Warm p95 ms: {percentile(times_ms, 0.95):.1f}")
        print(f"Warm max ms: {max(times_ms):.1f}")

        for key, values in summary.items():
            nonzero = [v for v in values if v > 0]
            print(
                f"{key:7s}: detected {len(nonzero)}/{len(values)}, "
                f"best={max(values):.3f}, "
                f"mean_nonzero={(statistics.mean(nonzero) if nonzero else 0.0):.3f}"
            )


if __name__ == "__main__":
    main()
