#!/usr/bin/env python3
"""Create a labeled montage of exact-image YOLO baseline failures."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import cv2
import numpy as np


def truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def main() -> int:
    project_root = Path(os.environ.get("AI_ROBOT_ROOT", str(Path.home() / "ai_robot")))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=Path,
        default=project_root / "data" / "vision_eval" / "yolo_offline_baseline.csv",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=project_root / "data" / "vision_eval",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "data" / "vision_eval" / "yolo_failures_montage.jpg",
    )
    parser.add_argument("--columns", type=int, default=2)
    args = parser.parse_args()

    with args.csv.expanduser().open("r", encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if not truthy(row["sample_ok"])]
    if not rows:
        print("No failures found")
        return 0

    tiles: list[np.ndarray] = []
    tile_w, tile_h = 720, 480
    for row in rows:
        image_path = args.dataset.expanduser() / row["image"]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue

        try:
            detections = json.loads(row.get("detections_json", "[]"))
        except json.JSONDecodeError:
            detections = []
        for det in detections:
            x1, y1, x2, y2 = det["bbox_xyxy"]
            label = f"{det['class_name']} {det['confidence']:.2f}"
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), 2)
            cv2.putText(
                image,
                label,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        canvas = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
        header_h = 100
        available_h = tile_h - header_h
        scale = min(tile_w / image.shape[1], available_h / image.shape[0])
        resized = cv2.resize(
            image,
            (max(1, int(image.shape[1] * scale)), max(1, int(image.shape[0] * scale))),
        )
        x = (tile_w - resized.shape[1]) // 2
        y = header_h + (available_h - resized.shape[0]) // 2
        canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized

        lines = [
            f"{row['scenario']} | {row['sample_id'][-12:]}",
            f"people exp/det: {row['expected_people']}/{row['detected_people']}",
            f"objects exp: {row['expected_objects'] or '-'}",
            f"objects det: {row['detected_objects'] or '-'}",
        ]
        for i, text in enumerate(lines):
            cv2.putText(
                canvas,
                text,
                (10, 22 + i * 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        tiles.append(canvas)

    columns = max(1, args.columns)
    rows_count = (len(tiles) + columns - 1) // columns
    blank = np.zeros_like(tiles[0])
    while len(tiles) < rows_count * columns:
        tiles.append(blank.copy())
    montage_rows = []
    for i in range(rows_count):
        montage_rows.append(np.hstack(tiles[i * columns : (i + 1) * columns]))
    montage = np.vstack(montage_rows)
    args.output.expanduser().parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output.expanduser()), montage)
    print(f"Failures: {len(rows)}")
    print(f"Montage: {args.output.expanduser()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
