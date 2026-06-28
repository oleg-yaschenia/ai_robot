#!/usr/bin/env python3
"""Calculate simple presence/count baseline metrics from manifest.jsonl.

This is intentionally not a bounding-box mAP evaluator. Version 1 measures
whether the current perception stack notices expected classes and people at all.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_manifest(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSON at line {line_number}: {exc}") from exc
    return records


def detected_classes(record: dict[str, Any]) -> list[str]:
    state = record.get("baseline", {}).get("perception_state") or {}
    detections = state.get("detections") or []
    result: list[str] = []
    for detection in detections:
        name = detection.get("class_name")
        if isinstance(name, str):
            result.append(name)
    return result


def detected_people_count(record: dict[str, Any]) -> int:
    state = record.get("baseline", {}).get("perception_state") or {}
    persons = state.get("persons")
    if isinstance(persons, list):
        return len(persons)
    counts = state.get("counts") or {}
    value = counts.get("person", 0)
    return int(value) if isinstance(value, (int, float)) else 0


def safe_div(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "manifest",
        type=Path,
        nargs="?",
        default=Path.home() / "ai_robot" / "data" / "vision_eval" / "manifest.jsonl",
    )
    parser.add_argument("--csv", type=Path, default=None)
    args = parser.parse_args()

    records = load_manifest(args.manifest.expanduser())
    if not records:
        raise SystemExit("Manifest is empty")

    per_class = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "support": 0})
    rows: list[dict[str, Any]] = []
    person_exact = 0
    missing_perception = 0

    for record in records:
        ground_truth = record.get("ground_truth", {})
        expected_objects = set(ground_truth.get("expected_objects") or [])
        expected_pets = set(ground_truth.get("expected_pets") or [])
        expected = expected_objects | expected_pets
        detected = set(detected_classes(record)) - {"person"}

        perception_state = record.get("baseline", {}).get("perception_state")
        if not perception_state:
            missing_perception += 1

        for class_name in sorted(expected | detected):
            stats = per_class[class_name]
            if class_name in expected:
                stats["support"] += 1
            if class_name in expected and class_name in detected:
                stats["tp"] += 1
            elif class_name in detected and class_name not in expected:
                stats["fp"] += 1
            elif class_name in expected and class_name not in detected:
                stats["fn"] += 1

        expected_people = int(ground_truth.get("expected_people", 0))
        actual_people = detected_people_count(record)
        if expected_people == actual_people:
            person_exact += 1

        rows.append(
            {
                "sample_id": record.get("sample_id", ""),
                "scenario": record.get("scenario", ""),
                "expected_people": expected_people,
                "detected_people": actual_people,
                "expected_objects": ",".join(sorted(expected)),
                "detected_objects": ",".join(sorted(detected)),
                "perception_present": bool(perception_state),
            }
        )

    print(f"Samples: {len(records)}")
    print(f"Missing perception state: {missing_perception}")
    print(
        "Person count exact accuracy: "
        f"{person_exact}/{len(records)} = {safe_div(person_exact, len(records)):.1%}"
    )
    print("\nObject presence metrics:")
    print(f"{'class':20} {'support':>7} {'precision':>10} {'recall':>8} {'tp':>4} {'fp':>4} {'fn':>4}")
    for class_name, stats in sorted(per_class.items()):
        precision = safe_div(stats["tp"], stats["tp"] + stats["fp"])
        recall = safe_div(stats["tp"], stats["tp"] + stats["fn"])
        print(
            f"{class_name:20} {stats['support']:7d} {precision:10.1%} "
            f"{recall:8.1%} {stats['tp']:4d} {stats['fp']:4d} {stats['fn']:4d}"
        )

    csv_path = args.csv or args.manifest.with_name("baseline_presence.csv")
    with csv_path.expanduser().open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nPer-sample report: {csv_path.expanduser()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
