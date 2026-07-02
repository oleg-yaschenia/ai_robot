#!/usr/bin/env python3
"""Evaluate Assistant Router against a labeled JSONL dataset."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

from robot_vision_assistant.assistant_router import (
    ROUTE_MODES,
    classify_route,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--report-root",
        default="~/ai_robot_artifacts/assistant_router_eval",
    )
    parser.add_argument("--min-accuracy", type=float, default=0.90)
    parser.add_argument(
        "--max-false-action-rate",
        type=float,
        default=0.02,
        help=(
            "Maximum fraction of non-action cases classified as "
            "action/action+report."
        ),
    )
    parser.add_argument(
        "--max-missed-action-rate",
        type=float,
        default=0.10,
        help=(
            "Maximum fraction of action/action+report cases classified as "
            "chat/visual_chat."
        ),
    )
    parser.add_argument(
        "--max-low-confidence-rate",
        type=float,
        default=0.40,
    )
    parser.add_argument(
        "--low-confidence-threshold",
        type=float,
        default=0.80,
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{line_number}: invalid JSON: {exc}"
                ) from exc

            required = {"id", "query", "expected_mode", "group"}
            missing = sorted(required - set(row))
            if missing:
                raise ValueError(
                    f"{path}:{line_number}: missing fields: {missing}"
                )
            if row["expected_mode"] not in ROUTE_MODES:
                raise ValueError(
                    f"{path}:{line_number}: invalid expected_mode "
                    f"{row['expected_mode']!r}"
                )
            rows.append(row)

    if not rows:
        raise ValueError("dataset is empty")
    return rows


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def make_markdown(report: Dict[str, Any]) -> str:
    lines = [
        "# Assistant Router Evaluation",
        "",
        f"- Status: **{report['status']}**",
        f"- Dataset: `{report['dataset']}`",
        f"- Cases: {report['totals']['cases']}",
        f"- Accuracy: {report['metrics']['accuracy']:.3f}",
        (
            "- False action rate: "
            f"{report['metrics']['false_action_rate']:.3f}"
        ),
        (
            "- Missed action rate: "
            f"{report['metrics']['missed_action_rate']:.3f}"
        ),
        (
            "- Low-confidence rate: "
            f"{report['metrics']['low_confidence_rate']:.3f}"
        ),
        "",
        "## Per-class metrics",
        "",
        "| Mode | Precision | Recall | F1 | Support |",
        "|---|---:|---:|---:|---:|",
    ]

    for mode in ROUTE_MODES:
        item = report["per_class"][mode]
        lines.append(
            f"| {mode} | {item['precision']:.3f} | "
            f"{item['recall']:.3f} | {item['f1']:.3f} | "
            f"{item['support']} |"
        )

    lines.extend([
        "",
        "## Confusion matrix",
        "",
        "| Expected \\\\ Predicted | "
        + " | ".join(ROUTE_MODES)
        + " |",
        "|---|" + "---:|" * len(ROUTE_MODES),
    ])

    for expected in ROUTE_MODES:
        values = [
            str(report["confusion_matrix"][expected][predicted])
            for predicted in ROUTE_MODES
        ]
        lines.append(
            f"| {expected} | " + " | ".join(values) + " |"
        )

    lines.extend([
        "",
        "## Failed gates",
        "",
    ])
    if report["failed_gates"]:
        for item in report["failed_gates"]:
            lines.append(f"- {item}")
    else:
        lines.append("- None")

    lines.extend([
        "",
        "## Misclassified cases",
        "",
    ])
    if report["misclassified"]:
        for item in report["misclassified"]:
            lines.append(
                f"- `{item['id']}` expected `{item['expected_mode']}`, "
                f"got `{item['predicted_mode']}` "
                f"(confidence={item['confidence']:.2f}): "
                f"{item['query']}"
            )
    else:
        lines.append("- None")

    lines.extend([
        "",
        "## False action cases",
        "",
    ])
    if report["false_action_cases"]:
        for item in report["false_action_cases"]:
            lines.append(
                f"- `{item['id']}` expected `{item['expected_mode']}`, "
                f"got `{item['predicted_mode']}`: {item['query']}"
            )
    else:
        lines.append("- None")

    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    dataset_path = Path(args.dataset).expanduser().resolve()
    rows = load_jsonl(dataset_path)

    modes = list(ROUTE_MODES)
    confusion = {
        expected: {predicted: 0 for predicted in modes}
        for expected in modes
    }

    results: List[Dict[str, Any]] = []
    correct = 0
    false_action_cases: List[Dict[str, Any]] = []
    missed_action_cases: List[Dict[str, Any]] = []
    low_confidence_cases: List[Dict[str, Any]] = []

    for row in rows:
        decision = classify_route(row["query"])
        predicted = decision["mode"]
        expected = row["expected_mode"]
        confidence = float(decision["confidence"])

        confusion[expected][predicted] += 1
        is_correct = predicted == expected
        correct += int(is_correct)

        result = {
            **row,
            "predicted_mode": predicted,
            "confidence": confidence,
            "reason": decision["reason"],
            "matched_rules": decision["matched_rules"],
            "correct": is_correct,
        }
        results.append(result)

        expected_is_action = expected in {"action", "action+report"}
        predicted_is_action = predicted in {"action", "action+report"}

        if not expected_is_action and predicted_is_action:
            false_action_cases.append(result)
        if expected_is_action and not predicted_is_action:
            missed_action_cases.append(result)
        if confidence < args.low_confidence_threshold:
            low_confidence_cases.append(result)

    accuracy = safe_div(correct, len(rows))
    non_action_total = sum(
        1 for row in rows
        if row["expected_mode"] not in {"action", "action+report"}
    )
    action_total = len(rows) - non_action_total

    false_action_rate = safe_div(
        len(false_action_cases),
        non_action_total,
    )
    missed_action_rate = safe_div(
        len(missed_action_cases),
        action_total,
    )
    low_confidence_rate = safe_div(
        len(low_confidence_cases),
        len(rows),
    )

    per_class: Dict[str, Dict[str, Any]] = {}
    for mode in modes:
        tp = confusion[mode][mode]
        fp = sum(
            confusion[expected][mode]
            for expected in modes
            if expected != mode
        )
        fn = sum(
            confusion[mode][predicted]
            for predicted in modes
            if predicted != mode
        )
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)
        per_class[mode] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(confusion[mode].values()),
        }

    failed_gates: List[str] = []
    if accuracy < args.min_accuracy:
        failed_gates.append(
            f"accuracy {accuracy:.3f} < {args.min_accuracy:.3f}"
        )
    if false_action_rate > args.max_false_action_rate:
        failed_gates.append(
            "false_action_rate "
            f"{false_action_rate:.3f} > "
            f"{args.max_false_action_rate:.3f}"
        )
    if missed_action_rate > args.max_missed_action_rate:
        failed_gates.append(
            "missed_action_rate "
            f"{missed_action_rate:.3f} > "
            f"{args.max_missed_action_rate:.3f}"
        )
    if low_confidence_rate > args.max_low_confidence_rate:
        failed_gates.append(
            "low_confidence_rate "
            f"{low_confidence_rate:.3f} > "
            f"{args.max_low_confidence_rate:.3f}"
        )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_dir = (
        Path(args.report_root).expanduser().resolve() / stamp
    )
    report_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "timestamp_utc": stamp,
        "status": "PASS" if not failed_gates else "FAIL",
        "dataset": str(dataset_path),
        "thresholds": {
            "min_accuracy": args.min_accuracy,
            "max_false_action_rate": args.max_false_action_rate,
            "max_missed_action_rate": args.max_missed_action_rate,
            "max_low_confidence_rate": args.max_low_confidence_rate,
            "low_confidence_threshold": (
                args.low_confidence_threshold
            ),
        },
        "totals": {
            "cases": len(rows),
            "correct": correct,
            "incorrect": len(rows) - correct,
            "non_action_cases": non_action_total,
            "action_cases": action_total,
        },
        "metrics": {
            "accuracy": accuracy,
            "false_action_rate": false_action_rate,
            "missed_action_rate": missed_action_rate,
            "low_confidence_rate": low_confidence_rate,
        },
        "per_class": per_class,
        "confusion_matrix": confusion,
        "failed_gates": failed_gates,
        "misclassified": [
            item for item in results if not item["correct"]
        ],
        "false_action_cases": false_action_cases,
        "missed_action_cases": missed_action_cases,
        "low_confidence_cases": low_confidence_cases,
        "results": results,
    }

    json_path = report_dir / "router_eval_report.json"
    md_path = report_dir / "router_eval_report.md"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    md_path.write_text(
        make_markdown(report),
        encoding="utf-8",
    )

    print(f"Status: {report['status']}")
    print(f"Cases: {len(rows)}")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"False action rate: {false_action_rate:.3f}")
    print(f"Missed action rate: {missed_action_rate:.3f}")
    print(f"Low-confidence rate: {low_confidence_rate:.3f}")
    print(f"Misclassified: {len(report['misclassified'])}")
    print(f"Report: {md_path}")

    return 0 if not failed_gates else 2


if __name__ == "__main__":
    raise SystemExit(main())
