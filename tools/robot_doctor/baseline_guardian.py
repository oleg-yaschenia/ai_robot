#!/usr/bin/env python3
"""
Baseline Guardian v1 for the AI robot project.

Creates a versionable baseline from a robot-doctor snapshot and compares
later snapshots against it.

The tool is read-only except for writing:
- a requested baseline JSON;
- comparison reports outside the repository, unless another path is supplied.

It does not modify ROS, restart services, use sudo, or access actuators.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TOOL_VERSION = "baseline-guardian-v1"


@dataclass
class Comparison:
    check_id: str
    status: str
    component: str
    summary: str
    baseline: Any
    current: Any


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_snapshot(path: str | None, output_root: Path) -> Path:
    if path:
        snapshot_dir = Path(path).expanduser().resolve()
    else:
        latest_file = output_root / "latest.txt"
        if not latest_file.exists():
            raise RuntimeError(
                f"No latest snapshot pointer found: {latest_file}"
            )
        snapshot_dir = Path(
            latest_file.read_text(encoding="utf-8").strip()
        ).expanduser().resolve()

    snapshot_json = snapshot_dir / "snapshot.json"
    if not snapshot_json.exists():
        raise RuntimeError(f"snapshot.json not found in {snapshot_dir}")
    return snapshot_dir


def command_map(snapshot: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(item.get("name")): item
        for item in snapshot.get("commands", [])
        if item.get("name")
    }


def command_output(
    commands: dict[str, dict[str, Any]],
    name: str,
    stream: str = "stdout",
) -> str:
    item = commands.get(name)
    if not item:
        return ""
    value = item.get(f"{stream}_file")
    if not value:
        return ""
    path = Path(value)
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def parse_pip_show(text: str) -> dict[str, str]:
    packages: dict[str, str] = {}
    current_name: str | None = None

    for line in text.splitlines():
        if line.startswith("Name: "):
            current_name = line.removeprefix("Name: ").strip().lower()
        elif line.startswith("Version: ") and current_name:
            packages[current_name] = line.removeprefix(
                "Version: "
            ).strip()
            current_name = None

    return packages


def parse_memory_available(text: str) -> int | None:
    for line in text.splitlines():
        if line.startswith("Mem:"):
            fields = line.split()
            if len(fields) >= 7:
                try:
                    return int(fields[6])
                except ValueError:
                    return None
    return None


def parse_disk_usage(text: str) -> int | None:
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        return None
    match = re.search(r"(\d+)%", lines[-1])
    return int(match.group(1)) if match else None


def normalized_file_hashes(snapshot: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in snapshot.get("important_files", []):
        path = item.get("path")
        sha256 = item.get("sha256")
        if path and sha256:
            result[str(path)] = str(sha256)
    return result


def read_git_value(
    commands: dict[str, dict[str, Any]], name: str
) -> str | None:
    value = command_output(commands, name).strip()
    return value or None


def extract_baseline(
    snapshot_dir: Path,
    name: str,
) -> dict[str, Any]:
    snapshot = load_json(snapshot_dir / "snapshot.json")
    commands = command_map(snapshot)

    health_path = snapshot_dir / "health_report.json"
    health = load_json(health_path) if health_path.exists() else {}

    nodes = sorted(
        line.strip()
        for line in command_output(commands, "ros_nodes").splitlines()
        if line.strip().startswith("/")
    )

    package_prefixes: dict[str, str] = {}
    for command_name in sorted(commands):
        if command_name.startswith("ros_pkg_"):
            package = command_name.removeprefix("ros_pkg_")
            package_prefixes[package] = command_output(
                commands, command_name
            ).strip()

    baseline = {
        "schema_version": 1,
        "tool_version": TOOL_VERSION,
        "baseline_name": name,
        "created_at_utc": iso_now(),
        "source_snapshot": str(snapshot_dir),
        "source_collected_at_utc": snapshot.get("collected_at_utc"),
        "host": snapshot.get("hostname"),
        "repo_root": snapshot.get("repo_root"),
        "git": {
            "branch": read_git_value(commands, "git_branch"),
            "commit": read_git_value(commands, "git_commit"),
            "describe": read_git_value(commands, "git_describe"),
        },
        "health": {
            "overall_status": health.get("overall_status"),
            "counts": health.get("counts", {}),
        },
        "required_nodes": nodes,
        "important_file_sha256": normalized_file_hashes(snapshot),
        "python_packages": parse_pip_show(
            command_output(commands, "python_packages")
        ),
        "ros_package_prefixes": package_prefixes,
        "runtime_reference": {
            "memory_available_bytes": parse_memory_available(
                command_output(commands, "memory_bytes")
            ),
            "root_disk_used_percent": parse_disk_usage(
                command_output(commands, "disk_root_bytes")
            ),
            "home_disk_used_percent": parse_disk_usage(
                command_output(commands, "disk_home_bytes")
            ),
        },
        "comparison_policy": {
            "missing_required_node": "ERROR",
            "important_file_hash_change": "ERROR",
            "python_package_version_change": "WARNING",
            "ros_package_prefix_change": "WARNING",
            "memory_drop_warning_fraction": 0.25,
            "memory_drop_error_fraction": 0.50,
            "disk_growth_warning_percentage_points": 10,
            "disk_growth_error_percentage_points": 25,
        },
    }
    return baseline


def add_comparison(
    items: list[Comparison],
    check_id: str,
    status: str,
    component: str,
    summary: str,
    baseline: Any,
    current: Any,
) -> None:
    items.append(
        Comparison(
            check_id=check_id,
            status=status,
            component=component,
            summary=summary,
            baseline=baseline,
            current=current,
        )
    )


def compare(
    baseline: dict[str, Any],
    snapshot_dir: Path,
) -> dict[str, Any]:
    snapshot = load_json(snapshot_dir / "snapshot.json")
    commands = command_map(snapshot)
    comparisons: list[Comparison] = []
    policy = baseline.get("comparison_policy", {})

    current_nodes = sorted(
        line.strip()
        for line in command_output(commands, "ros_nodes").splitlines()
        if line.strip().startswith("/")
    )
    required_nodes = baseline.get("required_nodes", [])
    missing_nodes = sorted(set(required_nodes) - set(current_nodes))

    add_comparison(
        comparisons,
        "required_nodes",
        "ERROR" if missing_nodes else "OK",
        "ROS 2 graph",
        (
            "Required ROS nodes are missing"
            if missing_nodes
            else "All baseline ROS nodes are present"
        ),
        required_nodes,
        missing_nodes if missing_nodes else current_nodes,
    )

    baseline_hashes = baseline.get("important_file_sha256", {})
    current_hashes = normalized_file_hashes(snapshot)

    for path, expected_hash in sorted(baseline_hashes.items()):
        current_hash = current_hashes.get(path)
        if current_hash is None:
            status = "ERROR"
            summary = "Baseline file is missing from current snapshot"
        elif current_hash != expected_hash:
            status = "ERROR"
            summary = "Protected file hash changed"
        else:
            status = "OK"
            summary = "Protected file hash matches baseline"

        add_comparison(
            comparisons,
            f"file:{path}",
            status,
            "Protected configuration",
            summary,
            expected_hash,
            current_hash,
        )

    baseline_packages = baseline.get("python_packages", {})
    current_packages = parse_pip_show(
        command_output(commands, "python_packages")
    )

    for package, expected_version in sorted(
        baseline_packages.items()
    ):
        current_version = current_packages.get(package)
        status = "OK" if current_version == expected_version else "WARNING"
        add_comparison(
            comparisons,
            f"python_package:{package}",
            status,
            "Python packages",
            (
                "Package version matches baseline"
                if status == "OK"
                else "Package version changed or is unavailable"
            ),
            expected_version,
            current_version,
        )

    baseline_prefixes = baseline.get("ros_package_prefixes", {})
    current_prefixes: dict[str, str] = {}
    for command_name in commands:
        if command_name.startswith("ros_pkg_"):
            package = command_name.removeprefix("ros_pkg_")
            current_prefixes[package] = command_output(
                commands, command_name
            ).strip()

    for package, expected_prefix in sorted(
        baseline_prefixes.items()
    ):
        current_prefix = current_prefixes.get(package)
        status = "OK" if current_prefix == expected_prefix else "WARNING"
        add_comparison(
            comparisons,
            f"ros_package:{package}",
            status,
            "ROS package prefix",
            (
                "Package prefix matches baseline"
                if status == "OK"
                else "Package prefix changed or is unavailable"
            ),
            expected_prefix,
            current_prefix,
        )

    baseline_runtime = baseline.get("runtime_reference", {})
    current_memory = parse_memory_available(
        command_output(commands, "memory_bytes")
    )
    baseline_memory = baseline_runtime.get("memory_available_bytes")

    if baseline_memory and current_memory is not None:
        drop_fraction = max(
            0.0, (baseline_memory - current_memory) / baseline_memory
        )
        warning_fraction = float(
            policy.get("memory_drop_warning_fraction", 0.25)
        )
        error_fraction = float(
            policy.get("memory_drop_error_fraction", 0.50)
        )

        if drop_fraction >= error_fraction:
            status = "ERROR"
        elif drop_fraction >= warning_fraction:
            status = "WARNING"
        else:
            status = "OK"

        add_comparison(
            comparisons,
            "runtime:memory_available",
            status,
            "Jetson memory",
            (
                f"Available memory changed by "
                f"{drop_fraction * 100:.1f}% below baseline"
            ),
            baseline_memory,
            current_memory,
        )

    for command_name, baseline_key, label in (
        ("disk_root_bytes", "root_disk_used_percent", "Root disk"),
        ("disk_home_bytes", "home_disk_used_percent", "Home disk"),
    ):
        baseline_percent = baseline_runtime.get(baseline_key)
        current_percent = parse_disk_usage(
            command_output(commands, command_name)
        )
        if baseline_percent is None or current_percent is None:
            continue

        growth = current_percent - baseline_percent
        warning_points = int(
            policy.get("disk_growth_warning_percentage_points", 10)
        )
        error_points = int(
            policy.get("disk_growth_error_percentage_points", 25)
        )

        if growth >= error_points:
            status = "ERROR"
        elif growth >= warning_points:
            status = "WARNING"
        else:
            status = "OK"

        add_comparison(
            comparisons,
            f"runtime:{baseline_key}",
            status,
            "Disk",
            f"{label} usage changed by {growth:+d} percentage points",
            baseline_percent,
            current_percent,
        )

    rank = {"OK": 0, "WARNING": 1, "ERROR": 2}
    overall = max(
        (item.status for item in comparisons),
        key=lambda value: rank[value],
        default="ERROR",
    )

    return {
        "schema_version": 1,
        "tool_version": TOOL_VERSION,
        "compared_at_utc": iso_now(),
        "baseline_name": baseline.get("baseline_name"),
        "baseline_created_at_utc": baseline.get("created_at_utc"),
        "baseline_source_snapshot": baseline.get("source_snapshot"),
        "current_snapshot": str(snapshot_dir),
        "overall_status": overall,
        "counts": {
            "OK": sum(item.status == "OK" for item in comparisons),
            "WARNING": sum(
                item.status == "WARNING" for item in comparisons
            ),
            "ERROR": sum(
                item.status == "ERROR" for item in comparisons
            ),
        },
        "comparisons": [asdict(item) for item in comparisons],
    }


def write_reports(
    report: dict[str, Any],
    report_dir: Path,
) -> tuple[Path, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "baseline_comparison.json"
    md_path = report_dir / "baseline_comparison.md"

    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Baseline comparison",
        "",
        f"- Baseline: `{report.get('baseline_name')}`",
        f"- Overall status: **{report.get('overall_status')}**",
        f"- Current snapshot: `{report.get('current_snapshot')}`",
        (
            f"- Checks: OK={report['counts']['OK']}, "
            f"WARNING={report['counts']['WARNING']}, "
            f"ERROR={report['counts']['ERROR']}"
        ),
        "",
        "| Status | Component | Check | Baseline | Current |",
        "|---|---|---|---|---|",
    ]

    for item in report.get("comparisons", []):
        baseline_value = json.dumps(
            item.get("baseline"), ensure_ascii=False
        )
        current_value = json.dumps(
            item.get("current"), ensure_ascii=False
        )
        if len(baseline_value) > 90:
            baseline_value = baseline_value[:87] + "..."
        if len(current_value) > 90:
            current_value = current_value[:87] + "..."

        lines.append(
            f"| {item['status']} | {item['component']} | "
            f"{item['summary']} | `{baseline_value}` | "
            f"`{current_value}` |"
        )

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def parse_args() -> argparse.Namespace:
    default_snapshots = str(
        Path.home() / "ai_robot_artifacts/robot_doctor/snapshots"
    )
    default_reports = str(
        Path.home() / "ai_robot_artifacts/robot_doctor/comparisons"
    )

    parser = argparse.ArgumentParser(
        description="Create and compare robot-doctor baselines."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    create_parser = subparsers.add_parser(
        "create", help="Create a baseline from a snapshot."
    )
    create_parser.add_argument("--snapshot")
    create_parser.add_argument(
        "--snapshot-root", default=default_snapshots
    )
    create_parser.add_argument("--name", required=True)
    create_parser.add_argument("--output", required=True)
    create_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing baseline.",
    )

    compare_parser = subparsers.add_parser(
        "compare", help="Compare a snapshot with a baseline."
    )
    compare_parser.add_argument("--snapshot")
    compare_parser.add_argument(
        "--snapshot-root", default=default_snapshots
    )
    compare_parser.add_argument("--baseline", required=True)
    compare_parser.add_argument(
        "--report-dir",
        default=default_reports,
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        snapshot_root = Path(
            args.snapshot_root
        ).expanduser().resolve()
        snapshot_dir = resolve_snapshot(args.snapshot, snapshot_root)

        if args.command == "create":
            output = Path(args.output).expanduser().resolve()
            if output.exists() and not args.force:
                raise RuntimeError(
                    f"Baseline already exists: {output}. "
                    "Use --force only after deliberate review."
                )

            baseline = extract_baseline(snapshot_dir, args.name)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                json.dumps(
                    baseline, ensure_ascii=False, indent=2
                ),
                encoding="utf-8",
            )
            print(output)
            return 0

        if args.command == "compare":
            baseline_path = Path(
                args.baseline
            ).expanduser().resolve()
            baseline = load_json(baseline_path)
            report = compare(baseline, snapshot_dir)

            report_root = Path(
                args.report_dir
            ).expanduser().resolve()
            stamp = datetime.now(timezone.utc).strftime(
                "%Y%m%dT%H%M%SZ"
            )
            report_dir = (
                report_root
                / str(report.get("baseline_name", "baseline"))
                / stamp
            )

            json_path, md_path = write_reports(report, report_dir)
            print(md_path)
            print(json_path)
            return 0 if report["overall_status"] == "OK" else 2

    except Exception as exc:
        print(f"baseline-guardian failed: {exc}", file=sys.stderr)
        return 1

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
