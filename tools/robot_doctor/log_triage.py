#!/usr/bin/env python3
"""
Log Triage v1 for the AI robot project.

Read-only deterministic log analyzer. It scans recent engineering and ROS logs,
groups repeated issues, classifies severity, and proposes the next safe checks.

Safety boundaries:
- no sudo;
- no service restarts;
- no ROS parameter changes;
- no firmware flashing;
- no motor, servo, UART command, or actuator access;
- no automatic source-code or configuration changes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


TOOL_VERSION = "log-triage-v1"
TEXT_SUFFIXES = {
    ".log",
    ".txt",
    ".out",
    ".err",
    ".stdout",
    ".stderr",
}


@dataclass
class IssueGroup:
    issue_id: str
    severity: str
    category: str
    normalized_message: str
    count: int
    first_source: str
    first_line: int
    samples: list[str]
    sources: list[str]
    ignored: bool
    ignore_reason: str | None
    recommended_checks: list[str]


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_pointer(pointer: Path) -> Path | None:
    if not pointer.exists():
        return None
    try:
        value = pointer.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not value:
        return None
    path = Path(value).expanduser()
    return path.resolve() if path.exists() else None


def resolve_symlink_or_path(path: Path) -> Path | None:
    try:
        resolved = path.resolve()
    except OSError:
        return None
    return resolved if resolved.exists() else None


def detect_repo_root(explicit: str | None) -> Path:
    if explicit:
        root = Path(explicit).expanduser().resolve()
        if not (root / ".git").exists():
            raise RuntimeError(f"Not a Git repository: {root}")
        return root

    script_path = Path(__file__).resolve()
    for parent in [script_path.parent, *script_path.parents]:
        if (parent / ".git").exists() and (parent / "ros2_ws").exists():
            return parent

    cwd = Path.cwd().resolve()
    for parent in [cwd, *cwd.parents]:
        if (parent / ".git").exists() and (parent / "ros2_ws").exists():
            return parent

    raise RuntimeError("Cannot detect repository root. Use --repo-root.")


def discover_sources(
    repo_root: Path,
    explicit_paths: list[str],
) -> list[Path]:
    sources: list[Path] = []

    for raw in explicit_paths:
        path = Path(raw).expanduser().resolve()
        if path.exists():
            sources.append(path)

    home = Path.home()

    regression_latest = resolve_pointer(
        home / "ai_robot_artifacts/robot_doctor/regressions/latest.txt"
    )
    if regression_latest:
        commands_dir = regression_latest / "commands"
        if commands_dir.exists():
            sources.append(commands_dir)

    snapshot_latest = resolve_pointer(
        home / "ai_robot_artifacts/robot_doctor/snapshots/latest.txt"
    )
    if snapshot_latest:
        commands_dir = snapshot_latest / "commands"
        if commands_dir.exists():
            sources.append(commands_dir)

    for candidate in (
        repo_root / "ros2_ws/log/latest_build",
        repo_root / "ros2_ws/log/latest_test",
        repo_root / "ros2_ws/log/latest",
        home / ".ros/log/latest",
    ):
        resolved = resolve_symlink_or_path(candidate)
        if resolved:
            sources.append(resolved)

    unique: list[Path] = []
    seen: set[str] = set()
    for source in sources:
        key = str(source)
        if key not in seen:
            seen.add(key)
            unique.append(source)
    return unique


def iter_log_files(
    sources: Iterable[Path],
    max_files: int,
    max_file_bytes: int,
) -> list[Path]:
    files: list[Path] = []

    for source in sources:
        if source.is_file():
            candidates = [source]
        else:
            candidates = [
                path
                for path in source.rglob("*")
                if path.is_file()
                and (
                    path.suffix.lower() in TEXT_SUFFIXES
                    or path.name.endswith(".stdout.txt")
                    or path.name.endswith(".stderr.txt")
                )
            ]

        for path in sorted(
            candidates,
            key=lambda item: item.stat().st_mtime if item.exists() else 0,
            reverse=True,
        ):
            try:
                size = path.stat().st_size
            except OSError:
                continue
            if size == 0 or size > max_file_bytes:
                continue
            files.append(path)
            if len(files) >= max_files:
                return files

    return files


def compile_patterns(items: list[str]) -> list[re.Pattern[str]]:
    return [re.compile(item, re.IGNORECASE) for item in items]


def matches_any(patterns: list[re.Pattern[str]], text: str) -> bool:
    return any(pattern.search(text) for pattern in patterns)


def strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text)


def normalize_message(line: str) -> str:
    text = strip_ansi(line).strip()
    text = re.sub(
        r"^\[[A-Z]+\]\s+\[[^\]]+\]\s*(\[[^\]]+\])?\s*",
        "",
        text,
    )
    text = re.sub(
        r"^\d{4}-\d{2}-\d{2}[T ][0-9:.+\-Z]+\s*",
        "",
        text,
    )
    text = re.sub(r"\b0x[0-9a-fA-F]+\b", "<HEX>", text)
    text = re.sub(r"\bpid[=: ]+\d+\b", "pid=<N>", text, flags=re.I)
    text = re.sub(r"\b\d+\.\d+\.\d+\.\d+\b", "<IP>", text)
    text = re.sub(r"/home/[^/\s]+", "~", text)
    text = re.sub(r"\b\d+(?:\.\d+)?(?:ms|s|MB|GB|KiB|MiB|GiB|%)\b", "<N>", text)
    text = re.sub(r"\b\d{5,}\b", "<N>", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:500]


def classify(
    line: str,
    config: dict[str, Any],
) -> tuple[str, str]:
    categories = config["category_patterns"]
    for category, patterns in categories.items():
        if matches_any(compile_patterns(patterns), line):
            severity = config["category_severity"].get(category, "WARNING")
            return severity, category

    severity_patterns = {
        key: compile_patterns(value)
        for key, value in config["severity_patterns"].items()
    }

    for severity in ("ERROR", "WARNING"):
        if matches_any(severity_patterns.get(severity, []), line):
            return severity, "general"

    return "INFO", "general"


def recommended_checks(category: str, severity: str) -> list[str]:
    mapping = {
        "python_exception": [
            "Open the first traceback in the primary source file.",
            "Identify the earliest project-file frame, not the final wrapper error.",
            "Run py_compile or the smallest reproducer before changing configuration.",
        ],
        "ros_transport": [
            "Check that the expected ROS nodes and topics exist.",
            "Inspect QoS compatibility and publisher/subscriber counts.",
            "Verify that the source node is alive before restarting anything.",
        ],
        "camera": [
            "Check camera node presence and image topic publication.",
            "Verify sensor-id and physical left/right mapping.",
            "Inspect the earliest Argus/GStreamer error in the same log.",
        ],
        "cuda_memory": [
            "Record RAM, swap and tegrastats before retrying.",
            "Check whether multiple model processes are loaded.",
            "Reduce concurrent model load before changing model parameters.",
        ],
        "serial": [
            "Check /dev/ttyTHS1 presence and dialout permissions.",
            "Verify that only one process owns the serial device.",
            "Do not send actuator commands during diagnosis.",
        ],
        "build": [
            "Inspect the first compiler or packaging error.",
            "Rebuild only the affected package before a full workspace build.",
            "Confirm the active ROS underlay and workspace setup order.",
        ],
        "timeout": [
            "Determine whether the timeout is intentional or a stalled component.",
            "Check the component's last successful message and current process state.",
            "Avoid increasing the timeout until the blocking cause is known.",
        ],
        "general": [
            "Inspect the earliest occurrence and its preceding context.",
            "Confirm whether the issue reproduces in the latest regression run.",
            "Change only one variable and rerun the smallest relevant check.",
        ],
    }
    checks = mapping.get(category, mapping["general"]).copy()
    if severity == "ERROR":
        checks.insert(0, "Stop further configuration changes until this error is explained.")
    return checks


def analyze_files(
    files: list[Path],
    config: dict[str, Any],
    context_lines: int,
) -> tuple[list[IssueGroup], dict[str, Any]]:
    ignore_patterns = compile_patterns(config["ignore_patterns"])
    include_patterns = compile_patterns(config["include_patterns"])

    groups: dict[tuple[str, str, str, bool], dict[str, Any]] = {}
    scanned_lines = 0
    matched_lines = 0
    ignored_lines = 0

    for path in files:
        try:
            lines = path.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines()
        except OSError:
            continue

        for index, raw in enumerate(lines, start=1):
            scanned_lines += 1
            line = strip_ansi(raw).strip()
            if not line:
                continue

            severity, category = classify(line, config)
            should_include = severity in {"ERROR", "WARNING"} or matches_any(
                include_patterns, line
            )
            if not should_include:
                continue

            matched_lines += 1
            ignored = matches_any(ignore_patterns, line)
            if ignored:
                ignored_lines += 1

            normalized = normalize_message(line)
            if not normalized:
                continue

            key = (severity, category, normalized, ignored)
            group = groups.setdefault(
                key,
                {
                    "severity": severity,
                    "category": category,
                    "normalized_message": normalized,
                    "count": 0,
                    "first_source": str(path),
                    "first_line": index,
                    "samples": [],
                    "sources": set(),
                    "ignored": ignored,
                    "ignore_reason": (
                        "Matched configured benign/expected pattern"
                        if ignored
                        else None
                    ),
                },
            )
            group["count"] += 1
            group["sources"].add(str(path))

            if len(group["samples"]) < 3:
                start = max(0, index - 1 - context_lines)
                end = min(len(lines), index + context_lines)
                sample = "\n".join(lines[start:end])
                group["samples"].append(sample[:2000])

    result: list[IssueGroup] = []
    for group in groups.values():
        result.append(
            IssueGroup(
                issue_id=hashlib.sha1(
                    (
                        group["severity"]
                        + "|"
                        + group["category"]
                        + "|"
                        + group["normalized_message"]
                    ).encode("utf-8")
                ).hexdigest()[:12],
                severity=group["severity"],
                category=group["category"],
                normalized_message=group["normalized_message"],
                count=group["count"],
                first_source=group["first_source"],
                first_line=group["first_line"],
                samples=group["samples"],
                sources=sorted(group["sources"]),
                ignored=group["ignored"],
                ignore_reason=group["ignore_reason"],
                recommended_checks=recommended_checks(
                    group["category"], group["severity"]
                ),
            )
        )

    severity_rank = {"ERROR": 0, "WARNING": 1, "INFO": 2}
    result.sort(
        key=lambda item: (
            item.ignored,
            severity_rank.get(item.severity, 3),
            -item.count,
            item.first_source,
            item.first_line,
        )
    )

    stats = {
        "files_scanned": len(files),
        "lines_scanned": scanned_lines,
        "matched_lines": matched_lines,
        "ignored_lines": ignored_lines,
        "active_groups": sum(not item.ignored for item in result),
        "ignored_groups": sum(item.ignored for item in result),
    }
    return result, stats


def overall_status(groups: list[IssueGroup]) -> str:
    active = [item for item in groups if not item.ignored]
    if any(item.severity == "ERROR" for item in active):
        return "ERROR"
    if any(item.severity == "WARNING" for item in active):
        return "WARNING"
    return "OK"


def write_reports(
    run_dir: Path,
    sources: list[Path],
    files: list[Path],
    groups: list[IssueGroup],
    stats: dict[str, Any],
) -> tuple[Path, Path]:
    status = overall_status(groups)
    active = [item for item in groups if not item.ignored]
    primary = active[0] if active else None

    report = {
        "schema_version": 1,
        "tool_version": TOOL_VERSION,
        "created_at_utc": iso_now(),
        "overall_status": status,
        "sources": [str(item) for item in sources],
        "files": [str(item) for item in files],
        "stats": stats,
        "primary_issue_id": primary.issue_id if primary else None,
        "groups": [asdict(item) for item in groups],
    }

    json_path = run_dir / "log_triage_report.json"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Log Triage report",
        "",
        f"- Overall status: **{status}**",
        f"- Created: `{report['created_at_utc']}`",
        f"- Files scanned: {stats['files_scanned']}",
        f"- Lines scanned: {stats['lines_scanned']}",
        f"- Active issue groups: {stats['active_groups']}",
        f"- Ignored expected groups: {stats['ignored_groups']}",
        "",
    ]

    if primary:
        lines.extend(
            [
                "## Primary issue",
                "",
                f"- Severity: **{primary.severity}**",
                f"- Category: `{primary.category}`",
                f"- Message: `{primary.normalized_message}`",
                f"- Count: {primary.count}",
                f"- First source: `{primary.first_source}:{primary.first_line}`",
                "",
                "Recommended next checks:",
                "",
            ]
        )
        for check in primary.recommended_checks:
            lines.append(f"1. {check}")
        lines.append("")
    else:
        lines.extend(
            [
                "## Primary issue",
                "",
                "No active errors or warnings were detected.",
                "",
            ]
        )

    lines.extend(
        [
            "## Active groups",
            "",
            "| Severity | Category | Count | First source | Message |",
            "|---|---|---:|---|---|",
        ]
    )

    if not active:
        lines.append("| OK | — | 0 | — | No active issues |")
    else:
        for item in active:
            message = item.normalized_message.replace("|", r"\|")
            if len(message) > 180:
                message = message[:177] + "..."
            source = f"{item.first_source}:{item.first_line}"
            lines.append(
                f"| {item.severity} | {item.category} | {item.count} | "
                f"`{source}` | {message} |"
            )

    ignored = [item for item in groups if item.ignored]
    lines.extend(
        [
            "",
            "## Ignored expected groups",
            "",
            "| Severity | Category | Count | Message |",
            "|---|---|---:|---|",
        ]
    )
    if not ignored:
        lines.append("| — | — | 0 | None |")
    else:
        for item in ignored:
            message = item.normalized_message.replace("|", r"\|")
            if len(message) > 180:
                message = message[:177] + "..."
            lines.append(
                f"| {item.severity} | {item.category} | "
                f"{item.count} | {message} |"
            )

    lines.extend(["", "## Detailed evidence", ""])
    for item in groups:
        label = "IGNORED" if item.ignored else item.severity
        lines.extend(
            [
                f"### {label}: {item.normalized_message}",
                "",
                f"- ID: `{item.issue_id}`",
                f"- Category: `{item.category}`",
                f"- Count: {item.count}",
                f"- Sources: {', '.join(f'`{source}`' for source in item.sources)}",
            ]
        )
        if item.ignore_reason:
            lines.append(f"- Ignore reason: {item.ignore_reason}")
        lines.append("")
        for sample in item.samples:
            lines.extend(["```text", sample, "```", ""])

    md_path = run_dir / "log_triage_report.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only deterministic ROS and engineering log triage."
    )
    parser.add_argument("--repo-root")
    parser.add_argument(
        "--config",
        help=(
            "Config path. Defaults to "
            "config/robot_doctor/log_triage_v1.json."
        ),
    )
    parser.add_argument(
        "--path",
        action="append",
        default=[],
        help="Additional file or directory to scan. Repeatable.",
    )
    parser.add_argument(
        "--output-root",
        default=str(
            Path.home() / "ai_robot_artifacts/robot_doctor/log_triage"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        repo_root = detect_repo_root(args.repo_root)
        config_path = (
            Path(args.config).expanduser().resolve()
            if args.config
            else repo_root / "config/robot_doctor/log_triage_v1.json"
        )
        if not config_path.exists():
            raise RuntimeError(f"Config not found: {config_path}")

        config = load_json(config_path)
        sources = discover_sources(repo_root, args.path)
        if not sources:
            raise RuntimeError(
                "No log sources found. Use --path FILE_OR_DIRECTORY."
            )

        files = iter_log_files(
            sources,
            max_files=int(config.get("max_files", 200)),
            max_file_bytes=int(
                config.get("max_file_bytes", 5 * 1024 * 1024)
            ),
        )
        if not files:
            raise RuntimeError("No readable log files found.")

        groups, stats = analyze_files(
            files,
            config=config,
            context_lines=int(config.get("context_lines", 2)),
        )

        output_root = Path(args.output_root).expanduser().resolve()
        run_dir = output_root / utc_stamp()
        run_dir.mkdir(parents=True, exist_ok=False)

        json_path, md_path = write_reports(
            run_dir, sources, files, groups, stats
        )
        (output_root / "latest.txt").write_text(
            str(run_dir) + "\n", encoding="utf-8"
        )

        print(md_path)
        print(json_path)

        status = overall_status(groups)
        if status == "OK":
            return 0
        if status == "WARNING":
            return 2
        return 3

    except Exception as exc:
        print(f"log-triage failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
