#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

VERSION = "log-triage-v1.1"


@dataclass
class Issue:
    issue_id: str
    severity: str
    category: str
    message: str
    count: int
    first_source: str
    first_line: int
    sources: list[str]
    ignored: bool
    ignore_reason: str | None


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def patterns(values: list[str]) -> list[re.Pattern[str]]:
    return [re.compile(value, re.IGNORECASE) for value in values]


def any_match(items: list[re.Pattern[str]], text: str) -> bool:
    return any(item.search(text) for item in items)


def strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text)


def repo_root(explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if (path / ".git").exists():
            return path
        raise RuntimeError(f"Not a Git repository: {path}")

    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents, Path.cwd(), *Path.cwd().parents]:
        if (parent / ".git").exists() and (parent / "ros2_ws").exists():
            return parent
    raise RuntimeError("Cannot detect repository root")


def pointer(path: Path) -> Path | None:
    if not path.exists():
        return None
    value = path.read_text(encoding="utf-8").strip()
    if not value:
        return None
    target = Path(value).expanduser()
    return target.resolve() if target.exists() else None


def discover(root: Path, extra: list[str]) -> list[Path]:
    result: list[Path] = []

    for raw in extra:
        path = Path(raw).expanduser().resolve()
        if path.exists():
            result.append(path)

    home = Path.home()
    for latest_file in (
        home / "ai_robot_artifacts/robot_doctor/regressions/latest.txt",
        home / "ai_robot_artifacts/robot_doctor/snapshots/latest.txt",
    ):
        run = pointer(latest_file)
        if run and (run / "commands").exists():
            result.append(run / "commands")

    for candidate in (
        root / "ros2_ws/log/latest_build",
        root / "ros2_ws/log/latest_test",
        home / ".ros/log/latest",
    ):
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved.exists():
            result.append(resolved)

    unique: list[Path] = []
    seen: set[str] = set()
    for item in result:
        key = str(item)
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def scan_files(sources: list[Path], cfg: dict[str, Any]) -> list[Path]:
    include = patterns(cfg["include_file_patterns"])
    exclude = patterns(cfg["exclude_file_patterns"])
    max_files = int(cfg.get("max_files", 120))
    max_bytes = int(cfg.get("max_file_bytes", 5 * 1024 * 1024))
    found: list[Path] = []

    for source in sources:
        candidates = [source] if source.is_file() else [
            item for item in source.rglob("*") if item.is_file()
        ]
        candidates.sort(key=lambda item: item.stat().st_mtime, reverse=True)

        for path in candidates:
            text = str(path)
            if not any_match(include, text) or any_match(exclude, text):
                continue
            try:
                size = path.stat().st_size
            except OSError:
                continue
            if size == 0 or size > max_bytes:
                continue
            found.append(path)
            if len(found) >= max_files:
                return found
    return found


def normalize(line: str) -> str:
    text = strip_ansi(line).strip()
    text = re.sub(r"^\[[0-9.]+\]\s*", "", text)
    text = re.sub(r"\b0x[0-9a-fA-F]+\b", "<HEX>", text)
    text = re.sub(r"\b\d+\.\d+\.\d+\.\d+\b", "<IP>", text)
    text = re.sub(r"/home/[^/\s]+", "~", text)
    text = re.sub(r"\b\d{5,}\b", "<N>", text)
    text = re.sub(r"\s+", " ", text)
    return text[:650]


def classify(line: str, cfg: dict[str, Any]) -> tuple[str, str, bool, str | None] | None:
    text = strip_ansi(line).strip()
    if not text:
        return None

    if any_match(patterns(cfg["ignore_line_patterns"]), text):
        return None

    if any_match(patterns(cfg["ignored_issue_patterns"]), text):
        if re.search(r"\b(?:ERROR|FATAL|WARN(?:ING)?|failed|not found|timed out)\b", text, re.I):
            return ("WARNING", category(text, cfg), True, "Known expected/non-actionable message")
        return None

    severity: str | None = None
    if any_match(patterns(cfg["error_patterns"]), text):
        severity = "ERROR"
    elif any_match(patterns(cfg["warning_patterns"]), text):
        severity = "WARNING"

    if severity is None:
        return None

    return (severity, category(text, cfg), False, None)


def category(text: str, cfg: dict[str, Any]) -> str:
    for name, values in cfg["category_patterns"].items():
        if any_match(patterns(values), text):
            return name
    return "general"


def analyze(files: list[Path], cfg: dict[str, Any]) -> tuple[list[Issue], dict[str, int]]:
    groups: dict[tuple[str, str, str, bool], dict[str, Any]] = {}
    lines_scanned = issue_lines = ignored_lines = 0

    for path in files:
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue

        for number, raw in enumerate(lines, start=1):
            lines_scanned += 1
            result = classify(raw, cfg)
            if result is None:
                continue

            severity, cat, ignored, reason = result
            issue_lines += 1
            ignored_lines += int(ignored)
            message = normalize(raw)
            key = (severity, cat, message, ignored)

            group = groups.setdefault(key, {
                "severity": severity,
                "category": cat,
                "message": message,
                "count": 0,
                "first_source": str(path),
                "first_line": number,
                "sources": set(),
                "ignored": ignored,
                "ignore_reason": reason,
            })
            group["count"] += 1
            group["sources"].add(str(path))

    issues: list[Issue] = []
    for group in groups.values():
        digest = hashlib.sha1(
            f"{group['severity']}|{group['category']}|{group['message']}".encode()
        ).hexdigest()[:12]
        issues.append(Issue(
            issue_id=digest,
            severity=group["severity"],
            category=group["category"],
            message=group["message"],
            count=group["count"],
            first_source=group["first_source"],
            first_line=group["first_line"],
            sources=sorted(group["sources"]),
            ignored=group["ignored"],
            ignore_reason=group["ignore_reason"],
        ))

    rank = {"ERROR": 0, "WARNING": 1}
    issues.sort(key=lambda item: (
        item.ignored,
        rank.get(item.severity, 2),
        -item.count,
        item.first_source,
        item.first_line,
    ))

    stats = {
        "files_scanned": len(files),
        "lines_scanned": lines_scanned,
        "issue_lines": issue_lines,
        "ignored_lines": ignored_lines,
        "active_groups": sum(not item.ignored for item in issues),
        "ignored_groups": sum(item.ignored for item in issues),
    }
    return issues, stats


def status(issues: list[Issue]) -> str:
    active = [item for item in issues if not item.ignored]
    if any(item.severity == "ERROR" for item in active):
        return "ERROR"
    if any(item.severity == "WARNING" for item in active):
        return "WARNING"
    return "OK"


def recommendations(category_name: str, severity: str) -> list[str]:
    mapping = {
        "python_exception": [
            "Open the first traceback and locate the earliest project-code frame.",
            "Run py_compile or the smallest reproducer before changing configuration.",
        ],
        "build": [
            "Inspect the first concrete compiler, CMake or package failure.",
            "Rebuild only the affected package before the full workspace.",
        ],
        "camera": [
            "Check camera/rectification nodes and image topic publication.",
            "Inspect the earliest Argus or GStreamer error from the same run.",
        ],
        "serial": [
            "Check /dev/ttyTHS1 ownership and dialout permissions.",
            "Verify only one process owns the serial device.",
        ],
        "ros_transport": [
            "Check the affected topic's publishers, subscribers and QoS.",
            "Verify the source node is alive before restarting anything.",
        ],
        "timeout": [
            "Decide whether the timeout is intentional or a stalled component.",
            "Check the component's last successful message and process state.",
        ],
        "general": [
            "Inspect the earliest occurrence and preceding context.",
            "Confirm whether it reproduces in the latest regression run.",
        ],
    }
    result = mapping.get(category_name, mapping["general"]).copy()
    if severity == "ERROR":
        result.insert(0, "Stop additional configuration changes until this error is explained.")
    return result


def write_report(run_dir: Path, sources: list[Path], files: list[Path], issues: list[Issue], stats: dict[str, int]) -> tuple[Path, Path]:
    overall = status(issues)
    active = [item for item in issues if not item.ignored]
    ignored = [item for item in issues if item.ignored]
    primary = active[0] if active else None

    payload = {
        "schema_version": 2,
        "tool_version": VERSION,
        "created_at_utc": now_iso(),
        "overall_status": overall,
        "sources": [str(item) for item in sources],
        "files": [str(item) for item in files],
        "stats": stats,
        "primary_issue_id": primary.issue_id if primary else None,
        "issues": [asdict(item) for item in issues],
    }

    json_path = run_dir / "log_triage_report.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Log Triage report",
        "",
        f"- Tool: `{VERSION}`",
        f"- Overall status: **{overall}**",
        f"- Files scanned: {stats['files_scanned']}",
        f"- Lines scanned: {stats['lines_scanned']}",
        f"- Active issue groups: {stats['active_groups']}",
        f"- Ignored expected groups: {stats['ignored_groups']}",
        "",
        "## Primary issue",
        "",
    ]

    if primary is None:
        lines.extend(["No active errors or warnings were detected.", ""])
    else:
        lines.extend([
            f"- Severity: **{primary.severity}**",
            f"- Category: `{primary.category}`",
            f"- Message: `{primary.message}`",
            f"- Count: {primary.count}",
            f"- First source: `{primary.first_source}:{primary.first_line}`",
            "",
            "Recommended next checks:",
            "",
        ])
        for item in recommendations(primary.category, primary.severity):
            lines.append(f"1. {item}")
        lines.append("")

    lines.extend([
        "## Active groups",
        "",
        "| Severity | Category | Count | First source | Message |",
        "|---|---|---:|---|---|",
    ])

    if not active:
        lines.append("| OK | — | 0 | — | No active issues |")
    else:
        for item in active:
            message = item.message.replace("|", r"\|")[:180]
            lines.append(
                f"| {item.severity} | {item.category} | {item.count} | "
                f"`{item.first_source}:{item.first_line}` | {message} |"
            )

    lines.extend([
        "",
        "## Ignored expected groups",
        "",
        "| Category | Count | Message |",
        "|---|---:|---|",
    ])

    if not ignored:
        lines.append("| — | 0 | None |")
    else:
        for item in ignored:
            escaped_message = item.message.replace("|", r"\|")[:180]
            lines.append(
                f"| {item.category} | {item.count} | "
                f"{escaped_message} |"
            )

    md_path = run_dir / "log_triage_report.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path, json_path


def self_test(cfg: dict[str, Any]) -> int:
    cases = [
        ("colcon DEBUG Parsed command line arguments: Namespace(continue_on_error=False)", None),
        ("WARNING: Package(s) not found: opencv-python", "IGNORED"),
        ("License: GPL-3.0-with-GCC-exception", None),
        ("crw-rw---- root dialout /dev/ttyTHS1", None),
        ("QOS COMPATIBILITY LIST", None),
        ("Invoked command returned '0': -W ignore:setup.py install is deprecated", None),
        ("[ERROR] [camera]: failed to open nvarguscamerasrc", "ERROR:camera"),
        ("Traceback (most recent call last):", "ERROR:python_exception"),
        ("CMake Error at CMakeLists.txt: package not found", "ERROR:build"),
        ("[WARN] [stereo_disparity]: dropped frame", "WARNING:general"),
    ]

    failures = 0
    for line, expected in cases:
        result = classify(line, cfg)
        if result is None:
            actual = None
        elif result[2]:
            actual = "IGNORED"
        else:
            actual = f"{result[0]}:{result[1]}"

        marker = "PASS" if actual == expected else "FAIL"
        print(f"{marker}: expected={expected!r} actual={actual!r} :: {line}")
        failures += int(actual != expected)

    print(f"Self-test: {len(cases) - failures}/{len(cases)} passed")
    return 0 if failures == 0 else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only robot log triage")
    parser.add_argument("--repo-root")
    parser.add_argument("--config")
    parser.add_argument("--path", action="append", default=[])
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument(
        "--output-root",
        default=str(Path.home() / "ai_robot_artifacts/robot_doctor/log_triage"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        root = repo_root(args.repo_root)
        cfg_path = (
            Path(args.config).expanduser().resolve()
            if args.config
            else root / "config/robot_doctor/log_triage_v1.json"
        )
        cfg = load_json(cfg_path)

        if args.self_test:
            return self_test(cfg)

        sources = discover(root, args.path)
        files = scan_files(sources, cfg)
        if not files:
            raise RuntimeError("No relevant log files found")

        issues, stats = analyze(files, cfg)
        output_root = Path(args.output_root).expanduser().resolve()
        run_dir = output_root / stamp()
        run_dir.mkdir(parents=True)
        md_path, json_path = write_report(run_dir, sources, files, issues, stats)
        (output_root / "latest.txt").write_text(str(run_dir) + "\n", encoding="utf-8")

        print(md_path)
        print(json_path)

        return {"OK": 0, "WARNING": 2, "ERROR": 3}[status(issues)]
    except Exception as exc:
        print(f"log-triage failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
