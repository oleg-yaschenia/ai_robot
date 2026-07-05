#!/usr/bin/env python3
"""
robot-doctor v1.1

Read-only diagnostic collector and deterministic health analyzer for the
AI robot project.

Safety boundaries:
- no sudo;
- no ROS parameter changes;
- no service restarts;
- no motor, UART command, firmware, or actuator access;
- no environment-variable or API-key collection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import shlex
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


COLLECTOR_VERSION = "robot-doctor-v1.1"

EXPECTED_NODES = {
    "/asr_node",
    "/assistant_router_node",
    "/assistant_core_shadow_node",
    "/camera/left/left_rectify",
    "/camera/right/right_rectify",
    "/esp32_bridge_node",
    "/head_state_manager",
    "/perception_entity_adapter_node",
    "/scene_interpreter_node",
    "/stereo_camera_node",
    "/stereo_disparity",
    "/tts_node",
    "/vision_assistant_node",
    "/voice_led_bridge_node",
    "/voice_manager_node",
    "/yolo_perception_node",
}

CRITICAL_NODES = {
    "/esp32_bridge_node",
    "/perception_entity_adapter_node",
    "/stereo_camera_node",
    "/stereo_disparity",
    "/voice_manager_node",
    "/yolo_perception_node",
}


@dataclass
class CommandResult:
    name: str
    command: str
    return_code: int | None
    duration_sec: float
    timed_out: bool
    stdout_file: str
    stderr_file: str
    available: bool
    accepted_return_codes: list[int]
    success: bool


@dataclass
class HealthCheck:
    check_id: str
    component: str
    status: str
    summary: str
    details: str
    evidence: str | None = None


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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

    raise RuntimeError(
        "Cannot detect repository root. Pass --repo-root ~/ai_robot."
    )


def safe_name(name: str) -> str:
    return "".join(
        ch if ch.isalnum() or ch in "-_." else "_" for ch in name
    )


def run_command(
    name: str,
    command: list[str],
    output_dir: Path,
    timeout_sec: int = 15,
    cwd: Path | None = None,
    accepted_return_codes: Iterable[int] = (0,),
) -> CommandResult:
    stdout_path = output_dir / f"{safe_name(name)}.stdout.txt"
    stderr_path = output_dir / f"{safe_name(name)}.stderr.txt"
    executable = command[0]
    accepted = list(accepted_return_codes)
    available = shutil.which(executable) is not None

    if not available:
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text(
            f"Command not found: {executable}\n", encoding="utf-8"
        )
        return CommandResult(
            name=name,
            command=shlex.join(command),
            return_code=None,
            duration_sec=0.0,
            timed_out=False,
            stdout_file=str(stdout_path),
            stderr_file=str(stderr_path),
            available=False,
            accepted_return_codes=accepted,
            success=False,
        )

    start = time.monotonic()
    timed_out = False
    return_code: int | None = None
    stdout = ""
    stderr = ""

    try:
        proc = subprocess.run(
            command,
            cwd=str(cwd) if cwd else None,
            text=True,
            capture_output=True,
            timeout=timeout_sec,
            check=False,
        )
        return_code = proc.returncode
        stdout = proc.stdout
        stderr = proc.stderr
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        return_code = 124
        stdout = exc.stdout or ""
        stderr = (exc.stderr or "") + (
            f"\nCollector timeout after {timeout_sec} seconds.\n"
        )
    except Exception as exc:
        return_code = 125
        stderr = f"{type(exc).__name__}: {exc}\n"

    duration = time.monotonic() - start
    stdout_path.write_text(stdout, encoding="utf-8", errors="replace")
    stderr_path.write_text(stderr, encoding="utf-8", errors="replace")

    success = available and not timed_out and return_code in accepted

    return CommandResult(
        name=name,
        command=shlex.join(command),
        return_code=return_code,
        duration_sec=round(duration, 3),
        timed_out=timed_out,
        stdout_file=str(stdout_path),
        stderr_file=str(stderr_path),
        available=available,
        accepted_return_codes=accepted,
        success=success,
    )


def command_map(snapshot: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(item.get("name")): item
        for item in snapshot.get("commands", [])
        if item.get("name")
    }


def read_command_output(
    commands: dict[str, dict[str, Any]],
    name: str,
    stream: str = "stdout",
) -> str:
    item = commands.get(name)
    if not item:
        return ""
    path_key = f"{stream}_file"
    path = item.get(path_key)
    if not path:
        return ""
    try:
        return Path(path).read_text(
            encoding="utf-8", errors="replace"
        )
    except OSError:
        return ""


def command_success(item: dict[str, Any] | None) -> bool:
    if not item:
        return False
    if "success" in item:
        return bool(item["success"])

    # Compatibility with v1 snapshots.
    rc = item.get("return_code")
    if item.get("name") == "tegrastats_sample" and rc == 124:
        return True
    return bool(item.get("available")) and rc == 0


def parse_free_bytes(text: str) -> tuple[int, int] | None:
    for line in text.splitlines():
        if line.startswith("Mem:"):
            fields = line.split()
            if len(fields) >= 7:
                try:
                    total = int(fields[1])
                    available = int(fields[6])
                    return total, available
                except ValueError:
                    return None
    return None


def parse_df_percent(text: str) -> int | None:
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        return None
    match = re.search(r"(\d+)%", lines[-1])
    return int(match.group(1)) if match else None


def status_rank(status: str) -> int:
    return {"OK": 0, "WARNING": 1, "ERROR": 2}.get(status, 2)


def overall_status(checks: list[HealthCheck]) -> str:
    if not checks:
        return "ERROR"
    return max(checks, key=lambda item: status_rank(item.status)).status


def analyze_snapshot(snapshot_dir: Path) -> tuple[Path, Path]:
    snapshot_path = snapshot_dir / "snapshot.json"
    if not snapshot_path.exists():
        raise RuntimeError(f"Snapshot not found: {snapshot_path}")

    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    commands = command_map(snapshot)
    checks: list[HealthCheck] = []

    # Collector command integrity.
    failed = [
        name
        for name, item in commands.items()
        if not command_success(item)
    ]
    if failed:
        checks.append(
            HealthCheck(
                "collector_commands",
                "robot-doctor",
                "WARNING",
                f"{len(failed)} diagnostic command(s) failed",
                ", ".join(sorted(failed)),
                str(snapshot_path),
            )
        )
    else:
        checks.append(
            HealthCheck(
                "collector_commands",
                "robot-doctor",
                "OK",
                "All diagnostic commands completed as expected",
                f"{len(commands)} commands checked",
                str(snapshot_path),
            )
        )

    # ROS nodes.
    ros_nodes_text = read_command_output(commands, "ros_nodes")
    current_nodes = {
        line.strip()
        for line in ros_nodes_text.splitlines()
        if line.strip().startswith("/")
    }
    missing = sorted(EXPECTED_NODES - current_nodes)
    missing_critical = sorted(CRITICAL_NODES - current_nodes)

    if missing_critical:
        checks.append(
            HealthCheck(
                "ros_nodes",
                "ROS 2 graph",
                "ERROR",
                "Critical ROS nodes are missing",
                ", ".join(missing_critical),
                commands.get("ros_nodes", {}).get("stdout_file"),
            )
        )
    elif missing:
        checks.append(
            HealthCheck(
                "ros_nodes",
                "ROS 2 graph",
                "WARNING",
                "Some expected ROS nodes are missing",
                ", ".join(missing),
                commands.get("ros_nodes", {}).get("stdout_file"),
            )
        )
    else:
        checks.append(
            HealthCheck(
                "ros_nodes",
                "ROS 2 graph",
                "OK",
                "All expected robot nodes are present",
                f"{len(EXPECTED_NODES)} expected nodes detected",
                commands.get("ros_nodes", {}).get("stdout_file"),
            )
        )

    # Git cleanliness.
    git_status = read_command_output(commands, "git_status").strip()
    if git_status:
        checks.append(
            HealthCheck(
                "git_status",
                "Git",
                "WARNING",
                "Repository has uncommitted changes",
                git_status[:2000],
                commands.get("git_status", {}).get("stdout_file"),
            )
        )
    else:
        checks.append(
            HealthCheck(
                "git_status",
                "Git",
                "OK",
                "Repository working tree is clean",
                "No modified or untracked files reported",
                commands.get("git_status", {}).get("stdout_file"),
            )
        )

    # Memory.
    free_text = read_command_output(commands, "memory_bytes")
    parsed_memory = parse_free_bytes(free_text)
    if parsed_memory:
        total, available = parsed_memory
        ratio = available / total if total else 0.0
        available_gib = available / (1024**3)
        if ratio < 0.10:
            status = "ERROR"
        elif ratio < 0.20:
            status = "WARNING"
        else:
            status = "OK"
        checks.append(
            HealthCheck(
                "memory_available",
                "Jetson memory",
                status,
                f"{available_gib:.2f} GiB available",
                f"{ratio * 100:.1f}% of total memory is available",
                commands.get("memory_bytes", {}).get("stdout_file"),
            )
        )
    else:
        checks.append(
            HealthCheck(
                "memory_available",
                "Jetson memory",
                "WARNING",
                "Could not parse memory information",
                "Check memory_bytes command output",
                commands.get("memory_bytes", {}).get("stdout_file"),
            )
        )

    # Disk.
    for command_name, label in (
        ("disk_root_bytes", "Root filesystem"),
        ("disk_home_bytes", "Home filesystem"),
    ):
        percent = parse_df_percent(
            read_command_output(commands, command_name)
        )
        if percent is None:
            checks.append(
                HealthCheck(
                    command_name,
                    "Disk",
                    "WARNING",
                    f"Could not parse {label} usage",
                    "Check df output",
                    commands.get(command_name, {}).get("stdout_file"),
                )
            )
            continue

        status = "ERROR" if percent >= 95 else (
            "WARNING" if percent >= 85 else "OK"
        )
        checks.append(
            HealthCheck(
                command_name,
                "Disk",
                status,
                f"{label} usage is {percent}%",
                "Thresholds: warning at 85%, error at 95%",
                commands.get(command_name, {}).get("stdout_file"),
            )
        )

    # Failed systemd units.
    failed_units = read_command_output(
        commands, "system_failed_units"
    )
    no_failed_units = (
        "0 loaded units listed" in failed_units
        or "0 unit files listed" in failed_units
    )
    checks.append(
        HealthCheck(
            "system_failed_units",
            "systemd",
            "OK" if no_failed_units else "WARNING",
            (
                "No failed systemd units"
                if no_failed_units
                else "Failed systemd units may be present"
            ),
            failed_units.strip()[:2000] or "No output",
            commands.get("system_failed_units", {}).get("stdout_file"),
        )
    )

    # Serial device.
    serial_item = commands.get("serial_device")
    checks.append(
        HealthCheck(
            "serial_device",
            "ESP32 transport",
            "OK" if command_success(serial_item) else "WARNING",
            (
                "/dev/ttyTHS1 is present"
                if command_success(serial_item)
                else "/dev/ttyTHS1 was not found"
            ),
            read_command_output(commands, "serial_device").strip()
            or read_command_output(
                commands, "serial_device", stream="stderr"
            ).strip(),
            (
                serial_item.get("stdout_file")
                if serial_item
                else None
            ),
        )
    )

    # tegrastats.
    tegra_text = read_command_output(commands, "tegrastats_sample").strip()
    checks.append(
        HealthCheck(
            "tegrastats_sample",
            "Jetson telemetry",
            "OK" if tegra_text else "WARNING",
            (
                "tegrastats sample collected"
                if tegra_text
                else "tegrastats produced no sample"
            ),
            tegra_text[:2000] or "No output",
            commands.get("tegrastats_sample", {}).get("stdout_file"),
        )
    )

    # ROS package prefixes.
    package_failures = []
    for name, item in commands.items():
        if name.startswith("ros_pkg_") and not command_success(item):
            package_failures.append(name.removeprefix("ros_pkg_"))
    checks.append(
        HealthCheck(
            "ros_package_prefixes",
            "ROS 2 packages",
            "WARNING" if package_failures else "OK",
            (
                "Some expected ROS packages are unavailable"
                if package_failures
                else "Expected ROS packages are available"
            ),
            (
                ", ".join(package_failures)
                if package_failures
                else "All checked package prefixes resolved"
            ),
            str(snapshot_path),
        )
    )

    report = {
        "schema_version": 1,
        "analyzer_version": COLLECTOR_VERSION,
        "analyzed_at_utc": iso_now(),
        "snapshot": str(snapshot_dir),
        "overall_status": overall_status(checks),
        "counts": {
            "OK": sum(item.status == "OK" for item in checks),
            "WARNING": sum(
                item.status == "WARNING" for item in checks
            ),
            "ERROR": sum(item.status == "ERROR" for item in checks),
        },
        "checks": [asdict(item) for item in checks],
    }

    json_path = snapshot_dir / "health_report.json"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    md_lines = [
        "# Robot health report",
        "",
        f"- Overall status: **{report['overall_status']}**",
        f"- Snapshot: `{snapshot_dir}`",
        f"- Analyzed: `{report['analyzed_at_utc']}`",
        (
            f"- Checks: OK={report['counts']['OK']}, "
            f"WARNING={report['counts']['WARNING']}, "
            f"ERROR={report['counts']['ERROR']}"
        ),
        "",
        "| Status | Component | Check | Result |",
        "|---|---|---|---|",
    ]

    for item in checks:
        compact_details = " ".join(item.details.split())
        if len(compact_details) > 180:
            compact_details = compact_details[:177] + "..."
        md_lines.append(
            f"| {item.status} | {item.component} | "
            f"{item.summary} | {compact_details} |"
        )

    md_lines.extend(["", "## Detailed checks", ""])
    for item in checks:
        md_lines.extend(
            [
                f"### {item.status}: {item.summary}",
                "",
                f"- ID: `{item.check_id}`",
                f"- Component: `{item.component}`",
                f"- Details: {item.details or 'No details'}",
                (
                    f"- Evidence: `{item.evidence}`"
                    if item.evidence
                    else "- Evidence: unavailable"
                ),
                "",
            ]
        )

    md_path = snapshot_dir / "health_report.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    return json_path, md_path


def collect(repo_root: Path, output_root: Path) -> Path:
    stamp = utc_timestamp()
    snapshot_dir = output_root / stamp
    commands_dir = snapshot_dir / "commands"
    snapshot_dir.mkdir(parents=True, exist_ok=False)
    commands_dir.mkdir(parents=True)

    results: list[CommandResult] = []

    def add(
        name: str,
        cmd: list[str],
        timeout: int = 15,
        cwd: Path | None = None,
        accepted_return_codes: Iterable[int] = (0,),
    ) -> None:
        results.append(
            run_command(
                name=name,
                command=cmd,
                output_dir=commands_dir,
                timeout_sec=timeout,
                cwd=cwd,
                accepted_return_codes=accepted_return_codes,
            )
        )

    # System.
    add("uname", ["uname", "-a"])
    add("uptime", ["uptime"])
    add("memory_human", ["free", "-h"])
    add("memory_bytes", ["free", "-b"])
    add("disk_root_human", ["df", "-h", "/"])
    add("disk_home_human", ["df", "-h", str(Path.home())])
    add("disk_root_bytes", ["df", "-P", "-B1", "/"])
    add(
        "disk_home_bytes",
        ["df", "-P", "-B1", str(Path.home())],
    )
    add("python_version", ["python3", "--version"])
    add("system_failed_units", ["systemctl", "--failed", "--no-pager"])
    add(
        "top_processes",
        [
            "ps",
            "-eo",
            "pid,ppid,stat,%cpu,%mem,etime,cmd",
            "--sort=-%cpu",
        ],
    )
    add("serial_device", ["ls", "-l", "/dev/ttyTHS1"])

    # `timeout` intentionally returns 124 after collecting a bounded sample.
    add(
        "tegrastats_sample",
        ["timeout", "3", "tegrastats"],
        timeout=5,
        accepted_return_codes=(0, 124),
    )
    add("jetson_release", ["cat", "/etc/nv_tegra_release"])
    add("os_release", ["cat", "/etc/os-release"])

    # Git.
    add("git_branch", ["git", "branch", "--show-current"], cwd=repo_root)
    add("git_commit", ["git", "rev-parse", "HEAD"], cwd=repo_root)
    add(
        "git_describe",
        ["git", "describe", "--tags", "--always", "--dirty"],
        cwd=repo_root,
    )
    add("git_status", ["git", "status", "--short"], cwd=repo_root)
    add(
        "git_recent_log",
        ["git", "log", "-8", "--oneline", "--decorate"],
        cwd=repo_root,
    )

    # Selected Python packages only.
    add(
        "python_packages",
        [
            "python3",
            "-m",
            "pip",
            "show",
            "torch",
            "torchvision",
            "ultralytics",
            "opencv-python",
            "numpy",
        ],
        timeout=20,
    )

    # ROS 2.
    add("ros_doctor_report", ["ros2", "doctor", "--report"], timeout=30)
    add("ros_nodes", ["ros2", "node", "list"], timeout=15)
    add("ros_topics", ["ros2", "topic", "list", "-t"], timeout=15)
    add("ros_services", ["ros2", "service", "list", "-t"], timeout=15)
    add("ros_actions", ["ros2", "action", "list", "-t"], timeout=15)

    for package in (
        "robot_bringup",
        "robot_camera",
        "robot_vision_assistant",
        "robot_audio",
        "robot_esp32_bridge",
    ):
        add(
            f"ros_pkg_{package}",
            ["ros2", "pkg", "prefix", package],
            timeout=10,
        )

    important_files = [
        repo_root / "ros2_ws/src/robot_camera/launch/stereo_disparity.launch.py",
        repo_root / "ros2_ws/src/robot_bringup/launch/robot_assistant_full.launch.py",
        repo_root / "ros2_ws/src/robot_vision_assistant/launch/local_assistant.launch.py",
        repo_root / "ros2_ws/src/robot_vision_assistant/robot_vision_assistant/assistant_response_contract.py",
        repo_root / "ros2_ws/src/robot_vision_assistant/robot_vision_assistant/qwen_vl_runtime_node.py",
        repo_root / "ros2_ws/src/robot_vision_assistant/robot_vision_assistant/response_orchestrator.py",
        repo_root / "ros2_ws/src/robot_vision_assistant/robot_vision_assistant/response_orchestrator_node.py",
        repo_root / "ros2_ws/src/robot_vision_assistant/robot_vision_assistant/vision_assistant_node.py",
        repo_root / "ros2_ws/src/robot_vision_assistant/robot_vision_assistant/yolo_perception_node.py",
        repo_root / "ros2_ws/src/robot_vision_assistant/robot_vision_assistant/yolo_tensorrt_node.py",
        repo_root / "ros2_ws/src/robot_vision_assistant/robot_vision_assistant/yolo_tensorrt_runtime.py",
        repo_root / "ros2_ws/src/robot_vision_assistant/setup.py",
    ]

    files_meta: list[dict[str, Any]] = []
    for path in important_files:
        try:
            relative = str(path.relative_to(repo_root))
        except ValueError:
            relative = str(path)

        entry: dict[str, Any] = {
            "path": relative,
            "exists": path.exists(),
        }
        if path.exists() and path.is_file():
            raw = path.read_bytes()
            entry.update(
                {
                    "size_bytes": len(raw),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "mtime_utc": datetime.fromtimestamp(
                        path.stat().st_mtime, tz=timezone.utc
                    ).isoformat(),
                }
            )
        files_meta.append(entry)

    snapshot: dict[str, Any] = {
        "schema_version": 1,
        "collector_version": COLLECTOR_VERSION,
        "collected_at_utc": iso_now(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "repo_root": str(repo_root),
        "output_dir": str(snapshot_dir),
        "commands": [asdict(item) for item in results],
        "important_files": files_meta,
        "expected_nodes": sorted(EXPECTED_NODES),
        "critical_nodes": sorted(CRITICAL_NODES),
        "notes": [
            "Read-only collection.",
            "Environment variables and API keys are intentionally not collected.",
            "tegrastats return code 124 is expected from the bounded timeout command.",
            "Command failures are recorded and do not abort the snapshot.",
        ],
    }

    snapshot_path = snapshot_dir / "snapshot.json"
    snapshot_path.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    successful = sum(item.success for item in results)
    failed = sum(item.available and not item.success for item in results)
    unavailable = sum(not item.available for item in results)
    timed_out = sum(item.timed_out for item in results)

    summary_lines = [
        f"{COLLECTOR_VERSION} snapshot",
        f"Collected: {snapshot['collected_at_utc']}",
        f"Host: {snapshot['hostname']}",
        f"Repository: {repo_root}",
        f"Snapshot: {snapshot_dir}",
        "",
        f"Commands successful: {successful}",
        f"Commands failed: {failed}",
        f"Commands unavailable: {unavailable}",
        f"Commands timed out: {timed_out}",
        "",
        "Non-successful or unavailable commands:",
    ]

    non_successful = [item for item in results if not item.success]
    if not non_successful:
        summary_lines.append("- none")
    else:
        for item in non_successful:
            summary_lines.append(
                f"- {item.name}: available={item.available}, "
                f"return_code={item.return_code}, "
                f"timed_out={item.timed_out}"
            )

    (snapshot_dir / "summary.txt").write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )

    latest_link = output_root / "latest"
    try:
        if latest_link.is_symlink() or latest_link.exists():
            latest_link.unlink()
        latest_link.symlink_to(snapshot_dir.name)
    except OSError:
        pass

    (output_root / "latest.txt").write_text(
        str(snapshot_dir) + "\n", encoding="utf-8"
    )

    analyze_snapshot(snapshot_dir)
    return snapshot_dir


def latest_snapshot(output_root: Path) -> Path:
    latest_file = output_root / "latest.txt"
    if not latest_file.exists():
        raise RuntimeError("No snapshots found.")
    return Path(latest_file.read_text(encoding="utf-8").strip())


def parse_args() -> argparse.Namespace:
    default_output = str(
        Path.home() / "ai_robot_artifacts/robot_doctor/snapshots"
    )

    parser = argparse.ArgumentParser(
        description="Read-only AI robot diagnostics and health analysis."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect_parser = subparsers.add_parser(
        "collect", help="Collect and analyze a new snapshot."
    )
    collect_parser.add_argument("--repo-root")
    collect_parser.add_argument(
        "--output-root", default=default_output
    )

    latest_parser = subparsers.add_parser(
        "latest", help="Print the latest snapshot path."
    )
    latest_parser.add_argument(
        "--output-root", default=default_output
    )

    analyze_parser = subparsers.add_parser(
        "analyze", help="Analyze an existing or latest snapshot."
    )
    analyze_parser.add_argument(
        "--snapshot",
        help="Snapshot directory. Latest is used when omitted.",
    )
    analyze_parser.add_argument(
        "--output-root", default=default_output
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        if args.command == "collect":
            repo_root = detect_repo_root(args.repo_root)
            output_root = Path(args.output_root).expanduser().resolve()
            output_root.mkdir(parents=True, exist_ok=True)
            snapshot_dir = collect(repo_root, output_root)
            print(snapshot_dir)
            print(snapshot_dir / "summary.txt")
            print(snapshot_dir / "health_report.md")
            print(snapshot_dir / "health_report.json")
            return 0

        if args.command == "latest":
            output_root = Path(args.output_root).expanduser().resolve()
            print(latest_snapshot(output_root))
            return 0

        if args.command == "analyze":
            output_root = Path(args.output_root).expanduser().resolve()
            snapshot_dir = (
                Path(args.snapshot).expanduser().resolve()
                if args.snapshot
                else latest_snapshot(output_root)
            )
            json_path, md_path = analyze_snapshot(snapshot_dir)
            print(md_path)
            print(json_path)
            return 0

    except Exception as exc:
        print(f"robot-doctor failed: {exc}", file=sys.stderr)
        return 1

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
