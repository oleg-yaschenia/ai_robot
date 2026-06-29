#!/usr/bin/env python3
"""
Regression Agent v1 for the AI robot project.

Runs a controlled, deterministic engineering verification pipeline:

1. Git whitespace checks.
2. Python syntax compilation.
3. Selected ROS 2 package build.
4. robot-doctor health snapshot.
5. Baseline Guardian comparison.
6. Final PASS / WARNING / FAIL report.

Safety boundaries:
- no sudo;
- no firmware flashing;
- no service restarts;
- no ROS parameter changes;
- no motor, servo, UART command, or actuator control;
- does not launch the robot automatically.

The robot should already be running when health and baseline checks are used.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TOOL_VERSION = "regression-agent-v1"


@dataclass
class StepResult:
    step_id: str
    title: str
    status: str
    return_code: int | None
    duration_sec: float
    command: str | None
    stdout_file: str | None
    stderr_file: str | None
    details: str


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8", errors="replace")


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


def run_command(
    step_id: str,
    title: str,
    command: list[str],
    output_dir: Path,
    cwd: Path,
    timeout_sec: int,
    accepted_return_codes: set[int] | None = None,
) -> StepResult:
    accepted = accepted_return_codes or {0}
    stdout_path = output_dir / f"{step_id}.stdout.txt"
    stderr_path = output_dir / f"{step_id}.stderr.txt"

    executable = command[0]
    if shutil.which(executable) is None:
        write_text(stdout_path, "")
        write_text(stderr_path, f"Command not found: {executable}\n")
        return StepResult(
            step_id=step_id,
            title=title,
            status="FAIL",
            return_code=None,
            duration_sec=0.0,
            command=shlex.join(command),
            stdout_file=str(stdout_path),
            stderr_file=str(stderr_path),
            details=f"Command not found: {executable}",
        )

    started = time.monotonic()
    try:
        proc = subprocess.run(
            command,
            cwd=str(cwd),
            text=True,
            capture_output=True,
            timeout=timeout_sec,
            check=False,
        )
        duration = time.monotonic() - started
        write_text(stdout_path, proc.stdout)
        write_text(stderr_path, proc.stderr)

        status = "PASS" if proc.returncode in accepted else "FAIL"
        details = (
            f"return_code={proc.returncode}"
            if status == "PASS"
            else f"Unexpected return code: {proc.returncode}"
        )
        return StepResult(
            step_id=step_id,
            title=title,
            status=status,
            return_code=proc.returncode,
            duration_sec=round(duration, 3),
            command=shlex.join(command),
            stdout_file=str(stdout_path),
            stderr_file=str(stderr_path),
            details=details,
        )
    except subprocess.TimeoutExpired as exc:
        duration = time.monotonic() - started
        stdout = exc.stdout or ""
        stderr = (exc.stderr or "") + (
            f"\nTimed out after {timeout_sec} seconds.\n"
        )
        write_text(stdout_path, stdout)
        write_text(stderr_path, stderr)
        return StepResult(
            step_id=step_id,
            title=title,
            status="FAIL",
            return_code=124,
            duration_sec=round(duration, 3),
            command=shlex.join(command),
            stdout_file=str(stdout_path),
            stderr_file=str(stderr_path),
            details=f"Timed out after {timeout_sec} seconds",
        )
    except Exception as exc:
        duration = time.monotonic() - started
        write_text(stdout_path, "")
        write_text(stderr_path, f"{type(exc).__name__}: {exc}\n")
        return StepResult(
            step_id=step_id,
            title=title,
            status="FAIL",
            return_code=125,
            duration_sec=round(duration, 3),
            command=shlex.join(command),
            stdout_file=str(stdout_path),
            stderr_file=str(stderr_path),
            details=f"{type(exc).__name__}: {exc}",
        )


def git_tracked_python_files(repo_root: Path, roots: list[str]) -> list[Path]:
    proc = subprocess.run(
        ["git", "ls-files", "*.py"],
        cwd=str(repo_root),
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "git ls-files failed")

    selected: list[Path] = []
    normalized_roots = [item.rstrip("/") + "/" for item in roots]

    for line in proc.stdout.splitlines():
        relative = line.strip()
        if not relative:
            continue
        if any(
            relative == root.rstrip("/") or relative.startswith(root)
            for root in normalized_roots
        ):
            path = repo_root / relative
            if path.exists():
                selected.append(path)

    return sorted(selected)


def parse_printed_path(text: str, suffix: str) -> Path | None:
    for line in reversed(text.splitlines()):
        candidate = Path(line.strip()).expanduser()
        if candidate.name == suffix and candidate.exists():
            return candidate.resolve()
    return None


def status_rank(status: str) -> int:
    return {"PASS": 0, "WARNING": 1, "FAIL": 2}.get(status, 2)


def overall_status(steps: list[StepResult]) -> str:
    return max(
        (step.status for step in steps),
        key=status_rank,
        default="FAIL",
    )


def update_step_from_health(
    step: StepResult,
    health_path: Path | None,
) -> StepResult:
    if health_path is None or not health_path.exists():
        step.status = "FAIL"
        step.details = "health_report.json was not produced"
        return step

    health = load_json(health_path)
    overall = str(health.get("overall_status", "ERROR"))

    if overall == "OK":
        step.status = "PASS"
    elif overall == "WARNING":
        step.status = "WARNING"
    else:
        step.status = "FAIL"

    counts = health.get("counts", {})
    step.details = (
        f"health={overall}; "
        f"OK={counts.get('OK', 0)}, "
        f"WARNING={counts.get('WARNING', 0)}, "
        f"ERROR={counts.get('ERROR', 0)}"
    )
    return step


def update_step_from_baseline(
    step: StepResult,
    report_path: Path | None,
) -> StepResult:
    if report_path is None or not report_path.exists():
        step.status = "FAIL"
        step.details = "baseline_comparison.json was not produced"
        return step

    report = load_json(report_path)
    overall = str(report.get("overall_status", "ERROR"))

    if overall == "OK":
        step.status = "PASS"
    elif overall == "WARNING":
        step.status = "WARNING"
    else:
        step.status = "FAIL"

    counts = report.get("counts", {})
    step.details = (
        f"baseline={overall}; "
        f"OK={counts.get('OK', 0)}, "
        f"WARNING={counts.get('WARNING', 0)}, "
        f"ERROR={counts.get('ERROR', 0)}"
    )
    return step


def run_pipeline(
    repo_root: Path,
    config: dict[str, Any],
    output_root: Path,
    skip_build: bool,
    skip_health: bool,
    skip_baseline: bool,
) -> Path:
    run_dir = output_root / utc_stamp()
    commands_dir = run_dir / "commands"
    run_dir.mkdir(parents=True, exist_ok=False)
    commands_dir.mkdir(parents=True)

    steps: list[StepResult] = []

    git_check = run_command(
        "git_diff_check",
        "Git whitespace and conflict-marker check",
        ["git", "diff", "--check", "HEAD"],
        commands_dir,
        repo_root,
        timeout_sec=30,
    )
    steps.append(git_check)

    python_roots = config.get(
        "python_roots",
        [
            "tools/robot_doctor",
            "ros2_ws/src/robot_camera",
            "ros2_ws/src/robot_vision_assistant",
        ],
    )

    try:
        python_files = git_tracked_python_files(repo_root, python_roots)
    except Exception as exc:
        python_files = []
        steps.append(
            StepResult(
                step_id="python_discovery",
                title="Discover Python files",
                status="FAIL",
                return_code=None,
                duration_sec=0.0,
                command=None,
                stdout_file=None,
                stderr_file=None,
                details=str(exc),
            )
        )

    if python_files:
        py_compile_command = [
            sys.executable,
            "-m",
            "py_compile",
            *[str(path) for path in python_files],
        ]
        steps.append(
            run_command(
                "python_syntax",
                f"Python syntax check ({len(python_files)} files)",
                py_compile_command,
                commands_dir,
                repo_root,
                timeout_sec=int(config.get("syntax_timeout_sec", 120)),
            )
        )
    else:
        steps.append(
            StepResult(
                step_id="python_syntax",
                title="Python syntax check",
                status="FAIL",
                return_code=None,
                duration_sec=0.0,
                command=None,
                stdout_file=None,
                stderr_file=None,
                details="No tracked Python files found in configured roots",
            )
        )

    if skip_build:
        steps.append(
            StepResult(
                step_id="colcon_build",
                title="ROS 2 package build",
                status="WARNING",
                return_code=None,
                duration_sec=0.0,
                command=None,
                stdout_file=None,
                stderr_file=None,
                details="Skipped by --skip-build",
            )
        )
    else:
        packages = config.get(
            "packages",
            [
                "robot_camera",
                "robot_vision_assistant",
                "robot_bringup",
                "robot_audio",
                "robot_esp32_bridge",
            ],
        )
        build_command = [
            "colcon",
            "build",
            "--symlink-install",
            "--packages-select",
            *packages,
        ]
        steps.append(
            run_command(
                "colcon_build",
                f"ROS 2 package build ({len(packages)} packages)",
                build_command,
                commands_dir,
                repo_root / "ros2_ws",
                timeout_sec=int(config.get("build_timeout_sec", 600)),
            )
        )

    doctor_script = (
        repo_root / "tools/robot_doctor/robot_doctor.py"
    )

    if skip_health:
        steps.append(
            StepResult(
                step_id="robot_health",
                title="Robot health snapshot",
                status="WARNING",
                return_code=None,
                duration_sec=0.0,
                command=None,
                stdout_file=None,
                stderr_file=None,
                details="Skipped by --skip-health",
            )
        )
        latest_snapshot = None
    else:
        health_step = run_command(
            "robot_health",
            "Robot health snapshot",
            [
                sys.executable,
                str(doctor_script),
                "collect",
                "--repo-root",
                str(repo_root),
            ],
            commands_dir,
            repo_root,
            timeout_sec=int(config.get("health_timeout_sec", 120)),
        )
        latest_snapshot = None
        if health_step.stdout_file:
            stdout = Path(health_step.stdout_file).read_text(
                encoding="utf-8", errors="replace"
            )
            health_json_path = parse_printed_path(
                stdout, "health_report.json"
            )
            if health_json_path:
                latest_snapshot = health_json_path.parent
            health_step = update_step_from_health(
                health_step, health_json_path
            )
        steps.append(health_step)

    baseline_path = (
        repo_root
        / config.get(
            "baseline_path",
            "config/robot_doctor/baselines/perception_v5.json",
        )
    )
    guardian_script = (
        repo_root / "tools/robot_doctor/baseline_guardian.py"
    )

    if skip_baseline:
        steps.append(
            StepResult(
                step_id="baseline_compare",
                title="Baseline comparison",
                status="WARNING",
                return_code=None,
                duration_sec=0.0,
                command=None,
                stdout_file=None,
                stderr_file=None,
                details="Skipped by --skip-baseline",
            )
        )
    elif latest_snapshot is None:
        steps.append(
            StepResult(
                step_id="baseline_compare",
                title="Baseline comparison",
                status="FAIL",
                return_code=None,
                duration_sec=0.0,
                command=None,
                stdout_file=None,
                stderr_file=None,
                details="No fresh robot-doctor snapshot is available",
            )
        )
    elif not baseline_path.exists():
        steps.append(
            StepResult(
                step_id="baseline_compare",
                title="Baseline comparison",
                status="FAIL",
                return_code=None,
                duration_sec=0.0,
                command=None,
                stdout_file=None,
                stderr_file=None,
                details=f"Baseline file not found: {baseline_path}",
            )
        )
    else:
        baseline_step = run_command(
            "baseline_compare",
            "Baseline comparison",
            [
                sys.executable,
                str(guardian_script),
                "compare",
                "--snapshot",
                str(latest_snapshot),
                "--baseline",
                str(baseline_path),
            ],
            commands_dir,
            repo_root,
            timeout_sec=int(
                config.get("baseline_timeout_sec", 120)
            ),
            accepted_return_codes={0, 2},
        )
        if baseline_step.stdout_file:
            stdout = Path(baseline_step.stdout_file).read_text(
                encoding="utf-8", errors="replace"
            )
            report_path = parse_printed_path(
                stdout, "baseline_comparison.json"
            )
            baseline_step = update_step_from_baseline(
                baseline_step, report_path
            )
        steps.append(baseline_step)

    overall = overall_status(steps)
    report = {
        "schema_version": 1,
        "tool_version": TOOL_VERSION,
        "created_at_utc": iso_now(),
        "repo_root": str(repo_root),
        "run_dir": str(run_dir),
        "overall_status": overall,
        "counts": {
            "PASS": sum(step.status == "PASS" for step in steps),
            "WARNING": sum(
                step.status == "WARNING" for step in steps
            ),
            "FAIL": sum(step.status == "FAIL" for step in steps),
        },
        "steps": [asdict(step) for step in steps],
    }

    json_path = run_dir / "regression_report.json"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Regression Agent report",
        "",
        f"- Overall status: **{overall}**",
        f"- Created: `{report['created_at_utc']}`",
        f"- Repository: `{repo_root}`",
        f"- Run directory: `{run_dir}`",
        (
            f"- Steps: PASS={report['counts']['PASS']}, "
            f"WARNING={report['counts']['WARNING']}, "
            f"FAIL={report['counts']['FAIL']}"
        ),
        "",
        "| Status | Step | Duration | Details |",
        "|---|---|---:|---|",
    ]

    for step in steps:
        lines.append(
            f"| {step.status} | {step.title} | "
            f"{step.duration_sec:.3f}s | {step.details} |"
        )

    lines.extend(["", "## Evidence", ""])
    for step in steps:
        lines.extend(
            [
                f"### {step.status}: {step.title}",
                "",
                f"- Step ID: `{step.step_id}`",
                f"- Command: `{step.command or 'not executed'}`",
                f"- Details: {step.details}",
                (
                    f"- Stdout: `{step.stdout_file}`"
                    if step.stdout_file
                    else "- Stdout: unavailable"
                ),
                (
                    f"- Stderr: `{step.stderr_file}`"
                    if step.stderr_file
                    else "- Stderr: unavailable"
                ),
                "",
            ]
        )

    md_path = run_dir / "regression_report.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")

    latest_file = output_root / "latest.txt"
    latest_file.write_text(str(run_dir) + "\n", encoding="utf-8")

    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Controlled AI robot regression verification."
    )
    parser.add_argument("--repo-root")
    parser.add_argument(
        "--config",
        help=(
            "JSON config path. Defaults to "
            "config/robot_doctor/regression_v1.json."
        ),
    )
    parser.add_argument(
        "--output-root",
        default=str(
            Path.home()
            / "ai_robot_artifacts/robot_doctor/regressions"
        ),
    )
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-health", action="store_true")
    parser.add_argument("--skip-baseline", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        repo_root = detect_repo_root(args.repo_root)
        config_path = (
            Path(args.config).expanduser().resolve()
            if args.config
            else (
                repo_root
                / "config/robot_doctor/regression_v1.json"
            )
        )
        if not config_path.exists():
            raise RuntimeError(f"Config file not found: {config_path}")

        config = load_json(config_path)
        output_root = Path(args.output_root).expanduser().resolve()
        output_root.mkdir(parents=True, exist_ok=True)

        run_dir = run_pipeline(
            repo_root=repo_root,
            config=config,
            output_root=output_root,
            skip_build=args.skip_build,
            skip_health=args.skip_health,
            skip_baseline=args.skip_baseline,
        )

        report = load_json(run_dir / "regression_report.json")
        print(run_dir / "regression_report.md")
        print(run_dir / "regression_report.json")

        overall = report.get("overall_status")
        if overall == "PASS":
            return 0
        if overall == "WARNING":
            return 2
        return 3

    except Exception as exc:
        print(f"regression-agent failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
