#!/usr/bin/env python3
"""
Configuration Review Agent v1.

Read-only deterministic review of ROS 2 launch/configuration files.

Checks:
- Python launch syntax;
- YAML/JSON/XML syntax;
- duplicate YAML keys;
- unresolved Git conflict markers;
- dangerous shell/system operations;
- hard-coded user paths;
- missing directly referenced local files;
- duplicate remap sources inside literal remappings lists;
- selected unsafe boolean configuration keys.

Safety:
- never edits project files;
- never runs launch files;
- never sends ROS/UART/actuator commands;
- never uses sudo;
- writes reports outside the Git repository.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import py_compile
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

VERSION = "configuration-review-v1.2"


@dataclass
class Finding:
    finding_id: str
    severity: str
    category: str
    file: str
    line: int
    message: str
    evidence: str
    recommendation: str


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def detect_repo_root(explicit: str | None) -> Path:
    if explicit:
        root = Path(explicit).expanduser().resolve()
        if not (root / ".git").exists():
            raise RuntimeError(f"Not a Git repository: {root}")
        return root

    current = Path(__file__).resolve()
    for parent in [current.parent, *current.parents, Path.cwd(), *Path.cwd().parents]:
        if (parent / ".git").exists() and (parent / "ros2_ws").exists():
            return parent

    raise RuntimeError("Cannot detect repository root. Use --repo-root.")


def git_tracked_files(repo_root: Path) -> list[Path]:
    import subprocess

    proc = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=str(repo_root),
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            proc.stderr.decode("utf-8", errors="replace").strip()
            or "git ls-files failed"
        )

    files: list[Path] = []
    for raw in proc.stdout.split(b"\0"):
        if not raw:
            continue
        path = repo_root / raw.decode("utf-8", errors="replace")
        if path.is_file():
            files.append(path)
    return files


def compile_patterns(values: Iterable[str]) -> list[re.Pattern[str]]:
    return [re.compile(value, re.IGNORECASE) for value in values]


def matches_any(patterns: list[re.Pattern[str]], text: str) -> bool:
    return any(pattern.search(text) for pattern in patterns)


def line_number(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def add_finding(
    findings: list[Finding],
    severity: str,
    category: str,
    path: Path,
    line: int,
    message: str,
    evidence: str,
    recommendation: str,
) -> None:
    raw = f"{severity}|{category}|{path}|{line}|{message}|{evidence}"
    finding_id = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    findings.append(
        Finding(
            finding_id=finding_id,
            severity=severity,
            category=category,
            file=str(path),
            line=line,
            message=message,
            evidence=evidence[:500],
            recommendation=recommendation,
        )
    )


def selected_files(
    repo_root: Path,
    config: dict[str, Any],
) -> list[Path]:
    include_roots = [
        (repo_root / item).resolve()
        for item in config.get(
            "include_roots",
            ["ros2_ws/src", "config"],
        )
    ]
    include_suffixes = tuple(
        config.get(
            "include_suffixes",
            [".launch.py", ".yaml", ".yml", ".json", ".xml"],
        )
    )
    include_names = set(config.get("include_names", ["package.xml"]))
    exclude_patterns = compile_patterns(
        config.get(
            "exclude_path_patterns",
            [r"/build/", r"/install/", r"/log/", r"/__pycache__/"],
        )
    )

    result: list[Path] = []
    for path in git_tracked_files(repo_root):
        resolved = path.resolve()

        if not any(
            resolved == root or root in resolved.parents
            for root in include_roots
        ):
            continue

        relative = str(path.relative_to(repo_root))
        if matches_any(exclude_patterns, "/" + relative):
            continue

        name_match = path.name in include_names
        suffix_match = any(path.name.endswith(suffix) for suffix in include_suffixes)
        if name_match or suffix_match:
            result.append(path)

    return sorted(result)


def check_conflict_markers(
    path: Path,
    text: str,
    findings: list[Finding],
) -> None:
    pattern = re.compile(r"^(<<<<<<<|=======|>>>>>>>)", re.MULTILINE)
    for match in pattern.finditer(text):
        add_finding(
            findings,
            "ERROR",
            "merge_conflict",
            path,
            line_number(text, match.start()),
            "Unresolved Git conflict marker",
            match.group(0),
            "Resolve the conflict before building or launching the robot.",
        )


def check_dangerous_patterns(
    path: Path,
    text: str,
    config: dict[str, Any],
    findings: list[Finding],
) -> None:
    allow_patterns = compile_patterns(
        config.get("dangerous_pattern_allowlist", [])
    )

    for item in config.get("dangerous_patterns", []):
        severity = item.get("severity", "ERROR")
        category = item.get("category", "dangerous_operation")
        recommendation = item.get(
            "recommendation",
            "Remove the operation or move it behind an explicit reviewed safety gate.",
        )
        pattern = re.compile(item["pattern"], re.IGNORECASE)

        for match in pattern.finditer(text):
            evidence = match.group(0)
            context_start = max(0, match.start() - 100)
            context_end = min(len(text), match.end() + 100)
            context = text[context_start:context_end]
            if matches_any(allow_patterns, context):
                continue

            add_finding(
                findings,
                severity,
                category,
                path,
                line_number(text, match.start()),
                item["message"],
                evidence,
                recommendation,
            )


def check_hardcoded_paths(
    repo_root: Path,
    path: Path,
    text: str,
    config: dict[str, Any],
    findings: list[Finding],
) -> None:
    relative = "/" + str(path.relative_to(repo_root))
    excluded = compile_patterns(
        config.get("hardcoded_path_exclude_patterns", [])
    )
    if matches_any(excluded, relative):
        return

    pattern = re.compile(r"/home/[A-Za-z0-9._-]+/[^\s'\"),\]}]+")
    for match in pattern.finditer(text):
        add_finding(
            findings,
            "WARNING",
            "hardcoded_path",
            path,
            line_number(text, match.start()),
            "Hard-coded user-specific path",
            match.group(0),
            "Use a package share directory, environment variable, LaunchConfiguration, or repository-relative path.",
        )


def _call_name(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _literal_pathjoin_candidates(
    repo_root: Path,
    call: ast.Call,
) -> tuple[set[int], list[Path]]:
    """
    Return string-node ids handled by PathJoinSubstitution and candidate paths.

    Dynamic leading elements such as LaunchConfiguration("project_root") are
    ignored. Literal suffix components are joined against known project roots.
    """
    handled: set[int] = set()
    components: list[str] = []

    if not call.args:
        return handled, []

    sequence = call.args[0]
    if not isinstance(sequence, (ast.List, ast.Tuple)):
        return handled, []

    for element in sequence.elts:
        if isinstance(element, ast.Constant) and isinstance(element.value, str):
            handled.add(id(element))
            components.append(element.value)

    if not components:
        return handled, []

    suffix = Path(*components)
    candidates = [
        repo_root / suffix,
        repo_root / "ros2_ws" / suffix,
        repo_root / "data" / suffix,
    ]
    return handled, candidates


def _literal_path_expression_candidates(
    repo_root: Path,
    node: ast.AST,
) -> tuple[set[int], list[Path]]:
    """
    Resolve literal suffixes from pathlib-style expressions such as:

        project_root / "calib" / "stereo" / "left.yaml"

    The dynamic root expression is ignored. Literal components are joined
    against known repository roots.
    """
    handled: set[int] = set()
    components: list[str] = []

    def visit(current: ast.AST) -> None:
        if isinstance(current, ast.BinOp) and isinstance(current.op, ast.Div):
            visit(current.left)
            visit(current.right)
            return
        if isinstance(current, ast.Constant) and isinstance(current.value, str):
            handled.add(id(current))
            components.append(current.value)

    visit(node)

    if not components:
        return handled, []

    suffix = Path(*components)
    candidates = [
        repo_root / suffix,
        repo_root / "ros2_ws" / suffix,
        repo_root / "data" / suffix,
    ]
    return handled, candidates


def parse_python_launch(
    repo_root: Path,
    path: Path,
    text: str,
    config: dict[str, Any],
    findings: list[Finding],
) -> None:
    try:
        py_compile.compile(str(path), doraise=True)
    except py_compile.PyCompileError as exc:
        add_finding(
            findings,
            "ERROR",
            "python_syntax",
            path,
            1,
            "Python launch/configuration syntax error",
            str(exc),
            "Fix syntax before running colcon or ros2 launch.",
        )
        return

    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError as exc:
        add_finding(
            findings,
            "ERROR",
            "python_syntax",
            path,
            exc.lineno or 1,
            "Python AST parsing failed",
            exc.msg,
            "Fix syntax before running the configuration.",
        )
        return

    # Find literal remappings=[(...), (...)] arguments.
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for keyword in node.keywords:
            if keyword.arg != "remappings":
                continue
            try:
                value = ast.literal_eval(keyword.value)
            except Exception:
                continue
            if not isinstance(value, (list, tuple)):
                continue

            sources: dict[str, set[str]] = {}
            for pair in value:
                if (
                    isinstance(pair, (list, tuple))
                    and len(pair) == 2
                    and all(isinstance(item, str) for item in pair)
                ):
                    src, dst = pair
                    sources.setdefault(src, set()).add(dst)

            for src, destinations in sources.items():
                if len(destinations) > 1:
                    add_finding(
                        findings,
                        "ERROR",
                        "remap_collision",
                        path,
                        getattr(keyword.value, "lineno", 1),
                        "One source topic is remapped to multiple destinations in the same literal remappings list",
                        f"{src} -> {sorted(destinations)}",
                        "Keep one unambiguous destination per source topic for that node.",
                    )

    reference_suffixes = tuple(
        config.get(
            "reference_suffixes",
            [".pt", ".pth", ".onnx", ".engine", ".plan", ".trt", ".yaml", ".yml", ".json"],
        )
    )
    ignored_reference_patterns = compile_patterns(
        config.get("ignored_reference_patterns", [])
    )

    # PathJoinSubstitution contains path fragments, not independent paths.
    handled_string_nodes: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if _call_name(node) != "PathJoinSubstitution":
            continue

        handled, candidates = _literal_pathjoin_candidates(repo_root, node)
        handled_string_nodes.update(handled)

        literal_files = [
            element
            for element in ast.walk(node)
            if isinstance(element, ast.Constant)
            and isinstance(element.value, str)
            and element.value.endswith(reference_suffixes)
        ]
        if not literal_files:
            continue

        target = literal_files[-1]
        if candidates and not any(candidate.exists() for candidate in candidates):
            add_finding(
                findings,
                "WARNING",
                "missing_reference",
                path,
                getattr(target, "lineno", 1),
                "PathJoinSubstitution target was not found under known project roots",
                target.value,
                "Verify the joined path or the LaunchConfiguration default used as its root.",
            )

    # pathlib-style project_root / "dir" / "file.ext" expressions also
    # contain fragments that must not be checked independently.
    for node in ast.walk(tree):
        if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Div):
            continue

        literal_files = [
            element
            for element in ast.walk(node)
            if isinstance(element, ast.Constant)
            and isinstance(element.value, str)
            and element.value.endswith(reference_suffixes)
        ]
        if not literal_files:
            continue

        handled, candidates = _literal_path_expression_candidates(
            repo_root, node
        )
        handled_string_nodes.update(handled)
        target = literal_files[-1]

        if candidates and not any(candidate.exists() for candidate in candidates):
            add_finding(
                findings,
                "WARNING",
                "missing_reference",
                path,
                getattr(target, "lineno", 1),
                "Path expression target was not found under known project roots",
                target.value,
                "Verify the joined pathlib path and its project-root source.",
            )

    # Check directly referenced model/config files in other string literals.
    for node in ast.walk(tree):
        if (
            not isinstance(node, ast.Constant)
            or not isinstance(node.value, str)
            or id(node) in handled_string_nodes
        ):
            continue

        value = node.value.strip()
        if not value.endswith(reference_suffixes):
            continue
        if "$" in value or "{" in value or "}" in value:
            continue
        if matches_any(ignored_reference_patterns, value):
            continue

        raw = Path(os.path.expanduser(value))
        if raw.is_absolute():
            candidates = [raw]
        else:
            candidates = [
                repo_root / raw,
                repo_root / "ros2_ws" / raw,
                path.parent / raw,
                repo_root / "data" / raw,
            ]

        if not any(candidate.exists() for candidate in candidates):
            add_finding(
                findings,
                "WARNING",
                "missing_reference",
                path,
                getattr(node, "lineno", 1),
                "Directly referenced local file was not found",
                value,
                "Verify the path or resolve it through package share/environment configuration.",
            )


class DuplicateKeyError(Exception):
    pass


def parse_yaml_with_duplicate_check(text: str) -> Any:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is not installed") from exc

    class UniqueKeyLoader(yaml.SafeLoader):
        pass

    def construct_mapping(loader: Any, node: Any, deep: bool = False) -> dict[Any, Any]:
        mapping: dict[Any, Any] = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            if key in mapping:
                mark = getattr(key_node, "start_mark", None)
                line = mark.line + 1 if mark is not None else 1
                raise DuplicateKeyError(f"Duplicate YAML key {key!r} at line {line}")
            mapping[key] = loader.construct_object(value_node, deep=deep)
        return mapping

    UniqueKeyLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        construct_mapping,
    )
    return yaml.load(text, Loader=UniqueKeyLoader)


def walk_mapping(value: Any, prefix: str = "") -> Iterable[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key)
            current = f"{prefix}.{key_text}" if prefix else key_text
            yield current, child
            yield from walk_mapping(child, current)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            current = f"{prefix}[{index}]"
            yield from walk_mapping(child, current)


def check_structured_file(
    path: Path,
    text: str,
    config: dict[str, Any],
    findings: list[Finding],
) -> None:
    data: Any = None
    try:
        if path.suffix.lower() in {".yaml", ".yml"}:
            data = parse_yaml_with_duplicate_check(text)
        elif path.suffix.lower() == ".json":
            data = json.loads(text)
        elif path.suffix.lower() == ".xml" or path.name == "package.xml":
            ET.fromstring(text)
            return
    except DuplicateKeyError as exc:
        match = re.search(r"line (\d+)", str(exc))
        line = int(match.group(1)) if match else 1
        add_finding(
            findings,
            "ERROR",
            "duplicate_key",
            path,
            line,
            "Duplicate YAML key",
            str(exc),
            "Remove the duplicate key; only the final value may otherwise be applied.",
        )
        return
    except RuntimeError as exc:
        add_finding(
            findings,
            "WARNING",
            "parser_unavailable",
            path,
            1,
            "Structured configuration parser unavailable",
            str(exc),
            "Install or restore the parser dependency before relying on this review.",
        )
        return
    except (json.JSONDecodeError, ET.ParseError, Exception) as exc:
        add_finding(
            findings,
            "ERROR",
            "structured_syntax",
            path,
            getattr(exc, "lineno", 1) or 1,
            "Invalid structured configuration syntax",
            str(exc),
            "Fix the file syntax before launching the corresponding component.",
        )
        return

    unsafe_keys = {
        key.lower(): severity
        for key, severity in config.get("unsafe_boolean_keys", {}).items()
    }
    for key_path, value in walk_mapping(data):
        leaf = key_path.split(".")[-1].split("[")[0].lower()
        if leaf in unsafe_keys and value is True:
            add_finding(
                findings,
                unsafe_keys[leaf],
                "unsafe_setting",
                path,
                1,
                "Potentially unsafe setting is enabled",
                f"{key_path}=true",
                "Keep this disabled unless it is explicitly reviewed for the current test.",
            )


def review(
    repo_root: Path,
    config: dict[str, Any],
) -> tuple[list[Path], list[Finding]]:
    files = selected_files(repo_root, config)
    findings: list[Finding] = []

    for path in files:
        text = path.read_text(encoding="utf-8", errors="replace")
        check_conflict_markers(path, text, findings)
        check_dangerous_patterns(path, text, config, findings)
        check_hardcoded_paths(repo_root, path, text, config, findings)

        if path.name.endswith(".launch.py"):
            parse_python_launch(
                repo_root,
                path,
                text,
                config,
                findings,
            )
        elif (
            path.suffix.lower() in {".yaml", ".yml", ".json", ".xml"}
            or path.name == "package.xml"
        ):
            check_structured_file(path, text, config, findings)

    severity_rank = {"ERROR": 0, "WARNING": 1, "INFO": 2}
    findings.sort(
        key=lambda item: (
            severity_rank.get(item.severity, 3),
            item.file,
            item.line,
            item.category,
        )
    )
    return files, findings


def overall_status(findings: list[Finding]) -> str:
    if any(item.severity == "ERROR" for item in findings):
        return "ERROR"
    if any(item.severity == "WARNING" for item in findings):
        return "WARNING"
    return "OK"


def write_reports(
    run_dir: Path,
    repo_root: Path,
    files: list[Path],
    findings: list[Finding],
) -> tuple[Path, Path]:
    status = overall_status(findings)
    counts = {
        "ERROR": sum(item.severity == "ERROR" for item in findings),
        "WARNING": sum(item.severity == "WARNING" for item in findings),
        "INFO": sum(item.severity == "INFO" for item in findings),
    }

    payload = {
        "schema_version": 1,
        "tool_version": VERSION,
        "created_at_utc": now_iso(),
        "repo_root": str(repo_root),
        "overall_status": status,
        "files_scanned": [str(path) for path in files],
        "counts": counts,
        "findings": [asdict(item) for item in findings],
    }

    json_path = run_dir / "configuration_review.json"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Configuration Review report",
        "",
        f"- Tool: `{VERSION}`",
        f"- Overall status: **{status}**",
        f"- Files scanned: {len(files)}",
        (
            f"- Findings: ERROR={counts['ERROR']}, "
            f"WARNING={counts['WARNING']}, INFO={counts['INFO']}"
        ),
        f"- Repository: `{repo_root}`",
        "",
        "## Findings",
        "",
        "| Severity | Category | File | Line | Message |",
        "|---|---|---|---:|---|",
    ]

    if not findings:
        lines.append("| OK | — | — | — | No configuration problems detected |")
    else:
        for item in findings:
            message = item.message.replace("|", r"\|")
            relative = Path(item.file)
            try:
                relative = relative.relative_to(repo_root)
            except ValueError:
                pass
            lines.append(
                f"| {item.severity} | {item.category} | "
                f"`{relative}` | {item.line} | {message} |"
            )

    lines.extend(["", "## Detailed evidence", ""])
    for item in findings:
        try:
            display_file = Path(item.file).relative_to(repo_root)
        except ValueError:
            display_file = Path(item.file)

        lines.extend(
            [
                f"### {item.severity}: {item.message}",
                "",
                f"- ID: `{item.finding_id}`",
                f"- Category: `{item.category}`",
                f"- File: `{display_file}:{item.line}`",
                f"- Evidence: `{item.evidence}`",
                f"- Recommendation: {item.recommendation}",
                "",
            ]
        )

    md_path = run_dir / "configuration_review.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path, json_path


def self_test(config: dict[str, Any]) -> int:
    with tempfile_directory() as temp:
        root = temp / "repo"
        (root / ".git").mkdir(parents=True)
        (root / "ros2_ws/src/test_pkg/launch").mkdir(parents=True)
        (root / "ros2_ws/src/test_pkg/config").mkdir(parents=True)

        valid_launch = root / "ros2_ws/src/test_pkg/launch/valid.launch.py"
        valid_launch.write_text(
            "from pathlib import Path\n"
            "def generate_launch_description():\n"
            "    remappings = [('image', '/camera/image')]\n"
            "    model = 'model.pt'\n"
            "    return []\n",
            encoding="utf-8",
        )
        (root / "model.pt").write_text("test", encoding="utf-8")

        bad_yaml = root / "ros2_ws/src/test_pkg/config/bad.yaml"
        bad_yaml.write_text(
            "node:\n  ros__parameters:\n    disable_watchdog: true\n",
            encoding="utf-8",
        )

        text = valid_launch.read_text(encoding="utf-8")
        findings: list[Finding] = []
        parse_python_launch(root, valid_launch, text, config, findings)
        check_structured_file(
            bad_yaml,
            bad_yaml.read_text(encoding="utf-8"),
            config,
            findings,
        )

        expected_warning = any(
            item.category == "unsafe_setting"
            and item.severity == "ERROR"
            for item in findings
        )
        missing_reference = any(
            item.category == "missing_reference"
            for item in findings
        )

        conflict_findings: list[Finding] = []
        check_conflict_markers(
            valid_launch,
            "<<<<<<< HEAD\nvalue\n=======\nother\n>>>>>>> branch\n",
            conflict_findings,
        )
        conflict_ok = any(
            item.category == "merge_conflict"
            and item.severity == "ERROR"
            for item in conflict_findings
        )

        joined_launch = root / "ros2_ws/src/test_pkg/launch/joined.launch.py"
        joined_target = root / "data/tts/piper/voice.onnx"
        joined_target.parent.mkdir(parents=True)
        joined_target.write_text("test", encoding="utf-8")
        joined_launch.write_text(
            "from launch.substitutions import PathJoinSubstitution\n"
            "def generate_launch_description():\n"
            "    model = PathJoinSubstitution([project_root, 'data', 'tts', 'piper', 'voice.onnx'])\n"
            "    return []\n",
            encoding="utf-8",
        )
        joined_findings: list[Finding] = []
        parse_python_launch(
            root,
            joined_launch,
            joined_launch.read_text(encoding="utf-8"),
            config,
            joined_findings,
        )
        joined_ok = not any(
            item.category == "missing_reference"
            for item in joined_findings
        )

        ros2_model = root / "ros2_ws/yolo11s.pt"
        ros2_model.write_text("test", encoding="utf-8")
        relative_launch = root / "ros2_ws/src/test_pkg/launch/relative.launch.py"
        relative_launch.write_text(
            "def generate_launch_description():\n"
            "    model = 'yolo11s.pt'\n"
            "    return []\n",
            encoding="utf-8",
        )
        relative_findings: list[Finding] = []
        parse_python_launch(
            root,
            relative_launch,
            relative_launch.read_text(encoding="utf-8"),
            config,
            relative_findings,
        )
        relative_ok = not any(
            item.category == "missing_reference"
            for item in relative_findings
        )

        path_target = root / "calib/stereo/left.yaml"
        path_target.parent.mkdir(parents=True)
        path_target.write_text("test", encoding="utf-8")
        path_launch = root / "ros2_ws/src/test_pkg/launch/path_expr.launch.py"
        path_launch.write_text(
            "from pathlib import Path\n"
            "def generate_launch_description():\n"
            "    project_root = Path('/dynamic')\n"
            "    calibration = project_root / 'calib' / 'stereo' / 'left.yaml'\n"
            "    return []\n",
            encoding="utf-8",
        )
        path_findings: list[Finding] = []
        parse_python_launch(
            root,
            path_launch,
            path_launch.read_text(encoding="utf-8"),
            config,
            path_findings,
        )
        path_expr_ok = not any(
            item.category == "missing_reference"
            for item in path_findings
        )

        self_config_excluded = matches_any(
            compile_patterns(config.get("exclude_path_patterns", [])),
            "/config/robot_doctor/config_review_v1.json",
        )

        results = [
            ("existing model reference accepted", not missing_reference),
            ("PathJoinSubstitution reference accepted", joined_ok),
            ("pathlib path expression accepted", path_expr_ok),
            ("ros2_ws relative model accepted", relative_ok),
            ("self configuration excluded", self_config_excluded),
            ("unsafe boolean detected", expected_warning),
            ("merge conflict detected", conflict_ok),
        ]

        failures = 0
        for name, passed in results:
            print(f"{'PASS' if passed else 'FAIL'}: {name}")
            failures += int(not passed)

        print(f"Self-test: {len(results) - failures}/{len(results)} passed")
        return 0 if failures == 0 else 1


class tempfile_directory:
    def __enter__(self) -> Path:
        import tempfile

        self._temporary = tempfile.TemporaryDirectory()
        return Path(self._temporary.name)

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self._temporary.cleanup()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only ROS 2 configuration review agent."
    )
    parser.add_argument("--repo-root")
    parser.add_argument("--config")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument(
        "--output-root",
        default=str(
            Path.home()
            / "ai_robot_artifacts/robot_doctor/configuration_reviews"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        root = detect_repo_root(args.repo_root)
        config_path = (
            Path(args.config).expanduser().resolve()
            if args.config
            else root / "config/robot_doctor/config_review_v1.json"
        )
        config = load_json(config_path)

        if args.self_test:
            return self_test(config)

        files, findings = review(root, config)

        output_root = Path(args.output_root).expanduser().resolve()
        run_dir = output_root / stamp()
        run_dir.mkdir(parents=True, exist_ok=False)

        md_path, json_path = write_reports(
            run_dir,
            root,
            files,
            findings,
        )
        (output_root / "latest.txt").write_text(
            str(run_dir) + "\n",
            encoding="utf-8",
        )

        print(md_path)
        print(json_path)

        return {
            "OK": 0,
            "WARNING": 2,
            "ERROR": 3,
        }[overall_status(findings)]

    except Exception as exc:
        print(f"configuration-review failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
