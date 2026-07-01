#!/usr/bin/env python3
"""Reproducible Ultralytics YOLO -> TensorRT FP16 export for Jetson.

The script exports in a temporary directory, smoke-tests the generated engine,
then copies it atomically to the requested output path. Existing engines are
never overwritten unless --force is supplied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_device(value: str) -> int | str:
    value = value.strip()
    return int(value) if value.isdigit() else value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a YOLO .pt model to a fixed-shape TensorRT FP16 engine."
    )
    parser.add_argument("--model", required=True, help="Source .pt model path.")
    parser.add_argument(
        "--output",
        help="Destination .engine path. Defaults to the source path with .engine suffix.",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument(
        "--workspace",
        type=float,
        default=2.0,
        help="TensorRT workspace limit in GiB. Default: 2.0.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing destination engine after a successful smoke test.",
    )
    parser.add_argument(
        "--skip-smoke-test",
        action="store_true",
        help="Skip loading and running one synthetic inference on the engine.",
    )
    parser.add_argument(
        "--report-root",
        default="~/ai_robot_artifacts/model_exports",
        help="Directory where JSON reports are written.",
    )
    return parser.parse_args()


def collect_versions() -> dict[str, Any]:
    import torch
    import ultralytics

    versions: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "ultralytics": getattr(ultralytics, "__version__", "unknown"),
        "torch": getattr(torch, "__version__", "unknown"),
        "torch_cuda": getattr(torch.version, "cuda", None),
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        ),
    }
    try:
        import tensorrt as trt

        versions["tensorrt"] = getattr(trt, "__version__", "unknown")
    except Exception as exc:
        versions["tensorrt"] = None
        versions["tensorrt_import_error"] = f"{type(exc).__name__}: {exc}"
    return versions


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    stamp = utc_timestamp()

    source = Path(args.model).expanduser().resolve()
    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else source.with_suffix(".engine")
    )
    report_dir = Path(args.report_root).expanduser().resolve() / stamp
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "export_report.json"

    report: dict[str, Any] = {
        "timestamp_utc": stamp,
        "status": "started",
        "source_model": str(source),
        "output_engine": str(output),
        "parameters": {
            "format": "engine",
            "imgsz": args.imgsz,
            "device": args.device,
            "batch": args.batch,
            "dynamic": False,
            "precision": "FP16",
            "workspace_gib": args.workspace,
            "nms": False,
        },
    }

    try:
        if not source.is_file():
            raise FileNotFoundError(f"Source model does not exist: {source}")
        if source.suffix.lower() != ".pt":
            raise ValueError(f"Source model must be a .pt file: {source}")
        if output.suffix.lower() != ".engine":
            raise ValueError(f"Output must use the .engine suffix: {output}")
        if output.exists() and not args.force:
            raise FileExistsError(
                f"Destination already exists: {output}. "
                "Use a different --output or pass --force."
            )
        if args.imgsz <= 0 or args.batch <= 0:
            raise ValueError("--imgsz and --batch must be positive")

        versions = collect_versions()
        report["versions"] = versions
        if not versions["cuda_available"]:
            raise RuntimeError("CUDA is unavailable; TensorRT export requires a GPU")
        if not versions.get("tensorrt"):
            raise RuntimeError(
                "TensorRT Python module is unavailable: "
                f"{versions.get('tensorrt_import_error', 'unknown error')}"
            )

        from ultralytics import YOLO

        try:
            from ultralytics.cfg import DEFAULT_CFG_DICT
        except Exception:
            DEFAULT_CFG_DICT = {}

        device = parse_device(args.device)
        report["source_sha256"] = sha256_file(source)
        output.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(prefix="yolo_trt_export_") as tmp:
            workdir = Path(tmp)
            temp_pt = workdir / source.name
            shutil.copy2(source, temp_pt)

            model = YOLO(str(temp_pt), task="detect")
            export_kwargs: dict[str, Any] = {
                "format": "engine",
                "imgsz": args.imgsz,
                "device": device,
                "batch": args.batch,
                "dynamic": False,
                "workspace": args.workspace,
                "nms": False,
                "verbose": True,
            }

            # Newer Ultralytics versions use quantize=16. Older Jetson builds
            # use half=True. Select the API exposed by the installed version.
            if "quantize" in DEFAULT_CFG_DICT:
                export_kwargs["quantize"] = 16
                report["parameters"]["precision_argument"] = "quantize=16"
            else:
                export_kwargs["half"] = True
                report["parameters"]["precision_argument"] = "half=True"

            print(f"Source: {source}")
            print(f"Output: {output}")
            print(f"Export parameters: {export_kwargs}")
            export_result = model.export(**export_kwargs)
            temp_engine = Path(str(export_result)).expanduser().resolve()

            if not temp_engine.is_file():
                raise RuntimeError(
                    f"Ultralytics returned an engine path that does not exist: "
                    f"{temp_engine}"
                )

            report["temporary_engine"] = str(temp_engine)
            report["engine_size_bytes"] = temp_engine.stat().st_size
            report["engine_sha256"] = sha256_file(temp_engine)

            if not args.skip_smoke_test:
                import numpy as np
                import torch

                engine_model = YOLO(str(temp_engine), task="detect")
                frame = np.zeros(
                    (args.imgsz, args.imgsz, 3),
                    dtype=np.uint8,
                )
                smoke_started = time.perf_counter()
                results = engine_model.predict(
                    source=frame,
                    imgsz=args.imgsz,
                    conf=0.01,
                    iou=0.45,
                    max_det=100,
                    device=device,
                    verbose=False,
                    stream=False,
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                report["smoke_test"] = {
                    "status": "passed",
                    "duration_sec": round(
                        time.perf_counter() - smoke_started,
                        3,
                    ),
                    "result_count": len(results),
                }
            else:
                report["smoke_test"] = {"status": "skipped"}

            staging = output.with_name(
                f".{output.name}.{stamp}.tmp"
            )
            shutil.copy2(temp_engine, staging)
            os.replace(staging, output)

        report["status"] = "passed"
        report["output_size_bytes"] = output.stat().st_size
        report["output_sha256"] = sha256_file(output)
        report["duration_sec"] = round(time.perf_counter() - started, 3)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        print()
        print("Export completed successfully.")
        print(f"Engine: {output}")
        print(f"Report: {report_path}")
        print(f"SHA256: {report['output_sha256']}")
        return 0

    except Exception as exc:
        report["status"] = "failed"
        report["duration_sec"] = round(time.perf_counter() - started, 3)
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"ERROR: {report['error']}", file=sys.stderr)
        print(f"Failure report: {report_path}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
