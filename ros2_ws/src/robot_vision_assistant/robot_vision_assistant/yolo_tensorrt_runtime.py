#!/usr/bin/env python3

"""Minimal TensorRT runtime for Ultralytics YOLO detection engines.

The module intentionally does not import torch or ultralytics. It reads the
small metadata header written by Ultralytics, deserializes the embedded native
TensorRT plan, reuses CUDA buffers, and returns ordinary Python detections.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np


class TensorRTRuntimeError(RuntimeError):
    """Raised when the direct TensorRT detector cannot be initialized or run."""


def read_ultralytics_engine(
    engine_path: str,
) -> Tuple[Dict[str, Any], bytes]:
    """Return Ultralytics metadata and the native TensorRT plan payload."""

    path = Path(engine_path).expanduser()
    raw = path.read_bytes()
    metadata: Dict[str, Any] = {}
    plan = raw

    if len(raw) >= 8:
        metadata_length = int.from_bytes(
            raw[:4],
            byteorder="little",
            signed=True,
        )
        if 0 < metadata_length < min(len(raw) - 4, 1_000_000):
            metadata_bytes = raw[4:4 + metadata_length]
            try:
                parsed = json.loads(metadata_bytes.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                parsed = None

            if isinstance(parsed, dict):
                metadata = parsed
                plan = raw[4 + metadata_length:]

    if not plan:
        raise TensorRTRuntimeError(
            f"TensorRT plan payload is empty: {path}"
        )

    return metadata, plan


def normalize_class_names(value: Any) -> Dict[int, str]:
    """Normalize metadata class names to an integer-keyed dictionary."""

    if isinstance(value, dict):
        names: Dict[int, str] = {}
        for key, name in value.items():
            try:
                class_id = int(key)
            except (TypeError, ValueError):
                continue
            names[class_id] = str(name)
        return names

    if isinstance(value, (list, tuple)):
        return {
            index: str(name)
            for index, name in enumerate(value)
        }

    return {}


def letterbox_bgr(
    frame_bgr: np.ndarray,
    target_size: int,
) -> Tuple[np.ndarray, float, Tuple[float, float]]:
    """Resize and pad an OpenCV BGR frame to YOLO NCHW float input."""

    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError(
            f"expected HxWx3 BGR frame, got {frame_bgr.shape}"
        )
    if target_size <= 0:
        raise ValueError(f"invalid target_size: {target_size}")

    original_height, original_width = frame_bgr.shape[:2]
    ratio = min(
        float(target_size) / max(float(original_height), 1.0),
        float(target_size) / max(float(original_width), 1.0),
    )

    resized_width = max(1, int(round(original_width * ratio)))
    resized_height = max(1, int(round(original_height * ratio)))

    if (resized_width, resized_height) != (
        original_width,
        original_height,
    ):
        resized = cv2.resize(
            frame_bgr,
            (resized_width, resized_height),
            interpolation=cv2.INTER_LINEAR,
        )
    else:
        resized = frame_bgr

    pad_width = target_size - resized_width
    pad_height = target_size - resized_height
    pad_left = int(round(pad_width / 2.0 - 0.1))
    pad_right = int(round(pad_width / 2.0 + 0.1))
    pad_top = int(round(pad_height / 2.0 - 0.1))
    pad_bottom = int(round(pad_height / 2.0 + 0.1))

    padded = cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    if padded.shape[0] != target_size or padded.shape[1] != target_size:
        raise TensorRTRuntimeError(
            "letterbox produced unexpected shape: "
            f"{padded.shape}, target={target_size}"
        )

    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    tensor = np.ascontiguousarray(
        rgb.transpose(2, 0, 1)[None],
        dtype=np.float32,
    )
    tensor *= 1.0 / 255.0

    return tensor, ratio, (float(pad_left), float(pad_top))


def _normalize_nms_indexes(indexes: Any) -> List[int]:
    if indexes is None:
        return []
    array = np.asarray(indexes).reshape(-1)
    return [int(index) for index in array.tolist()]


def decode_yolo_output(
    output: np.ndarray,
    original_shape: Tuple[int, int],
    ratio: float,
    pad: Tuple[float, float],
    confidence_threshold: float,
    iou_threshold: float,
    max_det: int,
) -> List[Dict[str, Any]]:
    """Decode a standard YOLO11 output tensor and run class-aware NMS."""

    prediction = np.asarray(output)
    if prediction.ndim == 3:
        if prediction.shape[0] != 1:
            raise TensorRTRuntimeError(
                f"unsupported output batch shape: {prediction.shape}"
            )
        prediction = prediction[0]

    if prediction.ndim != 2:
        raise TensorRTRuntimeError(
            f"expected 2D/3D YOLO output, got {prediction.shape}"
        )

    # Standard exported YOLO detection output is [4 + classes, anchors].
    if prediction.shape[0] <= 512 and prediction.shape[0] < prediction.shape[1]:
        prediction = prediction.T

    if prediction.shape[1] < 5:
        raise TensorRTRuntimeError(
            f"unsupported YOLO output shape: {prediction.shape}"
        )

    boxes_xywh = prediction[:, :4].astype(np.float32, copy=False)
    class_scores = prediction[:, 4:].astype(np.float32, copy=False)
    class_ids = np.argmax(class_scores, axis=1).astype(np.int32)
    confidences = class_scores[
        np.arange(class_scores.shape[0]),
        class_ids,
    ]

    valid = (
        np.isfinite(confidences)
        & (confidences >= float(confidence_threshold))
    )
    if not np.any(valid):
        return []

    boxes_xywh = boxes_xywh[valid]
    class_ids = class_ids[valid]
    confidences = confidences[valid]

    x_center = boxes_xywh[:, 0]
    y_center = boxes_xywh[:, 1]
    box_width = boxes_xywh[:, 2]
    box_height = boxes_xywh[:, 3]

    boxes_xyxy = np.stack(
        [
            x_center - box_width / 2.0,
            y_center - box_height / 2.0,
            x_center + box_width / 2.0,
            y_center + box_height / 2.0,
        ],
        axis=1,
    )

    selected_global: List[int] = []
    for class_id in np.unique(class_ids):
        class_indexes = np.flatnonzero(class_ids == class_id)
        class_boxes = boxes_xyxy[class_indexes]
        nms_boxes = [
            [
                float(box[0]),
                float(box[1]),
                float(max(0.0, box[2] - box[0])),
                float(max(0.0, box[3] - box[1])),
            ]
            for box in class_boxes
        ]
        class_confidences = [
            float(value)
            for value in confidences[class_indexes]
        ]
        kept_local = _normalize_nms_indexes(
            cv2.dnn.NMSBoxes(
                nms_boxes,
                class_confidences,
                float(confidence_threshold),
                float(iou_threshold),
            )
        )
        selected_global.extend(
            int(class_indexes[index])
            for index in kept_local
        )

    selected_global.sort(
        key=lambda index: float(confidences[index]),
        reverse=True,
    )
    selected_global = selected_global[: max(0, int(max_det))]

    image_height, image_width = original_shape
    pad_x, pad_y = pad
    safe_ratio = max(float(ratio), 1e-9)
    detections: List[Dict[str, Any]] = []

    for index in selected_global:
        x1, y1, x2, y2 = boxes_xyxy[index]
        x1 = (float(x1) - pad_x) / safe_ratio
        y1 = (float(y1) - pad_y) / safe_ratio
        x2 = (float(x2) - pad_x) / safe_ratio
        y2 = (float(y2) - pad_y) / safe_ratio

        x1 = max(0.0, min(float(image_width), x1))
        x2 = max(0.0, min(float(image_width), x2))
        y1 = max(0.0, min(float(image_height), y1))
        y2 = max(0.0, min(float(image_height), y2))

        if x2 <= x1 or y2 <= y1:
            continue

        detections.append({
            "class_id": int(class_ids[index]),
            "confidence": float(confidences[index]),
            "bbox_xyxy": [
                int(round(x1)),
                int(round(y1)),
                int(round(x2)),
                int(round(y2)),
            ],
        })

    return detections


class _CudaRuntime:
    SUCCESS = 0
    MEMCPY_HOST_TO_DEVICE = 1
    MEMCPY_DEVICE_TO_HOST = 2

    def __init__(self, device_id: int) -> None:
        candidates = [
            ctypes.util.find_library("cudart"),
            "libcudart.so",
            "libcudart.so.12",
        ]
        library = None
        errors = []
        for candidate in candidates:
            if not candidate:
                continue
            try:
                library = ctypes.CDLL(candidate)
                break
            except OSError as exc:
                errors.append(f"{candidate}: {exc}")
        if library is None:
            raise TensorRTRuntimeError(
                "unable to load CUDA runtime: " + "; ".join(errors)
            )

        self.lib = library
        self._configure_signatures()
        self.check(self.lib.cudaSetDevice(int(device_id)), "cudaSetDevice")

    def _configure_signatures(self) -> None:
        self.lib.cudaSetDevice.argtypes = [ctypes.c_int]
        self.lib.cudaSetDevice.restype = ctypes.c_int
        self.lib.cudaMalloc.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_size_t,
        ]
        self.lib.cudaMalloc.restype = ctypes.c_int
        self.lib.cudaFree.argtypes = [ctypes.c_void_p]
        self.lib.cudaFree.restype = ctypes.c_int
        self.lib.cudaMemcpyAsync.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        self.lib.cudaMemcpyAsync.restype = ctypes.c_int
        self.lib.cudaStreamCreate.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self.lib.cudaStreamCreate.restype = ctypes.c_int
        self.lib.cudaStreamDestroy.argtypes = [ctypes.c_void_p]
        self.lib.cudaStreamDestroy.restype = ctypes.c_int
        self.lib.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
        self.lib.cudaStreamSynchronize.restype = ctypes.c_int
        self.lib.cudaGetErrorString.argtypes = [ctypes.c_int]
        self.lib.cudaGetErrorString.restype = ctypes.c_char_p

    def check(self, code: int, operation: str) -> None:
        if code == self.SUCCESS:
            return
        message = self.lib.cudaGetErrorString(code)
        text = (
            message.decode("utf-8", errors="replace")
            if message
            else "unknown CUDA error"
        )
        raise TensorRTRuntimeError(
            f"{operation} failed: CUDA {code}: {text}"
        )

    def malloc(self, size: int) -> ctypes.c_void_p:
        pointer = ctypes.c_void_p()
        self.check(
            self.lib.cudaMalloc(
                ctypes.byref(pointer),
                ctypes.c_size_t(size),
            ),
            "cudaMalloc",
        )
        return pointer

    def free(self, pointer: Optional[ctypes.c_void_p]) -> None:
        if pointer is not None and pointer.value:
            self.check(self.lib.cudaFree(pointer), "cudaFree")

    def create_stream(self) -> ctypes.c_void_p:
        stream = ctypes.c_void_p()
        self.check(
            self.lib.cudaStreamCreate(ctypes.byref(stream)),
            "cudaStreamCreate",
        )
        return stream

    def destroy_stream(self, stream: Optional[ctypes.c_void_p]) -> None:
        if stream is not None and stream.value:
            self.check(
                self.lib.cudaStreamDestroy(stream),
                "cudaStreamDestroy",
            )

    def copy_h2d_async(
        self,
        device_pointer: ctypes.c_void_p,
        host_array: np.ndarray,
        stream: ctypes.c_void_p,
    ) -> None:
        self.check(
            self.lib.cudaMemcpyAsync(
                device_pointer,
                ctypes.c_void_p(host_array.ctypes.data),
                ctypes.c_size_t(host_array.nbytes),
                self.MEMCPY_HOST_TO_DEVICE,
                stream,
            ),
            "cudaMemcpyAsync(H2D)",
        )

    def copy_d2h_async(
        self,
        host_array: np.ndarray,
        device_pointer: ctypes.c_void_p,
        stream: ctypes.c_void_p,
    ) -> None:
        self.check(
            self.lib.cudaMemcpyAsync(
                ctypes.c_void_p(host_array.ctypes.data),
                device_pointer,
                ctypes.c_size_t(host_array.nbytes),
                self.MEMCPY_DEVICE_TO_HOST,
                stream,
            ),
            "cudaMemcpyAsync(D2H)",
        )

    def synchronize(self, stream: ctypes.c_void_p) -> None:
        self.check(
            self.lib.cudaStreamSynchronize(stream),
            "cudaStreamSynchronize",
        )


def _volume(shape: Iterable[int]) -> int:
    result = 1
    for value in shape:
        result *= int(value)
    return result


class TensorRTYoloRuntime:
    """Reusable direct TensorRT YOLO inference runtime."""

    def __init__(
        self,
        engine_path: str,
        imgsz: int,
        device_id: int = 0,
    ) -> None:
        self.engine_path = str(Path(engine_path).expanduser())
        self.imgsz = int(imgsz)
        self.device_id = int(device_id)
        self.metadata, serialized_plan = read_ultralytics_engine(
            self.engine_path
        )
        self.names = normalize_class_names(
            self.metadata.get("names")
        )

        try:
            import tensorrt as trt
        except ImportError as exc:
            raise TensorRTRuntimeError(
                "TensorRT Python bindings are unavailable"
            ) from exc

        self.trt = trt
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(
            serialized_plan
        )
        if self.engine is None:
            raise TensorRTRuntimeError(
                f"failed to deserialize TensorRT engine: {self.engine_path}"
            )

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise TensorRTRuntimeError(
                "failed to create TensorRT execution context"
            )

        self.cuda = _CudaRuntime(self.device_id)
        self.stream = self.cuda.create_stream()
        self.host_buffers: Dict[str, np.ndarray] = {}
        self.device_buffers: Dict[str, ctypes.c_void_p] = {}
        self.input_names: List[str] = []
        self.output_names: List[str] = []
        self.closed = False

        try:
            self._allocate_buffers()
        except Exception:
            self.close()
            raise

    def _resolve_input_shape(
        self,
        tensor_name: str,
    ) -> Tuple[int, ...]:
        shape = tuple(
            int(value)
            for value in self.engine.get_tensor_shape(tensor_name)
        )
        if all(value > 0 for value in shape):
            return shape

        resolved: List[int] = []
        for index, value in enumerate(shape):
            if value > 0:
                resolved.append(value)
            elif index == 0:
                resolved.append(1)
            elif index in (2, 3):
                resolved.append(self.imgsz)
            else:
                raise TensorRTRuntimeError(
                    "cannot resolve dynamic TensorRT input shape: "
                    f"{tensor_name}={shape}"
                )
        return tuple(resolved)

    def _allocate_buffers(self) -> None:
        trt = self.trt

        for index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
                shape = self._resolve_input_shape(name)
                if not self.context.set_input_shape(name, shape):
                    raise TensorRTRuntimeError(
                        f"failed to set TensorRT input shape {name}={shape}"
                    )
            else:
                self.output_names.append(name)

        if len(self.input_names) != 1:
            raise TensorRTRuntimeError(
                "expected one YOLO input tensor, found "
                f"{self.input_names}"
            )
        if not self.output_names:
            raise TensorRTRuntimeError(
                "TensorRT engine has no output tensors"
            )

        for index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            shape = tuple(
                int(value)
                for value in self.context.get_tensor_shape(name)
            )
            if any(value <= 0 for value in shape):
                raise TensorRTRuntimeError(
                    f"unresolved TensorRT tensor shape: {name}={shape}"
                )

            dtype = np.dtype(
                trt.nptype(self.engine.get_tensor_dtype(name))
            )
            host = np.empty(shape, dtype=dtype)
            device = self.cuda.malloc(host.nbytes)
            if not self.context.set_tensor_address(
                name,
                int(device.value),
            ):
                self.cuda.free(device)
                raise TensorRTRuntimeError(
                    f"failed to bind TensorRT tensor: {name}"
                )

            self.host_buffers[name] = host
            self.device_buffers[name] = device

    def predict(
        self,
        frame_bgr: np.ndarray,
        confidence_threshold: float,
        iou_threshold: float,
        max_det: int,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
        if self.closed:
            raise TensorRTRuntimeError("TensorRT runtime is closed")

        preprocess_started = time.perf_counter()
        input_tensor, ratio, pad = letterbox_bgr(
            frame_bgr,
            self.imgsz,
        )
        preprocess_ms = (
            time.perf_counter() - preprocess_started
        ) * 1000.0

        input_name = self.input_names[0]
        input_buffer = self.host_buffers[input_name]
        if input_tensor.shape != input_buffer.shape:
            raise TensorRTRuntimeError(
                "preprocessed input shape does not match engine: "
                f"input={input_tensor.shape}, engine={input_buffer.shape}"
            )
        np.copyto(input_buffer, input_tensor, casting="same_kind")

        inference_started = time.perf_counter()
        self.cuda.copy_h2d_async(
            self.device_buffers[input_name],
            input_buffer,
            self.stream,
        )
        if not self.context.execute_async_v3(
            stream_handle=int(self.stream.value)
        ):
            raise TensorRTRuntimeError(
                "TensorRT execute_async_v3 returned false"
            )
        for output_name in self.output_names:
            self.cuda.copy_d2h_async(
                self.host_buffers[output_name],
                self.device_buffers[output_name],
                self.stream,
            )
        self.cuda.synchronize(self.stream)
        inference_ms = (
            time.perf_counter() - inference_started
        ) * 1000.0

        postprocess_started = time.perf_counter()
        output = self.host_buffers[self.output_names[0]]
        detections = decode_yolo_output(
            output,
            original_shape=frame_bgr.shape[:2],
            ratio=ratio,
            pad=pad,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            max_det=max_det,
        )
        postprocess_ms = (
            time.perf_counter() - postprocess_started
        ) * 1000.0

        return detections, {
            "preprocess": preprocess_ms,
            "inference": inference_ms,
            "postprocess": postprocess_ms,
        }

    def warm_up(self, frame_bgr: np.ndarray, repetitions: int = 2) -> None:
        for _ in range(max(1, int(repetitions))):
            self.predict(
                frame_bgr,
                confidence_threshold=0.10,
                iou_threshold=0.45,
                max_det=100,
            )

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True

        for pointer in self.device_buffers.values():
            try:
                self.cuda.free(pointer)
            except Exception:
                pass
        self.device_buffers.clear()
        self.host_buffers.clear()

        try:
            self.cuda.destroy_stream(self.stream)
        except Exception:
            pass

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
