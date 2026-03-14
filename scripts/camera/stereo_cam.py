# stereo_cam.py
# Headless stereo capture (IMX219 dual via Argus) with pair saving on ENTER.
#
# RUN (important for NoMachine):
#   env -u DISPLAY -u WAYLAND_DISPLAY -u XAUTHORITY python3 stereo_cam.py
#
# Controls (type then Enter):
#   [Enter]  -> save a pair (left/right) as PNG
#   q        -> quit
#
# Notes:
# - Default is 1920x1080@30 for better quality than 1280x720.
# - If you want maximum detail snapshots, switch SNAPSHOT_MODE="fullres" below.

import os
import sys
import time
import select
from datetime import datetime
import cv2


# -------------------- CONFIG --------------------
# Video streaming (what you continuously read)
STREAM_W, STREAM_H, STREAM_FPS = 1920, 1080, 30

# Snapshot options:
#   "stream"  -> save at STREAM_* resolution (fast)
#   "fullres" -> save a single-frame full resolution (slow, best for sharpness check)
SNAPSHOT_MODE = "stream"  # change to "fullres" when you need max detail

# Full-res mode for IMX219 (from your Argus list):
FULL_W, FULL_H, FULL_FPS = 3280, 2464, 21
FULL_SENSOR_MODE = 0      # corresponds to 3280x2464@21 on your system

# 1080p mode often is sensor-mode=2 on IMX219, but it can vary.
# If you are unsure or get negotiation issues, set STREAM_SENSOR_MODE=None.
STREAM_SENSOR_MODE = None  # e.g. 2, or None to auto

OUT_DIR = "stereo_pairs"
SAVE_FORMAT = "png"  # "png" or "jpg"
JPG_QUALITY = 95     # used only if SAVE_FORMAT=="jpg"
# -----------------------------------------------


def make_argus_to_bgr_pipeline(sensor_id: int, w: int, h: int, fps: int, sensor_mode=None) -> str:
    sm = f" sensor-mode={sensor_mode}" if sensor_mode is not None else ""
    return (
        f"nvarguscamerasrc sensor-id={sensor_id}{sm} ! "
        f"video/x-raw(memory:NVMM),width={w},height={h},framerate={fps}/1,format=NV12 ! "
        "nvvidconv ! video/x-raw,format=BGRx ! "
        "videoconvert ! video/x-raw,format=BGR ! "
        "appsink drop=1 max-buffers=1 sync=false"
    )


def open_cam(sensor_id: int, w: int, h: int, fps: int, sensor_mode=None) -> cv2.VideoCapture:
    pipe = make_argus_to_bgr_pipeline(sensor_id, w, h, fps, sensor_mode=sensor_mode)
    cap = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open sensor-id={sensor_id}\nPipeline:\n{pipe}")
    return cap


def warmup(cap: cv2.VideoCapture, n: int = 20) -> None:
    for _ in range(n):
        cap.read()


def stdin_key_nonblocking(timeout_s: float = 0.0):
    r, _, _ = select.select([sys.stdin], [], [], timeout_s)
    if not r:
        return None
    line = sys.stdin.readline()
    if line == "":
        return None
    line = line.strip().lower()
    if line == "":
        return "ENTER"
    if line == "q":
        return "q"
    return None


def save_image(path: str, frame) -> bool:
    if SAVE_FORMAT.lower() == "jpg":
        return cv2.imwrite(path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(JPG_QUALITY)])
    return cv2.imwrite(path, frame)


def snapshot_pair_fullres(pair_idx: int) -> None:
    # Open full-res streams only for the snapshot (slower, but max detail).
    left = open_cam(0, FULL_W, FULL_H, FULL_FPS, sensor_mode=FULL_SENSOR_MODE)
    right = open_cam(1, FULL_W, FULL_H, FULL_FPS, sensor_mode=FULL_SENSOR_MODE)

    warmup(left, 8)
    warmup(right, 8)

    retL, frameL = left.read()
    retR, frameR = right.read()

    left.release()
    right.release()

    if not retL or frameL is None:
        print("Fullres left grab failed")
        return
    if not retR or frameR is None:
        print("Fullres right grab failed")
        return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    ext = SAVE_FORMAT.lower()
    lp = os.path.join(OUT_DIR, f"left_full_{pair_idx:04d}_{ts}.{ext}")
    rp = os.path.join(OUT_DIR, f"right_full_{pair_idx:04d}_{ts}.{ext}")

    ok1 = save_image(lp, frameL)
    ok2 = save_image(rp, frameR)
    if ok1 and ok2:
        print(f"Saved FULLRES pair {pair_idx:04d}:")
        print(f"  {lp}")
        print(f"  {rp}")
    else:
        print("Failed to save fullres images")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Opening stereo cameras...")
    left = open_cam(0, STREAM_W, STREAM_H, STREAM_FPS, sensor_mode=STREAM_SENSOR_MODE)
    right = open_cam(1, STREAM_W, STREAM_H, STREAM_FPS, sensor_mode=STREAM_SENSOR_MODE)

    warmup(left, 20)
    warmup(right, 20)

    print("OK. Capturing.")
    print("Press ENTER then Enter to save a pair. Type 'q' then Enter to quit.")
    print(f"Stream: {STREAM_W}x{STREAM_H}@{STREAM_FPS}, snapshot_mode={SNAPSHOT_MODE}, format={SAVE_FORMAT}")
    print(f"Output folder: {os.path.abspath(OUT_DIR)}")

    pair_idx = 0
    frames = 0
    t0 = time.time()
    last_report = t0

    while True:
        retL, frameL = left.read()
        retR, frameR = right.read()

        if not retL or frameL is None:
            print("Left grab failed")
            break
        if not retR or frameR is None:
            print("Right grab failed")
            break

        frames += 1
        now = time.time()
        if now - last_report >= 2.0:
            fps_now = frames / (now - t0)
            print(f"Running... approx FPS: {fps_now:.1f}")
            last_report = now

        key = stdin_key_nonblocking(0.0)
        if key == "q":
            print("Quit.")
            break

        if key == "ENTER":
            if SNAPSHOT_MODE == "fullres":
                snapshot_pair_fullres(pair_idx)
                pair_idx += 1
            else:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                ext = SAVE_FORMAT.lower()
                lp = os.path.join(OUT_DIR, f"left_{pair_idx:04d}_{ts}.{ext}")
                rp = os.path.join(OUT_DIR, f"right_{pair_idx:04d}_{ts}.{ext}")

                ok1 = save_image(lp, frameL)
                ok2 = save_image(rp, frameR)
                if ok1 and ok2:
                    print(f"Saved pair {pair_idx:04d}:")
                    print(f"  {lp}")
                    print(f"  {rp}")
                    pair_idx += 1
                else:
                    print("Failed to save images (check disk permissions/space).")

        time.sleep(0.001)

    left.release()
    right.release()


if __name__ == "__main__":
    main()
