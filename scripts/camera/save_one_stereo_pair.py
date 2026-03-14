import os
from datetime import datetime
import cv2

STREAM_W, STREAM_H, STREAM_FPS = 1920, 1080, 30
STREAM_SENSOR_MODE = None

OUT_DIR = "/home/warxen/ai_robot/data/stereo_pairs"
SAVE_FORMAT = "png"
JPG_QUALITY = 95


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


def save_image(path: str, frame) -> bool:
    if SAVE_FORMAT.lower() == "jpg":
        return cv2.imwrite(path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(JPG_QUALITY)])
    return cv2.imwrite(path, frame)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Opening stereo cameras...")
    left = open_cam(0, STREAM_W, STREAM_H, STREAM_FPS, sensor_mode=STREAM_SENSOR_MODE)
    right = open_cam(1, STREAM_W, STREAM_H, STREAM_FPS, sensor_mode=STREAM_SENSOR_MODE)

    warmup(left, 20)
    warmup(right, 20)

    retL, frameL = left.read()
    retR, frameR = right.read()

    left.release()
    right.release()

    if not retL or frameL is None:
        raise RuntimeError("Left grab failed")
    if not retR or frameR is None:
        raise RuntimeError("Right grab failed")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    ext = SAVE_FORMAT.lower()
    lp = os.path.join(OUT_DIR, f"left_{ts}.{ext}")
    rp = os.path.join(OUT_DIR, f"right_{ts}.{ext}")

    ok1 = save_image(lp, frameL)
    ok2 = save_image(rp, frameR)

    if not (ok1 and ok2):
        raise RuntimeError("Failed to save stereo pair")

    print("Saved stereo pair:")
    print(lp)
    print(rp)


if __name__ == "__main__":
    main()
