"""Blur people's heads in a video, so it can be shared without their consent.

Two passes: heads are detected on every `stride`-th frame, then every frame is
blurred with the boxes found within `hold` seconds around it. The time window
and the enlarged boxes cover frames where the detector misses a head.
"""

import logging
import subprocess

import cv2
import numpy as np

from .geometry import box_area, box_center, clip_box, point_in_box, to_int_box

log = logging.getLogger(__name__)

PROMPT = "a human head . a face . a person ."
TEXT_THRESHOLD = 0.25
CONFIDENT_HEAD = 0.45  # accepted on its own
PERSON_THRESHOLD = 0.3
HEAD_ZONE = 0.4  # a weaker head is accepted in the top 40% of a person box
MAX_HEAD_AREA_RATIO = 0.05  # larger "faces" are false positives (e.g. the table)
GROW = 0.25


def find_heads(detections, frame_shape):
    """Head boxes among the detections, ignoring head-like objects such as figurines."""
    frame_area = frame_shape[0] * frame_shape[1]
    persons = [to_int_box(d.box) for d in detections
               if "person" in d.label and d.score >= PERSON_THRESHOLD]
    heads = []
    for det in detections:
        box = to_int_box(det.box)
        if not ("head" in det.label or "face" in det.label):
            continue
        if box_area(box) / frame_area > MAX_HEAD_AREA_RATIO:
            continue
        cx, cy = box_center(box)
        in_head_zone = any(
            point_in_box((cx, cy), p) and cy <= p[1] + HEAD_ZONE * (p[3] - p[1]) for p in persons
        )
        if det.score >= CONFIDENT_HEAD or in_head_zone:
            heads.append(box)
    return heads


def grow_box(box, ratio):
    x1, y1, x2, y2 = box
    dx, dy = int((x2 - x1) * ratio / 2), int((y2 - y1) * ratio / 2)
    return x1 - dx, y1 - dy, x2 + dx, y2 + dy


def blur_box(frame, box):
    """Pixelate then blur a box, in place, strongly enough to hide a face."""
    box = clip_box(grow_box(box, GROW), frame.shape)
    if box is None:
        return
    x1, y1, x2, y2 = box
    roi = frame[y1:y2, x1:x2]
    small = cv2.resize(roi, (max(1, (x2 - x1) // 12), max(1, (y2 - y1) // 12)), interpolation=cv2.INTER_AREA)
    pixelated = cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
    k = max(3, ((x2 - x1) // 4) | 1)
    frame[y1:y2, x1:x2] = cv2.GaussianBlur(pixelated, (k, k), 0)


def detect_heads(path, detector, stride):
    """{frame index: [head boxes]} for every `stride`-th frame."""
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    heads, index = {}, 0
    while True:
        if index % stride == 0:
            ok, frame = cap.read()
            if not ok:
                break
            heads[index] = find_heads(detector(frame, PROMPT, TEXT_THRESHOLD), frame.shape)
            if len(heads) % 100 == 0:
                log.info("Detected heads on %d/%d frames", index, total)
        elif not cap.grab():
            break
        index += 1
    cap.release()
    return heads


def _frame_rate(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=r_frame_rate", "-of", "csv=p=0", path],
                         capture_output=True, text=True, check=True)
    return out.stdout.strip()


def blur_heads(src, dst, detector, stride=3, hold=0.4):
    """Write `dst`: `src` with every detected head blurred, audio kept as is."""
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    log.info("Pass 1/2: detecting heads in %s", src)
    heads = detect_heads(src, detector, stride)
    window = int(round(hold * fps))
    log.info("Found heads on %d of %d sampled frames", sum(1 for b in heads.values() if b), len(heads))

    log.info("Pass 2/2: writing %s", dst)
    ffmpeg = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", _frame_rate(src), "-i", "-",
         "-i", src, "-map", "0:v", "-map", "1:a?",
         "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p", "-c:a", "copy", dst],
        stdin=subprocess.PIPE,
    )
    cap = cv2.VideoCapture(src)
    index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            start = (index - window) // stride * stride
            for j in range(max(0, start), index + window + 1, stride):
                for box in heads.get(j, ()):
                    blur_box(frame, box)
            ffmpeg.stdin.write(np.ascontiguousarray(frame).tobytes())
            index += 1
    finally:
        cap.release()
        ffmpeg.stdin.close()
        if ffmpeg.wait() != 0:
            raise RuntimeError(f"ffmpeg failed writing {dst}")
    log.info("Saved %d frames to %s", index, dst)
