"""Image effects applied in place on frames."""

import logging

import cv2
import numpy as np

from .geometry import clip_box

log = logging.getLogger(__name__)

# HSV ranges (OpenCV hue scale 0-180)
BLUE_BALL_HSV = (np.array([100, 50, 50]), np.array([140, 255, 255]))
BLUE_BALL_HSV_WIDE = (np.array([90, 30, 30]), np.array([150, 255, 255]))
GREEN_SCREEN_HSV = (np.array([40, 40, 40]), np.array([80, 255, 255]))

_KERNEL_5 = np.ones((5, 5), np.uint8)


def color_mask(image, hsv_range, clean=False):
    """Binary mask of the pixels within an HSV range, optionally opened then closed."""
    mask = cv2.inRange(cv2.cvtColor(image, cv2.COLOR_BGR2HSV), *hsv_range)
    if clean:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _KERNEL_5)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _KERNEL_5)
    return mask


def recolor_ball(frame, box, mode):
    """Turn the blue pixels inside the box green (mode="green") or grey (mode="gray")."""
    box = clip_box(box, frame.shape)
    if box is None:
        return
    x1, y1, x2, y2 = box
    roi = frame[y1:y2, x1:x2]
    blue = color_mask(roi, BLUE_BALL_HSV) > 0
    if mode == "green":
        roi[blue, 0] = 0
        roi[blue, 2] = 0
    elif mode == "gray":
        gray = cv2.cvtColor(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        roi[blue] = gray[blue]
    else:
        raise ValueError(f"unknown mode {mode!r}")


def paste_masked(frame, patch, mask, center):
    """Copy the masked pixels of `patch` onto `frame`, centred on `center` and clipped to the frame."""
    ph, pw = mask.shape[:2]
    top, left = int(center[1]) - ph // 2, int(center[0]) - pw // 2
    fh, fw = frame.shape[:2]
    y1, x1 = max(0, top), max(0, left)
    y2, x2 = min(fh, top + ph), min(fw, left + pw)
    if y2 <= y1 or x2 <= x1:
        return
    src = (slice(y1 - top, y2 - top), slice(x1 - left, x2 - left))
    keep = mask[src] > 0
    frame[y1:y2, x1:x2][keep] = patch[src][keep]


def restore_background(frame, box, background):
    """Paint the background plate over a box."""
    box = clip_box(box, frame.shape)
    if box is not None:
        x1, y1, x2, y2 = box
        frame[y1:y2, x1:x2] = background[y1:y2, x1:x2]


def composite_green_screen(frame, box, overlay, background):
    """Replace a box with `overlay`, keying out its green pixels to show the background plate."""
    x1, y1, x2, y2 = box
    plate = background[y1:y2, x1:x2]
    out = cv2.resize(overlay, (x2 - x1, y2 - y1))
    green = color_mask(out, GREEN_SCREEN_HSV) > 0
    out[green] = plate[green]
    frame[y1:y2, x1:x2] = out


def enlarge_object(frame, box, background, diff_threshold):
    """Draw the object in `box` three times bigger, centred on itself.

    Pixels that barely differ from the background plate are treated as see-through.
    """
    box = clip_box(box, frame.shape)
    if box is None:
        return
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1

    obj = frame[y1:y2, x1:x2]
    diff = cv2.cvtColor(cv2.absdiff(obj, background[y1:y2, x1:x2]), cv2.COLOR_BGR2GRAY)
    big = cv2.resize(obj, (w * 3, h * 3))
    see_through = cv2.resize(diff, (w * 3, h * 3)) < diff_threshold

    fh, fw = frame.shape[:2]
    ys, ye = max(0, y1 - h), min(fh, y2 + h)
    xs, xe = max(0, x1 - w), min(fw, x2 + w)
    region = frame[ys:ye, xs:xe]
    if region.shape == big.shape:
        big[see_through] = region[see_through]
    frame[ys:ye, xs:xe] = cv2.resize(big, (xe - xs, ye - ys))


class LoopingVideo:
    """Serves the frames of a video file forever, rewinding at the end."""

    def __init__(self, path):
        self.path = path
        self._cap = None
        self._warned = False

    def next_frame(self):
        if self._cap is None:
            self._cap = cv2.VideoCapture(self.path)
        ok, frame = self._cap.read()
        if not ok:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self._cap.read()
        if not ok:
            if not self._warned:
                log.error("Cannot read overlay video %s", self.path)
                self._warned = True
            return None
        return frame

    def release(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None
