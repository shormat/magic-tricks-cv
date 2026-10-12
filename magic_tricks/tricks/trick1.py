"""Trick 1: the napkin disappears.

A green napkin covers objects on the table. While it lies on the table and the
ball, the can and the figurine are not all visible, the napkin is replaced by
the background plate so it looks invisible.
"""

import cv2
import numpy as np

from ..effects import color_mask
from ..geometry import box_area, box_center, contour_on_box, point_in_box, suppress_overlaps, to_int_box
from .base import TrickSpec

PROMPT = (
    "a soda can . a blue ball . a small figurine . a wooden table . a desk . a overhead projector. "
    "a projector arm . an alimentation cable . projector button . a grey cylindrical object . "
    "a black circle . a coat rack . a coat hanger . a coat ."
)
TEXT_THRESHOLD = 0.20

TABLE_THRESHOLD = 0.3
TABLE_MIN_AREA_RATIO = 0.05
OBJECT_THRESHOLD = 0.4
# Larger boxes are false positives (usually the whole table).
OBJECT_MAX_AREA_RATIO = {"ball": 0.03, "can": 0.05, "figurine": 0.05}
ALL_OBJECTS = set(OBJECT_MAX_AREA_RATIO)

NAPKIN_HSV = (np.array([25, 20, 20]), np.array([85, 255, 255]))
STATIC_BACKGROUND_HSV = (np.array([25, 40, 40]), np.array([85, 255, 255]))
MIN_NAPKIN_AREA = 1000
_DILATE_KERNEL = np.ones((15, 15), np.uint8)


def _category(label):
    label = label.lower()
    if "ball" in label:
        return "ball"
    if "soda" in label or "can" in label:
        return "can"
    if "figurine" in label:
        return "figurine"
    return None


def find_table(detections, frame_shape):
    frame_area = frame_shape[0] * frame_shape[1]
    candidates = []
    for det in detections:
        label = det.label.lower()
        box = to_int_box(det.box)
        if (det.score >= TABLE_THRESHOLD and ("table" in label or "desk" in label)
                and box_area(box) / frame_area >= TABLE_MIN_AREA_RATIO):
            candidates.append((det.score, box))
    return max(candidates, key=lambda c: c[0])[1] if candidates else None


def analyze(frame, detections, ctx):
    table = find_table(detections, frame.shape)
    frame_area = frame.shape[0] * frame.shape[1]

    objects = []
    for det in sorted(detections, key=lambda d: d.score, reverse=True):
        category = _category(det.label)
        box = to_int_box(det.box)
        if det.score < OBJECT_THRESHOLD or category is None:
            continue
        if box_area(box) / frame_area > OBJECT_MAX_AREA_RATIO[category]:
            continue
        if table is not None and not point_in_box(box_center(box), table):
            continue
        objects.append((det.score, box, category))

    found = {category for _, _, category in suppress_overlaps(objects, touching=True)}
    data = {"table": table, "all_objects_visible": found >= ALL_OBJECTS}
    if ctx.markers is not None:
        data["corners"], data["ids"] = ctx.markers.detect(frame)
    return data


def hide_napkin(frame, background, background_green, table):
    """Replace the largest new green blob lying on the table with the background."""
    moving_green = cv2.bitwise_and(color_mask(frame, NAPKIN_HSV), cv2.bitwise_not(background_green))
    kernel = np.ones((5, 5), np.uint8)
    moving_green = cv2.morphologyEx(moving_green, cv2.MORPH_OPEN, kernel)
    moving_green = cv2.morphologyEx(moving_green, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(moving_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    on_table = [c for c in contours if cv2.contourArea(c) > MIN_NAPKIN_AREA and contour_on_box(c, table)]
    if not on_table:
        return

    napkin = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(napkin, [max(on_table, key=cv2.contourArea)], 255)
    napkin = cv2.dilate(napkin, _DILATE_KERNEL) == 255
    frame[napkin] = background[napkin]


def render(frame, frame_num, data, ctx):
    if data["table"] is None or data["all_objects_visible"]:
        return frame

    if ctx.markers is None:
        background, background_green = ctx.first_frame, ctx.state["background_green"]
    else:
        background, _ = ctx.markers.warped_background(ctx.first_frame, data["corners"], data["ids"])
        background_green = color_mask(background, NAPKIN_HSV, clean=True)

    hide_napkin(frame, background, background_green, data["table"])
    return frame


def _init_static_state(ctx):
    return {"background_green": color_mask(ctx.first_frame, STATIC_BACKGROUND_HSV, clean=True)}


STATIC = TrickSpec(PROMPT, TEXT_THRESHOLD, analyze, render, init_state=_init_static_state)
DYNAMIC = TrickSpec(PROMPT, TEXT_THRESHOLD, analyze, render, uses_markers=True)
