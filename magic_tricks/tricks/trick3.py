"""Trick 3: telekinesis.

Once the wand gets close to the ball, the ball is erased from its real position
and drawn so that it follows the wand.
"""

import logging

import numpy as np

from ..detection import best_per_label
from ..effects import BLUE_BALL_HSV_WIDE, color_mask, erase_masked, paste_masked
from ..geometry import box_center, clamp_point_to_polygon, clip_box, distance, project_point, suppress_overlaps, to_int_box
from .base import TrickSpec

log = logging.getLogger(__name__)

PROMPT = (
    "a blue ball . a magic wand . a overhead projector. a projector arm . an alimentation cable . "
    "a black circle . a coat rack . a coat hanger . a coat ."
)
TEXT_THRESHOLD = 0.20
DETECTION_THRESHOLD = 0.3

STATIC_TOUCH_DISTANCE = 250
DYNAMIC_TOUCH_DISTANCE = 350
SMOOTHING_FACTOR = 0.4  # lower = smoother but slower to respond


def _boxes_by_label(detections):
    return {det.label: to_int_box(det.box) for det in detections}


def analyze_static(frame, detections, ctx):
    boxes = _boxes_by_label(best_per_label(detections, DETECTION_THRESHOLD))
    wand_box, ball_box = boxes.get("a magic wand"), boxes.get("a blue ball")
    wand = ((wand_box[0] + wand_box[2]) // 2, wand_box[3]) if wand_box else None  # tip: bottom centre
    ball = box_center(ball_box) if ball_box else None

    corners, ids = ctx.markers.detect(frame)
    return {
        "corners": corners,
        "ids": ids,
        "wand": wand,
        "ball": ball,
        # The ball is often hidden by the hand: reuse the last box seen.
        "ball_box": ball_box if ball_box is not None else ctx.state.get("last_ball_box"),
    }


def analyze_dynamic(frame, detections, ctx):
    boxes = _boxes_by_label(suppress_overlaps(best_per_label(detections, DETECTION_THRESHOLD)))
    wand_box, ball_box = boxes.get("a magic wand"), boxes.get("a blue ball")
    wand = box_center(wand_box) if wand_box else None
    ball = box_center(ball_box) if ball_box else None

    corners, ids = ctx.markers.detect(frame)
    return {"corners": corners, "ids": ids, "wand": wand, "ball": ball, "ball_box": ball_box}


def _cut_ball(frame, box):
    x1, y1, x2, y2 = box
    patch = frame[y1:y2, x1:x2].copy()
    return patch, color_mask(patch, BLUE_BALL_HSV_WIDE)


def render_static(frame, frame_num, data, ctx):
    state = ctx.state
    roi = ctx.markers.update_roi(data["corners"], data["ids"])
    prev_wand = state.get("last_wand")
    wand, ball = data["wand"], data["ball"]

    if wand is not None:
        state["last_wand"] = wand
        if not state["touched"] and ball is not None and distance(wand, ball) < STATIC_TOUCH_DISTANCE:
            state["touched"] = True
            state["ball_pos"] = ball
            log.info("Wand touched the ball at frame %d", frame_num)
    else:
        wand = prev_wand

    ball_box = data["ball_box"]
    if ball_box is not None:
        state["last_ball_box"] = ball_box

    if not state["touched"]:
        return frame

    # Move the ball like the wand; vertical moves are scaled down to follow the table perspective.
    dx = dy = 0.0
    if prev_wand is not None and wand is not None:
        dx, dy = wand[0] - prev_wand[0], wand[1] - prev_wand[1]
        if roi is not None:
            roi_height = roi[:, 1].max() - roi[:, 1].min()
            dy *= roi_height / (frame.shape[0] - roi[:, 1].min() + 100)
    state["ball_pos"] = (state["ball_pos"][0] + dx, state["ball_pos"][1] + dy)

    ball_box = clip_box(ball_box, frame.shape) if ball_box is not None else None
    if wand is not None and ball_box is not None:
        patch, mask = _cut_ball(frame, ball_box)
        erase_masked(frame, ball_box, mask)
        paste_masked(frame, patch, mask, state["ball_pos"])
    return frame


def render_dynamic(frame, frame_num, data, ctx):
    state = ctx.state
    roi = ctx.markers.update_roi(data["corners"], data["ids"])
    h_matrix = ctx.markers.homography(data["corners"], data["ids"])
    wand, ball = data["wand"], data["ball"]
    ball_box = clip_box(data["ball_box"], frame.shape) if data["ball_box"] is not None else None

    if wand is not None:
        state["last_wand"] = wand
        if (not state["touched"] and ball is not None and ball_box is not None and h_matrix is not None
                and distance(wand, ball) < DYNAMIC_TOUCH_DISTANCE):
            # Remember the ball's look and its offset from the wand on the table plane.
            state["touched"] = True
            state["ball_patch"], state["ball_mask"] = _cut_ball(frame, ball_box)
            to_table = np.linalg.inv(h_matrix)
            state["ball_offset"] = project_point(ball, to_table) - project_point(wand, to_table)
            log.info("Wand touched the ball at frame %d", frame_num)
    else:
        wand = state.get("last_wand")

    if not (state["touched"] and wand is not None and h_matrix is not None and ball_box is not None):
        return frame

    ball_on_table = project_point(wand, np.linalg.inv(h_matrix)) + state["ball_offset"]
    target = project_point(ball_on_table, h_matrix)
    smoothed = state.get("smoothed_pos")
    smoothed = target if smoothed is None else SMOOTHING_FACTOR * target + (1 - SMOOTHING_FACTOR) * smoothed
    state["smoothed_pos"] = smoothed

    position = clamp_point_to_polygon((int(smoothed[0]), int(smoothed[1])), roi)
    erase_masked(frame, ball_box, _cut_ball(frame, ball_box)[1])
    paste_masked(frame, state["ball_patch"], state["ball_mask"], position)
    return frame


def _init_state(ctx):
    return {"touched": False}


STATIC = TrickSpec(PROMPT, TEXT_THRESHOLD, analyze_static, render_static,
                   uses_markers=True, init_state=_init_state)
DYNAMIC = TrickSpec(PROMPT, TEXT_THRESHOLD, analyze_dynamic, render_dynamic,
                    uses_markers=True, init_state=_init_state)
