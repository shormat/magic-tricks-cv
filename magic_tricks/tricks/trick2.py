"""Trick 2: the magic wand transforms objects.

Each touch of the wand advances an object's state:
  * ball: 1st touch turns it green, 2nd turns it grey, 3rd restores it;
  * can (after the ball's 2nd touch): 1st touch makes it giant, 2nd restores it;
  * figurine (after the can): replaced by an animated green-screen video.
"""

import logging
from dataclasses import dataclass

from ..detection import best_per_label
from ..effects import composite_green_screen, enlarge_object, recolor_ball
from ..geometry import box_center, clip_box, distance, pad_box, suppress_overlaps, to_int_box
from .base import TrickSpec

log = logging.getLogger(__name__)

PROMPT = "a soda can . a blue ball . a small figurine . a magic wand . a black circle ."
DETECTION_THRESHOLD = 0.3
FIGURINE_PADDING = 20


@dataclass(frozen=True)
class Params:
    text_threshold: float
    touch_cooldown: float  # seconds between two touches of the same object
    can_touch_distance: float
    figurine_min_can_state: int
    can_diff_threshold: int


STATIC_PARAMS = Params(text_threshold=0.20, touch_cooldown=1, can_touch_distance=150,
                       figurine_min_can_state=1, can_diff_threshold=10)
DYNAMIC_PARAMS = Params(text_threshold=0.3, touch_cooldown=2, can_touch_distance=80,
                        figurine_min_can_state=2, can_diff_threshold=15)
TOUCH_DISTANCE = 80


def _object_name(label):
    if label == "a magic wand":
        return "wand"
    if label == "a blue ball":
        return "ball"
    if label == "a small figurine":
        return "figurine"
    if "soda" in label.lower() or "can" in label.lower():
        return "can"
    return None


def _anchors(name, box, dynamic):
    """Points used to measure the wand-object distance.

    With a moving camera, boxes are less stable, so several corners are tried.
    """
    x1, y1, x2, y2 = box
    cx, cy = box_center(box)
    if dynamic:
        return {
            "wand": [(x2, y2), (x1, y2)],
            "ball": [(x1, y1), (x2, y1)],
            "figurine": [(x1, y1), (x2, y1), (cx, y1)],
            "can": [(x1, y1), (x2, y1)],
        }[name]
    return {"wand": [(cx, y2)], "ball": [(cx, cy)], "figurine": [(cx, y1)], "can": [(cx, y1)]}[name]


def _analyze(frame, detections, ctx, dynamic):
    selected = best_per_label(detections, DETECTION_THRESHOLD)
    if dynamic:
        selected = suppress_overlaps(selected)

    boxes = {}
    for det in selected:
        name = _object_name(det.label)
        if name is not None:
            boxes[name] = to_int_box(det.box)

    distances = {}
    if "wand" in boxes:
        wand_points = _anchors("wand", boxes["wand"], dynamic)
        for name in ("ball", "figurine", "can"):
            if name in boxes:
                distances[name] = min(distance(w, p) for w in wand_points
                                      for p in _anchors(name, boxes[name], dynamic))

    data = {"boxes": boxes, "distances": distances}
    if ctx.markers is not None:
        data["corners"], data["ids"] = ctx.markers.detect(frame)
    return data


def analyze_static(frame, detections, ctx):
    return _analyze(frame, detections, ctx, dynamic=False)


def analyze_dynamic(frame, detections, ctx):
    return _analyze(frame, detections, ctx, dynamic=True)


def _update_states(data, state, now, params):
    """Count a touch when the wand gets close to an object, with a cooldown per object."""
    dist = data["distances"]
    rules = (
        ("ball", TOUCH_DISTANCE, True),
        ("figurine", TOUCH_DISTANCE, state["can"] >= params.figurine_min_can_state),
        ("can", params.can_touch_distance, state["ball"] >= 2),
    )
    for name, max_distance, unlocked in rules:
        if (unlocked and name in dist and dist[name] < max_distance
                and now - state["last_touch"][name] > params.touch_cooldown):
            state[name] += 1
            state["last_touch"][name] = now
            log.info("Wand touched the %s (state %d) at %.1fs", name, state[name], now)


def _render(frame, frame_num, data, ctx, params):
    state = ctx.state
    _update_states(data, state, frame_num / ctx.fps, params)
    boxes = data["boxes"]

    if "ball" in boxes and state["ball"] in (1, 2):
        recolor_ball(frame, boxes["ball"], "green" if state["ball"] == 1 else "gray")

    needs_figurine = state["figurine"] == 1 and "figurine" in boxes
    needs_can = state["can"] == 1 and "can" in boxes
    if not (needs_figurine or needs_can):
        return frame

    background = ctx.first_frame
    if ctx.markers is not None:
        background, _ = ctx.markers.warped_background(ctx.first_frame, data["corners"], data["ids"])

    if needs_figurine:
        box = clip_box(pad_box(boxes["figurine"], FIGURINE_PADDING), frame.shape)
        overlay = ctx.overlay.next_frame()
        if box is not None and overlay is not None:
            composite_green_screen(frame, box, overlay, background)

    if needs_can:
        enlarge_object(frame, boxes["can"], background, params.can_diff_threshold)

    return frame


def render_static(frame, frame_num, data, ctx):
    return _render(frame, frame_num, data, ctx, STATIC_PARAMS)


def render_dynamic(frame, frame_num, data, ctx):
    return _render(frame, frame_num, data, ctx, DYNAMIC_PARAMS)


def _init_state(ctx):
    return {"ball": 0, "can": 0, "figurine": 0, "last_touch": {"ball": 0.0, "can": 0.0, "figurine": 0.0}}


STATIC = TrickSpec(PROMPT, STATIC_PARAMS.text_threshold, analyze_static, render_static,
                   init_state=_init_state)
DYNAMIC = TrickSpec(PROMPT, DYNAMIC_PARAMS.text_threshold, analyze_dynamic, render_dynamic,
                    uses_markers=True, init_state=_init_state)
