from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

from ..effects import LoopingVideo
from ..geometry import MarkerTracker

OUTPUT_FPS = 30.0


@dataclass
class TrickContext:
    """Everything a trick needs besides the current frame."""

    first_frame: np.ndarray
    markers: Optional[MarkerTracker] = None
    overlay: Optional[LoopingVideo] = None
    fps: float = OUTPUT_FPS
    # Mutable per-video state. Only `render` writes to it.
    state: dict = field(default_factory=dict)


@dataclass(frozen=True)
class TrickSpec:
    """A trick is split in two stages.

    analyze(frame, detections, ctx) -> data
        Runs on several worker threads in any order; must not write to ctx.state.
    render(frame, frame_num, data, ctx) -> frame
        Runs on a single thread, in frame order; may update ctx.state.
    """

    prompt: str
    text_threshold: float
    analyze: Callable[..., Any]
    render: Callable[..., np.ndarray]
    uses_markers: bool = False
    init_state: Callable[[TrickContext], dict] = lambda ctx: {}
