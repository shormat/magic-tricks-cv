from . import trick1, trick2, trick3
from .base import OUTPUT_FPS, TrickContext, TrickSpec

TRICKS = {
    (1, "Static"): trick1.STATIC,
    (1, "Dynamic"): trick1.DYNAMIC,
    (2, "Static"): trick2.STATIC,
    (2, "Dynamic"): trick2.DYNAMIC,
    (3, "Static"): trick3.STATIC,
    (3, "Dynamic"): trick3.DYNAMIC,
}


def get_trick(trick_id, video_type):
    try:
        return TRICKS[(trick_id, video_type)]
    except KeyError:
        raise ValueError(f"No trick {trick_id} for {video_type} videos") from None


__all__ = ["OUTPUT_FPS", "TRICKS", "TrickContext", "TrickSpec", "get_trick"]
