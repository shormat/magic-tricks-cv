"""Blur people's heads in videos before sharing them.

Run it on the source videos, then run main.py on the blurred copies: the intro,
the tricks and the background plates are all anonymized, while the effects
added afterwards (e.g. the green-screen overlay) stay sharp.

Example:
    python blur_faces.py videos/*.mp4 --out-dir blurred/
"""

import argparse
import logging
from pathlib import Path

from magic_tricks.anonymize import blur_heads
from magic_tricks.detection import MODEL_ID, GroundingDino


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("videos", nargs="+", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True, help="where to write the blurred videos")
    parser.add_argument("--stride", type=int, default=3, help="detect heads on every n-th frame")
    parser.add_argument("--hold", type=float, default=0.4,
                        help="seconds a detected head stays blurred before and after its detection")
    parser.add_argument("--model", default=MODEL_ID, help="Grounding DINO model id")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    for name in ("httpx", "httpx2"):  # Hugging Face Hub HTTP request logs
        logging.getLogger(name).setLevel(logging.WARNING)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    detector = GroundingDino(args.model)
    for video in args.videos:
        out = args.out_dir / video.name
        if out.resolve() == video.resolve():
            raise SystemExit(f"Refusing to overwrite {video}")
        blur_heads(str(video), str(out), detector, stride=args.stride, hold=args.hold)


if __name__ == "__main__":
    main()
