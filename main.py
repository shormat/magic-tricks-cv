"""Apply the magic tricks to the annotated videos and assemble the final cut.

Examples:
    python main.py --groups 3 8 6 10 --mode Dynamic
    python main.py --groups 2 --mode Static --prefix-video-name --tricks 2
    python main.py --groups 0 --tricks 1 --preview
"""

import argparse
import logging
from pathlib import Path

from magic_tricks.annotation import Annotation
from magic_tricks.assemble import assemble
from magic_tricks.detection import MODEL_ID, GroundingDino
from magic_tricks.pipeline import TrickPipeline, output_path_for

ALL_TRICKS = (1, 2, 3)
DEFAULT_OVERLAY = Path(__file__).parent / "assets" / "gojo_green_screen.mp4"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--groups", type=int, nargs="+", required=True,
                        help="group ids; reads annotations_group_<id>.json")
    parser.add_argument("--mode", choices=["Static", "Dynamic"], default="Static",
                        help="fixed camera (Static) or moving camera (Dynamic)")
    parser.add_argument("--tricks", type=int, nargs="+", choices=ALL_TRICKS, default=list(ALL_TRICKS),
                        help="tricks to (re)process")
    parser.add_argument("--threads", type=int, default=8, help="number of processing threads")
    parser.add_argument("--annotations", default="annotations_group_{}.json",
                        help="annotation file pattern, {} is replaced by the group id")
    parser.add_argument("--prefix-video-name", action="store_true",
                        help='video files are named "video_<video_name>"')
    parser.add_argument("--overlay", default=str(DEFAULT_OVERLAY),
                        help="green-screen video replacing the figurine in trick 2")
    parser.add_argument("--model", default=MODEL_ID, help="Grounding DINO model id")
    parser.add_argument("--preview", action="store_true",
                        help="show the result in a window instead of writing files (press q to stop)")
    parser.add_argument("--no-assemble", action="store_true", help="do not build the final video")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    for name in ("httpx", "httpx2"):  # Hugging Face Hub HTTP request logs
        logging.getLogger(name).setLevel(logging.WARNING)

    detector = GroundingDino(args.model)

    for group_id in args.groups:
        annotation_file = args.annotations.format(group_id)
        for trick_id in args.tricks:
            logging.info("=== Group %d, trick %d, %s camera ===", group_id, trick_id, args.mode)
            TrickPipeline(
                annotation_file, trick_id, args.mode,
                detector=detector,
                nthreads=args.threads,
                prefix_video_name=args.prefix_video_name,
                preview=args.preview,
                overlay_path=args.overlay,
            ).run()

        if args.preview or args.no_assemble:
            continue

        # Assemble with every trick available, including ones processed in earlier runs.
        annotation = Annotation(annotation_file, args.mode, args.prefix_video_name)
        assemble(
            source_video=annotation.get_video_name(),
            intro_end=annotation.get_start_time(1),
            segments=[output_path_for(annotation_file, t, args.mode) for t in ALL_TRICKS],
            output=f"final_output_group_{group_id}_{args.mode}.mp4",
        )


if __name__ == "__main__":
    main()
