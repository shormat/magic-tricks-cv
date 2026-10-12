# Magic Tricks — computer-vision magic on video

A Python pipeline that adds **magic effects** to recorded videos: objects vanish, change color, grow, or follow a magic wand.
Objects are found with **Grounding DINO** zero-shot detection (Hugging Face Transformers). The effects are built with **OpenCV**, and **ArUco markers** track the table when the camera moves.

## The tricks

| # | Trick | Effect |
|---|-------|--------|
| 1 | **Invisible napkin** | The green napkin on the table is replaced by the background as long as the ball, the can and the figurine are not all visible. |
| 2 | **Transforming wand** | Each touch of the wand changes an object: the ball turns green then grey, the can becomes giant, and the figurine is replaced by a green-screen video. |
| 3 | **Telekinesis** | After a touch, the ball is erased from its real position and redrawn so it follows the wand. With a moving camera, its position is computed on the table plane (homography). |

Every trick runs in two modes:

- **Static**: fixed camera. The first frame of the video is the background plate.
- **Dynamic**: moving camera. The background plate is aligned to each frame with a homography computed from ArUco markers (`DICT_4X4_50`, ids 0–3) on the table corners.

## Architecture

```
reader ─▶ detector (GPU) ─▶ N "analyze" workers ─▶ "render" (in order) ─▶ writer / preview
```

- The **detector** runs Grounding DINO on every frame.
- The **analyze** workers extract object and marker positions in parallel.
- **render** applies the effect frame by frame, in order, because tricks keep state between frames (touch counts, ball position, etc.).
- Each stage sends an end-of-stream signal to the next one, so every thread stops on its own.

```
main.py                     CLI: processing + final cut
magic_tricks/
├── annotation.py           annotation file reader
├── detection.py            Grounding DINO wrapper
├── pipeline.py             multi-threaded pipeline
├── geometry.py             boxes, contours, ArUco, homography
├── effects.py              image effects (recoloring, green-screen compositing, scaling…)
├── assemble.py             ffmpeg assembly (intro + tricks)
└── tricks/                 one module per trick (static + dynamic)
```

## Installation

Requirements: Python ≥ 3.10, [uv](https://docs.astral.sh/uv/), and `ffmpeg` on your `PATH`.

```bash
git clone https://github.com/shormat/magic-tricks-cv.git
cd magic-tricks-cv
uv sync
```

The model uses a GPU automatically when one is available. Otherwise it runs on the CPU, which is much slower.

## Input data

Each group needs an `annotations_group_<id>.json` file:

```json
{
  "video_fixed": {
    "video_name": "group2_fixed.mp4",
    "illusions": [
      {"name": "Invisible napkin", "start_time": "00:00:12"},
      {"name": "Transforming wand", "start_time": "00:01:05"},
      {"name": "Telekinesis",       "start_time": "00:02:10"}
    ]
  },
  "video_dynamic": { "video_name": "group2_moving.mp4", "illusions": [ ... ] }
}
```

Each trick runs from its `start_time` to the start of the next trick, or to the end of the video for the last one.
Trick 2 needs a green-screen video (`video.mp4` by default, see `--overlay`).

## Usage

```bash
# All tricks, moving camera, several groups
uv run main.py --groups 3 8 6 10 --mode Dynamic

# Fixed camera, video files named "video_<name>"
uv run main.py --groups 2 --mode Static --prefix-video-name

# Reprocess only trick 3; the final cut reuses tricks 1 and 2 from earlier runs
uv run main.py --groups 3 --mode Dynamic --tricks 3

# Live preview without writing files (press q to quit)
uv run main.py --groups 0 --tricks 1 --preview
```

Outputs:

- `output_trick_<trick>_<mode>_<group>.mp4`: each processed trick (1080p, 30 fps);
- `final_output_group_<group>_<mode>.mp4`: the original intro followed by the three tricks, normalized to H.264/AAC.

Main options: `--threads`, `--overlay`, `--model`, `--annotations`, `--no-assemble`, `-v`. Run `uv run main.py --help` for the full list.

## Authors

- [@shormat](https://github.com/shormat)
- [@CorentinXM](https://github.com/CorentinXM)
- [@GuenfoudiIhabe](https://github.com/GuenfoudiIhabe)
