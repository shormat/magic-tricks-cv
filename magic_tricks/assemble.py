"""Assemble the final video with ffmpeg: untouched intro + the processed trick segments."""

import logging
import subprocess
import tempfile
from pathlib import Path

log = logging.getLogger(__name__)


def _ffmpeg(*args):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args], check=True)


def has_audio(path):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    )
    return result.stdout.strip() != ""


def normalize(src, out, fps):
    """Re-encode to 1080p H.264 at `fps` + stereo AAC (adding a silent track if needed) so the parts
    can be concatenated without re-encoding."""
    args = ["-i", str(src)]
    if not has_audio(src):
        args += ["-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000", "-shortest"]
    args += ["-vf", "scale=1920:1080", "-r", str(fps), "-pix_fmt", "yuv420p",
             "-c:v", "libx264", "-crf", "18", "-preset", "slow",
             "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2", str(out)]
    _ffmpeg(*args)


def assemble(source_video, intro_end, segments, output, fps=30):
    """Concatenate source_video[0:intro_end] followed by the given segment files."""
    with tempfile.TemporaryDirectory(prefix="magic_tricks_") as tmp:
        tmp = Path(tmp)
        parts = []

        intro = tmp / "intro.mp4"
        if intro_end > 0:
            try:
                _ffmpeg("-ss", "0", "-to", str(intro_end), "-i", str(source_video), "-c", "copy", str(intro))
                parts.append(intro)
            except subprocess.CalledProcessError:
                log.warning("Could not cut the intro from %s, skipping it", source_video)

        for segment in segments:
            if Path(segment).exists():
                parts.append(Path(segment))
            else:
                log.warning("Missing segment %s, skipping it", segment)

        if not parts:
            raise FileNotFoundError("Nothing to assemble")

        normalized = []
        for i, part in enumerate(parts):
            out = tmp / f"part_{i}.mp4"
            normalize(part, out, fps)
            normalized.append(out)

        concat_list = tmp / "concat.txt"
        concat_list.write_text("".join(f"file '{p.resolve()}'\n" for p in normalized))
        _ffmpeg("-f", "concat", "-safe", "0", "-i", str(concat_list), "-c", "copy", str(output))

    log.info("Final video saved as %s", output)
