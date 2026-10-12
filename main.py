import os
from MagicTricks import MagicTricks
import subprocess
def normalize_input(src, out):
    """Re-encode input to clean H.264 + AAC 1080p, visually lossless."""
    cmd = (
        f"ffmpeg -y -i '{src}' "
        f"-vf scale=1920:1080 "
        f"-r 30 "
        f"-c:v libx264 -crf 18 -preset slow "
        f"-c:a aac -b:a 192k "
        f"'{out}'"
    )
    os.system(cmd)


def has_audio(video_path):
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "a",
        "-show_entries", "stream=index", "-of", "csv=p=0", video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout.strip() != ""


def ensure_audio(video_path):
    """Ensure the video has an audio track (silent if needed)."""
    if has_audio(video_path):
        return video_path

    out = video_path.replace(".mp4", "_audio.mp4")
    cmd = (
        f"ffmpeg -y -i '{video_path}' "
        f"-f lavfi -i anullsrc=channel_layout=stereo:sample_rate=48000 "
        f"-shortest -c:v copy -c:a aac -b:a 192k '{out}'"
    )
    os.system(cmd)
    return out

from Annotation import Annotation
if __name__ == "__main__":
    group_ids = [3,8,6,10]
    for group_id in group_ids:
        annotation_file = f"annotations_group_{group_id}.json"
        nthreads = 12
        video_types = ["Dynamic"]
        prefix_video_name = False
        trick_ids = [3]
        debug = False
        for video_type in video_types:
            for trick_id in trick_ids:
                print(f"\n=== Traitement de la astuce {trick_id} en mode {video_type} ===")
                magic_tricks = MagicTricks(annotation_file, nthreads, video_type, prefix_video_name, trick_id, debug)
                magic_tricks.run()
        video_type = "Dynamic"
        trick_ids = [1,2,3]
        output_videos = [
            f"output_trick_{trick_id}_{vt}_{group_id}.mp4"
            for vt in video_types
            for trick_id in trick_ids
        ]

        begin_end = Annotation(annotation_file, video_type=video_types[0])
        start_time = begin_end.get_start_time(trick_ids[0])
        video_name = begin_end.get_video_name()

        pre_trick_raw = f"pre_trick_raw_{group_id}.mp4"
        cmd_pre = f"ffmpeg -y -ss 0 -to {start_time} -i '{video_name}' -c copy '{pre_trick_raw}'"
        os.system(cmd_pre)
        if not os.path.exists(pre_trick_raw):
            print(f"[WARNING] Pre-trick video not created, skipping it")
            videos_to_process = output_videos
        else:
            videos_to_process = [pre_trick_raw] + output_videos

        normalized = []
        for i, vid in enumerate(videos_to_process):
            if not os.path.exists(vid):
                print(f"[WARNING] Skipping missing video: {vid}")
                continue
            out = f"normalized_{i}_video_{group_id}_{video_types[0]}.mp4"
            normalize_input(vid, out)
            normalized.append(out)

        final_inputs = []
        for x in normalized:
            if os.path.exists(x):
                audio_version = ensure_audio(x)
                if os.path.exists(audio_version):
                    final_inputs.append(audio_version)
                else:
                    print(f"[WARNING] Audio version not created for {x}")
            else:
                print(f"[WARNING] Normalized video not found: {x}")

        list_file = "concat_list.txt"
        with open(list_file, "w") as f:
            for v in final_inputs:
                if os.path.exists(v):
                    f.write(f"file '{os.path.abspath(v)}'\n")
                else:
                    print(f"[WARNING] Skipping non-existent file in concat: {v}")

        final_output = f"final_output_group_{group_id}_{video_type}.mp4"
        cmd_concat = (
            f"ffmpeg -y -f concat -safe 0 -i {list_file} "
            f"-c:v copy -c:a copy "
            f"'{final_output}'"
        )
        os.system(cmd_concat)

        print(f"[INFO] Final video saved as: {final_output}")
