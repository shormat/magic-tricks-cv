import json

VIDEO_KEYS = {"Static": "video_fixed", "Dynamic": "video_dynamic"}


class Annotation:
    """Reads one group's annotation file.

    Expected layout::

        {
          "video_fixed":   {"video_name": "...", "illusions": [{"name": "...", "start_time": "HH:MM:SS"}, ...]},
          "video_dynamic": {"video_name": "...", "illusions": [...]}
        }

    Trick ids are 1-based, in the order of the "illusions" list.
    """

    def __init__(self, filename, video_type="Static", prefix_video_name=False):
        if video_type not in VIDEO_KEYS:
            raise ValueError(f"video_type must be one of {list(VIDEO_KEYS)}, got {video_type!r}")

        with open(filename, "r", encoding="utf-8") as f:
            self.annotations = json.load(f)[VIDEO_KEYS[video_type]]

        self.prefix_video_name = prefix_video_name

    def get_video_name(self):
        name = self.annotations["video_name"]
        return f"video_{name}" if self.prefix_video_name else name

    @property
    def trick_count(self):
        return len(self.annotations["illusions"])

    @staticmethod
    def time_to_seconds(time_str):
        h, m, s = map(int, time_str.split(":"))
        return h * 3600 + m * 60 + s

    def get_start_time(self, trick_id):
        return self.time_to_seconds(self._illusion(trick_id)["start_time"])

    def get_trick_name(self, trick_id):
        return self._illusion(trick_id)["name"]

    def _illusion(self, trick_id):
        if not 1 <= trick_id <= self.trick_count:
            raise IndexError(f"trick_id {trick_id} out of range 1..{self.trick_count}")
        return self.annotations["illusions"][trick_id - 1]
