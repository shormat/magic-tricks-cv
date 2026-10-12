import json

class Annotation():
    def __init__(self, filename, video_type="Static", prefix_video_name=False):
        
        with open(filename, "r") as f:
            self.annotations = json.load(f)

        self.prefix_video_name = prefix_video_name

        if video_type == "Static":
            self.annotations = self.annotations["video_fixed"]
        else:
            self.annotations = self.annotations["video_dynamic"]

    def get_video_name(self):   
        name = self.annotations["video_name"]
        
        if self.prefix_video_name:
            name = f"video_{name}"

        return name

    def time_to_seconds(self, time_str):
        h, m, s = map(int, time_str.split(':'))
        return h * 3600 + m * 60 + s
    
    def get_start_time(self, trick_id):
        # trick_id starts from 1 !!!
        time = self.time_to_seconds(self.annotations["illusions"][trick_id-1]["start_time"])
        return time
    
    def get_trick_name(self, trick_id):
        # trick_id starts from 1 !!!
        name = self.annotations["illusions"][trick_id-1]["name"]
        return name