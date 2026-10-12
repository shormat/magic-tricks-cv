from MagicTricks import MagicTricks

if __name__ == "__main__":
    annotation_file = "annotations_group_0.json"
    nthreads = 4
    video_type = "Static"
    prefix_video_name = True
    trick_id = 1
    debug = True

    magic_tricks = MagicTricks(annotation_file, nthreads, video_type, prefix_video_name, trick_id, debug)
    magic_tricks.run()