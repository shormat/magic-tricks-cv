import cv2
import numpy as np
ROI_IDS = [0, 1, 2, 3]
def get_roi(corners, ids):
    if ids is None: 
        return None

    centers = []
    for mid in ROI_IDS:
        if mid not in ids: 
            continue
        idx = list(ids).index(mid)
        centers.append(corners[idx].reshape(4, 2).mean(axis=0))

    centers = np.array(centers, dtype=np.float32)

    centroid = centers.mean(axis=0)
    angles = np.arctan2(centers[:, 1] - centroid[1], centers[:, 0] - centroid[0])
    return centers[np.argsort(angles)].astype(np.int32)
def intersects(b1, b2):
    x1, y1, x2, y2 = map(int, b1)
    x1p, y1p, x2p, y2p = map(int, b2)
    return (x1 < x2p) and (x2 > x1p) and (y1 < y2p) and (y2 > y1p)
def trick_3_static_part1(frame, selected, data_video):
    filtered = []
    for score, box, label in selected:
        skip = False
        for _, chosen_box, _ in filtered:
            if intersects(box, chosen_box):
                skip = True
                break
        if not skip:
            filtered.append((score, box, label))
    boxes_sel = [b for (_, b, _) in selected]
    scores_sel = [s for (s, _, _) in selected]
    labels_sel = [lab for (_, _, lab) in selected]
    pos_wand = None
    pos_ball = None
    ball_bbox = None
    dist_wand_ball = None
    for box, score, label in zip(boxes_sel, scores_sel, labels_sel):
        x1, y1, x2, y2 = map(int, box)
        # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label_str = str(label)
        label_text = f"{label_str} {score:.2f}"

        if label_str == "a magic wand":
            area = (box[2] - box[0]) * (box[3] - box[1])
            pos_wand = ((x1 + x2) // 2, y2)
            wand_bbox = (x1, y1, x2, y2)
        elif label_str == "a blue ball":
            pos_ball = ((x1 + x2) // 2, (y1 + y2) // 2)
            ball_bbox = (x1, y1, x2, y2)
        # cv2.putText(frame, label_text,
        # (x1, max(0, y1 - 10)),
        # cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36, 255, 12), 2)
 
    if pos_wand is not None:
        if pos_ball is not None:
            dist_wand_ball = np.linalg.norm(np.array(pos_wand) - np.array(pos_ball))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detector = cv2.aruco.ArucoDetector(data_video["aruco_dict"], data_video["aruco_params"])
    corners, ids, rejected = detector.detectMarkers(gray)
    ids = ids.flatten() if ids is not None else None 
    if ball_bbox is None:
        ball_bbox = data_video.get("last_ball_bbox", None) 
    return {
            "ids" : ids,
            "corners" : corners,
            "wand" : {"pos" : pos_wand, "bbox" : None}, 
            "ball" : {"pos" : pos_ball, "bbox" : ball_bbox, "dist" : dist_wand_ball}, 
            }

def trick_3_static_part2(frame,frame_num, first_frame, data, data_video):
    fps = 30
    try:
        roi = get_roi(data["corners"], data["ids"])
    except Exception as e:
        print("Error in get_roi:", e)
        roi = None
    if roi is not None:
        data_video["last_roi"] = roi

    
    bg = first_frame.copy()
    copy_frame = frame.copy()
    warped_bg = None
    H = None
    roi_use = roi if roi is not None else data_video["last_roi"] if "last_roi" in data_video else None
    if roi_use is not None:
        if data_video["first_frame_corners"] is not None and len(data_video["first_frame_corners"]) > 0 and len(data["corners"]) > 0:
            src_pts = []
            dst_pts = []
            for mid in ROI_IDS:
                if mid in data_video["first_frame_ids"] and mid in data["ids"]:
                    src_idx = list(data_video["first_frame_ids"]).index(mid)
                    dst_idx = list(data["ids"]).index(mid)
                    src_pts.extend(data_video["first_frame_corners"][src_idx].reshape(4, 2))
                    dst_pts.extend(data["corners"][dst_idx].reshape(4, 2))
            # print("Points de correspondance pour l'homographie:", len(src_pts))
            if len(src_pts) >= 4:
                src_pts = np.float32(src_pts)
                dst_pts = np.float32(dst_pts)
                H = cv2.findHomography(src_pts, dst_pts)[0]
                warped_bg = cv2.warpPerspective(bg, H, (frame.shape[1], frame.shape[0]))
            else:
                warped_bg = bg
        bg_var = warped_bg
        height_roi = roi_use[:, 1].max() - roi_use[:, 1].min()
        height_frame = frame.shape[0] - roi_use[:, 1].min() + 100
        width_roi = roi_use[:, 0].max() - roi_use[:, 0].min()
    current_time = frame_num / fps
    prev_wand_pos = data_video.get("last_wand_pos", None)
    if data["wand"]["pos"] is not None:
        data_video["last_wand_pos"] = data["wand"]["pos"]
        if data["ball"]["pos"] is not None:
            if(data["ball"]["dist"] < 250 and data_video["state_ball"] == 0):
                data_video["state_ball"] += 1
                last_detection_ball = current_time
                data_video["wand_pos_dct"] = data["wand"]["pos"]
                data_video["ball_pos_dct"] = data["ball"]["pos"]
                data_video["new_pos_ball"] = data["ball"]["pos"]
                data_video["vector_wand_ball"] = (np.array(data_video["ball_pos_dct"]) - np.array(data_video["wand_pos_dct"]))
                print(f"   >> Balle touchée! Total: {data_video['state_ball']}")

    if "state_ball" not in data_video:
        data_video["state_ball"] = 0
    if data["wand"]["pos"] is None:
        data["wand"]["pos"] = data_video.get("last_wand_pos", None)
    if data["ball"]["bbox"] is not None:
        data_video["last_ball_bbox"] = data["ball"]["bbox"]
    if data_video["state_ball"] >= 1:
        dx = data["wand"]["pos"][0] - prev_wand_pos[0] if prev_wand_pos is not None and data["wand"]["pos"] is not None else 0
        dy = data["wand"]["pos"][1] - prev_wand_pos[1] if prev_wand_pos is not None and data["wand"]["pos"] is not None else 0
        adjusted_displacement = dy * (height_roi / (height_frame)) if height_roi is not None else dy
        data_video["new_pos_ball"] = (data_video["new_pos_ball"][0] + dx, data_video["new_pos_ball"][1] + adjusted_displacement)
    if data_video["state_ball"] >= 1 and data["wand"]["pos"] is not None:
        new_pos_ball = data_video["new_pos_ball"]
        ball = frame[data["ball"]["bbox"][1]:data["ball"]["bbox"][3], data["ball"]["bbox"][0]:data["ball"]["bbox"][2]].copy()
        hsv_ball = cv2.cvtColor(ball, cv2.COLOR_BGR2HSV)
        blue_mask = cv2.inRange(hsv_ball, np.array([90, 30, 30]), np.array([150, 255, 255]))
        frame[data["ball"]["bbox"][1]:data["ball"]["bbox"][3], data["ball"]["bbox"][0]:data["ball"]["bbox"][2]] = first_frame[data["ball"]["bbox"][1]:data["ball"]["bbox"][3], data["ball"]["bbox"][0]:data["ball"]["bbox"][2]]
        for i in range(blue_mask.shape[0]):
            for j in range(blue_mask.shape[1]):
                if blue_mask[i, j] > 0:
                    y = int(new_pos_ball[1]) - (ball.shape[0] // 2) + i
                    x = int(new_pos_ball[0]) - (ball.shape[1] // 2) + j
                    if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
                        frame[y, x] = ball[i, j]
        
    return frame