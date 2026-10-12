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

def trick_2_dynamic_part1(frame, selected, data_video):
    filtered = []
    for score, box, label in selected:
        skip = False
        for _, chosen_box, _ in filtered:
            if intersects(box, chosen_box):
                skip = True
                break
        if not skip:
            filtered.append((score, box, label))

    boxes_sel = [b for (_, b, _) in filtered]
    scores_sel = [s for (s, _, _) in filtered]
    labels_sel = [lab for (_, _, lab) in filtered]

    pos_wand = None
    pos_wand_2 = None
    pos_ball = None
    pos_ball_2 = None
    pos_zombie = None
    pos_zombie_2 = None
    pos_zombie_3 = None
    pos_can = None
    pos_can_2 = None

    dist_wand_ball = None
    dist_wand_ball_2 = None
    dist_wand_zombie = None
    dist_wand_zombie_2 = None
    dist_wand_zombie_3 = None
    dist_wand_can = None

    wand_bbox = None
    ball_bbox = None
    zombie_bbox = None
    can_bbox = None

    for box, score, label in zip(boxes_sel, scores_sel, labels_sel):
        x1, y1, x2, y2 = map(int, box)
        # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label_str = str(label)
        label_text = f"{label_str} {score:.2f}"
        # cv2.putText(frame, label_text,
        #             (x1, max(0, y1 - 10)),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36, 255, 12), 2)
        if label_str == "a magic wand":
            pos_wand = (x2, y2)
            pos_wand_2 = (x1, y2)
            wand_bbox = (x1, y1, x2, y2)
        elif label_str == "a blue ball":
            pos_ball = (x1, y1)
            pos_ball_2 = (x2, y1)
            ball_bbox = (x1, y1, x2, y2)
        elif label_str == "a small figurine":
            pos_zombie = (x1, y1)
            pos_zombie_2 = (x2, y1)
            pos_zombie_3 = ((x1 + x2) // 2, y1)
            zombie_bbox = (x1, y1, x2, y2)
        elif "soda" in label_str.lower() or "can" in label_str.lower():
            pos_can = (x1, y1)
            pos_can_2 = (x2, y1)
            can_bbox = (x1, y1, x2, y2)
    if pos_wand is not None:
        if pos_ball is not None:
            dist_wand_ball = min(np.linalg.norm(np.array(pos_wand) - np.array(pos_ball)), np.linalg.norm(np.array(pos_wand_2) - np.array(pos_ball)))
            dist_wand_ball_2 = min(np.linalg.norm(np.array(pos_wand) - np.array(pos_ball_2)), np.linalg.norm(np.array(pos_wand_2) - np.array(pos_ball_2)))
            dist_wand_ball = min(dist_wand_ball, dist_wand_ball_2)
        if pos_zombie is not None:
            dist_wand_zombie = min(np.linalg.norm(np.array(pos_wand) - np.array(pos_zombie)), np.linalg.norm(np.array(pos_wand_2) - np.array(pos_zombie)))
            dist_wand_zombie_2 = min(np.linalg.norm(np.array(pos_wand) - np.array(pos_zombie_2)), np.linalg.norm(np.array(pos_wand_2) - np.array(pos_zombie_2)))
            dist_wand_zombie = min(dist_wand_zombie, dist_wand_zombie_2)
            dist_wand_zombie_3 = min(np.linalg.norm(np.array(pos_wand) - np.array(pos_zombie_3)), np.linalg.norm(np.array(pos_wand_2) - np.array(pos_zombie_3)))
            dist_wand_zombie = min(dist_wand_zombie, dist_wand_zombie_3)
        if pos_can is not None:
            dist_wand_can = min(np.linalg.norm(np.array(pos_wand) - np.array(pos_can)), np.linalg.norm(np.array(pos_wand_2) - np.array(pos_can)))
            dist_wand_can_2 = min(np.linalg.norm(np.array(pos_wand) - np.array(pos_can_2)), np.linalg.norm(np.array(pos_wand_2) - np.array(pos_can_2)))
            dist_wand_can = min(dist_wand_can, dist_wand_can_2)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detector = cv2.aruco.ArucoDetector(data_video["aruco_dict"], data_video["aruco_params"])
    corners, ids, rejected = detector.detectMarkers(gray)
    ids = ids.flatten() if ids is not None else None
    return {
        "ids" : ids,
        "corners" : corners,
        "wand" : {"pos" : pos_wand, "bbox": None},
        "ball" : {"pos" : pos_ball, "bbox": ball_bbox, "dist" : dist_wand_ball},
        "figurine" : {"pos" : pos_zombie, "bbox": zombie_bbox, "dist" : dist_wand_zombie},
        "can" : {"pos" : pos_can, "bbox": can_bbox, "dist" : dist_wand_can},
    }

def trick_2_dynamic_part2(frame, frame_num, first_frame, data, data_video):
    fps = 30
    current_time = frame_num / fps
    bg_var = None
    if data["wand"]["pos"] is not None:
        
        if data["ball"]["pos"] is not None:
            if(data["ball"]["dist"] < 80 and current_time - data_video["last_detection_ball"] > 2):
                data_video["state_ball"] += 1
                print("Ball detected : ", data_video["state_ball"])
                data_video["last_detection_ball"] = current_time
                print("last detection ball time updated to", current_time)
        if data["figurine"]["pos"] is not None:
            if(data["figurine"]["dist"] < 80 and current_time - data_video["last_detection_zombie"] > 2) and data_video["state_can"] > 1:
                data_video["state_zombie"] += 1
                print("Zombie detected : ", data_video["state_zombie"])
                data_video["last_detection_zombie"] = current_time
                print("last detection zombie time updated to", current_time)
        if data["can"]["pos"] is not None:
            if(data["can"]["dist"] < 80 and current_time - data_video["last_detection_can"] > 2) and data_video["state_ball"] >= 2:
                data_video["state_can"] += 1
                print("Can detected : ", data_video["state_can"])
                data_video["last_detection_can"] = current_time
                print("last detection can time updated to", current_time)

    print(f"DICT : {data_video}")
    
    try:
        roi = get_roi(data["corners"], data["ids"])
    except Exception as e:
        roi = None
    if roi is not None:
        data_video["last_roi"] = roi


    bg = first_frame.copy()
    copy_frame = frame.copy()
    warped_bg = None
    roi_use = roi if roi is not None else data_video["last_roi"]
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
    if warped_bg is None:
        warped_bg = bg
    if data_video["state_ball"] == 1 and data["ball"]["bbox"] is not None:
        try:
            x1, y1, x2, y2 = data["ball"]["bbox"]
        except NameError:
            pass
        else:
            x_b, y_b = x1, y1
            w_b, h_b = x2 - x1, y2 - y1
            copy = frame.copy()
            x0, x1_c = max(0, x_b), min(frame.shape[1], x_b + w_b)
            y0, y1_c = max(0, y_b), min(frame.shape[0], y_b + h_b)
            if x1_c > x0 and y1_c > y0:
                roi_ball = frame[y0:y1_c, x0:x1_c]
                hsv_ball = cv2.cvtColor(roi_ball, cv2.COLOR_BGR2HSV)
                blue_mask = cv2.inRange(hsv_ball, np.array([100, 50, 50]), np.array([140, 255, 255]))
                green_ball = roi_ball.copy()
                green_ball[blue_mask > 0, 0] = 0
                green_ball[blue_mask > 0, 2] = 0
                copy[y0:y1_c, x0:x1_c] = green_ball
                frame[y0:y1_c, x0:x1_c] = green_ball
    elif data_video["state_ball"] == 2 and data["ball"]["bbox"] is not None:
        try:
            x1, y1, x2, y2 = data["ball"]["bbox"]
        except NameError:
            pass
        else:
            x_b, y_b = x1, y1
            w_b, h_b = x2 - x1, y2 - y1
            copy = frame.copy()
            x0, x1_c = max(0, x_b), min(frame.shape[1], x_b + w_b)
            y0, y1_c = max(0, y_b), min(frame.shape[0], y_b + h_b)
            if x1_c > x0 and y1_c > y0:
                roi_ball = frame[y0:y1_c, x0:x1_c]
                hsv_ball = cv2.cvtColor(roi_ball, cv2.COLOR_BGR2HSV)
                blue_mask = cv2.inRange(hsv_ball, np.array([100, 50, 50]), np.array([140, 255, 255]))
                gray_ball = roi_ball.copy()
                gray_ball_region = cv2.cvtColor(roi_ball, cv2.COLOR_BGR2GRAY)
                gray_ball_region = cv2.cvtColor(gray_ball_region, cv2.COLOR_GRAY2BGR)
                gray_ball[blue_mask > 0] = gray_ball_region[blue_mask > 0]
                copy[y0:y1_c, x0:x1_c] = gray_ball   
                frame[y0:y1_c, x0:x1_c] = gray_ball 
    if data_video["state_zombie"] == 1 and data["figurine"]["bbox"] is not None:
        try:
            x0, y0, x2_v, y2_v = data["figurine"]["bbox"]
            if x2_v > x0 and y2_v > y0:
                x_f = int(x0)
                y_f = int(y0)
                w_f = int(x2_v - x0)
                h_f = int(y2_v - y0)
            else:
                x_f, y_f, w_f, h_f = map(int, data["figurine"]["bbox"])
            pad = 20
            x_f -= pad
            y_f -= pad
            w_f += pad * 2
            h_f += pad * 2

            fh, fw = frame.shape[:2]
            x_f = max(0, x_f)
            y_f = max(0, y_f)
            w_f = max(1, min(w_f, fw - x_f))
            h_f = max(1, min(h_f, fh - y_f))
            

            if data_video["amogus_img"] is None:
                tung_path = "video.mp4"
                data_video["amogus_cap2_reader"] = cv2.VideoCapture(tung_path)
                ok, amogus = data_video["amogus_cap2_reader"].read()
                if not ok:
                    print("ERROR: Cannot read Amogus video")
                    data_video["amogus_img"] = None
                else:
                    amogus = cv2.resize(amogus, (w_f, h_f))
                    data_video["amogus_img"] = amogus.copy()
                    data_video["amogus_bbox"] = (x_f, y_f, w_f, h_f)
                    data_video["amogus_cap2_reader"].set(cv2.CAP_PROP_POS_FRAMES, 1)
        except Exception:
            pass


        else:
            ok, amogus = data_video["amogus_cap2_reader"].read()
            if not ok:
                data_video["amogus_cap2_reader"].set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, amogus = data_video["amogus_cap2_reader"].read()
            if ok:
                amogus = cv2.resize(amogus, (data_video["amogus_bbox"][2], data_video["amogus_bbox"][3]))
    
        if ok:
            hsv_amogus = cv2.cvtColor(amogus, cv2.COLOR_BGR2HSV)
            lower_green = np.array([40, 40, 40])
            upper_green = np.array([80, 255, 255])
            green_mask = cv2.inRange(hsv_amogus, lower_green, upper_green)
            if green_mask.shape[:2] != amogus.shape[:2]:
                green_mask = cv2.resize(green_mask, (amogus.shape[1], amogus.shape[0]))
            green_mask = green_mask.astype(np.uint8)
            green_mask_inv = cv2.bitwise_not(green_mask)
            fh_cur, fw_cur = frame.shape[:2]
            x1_src = max(0, x_f)
            y1_src = max(0, y_f)
            x2_src = min(fw_cur, x_f + w_f)
            y2_src = min(fh_cur, y_f + h_f)
            if x2_src <= x1_src or y2_src <= y1_src:
                pass
            else:
                bg = bg if bg is not None else first_frame
                frame[y1_src:y2_src, x1_src:x2_src] = warped_bg[y1_src:y2_src, x1_src:x2_src]
                original_region = frame[y1_src:y2_src, x1_src:x2_src]
                
                if original_region.shape[:2] != amogus.shape[:2]:
                    original_region = cv2.resize(original_region, (amogus.shape[1], amogus.shape[0]))

                amogus_masked = cv2.bitwise_and(amogus, amogus, mask=green_mask_inv)
                original_masked = cv2.bitwise_and(original_region, original_region, mask=green_mask)
                result = cv2.add(amogus_masked, original_masked)
                x_dest = x1_src
                y_dest = y1_src
                res_h, res_w = result.shape[:2]
                max_w = fw_cur - x_dest
                max_h = fh_cur - y_dest
                paste_w = min(res_w, max_w)
                paste_h = min(res_h, max_h)
                if paste_w <= 0 or paste_h <= 0:
                    pass
                else:
                    result_cropped = result[0:paste_h, 0:paste_w]
                    bg = bg_var if bg_var is not None else bg if bg is not None else first_frame
                    frame[y_dest:y_dest + paste_h, x_dest:x_dest + paste_w] = warped_bg[y_dest:y_dest + paste_h, x_dest:x_dest + paste_w]
                    frame[y_dest:y_dest + paste_h, x_dest:x_dest + paste_w] = result_cropped   

    print("JARRIVE LA")
    if data_video["state_can"] == 1 and data["can"]["bbox"] is not None:
        x1, y1, x2, y2 = data["can"]["bbox"]
        x_c, y_c = x1, y1
        w_c = x2 - x1
        h_c = y2 - y1
        can_roi = frame[y_c:y_c+h_c, x_c:x_c+w_c]
        can_hsv = cv2.cvtColor(can_roi, cv2.COLOR_BGR2HSV)
        bg_var_at_can = bg_var[y_c:y_c+h_c, x_c:x_c+w_c] if bg_var is not None else bg[y_c:y_c+h_c, x_c:x_c+w_c]
        dist = cv2.absdiff(can_roi, bg_var_at_can)
        bigger_mask = cv2.resize(cv2.cvtColor(dist, cv2.COLOR_BGR2GRAY), (w_c*3, h_c*3))
        bigger_can_roi = cv2.resize(can_roi, (w_c*3, h_c*3))
        region = frame[y_c - h_c:y_c + 2*h_c, x_c - w_c:x_c + 2*w_c]
        mask_indices = bigger_mask < 15
        if region.shape == bigger_can_roi.shape:
            bigger_can_roi[mask_indices] = region[mask_indices]
        fh, fw = frame.shape[:2]
        y_start = max(0, y_c - h_c)
        y_end = min(fh, y_c + 2*h_c)
        x_start = max(0, x_c - w_c)
        x_end = min(fw, x_c + 2*w_c)
        region_h = y_end - y_start
        region_w = x_end - x_start
        if region_h > 0 and region_w > 0:
            bigger_can_roi_resized = cv2.resize(bigger_can_roi, (region_w, region_h))
            frame[y_start:y_end, x_start:x_end] = bigger_can_roi_resized


    return frame