import cv2
import numpy as np

def trick_2_static_part1(frame, selected):
    boxes_sel = [b for (_, b, _) in selected]
    scores_sel = [s for (s, _, _) in selected]
    labels_sel = [lab for (_, _, lab) in selected]
    pos_wand = None
    pos_ball = None
    pos_zombie = None
    can_bbox = None
    wand_bbox = None
    ball_bbox = None
    zombie_bbox = None
    pos_can = None
    dist_wand_ball = None
    dist_wand_zombie = None
    dist_wand_can = None
    zombie_bbox = None
    for box, score, label in zip(boxes_sel, scores_sel, labels_sel):
        x1, y1, x2, y2 = map(int, box)
        # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label_str = str(label)
        label_text = f"{label_str} {score:.2f}"
        # cv2.putText(frame, label_text,
        #             (x1, max(0, y1 - 10)),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36, 255, 12), 2)
        if label_str == "a magic wand":
            pos_wand = ((x1 + x2) // 2, y2)
            wand_bbox = (x1, y1, x2, y2)
        elif label_str == "a blue ball":
            pos_ball = ((x1 + x2) // 2, (y1 + y2) // 2)
            ball_bbox = (x1, y1, x2, y2)
        elif label_str == "a small figurine":
            pos_zombie = ((x1 + x2) // 2, y1)
            zombie_bbox = (x1, y1, x2, y2)
        elif "soda" in label_str.lower() or "can" in label_str.lower():
            pos_can = ((x1 + x2) // 2, y1)
            can_bbox = (x1, y1, x2, y2)
 
    if pos_wand is not None:
        if pos_ball is not None:
            dist_wand_ball = np.linalg.norm(np.array(pos_wand) - np.array(pos_ball))
        if pos_zombie is not None:
            dist_wand_zombie = np.linalg.norm(np.array(pos_wand) - np.array(pos_zombie))
        if pos_can is not None:
            dist_wand_can = np.linalg.norm(np.array(pos_wand) - np.array(pos_can))
  
    return {
            "wand" : {"pos" : pos_wand, "bbox" : None}, 
            "ball" : {"pos" : pos_ball, "bbox" : ball_bbox, "dist" : dist_wand_ball}, 
            "figurine" : {"pos" : pos_zombie, "bbox" : zombie_bbox, "dist" : dist_wand_zombie}, 
            "can" : {"pos" : pos_can, "bbox" : can_bbox, "dist" : dist_wand_can},
            }

def trick_2_static_part2(frame, frame_num, first_frame, data, data_video):
    fps = 30
    current_time = frame_num / fps
    if data["wand"]["pos"] is not None:
        print("i enter here")
        if data["ball"]["pos"] is not None:
            if(data["ball"]["dist"] < 80 and current_time - data_video["last_detection_ball"] > 1):
                data_video["state_ball"] += 1
                print("Ball detected : ", data_video["state_ball"])
                data_video["last_detection_ball"] = current_time
                print("last detection ball time updated to", current_time)
        if data["figurine"]["pos"] is not None:
            if(data["figurine"]["dist"] < 80 and current_time - data_video["last_detection_zombie"] > 1 and data_video["state_can"] >= 1):
                data_video["state_zombie"] += 1
                print("Zombie detected : ", data_video["state_zombie"])
                data_video["last_detection_zombie"] = current_time
                print("last detection zombie time updated to", current_time)
        if data["can"]["pos"] is not None:
            if(data["can"]["dist"] < 150 and current_time - data_video["last_detection_can"] > 1 and data_video["state_ball"] >= 2):
                data_video["state_can"] += 1
                print("Can detected : ", data_video["state_can"])
                data_video["last_detection_can"] = current_time
                print("last detection can time updated to", current_time)
    
    print(f"data WITHIN : {data}")
    print(f"data_video WITHIN : {data_video}")

    if data_video["state_ball"] == 1:
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
    elif data_video["state_ball"] == 2:
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
        except Exception:
            return frame

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
                return frame
            frame[y1_src:y2_src, x1_src:x2_src] = first_frame[y1_src:y2_src, x1_src:x2_src]
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
                return frame
 
            result_cropped = result[0:paste_h, 0:paste_w]
            frame[y_dest:y_dest + paste_h, x_dest:x_dest + paste_w] = first_frame[y_dest:y_dest + paste_h, x_dest:x_dest + paste_w]
            frame[y_dest:y_dest + paste_h, x_dest:x_dest + paste_w] = result_cropped

    if  data_video["state_can"] == 1 and data["can"]["bbox"] is not None:
        x1, y1, x2, y2 = data["can"]["bbox"]
        x_c, y_c = x1, y1
        w_c = x2 - x1
        h_c = y2 - y1
        can_roi = frame[y_c:y_c+h_c, x_c:x_c+w_c]
        can_hsv = cv2.cvtColor(can_roi, cv2.COLOR_BGR2HSV)
        bg_var_at_can = first_frame[y_c:y_c+h_c, x_c:x_c+w_c]
        dist = cv2.absdiff(can_roi, bg_var_at_can)
        bigger_mask = cv2.resize(cv2.cvtColor(dist, cv2.COLOR_BGR2GRAY), (w_c*3, h_c*3))
        bigger_can_roi = cv2.resize(can_roi, (w_c*3, h_c*3))
        region = frame[y_c - h_c:y_c + 2*h_c, x_c - w_c:x_c + 2*w_c]
        mask_indices = bigger_mask < 10
        if region.shape == bigger_can_roi.shape:
            bigger_can_roi[mask_indices] = region[mask_indices]
        frame[y_c - h_c:y_c + 2*h_c, x_c - w_c:x_c + 2*w_c] = bigger_can_roi

    return frame