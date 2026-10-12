import cv2
import numpy as np

ROI_IDS = [0, 1, 2, 3]
SMOOTHING_FACTOR = 0.4  # Lower = smoother but slower response, higher = more responsive

def clamp_point_to_polygon(point, polygon):
    if polygon is None or len(polygon) < 3:
        return point

    point = np.array(point, dtype=np.float32)
    polygon = np.array(polygon, dtype=np.float32)

    # Check if point is inside the polygon
    dist = cv2.pointPolygonTest(polygon, (float(point[0]), float(point[1])), True)

    if dist >= 0:  # Point is inside or on boundary
        return (int(point[0]), int(point[1]))

    # Not inside the polygon, find closest point on polygon edges
    min_dist = float('inf')
    closest_point = point

    n = len(polygon)
    for i in range(n):
        p1 = polygon[i]
        p2 = polygon[(i + 1) % n]

        # we do a projection of point onto the edge p1-p2
        edge = p2 - p1
        edge_len_sq = np.dot(edge, edge)

        if edge_len_sq == 0:  # its the same point
            candidate = p1
        else:
            t = max(0, min(1, np.dot(point - p1, edge) / edge_len_sq))
            candidate = p1 + t * edge

        dist = np.linalg.norm(point - candidate)
        if dist < min_dist:
            min_dist = dist
            closest_point = candidate

    return (int(closest_point[0]), int(closest_point[1]))

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

def trick_3_dynamic_part1(frame, selected, data_video):
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
            pos_wand = ((x1 + x2) // 2, (y1 + y2) // 2)  # center of bbox - more stable
            pos_wand_2 = (x1, y2)
            wand_bbox = (x1, y1, x2, y2)
        elif label_str == "a blue ball":
            pos_ball = ((x1 + x2) // 2, (y1 + y2) // 2)  # center of bbox - more stable
            pos_ball_2 = (x2, y1)
            ball_bbox = (x1, y1, x2, y2)
    if pos_wand is not None:
        if pos_ball is not None:
            dist_wand_ball = np.linalg.norm(np.array(pos_wand) - np.array(pos_ball))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detector = cv2.aruco.ArucoDetector(data_video["aruco_dict"], data_video["aruco_params"])
    corners, ids, rejected = detector.detectMarkers(gray)
    ids = ids.flatten() if ids is not None else None
    return {
        "ids" : ids,
        "corners" : corners,
        "wand" : {"pos" : pos_wand, "bbox": None},
        "ball" : {"pos" : pos_ball, "bbox": ball_bbox, "dist" : dist_wand_ball},
    }

def trick_3_dynamic_part2(frame,frame_num, first_frame, data, data_video):
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
    # Use current or last valid ROI
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
    if data["wand"]["pos"] is not None:
        data_video["last_wand_pos"] = data["wand"]["pos"]
        if data["ball"]["pos"] is not None:
            dist_wand_ball = np.linalg.norm(np.array(data["wand"]["pos"]) - np.array(data["ball"]["pos"]))
            if(dist_wand_ball < 350 and data_video.get("state_ball", 0) == 0):
                data_video["state_ball"] += 1
                data_video["place_detection_wand"] = data["wand"]["pos"]
                data_video["place_detection_ball"] = data["ball"]["pos"]
                data_video["vector_wand_ball"] = (np.array(data_video["place_detection_ball"]) - np.array(data_video["place_detection_wand"]))
                if H is not None and data_video.get("initial_homography", None) is None:
                    data_video["initial_homography"] = H
                    # we save original ball image and mask for good rendering
                    bbox = data["ball"]["bbox"]
                    original_ball = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]].copy()
                    hsv_ball = cv2.cvtColor(original_ball, cv2.COLOR_BGR2HSV)
                    original_mask = cv2.inRange(hsv_ball, np.array([90, 30, 30]), np.array([150, 255, 255]))
                    data_video["original_ball"] = original_ball
                    data_video["original_ball_mask"] = original_mask
                    data_video["original_ball_size"] = (original_ball.shape[1], original_ball.shape[0])
                else:
                    data_video["state_ball"] -= 1  # reset if no homography
                print(f"   >> Balle touchée! Total: {data_video['state_ball']}")
    if data["wand"]["pos"] is None:
        data["wand"]["pos"] = data_video.get("last_wand_pos", None)
    if "state_ball" not in data_video:
        data_video["state_ball"] = 0
    if "initial_homography" not in data_video:
        data_video["initial_homography"] = None
    if data_video["state_ball"] >= 1 and data_video["initial_homography"] is not None and data["wand"]["pos"] is not None and H is not None and data["ball"]["bbox"] is not None:
        # we compute inverse homography to project wand position onto table plane
        H_inv = np.linalg.inv(H)

        # Project current wand position to table coordinate system (first frame space)
        wand_screen = np.array([[data["wand"]["pos"]]], dtype=np.float32)
        wand_table = cv2.perspectiveTransform(wand_screen, H_inv)[0][0]

        # Keep the same distance from wand to ball as when first touched, but in table space
        if "ball_offset_table" not in data_video:
            # Store the initial distance in table coordinates
            initial_wand_table = cv2.perspectiveTransform(
                np.array([[data_video["place_detection_wand"]]], dtype=np.float32),
                np.linalg.inv(data_video["initial_homography"])
            )[0][0]
            initial_ball_table = cv2.perspectiveTransform(
                np.array([[data_video["place_detection_ball"]]], dtype=np.float32),
                np.linalg.inv(data_video["initial_homography"])
            )[0][0]
            data_video["ball_offset_table"] = initial_ball_table - initial_wand_table

        # Calculate ball position in table space (follows wand with fixed offset)
        ball_table = wand_table + data_video["ball_offset_table"]

        # Project ball position back to screen space using current homography
        ball_screen = cv2.perspectiveTransform(
            np.array([[ball_table]], dtype=np.float32), H
        )[0][0]

        # Apply smoothing to reduce jitter
        if "smoothed_ball_pos" not in data_video:
            data_video["smoothed_ball_pos"] = ball_screen
        else:
            data_video["smoothed_ball_pos"] = (
                SMOOTHING_FACTOR * ball_screen +
                (1 - SMOOTHING_FACTOR) * data_video["smoothed_ball_pos"]
            )

        new_pos_ball = (int(data_video["smoothed_ball_pos"][0]), int(data_video["smoothed_ball_pos"][1]))

        if roi_use is not None:
            new_pos_ball = clamp_point_to_polygon(new_pos_ball, roi_use)

        if "original_ball" in data_video:
            ball = data_video["original_ball"]
            blue_mask = data_video["original_ball_mask"]
        else:
            ball = frame[data["ball"]["bbox"][1]:data["ball"]["bbox"][3], data["ball"]["bbox"][0]:data["ball"]["bbox"][2]].copy()
            hsv_ball = cv2.cvtColor(ball, cv2.COLOR_BGR2HSV)
            blue_mask = cv2.inRange(hsv_ball, np.array([90, 30, 30]), np.array([150, 255, 255]))

        # remove original ball from frame using warped background
        frame[data["ball"]["bbox"][1]:data["ball"]["bbox"][3], data["ball"]["bbox"][0]:data["ball"]["bbox"][2]] = bg_var[data["ball"]["bbox"][1]:data["ball"]["bbox"][3], data["ball"]["bbox"][0]:data["ball"]["bbox"][2]]

        # Put the scaled ball at new_pos_ball
        for i in range(blue_mask.shape[0]):
            for j in range(blue_mask.shape[1]):
                if blue_mask[i, j] > 0:
                    y = new_pos_ball[1] - (ball.shape[0] // 2) + i
                    x = new_pos_ball[0] - (ball.shape[1] // 2) + j
                    if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
                        frame[y, x] = ball[i, j]
    return frame
        
