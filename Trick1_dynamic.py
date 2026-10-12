import cv2
import numpy as np

ROI_IDS = [0, 1, 2, 3]

# If two boxes intersect, keep only the one with the highest score.
def intersects(b1, b2):
    x1, y1, x2, y2 = map(int, b1)
    x1p, y1p, x2p, y2p = map(int, b2)
    return (x1 < x2p) and (x2 > x1p) and (y1 < y2p) and (y2 > y1p)
def contour_inside_box(contour, box):
    """Vérifie si le centre du contour est dans la bounding box"""
    x1, y1, x2, y2 = map(int, box)
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return False
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return x1 <= cx <= x2 and y1 <= cy <= y2

def contour_overlaps_box(contour, box, threshold=0.3):
    """Vérifie si un contour chevauche une bounding box"""
    x1, y1, x2, y2 = map(int, box)
    cx, cy, cw, ch = cv2.boundingRect(contour)
    
    inter_x1 = max(x1, cx)
    inter_y1 = max(y1, cy)
    inter_x2 = min(x2, cx + cw)
    inter_y2 = min(y2, cy + ch)
    
    if inter_x1 >= inter_x2 or inter_y1 >= inter_y2:
        return False
    
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    contour_area = cw * ch
    
    return inter_area / contour_area > threshold if contour_area > 0 else False

def box_center_inside_box(inner_box, outer_box):
    """Vérifie si le centre d'une box est dans une autre box"""
    x1, y1, x2, y2 = map(int, inner_box)
    ox1, oy1, ox2, oy2 = map(int, outer_box)
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    return ox1 <= cx <= ox2 and oy1 <= cy <= oy2

def boxes_intersect(box1, box2):
    """Vérifie si deux boxes se chevauchent"""
    x1, y1, x2, y2 = map(int, box1)
    x1b, y1b, x2b, y2b = map(int, box2)
    return not (x2 < x1b or x2b < x1 or y2 < y1b or y2b < y1)

def compute_iou(box1, box2):
    """Calcule l'IoU entre deux boxes"""
    x1, y1, x2, y2 = map(int, box1)
    x1b, y1b, x2b, y2b = map(int, box2)
    
    inter_x1 = max(x1, x1b)
    inter_y1 = max(y1, y1b)
    inter_x2 = min(x2, x2b)
    inter_y2 = min(y2, y2b)
    
    if inter_x1 >= inter_x2 or inter_y1 >= inter_y2:
        return 0.0
    
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2b - x1b) * (y2b - y1b)
    union_area = area1 + area2 - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0

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

def trick_1_dynamic_part1(frame, table_bbox, all_detections, data_video):
    all_objects = []
    box_threshold = 0.4
    width = frame.shape[1]
    height = frame.shape[0]
    can_found = ball_found = figurine_found = False
    for box, score, label in all_detections:
        if score < box_threshold:
            continue

        x1, y1, x2, y2 = map(int, box)
        label_str = str(label).lower()

        bbox_area = (x2 - x1) * (y2 - y1)
        frame_area = width * height
        bbox_ratio = bbox_area / frame_area

        category = None
        if "ball" in label_str:
            if bbox_ratio > 0.03:
                continue
            category = "ball"
            ball_found = True
        elif "soda" in label_str or "can" in label_str:
            if bbox_ratio > 0.05:
                continue
            category = "can"
            can_found = True
        elif "figurine" in label_str:
            if bbox_ratio > 0.05:
                continue
            category = "figurine"
            figurine_found = True
        else:
            continue
        if table_bbox is not None:
            if not box_center_inside_box((x1, y1, x2, y2), table_bbox):
                continue 

        all_objects.append((score, (x1, y1, x2, y2), category, label_str))

    all_objects.sort(key=lambda x: x[0], reverse=True)
    filtered_objects = []
    for score, bbox, category, label_str in all_objects:
        dominated = False
        for other_score, other_bbox, other_category, other_label in filtered_objects:
            if boxes_intersect(bbox, other_bbox):
                dominated = True
                break
        if not dominated:
            filtered_objects.append((score, bbox, category, label_str))

    best_per_category = {}
    for score, bbox, category, label_str in filtered_objects:
        if category not in best_per_category or score > best_per_category[category][0]:
            best_per_category[category] = (score, bbox, label_str)
    can_found = "can" in best_per_category
    ball_found = "ball" in best_per_category
    figurine_found = "figurine" in best_per_category

    if table_bbox is not None:
        x1, y1, x2, y2 = table_bbox
        # cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        # cv2.putText(frame, f"TABLE",
        #             (x1, max(0, y1 - 10)),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    for category, (score, bbox, label_str) in best_per_category.items():
        x1, y1, x2, y2 = bbox
        # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # cv2.putText(frame, f"{label_str} {score:.2f}",
        #             (x1, max(0, y1 - 10)),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36, 255, 12), 2)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detector = cv2.aruco.ArucoDetector(data_video["aruco_dict"], data_video["aruco_params"])
    corners, ids, rejected = detector.detectMarkers(gray)
    ids = ids.flatten() if ids is not None else None
    return {
        "ids" : ids,
        "corners" : corners,
        "table_bbox" : table_bbox,
        "can_found" : can_found,
        "ball_found" : ball_found,
        "figurine_found" : figurine_found
    }

def trick_1_dynamic_part2(frame, first_frame, data, data_video):
    original_frame = frame.copy()
    hsv = cv2.cvtColor(original_frame, cv2.COLOR_BGR2HSV)
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
    lower_green = np.array([35, 20, 20])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    bg_hsv = cv2.cvtColor(warped_bg, cv2.COLOR_BGR2HSV)
    bg_green_mask = cv2.inRange(bg_hsv, lower_green, upper_green)
    kernel = np.ones((5, 5), np.uint8)
    bg_green_mask = cv2.morphologyEx(bg_green_mask, cv2.MORPH_OPEN, kernel)
    bg_green_mask = cv2.morphologyEx(bg_green_mask, cv2.MORPH_CLOSE, kernel)
    inv_bg = cv2.bitwise_not(bg_green_mask)
    dynamic_mask = cv2.bitwise_and(mask, inv_bg)
    kernel = np.ones((5, 5), np.uint8)
    dynamic_mask = cv2.morphologyEx(dynamic_mask, cv2.MORPH_OPEN, kernel)
    dynamic_mask = cv2.morphologyEx(dynamic_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(dynamic_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_contour_on_table = None
    largest_area = 0
    for contour in contours:
        # cv2.drawContours(frame, [contour], -1, (0, 255, 255), 2)
        area = cv2.contourArea(contour)
        towel_on_table = False
        if data["table_bbox"] is not None:
            towel_on_table = contour_inside_box(contour, data["table_bbox"])
            if not towel_on_table:
                towel_on_table = contour_overlaps_box(contour, data["table_bbox"], threshold=0.5)
        if area > largest_area and area > 1000 and towel_on_table:
            largest_area = area
            largest_contour_on_table = contour
    largest_contour = largest_contour_on_table
    if largest_contour is not None:
        x, y, w, h = cv2.boundingRect(largest_contour)
        towel_on_table = False
        if data["table_bbox"] is not None:
            towel_on_table = contour_inside_box(largest_contour, data["table_bbox"])
            if not towel_on_table:
                towel_on_table = contour_overlaps_box(largest_contour, data["table_bbox"], threshold=0.5)
        
        if towel_on_table and not (data.get("can_found", False) and data.get("ball_found", False) and data.get("figurine_found", False)):
            towel_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.fillPoly(towel_mask, [largest_contour], 255)
            towel_mask = cv2.dilate(towel_mask, np.ones((15, 15), np.uint8), iterations=1)
            frame[towel_mask == 255] = warped_bg[towel_mask == 255]
            
            # cv2.putText(frame, "NAPKIN HIDDEN", (10, 30),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            # Dessiner la serviette (visible)
            # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            # cv2.putText(frame, "Napkin (not on table)", (x, y - 10),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            pass

    return frame