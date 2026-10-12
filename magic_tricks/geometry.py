"""Bounding-box, contour and ArUco helpers shared by the tricks.

Boxes are (x1, y1, x2, y2) in pixels.
"""

import cv2
import numpy as np

# ArUco markers (DICT_4X4_50) stuck on the four corners of the table.
ROI_IDS = (0, 1, 2, 3)


def to_int_box(box):
    x1, y1, x2, y2 = map(int, box)
    return x1, y1, x2, y2


def box_area(box):
    x1, y1, x2, y2 = box
    return (x2 - x1) * (y2 - y1)


def box_center(box):
    x1, y1, x2, y2 = box
    return (x1 + x2) // 2, (y1 + y2) // 2


def clip_box(box, frame_shape):
    """Clamp a box to the frame; None if nothing is left."""
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = to_int_box(box)
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def pad_box(box, pad):
    x1, y1, x2, y2 = box
    return x1 - pad, y1 - pad, x2 + pad, y2 + pad


def boxes_intersect(b1, b2, touching=False):
    """Whether two boxes overlap. With touching=True, boxes sharing an edge count too."""
    x1, y1, x2, y2 = to_int_box(b1)
    u1, v1, u2, v2 = to_int_box(b2)
    if touching:
        return x1 <= u2 and u1 <= x2 and y1 <= v2 and v1 <= y2
    return x1 < u2 and u1 < x2 and y1 < v2 and v1 < y2


def suppress_overlaps(items, touching=False):
    """Greedy suppression over (score, box, ...) items sorted by decreasing score:
    an item is dropped if it overlaps one already kept."""
    kept = []
    for item in items:
        if not any(boxes_intersect(item[1], other[1], touching) for other in kept):
            kept.append(item)
    return kept


def point_in_box(point, box):
    x, y = point
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def contour_on_box(contour, box, min_overlap=0.5):
    """A contour is on the box if its centroid is inside it, or if more than
    `min_overlap` of its bounding rectangle lies inside it."""
    box = to_int_box(box)
    m = cv2.moments(contour)
    if m["m00"] != 0 and point_in_box((int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])), box):
        return True

    x1, y1, x2, y2 = box
    cx, cy, cw, ch = cv2.boundingRect(contour)
    inter_w = min(x2, cx + cw) - max(x1, cx)
    inter_h = min(y2, cy + ch) - max(y1, cy)
    if inter_w <= 0 or inter_h <= 0 or cw * ch == 0:
        return False
    return inter_w * inter_h / (cw * ch) > min_overlap


def distance(p1, p2):
    return float(np.linalg.norm(np.subtract(p1, p2, dtype=np.float64)))


def project_point(point, homography):
    return cv2.perspectiveTransform(np.array([[point]], dtype=np.float32), homography)[0][0]


def clamp_point_to_polygon(point, polygon):
    """Return the point itself if inside the polygon, otherwise the closest point on its edges."""
    if polygon is None or len(polygon) < 3:
        return point

    p = np.array(point, dtype=np.float32)
    poly = np.array(polygon, dtype=np.float32)
    if cv2.pointPolygonTest(poly, (float(p[0]), float(p[1])), False) >= 0:
        return int(p[0]), int(p[1])

    best, best_dist = p, float("inf")
    for a, b in zip(poly, np.roll(poly, -1, axis=0)):
        edge = b - a
        edge_len_sq = float(np.dot(edge, edge))
        t = 0.0 if edge_len_sq == 0 else max(0.0, min(1.0, float(np.dot(p - a, edge)) / edge_len_sq))
        candidate = a + t * edge
        d = float(np.linalg.norm(p - candidate))
        if d < best_dist:
            best, best_dist = candidate, d
    return int(best[0]), int(best[1])


def table_polygon(corners, ids):
    """Polygon joining the centres of the visible table markers, ordered by angle."""
    if ids is None:
        return None
    ids = list(ids)
    centers = [corners[ids.index(mid)].reshape(4, 2).mean(axis=0) for mid in ROI_IDS if mid in ids]
    if not centers:
        return None
    centers = np.array(centers, dtype=np.float32)
    centroid = centers.mean(axis=0)
    angles = np.arctan2(centers[:, 1] - centroid[1], centers[:, 0] - centroid[0])
    return centers[np.argsort(angles)].astype(np.int32)


class MarkerTracker:
    """Detects the table markers and maps the reference (first) frame onto the current view."""

    def __init__(self, reference_frame):
        self._dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self._params = cv2.aruco.DetectorParameters()
        self.ref_corners, self.ref_ids = self.detect(reference_frame)
        self.last_roi = None

    def detect(self, frame):
        # A fresh detector per call keeps this safe to use from several worker threads.
        detector = cv2.aruco.ArucoDetector(self._dictionary, self._params)
        corners, ids, _ = detector.detectMarkers(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        return corners, (ids.flatten() if ids is not None else None)

    def update_roi(self, corners, ids):
        """Table polygon in the current frame, falling back to the last one seen."""
        roi = table_polygon(corners, ids)
        if roi is not None:
            self.last_roi = roi
        return self.last_roi

    def homography(self, corners, ids):
        """Homography reference frame -> current frame from the shared table markers, or None."""
        if self.ref_ids is None or ids is None:
            return None
        ref_ids, cur_ids = list(self.ref_ids), list(ids)
        src, dst = [], []
        for mid in ROI_IDS:
            if mid in ref_ids and mid in cur_ids:
                src.extend(self.ref_corners[ref_ids.index(mid)].reshape(4, 2))
                dst.extend(corners[cur_ids.index(mid)].reshape(4, 2))
        if len(src) < 4:
            return None
        return cv2.findHomography(np.float32(src), np.float32(dst))[0]

    def warped_background(self, background, corners, ids):
        """(background seen from the current viewpoint, homography). Unwarped if no homography."""
        h_matrix = self.homography(corners, ids)
        if h_matrix is None:
            return background, None
        h, w = background.shape[:2]
        return cv2.warpPerspective(background, h_matrix, (w, h)), h_matrix
