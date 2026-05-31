"""
gap_measurement.py — Dynamic gap detection between parked motorcycles.
======================================================================
Algorithm overview
------------------
1. Extract the **bottom-center** of each motorcycle bounding box as an
   approximate ground contact point.
2. If BEV is enabled, transform these points into Bird's Eye View
   coordinates for more uniform distance measurement.
3. If ROI is enabled, discard points outside the ROI polygon.
4. Sort the remaining points along the **dominant axis** (the axis with
   the larger spread).
5. Compute the Euclidean distance between each pair of adjacent points.
6. If the distance is within ``[min_gap_pixels, max_gap_pixels]`` the
   space is marked as a candidate *available gap*.
7. **Temporal smoothing**: a gap must persist for at least
   ``min_gap_frames`` consecutive frames before it is displayed.  This
   prevents flickering when a person walks through momentarily.
"""

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from backend.core.bev import transform_points
from backend.core.utils import (
    COLOR_GAP_AVAILABLE,
    bbox_bottom_center,
    point_in_polygon,
)

def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """Calculate IoU between two boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    if inter_area == 0:
        return 0.0

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou



class GapAnalyzer:
    """
    Stateful gap analyser — call :meth:`update` once per processed
    frame to get the current list of available gaps.
    """
    def __init__(
        self,
        gap_threshold_meters: float = 0.8,
        bev_pixels_per_meter: float = 200.0,
        min_gap_pixels: Optional[float] = None,
        max_gap_pixels: Optional[float] = None,
        min_gap_frames: int = 5,
        smoothing_window: int = 5,
        max_display: Optional[int] = None,
    ) -> None:
        self.gap_threshold_meters = gap_threshold_meters
        self.bev_pixels_per_meter = bev_pixels_per_meter
        self.min_gap_pixels = min_gap_pixels
        self.max_gap = max_gap_pixels
        self.min_gap_frames = min_gap_frames
        self.smoothing_window = smoothing_window
        self.max_display = max_display

        self._gap_history: Dict[int, int] = defaultdict(int)
        self._prev_gap_count: int = 0

    def update(
        self,
        detections: List[Dict],
        bev_matrix: Optional[np.ndarray],
        roi_polygon: Optional[np.ndarray],
        bev_enabled: bool = True,
    ) -> List[Dict]:
        
        person_boxes = [det["bbox"] for det in detections if det["class_name"] == "person"]

        # ---- 1. Extract Candidates and Deduplicate (Custom NMS) ----
        mc_candidates: List[Dict] = []
        for det in detections:
            if det["class_name"] == "motorcycle":
                mc_candidates.append(det)

        # Confidence-based Custom NMS
        mc_points_orig = []
        # Sort candidates by confidence descending
        mc_candidates.sort(key=lambda x: x.get("confidence", 0.0), reverse=True)
        
        for det in mc_candidates:
            bc = det["ground_point"]
            
            # Check Euclidean distance in original space to avoid overlapping bounding boxes
            is_duplicate = False
            for existing_pt in mc_points_orig:
                dist = np.hypot(bc[0] - existing_pt[0], bc[1] - existing_pt[1])
                if dist < 30.0:  # 30px threshold for duplicates
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                mc_points_orig.append(bc)

        if len(mc_points_orig) < 2:
            self._gap_history.clear()
            self._prev_gap_count = 0
            return []

        pts_array = np.array(mc_points_orig, dtype=np.float32)

        # ---- 2. ROI filter (secondary safety net) ----
        if roi_polygon is not None:
            keep_mask = [
                point_in_polygon(tuple(pt), roi_polygon) for pt in pts_array
            ]
            pts_array = pts_array[keep_mask]

        if len(pts_array) < 2:
            self._gap_history.clear()
            self._prev_gap_count = 0
            return []

        # ---- 3. Linear Sort by X-axis (original space) ----
        # Guaranteed no crossing diagonal lines
        order = np.argsort(pts_array[:, 0])
        pts_array = pts_array[order]

        # ---- 4. BEV transform ----
        if bev_enabled and bev_matrix is not None:
            pts_bev = transform_points(pts_array, bev_matrix)
        else:
            pts_bev = pts_array.copy()

        # ---- 5. Euclidean distances between adjacent pairs on BEV ----
        diffs = np.diff(pts_bev, axis=0)
        distances_bev = np.linalg.norm(diffs, axis=1)

        # ---- 6 & 7. Threshold + temporal smoothing ----
        new_history: Dict[int, int] = defaultdict(int)
        gaps: List[Dict] = []

        for i, dist_bev in enumerate(distances_bev):
            # Calculate physical meters
            dist_meters = dist_bev / self.bev_pixels_per_meter

            # Compare to threshold in meters
            if dist_meters < self.gap_threshold_meters or dist_meters > 2.5:
                continue

            prev_count = self._gap_history.get(i, 0)
            count = prev_count + 1
            new_history[i] = count

            mid_bev = (pts_bev[i] + pts_bev[i + 1]) / 2.0
            mid_orig = (pts_array[i] + pts_array[i + 1]) / 2.0

            status = "available" if count >= self.min_gap_frames else "smoothing"

            gaps.append({
                "gap_id": i,
                "midpoint": tuple(mid_orig.tolist()),
                "midpoint_bev": tuple(mid_bev.tolist()),
                "pt1": tuple(pts_array[i].tolist()),
                "pt2": tuple(pts_array[i + 1].tolist()),
                "distance_m": float(dist_meters),
                "distance_bev": float(dist_bev),
                "status": status,
            })

        self._gap_history = new_history
        self._prev_gap_count = len(gaps)

        if self.max_display is not None:
            available = [g for g in gaps if g["status"] == "available"]
            smoothing = [g for g in gaps if g["status"] == "smoothing"]
            available.sort(key=lambda g: g["distance_m"], reverse=True)
            gaps = available[: self.max_display] + smoothing

        return gaps


def draw_gaps(
    frame: np.ndarray,
    gaps: List[Dict],
    radius: int = 14,
    draw_labels: bool = True,
    draw_distance: bool = True,
) -> np.ndarray:
    """
    Draw gap markers on *frame*.
    """
    for gap in gaps:
        mx, my = int(gap["midpoint"][0]), int(gap["midpoint"][1])

        if gap["status"] == "available":
            pt1 = (int(gap["pt1"][0]), int(gap["pt1"][1]))
            pt2 = (int(gap["pt2"][0]), int(gap["pt2"][1]))
            
            # Draw dashed yellow line between the two bikes
            draw_dashed_line(frame, pt1, pt2, (0, 215, 255), thickness=3, dash_length=15)

            if draw_labels:
                text = "GAP"
                if draw_distance:
                    text += f": ~{gap.get('distance_m', 0.0):.1f}m"
                    
                # Text background
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (mx - tw//2 - 5, my - th//2 - 5), (mx + tw//2 + 5, my + th//2 + 5), (0, 0, 0), -1)
                
                # Text foreground
                cv2.putText(
                    frame,
                    text,
                    (mx - tw//2, my + th//2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 215, 255),
                    2,
                    cv2.LINE_AA,
                )

    return frame

def draw_dashed_line(img, pt1, pt2, color, thickness=1, dash_length=10):
    dist = np.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1])
    dashes = max(1, int(dist / dash_length))
    for i in range(dashes):
        if i % 2 == 0:  # draw every other segment
            start = (
                int(pt1[0] + (pt2[0] - pt1[0]) * (i / dashes)),
                int(pt1[1] + (pt2[1] - pt1[1]) * (i / dashes))
            )
            end = (
                int(pt1[0] + (pt2[0] - pt1[0]) * ((i + 1) / dashes)),
                int(pt1[1] + (pt2[1] - pt1[1]) * ((i + 1) / dashes))
            )
            cv2.line(img, start, end, color, thickness)
