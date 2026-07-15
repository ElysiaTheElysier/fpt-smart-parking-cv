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
        gap_threshold_meters: float = 1.3,
        bev_pixels_per_meter: float = 200.0,
        min_gap_pixels: Optional[float] = None,
        max_gap_pixels: Optional[float] = None,
        min_gap_frames: int = 5,
        smoothing_window: int = 5,
        max_display: Optional[int] = None,
        exclusion_zones: Optional[List[np.ndarray]] = None,
    ) -> None:
        self.gap_threshold_meters = gap_threshold_meters
        self.bev_pixels_per_meter = bev_pixels_per_meter
        self.min_gap_pixels = min_gap_pixels
        self.max_gap = max_gap_pixels
        self.min_gap_frames = min_gap_frames
        self.smoothing_window = smoothing_window
        self.max_display = max_display
        self.exclusion_zones = exclusion_zones or []

        self._gap_history: Dict[int, int] = defaultdict(int)
        self._prev_gap_count: int = 0
        # Person Freeze: store last stable gaps computed when no persons were present
        self._stable_gaps: List[Dict] = []

    def _midpoint_in_exclusion_zone(self, midpoint: Tuple[float, float]) -> bool:
        """Check if a gap midpoint falls inside any exclusion zone polygon."""
        for zone in self.exclusion_zones:
            if point_in_polygon(midpoint, zone):
                return True
        return False

    def _gap_overlaps_any_bbox(
        self,
        pt1_orig: np.ndarray,
        pt2_orig: np.ndarray,
        all_bboxes: List[List[float]],
        pair_indices: Tuple[int, int],
    ) -> bool:
        """
        Check if any motorcycle bbox (other than the two forming this gap)
        overlaps with the virtual gap region.
        
        The virtual gap box spans from the right edge of the left bike's bbox
        to the left edge of the right bike's bbox, vertically covering the
        union of both bboxes' Y ranges.
        """
        idx_a, idx_b = pair_indices
        if len(all_bboxes) < 3:
            return False

        # Get the bboxes of the two gap-forming bikes
        bbox_a = all_bboxes[idx_a]
        bbox_b = all_bboxes[idx_b]

        # Virtual gap box: from right edge of left bike to left edge of right bike
        gap_x1 = min(bbox_a[2], bbox_b[2])  # right edge of left bike
        gap_x2 = max(bbox_a[0], bbox_b[0])  # left edge of right bike
        gap_y1 = min(bbox_a[1], bbox_b[1])
        gap_y2 = max(bbox_a[3], bbox_b[3])

        # Ensure valid box - if the two bikes overlap horizontally or vertically,
        # we still want to check if a THIRD bike is in the middle. Just make sure
        # the gap box coordinates are valid (min < max).
        if gap_x1 >= gap_x2:
            gap_x1, gap_x2 = gap_x2, gap_x1
        if gap_y1 >= gap_y2:
            gap_y1, gap_y2 = gap_y2, gap_y1

        gap_box = [gap_x1, gap_y1, gap_x2, gap_y2]

        for k, bbox in enumerate(all_bboxes):
            if k == idx_a or k == idx_b:
                continue
            iou = calculate_iou(gap_box, bbox)
            if iou > 0.0:
                return True
        return False

    def update(
        self,
        detections: List[Dict],
        bev_matrix: Optional[np.ndarray],
        roi_polygon: Optional[np.ndarray],
        bev_enabled: bool = True,
    ) -> List[Dict]:

        person_count = sum(1 for det in detections if det["class_name"] == "person")

        # ── Person Freeze: if any person is in the scene, return last stable gaps ──
        if person_count > 0:
            return self._stable_gaps

        # ---- 1. Extract Candidates and Deduplicate (Custom NMS) ----
        mc_candidates: List[Dict] = []
        for det in detections:
            if det["class_name"] == "motorcycle":
                mc_candidates.append(det)

        # Confidence-based Custom NMS
        mc_points_orig = []
        mc_bboxes = []  # Track bboxes in parallel for overlap checking
        mc_track_ids = []

        # Sort candidates by confidence descending
        mc_candidates.sort(key=lambda x: x.get("confidence", 0.0), reverse=True)
        
        filtered_candidates = []
        for det in mc_candidates:
            is_overlap = False
            for existing_det in filtered_candidates:
                if calculate_iou(det["bbox"], existing_det["bbox"]) > 0.45:
                    is_overlap = True
                    break
            if not is_overlap:
                filtered_candidates.append(det)
        
        for det in filtered_candidates:
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
                mc_bboxes.append(det["bbox"])
                mc_track_ids.append(det.get("track_id", None))

        if len(mc_points_orig) < 2:
            self._gap_history.clear()
            self._prev_gap_count = 0
            self._stable_gaps = []
            return []

        pts_array = np.array(mc_points_orig, dtype=np.float32)

        # ---- 2. ROI filtering ----
        if roi_polygon is not None:
            keep_mask = [
                point_in_polygon(tuple(pt), roi_polygon) for pt in pts_array
            ]
            pts_array = pts_array[keep_mask]
            mc_bboxes = [b for b, k in zip(mc_bboxes, keep_mask) if k]
            mc_track_ids = [t for t, k in zip(mc_track_ids, keep_mask) if k]

        if len(pts_array) < 2:
            self._gap_history.clear()
            self._prev_gap_count = 0
            self._stable_gaps = []
            return []

        # ---- 3. Linear Sort by X-axis (original space) ----
        # Guaranteed no crossing diagonal lines
        order = np.argsort(pts_array[:, 0])
        pts_array = pts_array[order]
        mc_bboxes = [mc_bboxes[o] for o in order]
        mc_track_ids = [mc_track_ids[o] for o in order]

        # ---- 4. BEV transform ----
        if bev_enabled and bev_matrix is not None:
            pts_bev = transform_points(pts_array, bev_matrix)
        else:
            pts_bev = pts_array.copy()

        # ---- 5. Euclidean distances between adjacent pairs on BEV ----
        diffs = np.diff(pts_bev, axis=0)
        distances_bev = np.linalg.norm(diffs, axis=1)

        # ---- 6 & 7. Threshold + temporal smoothing ----
        new_history: Dict[str, int] = defaultdict(int)
        gaps: List[Dict] = []

        for i, dist_bev in enumerate(distances_bev):
            # Calculate physical meters
            dist_meters = dist_bev / self.bev_pixels_per_meter

            # 1. Distance thresholds
            if dist_meters < self.gap_threshold_meters or dist_meters > 10.0:
                continue
                
            # Vertical gap blocker: Y-distance in BEV must not exceed 3.0 meters
            y_dist_bev = abs(pts_bev[i][1] - pts_bev[i + 1][1])
            if y_dist_bev > 3.0 * self.bev_pixels_per_meter:
                continue

            mid_orig = (pts_array[i] + pts_array[i + 1]) / 2.0

            # ── Exclusion Zone Check ──
            if self._midpoint_in_exclusion_zone(tuple(mid_orig.tolist())):
                continue

            # ── BBox Overlap Check ──
            if self._gap_overlaps_any_bbox(
                pts_array[i], pts_array[i + 1],
                mc_bboxes, (i, i + 1)
            ):
                continue

            mid_bev = (pts_bev[i] + pts_bev[i + 1]) / 2.0

            # Match to existing gaps spatially to be robust to tracker/index failures
            # This ensures gaps don't flicker even if a person occludes the motorcycles.
            best_id = None
            best_dist = float('inf')
            for gid in self._gap_history.keys():
                if gid.startswith("spatial_"):
                    try:
                        _, x_str, y_str = gid.split("_")
                        gx, gy = float(x_str), float(y_str)
                        dist = np.linalg.norm(np.array([gx, gy]) - mid_bev)
                        if dist < best_dist and dist < 0.8 * self.bev_pixels_per_meter: # 0.8m tolerance
                            best_dist = dist
                            best_id = gid
                    except:
                        pass
            
            if best_id is not None:
                # Update key with new smoothed position
                _, x_str, y_str = best_id.split("_")
                gx, gy = float(x_str), float(y_str)
                new_x = 0.8 * gx + 0.2 * mid_bev[0]
                new_y = 0.8 * gy + 0.2 * mid_bev[1]
                gap_id = f"spatial_{new_x:.1f}_{new_y:.1f}"
                prev_count = self._gap_history.get(best_id, 0)
            else:
                gap_id = f"spatial_{mid_bev[0]:.1f}_{mid_bev[1]:.1f}"
                prev_count = 0

            count = prev_count + 1
            new_history[gap_id] = count

            status = "available" if count >= self.min_gap_frames else "smoothing"

            gaps.append({
                "gap_id": gap_id,
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

        # Update stable gaps (person-free snapshot)
        self._stable_gaps = gaps

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
