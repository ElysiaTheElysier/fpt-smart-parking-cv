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


class GapAnalyzer:
    """
    Stateful gap analyser — call :meth:`update` once per processed
    frame to get the current list of available gaps.

    Parameters
    ----------
    gap_threshold : float
        Minimum pixel distance in BEV space to consider a gap
        "available".  Alias for *min_gap_pixels*.
    min_gap_pixels : float or None
        Explicit minimum gap distance.  Overrides *gap_threshold* if
        provided.
    max_gap_pixels : float or None
        Gaps wider than this are discarded (edge / noise artefacts).
    min_gap_frames : int
        A gap must be seen for this many consecutive frames before it
        is reported as available.
    smoothing_window : int
        Reserved for future use.
    max_display : int or None
        Maximum number of available gaps returned per frame (for demo
        clarity).  ``None`` means unlimited.
    """

    def __init__(
        self,
        gap_threshold: float = 120,
        min_gap_pixels: Optional[float] = None,
        max_gap_pixels: Optional[float] = None,
        min_gap_frames: int = 5,
        smoothing_window: int = 5,
        max_display: Optional[int] = None,
    ) -> None:
        # min_gap_pixels takes precedence over gap_threshold
        self.min_gap = min_gap_pixels if min_gap_pixels is not None else gap_threshold
        self.max_gap = max_gap_pixels
        self.min_gap_frames = min_gap_frames
        self.smoothing_window = smoothing_window
        self.max_display = max_display

        # _gap_history[slot_index] = consecutive-frame-count
        self._gap_history: Dict[int, int] = defaultdict(int)
        self._prev_gap_count: int = 0

    def update(
        self,
        detections: List[Dict],
        bev_matrix: Optional[np.ndarray],
        roi_polygon: Optional[np.ndarray],
        bev_enabled: bool = True,
    ) -> List[Dict]:
        """
        Compute gaps for the current frame.

        Parameters
        ----------
        detections : list[dict]
            Detection list — **should already be ROI-filtered** by the
            caller if ``roi_only`` is enabled.
        bev_matrix : (3, 3) ndarray or None
        roi_polygon : (N, 2) ndarray or None
        bev_enabled : bool

        Returns
        -------
        list[dict]
        """
        # ---- 1. Extract motorcycle bottom-center points ----
        mc_points_orig: List[Tuple[float, float]] = []

        for det in detections:
            if det["class_name"] != "motorcycle":
                continue
            bc = bbox_bottom_center(det["bbox"])
            mc_points_orig.append(bc)

        if len(mc_points_orig) < 2:
            self._gap_history.clear()
            self._prev_gap_count = 0
            return []

        pts_array = np.array(mc_points_orig, dtype=np.float32)

        # ---- 2. BEV transform ----
        if bev_enabled and bev_matrix is not None:
            pts_bev = transform_points(pts_array, bev_matrix)
        else:
            pts_bev = pts_array.copy()

        # ---- 3. ROI filter (secondary safety net) ----
        if roi_polygon is not None:
            keep_mask = [
                point_in_polygon(tuple(pt), roi_polygon) for pt in pts_array
            ]
            pts_bev = pts_bev[keep_mask]
            pts_array = pts_array[keep_mask]

        if len(pts_bev) < 2:
            self._gap_history.clear()
            self._prev_gap_count = 0
            return []

        # ---- 4. Sort by dominant axis ----
        spread_x = np.ptp(pts_bev[:, 0])
        spread_y = np.ptp(pts_bev[:, 1])
        sort_axis = 0 if spread_x >= spread_y else 1
        order = np.argsort(pts_bev[:, sort_axis])
        pts_bev = pts_bev[order]
        pts_array = pts_array[order]

        # ---- 5. Euclidean distances between adjacent ----
        diffs = np.diff(pts_bev, axis=0)
        distances = np.linalg.norm(diffs, axis=1)
        print(f"[DEBUG GAP] sort_axis={sort_axis}, distances: {np.round(distances)}")

        # ---- 6 & 7. Threshold + temporal smoothing ----
        new_history: Dict[int, int] = defaultdict(int)
        gaps: List[Dict] = []

        for i, dist in enumerate(distances):
            # Must be within [min_gap, max_gap]
            if dist < self.min_gap:
                continue
            if self.max_gap is not None and dist > self.max_gap:
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
                "distance": float(dist),
                "status": status,
            })

        self._gap_history = new_history
        self._prev_gap_count = len(gaps)

        # ---- Cap the number of displayed available gaps ----
        if self.max_display is not None:
            available = [g for g in gaps if g["status"] == "available"]
            smoothing = [g for g in gaps if g["status"] == "smoothing"]
            # Keep only top N available (sorted by distance, largest first)
            available.sort(key=lambda g: g["distance"], reverse=True)
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

    Parameters
    ----------
    draw_labels : bool
        If False, only draw a circle marker (no text).
    draw_distance : bool
        If False, suppress the "NNNpx" distance label.
    """
    for gap in gaps:
        mx, my = int(gap["midpoint"][0]), int(gap["midpoint"][1])

        if gap["status"] == "available":
            cv2.circle(frame, (mx, my), radius, COLOR_GAP_AVAILABLE, 2)

            if draw_labels and draw_distance:
                dist_text = f"{gap['distance']:.0f}px"
                cv2.putText(
                    frame,
                    dist_text,
                    (mx + radius + 4, my + 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    COLOR_GAP_AVAILABLE,
                    1,
                    cv2.LINE_AA,
                )
        # "smoothing" gaps are NOT drawn at all in demo mode
        # (the caller can still access them via the returned list)

    return frame
