"""
parking_rules.py — Simple rule-based parking violation detection.
================================================================
Current rules (intentionally minimal):

* If ``parking_rules_enabled`` is False in the config, **no violations
  are reported** (safe for demo mode).
* If a dedicated ``lane_polygon`` is provided, motorcycles whose
  bottom-center falls inside the lane are flagged.
* The ROI polygon alone does **not** trigger violations — it is only
  an analysis region, not a punishment boundary.

Additional rules (e.g. angle-based) can be added here later without
changing the rest of the pipeline.
"""

from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from backend.core.utils import (
    COLOR_VIOLATION,
    bbox_bottom_center,
    draw_text_with_bg,
    point_in_polygon,
)


def check_violations(
    detections: List[Dict],
    roi_polygon: Optional[np.ndarray],
    *,
    enabled: bool = True,
    lane_polygon: Optional[np.ndarray] = None,
) -> List[Dict]:
    """
    Check each motorcycle detection against parking rules.

    Parameters
    ----------
    detections : list[dict]
        Detection list from the detector.
    roi_polygon : (N, 2) ndarray or None
        Allowed parking zone (for reference only — NOT used for
        violation flagging by itself).
    enabled : bool
        Master switch.  If False, returns an empty list immediately.
    lane_polygon : (N, 2) ndarray or None
        Optional lane polygon.  Motorcycles inside this polygon are
        flagged as "Possible lane blocking".

    Returns
    -------
    list[dict]
    """
    # Master switch — demo mode turns this off
    if not enabled:
        return []

    # Without a dedicated lane polygon, there is nothing to check
    if lane_polygon is None:
        return []

    violations: List[Dict] = []

    for det in detections:
        if det["class_name"] != "motorcycle":
            continue

        bc = bbox_bottom_center(det["bbox"])

        if point_in_polygon(bc, lane_polygon):
            violations.append({
                "detection": det,
                "bottom_center": bc,
                "rule": "inside_lane",
                "message": "Possible lane blocking",
            })

    return violations


def draw_violations(
    frame: np.ndarray,
    violations: List[Dict],
) -> np.ndarray:
    """
    Draw violation markers on *frame*.

    - Red bounding box outline around the offending motorcycle.
    - A warning label near the bottom-center point.
    """
    for viol in violations:
        bbox = viol["detection"]["bbox"]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        bx, by = int(viol["bottom_center"][0]), int(viol["bottom_center"][1])

        # Red outline
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_VIOLATION, 2)

        # Warning label
        draw_text_with_bg(
            frame,
            viol["message"],
            (bx - 60, by + 18),
            font_scale=0.5,
            color=COLOR_VIOLATION,
            bg_color=(40, 40, 40),
            thickness=1,
        )

    return frame
