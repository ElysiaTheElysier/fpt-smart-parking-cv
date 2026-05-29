"""
bev.py — Bird's Eye View (Perspective Transform) utilities.
=============================================================
Provides functions to:
- Load calibration points from JSON.
- Compute a perspective transform matrix.
- Warp frames / transform point coordinates into BEV space.
- Draw helper overlays (points, polygons).

NOTE ON CALIBRATION (pixel-level only)
---------------------------------------
The current implementation works entirely in **pixel space**.  No
real-world metric calibration is applied.  For a rough real-world
estimate you could use A4 paper sheets as reference objects:

    A4 = 21 cm × 29.7 cm
    4 sheets side by side (landscape) ≈ 84 cm wide
    This is close to the ~80 cm handlebar width of a typical motorcycle.

This can be added later without changing the API below.
"""

import json
import os
from typing import List, Optional, Tuple

import cv2
import numpy as np


# ── I/O ──────────────────────────────────────────────────────────────────────

def load_points(json_path: str) -> Optional[np.ndarray]:
    """
    Load calibration points from a JSON file.

    Expected format::

        {"points": [[x1, y1], [x2, y2], ...]}

    Returns an (N, 2) float32 numpy array, or *None* if the file does
    not exist.
    """
    if not os.path.isfile(json_path):
        print(f"[WARN] Calibration file not found: {json_path}")
        return None

    with open(json_path, "r") as f:
        data = json.load(f)

    points = np.array(data["points"], dtype=np.float32)
    print(f"[INFO] Loaded {len(points)} points from {json_path}")
    return points


def save_points(json_path: str, points: np.ndarray) -> None:
    """Save an (N, 2) array of points to JSON."""
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    data = {"points": points.tolist()}
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[INFO] Saved {len(points)} points to {json_path}")


# ── Perspective Transform ────────────────────────────────────────────────────

def compute_perspective_transform(
    src_points: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    """
    Compute the 3×3 perspective transform matrix that maps four source
    points to a rectangle of size (*width*, *height*).

    Parameters
    ----------
    src_points : (4, 2) float32 array
        Source quadrilateral corners in order:
        top-left, top-right, bottom-right, bottom-left.
    width : int
        Desired output width in pixels.
    height : int
        Desired output height in pixels.

    Returns
    -------
    np.ndarray
        3×3 homography matrix.
    """
    src = np.array(src_points, dtype=np.float32).reshape(4, 2)
    dst = np.array(
        [[0, 0], [width, 0], [width, height], [0, height]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(src, dst)
    return matrix


def warp_frame(
    frame: np.ndarray,
    matrix: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    """
    Apply a perspective warp to *frame* using the given homography
    *matrix*, producing an output image of (*width*, *height*).
    """
    return cv2.warpPerspective(frame, matrix, (width, height))


def transform_points(
    points: np.ndarray,
    matrix: np.ndarray,
) -> np.ndarray:
    """
    Transform an (N, 2) array of 2-D points through the 3×3 perspective
    *matrix*.  Returns an (N, 2) float32 array.
    """
    if len(points) == 0:
        return np.empty((0, 2), dtype=np.float32)

    pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(pts, matrix)
    return transformed.reshape(-1, 2)


# ── Drawing helpers ──────────────────────────────────────────────────────────

_POINT_COLORS = [
    (0, 255, 0),    # green  — top-left
    (0, 255, 255),  # yellow — top-right
    (0, 0, 255),    # red    — bottom-right
    (255, 0, 0),    # blue   — bottom-left
]

_POINT_LABELS = ["TL", "TR", "BR", "BL"]


def draw_points(
    frame: np.ndarray,
    points: np.ndarray,
    radius: int = 8,
    thickness: int = -1,
) -> np.ndarray:
    """
    Draw labelled circles on *frame* for each point in *points*.

    Returns the modified frame (drawn in-place).
    """
    overlay = frame.copy()
    for i, (x, y) in enumerate(points):
        color = _POINT_COLORS[i % len(_POINT_COLORS)]
        label = _POINT_LABELS[i % len(_POINT_LABELS)] if i < 4 else str(i)
        pt = (int(x), int(y))
        cv2.circle(overlay, pt, radius, color, thickness)
        cv2.putText(
            overlay,
            label,
            (pt[0] + 12, pt[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )
    return overlay


def draw_polygon(
    frame: np.ndarray,
    polygon: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
    alpha: float = 0.25,
    line_thickness: int = 2,
) -> np.ndarray:
    """
    Draw a filled semi-transparent polygon on *frame*.

    Parameters
    ----------
    polygon : (N, 2) array of int
        Vertices of the polygon.
    color : BGR tuple
    alpha : float
        Fill opacity (0 = fully transparent, 1 = fully opaque).
    line_thickness : int
        Outline thickness.

    Returns the modified frame.
    """
    overlay = frame.copy()
    pts = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))

    # Semi-transparent fill
    cv2.fillPoly(overlay, [pts], color)
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # Solid outline
    cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=line_thickness)

    return frame
