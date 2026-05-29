"""
utils.py — Shared utility functions for the parking CV pipeline.
================================================================
"""

import os
import sys
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import yaml


# ── Colour palette for drawing ───────────────────────────────────────────────

# BGR tuples
COLOR_MOTORCYCLE = (0, 200, 0)       # green
COLOR_PERSON = (255, 165, 0)         # orange-ish
COLOR_GAP_AVAILABLE = (0, 255, 255)  # yellow
COLOR_GAP_OCCUPIED = (128, 128, 128) # grey
COLOR_VIOLATION = (0, 0, 255)        # red
COLOR_ROI = (255, 200, 0)            # cyan-ish
COLOR_TEXT_BG = (40, 40, 40)         # dark grey
COLOR_WHITE = (255, 255, 255)


# COCO class-id → friendly name (only the ones we care about)
COCO_NAMES = {0: "person", 3: "motorcycle"}


# ── Config loading ───────────────────────────────────────────────────────────

def load_config(yaml_path: str) -> Dict[str, Any]:
    """
    Load a YAML configuration file and resolve relative paths against
    the **project root** (two levels up from ``backend/core/``).

    The project root is determined by going up from the config file's
    own directory.
    """
    yaml_path = os.path.abspath(yaml_path)
    if not os.path.isfile(yaml_path):
        print(f"[ERROR] Config file not found: {yaml_path}")
        sys.exit(1)

    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Resolve project root: config lives at  <root>/backend/config.yaml
    config_dir = os.path.dirname(yaml_path)
    project_root = os.path.dirname(config_dir)  # one level up from backend/

    # If the config is at the project root already, keep it
    # (handles both  backend/config.yaml  and  ./config.yaml)
    if not os.path.isdir(os.path.join(project_root, "backend")):
        project_root = config_dir

    cfg["_project_root"] = project_root

    # Resolve selected path keys relative to project root
    _path_keys = [
        "video_path",
        "output_video_path",
        "model_path",
    ]
    for key in _path_keys:
        if key in cfg and not os.path.isabs(cfg[key]):
            cfg[key] = os.path.join(project_root, cfg[key])

    print(f"[INFO] Config loaded from {yaml_path}")
    print(f"[INFO] Project root resolved to: {project_root}")
    return cfg


# ── Filesystem helpers ───────────────────────────────────────────────────────

def ensure_dir(path: str) -> str:
    """Create *path* (and parents) if it doesn't exist.  Returns *path*."""
    os.makedirs(path, exist_ok=True)
    return path


# ── Geometry helpers ─────────────────────────────────────────────────────────

def point_in_polygon(
    point: Tuple[float, float],
    polygon: np.ndarray,
) -> bool:
    """
    Return True if *point* ``(x, y)`` lies inside *polygon*
    (an (N, 2) array of vertices).

    Uses ``cv2.pointPolygonTest`` — a positive result means the point
    is inside or on the edge.
    """
    contour = np.array(polygon, dtype=np.float32).reshape((-1, 1, 2))
    dist = cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), False)
    return dist >= 0


def bbox_bottom_center(bbox: List[float]) -> Tuple[float, float]:
    """
    Given a bounding box ``[x1, y1, x2, y2]``, return the bottom-center
    point ``(cx, y2)`` — used as an approximation of the ground
    contact point.
    """
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    return (cx, y2)


# ── Drawing helpers ──────────────────────────────────────────────────────────

def draw_text_with_bg(
    frame: np.ndarray,
    text: str,
    origin: Tuple[int, int],
    font_scale: float = 0.6,
    color: Tuple[int, int, int] = COLOR_WHITE,
    bg_color: Tuple[int, int, int] = COLOR_TEXT_BG,
    thickness: int = 1,
    padding: int = 4,
) -> None:
    """Draw *text* on *frame* with a filled background rectangle."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = origin
    cv2.rectangle(
        frame,
        (x - padding, y - th - padding),
        (x + tw + padding, y + baseline + padding),
        bg_color,
        cv2.FILLED,
    )
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
