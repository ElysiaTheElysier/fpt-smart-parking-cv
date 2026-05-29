"""
select_roi.py
=============
Interactive tool to draw a Region-of-Interest polygon around the
parking area on a video frame.

Usage:
    python scripts/select_roi.py \
        --video data/raw_videos/input_video.mp4 \
        --output data/calibration/roi_points.json

Controls:
    Left-click  — place a vertex
    r           — reset all vertices
    s           — save polygon to JSON + debug image
    q           — quit without saving
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so we can import backend modules
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from backend.core.bev import draw_polygon, save_points  # noqa: E402


# ── Constants ────────────────────────────────────────────────────────────────

_WINDOW_NAME = "Select ROI Polygon (parking area)"
_MIN_POINTS = 3  # a polygon needs at least 3 vertices
_DEBUG_DIR = "data/outputs/screenshots"
_DEBUG_FILENAME = "roi_points_debug.jpg"

_COLOR_VERTEX = (0, 200, 255)   # orange
_COLOR_LINE = (0, 255, 0)       # green
_COLOR_CLOSE = (0, 255, 255)    # yellow — closing segment


# ── Global state ─────────────────────────────────────────────────────────────

_points: list = []
_frame_clean: np.ndarray = None  # type: ignore[assignment]


def _mouse_callback(event: int, x: int, y: int, flags: int, param) -> None:
    """Handle left-click to add a vertex."""
    global _points
    if event == cv2.EVENT_LBUTTONDOWN:
        _points.append([x, y])
        print(f"[CLICK] Vertex {len(_points)}: ({x}, {y})")


def _draw_overlay(frame: np.ndarray, points: list) -> np.ndarray:
    """Draw current polygon outline + vertices on a copy of the frame."""
    vis = frame.copy()

    if len(points) >= _MIN_POINTS:
        # Semi-transparent fill
        vis = draw_polygon(vis, np.array(points), color=_COLOR_LINE, alpha=0.20)

    # Draw edges
    for i in range(len(points) - 1):
        cv2.line(vis, tuple(points[i]), tuple(points[i + 1]), _COLOR_LINE, 2)

    # Closing line (dashed feel — just use a different colour)
    if len(points) >= _MIN_POINTS:
        cv2.line(vis, tuple(points[-1]), tuple(points[0]), _COLOR_CLOSE, 1)

    # Draw vertices as circles
    for i, (px, py) in enumerate(points):
        cv2.circle(vis, (px, py), 6, _COLOR_VERTEX, -1)
        cv2.putText(
            vis,
            str(i + 1),
            (px + 10, py - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            _COLOR_VERTEX,
            2,
        )

    _draw_hud(vis, points)
    return vis


def _draw_hud(frame: np.ndarray, points: list) -> None:
    """Draw instruction text at the top-left."""
    lines = [
        f"Vertices: {len(points)}",
        "Click to add vertices around the parking area.",
        f"Need at least {_MIN_POINTS} vertices.",
        "[r] Reset  [s] Save  [q] Quit",
    ]
    y0 = 25
    for i, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (10, y0 + i * 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Select an ROI polygon around the parking area."
    )
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to the input video file.",
    )
    parser.add_argument(
        "--frame-index",
        type=int,
        default=0,
        help="Index of the frame to display (default: 0 = first frame).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/calibration/roi_points.json",
        help="Path to save the ROI polygon as JSON.",
    )
    return parser.parse_args()


def main():
    global _points, _frame_clean

    args = parse_args()

    # ---- Load frame ----
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {args.video}")
        sys.exit(1)

    if args.frame_index > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame_index)

    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"[ERROR] Cannot read frame {args.frame_index} from {args.video}")
        sys.exit(1)

    _frame_clean = frame.copy()
    _points = []

    print("=" * 60)
    print("ROI POLYGON SELECTION")
    print("=" * 60)
    print(f"  Video  : {args.video}")
    print(f"  Frame  : {args.frame_index}")
    print(f"  Output : {args.output}")
    print()
    print("  Click around the parking area to define the ROI polygon.")
    print(f"  Minimum {_MIN_POINTS} vertices required.")
    print()
    print("  Keys: [r] Reset  [s] Save  [q] Quit")
    print("=" * 60)

    # ---- Window setup ----
    cv2.namedWindow(_WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(_WINDOW_NAME, 1280, 720)
    cv2.setMouseCallback(_WINDOW_NAME, _mouse_callback)

    while True:
        vis = _draw_overlay(_frame_clean, _points)
        cv2.imshow(_WINDOW_NAME, vis)

        key = cv2.waitKey(30) & 0xFF

        if key == ord("q"):
            print("[INFO] Quit without saving.")
            break

        elif key == ord("r"):
            _points = []
            print("[INFO] All vertices reset.")

        elif key == ord("s"):
            if len(_points) < _MIN_POINTS:
                print(f"[WARN] Need at least {_MIN_POINTS} vertices "
                      f"(have {len(_points)}). Keep clicking.")
                continue

            # Save JSON
            pts_arr = np.array(_points, dtype=np.float32)
            save_points(args.output, pts_arr)

            # Save debug image
            os.makedirs(_DEBUG_DIR, exist_ok=True)
            debug_path = os.path.join(_DEBUG_DIR, _DEBUG_FILENAME)
            debug_img = _draw_overlay(_frame_clean, _points)
            cv2.imwrite(debug_path, debug_img)
            print(f"[INFO] Debug image saved to {debug_path}")

            print("[DONE] ROI polygon saved successfully.")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
