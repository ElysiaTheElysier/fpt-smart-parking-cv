"""
select_bev_points.py
====================
Interactive tool to select four ground-plane points on a video frame
for Bird's Eye View (perspective) transform calibration.

Usage:
    python scripts/select_bev_points.py \
        --video data/raw_videos/input_video.mp4 \
        --output data/calibration/bev_points.json

Controls:
    Left-click  — place a point (max 4)
    r           — reset all points
    s           — save points to JSON + debug image
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

from backend.core.bev import draw_points, save_points  # noqa: E402


# ── Constants ────────────────────────────────────────────────────────────────

_WINDOW_NAME = "Select BEV Points (4 corners: TL → TR → BR → BL)"
_POINT_ORDER = ["top-left", "top-right", "bottom-right", "bottom-left"]
_MAX_POINTS = 4
_DEBUG_DIR = "data/outputs/screenshots"
_DEBUG_FILENAME = "bev_points_debug.jpg"


# ── Global state (used by the OpenCV mouse callback) ─────────────────────────

_points: list = []
_frame_clean: np.ndarray = None  # type: ignore[assignment]


def _mouse_callback(event: int, x: int, y: int, flags: int, param) -> None:
    """Handle left-click to add a point."""
    global _points
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(_points) < _MAX_POINTS:
            _points.append([x, y])
            ordinal = _POINT_ORDER[len(_points) - 1]
            print(f"[CLICK] Point {len(_points)}/{_MAX_POINTS} "
                  f"({ordinal}): ({x}, {y})")
        else:
            print(f"[WARN] Already have {_MAX_POINTS} points. "
                  f"Press 'r' to reset or 's' to save.")


def _draw_overlay(frame: np.ndarray, points: list) -> np.ndarray:
    """Draw current points + connecting lines on a copy of the frame."""
    vis = frame.copy()
    if points:
        pts = np.array(points, dtype=np.float32)
        vis = draw_points(vis, pts)

        # Draw lines connecting consecutive points
        for i in range(len(points) - 1):
            cv2.line(
                vis,
                tuple(points[i]),
                tuple(points[i + 1]),
                (0, 255, 0),
                2,
            )
        # Close polygon if 4 points
        if len(points) == _MAX_POINTS:
            cv2.line(vis, tuple(points[-1]), tuple(points[0]), (0, 255, 0), 2)

    # HUD instructions
    _draw_hud(vis, points)
    return vis


def _draw_hud(frame: np.ndarray, points: list) -> None:
    """Draw a small instruction panel at the top-left."""
    lines = [
        f"Points: {len(points)}/{_MAX_POINTS}",
        "Click to place a point.",
        "Order: TL -> TR -> BR -> BL",
        "[r] Reset  [s] Save  [q] Quit",
    ]
    if len(points) < _MAX_POINTS:
        next_label = _POINT_ORDER[len(points)]
        lines.append(f"Next: {next_label}")
    else:
        lines.append("All points set. Press 's' to save.")

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
        description="Select 4 BEV calibration points on a video frame."
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
        default="data/calibration/bev_points.json",
        help="Path to save the selected points as JSON.",
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
    print("BEV POINT SELECTION")
    print("=" * 60)
    print(f"  Video  : {args.video}")
    print(f"  Frame  : {args.frame_index}")
    print(f"  Output : {args.output}")
    print()
    print("  Click 4 ground-plane corners in order:")
    print("    1) Top-Left    2) Top-Right")
    print("    3) Bottom-Right  4) Bottom-Left")
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
            print("[INFO] All points reset.")

        elif key == ord("s"):
            if len(_points) != _MAX_POINTS:
                print(f"[WARN] Need exactly {_MAX_POINTS} points "
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

            print("[DONE] BEV points saved successfully.")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
