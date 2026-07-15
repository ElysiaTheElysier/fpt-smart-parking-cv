"""
export_roboflow_frames.py
=========================
Export a small, evenly-spaced subset of video frames suitable for
uploading to Roboflow (or any labeling tool).

Usage:
    python scripts/export_roboflow_frames.py \
        --video data/raw_videos/input_video.mp4 \
        --fps 1 \
        --max-frames 120 \
        --output data/roboflow_upload

If data/calibration/roi_points.json exists, frames are cropped to the
bounding rectangle of the ROI polygon so annotators only see the
parking area.
"""

import argparse
import json
import os
import sys
import cv2
import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export frames for Roboflow labeling."
    )
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to the input video file.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=1.0,
        help="Target sampling FPS (default: 1).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=120,
        help="Maximum number of frames to export (default: 120).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/roboflow_upload",
        help="Output directory for exported frames.",
    )
    parser.add_argument(
        "--roi",
        type=str,
        default="data/calibration/roi_points.json",
        help="Path to ROI polygon JSON (optional, for cropping).",
    )
    return parser.parse_args()


def load_roi_bbox(roi_path: str):
    """
    Load an ROI polygon from JSON and return its axis-aligned bounding
    rectangle as (x, y, w, h).  Returns None if the file doesn't exist.
    """
    if not os.path.isfile(roi_path):
        return None

    try:
        with open(roi_path, "r") as f:
            data = json.load(f)

        # Expect {"points": [[x1,y1], [x2,y2], ...]}
        points = np.array(data["points"], dtype=np.int32)
        x, y, w, h = cv2.boundingRect(points)
        print(f"[INFO] ROI bounding rectangle loaded from {roi_path}: "
              f"x={x}, y={y}, w={w}, h={h}")
        return (x, y, w, h)
    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        print(f"[WARN] Could not parse ROI file ({exc}). Skipping crop.")
        return None


def export_frames(
    video_path: str,
    target_fps: float,
    max_frames: int,
    output_dir: str,
    roi_path: str,
) -> int:
    """
    Sample frames from *video_path* at *target_fps*, capped at
    *max_frames*, and save to *output_dir*.

    Returns the number of frames saved.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        sys.exit(1)

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / original_fps if original_fps > 0 else 0

    print("=" * 60)
    print("EXPORT FOR ROBOFLOW")
    print("=" * 60)
    print(f"  Video         : {video_path}")
    print(f"  Original FPS  : {original_fps:.2f}")
    print(f"  Total frames  : {total_frames}")
    print(f"  Duration      : {duration_sec:.1f}s")
    print(f"  Target FPS    : {target_fps}")
    print(f"  Max frames    : {max_frames}")
    print("=" * 60)

    # Compute sampling interval
    if target_fps <= 0 or target_fps > original_fps:
        frame_interval = 1
    else:
        frame_interval = int(round(original_fps / target_fps))

    # Total candidate frames after FPS sampling
    candidate_count = total_frames // frame_interval

    # If candidates exceed max_frames, widen the interval further
    if candidate_count > max_frames and max_frames > 0:
        frame_interval = total_frames // max_frames
        print(
            f"[INFO] Candidate frames ({candidate_count}) exceed "
            f"max_frames ({max_frames}). Adjusted interval to "
            f"{frame_interval}."
        )

    # Load optional ROI crop
    roi_bbox = load_roi_bbox(roi_path)

    os.makedirs(output_dir, exist_ok=True)

    saved_count = 0
    frame_idx = 0

    pbar = tqdm(total=total_frames, desc="Exporting frames", unit="frame")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            # Crop to ROI bounding rectangle if available
            if roi_bbox is not None:
                x, y, w, h = roi_bbox
                # Clamp to frame dimensions
                fh, fw = frame.shape[:2]
                x1 = max(0, x)
                y1 = max(0, y)
                x2 = min(fw, x + w)
                y2 = min(fh, y + h)
                frame_out = frame[y1:y2, x1:x2]
            else:
                frame_out = frame

            filename = os.path.join(
                output_dir, f"roboflow_{frame_idx:06d}.jpg"
            )
            cv2.imwrite(filename, frame_out)
            saved_count += 1

            if saved_count >= max_frames:
                pbar.update(total_frames - frame_idx)
                break

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()

    print(f"\n[DONE] Exported {saved_count} frames to: {output_dir}")
    print()
    print("=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("  1. Go to https://app.roboflow.com and create a new project.")
    print("  2. Upload all images from the output directory above.")
    print("  3. Label two classes:  motorcycle  and  person")
    print("  4. Export the dataset in YOLOv8 format.")
    print("  5. Place the exported dataset in:  data/yolo_dataset/")
    print("  6. Run training:  python scripts/train_yolo.py \\")
    print("       --data data/yolo_dataset/data.yaml")
    print("=" * 60)

    return saved_count


def main():
    args = parse_args()

    if not os.path.isfile(args.video):
        print(f"[ERROR] Video file not found: {args.video}")
        sys.exit(1)

    export_frames(
        args.video,
        args.fps,
        args.max_frames,
        args.output,
        args.roi,
    )


if __name__ == "__main__":
    main()
