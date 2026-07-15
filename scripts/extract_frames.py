"""
extract_frames.py
=================
Extract frames from a video file at a configurable sample rate (FPS).

Usage:
    python scripts/extract_frames.py \
        --video data/raw_videos/input_video.mp4 \
        --fps 1 \
        --output data/extracted_frames
"""

import argparse
import os
import sys
import cv2
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract frames from a video at a target sampling FPS."
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
        help="Target sampling FPS (default: 1 frame per second).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/extracted_frames",
        help="Output directory for extracted frames.",
    )
    return parser.parse_args()


def extract_frames(video_path: str, target_fps: float, output_dir: str) -> int:
    """
    Read *video_path*, sample frames at *target_fps*, and save them as
    JPEG images in *output_dir*.

    Returns the number of frames saved.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        sys.exit(1)

    # ---- Video metadata ----
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / original_fps if original_fps > 0 else 0

    minutes = int(duration_sec // 60)
    seconds = int(duration_sec % 60)

    print("=" * 60)
    print("VIDEO METADATA")
    print("=" * 60)
    print(f"  Path          : {video_path}")
    print(f"  Original FPS  : {original_fps:.2f}")
    print(f"  Total frames  : {total_frames}")
    print(f"  Duration      : {minutes}m {seconds}s ({duration_sec:.1f}s)")
    print(f"  Target FPS    : {target_fps}")
    print("=" * 60)

    # Interval between sampled frames (in source-frame indices)
    if target_fps <= 0 or target_fps > original_fps:
        print(
            f"[WARN] target_fps={target_fps} is invalid or exceeds "
            f"original FPS ({original_fps:.2f}). Using original FPS."
        )
        frame_interval = 1
    else:
        frame_interval = int(round(original_fps / target_fps))

    expected_samples = total_frames // frame_interval
    print(
        f"[INFO] Sampling every {frame_interval} frames "
        f"(~{expected_samples} images)."
    )
    print(
        f"[INFO] A {minutes}m{seconds}s video at {original_fps:.0f} FPS has "
        f"{total_frames} frames. We only sample a small subset for "
        f"annotation — labeling all frames is unnecessary."
    )

    # ---- Create output directory ----
    os.makedirs(output_dir, exist_ok=True)

    saved_count = 0
    frame_idx = 0

    pbar = tqdm(total=total_frames, desc="Extracting frames", unit="frame")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            filename = os.path.join(
                output_dir, f"frame_{frame_idx:06d}.jpg"
            )
            cv2.imwrite(filename, frame)
            saved_count += 1

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()

    print(f"\n[DONE] Saved {saved_count} frames to: {output_dir}")
    return saved_count


def main():
    args = parse_args()

    if not os.path.isfile(args.video):
        print(f"[ERROR] Video file not found: {args.video}")
        sys.exit(1)

    extract_frames(args.video, args.fps, args.output)


if __name__ == "__main__":
    main()
