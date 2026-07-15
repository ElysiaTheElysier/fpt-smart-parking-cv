"""
train_yolo.py — Train or fine-tune a YOLOv8 model on a custom dataset.
======================================================================
If a labelled YOLO dataset exists at the specified path, training
starts immediately.  Otherwise, step-by-step instructions are printed
to guide the user through the Roboflow labelling workflow.

Usage:
    python scripts/train_yolo.py \
        --data data/yolo_dataset/data.yaml \
        --epochs 50 \
        --imgsz 640 \
        --model yolov8n.pt
"""

import argparse
import os
import sys
import shutil


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 on a custom motorcycle/person dataset."
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/yolo_dataset/data.yaml",
        help="Path to the YOLO dataset data.yaml.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Training image size (default: 640).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Base model checkpoint (default: yolov8n.pt).",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size (default: 16).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/best.pt",
        help="Where to copy the best model after training.",
    )
    return parser.parse_args()


def print_labelling_guide():
    """Print instructions for creating a labelled dataset."""
    print()
    print("=" * 60)
    print("  DATASET NOT FOUND — LABELLING GUIDE")
    print("=" * 60)
    print()
    print("  To train a custom YOLO model you need labelled images.")
    print("  Follow these steps:")
    print()
    print("  1. Export frames from your video:")
    print("     python scripts/export_roboflow_frames.py \\")
    print("         --video data/raw_videos/input_video.mp4 \\")
    print("         --fps 1 --max-frames 120 \\")
    print("         --output data/roboflow_upload")
    print()
    print("  2. Go to https://app.roboflow.com and create a project.")
    print()
    print("  3. Upload images from data/roboflow_upload/")
    print()
    print("  4. Label two classes:")
    print("       - motorcycle")
    print("       - person")
    print()
    print("  5. Export the dataset in YOLOv8 format (zip).")
    print()
    print("  6. Unzip and place it at:  data/yolo_dataset/")
    print("     The folder structure should look like:")
    print("       data/yolo_dataset/")
    print("         data.yaml")
    print("         train/")
    print("           images/")
    print("           labels/")
    print("         valid/")
    print("           images/")
    print("           labels/")
    print()
    print("  7. Re-run this script:")
    print("     python scripts/train_yolo.py \\")
    print("         --data data/yolo_dataset/data.yaml")
    print()
    print("=" * 60)


def main():
    args = parse_args()

    # Check if dataset exists
    if not os.path.isfile(args.data):
        print(f"[ERROR] Dataset config not found: {args.data}")
        print_labelling_guide()
        sys.exit(0)

    print("=" * 60)
    print("YOLO TRAINING")
    print("=" * 60)
    print(f"  Dataset  : {args.data}")
    print(f"  Model    : {args.model}")
    print(f"  Epochs   : {args.epochs}")
    print(f"  Image sz : {args.imgsz}")
    print(f"  Batch    : {args.batch}")
    print("=" * 60)

    # Import here so the script can print help without ultralytics
    from ultralytics import YOLO

    model = YOLO(args.model)

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project="runs/train",
        name="smart_parking",
        exist_ok=True,
        verbose=True,
    )

    # Copy best.pt to models/
    best_pt = os.path.join("runs", "train", "smart_parking", "weights", "best.pt")
    if os.path.isfile(best_pt):
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        shutil.copy2(best_pt, args.output)
        print(f"\n[DONE] Best model copied to: {args.output}")
    else:
        print(f"[WARN] best.pt not found at {best_pt}. Check training output.")

    print("[DONE] Training complete.")


if __name__ == "__main__":
    main()
