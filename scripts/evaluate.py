"""
evaluate.py — Evaluate pipeline output metrics.
================================================
Reads metrics.csv and gap_log.csv produced by the inference pipeline
and prints summary statistics.

Usage:
    python scripts/evaluate.py \
        --metrics data/outputs/metrics.csv \
        --gaps data/outputs/gap_log.csv
"""

import argparse
import os
import sys

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate inference pipeline output metrics."
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="data/outputs/metrics.csv",
        help="Path to metrics.csv.",
    )
    parser.add_argument(
        "--gaps",
        type=str,
        default="data/outputs/gap_log.csv",
        help="Path to gap_log.csv.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/outputs/detection_summary.csv",
        help="Path to save summary CSV.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate a simple matplotlib plot.",
    )
    return parser.parse_args()


def evaluate_metrics(metrics_path: str) -> dict:
    """Load metrics.csv and compute summary statistics."""
    if not os.path.isfile(metrics_path):
        print(f"[ERROR] Metrics file not found: {metrics_path}")
        sys.exit(1)

    df = pd.read_csv(metrics_path)

    summary = {
        "total_frames": len(df),
        "avg_fps": round(df["fps"].mean(), 2),
        "avg_motorcycle_count": round(df["motorcycle_count"].mean(), 2),
        "max_motorcycle_count": int(df["motorcycle_count"].max()),
        "avg_person_count": round(df["person_count"].mean(), 2),
        "avg_available_gaps": round(df["available_gap_count"].mean(), 2),
        "max_available_gaps": int(df["available_gap_count"].max()),
    }

    return summary


def evaluate_gaps(gaps_path: str) -> dict:
    """Load gap_log.csv and compute gap statistics."""
    if not os.path.isfile(gaps_path):
        print(f"[WARN] Gap log not found: {gaps_path}. Skipping gap analysis.")
        return {
            "total_gap_records": 0,
            "unique_gap_ids": 0,
            "gap_status_changes": 0,
            "avg_gap_distance_px": 0,
        }

    df = pd.read_csv(gaps_path)

    if df.empty:
        return {
            "total_gap_records": 0,
            "unique_gap_ids": 0,
            "gap_status_changes": 0,
            "avg_gap_distance_px": 0,
        }

    # Count status transitions per gap_id
    status_changes = 0
    for _, group in df.groupby("gap_id"):
        statuses = group["status"].tolist()
        for i in range(1, len(statuses)):
            if statuses[i] != statuses[i - 1]:
                status_changes += 1

    summary = {
        "total_gap_records": len(df),
        "unique_gap_ids": int(df["gap_id"].nunique()),
        "gap_status_changes": status_changes,
        "avg_gap_distance_px": round(df["gap_distance_pixels"].mean(), 1),
    }

    return summary


def save_summary(summary: dict, output_path: str) -> None:
    """Save summary as a single-row CSV."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.DataFrame([summary])
    df.to_csv(output_path, index=False)
    print(f"[INFO] Summary saved to: {output_path}")


def plot_metrics(metrics_path: str) -> None:
    """Generate a simple plot of motorcycle count and gaps over time."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not installed. Skipping plot.")
        return

    df = pd.read_csv(metrics_path)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    axes[0].plot(df["timestamp"], df["motorcycle_count"], label="Motorcycles", color="green")
    axes[0].plot(df["timestamp"], df["person_count"], label="Persons", color="blue", alpha=0.6)
    axes[0].set_ylabel("Count")
    axes[0].legend()
    axes[0].set_title("Detection Counts Over Time")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(df["timestamp"], df["available_gap_count"], label="Available Gaps", color="orange")
    axes[1].set_ylabel("Gaps")
    axes[1].set_xlabel("Time (s)")
    axes[1].legend()
    axes[1].set_title("Available Gaps Over Time")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    plot_path = os.path.join(os.path.dirname(metrics_path), "metrics_plot.png")
    plt.savefig(plot_path, dpi=150)
    print(f"[INFO] Plot saved to: {plot_path}")
    plt.close()


def main():
    args = parse_args()

    print("=" * 60)
    print("EVALUATION")
    print("=" * 60)

    # Metrics summary
    metrics_summary = evaluate_metrics(args.metrics)
    print("\n--- Metrics Summary ---")
    print(f"  Total frames processed : {metrics_summary['total_frames']}")
    print(f"  Average FPS            : {metrics_summary['avg_fps']}")
    print(f"  Avg motorcycle count   : {metrics_summary['avg_motorcycle_count']}")
    print(f"  Max motorcycle count   : {metrics_summary['max_motorcycle_count']}")
    print(f"  Avg person count       : {metrics_summary['avg_person_count']}")
    print(f"  Avg available gaps     : {metrics_summary['avg_available_gaps']}")
    print(f"  Max available gaps     : {metrics_summary['max_available_gaps']}")

    # Gap summary
    gap_summary = evaluate_gaps(args.gaps)
    print("\n--- Gap Summary ---")
    print(f"  Total gap records      : {gap_summary['total_gap_records']}")
    print(f"  Unique gap IDs         : {gap_summary['unique_gap_ids']}")
    print(f"  Gap status changes     : {gap_summary['gap_status_changes']}")
    print(f"  Avg gap distance (px)  : {gap_summary['avg_gap_distance_px']}")

    # Merge and save
    combined = {**metrics_summary, **gap_summary}
    save_summary(combined, args.output)

    # Optional plot
    if args.plot:
        plot_metrics(args.metrics)

    print("\n" + "=" * 60)
    print("[DONE] Evaluation complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
