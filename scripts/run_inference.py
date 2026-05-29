"""
run_inference.py
================
CLI entry point for the full parking-CV inference pipeline.

Usage:
    python scripts/run_inference.py --config backend/config.yaml
"""

import argparse
import os
import sys

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from backend.core.utils import load_config          # noqa: E402
from backend.core.video_processor import VideoProcessor  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the smart-parking CV inference pipeline."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="backend/config.yaml",
        help="Path to the YAML config file (default: backend/config.yaml).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(_PROJECT_ROOT, config_path)

    config = load_config(config_path)
    processor = VideoProcessor(config)
    processor.process()


if __name__ == "__main__":
    main()
