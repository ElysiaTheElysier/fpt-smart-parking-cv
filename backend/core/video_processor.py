"""
video_processor.py — Full inference pipeline orchestrator.
==========================================================
Reads a video, runs YOLO detection + tracking frame-by-frame,
applies BEV transform, computes gaps, checks parking rules,
draws all overlays, and saves:

* Annotated output video (``data/outputs/annotated_video.mp4``)
* ``metrics.csv`` — per-frame statistics
* ``gap_log.csv`` — per-frame gap details
* At least one debug screenshot
"""

import os
import sys
import time
import math
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import imageio
import pandas as pd

from backend.core.bev import (
    compute_perspective_transform,
    draw_polygon as bev_draw_polygon,
    load_points,
)
from backend.core.detector import YOLODetector
from backend.core.gap_measurement import GapAnalyzer, draw_gaps
from backend.core.parking_rules import check_violations, draw_violations
from backend.core.tracker import SimpleIOUTracker
from backend.core.utils import (
    COLOR_MOTORCYCLE,
    COLOR_PERSON,
    COLOR_ROI,
    COLOR_WHITE,
    bbox_bottom_center,
    draw_text_with_bg,
    ensure_dir,
    load_config,
    point_in_polygon,
)


class VideoProcessor:
    """End-to-end video processing pipeline."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.cfg = config
        self.project_root: str = config.get("_project_root", ".")

        # ── Detector ─────────────────────────────────────────────────
        self.detector = YOLODetector(config)

        # ── Tracker (fallback) ───────────────────────────────────────
        self.fallback_tracker = SimpleIOUTracker(min_iou=0.3, max_age=15)
        self.use_builtin_tracking = True  # try Ultralytics first

        # ── BEV ──────────────────────────────────────────────────────
        self.bev_enabled: bool = config.get("bev_enabled", False)
        self.bev_matrix: Optional[np.ndarray] = None
        self.bev_w: int = int(config.get("bev_output_width", 1000))
        self.bev_h: int = int(config.get("bev_output_height", 700))

        if self.bev_enabled:
            bev_path = os.path.join(
                self.project_root, "data", "calibration", "bev_points.json"
            )
            bev_pts = load_points(bev_path)
            if bev_pts is not None and len(bev_pts) == 4:
                self.bev_matrix = compute_perspective_transform(
                    bev_pts, self.bev_w, self.bev_h
                )
                print("[INFO] BEV transform matrix computed.")
            else:
                print("[WARN] BEV points not found or invalid. BEV disabled.")
                self.bev_enabled = False

        # ── ROI ──────────────────────────────────────────────────────
        self.roi_enabled: bool = config.get("roi_enabled", False)
        self.roi_polygon: Optional[np.ndarray] = None

        if self.roi_enabled:
            roi_path = os.path.join(
                self.project_root, "data", "calibration", "roi_points.json"
            )
            roi_pts = load_points(roi_path)
            if roi_pts is not None and len(roi_pts) >= 3:
                self.roi_polygon = roi_pts
                print(f"[INFO] ROI polygon loaded ({len(roi_pts)} vertices).")
            else:
                print("[WARN] ROI points not found or invalid. ROI disabled.")
                self.roi_enabled = False

        # ── ROI filtering behaviour ──────────────────────────────────
        self.roi_only: bool = config.get("roi_only", False)
        self.draw_outside_roi: bool = config.get("draw_outside_roi", True)

        # ── Demo mode flags ──────────────────────────────────────────
        self.demo_mode: bool = config.get("demo_mode", False)
        self.parking_rules_enabled: bool = config.get("parking_rules_enabled", True)
        self.gap_enabled: bool = config.get("gap_enabled", True)
        self.draw_gap_labels: bool = config.get("draw_gap_labels", True)
        self.draw_gap_distance: bool = config.get("draw_gap_distance", True)
        self.draw_roi_fill: bool = config.get("draw_roi_fill", True)

        # ── Gap analyser ─────────────────────────────────────────────
        self.gap_analyzer = GapAnalyzer(
            gap_threshold=float(config.get("gap_threshold_pixels", 120)),
            min_gap_pixels=config.get("min_gap_pixels"),
            max_gap_pixels=config.get("max_gap_pixels"),
            min_gap_frames=int(config.get("min_gap_frames", 5)),
            smoothing_window=int(config.get("temporal_smoothing_window", 5)),
            max_display=config.get("max_display_gaps"),
        )

        # ── Misc config ─────────────────────────────────────────────
        self.stride: int = int(config.get("inference_stride", 2))
        self.show_person: bool = config.get("show_person", True)
        self.show_tracks: bool = config.get("show_tracks", True)
        self.save_metrics: bool = config.get("save_metrics", True)

    # ==================================================================
    #  ROI filtering
    # ==================================================================

    def _filter_by_roi(self, detections: List[Dict]) -> List[Dict]:
        """
        If ``roi_only`` is True and an ROI polygon is loaded, keep only
        detections whose bottom-center falls inside the ROI polygon.
        """
        if not self.roi_only or self.roi_polygon is None:
            return detections

        filtered: List[Dict] = []
        for det in detections:
            bc = bbox_bottom_center(det["bbox"])
            if point_in_polygon(bc, self.roi_polygon):
                filtered.append(det)
        return filtered

    # ==================================================================
    #  Main entry point
    # ==================================================================

    def process(self) -> None:
        """Run the full pipeline on the configured video."""
        video_path: str = self.cfg["video_path"]
        output_path: str = self.cfg["output_video_path"]

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open video: {video_path}")
            sys.exit(1)

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        max_frames_cfg = self.cfg.get("max_frames", 0)
        if max_frames_cfg > 0:
            total_frames = min(total_frames, max_frames_cfg)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print("=" * 60)
        print("VIDEO PROCESSING PIPELINE")
        print("=" * 60)
        print(f"  Input          : {video_path}")
        print(f"  Resolution     : {width}x{height}")
        print(f"  FPS            : {fps:.2f}")
        print(f"  Total frames   : {total_frames}")
        print(f"  Inference stride: every {self.stride} frame(s)")
        print(f"  BEV enabled    : {self.bev_enabled}")
        print(f"  ROI enabled    : {self.roi_enabled}")
        print(f"  ROI only       : {self.roi_only}")
        print(f"  Demo mode      : {self.demo_mode}")
        print(f"  Gap enabled    : {self.gap_enabled}")
        print(f"  Parking rules  : {self.parking_rules_enabled}")
        print(f"  Output         : {output_path}")
        print("=" * 60)

        # ── Video writer ─────────────────────────────────────────────
        ensure_dir(os.path.dirname(output_path))
        
        # We write to a temporary mp4v file first, because OpenCV on Windows 
        # often fails to write H.264 directly without extra DLLs.
        temp_output_path = output_path.replace(".mp4", "_temp.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            
        out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))

        # ── Metrics accumulators ─────────────────────────────────────
        metrics_rows: List[Dict] = []
        gap_log_rows: List[Dict] = []

        frame_idx = 0
        processed_count = 0
        last_detections: List[Dict] = []
        last_gaps: List[Dict] = []
        debug_screenshot_saved = False

        t_start = time.time()

        while True:
            ret, frame = cap.read()
            if not ret or (max_frames_cfg > 0 and frame_idx >= max_frames_cfg):
                break

            timestamp = frame_idx / fps if fps > 0 else 0.0

            # ── Run inference on selected frames ─────────────────────
            if frame_idx % self.stride == 0:
                raw_detections = self._detect_and_track(frame)

                # Filter to ROI before any downstream processing
                detections = self._filter_by_roi(raw_detections)
                last_detections = detections
                processed_count += 1

                # Gap analysis (detections are already ROI-filtered)
                if self.gap_enabled:
                    gaps = self.gap_analyzer.update(
                        detections,
                        self.bev_matrix,
                        self.roi_polygon,
                        self.bev_enabled,
                    )
                else:
                    gaps = []
                last_gaps = gaps
            else:
                detections = last_detections
                gaps = last_gaps

            # ── Counts (ROI-filtered only) ───────────────────────────
            mc_count = sum(1 for d in detections if d["class_name"] == "motorcycle")
            person_count = sum(1 for d in detections if d["class_name"] == "person")
            available_gaps = sum(1 for g in gaps if g["status"] == "available")

            # ── Parking rule violations ──────────────────────────────
            violations = check_violations(
                detections,
                self.roi_polygon,
                enabled=self.parking_rules_enabled,
            )

            # ── Draw overlays ────────────────────────────────────────
            annotated = frame.copy()

            # ROI polygon — outline only when draw_roi_fill is False
            if self.roi_polygon is not None:
                if self.draw_roi_fill:
                    annotated = bev_draw_polygon(
                        annotated, self.roi_polygon,
                        color=COLOR_ROI, alpha=0.10, line_thickness=2,
                    )
                else:
                    pts = np.array(self.roi_polygon, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(
                        annotated, [pts], isClosed=True,
                        color=COLOR_ROI, thickness=2,
                    )

            # Detection bounding boxes (ROI-filtered)
            self._draw_detections(annotated, detections)

            # Gap markers
            if self.gap_enabled:
                annotated = draw_gaps(
                    annotated, gaps,
                    draw_labels=self.draw_gap_labels,
                    draw_distance=self.draw_gap_distance,
                )

            # Violation markers
            annotated = draw_violations(annotated, violations)

            # HUD
            elapsed = time.time() - t_start
            proc_fps = processed_count / elapsed if elapsed > 0 else 0.0
            self._draw_hud(
                annotated, frame_idx, timestamp, proc_fps,
                mc_count, person_count, available_gaps,
            )

            # ── BEV minimap ──────────────────────────────────────────
            if self.bev_enabled and self.cfg.get("draw_bev_minimap", True) and self.bev_matrix is not None:
                # Warp current frame to BEV
                bev_view = cv2.warpPerspective(
                    frame, self.bev_matrix, (self.bev_w, self.bev_h)
                )

                # Draw motorcycle points in BEV
                mc_points_orig = [
                    bbox_bottom_center(d["bbox"]) for d in detections 
                    if d["class_name"] == "motorcycle"
                ]
                if mc_points_orig:
                    pts_array = np.array(mc_points_orig, dtype=np.float32).reshape(-1, 1, 2)
                    pts_bev = cv2.perspectiveTransform(pts_array, self.bev_matrix).reshape(-1, 2)
                    for x, y in pts_bev:
                        cv2.circle(bev_view, (int(x), int(y)), 8, COLOR_MOTORCYCLE, -1)

                # Resize to minimap (width=400)
                mini_w = 400
                mini_h = int((self.bev_h / self.bev_w) * mini_w)
                bev_mini = cv2.resize(bev_view, (mini_w, mini_h))

                # Label & border
                cv2.putText(bev_mini, "BEV Mini-map", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.rectangle(bev_mini, (0, 0), (mini_w-1, mini_h-1), (255, 255, 255), 2)

                # Place at bottom-right
                h_main, w_main = annotated.shape[:2]
                y_off = h_main - mini_h - 20
                x_off = w_main - mini_w - 20
                annotated[y_off:y_off+mini_h, x_off:x_off+mini_w] = bev_mini
            out.write(annotated)

            # ── Save one debug screenshot ────────────────────────────
            if not debug_screenshot_saved and processed_count >= 5:
                ss_dir = ensure_dir(
                    os.path.join(self.project_root, "data", "outputs", "screenshots")
                )
                ss_path = os.path.join(ss_dir, "pipeline_debug.jpg")
                cv2.imwrite(ss_path, annotated)
                print(f"[INFO] Debug screenshot saved: {ss_path}")
                debug_screenshot_saved = True

            # ── Metrics row ──────────────────────────────────────────
            if self.save_metrics:
                metrics_rows.append({
                    "frame_idx": frame_idx,
                    "timestamp": round(timestamp, 3),
                    "fps": round(proc_fps, 2),
                    "motorcycle_count": mc_count,
                    "person_count": person_count,
                    "available_gap_count": available_gaps,
                })

                for gap in gaps:
                    gap_log_rows.append({
                        "frame_idx": frame_idx,
                        "timestamp": round(timestamp, 3),
                        "gap_id": gap["gap_id"],
                        "gap_distance_pixels": round(gap["distance"], 1),
                        "status": gap["status"],
                    })

            # ── Progress ─────────────────────────────────────────────
            if frame_idx % 500 == 0:
                pct = frame_idx / total_frames * 100 if total_frames > 0 else 0
                print(f"[PROGRESS] Frame {frame_idx}/{total_frames} "
                      f"({pct:.1f}%)  MC={mc_count}  Gaps={available_gaps}")

            frame_idx += 1

        cap.release()
        out.release()

        # ── Convert to H.264 for web compatibility ───────────────────
        print(f"[INFO] Converting video to H.264 for web playback...")
        try:
            reader = imageio.get_reader(temp_output_path)
            writer = imageio.get_writer(output_path, fps=fps, codec='libx264')
            for frame in reader:
                writer.append_data(frame)
            writer.close()
            reader.close()
            os.remove(temp_output_path)
            print(f"[DONE] Annotated video saved to: {output_path}")
        except Exception as e:
            print(f"[ERROR] Video conversion failed: {e}")
            print(f"[INFO] Fallback: Please view the raw video at {temp_output_path}")

        # ── Save metrics ─────────────────────────────────────────────
        if self.save_metrics:
            self._save_csv(metrics_rows, gap_log_rows)

    # ==================================================================
    #  Detection + tracking
    # ==================================================================

    def _detect_and_track(self, frame: np.ndarray) -> List[Dict]:
        """
        Try Ultralytics built-in tracking first.  If it throws or
        returns no track IDs, fall back to plain detection + IoU
        tracker.
        """
        if self.use_builtin_tracking:
            try:
                detections = self.detector.track(frame)
                has_ids = any(d["track_id"] is not None for d in detections)
                if has_ids:
                    return detections
            except Exception as exc:
                print(f"[WARN] Built-in tracking failed ({exc}). "
                      f"Switching to fallback IoU tracker.")
                self.use_builtin_tracking = False

        # Fallback: plain detection + simple IoU tracker
        detections = self.detector.detect(frame)
        detections = self.fallback_tracker.update(detections)
        return detections

    # ==================================================================
    #  Drawing helpers
    # ==================================================================

    def _draw_detections(
        self, frame: np.ndarray, detections: List[Dict]
    ) -> None:
        """Draw bounding boxes, labels, confidence, and track IDs."""
        for det in detections:
            cls_name = det["class_name"]
            if cls_name == "person" and not self.show_person:
                continue

            bbox = det["bbox"]
            x1, y1, x2, y2 = [int(v) for v in bbox]
            conf = det["confidence"]
            track_id = det.get("track_id")

            color = COLOR_MOTORCYCLE if cls_name == "motorcycle" else COLOR_PERSON
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Label string
            parts = [f"{cls_name} {conf:.2f}"]
            if track_id is not None and self.show_tracks:
                parts.append(f"ID:{track_id}")
            label = " ".join(parts)

            draw_text_with_bg(
                frame, label, (x1, y1 - 4),
                font_scale=0.5, color=color,
            )

    def _draw_hud(
        self,
        frame: np.ndarray,
        frame_idx: int,
        timestamp: float,
        proc_fps: float,
        mc_count: int,
        person_count: int,
        gap_count: int,
    ) -> None:
        """Draw a heads-up display panel at the top of the frame."""
        lines = [
            f"Frame: {frame_idx}  |  Time: {timestamp:.1f}s  |  FPS: {proc_fps:.1f}",
            f"Motorcycles (ROI): {mc_count}  |  Persons (ROI): {person_count}",
            f"Available gaps: {gap_count}",
        ]

        y = 30
        for line in lines:
            draw_text_with_bg(frame, line, (10, y), font_scale=0.6,
                              color=COLOR_WHITE)
            y += 30

    # ==================================================================
    #  CSV export
    # ==================================================================

    def _save_csv(
        self,
        metrics_rows: List[Dict],
        gap_log_rows: List[Dict],
    ) -> None:
        """Save metrics.csv and gap_log.csv."""
        out_dir = ensure_dir(
            os.path.join(self.project_root, "data", "outputs")
        )

        if metrics_rows:
            metrics_path = os.path.join(out_dir, "metrics.csv")
            pd.DataFrame(metrics_rows).to_csv(metrics_path, index=False)
            print(f"[INFO] Metrics saved to: {metrics_path}")

        if gap_log_rows:
            gap_path = os.path.join(out_dir, "gap_log.csv")
            pd.DataFrame(gap_log_rows).to_csv(gap_path, index=False)
            print(f"[INFO] Gap log saved to: {gap_path}")
        else:
            print("[INFO] No gap data recorded (gap_log.csv not created).")
