# PROJECT STATUS — FPT Smart Parking CV

> Last updated: 2026-05-29

## ✅ ALL PHASES COMPLETE

## Phase 1: Foundation ✅ DONE

| File | Status | Description |
|------|--------|-------------|
| `requirements.txt` | ✅ | 8 core dependencies |
| `backend/config.yaml` | ✅ | Full pipeline config with demo mode flags |
| `backend/__init__.py` | ✅ | Package init |
| `backend/core/__init__.py` | ✅ | Subpackage init |
| `scripts/extract_frames.py` | ✅ | CLI frame extraction |
| `scripts/export_roboflow_frames.py` | ✅ | CLI Roboflow export with optional ROI crop |

## Phase 2: BEV & ROI Calibration ✅ DONE

| File | Status | Description |
|------|--------|-------------|
| `backend/core/bev.py` | ✅ | Perspective transform: load/save points, warp frame/points, draw overlays |
| `scripts/select_bev_points.py` | ✅ | Interactive GUI to click 4 BEV corners |
| `scripts/select_roi.py` | ✅ | Interactive GUI to draw ROI polygon |

## Phase 3: Backend Core Modules + Inference ✅ DONE

| File | Status | Description |
|------|--------|-------------|
| `backend/core/utils.py` | ✅ | Config loader, geometry helpers, drawing helpers |
| `backend/core/detector.py` | ✅ | YOLO wrapper: best.pt / yolov8s.pt fallback, imgsz support |
| `backend/core/tracker.py` | ✅ | Fallback IoU tracker |
| `backend/core/gap_measurement.py` | ✅ | GapAnalyzer: min/max band-pass, max_display cap, temporal smoothing |
| `backend/core/parking_rules.py` | ✅ | Gated by `enabled` flag; requires lane_polygon for violations |
| `backend/core/video_processor.py` | ✅ | Full pipeline with ROI filtering, demo mode |
| `scripts/run_inference.py` | ✅ | CLI entry point |

## Phase 4: Scripts + Frontend + Docs ✅ DONE

| File | Status | Description |
|------|--------|-------------|
| `scripts/evaluate.py` | ✅ | Summary stats, detection_summary.csv, optional matplotlib plot |
| `scripts/train_yolo.py` | ✅ | Train YOLO or print Roboflow labelling guide |
| `frontend/streamlit_app.py` | ✅ | Dashboard: config, run inference, video player, metrics, charts |
| `README.md` | ✅ | Full Vietnamese documentation |
