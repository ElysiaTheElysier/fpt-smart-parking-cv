# PROJECT STATUS — FPT Smart Parking CV

> Last updated: 2026-05-29

## Phase 1: Foundation ✅ DONE

| File | Status | Description |
|------|--------|-------------|
| `requirements.txt` | ✅ | 8 core dependencies (opencv, numpy, pandas, pyyaml, ultralytics, streamlit, matplotlib, tqdm) |
| `backend/config.yaml` | ✅ | Full pipeline config: paths, thresholds, class IDs, BEV/ROI/gap settings |
| `backend/__init__.py` | ✅ | Python package init |
| `backend/core/__init__.py` | ✅ | Python subpackage init |
| `scripts/extract_frames.py` | ✅ | CLI to extract frames at configurable FPS with video metadata logging |
| `scripts/export_roboflow_frames.py` | ✅ | CLI to export max N frames for Roboflow, optional ROI crop |

## Phase 2: BEV & ROI Calibration ✅ DONE

| File | Status | Description |
|------|--------|-------------|
| `backend/core/bev.py` | ✅ | Perspective transform: load/save points, warp frame/points, draw overlays |
| `scripts/select_bev_points.py` | ✅ | Interactive GUI to click 4 BEV ground-plane corners (r/s/q keys) |
| `scripts/select_roi.py` | ✅ | Interactive GUI to draw ROI polygon around parking area (r/s/q keys) |

## Phase 3: Backend Core Modules + Inference ✅ DONE

| File | Status | Description |
|------|--------|-------------|
| `backend/core/utils.py` | ✅ | Config loader, ensure_dir, point_in_polygon, bbox_bottom_center, draw helpers, colour palette |
| `backend/core/detector.py` | ✅ | YOLO wrapper: auto-loads best.pt or yolov8s.pt, detect() + track(), imgsz support |
| `backend/core/tracker.py` | ✅ | Fallback IoU tracker (greedy matching, max_age eviction) for when ByteTrack is unavailable |
| `backend/core/gap_measurement.py` | ✅ | GapAnalyzer: min/max gap band-pass, max_display cap, draw_labels/draw_distance flags |
| `backend/core/parking_rules.py` | ✅ | Gated by `enabled` flag; ROI alone no longer triggers violations; requires dedicated lane_polygon |
| `backend/core/video_processor.py` | ✅ | ROI-filter detections BEFORE counting/drawing/gaps; demo mode flags; outline-only ROI |
| `scripts/run_inference.py` | ✅ | CLI entry point: `python scripts/run_inference.py --config backend/config.yaml` |

## Hotfix: Demo Mode Cleanup ✅ DONE

Applied to make output clean for demo presentation:

| Change | Detail |
|--------|--------|
| `config.yaml` | `demo_mode`, `roi_only`, `draw_outside_roi`, `parking_rules_enabled`, `draw_roi_fill`, `draw_gap_labels`, `draw_gap_distance`, `min/max_gap_pixels`, `max_display_gaps` |
| `video_processor.py` | `_filter_by_roi()` filters detections before counting/drawing; ROI outline-only mode; HUD shows "(ROI)" suffix |
| `parking_rules.py` | `enabled` master switch; ROI ≠ violation boundary; requires `lane_polygon` |
| `gap_measurement.py` | Band-pass filter `[min_gap, max_gap]`; `max_display` cap (3); `draw_labels`/`draw_distance` flags |

## Phase 4: Scripts ⏳ PENDING

- `scripts/evaluate.py`
- `scripts/train_yolo.py`

## Phase 4: Frontend ⏳ PENDING

- `frontend/streamlit_app.py`

## Phase 5: Documentation ⏳ PENDING

- `README.md` (rewrite)
