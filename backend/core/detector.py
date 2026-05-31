"""
detector.py — YOLOv8 object detection wrapper.
===============================================
Loads either a custom-trained model (``models/best.pt``) or the
pretrained YOLOv8n COCO checkpoint.  Only person (class 0) and
motorcycle (class 3) detections are returned.
"""

import os
from typing import Any, Dict, List

import numpy as np
from ultralytics import YOLO

from backend.core.utils import bbox_bottom_center


class YOLODetector:
    """Thin wrapper around Ultralytics YOLO for detection + tracking."""

    def __init__(self, config: Dict[str, Any]) -> None:
        model_path: str = config.get("model_path", "models/best.pt")
        pretrained: str = config.get("pretrained_model", "yolov8n.pt")

        if os.path.isfile(model_path):
            print(f"[INFO] Loading custom model: {model_path}")
            self.model = YOLO(model_path)
        else:
            print(f"[WARN] Custom model not found at {model_path}. "
                  f"Falling back to pretrained: {pretrained}")
            self.model = YOLO(pretrained)

        self.conf = float(config.get("confidence_threshold", 0.15))
        self.iou = float(config.get("iou_threshold", 0.7))
        self.imgsz = int(config.get("inference_imgsz", 640))

        # COCO class IDs we care about
        classes_cfg = config.get("classes", {})
        self.target_classes = [
            int(classes_cfg.get("person", 0)),
            int(classes_cfg.get("motorcycle", 3)),
        ]

        # Friendly names keyed by class id
        self._class_names: Dict[int, str] = {
            int(classes_cfg.get("person", 0)): "person",
            int(classes_cfg.get("motorcycle", 3)): "motorcycle",
        }

        print(f"[INFO] Detector ready  |  conf={self.conf}  "
              f"iou={self.iou}  imgsz={self.imgsz}  "
              f"classes={self.target_classes}")

    # ── Plain detection (no tracking) ────────────────────────────────────

    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Run inference on a single frame.

        Returns a list of dicts, each containing::

            {
                "class_id": int,
                "class_name": str,
                "confidence": float,
                "bbox": [x1, y1, x2, y2],   # pixel coords
                "track_id": None,
            }
        """
        results = self.model.predict(
            frame,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            classes=self.target_classes,
            verbose=False,
        )
        return self._parse_results(results)

    # ── Detection + tracking (ByteTrack via Ultralytics) ─────────────────

    def track(self, frame: np.ndarray) -> List[Dict]:
        """
        Run detection **and** tracking (``model.track``).

        Same return format as :meth:`detect` but ``track_id`` is
        populated for tracked objects.
        """
        results = self.model.track(
            frame,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            classes=self.target_classes,
            persist=True,       # keep state across calls
            tracker="backend/custom_botsort.yaml", # Robust tracking for occlusion with 10s buffer
            verbose=False,
        )
        return self._parse_results(results)

    # ── Internal ─────────────────────────────────────────────────────────

    def _parse_results(self, results) -> List[Dict]:
        """Convert Ultralytics Results → list[dict]."""
        detections: List[Dict] = []
        if not results or len(results) == 0:
            return detections

        boxes = results[0].boxes
        keypoints = results[0].keypoints if hasattr(results[0], 'keypoints') else None
        
        if boxes is None:
            return detections

        for i, box in enumerate(boxes):
            cls_id = int(box.cls.item())
            bbox_coords = box.xyxy[0].tolist()
            
            # Keypoints extraction (YOLO-Pose fallback)
            ground_point = None
            if keypoints is not None and keypoints.data is not None and len(keypoints.data) > i:
                obj_kpts = keypoints.data[i]
                # Filter keypoints with confidence > 0.5
                valid_kpts = obj_kpts[obj_kpts[:, 2] > 0.5]
                if len(valid_kpts) > 0:
                    # Lowest point in image (Max Y) is typically the wheel/foot
                    lowest_pt = valid_kpts[valid_kpts[:, 1].argmax()]
                    ground_point = (float(lowest_pt[0]), float(lowest_pt[1]))
            
            # Fallback to bbox bottom center if no keypoints found
            if ground_point is None:
                ground_point = bbox_bottom_center(bbox_coords)

            det = {
                "class_id": cls_id,
                "class_name": self._class_names.get(cls_id, str(cls_id)),
                "confidence": float(box.conf.item()),
                "bbox": bbox_coords,  # [x1, y1, x2, y2]
                "track_id": int(box.id.item()) if box.id is not None else None,
                "ground_point": ground_point,
            }
            detections.append(det)

        return detections
