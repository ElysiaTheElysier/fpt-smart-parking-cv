"""
tracker.py — Fallback IoU-based tracker.
=========================================
The primary tracking strategy uses Ultralytics' built-in ByteTrack
integration (see ``detector.py :: YOLODetector.track``).

This module provides a **simple IoU-based tracker** as a fallback for
situations where ``model.track()`` fails or is unavailable.  The
algorithm is intentionally minimal:

1. For each new detection, compute IoU with every active track's last
   bounding box.
2. Greedily assign the detection to the track with the highest IoU
   (above ``min_iou``).
3. Unmatched detections start new tracks.
4. Tracks not updated for ``max_age`` frames are dropped.

This is **not** production-quality — but it is sufficient for a
student-project demo to reduce flickering.
"""

from typing import Any, Dict, List

import numpy as np


def _iou(box_a: List[float], box_b: List[float]) -> float:
    """Compute IoU between two ``[x1, y1, x2, y2]`` boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter == 0:
        return 0.0

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class SimpleIOUTracker:
    """
    Greedy IoU tracker — keeps a dict of active tracks keyed by
    ``track_id``.

    Parameters
    ----------
    min_iou : float
        Minimum IoU to consider a match (default 0.3).
    max_age : int
        Frames a track can survive without being updated (default 10).
    """

    def __init__(self, min_iou: float = 0.3, max_age: int = 10) -> None:
        self.min_iou = min_iou
        self.max_age = max_age
        self._next_id: int = 1
        self._tracks: Dict[int, Dict[str, Any]] = {}

    def update(self, detections: List[Dict]) -> List[Dict]:
        """
        Match *detections* (from ``YOLODetector.detect``) to existing
        tracks and assign ``track_id`` to each detection.

        Returns the same list with ``track_id`` populated.
        """
        # Increment age of all existing tracks
        for tid in list(self._tracks):
            self._tracks[tid]["age"] += 1

        matched_track_ids: set = set()
        matched_det_indices: set = set()

        # ---- Greedy matching by IoU ----
        # Build IoU matrix
        det_boxes = [d["bbox"] for d in detections]
        track_ids = list(self._tracks.keys())

        if track_ids and det_boxes:
            iou_matrix = np.zeros((len(track_ids), len(det_boxes)))
            for ti, tid in enumerate(track_ids):
                for di, dbox in enumerate(det_boxes):
                    iou_matrix[ti, di] = _iou(self._tracks[tid]["bbox"], dbox)

            # Greedy assignment — pick the best match iteratively
            while True:
                if iou_matrix.size == 0:
                    break
                best_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                best_iou = iou_matrix[best_idx]
                if best_iou < self.min_iou:
                    break

                ti, di = best_idx
                tid = track_ids[ti]

                detections[di]["track_id"] = tid
                self._tracks[tid]["bbox"] = det_boxes[di]
                self._tracks[tid]["age"] = 0

                matched_track_ids.add(tid)
                matched_det_indices.add(di)

                # Zero out the matched row and column
                iou_matrix[ti, :] = 0
                iou_matrix[:, di] = 0

        # ---- Create new tracks for unmatched detections ----
        for di, det in enumerate(detections):
            if di not in matched_det_indices:
                tid = self._next_id
                self._next_id += 1
                det["track_id"] = tid
                self._tracks[tid] = {
                    "bbox": det["bbox"],
                    "age": 0,
                }

        # ---- Remove stale tracks ----
        stale = [tid for tid, t in self._tracks.items() if t["age"] > self.max_age]
        for tid in stale:
            del self._tracks[tid]

        return detections
