"""
detector.py — YOLOv8 Person Detection Module

Detects 'person' class bounding boxes in a frame using Ultralytics YOLOv8.
Returns a list of (x1, y1, x2, y2) bounding boxes.
"""

from __future__ import annotations

import numpy as np
import cv2
from typing import List, Tuple

# Lazy import so startup error messages are clear
try:
    from ultralytics import YOLO
except ImportError as e:
    raise ImportError(
        "ultralytics is not installed. Run: pip install ultralytics"
    ) from e

# COCO class index for 'person'
PERSON_CLASS_ID = 0

BBox = Tuple[int, int, int, int]  # (x1, y1, x2, y2)


class PersonDetector:
    """
    Wraps YOLOv8 to detect persons in a given frame.

    Parameters
    ----------
    model_name : str
        YOLOv8 model variant. 'yolov8n.pt' is fastest; 'yolov8s.pt' is a
        good balance between speed and accuracy.
    conf_threshold : float
        Minimum confidence for a detection to be kept.
    iou_threshold : float
        IoU threshold used in NMS.
    input_size : int
        Inference resolution (square). Lower = faster; higher = more accurate.
    device : str | None
        'cpu', 'cuda', 'mps', or None (auto-detect).
    """

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        input_size: int = 640,
        device: str | None = None,
    ) -> None:
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.input_size = input_size

        print(f"[Detector] Loading model: {model_name}")
        self.model = YOLO(model_name)

        # Determine compute device
        if device is None:
            import torch
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        print(f"[Detector] Using device : {self.device}")

    # ── Public API ────────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> List[BBox]:
        """
        Run person detection on a single BGR frame.

        Parameters
        ----------
        frame : np.ndarray
            BGR image from OpenCV.

        Returns
        -------
        List[BBox]
            List of bounding boxes as (x1, y1, x2, y2) in pixel coordinates,
            sorted by area descending (largest person first).
        """
        results = self.model.predict(
            source=frame,
            imgsz=self.input_size,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=[PERSON_CLASS_ID],
            device=self.device,
            verbose=False,
        )

        bboxes: List[BBox] = []
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                bboxes.append((int(x1), int(y1), int(x2), int(y2)))

        # Sort by area (largest first) so the primary subject is bboxes[0]
        bboxes.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
        return bboxes

    # ── Utility ───────────────────────────────────────────────────────────────

    @staticmethod
    def bbox_area(bbox: BBox) -> int:
        """Return pixel area of a bounding box."""
        x1, y1, x2, y2 = bbox
        return max(0, x2 - x1) * max(0, y2 - y1)

    @staticmethod
    def pad_bbox(
        bbox: BBox,
        frame_shape: Tuple[int, int],
        pad_ratio: float = 0.1,
    ) -> BBox:
        """
        Expand a bounding box by `pad_ratio` on each side, clamped to the
        frame boundaries. Useful before passing to a pose estimator that
        needs some context around the person.

        Parameters
        ----------
        bbox : BBox
        frame_shape : (height, width)
        pad_ratio : float
            Fraction of the bbox side length to add as padding.

        Returns
        -------
        BBox
        """
        h, w = frame_shape[:2]
        x1, y1, x2, y2 = bbox
        bw, bh = x2 - x1, y2 - y1
        px, py = int(bw * pad_ratio), int(bh * pad_ratio)
        return (
            max(0, x1 - px),
            max(0, y1 - py),
            min(w, x2 + px),
            min(h, y2 + py),
        )
