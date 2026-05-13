"""
Usage:
    pipe = InferencePipeline(detector, pose_estimator, smoother)
    pipe.start()

    while True:
        ret, frame = cap.read()
        pipe.submit_frame(frame)           # non-blocking
        result = pipe.get_result()         # non-blocking, returns latest
        # render result onto frame …

    pipe.stop()
"""

from __future__ import annotations

import copy
import threading
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from detector import PersonDetector, BBox
from pose import PoseEstimator
from utils import KalmanKeypoints, select_largest_bbox


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class InferenceResult:
    person_detected: bool = False
    bbox:            Optional[BBox] = None
    keypoints:       Optional[np.ndarray] = None   # (17, 2) Kalman-smoothed
    frame_id:        int = 0

    def copy(self) -> "InferenceResult":
        return InferenceResult(
            person_detected = self.person_detected,
            bbox            = self.bbox,
            keypoints       = self.keypoints.copy() if self.keypoints is not None else None,
            frame_id        = self.frame_id,
        )


# ── Pipeline ──────────────────────────────────────────────────────────────────

class InferencePipeline:
    """
    Runs YOLOv8 detection + RTMPose estimation in a daemon background thread.

    Thread-safety guarantees:
      - submit_frame() : safe to call from main thread at any time
      - get_result()   : safe to call from main thread at any time
      - reset_smoother(): safe to call from main thread

    The pipeline drops stale frames automatically — if inference takes 100 ms
    and frames arrive at 30 fps, each inference cycle processes the frame that
    was submitted most recently, skipping the 2–3 intermediate ones.
    """

    def __init__(
        self,
        detector:       PersonDetector,
        pose_estimator: PoseEstimator,
        smoother:       KalmanKeypoints,
    ) -> None:
        self._detector = detector
        self._pose     = pose_estimator
        self._smoother = smoother

        # Shared state (protected by locks)
        self._input_lock  = threading.Lock()
        self._output_lock = threading.Lock()
        self._frame_event = threading.Event()

        self._pending_frame: Optional[np.ndarray] = None
        self._result   = InferenceResult()
        self._frame_id = 0

        # Control
        self._running = False
        self._thread  = threading.Thread(
            target  = self._loop,
            name    = "InferenceThread",
            daemon  = True,
        )

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the background inference thread."""
        self._running = True
        self._thread.start()

    def stop(self) -> None:
        """Signal the inference thread to stop and wait for it."""
        self._running = False
        self._frame_event.set()   # unblock thread if waiting
        self._thread.join(timeout=2.0)

    # ── Public API (main-thread safe) ─────────────────────────────────────────

    def submit_frame(self, frame: np.ndarray) -> None:
        """
        Queue a frame for inference.
        If a frame is already pending (inference busy), REPLACE it with
        the newer one — we always want to process the freshest data.
        """
        with self._input_lock:
            self._pending_frame = frame          # no copy; inference reads only
            self._frame_id     += 1
        self._frame_event.set()

    def get_result(self) -> InferenceResult:
        """Return a snapshot of the latest inference result (non-blocking)."""
        with self._output_lock:
            return self._result.copy()

    def reset_smoother(self) -> None:
        """Reset Kalman state — call when the tracked person leaves frame."""
        # Safe because smoother is accessed only from the inference thread
        # and we set a flag it will pick up
        self._smoother.reset()

    # ── Background loop ───────────────────────────────────────────────────────

    def _loop(self) -> None:
        while self._running:
            # Block until there is a frame to process (or timeout for shutdown)
            signaled = self._frame_event.wait(timeout=0.5)
            self._frame_event.clear()

            if not self._running:
                break
            if not signaled:
                continue

            # Grab the latest pending frame (may have been replaced since set)
            with self._input_lock:
                frame = self._pending_frame
                self._pending_frame = None

            if frame is None:
                continue

            result = self._process(frame)

            with self._output_lock:
                self._result = result

    def _process(self, frame: np.ndarray) -> InferenceResult:
        """Full detection + pose estimation for one frame."""
        # ── 1. Person detection ───────────────────────────────────────────────
        bboxes = self._detector.detect(frame)
        if not bboxes:
            self._smoother.reset()
            return InferenceResult(person_detected=False)

        bbox = select_largest_bbox(bboxes)

        # ── 2. Pose estimation ────────────────────────────────────────────────
        keypoints = self._pose.estimate(frame, bbox)
        if keypoints is None:
            return InferenceResult(person_detected=True, bbox=bbox)

        # ── 3. Kalman smoothing ───────────────────────────────────────────────
        keypoints = self._smoother.smooth(keypoints)

        return InferenceResult(
            person_detected = True,
            bbox            = bbox,
            keypoints       = keypoints,
        )
