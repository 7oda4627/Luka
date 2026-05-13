"""
temporal.py — Temporal Consistency Layer  [NEW MODULE v3]

WHY THIS EXISTS:
  The counter FSM previously transitioned on a SINGLE frame satisfying a
  threshold (e.g. knee_angle < 90° for one frame → immediately go DOWN).
  This caused:
    • Random noise spikes → false rep transitions
    • Partial reps counted (angle briefly crosses threshold then recovers)
    • Fast flailing motions counted as reps

HOW IT WORKS:
  TemporalValidator maintains a rolling buffer of the last N angle readings
  per joint. On every inference frame the buffer is updated. Counter and
  FeedbackEngine query it before making decisions.

KEY METHODS:
  add(angles)                     — call every inference frame
  direction(joint)                — "falling" | "rising" | "stable" | "unknown"
  amplitude(joint)                — peak-to-peak range in buffer (degrees)
  is_moving(joint, min_delta)     — angle moved at least min_delta in buffer
  is_stable(joint, max_var)       — variance below max_var (good for Plank)
  consecutive_below(joint, thr)   — how many consecutive recent frames below thr
  consecutive_above(joint, thr)   — how many consecutive recent frames above thr

USAGE IN counter.py:
  # Only transition DOWN after knee has been below 90° for 3+ frames
  if tv.consecutive_below("knee_avg", 90.0) >= 3:
      self.state = DOWN

  # Only count rep if total ROM >= 60°
  if tv.amplitude("knee_avg") >= 60.0:
      self.reps += 1
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, Optional
import numpy as np


# ── Configuration ─────────────────────────────────────────────────────────────

# How many inference frames to buffer (at ~15 infer fps → 8 frames ≈ 0.5 s)
_BUFFER_LEN = 8

# Minimum linear regression slope to call a direction "falling" or "rising"
_DIR_SLOPE_THRESHOLD = 1.5   # degrees per frame


# ═══════════════════════════════════════════════════════════════════════════════

class TemporalValidator:
    """
    Buffers the last N frames of joint angles and exposes queries for
    direction, amplitude, stability, and consecutive-frame conditions.

    One instance per session; call reset() when switching exercises.
    """

    def __init__(self, buffer_len: int = _BUFFER_LEN) -> None:
        self._n = buffer_len
        # joint_name → deque of float (degrees, newest at right)
        self._history: Dict[str, Deque[float]] = {}

    # ── Feed data ─────────────────────────────────────────────────────────────

    def add(self, angles: Dict[str, float]) -> None:
        """
        Push the current frame's angles into the buffer.
        Must be called once per processed inference frame.
        Zero values (missing keypoints) are NOT added — they do not
        contribute to history so stale good data is used for queries.
        """
        for name, val in angles.items():
            if val <= 0.0:
                continue   # skip missing-keypoint sentinel
            if name not in self._history:
                self._history[name] = deque(maxlen=self._n)
            self._history[name].append(float(val))

    def reset(self) -> None:
        """Clear all history (call on exercise switch)."""
        self._history.clear()

    # ── Queries ───────────────────────────────────────────────────────────────

    def direction(self, joint: str) -> str:
        """
        Return the dominant direction of the angle over the buffer.

        Uses linear regression slope over the stored values.

        Returns
        -------
        "falling"  — angle consistently decreasing (e.g. knee bending)
        "rising"   — angle consistently increasing (e.g. knee straightening)
        "stable"   — nearly flat motion
        "unknown"  — insufficient data
        """
        buf = self._get(joint)
        if buf is None or len(buf) < 3:
            return "unknown"
        arr = np.array(buf, dtype=np.float64)
        x   = np.arange(len(arr), dtype=np.float64)
        # slope via least-squares
        slope = float(np.polyfit(x, arr, 1)[0])
        if slope < -_DIR_SLOPE_THRESHOLD:
            return "falling"
        if slope > _DIR_SLOPE_THRESHOLD:
            return "rising"
        return "stable"

    def amplitude(self, joint: str) -> float:
        """
        Peak-to-peak range of motion in the buffer (degrees).
        Returns 0.0 if insufficient data.
        """
        buf = self._get(joint)
        if buf is None or len(buf) < 2:
            return 0.0
        arr = np.array(buf, dtype=np.float64)
        return float(arr.max() - arr.min())

    def latest(self, joint: str) -> Optional[float]:
        """Most recent value in the buffer, or None."""
        buf = self._get(joint)
        return buf[-1] if buf else None

    def mean(self, joint: str) -> Optional[float]:
        """Mean of buffered values, or None if empty."""
        buf = self._get(joint)
        if not buf:
            return None
        return float(np.mean(buf))

    def variance(self, joint: str) -> float:
        """Variance of buffered values (0.0 if insufficient data)."""
        buf = self._get(joint)
        if buf is None or len(buf) < 2:
            return 0.0
        return float(np.var(buf))

    def is_moving(self, joint: str, min_delta: float = 15.0) -> bool:
        """True if the angle has moved at least min_delta degrees in the buffer."""
        return self.amplitude(joint) >= min_delta

    def is_stable(self, joint: str, max_variance: float = 8.0) -> bool:
        """True if angle is nearly constant (good for Plank hold detection)."""
        return self.variance(joint) <= max_variance

    def consecutive_below(self, joint: str, threshold: float) -> int:
        """
        Count how many of the most recent consecutive frames are below threshold.
        Counts backwards from the newest frame.
        """
        buf = self._get(joint)
        if not buf:
            return 0
        count = 0
        for val in reversed(buf):
            if val < threshold:
                count += 1
            else:
                break
        return count

    def consecutive_above(self, joint: str, threshold: float) -> int:
        """
        Count how many of the most recent consecutive frames are above threshold.
        Counts backwards from the newest frame.
        """
        buf = self._get(joint)
        if not buf:
            return 0
        count = 0
        for val in reversed(buf):
            if val > threshold:
                count += 1
            else:
                break
        return count

    def peak_below(self, joint: str, threshold: float) -> float:
        """
        The minimum value seen in the buffer while below threshold.
        Returns threshold if never below.
        """
        buf = self._get(joint)
        if not buf:
            return threshold
        below = [v for v in buf if v < threshold]
        return min(below) if below else threshold

    def peak_above(self, joint: str, threshold: float) -> float:
        """
        The maximum value seen in the buffer while above threshold.
        Returns threshold if never above.
        """
        buf = self._get(joint)
        if not buf:
            return threshold
        above = [v for v in buf if v > threshold]
        return max(above) if above else threshold

    # ── Internal ──────────────────────────────────────────────────────────────

    def _get(self, joint: str) -> Optional[Deque[float]]:
        buf = self._history.get(joint)
        return buf if buf else None
