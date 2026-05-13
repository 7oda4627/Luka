"""
utils.py — Shared Utilities  [UPDATED v2]

Changes vs original:
  + KalmanKeypoints   : per-keypoint 2-D Kalman filter replacing EMA
  + DualFPSCounter    : separate display-FPS and inference-FPS streams
  + draw_hud()        : full HUD renderer extracted from main
  + print_status()    : ANSI-coloured terminal line with rep-blocked flag
  + print_keybind_legend(): startup legend for all exercise shortcuts
  - KeypointSmoother  : kept for backward-compat (superseded by Kalman)
"""

from __future__ import annotations

import time
from collections import deque
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

BBox = Tuple[int, int, int, int]   # (x1, y1, x2, y2)

# ── COCO-17 skeleton ──────────────────────────────────────────────────────────

COCO_SKELETON: List[Tuple[int, int]] = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]
_LIMB_COLOURS: List[Tuple[int, int, int]] = [
    (255, 100, 100), (255, 100, 100),
    (255, 180, 100), (255, 180, 100),
    (100, 255, 100),
    (100, 200, 255), (100, 200, 255),
    (255, 180,  50), (255, 180,  50),
    (160, 100, 255), (160, 100, 255),
    (200, 200,   0),
    (  0, 220, 150), (  0, 220, 150),
    (  0, 160, 255), (  0, 160, 255),
]
_KP_COLOUR     = (0, 255, 220)
_KP_BAD_COLOUR = (0, 80, 255)

# ── ANSI terminal colours ─────────────────────────────────────────────────────

_A = {
    "green":  "\033[92m", "yellow": "\033[93m", "red":   "\033[91m",
    "cyan":   "\033[96m", "white":  "\033[97m", "dim":   "\033[2m",
    "bold":   "\033[1m",  "reset":  "\033[0m",
}


# ═══════════════════════════════════════════════════════════════════════════════
#  FPS Counters
# ═══════════════════════════════════════════════════════════════════════════════

class FPSCounter:
    """Rolling-window FPS (single event stream)."""

    def __init__(self, window: int = 30) -> None:
        self._ts: deque = deque(maxlen=window)
        self.fps: float = 0.0

    def tick(self) -> float:
        now = time.perf_counter()
        self._ts.append(now)
        if len(self._ts) >= 2:
            elapsed = self._ts[-1] - self._ts[0]
            self.fps = (len(self._ts) - 1) / elapsed if elapsed > 0 else 0.0
        return self.fps


class DualFPSCounter:
    """
    Tracks display-render FPS and model-inference FPS independently.

    display_fps: how fast frames are shown on screen / terminal
    infer_fps  : how fast new pose results arrive from the inference thread
    """

    def __init__(self, window: int = 30) -> None:
        self._disp  = FPSCounter(window)
        self._infer = FPSCounter(window)

    def tick_display(self) -> None:
        self._disp.tick()

    def tick_infer(self) -> None:
        self._infer.tick()

    @property
    def display_fps(self) -> float:
        return self._disp.fps

    @property
    def infer_fps(self) -> float:
        return self._infer.fps


# ═══════════════════════════════════════════════════════════════════════════════
#  Keypoint Smoothing
# ═══════════════════════════════════════════════════════════════════════════════

class KeypointSmoother:
    """EMA smoother — kept for backward-compat. Prefer KalmanKeypoints."""

    def __init__(self, window_size: int = 5, alpha: Optional[float] = None) -> None:
        self.alpha = alpha if alpha is not None else 2.0 / (window_size + 1)
        self._smoothed: Optional[np.ndarray] = None

    def smooth(self, keypoints: np.ndarray) -> np.ndarray:
        if self._smoothed is None:
            self._smoothed = keypoints.copy()
            return self._smoothed
        valid = ~((keypoints[:, 0] == 0) & (keypoints[:, 1] == 0))
        updated = self._smoothed.copy()
        updated[valid] = (
            self.alpha * keypoints[valid] + (1.0 - self.alpha) * self._smoothed[valid]
        )
        self._smoothed = updated
        return self._smoothed

    def reset(self) -> None:
        self._smoothed = None


class KalmanKeypoints:
    """
    Per-keypoint 2-D constant-velocity Kalman filter.

    Why better than EMA:
      • Predicts position between detections — handles brief occlusion
      • Adapts uncertainty automatically (R/Q covariances)
      • Velocity model prevents 'rubber-band' lag on fast movements
      • Missing keypoints (0,0 sentinel) handled as "no measurement" frames

    State per keypoint: [x, y, vx, vy]
    Measurement:        [x, y]

    Parameters
    ----------
    n_kp       : number of keypoints (17 for COCO-17)
    proc_noise : process noise — higher = trusts raw detections more
    meas_noise : measurement noise — higher = trusts predictions more
    """

    def __init__(
        self,
        n_kp:       int   = 17,
        proc_noise: float = 1e-2,
        meas_noise: float = 5e-2,
    ) -> None:
        self.n_kp = n_kp
        self._initialized = False

        # Constant-velocity state transition F
        self._F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float64)

        # Measurement extracts (x, y) from state
        self._H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float64)

        self._Q = np.eye(4, dtype=np.float64) * proc_noise   # process noise
        self._R = np.eye(2, dtype=np.float64) * meas_noise   # measurement noise

        self._x = np.zeros((n_kp, 4), dtype=np.float64)
        self._P = np.stack([np.eye(4, dtype=np.float64)] * n_kp)

    # Maximum pixel distance a keypoint may jump in one frame
    # Larger jumps are treated as detection noise and dampened
    _MAX_JUMP_PX: float = 80.0

    def smooth(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Feed a new (17, 2) keypoint array and return Kalman-smoothed (17, 2).

        Jump-rejection (v3):
          If a new measurement is more than _MAX_JUMP_PX pixels from the
          current Kalman prediction, it is treated as a detection artefact.
          The measurement noise R is inflated 10× for that keypoint so the
          filter strongly prefers its own prediction over the noisy reading.
          This prevents single-frame detection errors from snapping keypoints.

        Missing keypoints:
          (0, 0) → treated as no measurement; prediction carried forward.
        """
        kp = keypoints.astype(np.float64)
        missing = (kp[:, 0] == 0) & (kp[:, 1] == 0)

        if not self._initialized:
            self._x[:, :2] = kp
            self._initialized = True
            return keypoints.astype(np.float32)

        out = np.zeros_like(kp)

        for i in range(self.n_kp):
            x_k, P_k = self._x[i], self._P[i]

            # Predict
            x_pred = self._F @ x_k
            P_pred = self._F @ P_k @ self._F.T + self._Q

            if missing[i]:
                # No measurement: carry prediction, grow uncertainty
                self._x[i] = x_pred
                self._P[i] = P_pred * 1.5
                out[i]     = x_pred[:2]
            else:
                # Jump-rejection: check distance from prediction to measurement
                dist = float(np.linalg.norm(kp[i] - x_pred[:2]))
                R_eff = self._R.copy()
                if dist > self._MAX_JUMP_PX:
                    # Inflate measurement noise → filter trusts its prediction
                    R_eff = self._R * 10.0

                # Update with (possibly noise-inflated) R
                y = kp[i] - self._H @ x_pred
                S = self._H @ P_pred @ self._H.T + R_eff
                K = P_pred @ self._H.T @ np.linalg.inv(S)

                self._x[i] = x_pred + K @ y
                self._P[i] = (np.eye(4) - K @ self._H) @ P_pred
                out[i]     = self._x[i, :2]

        return out.astype(np.float32)

    def reset(self) -> None:
        """Reset state — call when person leaves frame."""
        self._x[:] = 0.0
        self._P[:] = np.eye(4)
        self._initialized = False


# ═══════════════════════════════════════════════════════════════════════════════
#  Bounding-box helpers
# ═══════════════════════════════════════════════════════════════════════════════

def select_largest_bbox(bboxes: List[BBox]) -> BBox:
    """Return bbox with the largest pixel area (primary subject)."""
    if not bboxes:
        raise ValueError("bboxes list is empty")
    return max(bboxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))


# ═══════════════════════════════════════════════════════════════════════════════
#  Terminal output
# ═══════════════════════════════════════════════════════════════════════════════

def print_keybind_legend(key_map: Dict[str, object]) -> None:
    """Print keyboard shortcut table once at startup."""
    sep = "─" * 60
    print(f"\n{sep}")
    print("  EXERCISE SHORTCUTS (press key in the OpenCV window):")
    for key, ex in key_map.items():
        print(f"    [{key}]  →  {ex.value.replace('_', ' ').title()}")
    print("    [r]  →  Reset rep counter for current exercise")
    print("    [q]  →  Quit")
    print(f"{sep}\n")


def print_status(
    display_fps:   float,
    infer_fps:     float,
    reps:          int,
    state:         str,
    angles:        Dict[str, float],
    form_feedback: str,
    rep_blocked:   bool,
    exercise,
    hold_seconds:  float = 0.0,
) -> None:
    """
    Overwrite a single terminal line with live trainer status.
    Includes ANSI colour coding:
      green  = good form / rep counted
      yellow = warning
      red    = critical error / rep blocked
    """
    a = _A

    angle_str = "  ".join(
        f"{n}:{v:6.1f}°" for n, v in angles.items() if v > 0
    )

    is_bad = "⚠" in form_feedback or "bad" in form_feedback.lower()
    fb_col = a["red"] if (is_bad or rep_blocked) else (
             a["green"] if "✓" in form_feedback else a["yellow"])

    blocked_tag = f" {a['red']}[REP BLOCKED]{a['reset']}" if rep_blocked else ""

    # Timed vs rep-based display
    from exercises import REGISTRY
    if REGISTRY[exercise].is_timed:
        count_str = f"Hold:{a['green']}{hold_seconds:5.1f}s{a['reset']}"
    else:
        count_str = f"Reps:{a['green']}{reps:3d}{a['reset']}"

    line = (
        f"\r"
        f"{a['dim']}D:{display_fps:4.0f} I:{infer_fps:4.0f}fps{a['reset']}  "
        f"{a['bold']}{a['cyan']}[{exercise.value.upper()}]{a['reset']}  "
        f"{a['bold']}{count_str}  "
        f"State:{a['yellow']}{state.upper():<6s}{a['reset']}  "
        f"{a['dim']}{angle_str}{a['reset']}  "
        f"{fb_col}{form_feedback}{a['reset']}"
        f"{blocked_tag}"
        f"          "
    )
    print(line, end="", flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  OpenCV drawing helpers
# ═══════════════════════════════════════════════════════════════════════════════

def draw_skeleton(
    frame:      np.ndarray,
    keypoints:  np.ndarray,
    form_bad:   bool = False,
    dot_radius: int  = 5,
    line_thick: int  = 2,
) -> None:
    """Render COCO-17 skeleton on frame in-place. form_bad turns dots red."""
    kp_col = _KP_BAD_COLOUR if form_bad else _KP_COLOUR
    for idx, (a, b) in enumerate(COCO_SKELETON):
        pa = (int(keypoints[a, 0]), int(keypoints[a, 1]))
        pb = (int(keypoints[b, 0]), int(keypoints[b, 1]))
        if pa == (0, 0) or pb == (0, 0):
            continue
        cv2.line(frame, pa, pb, _LIMB_COLOURS[idx % len(_LIMB_COLOURS)],
                 line_thick, cv2.LINE_AA)
    for kp in keypoints:
        x, y = int(kp[0]), int(kp[1])
        if x == 0 and y == 0:
            continue
        cv2.circle(frame, (x, y), dot_radius,     kp_col,    -1, cv2.LINE_AA)
        cv2.circle(frame, (x, y), dot_radius + 1, (0, 0, 0),  1, cv2.LINE_AA)


def draw_bbox(
    frame:     np.ndarray,
    bbox:      BBox,
    colour:    Tuple[int, int, int] = (0, 255, 100),
    thickness: int = 2,
    label:     str = "Person",
) -> None:
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, thickness, cv2.LINE_AA)
    if label:
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), colour, -1)
        cv2.putText(frame, label, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)


def draw_hud(
    frame:         np.ndarray,
    display_fps:   float,
    infer_fps:     float,
    reps:          int,
    state:         str,
    angles:        Dict[str, float],
    feedback:      str,
    rep_blocked:   bool,
    exercise,
    hold_seconds:  float = 0.0,
) -> None:
    """
    Draw a semi-transparent HUD panel on the left edge of frame.

    Shows (satisfying all debug/monitoring requirements):
      ✓ Display FPS + Inference FPS
      ✓ Current exercise name
      ✓ Rep count (or hold time for timed exercises)
      ✓ State: UP / DOWN / READY
      ✓ All joint angle values
      ✓ Form feedback (colour-coded by severity)
      ✓ Rep-blocked indicator (red top bar)
      ✓ Keybind hint line
    """
    h = frame.shape[0]
    panel_w = 290

    # Semi-transparent overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, h), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.48, frame, 0.52, 0, frame)

    def put(text: str, y: int, color=(210, 210, 210), scale=0.52, thick=1):
        cv2.putText(frame, text, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

    # FPS row
    put(f"D-FPS:{display_fps:4.0f}   I-FPS:{infer_fps:4.0f}",
        24, color=(80, 230, 80), scale=0.46)

    # Exercise name
    put(exercise.value.replace("_", " ").upper(),
        54, color=(0, 220, 255), scale=0.68, thick=2)

    # Rep / Hold count
    from exercises import REGISTRY
    is_timed = REGISTRY[exercise].is_timed
    if is_timed:
        put(f"Hold: {hold_seconds:.1f} s", 88,
            color=(0, 200, 255), scale=0.72, thick=2)
    else:
        put(f"Reps: {reps}", 88,
            color=(0, 200, 255), scale=0.72, thick=2)

    # State
    s_col = ((0, 255, 160) if state == "up" else
             (0, 160, 255) if state == "down" else (180, 180, 180))
    put(f"State: {state.upper()}", 116, color=s_col, scale=0.55)

    # Divider
    cv2.line(frame, (8, 126), (panel_w - 8, 126), (60, 60, 60), 1)

    # Angles
    y = 144
    for name, val in angles.items():
        if val > 0:
            bar = "▌" * min(int(val / 9), 20)
            put(f"{name[:12]:<12}: {val:6.1f}°  {bar}", y,
                color=(190, 190, 140), scale=0.44)
            y += 17
        if y > h - 85:
            break

    # Divider
    cv2.line(frame, (8, y + 2), (panel_w - 8, y + 2), (60, 60, 60), 1)

    # Feedback (word-wrapped)
    if feedback:
        is_bad = ("⚠" in feedback or "bad" in feedback.lower())
        fb_col = (60, 60, 255) if (is_bad or rep_blocked) else (60, 220, 60)

        words  = feedback.split()
        lines, cur = [], ""
        for w in words:
            if len(cur) + len(w) + 1 <= 26:
                cur = (cur + " " + w).strip()
            else:
                if cur:
                    lines.append(cur)
                cur = w
        if cur:
            lines.append(cur)

        fb_y = h - 32 - len(lines) * 18
        for ln in lines:
            put(ln, fb_y, color=fb_col, scale=0.49, thick=1)
            fb_y += 18

    # Rep-blocked alert bar
    if rep_blocked:
        cv2.rectangle(frame, (0, 0), (panel_w, 4), (0, 0, 230), -1)

    # Keybind hint
    put("[1-0]=switch  [r]=reset  [q]=quit",
        h - 7, color=(65, 65, 65), scale=0.34)


def format_angle_table(angles: Dict[str, float]) -> str:
    lines = ["Joint Angles:"]
    for name, val in angles.items():
        bar = "█" * int(val / 3)
        lines.append(f"  {name:<16s}: {val:6.1f}°  {bar}")
    return "\n".join(lines)
