'''
angles.py — Joint Angle Calculations 
'''


from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from exercises import Exercise

Point = Tuple[float, float]

# ── COCO-17 indices ───────────────────────────────────────────────────────────
_L_SHOULDER = 5;  _R_SHOULDER = 6
_L_ELBOW    = 7;  _R_ELBOW    = 8
_L_WRIST    = 9;  _R_WRIST    = 10
_L_HIP      = 11; _R_HIP      = 12
_L_KNEE     = 13; _R_KNEE     = 14
_L_ANKLE    = 15; _R_ANKLE    = 16

# Maximum plausible angle change between consecutive inference frames
# Larger changes are treated as detection noise and filtered out
MAX_JUMP_DEG = 35.0

# Per-session previous angle cache (updated every call to get_joint_angles)
_prev_angles: Dict[str, float] = {}


# ═══════════════════════════════════════════════════════════════════════════════
#  Core geometry
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_angle(p1: Point, p2: Point, p3: Point) -> float:
    """
    Angle at vertex p2 via cosine rule.
    Returns 0.0 if any point is the (0,0) missing-keypoint sentinel.
    """
    for p in (p1, p2, p3):
        if p[0] == 0.0 and p[1] == 0.0:
            return 0.0

    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]], dtype=np.float64)
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]], dtype=np.float64)

    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0

    cos_a = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return math.degrees(math.acos(cos_a))


def angle_between_vectors(v1: Sequence[float], v2: Sequence[float]) -> float:
    a = np.array(v1, dtype=np.float64)
    b = np.array(v2, dtype=np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    cos = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
    return math.degrees(math.acos(cos))


# ═══════════════════════════════════════════════════════════════════════════════
#  Confidence filtering
# ═══════════════════════════════════════════════════════════════════════════════

def filter_keypoints_by_confidence(
    keypoints: np.ndarray,
    scores:    Optional[np.ndarray],
    threshold: float = 0.25,
) -> np.ndarray:
    """
    Zero out keypoints whose confidence score is below `threshold`.
    If scores is None (pose backend doesn't provide them), returns kp unchanged.

    Parameters
    ----------
    keypoints : (17, 2) float array
    scores    : (17,) float array of confidence scores, or None
    threshold : confidence below this → set kp to (0, 0)

    Returns
    -------
    (17, 2) float array with low-confidence joints zeroed
    """
    if scores is None:
        return keypoints

    kp = keypoints.copy()
    mask = scores < threshold         # (17,) bool
    kp[mask] = 0.0
    return kp


# ═══════════════════════════════════════════════════════════════════════════════
#  View angle detection
# ═══════════════════════════════════════════════════════════════════════════════

def detect_view_angle(kp: np.ndarray) -> str:
    """
    Estimate whether the camera is viewing the subject from the front or side.

    Heuristic: in a front-view frame, both shoulders and both hips are visible
    and roughly symmetric. In a side view, one side is occluded (zero).

    Returns
    -------
    "front"   — both shoulders and hips visible
    "side_l"  — only left side visible (right occluded)
    "side_r"  — only right side visible (left occluded)
    "unknown" — insufficient keypoints
    """
    ls_ok = _visible(kp, _L_SHOULDER)
    rs_ok = _visible(kp, _R_SHOULDER)
    lh_ok = _visible(kp, _L_HIP)
    rh_ok = _visible(kp, _R_HIP)

    if ls_ok and rs_ok and lh_ok and rh_ok:
        # Both sides visible — check horizontal symmetry
        # In side view, left and right shoulders are very close together
        ls_x = float(kp[_L_SHOULDER, 0])
        rs_x = float(kp[_R_SHOULDER, 0])
        width = abs(rs_x - ls_x)

        # Estimate body height as proxy for scale
        head_y  = float(kp[0, 1]) if _visible(kp, 0) else 0.0
        ankle_y = _avg_nz(float(kp[_L_ANKLE, 1]), float(kp[_R_ANKLE, 1]))
        body_h  = max(ankle_y - head_y, 100.0)

        ratio = width / body_h
        if ratio < 0.08:   # tightened: only flag side-view when shoulders nearly overlap
            # Shoulders very close horizontally → likely side view
            return "side_l" if lh_ok else "side_r"
        return "front"

    if (ls_ok or lh_ok) and not (rs_ok or rh_ok):
        return "side_l"
    if (rs_ok or rh_ok) and not (ls_ok or lh_ok):
        return "side_r"

    return "unknown"


# ═══════════════════════════════════════════════════════════════════════════════
#  Jump-spike guard
# ═══════════════════════════════════════════════════════════════════════════════

def _guarded(name: str, angle: float) -> float:
    """
    Reject implausible single-frame angle jumps.

    If the new angle differs from the previous by more than MAX_JUMP_DEG,
    return the previous value (treat new reading as noise).
    Otherwise accept the new value and update the cache.
    """
    global _prev_angles
    prev = _prev_angles.get(name, angle)   # first frame: accept as-is
    if angle == 0.0:
        # Missing keypoint — don't overwrite cache, return 0
        return 0.0
    if abs(angle - prev) > MAX_JUMP_DEG:
        # Spike detected — return previous stable value
        return prev
    _prev_angles[name] = angle
    return angle


def reset_angle_cache() -> None:
    """Call when switching exercises to clear the jump-guard cache."""
    global _prev_angles
    _prev_angles.clear()


# ═══════════════════════════════════════════════════════════════════════════════
#  Exercise-specific angle extraction
# ═══════════════════════════════════════════════════════════════════════════════

def get_joint_angles(
    keypoints: np.ndarray,
    exercise:  Exercise,
    scores:    Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute joint angles for the given exercise, with:
      • Confidence filtering (zero out low-confidence joints if scores given)
      • Side-view adaptation (use best-visible joint chain)
      • Jump-spike rejection (MAX_JUMP_DEG guard per named angle)

    Parameters
    ----------
    keypoints : (17, 2) COCO-17 pixel coords; (0,0) = missing
    exercise  : Exercise enum value
    scores    : (17,) confidence scores from pose estimator, or None
    """
    # Apply confidence filter first
    kp = filter_keypoints_by_confidence(keypoints, scores)

    # Detect view angle for adaptive logic
    view = detect_view_angle(kp)

    angles: Dict[str, float] = {}

    def xy(i: int) -> Point:
        return (float(kp[i, 0]), float(kp[i, 1]))

    def ga(name: str, p1: int, p2: int, p3: int) -> float:
        """Compute + guard a named angle."""
        return _guarded(name, calculate_angle(xy(p1), xy(p2), xy(p3)))

    def bilateral(name_l: str, name_r: str,
                  p1l: int, p2l: int, p3l: int,
                  p1r: int, p2r: int, p3r: int,
                  out_l: str, out_r: str,
                  ) -> Tuple[float, float]:
        """
        Compute both sides; in side view, return whichever side is visible.
        Returns (left_angle, right_angle).
        """
        al = ga(name_l, p1l, p2l, p3l)
        ar = ga(name_r, p1r, p2r, p3r)

        if view == "side_l":
            # Right side occluded — mirror left to right for display
            return al, al
        if view == "side_r":
            return ar, ar
        return al, ar

    # ── Squat ─────────────────────────────────────────────────────────────────
    if exercise == Exercise.SQUAT:
        lk, rk = bilateral(
            "sq_knee_l", "sq_knee_r",
            _L_HIP, _L_KNEE, _L_ANKLE,
            _R_HIP, _R_KNEE, _R_ANKLE,
            "Left Knee", "Right Knee",
        )
        angles["Left Knee"]  = lk
        angles["Right Knee"] = rk
        angles["Back"]       = _guarded("sq_back", _back_angle(kp))

    # ── Push-up ───────────────────────────────────────────────────────────────
    elif exercise == Exercise.PUSHUP:
        le, re = bilateral(
            "pu_elbow_l", "pu_elbow_r",
            _L_SHOULDER, _L_ELBOW, _L_WRIST,
            _R_SHOULDER, _R_ELBOW, _R_WRIST,
            "Left Elbow", "Right Elbow",
        )
        lh, rh = bilateral(
            "pu_hip_l", "pu_hip_r",
            _L_SHOULDER, _L_HIP, _L_KNEE,
            _R_SHOULDER, _R_HIP, _R_KNEE,
            "Left Hip", "Right Hip",
        )
        angles["Left Elbow"]  = le
        angles["Right Elbow"] = re
        angles["Left Hip"]    = lh
        angles["Right Hip"]   = rh

    # ── Jumping Jack ──────────────────────────────────────────────────────────
    elif exercise == Exercise.JUMPING_JACK:
        angles["L Arm Raise"] = ga("jj_arm_l", _L_HIP, _L_SHOULDER, _L_WRIST)
        angles["R Arm Raise"] = ga("jj_arm_r", _R_HIP, _R_SHOULDER, _R_WRIST)

    # ── High Knees ────────────────────────────────────────────────────────────
    elif exercise == Exercise.HIGH_KNEES:
        angles["L Knee Height"] = ga("hk_l", _L_SHOULDER, _L_HIP, _L_KNEE)
        angles["R Knee Height"] = ga("hk_r", _R_SHOULDER, _R_HIP, _R_KNEE)

    # ── Plank ─────────────────────────────────────────────────────────────────
    elif exercise == Exercise.PLANK:
        lh = ga("pl_hip_l", _L_SHOULDER, _L_HIP, _L_KNEE)
        rh = ga("pl_hip_r", _R_SHOULDER, _R_HIP, _R_KNEE)
        if view == "side_l":
            lh = rh = lh
        elif view == "side_r":
            lh = rh = rh
        angles["Left Hip"]   = lh
        angles["Right Hip"]  = rh
        angles["Body Align"] = _avg_nz(lh, rh)

    # ── Pull-up ───────────────────────────────────────────────────────────────
    elif exercise == Exercise.PULLUP:
        le, re = bilateral(
            "pup_elbow_l", "pup_elbow_r",
            _L_SHOULDER, _L_ELBOW, _L_WRIST,
            _R_SHOULDER, _R_ELBOW, _R_WRIST,
            "Left Elbow", "Right Elbow",
        )
        angles["Left Elbow"]  = le
        angles["Right Elbow"] = re

    # ── Sit-up ────────────────────────────────────────────────────────────────
    elif exercise == Exercise.SITUP:
        angles["Back Angle"] = _guarded("su_back", _back_angle(kp))

    # ── Lunge ─────────────────────────────────────────────────────────────────
    elif exercise == Exercise.LUNGE:
        lk, rk = bilateral(
            "lg_knee_l", "lg_knee_r",
            _L_HIP, _L_KNEE, _L_ANKLE,
            _R_HIP, _R_KNEE, _R_ANKLE,
            "Left Knee", "Right Knee",
        )
        angles["Left Knee"]  = lk
        angles["Right Knee"] = rk
        angles["Back"]       = _guarded("lg_back", _back_angle(kp))

    # ── Mountain Climber ──────────────────────────────────────────────────────
    elif exercise == Exercise.MOUNTAIN_CLIMBER:
        angles["L Hip Angle"] = ga("mc_hip_l", _L_SHOULDER, _L_HIP, _L_KNEE)
        angles["R Hip Angle"] = ga("mc_hip_r", _R_SHOULDER, _R_HIP, _R_KNEE)

    # ── Burpee ────────────────────────────────────────────────────────────────
    elif exercise == Exercise.BURPEE:
        lk, rk = bilateral(
            "bur_knee_l", "bur_knee_r",
            _L_HIP, _L_KNEE, _L_ANKLE,
            _R_HIP, _R_KNEE, _R_ANKLE,
            "Left Knee", "Right Knee",
        )
        angles["Left Knee"]  = lk
        angles["Right Knee"] = rk
        angles["Back Angle"] = _guarded("bur_back", _back_angle(kp))

    return angles


# ═══════════════════════════════════════════════════════════════════════════════
#  Shared helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _back_angle(kp: np.ndarray) -> float:
    """
    Torso inclination relative to vertical.
    ~180° = upright, ~90° = horizontal.

    Side-view aware: uses whichever shoulder/hip points are visible.
    """
    ls, rs = kp[_L_SHOULDER], kp[_R_SHOULDER]
    lh, rh = kp[_L_HIP],      kp[_R_HIP]

    s_pts = [p for p in (ls, rs) if not (p[0] == 0 and p[1] == 0)]
    h_pts = [p for p in (lh, rh) if not (p[0] == 0 and p[1] == 0)]
    if not s_pts or not h_pts:
        return 0.0

    smid = np.mean(s_pts, axis=0)
    hmid = np.mean(h_pts, axis=0)

    return calculate_angle(
        (float(smid[0]), float(smid[1])),
        (float(hmid[0]), float(hmid[1])),
        (float(hmid[0]), float(hmid[1]) + 100.0),
    )


def _visible(kp: np.ndarray, idx: int) -> bool:
    """True if keypoint at idx is non-zero (i.e. detected)."""
    return not (kp[idx, 0] == 0.0 and kp[idx, 1] == 0.0)


def _avg_nz(*values: float) -> float:
    valid = [v for v in values if v > 0.0]
    return sum(valid) / len(valid) if valid else 0.0
