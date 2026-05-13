"""
counter.py — Rep Counting & State Machine
State machine per exercise:
  SQUAT         : knee_avg < 90° for 3f → DOWN ; > 160° for 3f + amp≥60° → UP+rep
  PUSHUP        : elbow_avg < 90° for 3f → DOWN ; > 155° for 3f + amp≥55° → UP+rep
  JUMPING JACK  : arm_avg > 140° for 3f → UP   ; < 50° for 3f → DOWN+rep
  HIGH KNEES    : knee.y < hip.y-10px for 3f → UP ; lowered for 2f → DOWN+rep
  PLANK         : body flat (hip 155-205°) → accumulate hold time
  PULLUP        : elbow < 70° for 3f → UP ; > 155° for 3f + amp≥70° → DOWN+rep
  SITUP         : back < 105° for 3f → DOWN ; > 140° for 3f + amp≥40° → UP+rep
  LUNGE         : min_knee < 95° for 3f → DOWN ; avg_knee > 155° for 3f → UP+rep
  MTN CLIMBER   : active hip_angle < 70° for 3f → UP+rep
  BURPEE        : back < 115° for 3f → DOWN ; > 155° for 3f → UP+rep
"""

from __future__ import annotations

import time
from typing import Dict, Optional

import numpy as np

from exercises import Exercise, REGISTRY
from temporal import TemporalValidator


# ── FSM state labels ──────────────────────────────────────────────────────────

class State:
    READY = "ready"
    UP    = "up"
    DOWN  = "down"


# ── Temporal thresholds ───────────────────────────────────────────────────────

# Consecutive inference frames a condition must hold before transition fires
MIN_CONSEC = 3

# Minimum total range-of-motion (degrees) required for a rep to count
MIN_AMP: Dict[Exercise, float] = {
    Exercise.SQUAT:            55.0,   # knee must travel at least 55°
    Exercise.PUSHUP:           55.0,   # elbow must travel at least 55°
    Exercise.JUMPING_JACK:     80.0,   # arm must travel at least 80°
    Exercise.HIGH_KNEES:        0.0,   # pixel-based, skip angle amp check
    Exercise.PLANK:             0.0,   # timed, no rep amplitude
    Exercise.PULLUP:            70.0,
    Exercise.SITUP:             40.0,
    Exercise.LUNGE:             50.0,
    Exercise.MOUNTAIN_CLIMBER:  0.0,   # speed-based, lighter threshold
    Exercise.BURPEE:            40.0,
}


# ═══════════════════════════════════════════════════════════════════════════════

class RepCounter:
    """
    Strict FSM rep counter with temporal validation.

    Parameters
    ----------
    exercise : Exercise
    """

    def __init__(self, exercise: Exercise) -> None:
        self.exercise  = exercise
        self.reps: int = 0
        self.state: str = State.READY
        self.hold_seconds: float = 0.0

        self._thresholds  = REGISTRY[exercise]
        self._plank_start: Optional[float] = None
        self._rep_blocked = False
        self._active_knee = "left"

        # Per-rep angle tracking for amplitude validation
        self._rep_angle_min: float = 999.0
        self._rep_angle_max: float = 0.0

        # Consecutive-frame counter for pending transitions
        # Key: transition label (str) → consecutive frames condition was True
        self._consec: Dict[str, int] = {}

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def last_feedback(self) -> str:
        return ""

    # ── Public API ─────────────────────────────────────────────────────────────

    def update(
        self,
        angles:       Dict[str, float],
        keypoints:    Optional[np.ndarray] = None,
        is_rep_valid: bool = True,
        temporal:     Optional[TemporalValidator] = None,
    ) -> None:
        """
        Parameters
        ----------
        angles       : from angles.get_joint_angles()
        keypoints    : (17,2) for pixel-based checks
        is_rep_valid : False → rep blocked by FeedbackEngine
        temporal     : TemporalValidator for direction/amplitude queries
        """
        if not angles:
            return

        dispatch = {
            Exercise.SQUAT:            self._squat,
            Exercise.PUSHUP:           self._pushup,
            Exercise.JUMPING_JACK:     self._jumping_jack,
            Exercise.HIGH_KNEES:       self._high_knees,
            Exercise.PLANK:            self._plank,
            Exercise.PULLUP:           self._pullup,
            Exercise.SITUP:            self._situp,
            Exercise.LUNGE:            self._lunge,
            Exercise.MOUNTAIN_CLIMBER: self._mountain_climber,
            Exercise.BURPEE:           self._burpee,
        }
        fn = dispatch.get(self.exercise)
        if fn:
            fn(angles, keypoints, is_rep_valid, temporal)

    def reset(self) -> None:
        self.reps         = 0
        self.state        = State.READY
        self.hold_seconds = 0.0
        self._plank_start = None
        self._rep_blocked = False
        self._active_knee = "left"
        self._rep_angle_min = 999.0
        self._rep_angle_max = 0.0
        self._consec.clear()

    # ── Consecutive-frame helper ───────────────────────────────────────────────

    def _cond(self, key: str, condition: bool, required: int = MIN_CONSEC) -> bool:
        """
        Returns True only after `condition` has been True for `required`
        consecutive frames. Resets counter if condition is False.

        This is the core false-positive prevention mechanism.
        """
        if condition:
            self._consec[key] = self._consec.get(key, 0) + 1
        else:
            self._consec[key] = 0
        return self._consec.get(key, 0) >= required

    def _reset_consec(self, *keys: str) -> None:
        for k in keys:
            self._consec[k] = 0

    # ── Amplitude check helper ────────────────────────────────────────────────

    def _track_angle(self, angle: float) -> None:
        """Update per-rep min/max angle for amplitude validation."""
        if angle > 0:
            self._rep_angle_min = min(self._rep_angle_min, angle)
            self._rep_angle_max = max(self._rep_angle_max, angle)

    def _amp_ok(self, exercise: Exercise) -> bool:
        """True if the observed ROM meets the minimum for this exercise."""
        required = MIN_AMP.get(exercise, 0.0)
        if required <= 0.0:
            return True
        return (self._rep_angle_max - self._rep_angle_min) >= required

    def _reset_rep_tracking(self) -> None:
        self._rep_angle_min = 999.0
        self._rep_angle_max = 0.0

    # ── Count helper ───────────────────────────────────────────────────────────

    def _try_count(self, valid: bool) -> None:
        """
        Attempt to count a rep. Requires:
          1. is_rep_valid=True (no critical form error)
          2. rep was not blocked when it started (DOWN entry)
          3. amplitude is sufficient
        """
        if valid and not self._rep_blocked and self._amp_ok(self.exercise):
            self.reps += 1
        self._rep_blocked = False
        self._reset_rep_tracking()

    # ── Exercise state machines ────────────────────────────────────────────────

    def _squat(self, angles, kp, valid, tv) -> None:
        """
        DOWN: knee_avg < 90° for MIN_CONSEC frames
        UP  : knee_avg > 160° for MIN_CONSEC frames + amplitude >= 55°
        """
        knee = _avg_nz(angles.get("Left Knee", 0), angles.get("Right Knee", 0))
        if knee == 0:
            return
        self._track_angle(knee)

        if self.state in (State.READY, State.UP):
            if self._cond("squat_down", knee < 90.0):
                self.state = State.DOWN
                self._rep_blocked = not valid
                self._reset_consec("squat_up")
        elif self.state == State.DOWN:
            if self._cond("squat_up", knee > 160.0):
                self.state = State.UP
                self._try_count(valid)
                self._reset_consec("squat_down")

    def _pushup(self, angles, kp, valid, tv) -> None:
        """
        DOWN: elbow_avg < 90° for MIN_CONSEC frames
        UP  : elbow_avg > 155° for MIN_CONSEC frames + amplitude >= 55°
        """
        elbow = _avg_nz(angles.get("Left Elbow", 0), angles.get("Right Elbow", 0))
        if elbow == 0:
            return
        self._track_angle(elbow)

        if self.state in (State.READY, State.UP):
            if self._cond("pu_down", elbow < 90.0):
                self.state = State.DOWN
                self._rep_blocked = not valid
                self._reset_consec("pu_up")
        elif self.state == State.DOWN:
            if self._cond("pu_up", elbow > 155.0):
                self.state = State.UP
                self._try_count(valid)
                self._reset_consec("pu_down")

    def _jumping_jack(self, angles, kp, valid, tv) -> None:
        """
        UP  : arm_avg > 140° for MIN_CONSEC frames
        DOWN: arm_avg < 50° for MIN_CONSEC frames → rep counted
        """
        arm = _avg_nz(angles.get("L Arm Raise", 0), angles.get("R Arm Raise", 0))
        if arm == 0:
            return
        self._track_angle(arm)

        if self.state in (State.READY, State.DOWN):
            if self._cond("jj_up", arm > 140.0):
                self.state = State.UP
                self._rep_blocked = not valid
                self._reset_consec("jj_down")
        elif self.state == State.UP:
            if self._cond("jj_down", arm < 50.0):
                self.state = State.DOWN
                self._try_count(valid)
                self._reset_consec("jj_up")

    def _high_knees(self, angles, kp, valid, tv) -> None:
        """
        Pixel-based: knee.y < hip.y - 10px for 3 frames → UP
        Knee lowered for 2 frames → DOWN + rep (alternating)
        Requires 10px clearance (was 5px) to reduce false triggers.
        """
        if kp is None:
            return

        if self._active_knee == "left":
            knee_y, hip_y = kp[13, 1], kp[11, 1]
        else:
            knee_y, hip_y = kp[14, 1], kp[12, 1]

        if knee_y == 0 or hip_y == 0:
            return

        raised = float(knee_y) < float(hip_y) - 10.0  # 10px clearance (was 5)

        if self.state in (State.READY, State.DOWN):
            if self._cond("hk_up", raised):
                self.state = State.UP
                self._rep_blocked = not valid
                self._reset_consec("hk_down")
        elif self.state == State.UP:
            if self._cond("hk_down", not raised, required=2):
                self.state = State.DOWN
                if valid and not self._rep_blocked:
                    self.reps += 1
                self._rep_blocked = False
                self._reset_consec("hk_up")
                self._active_knee = "right" if self._active_knee == "left" else "left"

    def _plank(self, angles, kp, valid, tv) -> None:
        """
        Timed hold. Body flat = shoulder→hip→knee ≈ 155–205°.
        Requires stable reading (no single-frame spikes) via tv.
        """
        hip = _avg_nz(angles.get("Left Hip", 0), angles.get("Right Hip", 0))
        if hip == 0:
            return

        holding = 155.0 < hip < 205.0

        if self.state == State.READY and holding:
            self.state        = State.DOWN
            self._plank_start = time.time()
        elif self.state == State.DOWN:
            if holding and self._plank_start:
                self.hold_seconds = time.time() - self._plank_start
            else:
                self.state        = State.READY
                self._plank_start = None

    def _pullup(self, angles, kp, valid, tv) -> None:
        """
        Inverted: hanging = elbow > 155° (DOWN start)
        Chin up  = elbow < 70° for MIN_CONSEC frames → UP
        Return   = elbow > 155° for MIN_CONSEC frames + amp >= 70° → rep
        """
        elbow = _avg_nz(angles.get("Left Elbow", 0), angles.get("Right Elbow", 0))
        if elbow == 0:
            return
        self._track_angle(elbow)

        if self.state in (State.READY, State.DOWN):
            if self._cond("pu2_up", elbow < 70.0):
                self.state = State.UP
                self._rep_blocked = not valid
                self._reset_consec("pu2_down")
        elif self.state == State.UP:
            if self._cond("pu2_down", elbow > 155.0):
                self.state = State.DOWN
                self._try_count(valid)
                self._reset_consec("pu2_up")

    def _situp(self, angles, kp, valid, tv) -> None:
        """
        DOWN: back_angle < 105° for MIN_CONSEC frames (lying)
        UP  : back_angle > 140° for MIN_CONSEC frames + amp >= 40° → rep
        """
        back = angles.get("Back Angle", angles.get("Back", 0.0))
        if back == 0:
            return
        self._track_angle(back)

        if self.state in (State.READY, State.UP):
            if self._cond("su_down", back < 105.0):
                self.state = State.DOWN
                self._rep_blocked = not valid
                self._reset_consec("su_up")
        elif self.state == State.DOWN:
            if self._cond("su_up", back > 140.0):
                self.state = State.UP
                self._try_count(valid)
                self._reset_consec("su_down")

    def _lunge(self, angles, kp, valid, tv) -> None:
        """
        DOWN: min_knee < 95° for MIN_CONSEC frames
        UP  : avg_knee > 155° for MIN_CONSEC frames + amp >= 50° → rep
        """
        knee_l = angles.get("Left Knee",  0.0)
        knee_r = angles.get("Right Knee", 0.0)
        min_k  = _min_nz(knee_l, knee_r)
        avg_k  = _avg_nz(knee_l, knee_r)
        if min_k == 0:
            return
        self._track_angle(min_k)

        if self.state in (State.READY, State.UP):
            if self._cond("lg_down", min_k < 95.0):
                self.state = State.DOWN
                self._rep_blocked = not valid
                self._reset_consec("lg_up")
        elif self.state == State.DOWN:
            if self._cond("lg_up", avg_k > 155.0):
                self.state = State.UP
                self._try_count(valid)
                self._reset_consec("lg_down")

    def _mountain_climber(self, angles, kp, valid, tv) -> None:
        """
        Alternating knee drive.
        UP  : active hip_angle < 70° for MIN_CONSEC frames → rep
        DOWN: active hip_angle > 130° for MIN_CONSEC frames → ready for next
        """
        key = "L Hip Angle" if self._active_knee == "left" else "R Hip Angle"
        hip_a = angles.get(key, 0.0)
        if hip_a == 0:
            return

        if self.state in (State.READY, State.DOWN):
            if self._cond("mc_up", hip_a < 70.0):
                self.state = State.UP
                self._rep_blocked = not valid
                self._reset_consec("mc_down")
        elif self.state == State.UP:
            if self._cond("mc_down", hip_a > 130.0):
                self.state = State.DOWN
                if valid and not self._rep_blocked:
                    self.reps += 1
                self._rep_blocked = False
                self._reset_consec("mc_up")
                self._active_knee = "right" if self._active_knee == "left" else "left"

    def _burpee(self, angles, kp, valid, tv) -> None:
        """
        DOWN: back_angle < 115° for MIN_CONSEC frames (floor)
        UP  : back_angle > 155° for MIN_CONSEC frames + amp >= 40° → rep
        """
        back = angles.get("Back Angle", angles.get("Back", 0.0))
        if back == 0:
            return
        self._track_angle(back)

        if self.state in (State.READY, State.UP):
            if self._cond("bur_down", back < 115.0):
                self.state = State.DOWN
                self._rep_blocked = not valid
                self._reset_consec("bur_up")
        elif self.state == State.DOWN:
            if self._cond("bur_up", back > 155.0):
                self.state = State.UP
                self._try_count(valid)
                self._reset_consec("bur_down")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _avg_nz(*values: float) -> float:
    valid = [v for v in values if v > 0.0]
    return sum(valid) / len(valid) if valid else 0.0

def _min_nz(*values: float) -> float:
    valid = [v for v in values if v > 0.0]
    return min(valid) if valid else 0.0
