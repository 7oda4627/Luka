from __future__ import annotations

import time
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from exercises import Exercise, REGISTRY


# ── Priority ──────────────────────────────────────────────────────────────────

class Priority(Enum):
    CRITICAL = 0   # blocks rep; shown after PERSIST_FRAMES consecutive frames
    WARNING  = 1   # shown after PERSIST_FRAMES frames; rep still counts
    TIP      = 2   # coaching; shown after 5 frames; rarely


# Cooldown seconds before same message may show again
_COOLDOWN: Dict[Priority, float] = {
    Priority.CRITICAL: 2.0,
    Priority.WARNING:  4.0,
    Priority.TIP:      8.0,
}

# Issue must be detected this many consecutive frames before showing
PERSIST_FRAMES: Dict[Priority, int] = {
    Priority.CRITICAL: 3,
    Priority.WARNING:  3,
    Priority.TIP:      5,
}


class _Issue:
    __slots__ = ("message", "priority", "blocks_rep")

    def __init__(self, message: str, priority: Priority, blocks_rep: bool = False):
        self.message    = message
        self.priority   = priority
        self.blocks_rep = blocks_rep


# ═══════════════════════════════════════════════════════════════════════════════

class FeedbackEngine:
    """
    Stateful form-feedback engine with persistence gating and cooldowns.

    Persistence gating:
      Each detected issue gets a streak counter in _issue_streak.
      A message is only raised once the streak reaches PERSIST_FRAMES[priority].
      This eliminates single-frame noise triggers.
    """

    def __init__(self, cooldown_scale: float = 1.0):
        self._scale        = cooldown_scale
        self._last_shown:  Dict[str, float] = {}   # msg → last-shown timestamp
        self._issue_streak: Dict[str, int]  = {}   # msg → consecutive frame count
        self._active_msg:   str  = ""
        self._active_blocks: bool = False

    # ── Public API ─────────────────────────────────────────────────────────────

    def evaluate(
        self,
        exercise:  Exercise,
        angles:    Dict[str, float],
        state:     str,
        keypoints: Optional[np.ndarray],
        extra:     Optional[Dict] = None,
    ) -> Tuple[str, bool]:
        """
        Evaluate pose and return (feedback_message, is_rep_invalid).

        Only messages that have persisted for PERSIST_FRAMES consecutive frames
        AND have cleared their cooldown are shown.
        """
        all_issues = self._check(exercise, angles, state, keypoints, extra or {})

        # Update streak counters for every possible issue
        # Issues NOT detected this frame get their streak reset to 0
        active_msgs = {i.message for i in all_issues}
        all_possible = self._all_possible_messages(exercise)

        for msg in all_possible:
            if msg in active_msgs:
                self._issue_streak[msg] = self._issue_streak.get(msg, 0) + 1
            else:
                self._issue_streak[msg] = 0

        # Filter to issues that have persisted long enough
        persistent_issues = [
            i for i in all_issues
            if self._issue_streak.get(i.message, 0) >= PERSIST_FRAMES[i.priority]
        ]

        if not persistent_issues:
            self._active_msg    = "✓ Good form"
            self._active_blocks = False
            return "✓ Good form", False

        # Sort by priority (CRITICAL first)
        persistent_issues.sort(key=lambda i: i.priority.value)

        # Show highest-priority issue that has cooled down
        for issue in persistent_issues:
            if self._can_show(issue.message, issue.priority):
                self._last_shown[issue.message] = time.time()
                self._active_msg    = issue.message
                self._active_blocks = issue.blocks_rep
                return issue.message, issue.blocks_rep

        # All on cooldown — return cached (still honour blocks from persistent issues)
        has_blocking = any(i.blocks_rep for i in persistent_issues)
        return self._active_msg, has_blocking

    def reset(self) -> None:
        self._last_shown.clear()
        self._issue_streak.clear()
        self._active_msg    = ""
        self._active_blocks = False

    # ── Cooldown ──────────────────────────────────────────────────────────────

    def _can_show(self, msg: str, priority: Priority) -> bool:
        cooldown = _COOLDOWN[priority] * self._scale
        return time.time() - self._last_shown.get(msg, 0.0) >= cooldown

    # ── Router ────────────────────────────────────────────────────────────────

    def _check(
        self,
        exercise:  Exercise,
        angles:    Dict[str, float],
        state:     str,
        keypoints: Optional[np.ndarray],
        extra:     Dict,
    ) -> List[_Issue]:
        fn = {
            Exercise.SQUAT:            self._check_squat,
            Exercise.PUSHUP:           self._check_pushup,
            Exercise.JUMPING_JACK:     self._check_jumping_jack,
            Exercise.HIGH_KNEES:       self._check_high_knees,
            Exercise.PLANK:            self._check_plank,
            Exercise.PULLUP:           self._check_pullup,
            Exercise.SITUP:            self._check_situp,
            Exercise.LUNGE:            self._check_lunge,
            Exercise.MOUNTAIN_CLIMBER: self._check_mountain_climber,
            Exercise.BURPEE:           self._check_burpee,
        }.get(exercise)
        return fn(angles, state, keypoints, extra) if fn else []

    def _all_possible_messages(self, exercise: Exercise) -> List[str]:
        """Return all possible message strings for this exercise (for streak mgmt)."""
        reg = REGISTRY.get(exercise)
        if reg is None:
            return []
        return list(reg.mistake_messages.values())

    # ── Per-exercise checkers ─────────────────────────────────────────────────

    def _check_squat(self, angles, state, kp, extra) -> List[_Issue]:
        issues: List[_Issue] = []
        cfg   = REGISTRY[Exercise.SQUAT].form
        msgs  = REGISTRY[Exercise.SQUAT].mistake_messages

        back     = angles.get("Back", 0.0)
        knee_l   = angles.get("Left Knee",  0.0)
        knee_r   = angles.get("Right Knee", 0.0)
        knee_avg = _avg_nz(knee_l, knee_r)

        # Forward lean — use whichever side is visible (side-view tolerance)
        if back > 0 and back < cfg["back_min"]:
            issues.append(_Issue(msgs["lean_forward"], Priority.CRITICAL, blocks_rep=True))

        # Knee valgus collapse — only check if both knees and hips are visible
        if kp is not None and knee_avg > 0 and knee_avg < 135:
            lk_x = float(kp[13, 0]); rk_x = float(kp[14, 0])
            lh_x = float(kp[11, 0]); rh_x = float(kp[12, 0])
            # Only check if ALL four points are non-zero (i.e. visible)
            if lk_x > 0 and rk_x > 0 and lh_x > 0 and rh_x > 0:
                hip_w  = abs(rh_x - lh_x)
                knee_w = abs(rk_x - lk_x)
                if hip_w > 20 and knee_w / max(hip_w, 1) < cfg["knee_cave_ratio"]:
                    issues.append(_Issue(msgs["knees_cave"], Priority.CRITICAL, blocks_rep=True))

        # Depth — only in down state, only if angle is reliably measured
        if state == "down" and knee_avg > 0 and knee_avg > 115:
            issues.append(_Issue(msgs["not_deep"], Priority.WARNING))

        return issues

    def _check_pushup(self, angles, state, kp, extra) -> List[_Issue]:
        issues: List[_Issue] = []
        cfg  = REGISTRY[Exercise.PUSHUP].form
        msgs = REGISTRY[Exercise.PUSHUP].mistake_messages

        # Use best available side for hip (handles side view)
        hip   = _best_nz(angles.get("Left Hip", 0.0), angles.get("Right Hip", 0.0))
        elbow = _best_nz(angles.get("Left Elbow", 0.0), angles.get("Right Elbow", 0.0))

        if hip > 0 and hip < cfg["hip_sag_max"]:
            issues.append(_Issue(msgs["hip_sag"],  Priority.CRITICAL, blocks_rep=True))
        if hip > 0 and hip > cfg["hip_pike_min"]:
            issues.append(_Issue(msgs["hip_pike"], Priority.WARNING))
        if state == "down" and elbow > 0 and elbow > 115:
            issues.append(_Issue(msgs["not_deep"],   Priority.WARNING))
        if state == "up"   and elbow > 0 and elbow < 130:
            issues.append(_Issue(msgs["incomplete"], Priority.WARNING))

        return issues

    def _check_jumping_jack(self, angles, state, kp, extra) -> List[_Issue]:
        issues: List[_Issue] = []
        cfg  = REGISTRY[Exercise.JUMPING_JACK].form
        msgs = REGISTRY[Exercise.JUMPING_JACK].mistake_messages

        arm = _avg_nz(angles.get("L Arm Raise", 0.0), angles.get("R Arm Raise", 0.0))
        if state == "up" and arm > 0 and arm < cfg["arms_min_up"]:
            issues.append(_Issue(msgs["arms_not_up"], Priority.WARNING))
        return issues

    def _check_high_knees(self, angles, state, kp, extra) -> List[_Issue]:
        issues: List[_Issue] = []
        if kp is None:
            return issues

        active = extra.get("active_knee", "left")
        if active == "left":
            knee_y, hip_y = float(kp[13, 1]), float(kp[11, 1])
        else:
            knee_y, hip_y = float(kp[14, 1]), float(kp[12, 1])

        if knee_y > 0 and hip_y > 0:
            cfg = REGISTRY[Exercise.HIGH_KNEES].form
            if knee_y > hip_y - cfg["knee_above_hip_px"]:
                msgs = REGISTRY[Exercise.HIGH_KNEES].mistake_messages
                issues.append(_Issue(msgs["knees_low"], Priority.WARNING))
        return issues

    def _check_plank(self, angles, state, kp, extra) -> List[_Issue]:
        issues: List[_Issue] = []
        cfg  = REGISTRY[Exercise.PLANK].form
        msgs = REGISTRY[Exercise.PLANK].mistake_messages

        hip = _best_nz(angles.get("Left Hip", 0.0), angles.get("Right Hip", 0.0))
        if hip > 0 and hip < cfg["hip_sag_max"]:
            issues.append(_Issue(msgs["hip_sag"],  Priority.CRITICAL, blocks_rep=True))
        elif hip > 0 and hip > cfg["hip_pike_min"]:
            issues.append(_Issue(msgs["hip_pike"], Priority.WARNING))
        return issues

    def _check_pullup(self, angles, state, kp, extra) -> List[_Issue]:
        issues: List[_Issue] = []
        msgs  = REGISTRY[Exercise.PULLUP].mistake_messages
        elbow = _best_nz(angles.get("Left Elbow", 0.0), angles.get("Right Elbow", 0.0))

        if state == "down" and elbow > 0 and elbow < 140:
            issues.append(_Issue(msgs["not_extended"], Priority.WARNING))
        if state == "up"   and elbow > 0 and elbow > 90:
            issues.append(_Issue(msgs["not_pulled"],   Priority.WARNING))
        return issues

    def _check_situp(self, angles, state, kp, extra) -> List[_Issue]:
        issues: List[_Issue] = []
        msgs = REGISTRY[Exercise.SITUP].mistake_messages
        back = angles.get("Back Angle", angles.get("Back", 0.0))

        if state == "up" and back > 0 and back < 125:
            issues.append(_Issue(msgs["not_full"], Priority.WARNING))
        return issues

    def _check_lunge(self, angles, state, kp, extra) -> List[_Issue]:
        issues: List[_Issue] = []
        cfg  = REGISTRY[Exercise.LUNGE].form
        msgs = REGISTRY[Exercise.LUNGE].mistake_messages
        back     = angles.get("Back", 0.0)
        min_knee = _min_nz(angles.get("Left Knee", 0.0), angles.get("Right Knee", 0.0))

        if back > 0 and back < cfg["back_min"]:
            issues.append(_Issue(msgs["lean_forward"], Priority.WARNING))
        if state == "down" and min_knee > 0 and min_knee > 120:
            issues.append(_Issue(msgs["not_deep"], Priority.WARNING))
        return issues

    def _check_mountain_climber(self, angles, state, kp, extra) -> List[_Issue]:
        issues: List[_Issue] = []
        cfg  = REGISTRY[Exercise.MOUNTAIN_CLIMBER].form
        msgs = REGISTRY[Exercise.MOUNTAIN_CLIMBER].mistake_messages

        if kp is not None:
            lh_y = float(kp[11, 1]); rh_y = float(kp[12, 1])
            if lh_y > 0 and rh_y > 0:
                if abs(lh_y - rh_y) > cfg["hip_level_px"]:
                    issues.append(_Issue(msgs["hips_rotating"], Priority.WARNING))
        return issues

    def _check_burpee(self, angles, state, kp, extra) -> List[_Issue]:
        return []


# ── Helpers ────────────────────────────────────────────────────────────────────

def _avg_nz(*values: float) -> float:
    valid = [v for v in values if v > 0.0]
    return sum(valid) / len(valid) if valid else 0.0

def _min_nz(*values: float) -> float:
    valid = [v for v in values if v > 0.0]
    return min(valid) if valid else 0.0

def _best_nz(a: float, b: float) -> float:
    """
    Return whichever of a, b is non-zero.
    If both are non-zero, return their average.
    Handles side-view scenarios where one side may be occluded (zero).
    """
    if a > 0 and b > 0:
        return (a + b) / 2.0
    return a if a > 0 else b
