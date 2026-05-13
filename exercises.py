from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List


# ── Exercise catalogue ────────────────────────────────────────────────────────

class Exercise(Enum):
    SQUAT            = "squat"
    PUSHUP           = "pushup"
    JUMPING_JACK     = "jumping_jack"
    HIGH_KNEES       = "high_knees"
    PLANK            = "plank"
    PULLUP           = "pullup"
    SITUP            = "situp"
    LUNGE            = "lunge"
    MOUNTAIN_CLIMBER = "mountain_climber"
    BURPEE           = "burpee"


# ── Keyboard shortcut → Exercise mapping ─────────────────────────────────────

KEY_MAP: Dict[str, Exercise] = {
    "1": Exercise.SQUAT,
    "2": Exercise.PUSHUP,
    "3": Exercise.JUMPING_JACK,
    "4": Exercise.HIGH_KNEES,
    "5": Exercise.PLANK,
    "6": Exercise.PULLUP,
    "7": Exercise.SITUP,
    "8": Exercise.LUNGE,
    "9": Exercise.MOUNTAIN_CLIMBER,
    "0": Exercise.BURPEE,
}


# ── Exercise definition dataclass ─────────────────────────────────────────────

@dataclass
class ExerciseDef:
    """Complete specification of one exercise."""

    display_name:   str           # human-readable label shown on HUD
    keyboard_key:   str           # single character to switch to this exercise
    description:    str           # one-line description for --help / startup

    # Whether reps are time-based (Plank) vs motion-based (everything else)
    is_timed: bool = False

    # Primary angle thresholds
    # For most exercises: angle < down_threshold → DOWN position
    #                     angle > up_threshold   → UP  position
    # For inverted exercises (pull-up): these are still used but with
    # reversed comparison logic in counter._update_*
    down_threshold: float = 90.0
    up_threshold:   float = 160.0

    # Named form thresholds used by FeedbackEngine
    form: Dict[str, float] = field(default_factory=dict)

    # Angle names to display on the HUD (must match keys returned by get_joint_angles)
    display_joints: List[str] = field(default_factory=list)

    # Feedback message strings keyed by mistake code
    mistake_messages: Dict[str, str] = field(default_factory=dict)


# ── Exercise registry ─────────────────────────────────────────────────────────

REGISTRY: Dict[Exercise, ExerciseDef] = {

    # ── 1. Squat ──────────────────────────────────────────────────────────────
    Exercise.SQUAT: ExerciseDef(
        display_name  = "Squat",
        keyboard_key  = "1",
        description   = "Bodyweight squat — hip→knee→ankle angle",
        down_threshold = 90.0,   # knee angle below this = bottom of squat
        up_threshold   = 160.0,  # knee angle above this = standing
        form = {
            "back_min":         150.0,  # back angle below → excessive forward lean
            "knee_cave_ratio":  0.65,   # knee_width/hip_width below → valgus collapse
        },
        display_joints   = ["Left Knee", "Right Knee", "Back"],
        mistake_messages = {
            "lean_forward": "⚠ Chest up — don't lean forward!",
            "knees_cave":   "⚠ Knees caving in — drive them out!",
            "not_deep":     "⚠ Go deeper — break 90° at the knee",
        },
    ),

    # ── 2. Push-up ────────────────────────────────────────────────────────────
    Exercise.PUSHUP: ExerciseDef(
        display_name  = "Push-up",
        keyboard_key  = "2",
        description   = "Standard push-up — shoulder→elbow→wrist angle",
        down_threshold = 90.0,
        up_threshold   = 155.0,
        form = {
            "hip_sag_max":  152.0,  # shoulder→hip→knee below → hips drooping
            "hip_pike_min": 198.0,  # shoulder→hip→knee above → hips piked
        },
        display_joints   = ["Left Elbow", "Right Elbow", "Left Hip"],
        mistake_messages = {
            "hip_sag":    "⚠ Hips drooping — brace your core!",
            "hip_pike":   "⚠ Hips too high — lower them!",
            "not_deep":   "⚠ Lower your chest closer to the floor!",
            "incomplete": "⚠ Lock arms fully at the top!",
        },
    ),

    # ── 3. Jumping Jack ───────────────────────────────────────────────────────
    Exercise.JUMPING_JACK: ExerciseDef(
        display_name  = "Jumping Jack",
        keyboard_key  = "3",
        description   = "Jumping jacks — hip→shoulder→wrist arm elevation",
        down_threshold = 40.0,   # arm angle low  → arms at sides
        up_threshold   = 140.0,  # arm angle high → arms overhead
        form = {
            "arms_min_up": 130.0,  # arms must reach at least this angle overhead
        },
        display_joints   = ["L Arm Raise", "R Arm Raise"],
        mistake_messages = {
            "arms_not_up": "⚠ Raise arms fully overhead!",
        },
    ),

    # ── 4. High Knees ─────────────────────────────────────────────────────────
    Exercise.HIGH_KNEES: ExerciseDef(
        display_name  = "High Knees",
        keyboard_key  = "4",
        description   = "High knees — knee must rise above hip level",
        # Not angle-based; counter uses pixel position comparison
        down_threshold = 0.0,
        up_threshold   = 0.0,
        form = {
            "knee_above_hip_px": 5.0,  # knee.y must be < hip.y by at least this (px)
        },
        display_joints   = ["L Knee Height", "R Knee Height"],
        mistake_messages = {
            "knees_low": "⚠ Drive knees higher — above hip level!",
        },
    ),

    # ── 5. Plank ──────────────────────────────────────────────────────────────
    Exercise.PLANK: ExerciseDef(
        display_name  = "Plank",
        keyboard_key  = "5",
        description   = "Plank hold — time-based, body alignment tracked",
        is_timed      = True,
        form = {
            "hip_sag_max":  153.0,
            "hip_pike_min": 202.0,
        },
        display_joints   = ["Body Align", "Left Hip"],
        mistake_messages = {
            "hip_sag":  "⚠ Hips dropping — squeeze glutes!",
            "hip_pike": "⚠ Hips too high — straighten your back!",
        },
    ),

    # ── 6. Pull-up ────────────────────────────────────────────────────────────
    Exercise.PULLUP: ExerciseDef(
        display_name  = "Pull-up",
        keyboard_key  = "6",
        description   = "Pull-up — shoulder→elbow→wrist, inverted thresholds",
        # For pull-up: DOWN = arms extended (large elbow angle)
        #              UP   = arms flexed   (small elbow angle)
        down_threshold = 155.0,  # elbow angle ABOVE this → hanging position
        up_threshold   =  70.0,  # elbow angle BELOW this → chin above bar
        display_joints   = ["Left Elbow", "Right Elbow"],
        mistake_messages = {
            "not_extended": "⚠ Fully extend arms at the bottom!",
            "not_pulled":   "⚠ Pull chin above the bar!",
        },
    ),

    # ── 7. Sit-up ─────────────────────────────────────────────────────────────
    Exercise.SITUP: ExerciseDef(
        display_name  = "Sit-up",
        keyboard_key  = "7",
        description   = "Sit-up — torso back angle (low=lying, high=sitting)",
        # back_angle (shoulder→hip vs vertical): ~90° lying flat, ~160° sitting up
        down_threshold =  100.0,   # back_angle below → lying down
        up_threshold   =  140.0,   # back_angle above → fully up
        display_joints   = ["Back Angle"],
        mistake_messages = {
            "not_full":   "⚠ Come all the way up — don't half-rep!",
            "pull_neck":  "⚠ Don't yank your neck — use your core!",
        },
    ),

    # ── 8. Lunge ─────────────────────────────────────────────────────────────
    Exercise.LUNGE: ExerciseDef(
        display_name  = "Lunge",
        keyboard_key  = "8",
        description   = "Forward lunge — front knee angle, alternating legs",
        down_threshold = 95.0,
        up_threshold   = 155.0,
        form = {
            "back_min":  145.0,  # torso should stay upright
        },
        display_joints   = ["Left Knee", "Right Knee", "Back"],
        mistake_messages = {
            "lean_forward": "⚠ Torso upright — don't lean forward!",
            "not_deep":     "⚠ Lower thigh to parallel — go deeper!",
        },
    ),

    # ── 9. Mountain Climber ───────────────────────────────────────────────────
    Exercise.MOUNTAIN_CLIMBER: ExerciseDef(
        display_name  = "Mountain Climber",
        keyboard_key  = "9",
        description   = "Mountain climbers — alternating knee drives in plank",
        # shoulder→hip→knee angle: large = leg back, small = knee driven forward
        down_threshold = 130.0,   # leg extended back
        up_threshold   =  70.0,   # knee driven toward chest
        form = {
            "hip_level_px": 35.0,   # left/right hip y diff above → rotation
        },
        display_joints   = ["L Hip Angle", "R Hip Angle"],
        mistake_messages = {
            "hips_rotating": "⚠ Keep hips level — don't rotate!",
        },
    ),

    # ── 10. Burpee ────────────────────────────────────────────────────────────
    Exercise.BURPEE: ExerciseDef(
        display_name  = "Burpee",
        keyboard_key  = "0",
        description   = "Burpee — standing↔floor via back angle + knee angle",
        # Use back_angle to detect floor vs standing
        # Standing: back ~170°+, Floor (plank): back ~90°
        down_threshold = 110.0,   # back angle below → body horizontal (floor)
        up_threshold   = 155.0,   # back angle above → standing
        display_joints   = ["Left Knee", "Back Angle"],
        mistake_messages = {
            "no_extension": "⚠ Stand fully tall — jump at the top!",
        },
    ),
}


# ── Convenience helpers ───────────────────────────────────────────────────────

def list_exercises() -> str:
    """Return a formatted string listing all exercises and their keys."""
    lines = ["  Key | Exercise"]
    lines.append("  " + "-" * 30)
    for key, ex in KEY_MAP.items():
        d = REGISTRY[ex]
        lines.append(f"   {key}  | {d.display_name:<18s} — {d.description}")
    return "\n".join(lines)
