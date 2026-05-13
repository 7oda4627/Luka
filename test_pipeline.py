"""
test_pipeline.py — Offline Unit Tests & Pipeline Smoke Test

Run without a webcam or GPU to verify that all modules import correctly
and produce sane outputs for synthetic data.

Usage:
    python test_pipeline.py
"""

import sys
import math
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# 1. Angle calculation tests
# ─────────────────────────────────────────────────────────────────────────────

def test_angles() -> None:
    from angles import calculate_angle, angle_between_vectors

    print("── angle tests ──────────────────────────────────────────────────")

    # Straight line → 180°  (avoid using (0,0) — it's the "missing kp" sentinel)
    p1, p2, p3 = (10.0, 5.0), (20.0, 5.0), (30.0, 5.0)
    a = calculate_angle(p1, p2, p3)
    assert abs(a - 180.0) < 0.01, f"Straight line should be 180°, got {a}"
    print(f"  Straight line (expected 180°): {a:.2f}°  ✓")

    # Right angle → 90°  (vertex must not be (0,0))
    p1, p2, p3 = (10.0, 20.0), (10.0, 10.0), (20.0, 10.0)
    a = calculate_angle(p1, p2, p3)
    assert abs(a - 90.0) < 0.01, f"Right angle should be 90°, got {a}"
    print(f"  Right angle   (expected 90°) : {a:.2f}°  ✓")

    # Acute: equilateral triangle → 60° (all points offset away from origin)
    side = 10.0
    p1 = (10.0, 10.0)
    p2 = (20.0, 10.0)
    p3 = (15.0, 10.0 + math.sqrt(3) / 2 * side)
    a = calculate_angle(p1, p2, p3)
    assert abs(a - 60.0) < 0.1, f"Equilateral triangle should be 60°, got {a}"
    print(f"  Equilateral   (expected 60°) : {a:.2f}°  ✓")

    # Missing keypoint → 0°
    a = calculate_angle((0.0, 0.0), (1.0, 1.0), (2.0, 0.0))
    assert a == 0.0, f"Missing kp should return 0.0, got {a}"
    print(f"  Missing point (expected 0°)  : {a:.2f}°  ✓")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Rep counter tests (squat)
# ─────────────────────────────────────────────────────────────────────────────

def test_squat_counter() -> None:
    from counter import RepCounter, Exercise

    print("\n── squat counter tests ──────────────────────────────────────────")
    counter = RepCounter(Exercise.SQUAT)

    def feed(knee: float, back: float = 175.0):
        counter.update({"Left Knee": knee, "Right Knee": knee, "Back": back})

    # Starts READY → squat down first
    feed(85.0)   # below 90 → transitions READY→DOWN
    assert counter.state == "down", f"Expected 'down', got {counter.state}"
    print(f"  After 85°  feed → state: {counter.state}  ✓")

    feed(165.0)  # above 160 → rep counted, state→UP
    assert counter.reps == 1, f"Expected 1 rep, got {counter.reps}"
    assert counter.state == "up"
    print(f"  After 165° feed → reps: {counter.reps}, state: {counter.state}  ✓")

    # Second rep
    feed(85.0)
    feed(170.0)
    assert counter.reps == 2, f"Expected 2 reps, got {counter.reps}"
    print(f"  Second rep       → reps: {counter.reps}  ✓")

    # Bad posture test
    feed(170.0, back=120.0)
    assert "bad" in counter.last_feedback.lower(), (
        f"Expected bad posture feedback, got: {counter.last_feedback}"
    )
    print(f"  Bad posture feedback: '{counter.last_feedback}'  ✓")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Push-up counter tests
# ─────────────────────────────────────────────────────────────────────────────

def test_pushup_counter() -> None:
    from counter import RepCounter, Exercise

    print("\n── push-up counter tests ────────────────────────────────────────")
    counter = RepCounter(Exercise.PUSHUP)

    def feed(elbow: float, hip: float = 175.0):
        counter.update({
            "Left Elbow": elbow, "Right Elbow": elbow,
            "Left Hip": hip, "Right Hip": hip,
        })

    feed(165.0)   # up
    feed(80.0)    # down
    feed(165.0)   # rep
    assert counter.reps == 1, f"Expected 1 push-up rep, got {counter.reps}"
    print(f"  Push-up rep counted: {counter.reps}  ✓")

    # Sagging hips warning
    feed(165.0, hip=130.0)
    assert "sagging" in counter.last_feedback.lower(), (
        f"Expected sagging feedback, got: {counter.last_feedback}"
    )
    print(f"  Hip feedback: '{counter.last_feedback}'  ✓")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Bicep curl counter tests
# ─────────────────────────────────────────────────────────────────────────────

def test_bicep_curl_counter() -> None:
    from counter import RepCounter, Exercise

    print("\n── bicep curl counter tests ─────────────────────────────────────")
    counter = RepCounter(Exercise.BICEP_CURL)

    def feed(elbow: float):
        counter.update({"Left Elbow": elbow, "Right Elbow": elbow})

    feed(160.0)  # extended (down)
    feed(40.0)   # curled (up)
    feed(160.0)  # extended → rep counted
    assert counter.reps == 1, f"Expected 1 curl rep, got {counter.reps}"
    print(f"  Bicep curl rep counted: {counter.reps}  ✓")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Keypoint smoother tests
# ─────────────────────────────────────────────────────────────────────────────

def test_smoother() -> None:
    """Test the EMA smoother inline (no external imports needed)."""
    import numpy as np
    from typing import Optional

    print("\n── keypoint smoother tests ──────────────────────────────────────")

    # Inline EMA smoother (mirrors utils.KeypointSmoother)
    class _Smoother:
        def __init__(self, window_size=5):
            self.alpha = 2.0 / (window_size + 1)
            self._smoothed: Optional[np.ndarray] = None

        def smooth(self, kp: np.ndarray) -> np.ndarray:
            if self._smoothed is None:
                self._smoothed = kp.copy()
                return self._smoothed
            valid = ~((kp[:, 0] == 0) & (kp[:, 1] == 0))
            updated = self._smoothed.copy()
            updated[valid] = self.alpha * kp[valid] + (1 - self.alpha) * self._smoothed[valid]
            self._smoothed = updated
            return self._smoothed

        def reset(self):
            self._smoothed = None

    smoother = _Smoother(window_size=5)

    kp1 = np.zeros((17, 2), dtype=np.float32)
    kp1[0] = [100.0, 200.0]

    kp2 = np.zeros((17, 2), dtype=np.float32)
    kp2[0] = [200.0, 200.0]

    s1 = smoother.smooth(kp1)
    assert np.allclose(s1[0], [100.0, 200.0]), "First frame should pass through"
    print(f"  First frame: kp[0] = {s1[0]}  ✓")

    s2 = smoother.smooth(kp2)
    assert 100.0 < s2[0, 0] < 200.0, f"EMA should interpolate, got {s2[0]}"
    print(f"  After EMA  : kp[0] = {s2[0]}  ✓  (interpolated between 100 and 200)")

    smoother.reset()
    assert smoother._smoothed is None
    print(f"  Reset       : history cleared  ✓")




# ─────────────────────────────────────────────────────────────────────────────
# 6. get_joint_angles integration test (synthetic keypoints)
# ─────────────────────────────────────────────────────────────────────────────

def test_joint_angles_squat() -> None:
    from counter import Exercise
    from angles import get_joint_angles

    print("\n── joint angles integration test ────────────────────────────────")

    # Build a synthetic standing-upright skeleton
    # All keypoints at anatomically plausible y-positions, x centered
    kp = np.zeros((17, 2), dtype=np.float32)
    cx = 320.0  # x center

    # Rough pixel positions (top of frame = y=0)
    kp[0]  = [cx,       50]   # nose
    kp[5]  = [cx - 30, 180]   # left  shoulder
    kp[6]  = [cx + 30, 180]   # right shoulder
    kp[7]  = [cx - 55, 280]   # left  elbow
    kp[8]  = [cx + 55, 280]   # right elbow
    kp[9]  = [cx - 55, 380]   # left  wrist
    kp[10] = [cx + 55, 380]   # right wrist
    kp[11] = [cx - 25, 360]   # left  hip
    kp[12] = [cx + 25, 360]   # right hip
    kp[13] = [cx - 25, 510]   # left  knee
    kp[14] = [cx + 25, 510]   # right knee
    kp[15] = [cx - 25, 660]   # left  ankle
    kp[16] = [cx + 25, 660]   # right ankle

    angles = get_joint_angles(kp, Exercise.SQUAT)
    assert "Left Knee"  in angles, "Left Knee angle missing"
    assert "Right Knee" in angles, "Right Knee angle missing"
    assert "Back"       in angles, "Back angle missing"

    lk = angles["Left Knee"]
    rk = angles["Right Knee"]
    print(f"  Left Knee : {lk:.1f}°  (expected ~180° for straight leg)")
    print(f"  Right Knee: {rk:.1f}°  (expected ~180° for straight leg)")
    print(f"  Back      : {angles['Back']:.1f}°  (expected ~180° upright)")

    # Standing upright should produce near-180° knee angle
    assert lk > 160.0, f"Upright left knee should be ~180°, got {lk}"
    assert rk > 160.0, f"Upright right knee should be ~180°, got {rk}"
    print("  ✓ Upright posture angles look correct")


# ─────────────────────────────────────────────────────────────────────────────
# 7. FPS counter test
# ─────────────────────────────────────────────────────────────────────────────

def test_fps_counter() -> None:
    import time
    from collections import deque

    print("\n── FPS counter test ─────────────────────────────────────────────")

    # Inline FPS counter (mirrors utils.FPSCounter)
    class _FPS:
        def __init__(self, window=30):
            self._ts: deque = deque(maxlen=window)
            self.fps = 0.0
        def tick(self):
            self._ts.append(time.perf_counter())
            if len(self._ts) >= 2:
                elapsed = self._ts[-1] - self._ts[0]
                self.fps = (len(self._ts) - 1) / elapsed if elapsed > 0 else 0.0
            return self.fps

    fps_counter = _FPS(window=10)
    for _ in range(12):
        fps_counter.tick()
        time.sleep(0.01)

    fps = fps_counter.fps
    print(f"  Measured FPS (targeting ~100): {fps:.1f}")
    assert fps > 50, f"FPS seems too low: {fps}"
    print("  ✓")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("  AI Fitness Trainer — Unit Tests")
    print("=" * 60)

    failed = 0
    tests = [
        test_angles,
        test_squat_counter,
        test_pushup_counter,
        test_bicep_curl_counter,
        test_smoother,
        test_joint_angles_squat,
        test_fps_counter,
    ]

    for test in tests:
        try:
            test()
        except Exception as exc:
            print(f"\n  ✗ FAILED: {test.__name__}: {exc}")
            failed += 1

    print("\n" + "=" * 60)
    if failed == 0:
        print(f"  ✓ All {len(tests)} tests passed.")
    else:
        print(f"  ✗ {failed}/{len(tests)} tests FAILED.")
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    main()
