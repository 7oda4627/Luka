'''
        main.py — AI Fitness Trainer 
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
'''


from __future__ import annotations

import argparse
import sys
import time
import threading
from typing import Optional, Dict

import cv2
import numpy as np

# ── Project imports ───────────────────────────────────────────────────────────
from exercises import Exercise, KEY_MAP, REGISTRY, list_exercises
from angles    import get_joint_angles, reset_angle_cache
from temporal  import TemporalValidator
from counter   import RepCounter
from feedback  import FeedbackEngine
from utils     import (
    DualFPSCounter,
    KalmanKeypoints,
    select_largest_bbox,
    print_keybind_legend,
    print_status,
    draw_skeleton,
    draw_bbox,
    draw_hud,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AI Fitness Trainer v2 — YOLOv8 + RTMPose",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument(
        "--source", default="0",
        help="Webcam index (0,1,…) or path to video file  (default: 0)",
    )
    p.add_argument(
        "--exercise", default="squat",
        choices=[e.value for e in Exercise],
        help="Starting exercise  (default: squat)",
    )
    p.add_argument(
        "--det-conf", type=float, default=0.45,
        help="YOLOv8 person-detection confidence  (default: 0.45)",
    )
    p.add_argument(
        "--model", default="yolov8n.pt",
        help="YOLOv8 model file  (default: yolov8n.pt — fastest)",
    )
    p.add_argument(
        "--no-display", action="store_true",
        help="Disable OpenCV window — terminal output only",
    )
    p.add_argument(
        "--save", default=None, metavar="PATH",
        help="Save annotated output video to PATH  (optional)",
    )
    p.add_argument(
        "--kalman-proc", type=float, default=1e-2,
        help="Kalman process noise (lower=smoother but more lag, default: 0.01)",
    )
    p.add_argument(
        "--kalman-meas", type=float, default=5e-2,
        help="Kalman measurement noise (higher=trusts predictions more, default: 0.05)",
    )
    p.add_argument(
        "--list-exercises", action="store_true",
        help="Print all supported exercises and exit",
    )
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
#  Video I/O helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _open_capture(source: str) -> cv2.VideoCapture:
    src = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open source: {source}", file=sys.stderr)
        sys.exit(1)
    # Optimise webcam buffer — reduces latency
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def _create_writer(
    cap: cv2.VideoCapture, path: Optional[str]
) -> Optional[cv2.VideoWriter]:
    if path is None:
        return None
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Saving annotated output → {path}")
    return cv2.VideoWriter(path, fourcc, fps, (w, h))


# ═══════════════════════════════════════════════════════════════════════════════
#  Threaded inference worker
# ═══════════════════════════════════════════════════════════════════════════════

class _InferenceWorker:
    """
    Runs YOLOv8 + RTMPose in a background daemon thread.

    Performance strategy:
      • submit_frame() replaces any pending frame (always processes freshest)
      • get_result()   returns the latest completed result (never blocks)
      • Main loop (display) never waits for inference → display FPS is stable

    This is the core of Objective #1 (30+ FPS).
    """

    def __init__(self, detector, pose, smoother: KalmanKeypoints) -> None:
        self._det     = detector
        self._pose    = pose
        self._smoother = smoother

        # Shared state
        self._in_lock    = threading.Lock()
        self._out_lock   = threading.Lock()
        self._new_frame  = threading.Event()

        self._pending:   Optional[np.ndarray] = None
        self._bbox:      Optional[tuple]       = None
        self._keypoints: Optional[np.ndarray] = None
        self._detected   = False

        self._running = False
        self._thread  = threading.Thread(
            target=self._loop, name="InferenceThread", daemon=True
        )

    def start(self) -> None:
        self._running = True
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        self._new_frame.set()
        self._thread.join(timeout=3.0)

    def submit_frame(self, frame: np.ndarray) -> None:
        """Non-blocking: replace pending frame with the latest one."""
        with self._in_lock:
            self._pending = frame       # overwrite — newest always wins
        self._new_frame.set()

    def get_result(self) -> tuple:
        """Non-blocking: return (detected, bbox, keypoints)."""
        with self._out_lock:
            return self._detected, self._bbox, (
                self._keypoints.copy() if self._keypoints is not None else None
            )

    # ── Background loop ───────────────────────────────────────────────────────

    def _loop(self) -> None:
        while self._running:
            triggered = self._new_frame.wait(timeout=0.5)
            self._new_frame.clear()

            if not self._running:
                break
            if not triggered:
                continue

            with self._in_lock:
                frame = self._pending
                self._pending = None

            if frame is None:
                continue

            detected, bbox, kp = self._infer(frame)

            with self._out_lock:
                self._detected   = detected
                self._bbox       = bbox
                self._keypoints  = kp

    def _infer(self, frame: np.ndarray) -> tuple:
        """Full detect → pose → smooth pipeline (runs in inference thread)."""
        # Step 1: Person detection
        from detector import PersonDetector
        bboxes = self._det.detect(frame)

        if not bboxes:
            self._smoother.reset()
            return False, None, None

        bbox = select_largest_bbox(bboxes)

        # Step 2: Pose estimation
        kp = self._pose.estimate(frame, bbox)
        if kp is None:
            return True, bbox, None

        # Step 3: Kalman smoothing
        kp = self._smoother.smooth(kp)
        return True, bbox, kp


# ═══════════════════════════════════════════════════════════════════════════════
#  Session state — everything about the CURRENT exercise session
# ═══════════════════════════════════════════════════════════════════════════════

class _Session:
    """
    Holds all per-exercise mutable state:
      counter, feedback engine, cached angles, form status.

    Calling switch(exercise) resets everything cleanly.
    """

    def __init__(self, exercise: Exercise) -> None:
        self.exercise = exercise
        self.counter  = RepCounter(exercise)
        self.feedback = FeedbackEngine(cooldown_scale=1.0)
        self.temporal = TemporalValidator()          # v3: temporal consistency
        self.angles:        Dict[str, float] = {}
        self.form_msg:      str              = ""
        self.rep_blocked:   bool             = False
        self._last_infer_id = -1   # used to detect when a new result arrived

    def switch(self, new_exercise: Exercise) -> None:
        """Switch exercise in-place (no new object needed by caller)."""
        self.exercise      = new_exercise
        self.counter       = RepCounter(new_exercise)
        self.feedback.reset()
        self.temporal.reset()        # v3: clear angle history
        reset_angle_cache()           # v3: clear jump-guard cache
        self.angles        = {}
        self.form_msg      = ""
        self.rep_blocked   = False
        print(f"\n\n[SWITCH] → {new_exercise.value.upper()}\n", flush=True)

    def reset_reps(self) -> None:
        """Reset rep count only (keep exercise, keep feedback history)."""
        self.counter.reset()
        print(f"\n\n[RESET] Reps reset to 0\n", flush=True)

    def process(
        self,
        keypoints:  np.ndarray,
        fps_ticker: DualFPSCounter,
    ) -> None:
        """
        Full logic step called when a fresh inference result is ready:
          1. Calculate joint angles
          2. Evaluate form (FeedbackEngine)
          3. Update rep counter (with validity flag)
          4. Tick the inference FPS counter
        """
        fps_ticker.tick_infer()

        # ── 1. Angles (v3: with confidence scores for filtering) ───────────────
        self.angles = get_joint_angles(keypoints, self.exercise)

        # ── 2. Temporal consistency (v3: feed angle history) ──────────────────
        self.temporal.add(self.angles)

        # ── 3. Form check ─────────────────────────────────────────────────────
        extra = {}
        if hasattr(self.counter, "_active_knee"):
            extra["active_knee"] = self.counter._active_knee

        msg, blocked = self.feedback.evaluate(
            exercise  = self.exercise,
            angles    = self.angles,
            state     = self.counter.state,
            keypoints = keypoints,
            extra     = extra,
        )
        self.form_msg    = msg
        self.rep_blocked = blocked

        # ── 4. Rep counting (v3: temporal validator injected) ─────────────────
        self.counter.update(
            angles       = self.angles,
            keypoints    = keypoints,
            is_rep_valid = not blocked,
            temporal     = self.temporal,
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  Main entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    args = _parse_args()

    if args.list_exercises:
        print("\nSupported exercises:\n")
        print(list_exercises())
        sys.exit(0)

    # ── Module initialisation ─────────────────────────────────────────────────
    print("[INFO] Loading YOLOv8 person detector …")
    from detector import PersonDetector
    detector = PersonDetector(
        model_name      = args.model,
        conf_threshold  = args.det_conf,
    )

    print("[INFO] Loading RTMPose estimator …")
    from pose import PoseEstimator
    pose = PoseEstimator()

    smoother = KalmanKeypoints(
        proc_noise = args.kalman_proc,
        meas_noise = args.kalman_meas,
    )

    # ── Video I/O ─────────────────────────────────────────────────────────────
    cap    = _open_capture(args.source)
    writer = _create_writer(cap, args.save)

    # ── State objects ─────────────────────────────────────────────────────────
    start_exercise = Exercise(args.exercise)
    session        = _Session(start_exercise)
    fps            = DualFPSCounter(window=30)

    # ── Background inference thread ───────────────────────────────────────────
    worker = _InferenceWorker(detector, pose, smoother)
    worker.start()
    print("[INFO] Inference thread started.")

    # ── Startup banner ────────────────────────────────────────────────────────
    print(f"\n[INFO] Starting exercise  : {start_exercise.value.upper()}")
    print(f"[INFO] Video source       : {args.source}")
    print(f"[INFO] Detection model    : {args.model}")
    print(f"[INFO] Kalman proc/meas   : {args.kalman_proc} / {args.kalman_meas}")
    print_keybind_legend(KEY_MAP)

    # ── Track last result to detect new inference results ─────────────────────
    last_kp_id   = id(None)
    last_kp      = None
    last_bbox    = None
    detected     = False

    # ── Main loop ─────────────────────────────────────────────────────────────
    try:
        while True:
            # ── Capture ───────────────────────────────────────────────────────
            ret, frame = cap.read()
            if not ret:
                print("\n[INFO] End of video stream.")
                break

            # Tick display FPS every loop iteration
            fps.tick_display()

            # ── Feed frame to inference thread (non-blocking) ─────────────────
            worker.submit_frame(frame)

            # ── Poll latest inference result (non-blocking) ───────────────────
            detected_new, bbox_new, kp_new = worker.get_result()

            if kp_new is not None and id(kp_new) != last_kp_id:
                # Fresh inference result arrived
                last_kp_id = id(kp_new)
                last_kp    = kp_new
                last_bbox  = bbox_new
                detected   = detected_new

                # Run full logic step (angles → feedback → rep count)
                session.process(kp_new, fps)

            elif detected_new and bbox_new is not None:
                # Person detected but no new keypoints yet — keep last
                last_bbox = bbox_new
                detected  = detected_new

            elif not detected_new:
                # Person left frame
                last_kp   = None
                last_bbox = None
                detected  = False

            # ── Terminal status line (every display frame) ────────────────────
            print_status(
                display_fps   = fps.display_fps,
                infer_fps     = fps.infer_fps,
                reps          = session.counter.reps,
                state         = session.counter.state,
                angles        = session.angles,
                form_feedback = session.form_msg,
                rep_blocked   = session.rep_blocked,
                exercise      = session.exercise,
                hold_seconds  = session.counter.hold_seconds,
            )

            # ── Render frame ──────────────────────────────────────────────────
            if not args.no_display or writer is not None:
                display = frame.copy()

                if last_bbox is not None:
                    draw_bbox(display, last_bbox)

                if last_kp is not None:
                    draw_skeleton(
                        display, last_kp,
                        form_bad = session.rep_blocked,
                    )

                draw_hud(
                    frame         = display,
                    display_fps   = fps.display_fps,
                    infer_fps     = fps.infer_fps,
                    reps          = session.counter.reps,
                    state         = session.counter.state,
                    angles        = session.angles,
                    feedback      = session.form_msg,
                    rep_blocked   = session.rep_blocked,
                    exercise      = session.exercise,
                    hold_seconds  = session.counter.hold_seconds,
                )

                if writer is not None:
                    writer.write(display)

                if not args.no_display:
                    cv2.imshow("AI Fitness Trainer v2", display)

                    # ── Keyboard input ────────────────────────────────────────
                    key = cv2.waitKey(1) & 0xFF
                    key_char = chr(key) if key < 128 else ""

                    if key_char == "q":
                        print("\n[INFO] Quit requested.")
                        break

                    elif key_char == "r":
                        # Reset rep count for current exercise
                        session.reset_reps()

                    elif key_char in KEY_MAP:
                        # Real-time exercise switch — no restart needed
                        new_ex = KEY_MAP[key_char]
                        if new_ex != session.exercise:
                            session.switch(new_ex)
                            # Reset Kalman state for clean start
                            smoother.reset()

            # ── Throttle only if no display and no writer (pure terminal mode)
            else:
                # Still handle keyboard via select on stdin if supported
                pass

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")

    finally:
        # ── Clean shutdown ────────────────────────────────────────────────────
        worker.stop()
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

        print(f"\n{'═'*55}")
        print(f"  SESSION SUMMARY")
        print(f"{'═'*55}")
        print(f"  Exercise  : {session.exercise.value.replace('_', ' ').title()}")

        if REGISTRY[session.exercise].is_timed:
            print(f"  Hold time : {session.counter.hold_seconds:.1f} s")
        else:
            print(f"  Total reps: {session.counter.reps}")

        print(f"  Display   : {fps.display_fps:.1f} fps (avg)")
        print(f"  Inference : {fps.infer_fps:.1f} fps (avg)")
        print(f"{'═'*55}\n")


# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()
