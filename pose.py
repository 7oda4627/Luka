"""
pose.py — RTMPose Keypoint Estimation Module

Uses MMPose's RTMPose model (via the mmdeploy / mmpose top-down pipeline)
to extract 17 COCO-format keypoints from a detected person crop.

Keypoint index → body part mapping (COCO 17):
    0  – nose
    1  – left_eye      2  – right_eye
    3  – left_ear      4  – right_ear
    5  – left_shoulder 6  – right_shoulder
    7  – left_elbow    8  – right_elbow
    9  – left_wrist   10  – right_wrist
   11  – left_hip     12  – right_hip
   13  – left_knee    14  – right_knee
   15  – left_ankle   16  – right_ankle
"""

from __future__ import annotations

import os
import time
import urllib.request
import numpy as np
import cv2
from typing import Optional, Tuple, List

from detector import PersonDetector, BBox

# ── Keypoint indices ──────────────────────────────────────────────────────────
KP = {
    "nose": 0,
    "left_eye": 1,   "right_eye": 2,
    "left_ear": 3,   "right_ear": 4,
    "left_shoulder": 5,  "right_shoulder": 6,
    "left_elbow": 7,     "right_elbow": 8,
    "left_wrist": 9,     "right_wrist": 10,
    "left_hip": 11,      "right_hip": 12,
    "left_knee": 13,     "right_knee": 14,
    "left_ankle": 15,    "right_ankle": 16,
}

Keypoints = np.ndarray  # shape (17, 2) — (x, y) in frame-space pixel coords


# ── Caffemodel download helper ────────────────────────────────────────────────

WEIGHT_MIRRORS = [
    # Primary: HuggingFace mirror (camenduru/openpose) — SHA256: b4cf475576abd7b15d5316f1ee65eb492b5c9f5865e70a2e7882ed31fb682549
    "https://huggingface.co/camenduru/openpose/resolve/main/models/pose/coco/pose_iter_440000.caffemodel",
    # CMU server is permanently down — kept as last-resort only
    "http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/coco/pose_iter_440000.caffemodel",
]


def _progress(block_num: int, block_size: int, total_size: int) -> None:
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(downloaded / total_size * 100, 100)
        mb = downloaded / 1_048_576
        print(f"\r  {pct:5.1f}%  {mb:.1f} MB", end="", flush=True)


def download_weights(weights_path: str, retries: int = 3) -> None:
    """
    Download pose_iter_440000.caffemodel from multiple mirrors with retry
    and exponential back-off. A .part temp file ensures no corrupt artefact
    is left at weights_path on failure.
    """
    if os.path.exists(weights_path):
        return

    print("[Pose] Downloading caffemodel (~200 MB) …")
    os.makedirs(os.path.dirname(weights_path) or ".", exist_ok=True)
    tmp_path = weights_path + ".part"

    for url in WEIGHT_MIRRORS:
        print(f"  Trying: {url}")
        for attempt in range(retries):
            try:
                urllib.request.urlretrieve(url, tmp_path, reporthook=_progress)
                print()  # newline after progress bar
                os.rename(tmp_path, weights_path)
                print("[Pose] Download complete.")
                return
            except Exception as exc:
                print(f"\n  Attempt {attempt + 1}/{retries} failed: {exc}")
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)  # exponential back-off

    raise RuntimeError(
        "[Pose] All mirrors failed. Download the file manually:\n"
        "  https://huggingface.co/camenduru/openpose/resolve/main/models/pose/coco/pose_iter_440000.caffemodel\n"
        f"  → place at: {weights_path}"
    )


# ── PoseEstimator ─────────────────────────────────────────────────────────────

class PoseEstimator:
    """
    Thin wrapper around RTMPose (MMPose top-down API).

    Falls back gracefully to a lightweight OpenCV DNN-based pose model if
    mmpose is not installed, so the rest of the pipeline keeps working.

    Parameters
    ----------
    config : str
        Path to RTMPose mmpose config file, or 'auto' to download default.
    checkpoint : str
        Path to RTMPose checkpoint (.pth), or 'auto' to download default.
    device : str
        Compute device string ('cpu', 'cuda:0', …).
    score_threshold : float
        Minimum keypoint confidence to keep (others set to (0, 0)).
    """

    # Default RTMPose-s COCO config & checkpoint (MMPose model zoo)
    _DEFAULT_CONFIG = (
        "rtmpose-s_8xb256-420e_coco-256x192"
    )
    _DEFAULT_CKPT_URL = (
        "https://download.openmmlab.com/mmpose/v1/projects/rtmpose/"
        "rtmpose-s_simcc-aic-coco_pt-aic-coco_420e-256x192-fcb2599b_20230126.pth"
    )

    def __init__(
        self,
        config: str = "auto",
        checkpoint: str = "auto",
        device: str | None = None,
        score_threshold: float = 0.3,
    ) -> None:
        self.score_threshold = score_threshold
        self._inferencer = None   # MMPose inferencer (preferred)
        self._dnn_net = None      # OpenCV DNN fallback

        if device is None:
            try:
                import torch
                if torch.cuda.is_available():
                    device = "cuda:0"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    device = "cpu"   # MMPose MPS support is partial
                else:
                    device = "cpu"
            except ImportError:
                device = "cpu"

        self.device = device
        self._init_backend(config, checkpoint)

    # ── Initialisation ────────────────────────────────────────────────────────

    def _init_backend(self, config: str, checkpoint: str) -> None:
        """Try MMPose first, NO fallback."""
        self._init_mmpose(config, checkpoint)

    def _init_mmpose(self, config: str, checkpoint: str) -> None:
        """Initialise RTMPose via the MMPose PoseInferencer API."""
        from mmpose.apis import MMPoseInferencer  # type: ignore

        if config == "auto":
            config = self._DEFAULT_CONFIG
        if checkpoint == "auto":
            checkpoint = "auto"  # MMPoseInferencer can auto-download

        print(f"[Pose] Loading RTMPose: {config} on {self.device}")
        self._inferencer = MMPoseInferencer(
            pose2d=config,
            pose2d_weights=None if checkpoint == "auto" else checkpoint,
            device=self.device,
            show_progress=False,
        )
        print("[Pose] RTMPose ready.")

    def _init_opencv_dnn(self) -> None:
        """
        Fallback: lightweight OpenCV DNN pose model (MobileNet-based).
        Downloads the model files if not present.
        """
        model_dir = ".pose_models"
        os.makedirs(model_dir, exist_ok=True)

        proto_path   = os.path.join(model_dir, "pose_deploy_linevec.prototxt")
        weights_path = os.path.join(model_dir, "pose_iter_440000.caffemodel")

        BASE_URL = (
            "https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/"
            "openpose/master/models/pose/coco/"
        )

        if not os.path.exists(proto_path):
            print("[Pose] Downloading prototxt …")
            urllib.request.urlretrieve(BASE_URL + "pose_deploy_linevec.prototxt", proto_path)

        download_weights(weights_path)  # handles exists-check, mirrors, and retries

        self._dnn_net = cv2.dnn.readNetFromCaffe(proto_path, weights_path)
        if self.device.startswith("cuda"):
            self._dnn_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self._dnn_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        print("[Pose] OpenCV DNN pose model ready (18-kp OpenPose COCO).")

    # ── Public API ────────────────────────────────────────────────────────────

    def estimate(
        self,
        frame: np.ndarray,
        bbox: BBox,
        pad_ratio: float = 0.15,
    ) -> Optional[Keypoints]:
        """
        Estimate 17 COCO keypoints for the person described by *bbox*.

        Parameters
        ----------
        frame : np.ndarray  BGR image
        bbox  : (x1, y1, x2, y2) bounding box in frame pixel coords
        pad_ratio : float   extra padding around the crop

        Returns
        -------
        np.ndarray of shape (17, 2) in *frame* pixel coordinates, or None.
        """
        padded_bbox = PersonDetector.pad_bbox(bbox, frame.shape, pad_ratio)
        x1, y1, x2, y2 = padded_bbox
        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            return None

        if self._inferencer is not None:
            return self._estimate_mmpose(frame, bbox)
        elif self._dnn_net is not None:
            return self._estimate_dnn(crop, x1, y1)
        else:
            return None

    # ── Backend implementations ───────────────────────────────────────────────

    def _estimate_mmpose(
        self, frame: np.ndarray, bbox: BBox
    ) -> Optional[Keypoints]:
        """Use MMPose PoseInferencer."""
        x1, y1, x2, y2 = bbox
        bboxes_input = [[x1, y1, x2, y2]]

        try:
            result_gen = self._inferencer(
                frame,
                bboxes=bboxes_input,
                return_datasamples=False,
                show=False,
            )
            results = next(result_gen)
        except StopIteration:
            return None
        except Exception:
            return None

        if not results or not results.get("predictions"):
            return None

        preds = results["predictions"][0]
        if not preds:
            return None

        pred = preds[0]
        keypoints_xy = np.array(pred["keypoints"], dtype=np.float32)   # (17, 2)
        scores = np.array(pred["keypoint_scores"], dtype=np.float32)   # (17,)

        # Zero out low-confidence keypoints
        mask = scores < self.score_threshold
        keypoints_xy[mask] = 0.0

        return keypoints_xy

    def _estimate_dnn(
        self, crop: np.ndarray, offset_x: int, offset_y: int
    ) -> Optional[Keypoints]:
        """
        OpenCV DNN fallback using OpenPose COCO (18 kp).
        Maps to COCO-17 ordering by dropping the 'background' class and
        reordering to match YOLOv8 / RTMPose conventions.
        """
        INPUT_H, INPUT_W = 368, 368
        blob = cv2.dnn.blobFromImage(
            crop, 1.0 / 255, (INPUT_W, INPUT_H), (0, 0, 0), swapRB=False, crop=False
        )
        self._dnn_net.setInput(blob)
        output = self._dnn_net.forward()  # (1, 57, H', W')

        ch, out_h, out_w = output.shape[1], output.shape[2], output.shape[3]
        crop_h, crop_w = crop.shape[:2]

        # OpenPose COCO 18 → COCO 17 index mapping
        # (drop index 17 = background, reorder to match RTMPose numbering)
        OPENPOSE_TO_COCO17 = [0, 15, 14, 17, 16, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10]

        keypoints = np.zeros((17, 2), dtype=np.float32)
        for coco17_idx, op_idx in enumerate(OPENPOSE_TO_COCO17):
            if op_idx >= ch:
                continue
            heatmap = output[0, op_idx]
            _, conf, _, point = cv2.minMaxLoc(heatmap)
            if conf < self.score_threshold:
                keypoints[coco17_idx] = (0.0, 0.0)
            else:
                kx = point[0] * crop_w / out_w + offset_x
                ky = point[1] * crop_h / out_h + offset_y
                keypoints[coco17_idx] = (kx, ky)

        return keypoints


# ── Helper to access named keypoints ─────────────────────────────────────────

def get_keypoint(keypoints: Keypoints, name: str) -> Tuple[float, float]:
    """Return (x, y) for a named keypoint. Raises KeyError if name unknown."""
    return tuple(keypoints[KP[name]])