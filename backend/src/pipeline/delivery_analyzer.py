"""
Combined delivery analyzer.

Current pipeline:

    Existing delivery clip
            |
            +----------------------+
            |                      |
            v                      v
      Ball Detection        Shot Classification
            |                      |
            v                      v
       Kalman Tracker       EfficientNet-B0
            |                      |
            v                  Transformer
       Ball trajectory              |
            |                       v
            |                 Shot prediction
            |                       |
            +----------+------------+
                       |
                       v
                Combined result

IMPORTANT
---------
This module does NOT perform clipping.

It expects an already-created .mp4 delivery clip.

For now we only calculate:

SHOT
    - prediction
    - confidence
    - class probabilities

BALL
    - tracking success
    - frames processed
    - YOLO detections
    - Kalman trajectory

The following are intentionally NOT calculated yet:

    - bowling speed
    - bounce
    - line
    - length
    - swing
    - release angle
    - bounce angle

Those can be added after the clipping/tracking pipeline is validated.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from .efficientnet_transformer import ImprovedSOTAModel
from ..tracking.detector import BallDetector
from ..tracking.tracker import BallTracker


# ============================================================
# PATHS
# ============================================================

# File:
#
# backend/
#   src/
#     pipeline/
#       delivery_analyzer.py
#
# parents[0] = pipeline
# parents[1] = src
# parents[2] = backend

BASE_DIR = Path(__file__).resolve().parents[2]

SHOT_MODEL_PATH = (
    BASE_DIR
    / "models"
    / "cricket_model_transformer.ckpt"
)


# ============================================================
# SHOT CLASSES
# ============================================================

SHOT_CLASSES = [
    "Cover",
    "Defense",
    "Flick",
    "Hook",
    "Late Cut",
    "Lofted",
    "Pull",
    "Square Cut",
    "Straight",
    "Sweep",
]


N_FRAMES = 30
IMAGE_SIZE = 224


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

IMAGE_TRANSFORM = transforms.Compose(
    [
        transforms.ToPILImage(),

        transforms.Resize(
            (
                IMAGE_SIZE,
                IMAGE_SIZE,
            )
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406,
            ],
            std=[
                0.229,
                0.224,
                0.225,
            ],
        ),
    ]
)


# ============================================================
# CHECKPOINT
# ============================================================

def _extract_state_dict(
    checkpoint: Any,
) -> dict[str, torch.Tensor]:
    """
    Extract the model state dictionary from the checkpoint.
    """

    if not isinstance(
        checkpoint,
        dict,
    ):
        raise RuntimeError(
            "Unsupported checkpoint format. "
            "Expected a dictionary."
        )

    if "state_dict" in checkpoint:

        state_dict = checkpoint[
            "state_dict"
        ]

    elif "model_state_dict" in checkpoint:

        state_dict = checkpoint[
            "model_state_dict"
        ]

    else:

        state_dict = checkpoint

    if not isinstance(
        state_dict,
        dict,
    ):
        raise RuntimeError(
            "Checkpoint state_dict is not a dictionary."
        )

    cleaned = {}

    for key, value in state_dict.items():

        new_key = key

        # Remove common training wrappers.
        while (
            new_key.startswith("module.")
            or new_key.startswith("model.")
        ):

            if new_key.startswith(
                "module."
            ):
                new_key = new_key[
                    len("module.") :
                ]

            elif new_key.startswith(
                "model."
            ):
                new_key = new_key[
                    len("model.") :
                ]

        # The deployment model creates this
        # compatibility buffer itself.
        if new_key == "class_weights":
            continue

        cleaned[
            new_key
        ] = value

    return cleaned


def load_shot_model() -> ImprovedSOTAModel:
    """
    Load the trained EfficientNet-B0 + Transformer model.
    """

    if not SHOT_MODEL_PATH.is_file():
        raise FileNotFoundError(
            "Shot model checkpoint not found:\n"
            f"{SHOT_MODEL_PATH}"
        )

    print(
        "[SHOT] Loading checkpoint:",
        flush=True,
    )

    print(
        f"[SHOT] {SHOT_MODEL_PATH}",
        flush=True,
    )

    checkpoint = torch.load(
        SHOT_MODEL_PATH,
        map_location="cpu",
        weights_only=False,
    )

    state_dict = _extract_state_dict(
        checkpoint
    )

    # Exact architecture from the supplied
    # efficientnet_transformer.py.
    model = ImprovedSOTAModel(
        num_classes=len(
            SHOT_CLASSES
        ),
        temporal_dim=256,
        n_frames=N_FRAMES,
    )

    # Compatibility buffer used by the
    # deployment architecture.
    state_dict[
        "class_weights"
    ] = (
        model.state_dict()[
            "class_weights"
        ]
        .detach()
        .clone()
    )

    # Strict loading is intentional.
    # It catches architecture/checkpoint mismatches.
    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model = model.to(
        DEVICE
    )

    model.eval()

    print(
        f"[SHOT] Model loaded successfully "
        f"on {DEVICE}",
        flush=True,
    )

    return model


# ============================================================
# VIDEO INFORMATION
# ============================================================

def _get_video_info(
    clip_path: Path,
) -> tuple[
    int,
    float,
    int,
    int,
]:
    """
    Return:

        total_frames
        fps
        width
        height
    """

    cap = cv2.VideoCapture(
        str(clip_path)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video:\n"
            f"{clip_path}"
        )

    try:

        total_frames = int(
            cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        fps = float(
            cap.get(
                cv2.CAP_PROP_FPS
            )
        )

        width = int(
            cap.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )

        height = int(
            cap.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )

        if total_frames <= 0:
            raise RuntimeError(
                f"Video contains no frames:\n"
                f"{clip_path}"
            )

        if width <= 0 or height <= 0:
            raise RuntimeError(
                f"Video has invalid dimensions:\n"
                f"{clip_path}"
            )

        return (
            total_frames,
            fps,
            width,
            height,
        )

    finally:

        cap.release()


# ============================================================
# LOAD 30 FRAMES
# ============================================================

def load_30_frames(
    clip_path: str | Path,
) -> torch.Tensor:
    """
    Uniformly sample exactly 30 frames.

    Output:

        (1, 30, 3, 224, 224)
    """

    clip_path = Path(
        clip_path
    )

    total_frames, _, _, _ = (
        _get_video_info(
            clip_path
        )
    )

    cap = cv2.VideoCapture(
        str(clip_path)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video:\n"
            f"{clip_path}"
        )

    try:

        indices = np.linspace(
            0,
            total_frames - 1,
            N_FRAMES,
            dtype=np.int64,
        )

        tensors = []

        for index in indices:

            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                int(index),
            )

            success, frame = cap.read()

            if (
                not success
                or frame is None
            ):
                raise RuntimeError(
                    "Failed to read frame "
                    f"{int(index)} from:\n"
                    f"{clip_path}"
                )

            # BGR -> RGB
            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )

            tensor = IMAGE_TRANSFORM(
                frame
            )

            tensors.append(
                tensor
            )

        if len(tensors) != N_FRAMES:
            raise RuntimeError(
                f"Expected {N_FRAMES} frames, "
                f"but loaded {len(tensors)}."
            )

        frames = torch.stack(
            tensors,
            dim=0,
        )

        # (30,3,224,224)
        # ->
        # (1,30,3,224,224)
        return frames.unsqueeze(
            0
        )

    finally:

        cap.release()


# ============================================================
# SHOT PREDICTION
# ============================================================

@torch.inference_mode()
def predict_shot(
    model: ImprovedSOTAModel,
    frames: torch.Tensor,
) -> dict[str, Any]:
    """
    Run shot classification.
    """

    expected_shape = (
        1,
        N_FRAMES,
        3,
        IMAGE_SIZE,
        IMAGE_SIZE,
    )

    if tuple(
        frames.shape
    ) != expected_shape:

        raise ValueError(
            "Unexpected input tensor shape. "
            f"Expected {expected_shape}, "
            f"got {tuple(frames.shape)}"
        )

    frames = frames.to(
        DEVICE,
        non_blocking=(
            DEVICE.type == "cuda"
        ),
    )

    logits = model(
        frames
    )

    if logits.ndim != 2:
        raise RuntimeError(
            "Unexpected model output shape: "
            f"{tuple(logits.shape)}"
        )

    if logits.shape[0] != 1:
        raise RuntimeError(
            "Expected batch size 1, "
            f"got {logits.shape[0]}"
        )

    if (
        logits.shape[1]
        != len(SHOT_CLASSES)
    ):
        raise RuntimeError(
            "Unexpected number of classes. "
            f"Expected {len(SHOT_CLASSES)}, "
            f"got {logits.shape[1]}"
        )

    probabilities = F.softmax(
        logits,
        dim=1,
    )

    confidence, class_index = (
        torch.max(
            probabilities,
            dim=1,
        )
    )

    predicted_index = int(
        class_index.item()
    )

    predicted_confidence = float(
        confidence.item()
    )

    class_probabilities = {}

    for index, class_name in enumerate(
        SHOT_CLASSES
    ):

        class_probabilities[
            class_name
        ] = round(
            float(
                probabilities[
                    0,
                    index,
                ].item()
            ),
            6,
        )

    return {
        "prediction": (
            SHOT_CLASSES[
                predicted_index
            ]
        ),

        "confidence": round(
            predicted_confidence,
            6,
        ),

        "class_probabilities": (
            class_probabilities
        ),
    }


# ============================================================
# BALL TRACKING
# ============================================================

def track_ball(
    clip_path: str | Path,
) -> dict[str, Any]:
    """
    Run:

        YOLO ball detector
             ->
        Kalman ball tracker

    on one clip.

    Only raw tracking information is returned.
    """

    clip_path = Path(
        clip_path
    )

    cap = cv2.VideoCapture(
        str(clip_path)
    )

    if not cap.isOpened():

        return {
            "tracking_success": False,
            "message": (
                "Could not open clip "
                "for ball tracking."
            ),
            "frames_processed": 0,
            "detections": 0,
            "trajectory": [],
        }

    frames_processed = 0
    detections = 0
    trajectory = []

    try:

        print(
            "[BALL] Loading detector...",
            flush=True,
        )

        detector = BallDetector()

        tracker = BallTracker()

        while True:

            success, frame = (
                cap.read()
            )

            if not success:
                break

            frames_processed += 1

            # ----------------------------------------------
            # YOLO
            # ----------------------------------------------

            detection = (
                detector.detect(
                    frame
                )
            )

            if detection is not None:
                detections += 1

            # ----------------------------------------------
            # KALMAN
            # ----------------------------------------------

            tracked_position = (
                tracker.update(
                    detection,
                    frames_processed,
                )
            )

            if tracked_position is not None:

                trajectory.append(
                    {
                        "frame": (
                            frames_processed
                        ),
                        "x": int(
                            tracked_position[0]
                        ),
                        "y": int(
                            tracked_position[1]
                        ),
                    }
                )

        tracking_success = (
            detections > 0
            and len(trajectory) > 0
        )

        if tracking_success:

            message = (
                "Ball detected and tracked."
            )

        else:

            message = (
                "No usable ball trajectory "
                "was detected."
            )

        return {
            "tracking_success": (
                tracking_success
            ),

            "message": message,

            "frames_processed": (
                frames_processed
            ),

            "detections": detections,

            "trajectory": trajectory,
        }

    except Exception as exc:

        # Tracking must never destroy
        # the shot prediction.

        return {
            "tracking_success": False,

            "message": (
                "Ball tracking failed: "
                f"{type(exc).__name__}: {exc}"
            ),

            "frames_processed": (
                frames_processed
            ),

            "detections": detections,

            "trajectory": trajectory,
        }

    finally:

        cap.release()


# ============================================================
# COMBINED ANALYSIS
# ============================================================

def analyze_delivery_clip(
    clip_path: str | Path,
    model: ImprovedSOTAModel | None = None,
    delivery_number: int | None = None,
) -> dict[str, Any]:
    """
    Analyze one existing delivery clip.

    Shot classification and ball tracking are
    intentionally independent.
    """

    clip_path = Path(
        clip_path
    )

    if not clip_path.is_file():

        raise FileNotFoundError(
            f"Clip not found:\n"
            f"{clip_path}"
        )

    print()
    print(
        "=" * 70
    )

    print(
        f"ANALYZING CLIP: "
        f"{clip_path.name}"
    )

    print(
        "=" * 70
    )

    # ========================================================
    # SHOT
    # ========================================================

    print(
        "[1/2] Running shot classification...",
        flush=True,
    )

    if model is None:
        model = load_shot_model()

    frames = load_30_frames(
        clip_path
    )

    shot_result = predict_shot(
        model,
        frames,
    )

    print(
        "[SHOT] "
        f"{shot_result['prediction']} "
        f"("
        f"{shot_result['confidence'] * 100:.2f}%"
        f")",
        flush=True,
    )

    # ========================================================
    # BALL
    # ========================================================

    print(
        "[2/2] Running ball tracking...",
        flush=True,
    )

    ball_result = track_ball(
        clip_path
    )

    print(
        "[BALL] "
        f"frames="
        f"{ball_result['frames_processed']} "
        f"detections="
        f"{ball_result['detections']} "
        f"trajectory="
        f"{len(ball_result['trajectory'])}",
        flush=True,
    )

    # ========================================================
    # COMBINED RESULT
    # ========================================================

    result = {
        "delivery": delivery_number,

        "clip": clip_path.name,

        "shot": shot_result,

        "ball": ball_result,
    }

    print(
        "[OK] Combined analysis completed.",
        flush=True,
    )

    return result