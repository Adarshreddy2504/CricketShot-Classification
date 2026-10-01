from __future__ import annotations

import asyncio
import json
import os
import shutil
import statistics
import tempfile
import threading
import time
import uuid
from collections import deque
from contextlib import suppress
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from PIL import Image
from torchvision import transforms


# ============================================================================
# LOCAL IMPORTS
# ============================================================================

from src.pipeline.efficientnet_transformer import (
    ImprovedSOTAModel,
)

from src.tracking import (
    BallDetector,
    BallTracker,
    MAX_EARLY_POSITIONS,
    build_homography,
    classify_length,
    classify_line,
    compute_bounce_angle,
    compute_release_angle,
    compute_release_speed,
    detect_bounce,
    draw_trajectory_trail,
    estimate_swing,
    get_fps,
)


# ============================================================================
# PATHS
# ============================================================================

BASE_DIR = Path(__file__).resolve().parent

MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"
TMP_ROOT = BASE_DIR / "tmp"

BALL_MODEL = MODELS_DIR / "best.pt"

SHOT_MODEL = (
    MODELS_DIR
    / "cricket_model_transformer.ckpt"
)


# ============================================================================
# SHOT MODEL
# ============================================================================

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


# ============================================================================
# DELIVERY SETTINGS
# ============================================================================

MIN_DELIVERY_FRAMES = 8

MIN_FRAMES_BETWEEN_DELIVERIES = 60

PRE_ROLL_FRAMES = 10
POST_ROLL_FRAMES = 10

MIN_CLIP_FRAMES = 30
MAX_CLIP_FRAMES = 90


# ============================================================================
# RESOURCE SETTINGS
# ============================================================================

MAX_UPLOAD_BYTES = 500 * 1024 * 1024

TASK_TTL_SECONDS = 30 * 60

CLEANUP_INTERVAL_SECONDS = 60

MAX_TRAJECTORY_POINTS = 20000


# ============================================================================
# DEVICE
# ============================================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True


# ============================================================================
# APP
# ============================================================================

app = FastAPI(
    title="CricShot + CricketTracker API",
    version="4.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

TMP_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

app.mount(
    "/outputs",
    StaticFiles(
        directory=str(OUTPUT_DIR)
    ),
    name="outputs",
)


# ============================================================================
# GLOBAL STATE
# ============================================================================

_tasks: dict[str, dict[str, Any]] = {}

_tasks_lock = threading.RLock()

PROCESS_LOCK = threading.Lock()

_shot_model: ImprovedSOTAModel | None = None
_shot_model_lock = threading.Lock()

_ball_detector: BallDetector | None = None
_ball_detector_lock = threading.Lock()

_cleanup_started = False


# ============================================================================
# TIME
# ============================================================================

def now() -> float:
    return time.time()


# ============================================================================
# TASK MANAGEMENT
# ============================================================================

def create_task() -> str:

    task_id = str(
        uuid.uuid4()
    )

    with _tasks_lock:

        _tasks[task_id] = {
            "progress": 0,
            "status": "Queued",
            "cancelled": False,
            "created_at": now(),
            "updated_at": now(),
            "cancel_event": threading.Event(),
            "result": None,
            "error": None,
            "output_path": None,
        }

    return task_id


def update_task(
    task_id: str,
    progress: int,
    status: str,
    *,
    result: dict[str, Any] | None = None,
    error: str | None = None,
):

    with _tasks_lock:

        state = _tasks.get(task_id)

        if state is None:
            return

        state["progress"] = max(
            -1,
            min(
                100,
                int(progress),
            ),
        )

        state["status"] = status
        state["updated_at"] = now()

        if result is not None:
            state["result"] = result

        if error is not None:
            state["error"] = error


def public_task_state(
    task_id: str,
):

    with _tasks_lock:

        state = _tasks.get(task_id)

        if state is None:
            return None

        result = {
            "progress": int(
                state.get(
                    "progress",
                    0,
                )
            ),
            "status": str(
                state.get(
                    "status",
                    "unknown",
                )
            ),
            "cancelled": bool(
                state.get(
                    "cancelled",
                    False,
                )
            ),
        }

        if state.get("result") is not None:
            result["result"] = state["result"]

        if state.get("error"):
            result["error"] = state["error"]

        return result


def is_cancelled(
    task_id: str,
) -> bool:

    with _tasks_lock:

        state = _tasks.get(task_id)

        if state is None:
            return True

        event = state.get(
            "cancel_event"
        )

        return bool(
            event
            and event.is_set()
        )


def request_cancel(
    task_id: str,
) -> bool:

    with _tasks_lock:

        state = _tasks.get(task_id)

        if state is None:
            return False

        state["cancelled"] = True
        state["status"] = "Cancelling..."
        state["updated_at"] = now()

        event = state.get(
            "cancel_event"
        )

        if event is not None:
            event.set()

        return True


# ============================================================================
# CLEANUP
# ============================================================================

def start_cleanup_thread():

    global _cleanup_started

    if _cleanup_started:
        return

    _cleanup_started = True

    def cleanup_loop():

        while True:

            time.sleep(
                CLEANUP_INTERVAL_SECONDS
            )

            cutoff = (
                now()
                - TASK_TTL_SECONDS
            )

            expired = []

            with _tasks_lock:

                for task_id, state in list(
                    _tasks.items()
                ):

                    if (
                        state.get(
                            "updated_at",
                            0,
                        )
                        < cutoff
                    ):

                        expired.append(
                            task_id
                        )

                for task_id in expired:

                    state = _tasks.pop(
                        task_id,
                        None,
                    )

                    if not state:
                        continue

                    output_path = (
                        state.get(
                            "output_path"
                        )
                    )

                    if output_path:

                        with suppress(
                            OSError
                        ):

                            Path(
                                output_path
                            ).unlink()

    thread = threading.Thread(
        target=cleanup_loop,
        daemon=True,
        name="cricket-cleanup",
    )

    thread.start()


@app.on_event("startup")
async def startup():

    start_cleanup_thread()

    cutoff = (
        now()
        - TASK_TTL_SECONDS
    )

    for path in TMP_ROOT.iterdir():

        try:

            if (
                path.is_dir()
                and path.stat().st_mtime
                < cutoff
            ):

                shutil.rmtree(
                    path,
                    ignore_errors=True,
                )

        except OSError:
            pass


# ============================================================================
# FRAME PREPROCESSING
# ============================================================================

class ResizeWithPadding:

    def __init__(
        self,
        image_size: int = 224,
        fill: int = 0,
    ):

        self.image_size = image_size
        self.fill = fill

    def __call__(
        self,
        image: Image.Image,
    ):

        width, height = image.size

        if width <= 0 or height <= 0:
            raise ValueError(
                "Invalid image dimensions."
            )

        scale = min(
            self.image_size / width,
            self.image_size / height,
        )

        new_width = max(
            1,
            round(width * scale),
        )

        new_height = max(
            1,
            round(height * scale),
        )

        image = image.resize(
            (
                new_width,
                new_height,
            ),
            Image.Resampling.BILINEAR,
        )

        left = (
            self.image_size
            - new_width
        ) // 2

        top = (
            self.image_size
            - new_height
        ) // 2

        right = (
            self.image_size
            - new_width
            - left
        )

        bottom = (
            self.image_size
            - new_height
            - top
        )

        return transforms.functional.pad(
            image,
            [
                left,
                top,
                right,
                bottom,
            ],
            fill=self.fill,
        )


FRAME_TRANSFORM = transforms.Compose(
    [
        ResizeWithPadding(
            IMAGE_SIZE
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


# ============================================================================
# MODEL LOADING
# ============================================================================

def get_ball_detector():

    global _ball_detector

    with _ball_detector_lock:

        if _ball_detector is None:

            if not BALL_MODEL.exists():

                raise FileNotFoundError(
                    f"Ball model not found:\n"
                    f"{BALL_MODEL}"
                )

            _ball_detector = (
                BallDetector()
            )

        return _ball_detector


def get_shot_model():

    global _shot_model

    with _shot_model_lock:

        if _shot_model is not None:
            return _shot_model

        if not SHOT_MODEL.exists():

            raise FileNotFoundError(
                f"Shot checkpoint not found:\n"
                f"{SHOT_MODEL}"
            )

        print(
            "[SHOT] Loading "
            "EfficientNet-B0 + Transformer...",
            flush=True,
        )

        model = ImprovedSOTAModel(
            num_classes=len(
                SHOT_CLASSES
            ),
            temporal_dim=256,
            n_frames=N_FRAMES,
        )

        checkpoint = torch.load(
            SHOT_MODEL,
            map_location=DEVICE,
            weights_only=False,
        )

        if (
            isinstance(
                checkpoint,
                dict,
            )
            and "state_dict"
            in checkpoint
        ):

            state_dict = dict(
                checkpoint[
                    "state_dict"
                ]
            )

        else:

            state_dict = checkpoint

        if not isinstance(
            state_dict,
            dict,
        ):

            raise ValueError(
                "Unsupported checkpoint format."
            )

        state_dict.pop(
            "class_weights",
            None,
        )

        cleaned = {}

        for key, value in (
            state_dict.items()
        ):

            if key.startswith(
                "model."
            ):

                key = key[
                    len("model.") :
                ]

            cleaned[key] = value

        model.load_state_dict(
            cleaned,
            strict=True,
        )

        model.to(DEVICE)
        model.eval()

        _shot_model = model

        print(
            f"[SHOT] Transformer loaded "
            f"on {DEVICE}",
            flush=True,
        )

        return model


# ============================================================================
# SHOT CLASSIFICATION
# ============================================================================

def sample_video_frames(
    cap: cv2.VideoCapture,
    start_frame: int,
    end_frame: int,
):

    if end_frame < start_frame:
        raise ValueError(
            "Invalid frame range."
        )

    indices = np.linspace(
        start_frame,
        end_frame,
        N_FRAMES,
        dtype=np.int64,
    )

    tensors = []

    for frame_index in indices:

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            int(frame_index),
        )

        ok, frame = cap.read()

        if not ok or frame is None:
            continue

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        image = Image.fromarray(
            rgb
        )

        tensors.append(
            FRAME_TRANSFORM(image)
        )

    if len(tensors) != N_FRAMES:

        raise RuntimeError(
            f"Expected {N_FRAMES} frames, "
            f"read {len(tensors)}."
        )

    return torch.stack(
        tensors,
        dim=0,
    ).unsqueeze(0)


@torch.inference_mode()
def predict_frames(
    model,
    frames,
):

    frames = frames.to(
        DEVICE,
        non_blocking=(
            DEVICE.type == "cuda"
        ),
    )

    logits = model(frames)

    probabilities = F.softmax(
        logits,
        dim=1,
    )

    confidence, index = (
        torch.max(
            probabilities,
            dim=1,
        )
    )

    class_index = int(
        index.item()
    )

    result = {

        "prediction":
            SHOT_CLASSES[
                class_index
            ],

        "confidence":
            round(
                float(
                    confidence.item()
                ),
                6,
            ),

        "class_probabilities": {
            name: round(
                float(
                    probabilities[
                        0,
                        i,
                    ].item()
                ),
                6,
            )
            for i, name in enumerate(
                SHOT_CLASSES
            )
        },
    }

    return result


# ============================================================================
# CLIP RANGE
# ============================================================================

def make_clip_range(
    first_frame: int,
    last_frame: int,
    total_frames: int,
):

    start = max(
        0,
        first_frame
        - PRE_ROLL_FRAMES,
    )

    end = min(
        total_frames - 1,
        last_frame
        + POST_ROLL_FRAMES,
    )

    length = (
        end - start + 1
    )

    if length < MIN_CLIP_FRAMES:

        missing = (
            MIN_CLIP_FRAMES
            - length
        )

        left = missing // 2
        right = (
            missing - left
        )

        start = max(
            0,
            start - left,
        )

        end = min(
            total_frames - 1,
            end + right,
        )

        length = (
            end - start + 1
        )

        if (
            length
            < MIN_CLIP_FRAMES
        ):

            if start == 0:

                end = min(
                    total_frames - 1,
                    MIN_CLIP_FRAMES - 1,
                )

            elif (
                end
                == total_frames - 1
            ):

                start = max(
                    0,
                    total_frames
                    - MIN_CLIP_FRAMES,
                )

    if (
        end - start + 1
        > MAX_CLIP_FRAMES
    ):

        center = (
            start + end
        ) // 2

        half = (
            MAX_CLIP_FRAMES
            // 2
        )

        start = max(
            0,
            center - half,
        )

        end = min(
            total_frames - 1,
            start
            + MAX_CLIP_FRAMES
            - 1,
        )

    return start, end


# ============================================================================
# DELIVERY FINALIZATION
# ============================================================================

def finalise_delivery(
    positions,
    H,
    fps,
    frame_height,
    frame_width,
    delivery_id,
    deliveries,
):

    if (
        len(positions)
        < MIN_DELIVERY_FRAMES
    ):
        return False

    trajectory = [
        point
        for point, _ in positions
    ]

    bounce_idx = detect_bounce(
        trajectory,
        H,
    )

    if bounce_idx is None:

        bounce_x, bounce_y = max(
            trajectory,
            key=lambda point: point[1],
        )

        bounce_idx = trajectory.index(
            (
                bounce_x,
                bounce_y,
            )
        )

    else:

        bounce_x, bounce_y = (
            trajectory[bounce_idx]
        )

    early = positions[
        :MAX_EARLY_POSITIONS
    ]

    speed = compute_release_speed(
        early,
        fps,
        H,
    )

    record = {

        "ball":
            int(delivery_id),

        "frame_start":
            int(positions[0][1]),

        "frame_end":
            int(positions[-1][1]),

        "speed":
            speed,

        "length":
            classify_length(
                bounce_y,
                frame_height,
            ),

        "line":
            classify_line(
                bounce_x,
                frame_width,
            ),

        "swing":
            estimate_swing(
                trajectory,
                bounce_idx,
                H,
            ),

        "release_angle":
            compute_release_angle(
                positions
            ),

        "bounce_angle":
            compute_bounce_angle(
                trajectory,
                bounce_idx,
                H,
            ),

        "bounce_x":
            round(
                bounce_x
                / frame_width,
                3,
            ),

        "bounce_y":
            round(
                bounce_y
                / frame_height,
                3,
            ),

        "tracking_trajectory": [
            {
                "x": int(x),
                "y": int(y),
                "frame": int(frame),
            }
            for (x, y), frame
            in positions
        ],
    }

    deliveries.append(
        record
    )

    return True


# ============================================================================
# MAIN VIDEO PROCESSOR
# ============================================================================

def process_video(
    task_id,
    temp_dir,
    video_path,
    output_path,
):

    started = time.monotonic()

    cap = None
    writer = None

    try:

        update_task(
            task_id,
            2,
            "Opening video...",
        )

        cap = cv2.VideoCapture(
            str(video_path)
        )

        if not cap.isOpened():

            raise ValueError(
                "Could not open video."
            )

        total_frames = int(
            cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        fps = get_fps(cap)

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

        if (
            total_frames <= 0
            or fps <= 0
            or width <= 0
            or height <= 0
        ):

            raise ValueError(
                "Video metadata is invalid."
            )

        H = build_homography(
            width,
            height,
        )

        detector = (
            get_ball_detector()
        )

        # ------------------------------------------------------------
        # Output writer
        # ------------------------------------------------------------

        fourcc = (
            cv2.VideoWriter_fourcc(
                *"mp4v"
            )
        )

        writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            fps,
            (
                width,
                height,
            ),
        )

        if not writer.isOpened():

            raise RuntimeError(
                "Could not create "
                "processed video."
            )

        tracker = BallTracker()

        frame_number = 0

        delivery_id = 1

        last_delivery_frame = (
            -MIN_FRAMES_BETWEEN_DELIVERIES
        )

        deliveries = []

        trajectory_data = deque(
            maxlen=MAX_TRAJECTORY_POINTS
        )

        update_task(
            task_id,
            5,
            "Tracking ball trajectory...",
        )

        # ============================================================
        # TRACKING
        # ============================================================

        while True:

            if is_cancelled(
                task_id
            ):
                return

            ok, frame = cap.read()

            if not ok:
                break

            frame_number += 1

            detection = (
                detector.detect(frame)
            )

            position = tracker.update(
                detection,
                frame_number,
            )

            if position is not None:

                trajectory_data.append(
                    {
                        "frame":
                            frame_number,
                        "x":
                            int(
                                position[0]
                            ),
                        "y":
                            int(
                                position[1]
                            ),
                    }
                )

            # --------------------------------------------------------
            # Delivery finished
            # --------------------------------------------------------

            if (
                tracker.is_lost
                and tracker.in_delivery
            ):

                positions = list(
                    tracker.delivery_positions
                )

                if (
                    frame_number
                    - last_delivery_frame
                    >= MIN_FRAMES_BETWEEN_DELIVERIES
                ):

                    added = (
                        finalise_delivery(
                            positions,
                            H,
                            fps,
                            height,
                            width,
                            delivery_id,
                            deliveries,
                        )
                    )

                    if added:

                        last_delivery_frame = (
                            frame_number
                        )

                        delivery_id += 1

                tracker.reset_delivery()

            # --------------------------------------------------------
            # Draw tracking trail
            # --------------------------------------------------------

            trail = tracker.traj_points

            if len(trail) >= 2:

                draw_trajectory_trail(
                    frame,
                    trail,
                )

            writer.write(frame)

            if (
                frame_number % 10
                == 0
            ):

                progress = int(
                    5
                    + (
                        frame_number
                        / total_frames
                    )
                    * 50
                )

                update_task(
                    task_id,
                    progress,
                    (
                        "Tracking video "
                        f"{frame_number}/"
                        f"{total_frames}"
                    ),
                )

        # ============================================================
        # FINAL DELIVERY
        # ============================================================

        if (
            not is_cancelled(task_id)
            and tracker.in_delivery
        ):

            positions = list(
                tracker.delivery_positions
            )

            if len(positions) >= (
                MIN_DELIVERY_FRAMES
            ):

                if (
                    frame_number
                    - last_delivery_frame
                    >= MIN_FRAMES_BETWEEN_DELIVERIES
                ):

                    added = (
                        finalise_delivery(
                            positions,
                            H,
                            fps,
                            height,
                            width,
                            delivery_id,
                            deliveries,
                        )
                    )

                    if added:
                        delivery_id += 1

        cap.release()
        cap = None

        writer.release()
        writer = None

        if is_cancelled(
            task_id
        ):

            with suppress(
                OSError
            ):
                output_path.unlink()

            update_task(
                task_id,
                0,
                "Cancelled",
            )

            return

        if not deliveries:

            raise ValueError(
                "No valid ball deliveries "
                "were detected."
            )

        # ============================================================
        # SHOT CLASSIFICATION
        # ============================================================

        model = get_shot_model()

        update_task(
            task_id,
            55,
            (
                f"Classifying "
                f"{len(deliveries)} deliveries..."
            ),
        )

        classify_cap = cv2.VideoCapture(
            str(video_path)
        )

        if not classify_cap.isOpened():

            raise RuntimeError(
                "Could not reopen video "
                "for classification."
            )

        try:

            total_deliveries = (
                len(deliveries)
            )

            for index, delivery in enumerate(
                deliveries,
                start=1,
            ):

                if is_cancelled(
                    task_id
                ):
                    return

                clip_start, clip_end = (
                    make_clip_range(
                        delivery[
                            "frame_start"
                        ],
                        delivery[
                            "frame_end"
                        ],
                        total_frames,
                    )
                )

                frames = (
                    sample_video_frames(
                        classify_cap,
                        clip_start,
                        clip_end,
                    )
                )

                shot = predict_frames(
                    model,
                    frames,
                )

                delivery[
                    "clip_start"
                ] = clip_start

                delivery[
                    "clip_end"
                ] = clip_end

                delivery[
                    "shot"
                ] = shot

                progress = (
                    55
                    + int(
                        (
                            index
                            / total_deliveries
                        )
                        * 40
                    )
                )

                update_task(
                    task_id,
                    progress,
                    (
                        f"Classifying delivery "
                        f"{index}/"
                        f"{total_deliveries}"
                    ),
                )

        finally:

            classify_cap.release()

        if is_cancelled(
            task_id
        ):

            with suppress(
                OSError
            ):
                output_path.unlink()

            update_task(
                task_id,
                0,
                "Cancelled",
            )

            return

        # ============================================================
        # RESULT
        # ============================================================

        shot_distribution = {
            name: 0
            for name in SHOT_CLASSES
        }

        confidence_values = []

        timeline = []

        for delivery in deliveries:

            shot = delivery[
                "shot"
            ]

            prediction = shot[
                "prediction"
            ]

            shot_distribution[
                prediction
            ] += 1

            confidence = float(
                shot[
                    "confidence"
                ]
            )

            confidence_values.append(
                confidence
            )

            timeline.append(
                {
                    "frame":
                        delivery[
                            "ball"
                        ],

                    "confidence":
                        round(
                            confidence
                            * 100,
                            2,
                        ),

                    "prediction":
                        prediction,
                }
            )

        result = {

            "prediction":
                None,

            "confidence":
                0,

            "class_probabilities":
                {},

            "timeline":
                timeline,

            "clips_processed":
                len(deliveries),

            "deliveries":
                deliveries,

            "shot_distribution":
                shot_distribution,

            "average_shot_confidence":
                round(
                    statistics.fmean(
                        confidence_values
                    ),
                    6,
                )
                if confidence_values
                else 0,

            "trajectory":
                list(
                    trajectory_data
                ),

            "video_url":
                f"/outputs/"
                f"{output_path.name}",

            "processing_time":
                round(
                    time.monotonic()
                    - started,
                    2,
                ),

            "device":
                str(DEVICE),
        }

        with _tasks_lock:

            state = _tasks.get(
                task_id
            )

            if state:

                state[
                    "output_path"
                ] = str(
                    output_path
                )

        update_task(
            task_id,
            100,
            "Complete",
            result=result,
        )

        print(
            f"[DONE] {task_id}: "
            f"{len(deliveries)} deliveries",
            flush=True,
        )

    except Exception as exc:

        import traceback

        traceback.print_exc()

        with suppress(
            OSError
        ):
            if output_path.exists():
                output_path.unlink()

        if is_cancelled(
            task_id
        ):

            update_task(
                task_id,
                0,
                "Cancelled",
            )

        else:

            update_task(
                task_id,
                -1,
                "Error",
                error=str(exc),
            )

    finally:

        if cap is not None:
            with suppress(Exception):
                cap.release()

        if writer is not None:
            with suppress(Exception):
                writer.release()

        shutil.rmtree(
            temp_dir,
            ignore_errors=True,
        )

        if DEVICE.type == "cuda":

            with suppress(Exception):
                torch.cuda.empty_cache()


# ============================================================================
# WORKER
# ============================================================================

def run_task(
    task_id,
    temp_dir,
    video_path,
    output_path,
):

    acquired = False

    try:

        while not acquired:

            if is_cancelled(
                task_id
            ):

                update_task(
                    task_id,
                    0,
                    "Cancelled",
                )

                return

            acquired = (
                PROCESS_LOCK.acquire(
                    timeout=0.25
                )
            )

        process_video(
            task_id,
            temp_dir,
            video_path,
            output_path,
        )

    finally:

        if acquired:
            PROCESS_LOCK.release()


# ============================================================================
# UPLOAD
# ============================================================================

async def create_processing_task(
    file: UploadFile,
):

    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="No file provided.",
        )

    allowed_extensions = {
        ".mp4",
        ".avi",
        ".mov",
        ".webm",
        ".m4v",
    }

    extension = (
        Path(
            file.filename
        ).suffix.lower()
    )

    if (
        extension
        not in allowed_extensions
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported video format."
            ),
        )

    task_id = create_task()

    temp_dir = Path(
        tempfile.mkdtemp(
            prefix=(
                f"cricket_{task_id}_"
            ),
            dir=str(TMP_ROOT),
        )
    )

    video_path = (
        temp_dir
        / f"input{extension}"
    )

    output_path = (
        OUTPUT_DIR
        / f"{task_id}.mp4"
    )

    try:

        total_bytes = 0

        with video_path.open(
            "wb"
        ) as buffer:

            while True:

                chunk = await file.read(
                    1024 * 1024
                )

                if not chunk:
                    break

                total_bytes += len(
                    chunk
                )

                if (
                    total_bytes
                    > MAX_UPLOAD_BYTES
                ):

                    raise HTTPException(
                        status_code=413,
                        detail=(
                            "Video exceeds "
                            "500 MB limit."
                        ),
                    )

                buffer.write(chunk)

        await file.close()

        if total_bytes == 0:

            raise HTTPException(
                status_code=400,
                detail="Video is empty.",
            )

        with _tasks_lock:

            _tasks[task_id][
                "temp_dir"
            ] = str(temp_dir)

            _tasks[task_id][
                "output_path"
            ] = str(output_path)

        thread = threading.Thread(
            target=run_task,
            args=(
                task_id,
                temp_dir,
                video_path,
                output_path,
            ),
            daemon=True,
            name=(
                f"cricket-{task_id[:8]}"
            ),
        )

        thread.start()

        return task_id

    except Exception:

        await file.close()

        shutil.rmtree(
            temp_dir,
            ignore_errors=True,
        )

        with _tasks_lock:
            _tasks.pop(
                task_id,
                None,
            )

        raise


# ============================================================================
# API
# ============================================================================

@app.post("/api/predict")
async def predict(
    video: UploadFile = File(...),
):

    task_id = (
        await create_processing_task(
            video
        )
    )

    return {
        "task_id": task_id
    }


@app.delete(
    "/api/cancel/{task_id}"
)
async def cancel(
    task_id: str,
):

    if not request_cancel(
        task_id
    ):

        return JSONResponse(
            status_code=404,
            content={
                "detail":
                    "Task not found."
            },
        )

    return {
        "status": "success",
        "message":
            "Cancellation requested.",
    }


@app.get(
    "/api/progress/{task_id}"
)
async def progress(
    task_id: str,
    request: Request,
):

    if (
        public_task_state(
            task_id
        )
        is None
    ):

        return StreamingResponse(
            iter(
                [
                    "data: "
                    + json.dumps(
                        {
                            "progress":
                                -1,
                            "status":
                                "Task not found",
                            "cancelled":
                                False,
                        }
                    )
                    + "\n\n"
                ]
            ),
            media_type=(
                "text/event-stream"
            ),
        )

    async def generator():

        while True:

            state = (
                public_task_state(
                    task_id
                )
            )

            if state is None:

                yield (
                    "data: "
                    + json.dumps(
                        {
                            "progress":
                                -1,
                            "status":
                                "Task expired",
                            "cancelled":
                                False,
                        }
                    )
                    + "\n\n"
                )

                return

            yield (
                "data: "
                + json.dumps(
                    state
                )
                + "\n\n"
            )

            if (
                state["progress"]
                == 100
                or state["progress"]
                < 0
                or state["cancelled"]
                or state["status"]
                in {
                    "Cancelled",
                    "Error",
                }
            ):

                return

            if await request.is_disconnected():
                return

            await asyncio.sleep(
                0.25
            )

    return StreamingResponse(
        generator(),
        media_type=(
            "text/event-stream"
        ),
        headers={
            "Cache-Control":
                "no-cache",
            "Connection":
                "keep-alive",
            "X-Accel-Buffering":
                "no",
        },
    )


# ============================================================================
# HEALTH
# ============================================================================

@app.get("/api/health")
async def health():

    return {

        "status":
            "CricShot + CricketTracker Online",

        "device":
            str(DEVICE),

        "ball_model_exists":
            BALL_MODEL.exists(),

        "shot_model_exists":
            SHOT_MODEL.exists(),

        "ball_model_loaded":
            _ball_detector is not None,

        "shot_model_loaded":
            _shot_model is not None,
    }


# ============================================================================
# LEGACY BALL TRACKING API
# ============================================================================

@app.post("/upload")
async def legacy_upload(
    file: UploadFile = File(...),
):

    task_id = (
        await create_processing_task(
            file
        )
    )

    return {
        "message":
            "Processing started",
        "task_id":
            task_id,
    }


@app.get("/status")
async def legacy_status():

    with _tasks_lock:

        if not _tasks:

            return {
                "status":
                    "idle",
                "progress":
                    0,
                "result":
                    None,
                "processing_time":
                    0,
                "error_message":
                    None,
            }

        task_id = max(
            _tasks,
            key=lambda key:
                _tasks[key].get(
                    "created_at",
                    0,
                ),
        )

        state = _tasks[
            task_id
        ]

        result = state.get(
            "result"
        )

        wrapped = None

        if result is not None:

            wrapped = {
                "video_url":
                    result.get(
                        "video_url"
                    ),
                "json_data":
                    result,
            }

        return {

            "task_id":
                task_id,

            "status":
                str(
                    state.get(
                        "status",
                        "idle",
                    )
                ).lower(),

            "progress":
                state.get(
                    "progress",
                    0,
                ),

            "result":
                wrapped,

            "processing_time":
                (
                    result.get(
                        "processing_time",
                        0,
                    )
                    if result
                    else 0
                ),

            "error_message":
                state.get(
                    "error"
                ),
        }


# ============================================================================
# VISUALIZATION
# ============================================================================

@app.get("/api/visualization")
async def visualization():

    return {
        "status":
            "Visualization data available"
    }


# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(
            os.getenv(
                "PORT",
                "8000",
            )
        ),
        reload=False,
    )