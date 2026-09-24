import asyncio
import json
import shutil
import tempfile
import uuid
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image
from torchvision import transforms

from src.pipeline.efficientnet_gru import EfficientNetGRU
from src.pipeline.extract_frames import extract_frames
from src.segmentation.auto_clipper import AutoClipper


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

BALL_MODEL = MODELS_DIR / "Ball_Detection_Model.pt"
BAT_MODEL = MODELS_DIR / "Bat_Detection_Model.pt"
CRICKET_MODEL = MODELS_DIR / "cricket_model.ckpt"

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

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="CricShot API",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# EXACT VALIDATION/TEST PREPROCESSING
# ============================================================

class ResizeWithPadding:
    """
    Preserve aspect ratio and pad to 224x224.
    This matches the preprocessing used during training.
    """

    def __init__(self, image_size=224, fill=0):
        self.image_size = image_size
        self.fill = fill

    def __call__(self, img):
        width, height = img.size

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

        img = img.resize(
            (new_width, new_height),
            Image.Resampling.BILINEAR,
        )

        left = (self.image_size - new_width) // 2
        top = (self.image_size - new_height) // 2

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
            img,
            [left, top, right, bottom],
            fill=self.fill,
        )


FRAME_TRANSFORM = transforms.Compose([
    ResizeWithPadding(IMAGE_SIZE),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ============================================================
# TASK MANAGEMENT
# ============================================================

active_tasks = {}


def update_task(
    task_id,
    progress,
    status,
    **extra,
):
    """
    Update task state while preserving cancellation state.
    """

    previous = active_tasks.get(
        task_id,
        {},
    )

    active_tasks[task_id] = {
        "progress": progress,
        "status": status,
        "cancelled": previous.get(
            "cancelled",
            False,
        ),
        **extra,
    }


def is_cancelled(task_id):
    return active_tasks.get(
        task_id,
        {},
    ).get(
        "cancelled",
        False,
    )


# ============================================================
# MODEL LOADING
# ============================================================

def load_cricket_model():

    if not CRICKET_MODEL.exists():
        raise FileNotFoundError(
            f"Cricket model not found: {CRICKET_MODEL}"
        )

    print(
        "[*] Loading EfficientNet-B0 + BiGRU...",
        flush=True,
    )

    model = EfficientNetGRU(
        num_classes=len(SHOT_CLASSES),
        temporal_dim=256,
        n_frames=N_FRAMES,
    )

    checkpoint = torch.load(
        CRICKET_MODEL,
        map_location=DEVICE,
        weights_only=False,
    )

    if (
        isinstance(checkpoint, dict)
        and "state_dict" in checkpoint
    ):
        state_dict = checkpoint["state_dict"]

        # Lightning checkpoint contains this extra item.
        state_dict.pop(
            "class_weights",
            None,
        )

        cleaned_state_dict = {}

        for key, value in state_dict.items():

            if key.startswith("model."):
                key = key[len("model."):]

            cleaned_state_dict[key] = value

        state_dict = cleaned_state_dict

    else:
        state_dict = checkpoint

    # The architecture/checkpoint were already verified.
    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model.to(DEVICE)
    model.eval()

    print(
        f"[+] Cricket model loaded on {DEVICE}",
        flush=True,
    )

    return model


# ============================================================
# FRAME LOADING
# ============================================================

def load_frames(frames_dir):
    """
    Load exactly 30 extracted frames.

    extract_frames.py creates:
        frame_00.png
        ...
        frame_29.png
    """

    frames_dir = Path(frames_dir)

    frame_paths = sorted(
        frames_dir.glob("frame_*.png")
    )

    if len(frame_paths) != N_FRAMES:
        raise ValueError(
            f"Expected {N_FRAMES} frames, "
            f"found {len(frame_paths)} in {frames_dir}"
        )

    frames = []

    for frame_path in frame_paths:

        image = Image.open(
            frame_path
        ).convert("RGB")

        tensor = FRAME_TRANSFORM(
            image
        )

        frames.append(tensor)

    frames = torch.stack(
        frames,
        dim=0,
    )

    # (30, 3, 224, 224)
    frames = frames.unsqueeze(0)

    # (1, 30, 3, 224, 224)
    return frames


# ============================================================
# MODEL PREDICTION
# ============================================================

@torch.inference_mode()
def predict_clip(
    model,
    frames,
):

    frames = frames.to(
        DEVICE
    )

    logits = model(
        frames
    )

    probabilities = F.softmax(
        logits,
        dim=1,
    )

    confidence, class_index = torch.max(
        probabilities,
        dim=1,
    )

    class_index = class_index.item()
    confidence = confidence.item()

    prediction = SHOT_CLASSES[
        class_index
    ]

    class_probabilities = {}

    for i, class_name in enumerate(
        SHOT_CLASSES
    ):
        class_probabilities[
            class_name
        ] = round(
            probabilities[
                0,
                i,
            ].item(),
            6,
        )

    return {
        "prediction": prediction,
        "confidence": round(
            confidence,
            6,
        ),
        "class_probabilities": (
            class_probabilities
        ),
    }


# ============================================================
# VIDEO VALIDATION
# ============================================================

def validate_video(video_path):

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():
        return False

    frame_count = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    cap.release()

    return frame_count > 0


# ============================================================
# MAIN BACKGROUND PROCESS
# ============================================================

def process_video(
    task_id,
    temp_dir,
    video_path,
    clips_dir,
    frames_dir,
):

    try:

        # ----------------------------------------------------
        # INITIAL STATE
        # ----------------------------------------------------

        update_task(
            task_id,
            5,
            "Validating uploaded video...",
        )

        if is_cancelled(task_id):
            return

        if not validate_video(
            video_path
        ):
            raise ValueError(
                "Could not open uploaded video."
            )

        # ----------------------------------------------------
        # CHECK MODELS
        # ----------------------------------------------------

        if not BALL_MODEL.exists():
            raise FileNotFoundError(
                f"Ball model not found: {BALL_MODEL}"
            )

        if not BAT_MODEL.exists():
            raise FileNotFoundError(
                f"Bat model not found: {BAT_MODEL}"
            )

        if not CRICKET_MODEL.exists():
            raise FileNotFoundError(
                f"Cricket model not found: {CRICKET_MODEL}"
            )

        # ----------------------------------------------------
        # AUTO CLIPPER
        # ----------------------------------------------------

        update_task(
            task_id,
            10,
            "Loading ball and bat detection models...",
        )

        if is_cancelled(task_id):
            return

        clipper = AutoClipper(
            ball_model_path=str(
                BALL_MODEL
            ),
            bat_model_path=str(
                BAT_MODEL
            ),
            output_dir=str(
                clips_dir
            ),
            confidence=0.30,
            use_ocr=True,
            ocr_gpu=False,
        )

        update_task(
            task_id,
            15,
            "Extracting delivery clips...",
        )

        if is_cancelled(task_id):
            return

        extracted_clips = (
            clipper.process_match(
                str(video_path),
                clip_duration_sec=1.0,
                active_tasks=active_tasks,
                task_id=task_id,
                temp_dir=temp_dir,
            )
        )

        if is_cancelled(task_id):
            print(
                f"[INFO] Task {task_id} cancelled "
                f"after delivery extraction.",
                flush=True,
            )
            return

        # ----------------------------------------------------
        # NO DELIVERY
        # ----------------------------------------------------

        if not extracted_clips:

            update_task(
                task_id,
                100,
                "Complete",
                result={
                    "prediction": None,
                    "confidence": 0,
                    "class_probabilities": {},
                    "timeline": [],
                    "clips_processed": 0,
                    "message": (
                        "No valid delivery was detected "
                        "in the uploaded video."
                    ),
                },
            )

            return

        # ----------------------------------------------------
        # LOAD CRICKET MODEL
        # ----------------------------------------------------

        update_task(
            task_id,
            50,
            "Loading EfficientNet-B0 + BiGRU...",
        )

        if is_cancelled(task_id):
            return

        model = load_cricket_model()

        predictions = []

        total_clips = len(
            extracted_clips
        )

        # ----------------------------------------------------
        # CLASSIFY EACH DELIVERY
        # ----------------------------------------------------

        for clip_number, clip_path in enumerate(
            extracted_clips,
            start=1,
        ):

            if is_cancelled(task_id):

                print(
                    f"[INFO] Task {task_id} "
                    f"cancelled during inference.",
                    flush=True,
                )

                return

            clip_path = Path(
                clip_path
            )

            delivery_id = (
                clip_path.stem
            )

            current_frames_dir = (
                frames_dir / delivery_id
            )

            current_frames_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            # -----------------------------------------------
            # Progress
            # -----------------------------------------------

            progress = (
                50
                + int(
                    (
                        clip_number
                        / total_clips
                    )
                    * 35
                )
            )

            update_task(
                task_id,
                progress,
                (
                    "Extracting 30 uniform frames "
                    f"(Clip {clip_number}/{total_clips})..."
                ),
            )

            if is_cancelled(task_id):
                return

            # -----------------------------------------------
            # Extract frames
            # -----------------------------------------------

            success = extract_frames(
                clip_path,
                current_frames_dir,
            )

            if not success:

                print(
                    f"[WARNING] Frame extraction failed: "
                    f"{clip_path}",
                    flush=True,
                )

                continue

            if is_cancelled(task_id):
                return

            # -----------------------------------------------
            # Load + preprocess frames
            # -----------------------------------------------

            frames = load_frames(
                current_frames_dir
            )

            if is_cancelled(task_id):
                return

            # -----------------------------------------------
            # Model inference
            # -----------------------------------------------

            update_task(
                task_id,
                progress,
                (
                    "BiGRU temporal analysis "
                    f"(Clip {clip_number}/{total_clips})..."
                ),
            )

            result = predict_clip(
                model,
                frames,
            )

            result["delivery_id"] = (
                delivery_id
            )

            result["clip_number"] = (
                clip_number
            )

            predictions.append(
                result
            )

        # ----------------------------------------------------
        # CANCELLED
        # ----------------------------------------------------

        if is_cancelled(task_id):
            return

        # ----------------------------------------------------
        # NO CLASSIFIABLE CLIPS
        # ----------------------------------------------------

        if not predictions:
            raise ValueError(
                "Delivery clips were detected, "
                "but no clip could be classified."
            )

        # ----------------------------------------------------
        # AVERAGE CLASS PROBABILITIES
        # ----------------------------------------------------

        update_task(
            task_id,
            90,
            "Finalizing classification...",
        )

        averaged_probabilities = {}

        for class_name in SHOT_CLASSES:

            values = [
                prediction[
                    "class_probabilities"
                ][class_name]
                for prediction in predictions
            ]

            averaged_probabilities[
                class_name
            ] = round(
                float(
                    np.mean(values)
                ),
                6,
            )

        # ----------------------------------------------------
        # FINAL CLASS
        # ----------------------------------------------------

        final_class_index = int(
            np.argmax(
                [
                    averaged_probabilities[
                        class_name
                    ]
                    for class_name in SHOT_CLASSES
                ]
            )
        )

        final_prediction = (
            SHOT_CLASSES[
                final_class_index
            ]
        )

        final_confidence = (
            averaged_probabilities[
                final_prediction
            ]
        )

        # ----------------------------------------------------
        # TIMELINE
        # ----------------------------------------------------

        timeline = []

        for prediction in predictions:

            timeline.append({
                "frame": prediction[
                    "clip_number"
                ],
                "confidence": round(
                    prediction[
                        "confidence"
                    ] * 100,
                    2,
                ),
                "prediction": prediction[
                    "prediction"
                ],
            })

        # ----------------------------------------------------
        # FINAL RESULT
        # ----------------------------------------------------

        result = {
            "prediction": final_prediction,
            "confidence": round(
                final_confidence,
                6,
            ),
            "class_probabilities": (
                averaged_probabilities
            ),
            "timeline": timeline,
            "clips_processed": len(
                predictions
            ),
        }

        update_task(
            task_id,
            100,
            "Complete",
            result=result,
        )

        print(
            f"[+] Task {task_id} complete: "
            f"{final_prediction} "
            f"({final_confidence * 100:.2f}%)",
            flush=True,
        )

    except Exception as e:

        import traceback

        traceback.print_exc()

        print(
            f"[ERROR] Task {task_id}: "
            f"{type(e).__name__}: {e}",
            flush=True,
        )

        update_task(
            task_id,
            -1,
            "Error",
            error=str(e),
        )

    finally:

        if (
            temp_dir is not None
            and temp_dir.exists()
        ):
            shutil.rmtree(
                temp_dir,
                ignore_errors=True,
            )


# ============================================================
# PREDICT
# ============================================================

@app.post("/api/predict")
async def predict_shot(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
):

    task_id = str(
        uuid.uuid4()
    )

    active_tasks[task_id] = {
        "progress": 0,
        "status": "Uploading video...",
        "cancelled": False,
    }

    temp_dir = None

    try:

        # ----------------------------------------------------
        # Temporary directory
        # ----------------------------------------------------

        temp_dir = Path(
            tempfile.mkdtemp(
                prefix="cricket_"
            )
        )

        video_path = (
            temp_dir / "input_video.mp4"
        )

        clips_dir = (
            temp_dir / "delivery_clips"
        )

        frames_dir = (
            temp_dir / "frames"
        )

        clips_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        frames_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ----------------------------------------------------
        # Save upload
        # ----------------------------------------------------

        await video.seek(0)

        with open(
            video_path,
            "wb",
        ) as buffer:

            while True:

                chunk = await video.read(
                    1024 * 1024
                )

                if not chunk:
                    break

                buffer.write(chunk)

        # ----------------------------------------------------
        # Schedule background processing
        # ----------------------------------------------------

        background_tasks.add_task(
            process_video,
            task_id,
            temp_dir,
            video_path,
            clips_dir,
            frames_dir,
        )

        return {
            "task_id": task_id
        }

    except Exception as e:

        if (
            temp_dir is not None
            and temp_dir.exists()
        ):
            shutil.rmtree(
                temp_dir,
                ignore_errors=True,
            )

        update_task(
            task_id,
            -1,
            "Error",
            error=str(e),
        )

        return JSONResponse(
            status_code=500,
            content={
                "detail": str(e)
            },
        )


# ============================================================
# CANCEL
# ============================================================

@app.delete("/api/cancel/{task_id}")
async def cancel_task(
    task_id: str,
):

    print(
        f"[CANCEL] Received cancellation "
        f"request for {task_id}",
        flush=True,
    )

    if task_id not in active_tasks:

        return JSONResponse(
            status_code=404,
            content={
                "detail": "Task not found."
            },
        )

    active_tasks[
        task_id
    ]["cancelled"] = True

    active_tasks[
        task_id
    ]["status"] = "Cancelling..."

    print(
        f"[CANCEL] Task {task_id} marked "
        f"for cancellation.",
        flush=True,
    )

    return {
        "status": "success",
        "message": (
            f"Task {task_id} marked for cancellation."
        ),
    }


# ============================================================
# PROGRESS / SSE
# ============================================================

@app.get("/api/progress/{task_id}")
async def get_progress(
    task_id: str,
):

    async def event_generator():

        while True:

            if task_id not in active_tasks:

                yield (
                    "data: "
                    + json.dumps({
                        "progress": -1,
                        "status": "Task not found",
                    })
                    + "\n\n"
                )

                break

            state = active_tasks[
                task_id
            ]

            yield (
                "data: "
                + json.dumps(state)
                + "\n\n"
            )

            progress = state.get(
                "progress",
                0,
            )

            if (
                progress == 100
                or progress < 0
                or state.get("cancelled")
            ):
                break

            await asyncio.sleep(
                0.2
            )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ============================================================
# HEALTH
# ============================================================

@app.get("/api/health")
async def health_check():

    return {
        "status": "CricShot Pipeline Online",
        "device": str(DEVICE),
    }


# ============================================================
# VISUALIZATION
# ============================================================

@app.get("/api/visualization")
async def get_visualization():

    return {
        "status": "Visualization data available"
    }


# ============================================================
# LOCAL DEVELOPMENT
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )