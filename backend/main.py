import os
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="CricShot API",
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

FRAME_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# MODEL LOADING
# ============================================================

def load_cricket_model():
    if not CRICKET_MODEL.exists():
        raise FileNotFoundError(
            f"Cricket model not found: {CRICKET_MODEL}"
        )

    model = EfficientNetGRU(
        num_classes=len(SHOT_CLASSES)
    )

    checkpoint = torch.load(
        CRICKET_MODEL,
        map_location=DEVICE,
        weights_only=False
    )

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:

        state_dict = checkpoint["state_dict"]

        cleaned_state_dict = {}

        for key, value in state_dict.items():

            if key.startswith("model."):
                key = key[len("model."):]

            cleaned_state_dict[key] = value

        model.load_state_dict(
            cleaned_state_dict,
            strict=True
        )

    else:

        model.load_state_dict(
            checkpoint,
            strict=True
        )

    model.to(DEVICE)
    model.eval()

    print(f"[+] Cricket model loaded on {DEVICE}")

    return model


# ============================================================
# FRAME LOADING
# ============================================================

def load_frames(frames_dir):
    frames_dir = Path(frames_dir)

    frame_paths = sorted(
        frames_dir.glob("frame_*.png")
    )

    if len(frame_paths) != 30:
        raise ValueError(
            f"Expected 30 frames, found {len(frame_paths)} "
            f"in {frames_dir}"
        )

    frames = []

    for frame_path in frame_paths:

        image = Image.open(
            frame_path
        ).convert("RGB")

        tensor = FRAME_TRANSFORM(image)

        frames.append(tensor)

    frames = torch.stack(
        frames,
        dim=0
    )

    # (30, 3, 224, 224)
    frames = frames.unsqueeze(0)

    # (1, 30, 3, 224, 224)
    return frames


# ============================================================
# MODEL PREDICTION
# ============================================================

@torch.inference_mode()
def predict_clip(model, frames):

    frames = frames.to(DEVICE)

    logits = model(frames)

    probabilities = F.softmax(
        logits,
        dim=1
    )

    confidence, class_index = torch.max(
        probabilities,
        dim=1
    )

    class_index = class_index.item()
    confidence = confidence.item()

    prediction = SHOT_CLASSES[class_index]

    class_probabilities = {}

    for i, class_name in enumerate(SHOT_CLASSES):

        class_probabilities[class_name] = round(
            probabilities[0, i].item(),
            6
        )

    return {
        "prediction": prediction,
        "confidence": round(
            confidence,
            6
        ),
        "class_probabilities": class_probabilities
    }


# ============================================================
# TEMPORARY VIDEO STORAGE
# ============================================================

def save_upload(upload_file, destination):

    with open(destination, "wb") as buffer:

        while True:

            chunk = upload_file.file.read(
                1024 * 1024
            )

            if not chunk:
                break

            buffer.write(chunk)


# ============================================================
# PREDICT
# ============================================================

@app.post("/api/predict")
async def predict_shot(
    video: UploadFile = File(...)
):

    temp_dir = None

    try:

        # ----------------------------------------------------
        # Create temporary request directory
        # ----------------------------------------------------

        temp_dir = Path(
            tempfile.mkdtemp(
                prefix="cricket_"
            )
        )

        video_path = (
            temp_dir /
            "input_video.mp4"
        )

        clips_dir = (
            temp_dir /
            "delivery_clips"
        )

        frames_dir = (
            temp_dir /
            "frames"
        )

        clips_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        frames_dir.mkdir(
            parents=True,
            exist_ok=True
        )


        # ----------------------------------------------------
        # Save uploaded video
        # ----------------------------------------------------

        await video.seek(0)

        save_upload(
            video,
            video_path
        )

        print(
            f"[+] Uploaded video: {video_path}"
        )


        # ----------------------------------------------------
        # Validate video
        # ----------------------------------------------------

        cap = cv2.VideoCapture(
            str(video_path)
        )

        if not cap.isOpened():

            raise ValueError(
                "Could not open uploaded video."
            )

        cap.release()


        # ----------------------------------------------------
        # Load AutoClipper
        # ----------------------------------------------------

        if not BALL_MODEL.exists():
            raise FileNotFoundError(
                f"Ball model not found: {BALL_MODEL}"
            )

        if not BAT_MODEL.exists():
            raise FileNotFoundError(
                f"Bat model not found: {BAT_MODEL}"
            )


        clipper = AutoClipper(
            ball_model_path=str(BALL_MODEL),
            bat_model_path=str(BAT_MODEL),
            output_dir=str(clips_dir),
            use_ocr=True,
            ocr_gpu=torch.cuda.is_available()
        )


        # ----------------------------------------------------
        # Detect delivery clips
        # ----------------------------------------------------

        extracted_clips = (
            clipper.process_match(
                str(video_path),
                clip_duration_sec=1.0
            )
        )


        if not extracted_clips:

            return JSONResponse(
                content={
                    "prediction": None,
                    "confidence": 0,
                    "message": (
                        "No valid delivery was detected "
                        "in the uploaded video."
                    ),
                    "clips_processed": 0,
                    "class_probabilities": {}
                }
            )


        # ----------------------------------------------------
        # Load cricket model
        # ----------------------------------------------------

        model = load_cricket_model()


        # ----------------------------------------------------
        # Predict every detected delivery
        # ----------------------------------------------------

        predictions = []

        for clip_number, clip_path in enumerate(
            extracted_clips,
            start=1
        ):

            clip_path = Path(
                clip_path
            )

            delivery_id = clip_path.stem

            current_frames_dir = (
                frames_dir /
                delivery_id
            )

            current_frames_dir.mkdir(
                parents=True,
                exist_ok=True
            )


            # ----------------------------------------------
            # Extract 30 frames
            # ----------------------------------------------

            success = extract_frames(
                clip_path,
                current_frames_dir
            )

            if not success:

                print(
                    f"[WARNING] "
                    f"Frame extraction failed: "
                    f"{clip_path}"
                )

                continue


            # ----------------------------------------------
            # Load frames
            # ----------------------------------------------

            frames = load_frames(
                current_frames_dir
            )


            # ----------------------------------------------
            # Model prediction
            # ----------------------------------------------

            result = predict_clip(
                model,
                frames
            )

            result["delivery_id"] = delivery_id
            result["clip_number"] = clip_number

            predictions.append(
                result
            )


        if not predictions:

            raise ValueError(
                "Delivery clips were detected, "
                "but no clip could be classified."
            )


        # ----------------------------------------------------
        # Select highest-confidence prediction
        # ----------------------------------------------------

        best_prediction = max(
            predictions,
            key=lambda x: x["confidence"]
        )


        # ----------------------------------------------------
        # Average probabilities across clips
        # ----------------------------------------------------

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
                float(np.mean(values)),
                6
            )


        average_class_index = int(
            np.argmax(
                [
                    averaged_probabilities[
                        class_name
                    ]
                    for class_name in SHOT_CLASSES
                ]
            )
        )

        final_prediction = SHOT_CLASSES[
            average_class_index
        ]

        final_confidence = averaged_probabilities[
            final_prediction
        ]


        # ----------------------------------------------------
        # Timeline
        # ----------------------------------------------------

        timeline = []

        for prediction in predictions:

            timeline.append({
                "delivery_id": prediction[
                    "delivery_id"
                ],
                "prediction": prediction[
                    "prediction"
                ],
                "confidence": prediction[
                    "confidence"
                ]
            })


        # ----------------------------------------------------
        # Final response
        # ----------------------------------------------------

        return JSONResponse(
            content={
                "prediction": final_prediction,

                "confidence": round(
                    final_confidence,
                    6
                ),

                "class_probabilities":
                    averaged_probabilities,

                "timeline":
                    timeline,

                "clips_processed":
                    len(predictions)
            }
        )


    except Exception as e:

        print(
            f"[ERROR] {type(e).__name__}: {e}"
        )

        return JSONResponse(
            status_code=500,
            content={
                "error": str(e)
            }
        )


    finally:

        # ----------------------------------------------------
        # Delete temporary files
        # ----------------------------------------------------

        if temp_dir and temp_dir.exists():

            shutil.rmtree(
                temp_dir,
                ignore_errors=True
            )


# ============================================================
# HEALTH
# ============================================================

@app.get("/api/health")
async def health_check():

    return {
        "status": "CricShot Pipeline Online",
        "device": str(DEVICE)
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
        reload=False
    )