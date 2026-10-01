from __future__ import annotations

import json
import shutil
import sys
import threading
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles


# ============================================================
# PATH SETUP
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


# ============================================================
# IMPORT TESTED ANALYSIS PIPELINE
# ============================================================

from src.pipeline.delivery_analyzer import (  # noqa: E402
    analyze_delivery_clip,
    load_shot_model,
)


# ============================================================
# DIRECTORIES
# ============================================================

UPLOAD_DIR = BASE_DIR / "uploads"
RESULT_DIR = BASE_DIR / "results"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="Cricket Analysis API",
    description="Cricket shot classification and ball tracking backend",
    version="1.0.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# OPTIONAL STATIC RESULT DIRECTORY
# ============================================================

app.mount(
    "/results",
    StaticFiles(directory=str(RESULT_DIR)),
    name="results",
)


# ============================================================
# GLOBAL MODEL
# ============================================================

SHOT_MODEL = None
MODEL_LOCK = threading.Lock()


def get_shot_model():
    """
    Load the Transformer shot-classification model once
    and reuse it for subsequent requests.
    """
    global SHOT_MODEL

    if SHOT_MODEL is None:
        with MODEL_LOCK:
            if SHOT_MODEL is None:
                print("=" * 70)
                print("Loading cricket shot classification model...")
                print("=" * 70)

                SHOT_MODEL = load_shot_model()

                print("=" * 70)
                print("Shot classification model loaded successfully.")
                print("=" * 70)

    return SHOT_MODEL


# ============================================================
# TASK STORAGE
# ============================================================

TASKS: dict[str, dict[str, Any]] = {}


def create_task() -> str:
    task_id = str(uuid.uuid4())

    TASKS[task_id] = {
        "task_id": task_id,
        "status": "queued",
        "progress": 0,
        "message": "Task created.",
        "result": None,
        "error": None,
    }

    return task_id


def update_task(
    task_id: str,
    *,
    status: str | None = None,
    progress: int | None = None,
    message: str | None = None,
    result: Any | None = None,
    error: str | None = None,
) -> None:

    if task_id not in TASKS:
        return

    if status is not None:
        TASKS[task_id]["status"] = status

    if progress is not None:
        TASKS[task_id]["progress"] = max(0, min(100, progress))

    if message is not None:
        TASKS[task_id]["message"] = message

    if result is not None:
        TASKS[task_id]["result"] = result

    if error is not None:
        TASKS[task_id]["error"] = error


# ============================================================
# BASIC HEALTH ENDPOINT
# ============================================================

@app.get("/")
def root():
    return {
        "name": "Cricket Analysis API",
        "version": "1.0.0",
        "status": "running",
        "services": {
            "shot_classification": True,
            "ball_tracking": True,
            "auto_clipper": False,
        },
    }


@app.get("/api/health")
def health():
    return {
        "status": "healthy",
        "shot_model_loaded": SHOT_MODEL is not None,
        "shot_classification": "EfficientNet-B0 + Transformer",
        "ball_tracking": "YOLO + Kalman",
        "auto_clipper": "not integrated",
    }


# ============================================================
# MODEL LOAD ENDPOINT
# ============================================================

@app.post("/api/load-model")
def load_model():
    try:
        get_shot_model()

        return {
            "status": "success",
            "message": "Shot classification model loaded.",
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load shot model: {exc}",
        ) from exc


# ============================================================
# SINGLE CLIP PROCESSING
# ============================================================

def process_single_clip(
    task_id: str,
    clip_path: Path,
    delivery_number: int,
) -> None:

    try:
        update_task(
            task_id,
            status="processing",
            progress=10,
            message="Loading shot classification model...",
        )

        model = get_shot_model()

        update_task(
            task_id,
            progress=25,
            message="Analyzing delivery...",
        )

        result = analyze_delivery_clip(
            clip_path=clip_path,
            model=model,
            delivery_number=delivery_number,
        )

        update_task(
            task_id,
            status="completed",
            progress=100,
            message="Delivery analysis completed.",
            result=result,
        )

    except Exception as exc:
        update_task(
            task_id,
            status="error",
            progress=100,
            message="Delivery analysis failed.",
            error=str(exc),
        )

    finally:
        # Keep uploaded files for now so we can inspect/debug them.
        pass


# ============================================================
# PREDICT ENDPOINT
# ============================================================

@app.post("/api/predict")
async def predict(
    file: UploadFile = File(...),
):
    """
    Upload one already-extracted delivery clip.

    The clip is processed by:

        EfficientNet-B0 + Transformer
                    +
              YOLO + Kalman

    Returns a task_id that can be queried through
    /api/progress/{task_id}.
    """

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No filename provided.",
        )

    suffix = Path(file.filename).suffix.lower()

    allowed_extensions = {
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
        ".webm",
    }

    if suffix not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported video format: {suffix}. "
                f"Allowed formats: {sorted(allowed_extensions)}"
            ),
        )

    task_id = create_task()

    task_dir = UPLOAD_DIR / task_id
    task_dir.mkdir(parents=True, exist_ok=True)

    clip_path = task_dir / f"delivery{suffix}"

    try:
        with clip_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

    except Exception as exc:
        shutil.rmtree(task_dir, ignore_errors=True)

        update_task(
            task_id,
            status="error",
            progress=100,
            message="Failed to save uploaded video.",
            error=str(exc),
        )

        raise HTTPException(
            status_code=500,
            detail=f"Failed to save uploaded video: {exc}",
        ) from exc

    update_task(
        task_id,
        status="queued",
        progress=0,
        message="Video uploaded successfully.",
    )

    thread = threading.Thread(
        target=process_single_clip,
        args=(task_id, clip_path, 1),
        daemon=True,
    )

    thread.start()

    return {
        "status": "accepted",
        "task_id": task_id,
        "filename": file.filename,
        "message": "Delivery analysis started.",
    }


# ============================================================
# PROGRESS ENDPOINT
# ============================================================

@app.get("/api/progress/{task_id}")
def get_progress(task_id: str):
    if task_id not in TASKS:
        raise HTTPException(
            status_code=404,
            detail="Task not found.",
        )

    return TASKS[task_id]


# ============================================================
# DIRECT SYNCHRONOUS ANALYSIS ENDPOINT
# ============================================================

@app.post("/api/analyze")
async def analyze(
    file: UploadFile = File(...),
):
    """
    Synchronous endpoint.

    Useful for simple frontend testing because it waits
    until the delivery has been completely analyzed.
    """

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No filename provided.",
        )

    suffix = Path(file.filename).suffix.lower()

    allowed_extensions = {
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
        ".webm",
    }

    if suffix not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video format: {suffix}",
        )

    request_id = str(uuid.uuid4())

    task_dir = UPLOAD_DIR / request_id
    task_dir.mkdir(parents=True, exist_ok=True)

    clip_path = task_dir / f"delivery{suffix}"

    try:
        with clip_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        model = get_shot_model()

        result = analyze_delivery_clip(
            clip_path=clip_path,
            model=model,
            delivery_number=1,
        )

        return JSONResponse(
            content=result,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {exc}",
        ) from exc


# ============================================================
# MULTIPLE EXTRACTED CLIPS
# ============================================================

def process_multiple_clips(
    task_id: str,
    clip_paths: list[tuple[int, Path]],
) -> None:

    results = []

    try:
        model = get_shot_model()

        total = len(clip_paths)

        for index, (delivery_number, clip_path) in enumerate(
            clip_paths,
            start=1,
        ):

            progress = int(((index - 1) / total) * 100)

            update_task(
                task_id,
                status="processing",
                progress=progress,
                message=(
                    f"Analyzing delivery "
                    f"{delivery_number}/{total}..."
                ),
            )

            try:
                result = analyze_delivery_clip(
                    clip_path=clip_path,
                    model=model,
                    delivery_number=delivery_number,
                )

                results.append(result)

            except Exception as exc:
                results.append(
                    {
                        "delivery": delivery_number,
                        "clip": clip_path.name,
                        "error": str(exc),
                    }
                )

        successful = sum(
            1
            for result in results
            if "error" not in result
        )

        failed = total - successful

        final_result = {
            "total_clips": total,
            "completed": successful,
            "errors": failed,
            "results": results,
        }

        update_task(
            task_id,
            status="completed",
            progress=100,
            message="All delivery clips analyzed.",
            result=final_result,
        )

    except Exception as exc:
        update_task(
            task_id,
            status="error",
            progress=100,
            message="Multiple delivery analysis failed.",
            error=str(exc),
        )


@app.post("/api/analyze-multiple")
async def analyze_multiple(
    files: list[UploadFile] = File(...),
):
    """
    Analyze multiple already-extracted delivery clips.

    Files are processed in the order they are uploaded.
    """

    if not files:
        raise HTTPException(
            status_code=400,
            detail="No video files provided.",
        )

    task_id = create_task()

    task_dir = UPLOAD_DIR / task_id
    task_dir.mkdir(parents=True, exist_ok=True)

    clip_paths: list[tuple[int, Path]] = []

    allowed_extensions = {
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
        ".webm",
    }

    try:
        for delivery_number, file in enumerate(files, start=1):

            if not file.filename:
                continue

            suffix = Path(file.filename).suffix.lower()

            if suffix not in allowed_extensions:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Unsupported video format for "
                        f"{file.filename}: {suffix}"
                    ),
                )

            clip_path = (
                task_dir
                / f"delivery_{delivery_number:03d}{suffix}"
            )

            with clip_path.open("wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            clip_paths.append(
                (delivery_number, clip_path)
            )

    except HTTPException:
        shutil.rmtree(task_dir, ignore_errors=True)
        raise

    except Exception as exc:
        shutil.rmtree(task_dir, ignore_errors=True)

        raise HTTPException(
            status_code=500,
            detail=f"Failed to save uploaded clips: {exc}",
        ) from exc

    if not clip_paths:
        shutil.rmtree(task_dir, ignore_errors=True)

        raise HTTPException(
            status_code=400,
            detail="No valid video clips were uploaded.",
        )

    update_task(
        task_id,
        status="queued",
        progress=0,
        message=f"{len(clip_paths)} clips uploaded.",
    )

    thread = threading.Thread(
        target=process_multiple_clips,
        args=(task_id, clip_paths),
        daemon=True,
    )

    thread.start()

    return {
        "status": "accepted",
        "task_id": task_id,
        "total_clips": len(clip_paths),
        "message": "Multiple delivery analysis started.",
    }


# ============================================================
# DEBUG TASK ENDPOINT
# ============================================================

@app.get("/api/tasks")
def get_tasks():
    return {
        "count": len(TASKS),
        "tasks": TASKS,
    }


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
def startup_event():
    print()
    print("=" * 70)
    print("CRICKET ANALYSIS BACKEND")
    print("=" * 70)
    print(f"Backend directory : {BASE_DIR}")
    print(f"Upload directory  : {UPLOAD_DIR}")
    print(f"Result directory  : {RESULT_DIR}")
    print()
    print("Available endpoints:")
    print("  GET  /")
    print("  GET  /api/health")
    print("  POST /api/load-model")
    print("  POST /api/predict")
    print("  POST /api/analyze")
    print("  POST /api/analyze-multiple")
    print("  GET  /api/progress/{task_id}")
    print("  GET  /api/tasks")
    print("=" * 70)
    print()


# ============================================================
# LOCAL ENTRY POINT
# ============================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )