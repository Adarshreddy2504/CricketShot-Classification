"""
Complete Cricket Match Pipeline.

Pipeline:
    Full Match Video
        ->
    AutoClipper
        ->
    Delivery Clips
        ->
    EfficientNet-B0 + Transformer
        ->
    YOLO Ball Detection
        ->
    Kalman Ball Tracking
        ->
    Combined JSON Results
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .delivery_analyzer import (
    analyze_delivery_clip,
    load_shot_model,
)
from ..segmentation.auto_clipper import AutoClipper


# ============================================================
# PATHS
# ============================================================

BACKEND_DIR = Path(__file__).resolve().parents[2]

MODELS_DIR = BACKEND_DIR / "models"

BALL_CLIPPER_MODEL = (
    MODELS_DIR / "Ball_Detection_Model.pt"
)

BAT_CLIPPER_MODEL = (
    MODELS_DIR / "Bat_Detection_Model.pt"
)

DEFAULT_OUTPUT_DIR = (
    BACKEND_DIR / "match_results"
)


# ============================================================
# LOGGING
# ============================================================

logger = logging.getLogger(__name__)


# ============================================================
# VALIDATION HELPERS
# ============================================================

def _validate_file(
    path: Path,
    name: str,
) -> None:
    """
    Validate that a required file exists.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"{name} not found:\n{path}"
        )

    if not path.is_file():
        raise ValueError(
            f"{name} is not a file:\n{path}"
        )


def _validate_models() -> None:
    """
    Validate the models required by AutoClipper.
    """

    _validate_file(
        BALL_CLIPPER_MODEL,
        "Ball detection model",
    )

    _validate_file(
        BAT_CLIPPER_MODEL,
        "Bat detection model",
    )


# ============================================================
# JSON HELPERS
# ============================================================

def _json_safe(value: Any) -> Any:
    """
    Convert common Python / NumPy / Path values into
    JSON-safe values.
    """

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {
            str(key): _json_safe(val)
            for key, val in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            _json_safe(item)
            for item in value
        ]

    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except Exception:
            pass

    if hasattr(value, "tolist"):
        try:
            return _json_safe(value.tolist())
        except Exception:
            pass

    if isinstance(value, float):

        if value != value:
            return None

        if value == float("inf"):
            return None

        if value == float("-inf"):
            return None

        return value

    if isinstance(value, (str, int, bool)):
        return value

    if value is None:
        return None

    return str(value)


# ============================================================
# AUTOCLIPPER
# ============================================================

def create_auto_clipper(
    output_dir: str | Path,
    confidence: float = 0.30,
    use_ocr: bool = True,
    ocr_gpu: bool = False,
) -> AutoClipper:
    """
    Create the existing AutoClipper with the project's
    trained detection models.
    """

    _validate_models()

    output_dir = Path(
        output_dir
    ).resolve()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("[AUTOCLIPPER] Creating AutoClipper...")
    print(
        f"[AUTOCLIPPER] Ball model: "
        f"{BALL_CLIPPER_MODEL}"
    )
    print(
        f"[AUTOCLIPPER] Bat model: "
        f"{BAT_CLIPPER_MODEL}"
    )
    print(
        f"[AUTOCLIPPER] Output: "
        f"{output_dir}"
    )

    clipper = AutoClipper(
        ball_model_path=str(
            BALL_CLIPPER_MODEL
        ),
        bat_model_path=str(
            BAT_CLIPPER_MODEL
        ),
        output_dir=str(
            output_dir
        ),
        confidence=confidence,
        use_ocr=use_ocr,
        ocr_gpu=ocr_gpu,
    )

    return clipper


# ============================================================
# ANALYZE DELIVERY CLIPS
# ============================================================

def analyze_delivery_clips(
    clips_dir: str | Path,
    model=None,
) -> Dict[str, Any]:
    """
    Analyze every delivery clip.

    Shot classification:
        EfficientNet-B0 + Transformer

    Ball tracking:
        YOLO + Kalman

    IMPORTANT:
        delivery_analyzer returns ball tracking under:

            result["ball"]

        and not:

            result["ball_tracking"]
    """

    clips_dir = Path(
        clips_dir
    ).resolve()

    # --------------------------------------------------------
    # Validate directory
    # --------------------------------------------------------

    if not clips_dir.exists():
        raise FileNotFoundError(
            f"Delivery clips directory not found:\n"
            f"{clips_dir}"
        )

    if not clips_dir.is_dir():
        raise NotADirectoryError(
            f"Delivery clips path is not a directory:\n"
            f"{clips_dir}"
        )

    # --------------------------------------------------------
    # Find video clips
    # --------------------------------------------------------

    supported_extensions = {
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
    }

    clip_paths = sorted(
        [
            path
            for path in clips_dir.iterdir()
            if path.is_file()
            and path.suffix.lower()
            in supported_extensions
        ]
    )

    if not clip_paths:
        raise RuntimeError(
            f"No delivery clips found in:\n"
            f"{clips_dir}"
        )

    total_clips = len(
        clip_paths
    )

    print()
    print(
        f"[ANALYSIS] Found {total_clips} delivery clips."
    )

    # --------------------------------------------------------
    # Load shot model ONCE
    # --------------------------------------------------------

    if model is None:

        print()
        print(
            "[ANALYSIS] Loading "
            "EfficientNet-B0 + Transformer..."
        )

        model = load_shot_model()

    # --------------------------------------------------------
    # Counters
    # --------------------------------------------------------

    results: List[Dict[str, Any]] = []

    completed = 0
    errors = 0

    tracking_success = 0
    tracking_failures = 0

    # --------------------------------------------------------
    # Process clips
    # --------------------------------------------------------

    for index, clip_path in enumerate(
        clip_paths,
        start=1,
    ):

        print()
        print(
            "=" * 70
        )

        print(
            f"[{index}/{total_clips}] "
            f"Analyzing {clip_path.name}"
        )

        print(
            "=" * 70
        )

        try:

            # ------------------------------------------------
            # Run complete delivery analysis
            # ------------------------------------------------

            result = analyze_delivery_clip(
                clip_path=str(
                    clip_path
                ),
                model=model,
            )

            # ------------------------------------------------
            # Make result JSON-safe
            # ------------------------------------------------

            result = _json_safe(
                result
            )

            if not isinstance(
                result,
                dict,
            ):
                raise TypeError(
                    "analyze_delivery_clip() "
                    "must return a dictionary."
                )

            # ------------------------------------------------
            # Add clip information
            # ------------------------------------------------

            result["clip_path"] = str(
                clip_path
            )

            result["clip_name"] = (
                clip_path.name
            )

            # ------------------------------------------------
            # Get shot result
            # ------------------------------------------------

            shot_result = result.get(
                "shot",
                {},
            )

            if not isinstance(
                shot_result,
                dict,
            ):
                shot_result = {}

            shot_name = shot_result.get(
                "prediction",
                "Unknown",
            )

            shot_confidence = (
                shot_result.get(
                    "confidence",
                    0.0,
                )
            )

            # ------------------------------------------------
            # IMPORTANT:
            #
            # Ball tracking is inside result["ball"]
            # ------------------------------------------------

            ball_result = result.get(
                "ball",
                {},
            )

            if not isinstance(
                ball_result,
                dict,
            ):
                ball_result = {}

            tracking_ok = bool(
                ball_result.get(
                    "tracking_success",
                    False,
                )
            )

            frames_processed = (
                ball_result.get(
                    "frames_processed",
                    0,
                )
            )

            detections = (
                ball_result.get(
                    "detections",
                    0,
                )
            )

            trajectory = (
                ball_result.get(
                    "trajectory",
                    [],
                )
            )

            if not isinstance(
                trajectory,
                list,
            ):
                trajectory = []

            # ------------------------------------------------
            # Update counters
            # ------------------------------------------------

            completed += 1

            if tracking_ok:
                tracking_success += 1
            else:
                tracking_failures += 1

            # ------------------------------------------------
            # Add normalized tracking summary
            # ------------------------------------------------

            result["tracking_summary"] = {
                "success": tracking_ok,
                "frames_processed": (
                    frames_processed
                ),
                "detections": detections,
                "trajectory_points": len(
                    trajectory
                ),
            }

            # ------------------------------------------------
            # Store result
            # ------------------------------------------------

            results.append(
                result
            )

            # ------------------------------------------------
            # Print result
            # ------------------------------------------------

            print(
                f"  Shot       : "
                f"{shot_name}"
            )

            try:

                print(
                    f"  Confidence : "
                    f"{float(shot_confidence):.4f}"
                )

            except (
                TypeError,
                ValueError,
            ):

                print(
                    f"  Confidence : "
                    f"{shot_confidence}"
                )

            print(
                f"  Frames     : "
                f"{frames_processed}"
            )

            print(
                f"  Detections : "
                f"{detections}"
            )

            print(
                f"  Trajectory : "
                f"{len(trajectory)} points"
            )

            print(
                f"  Tracking   : "
                f"{'SUCCESS' if tracking_ok else 'FAILED'}"
            )

        except Exception as exc:

            errors += 1

            error_result = {
                "clip_path": str(
                    clip_path
                ),
                "clip_name": (
                    clip_path.name
                ),
                "success": False,
                "error": str(exc),
            }

            results.append(
                error_result
            )

            print()
            print(
                f"  ERROR: {exc}"
            )

    # --------------------------------------------------------
    # Final analysis summary
    # --------------------------------------------------------

    return {
        "clips_directory": str(
            clips_dir
        ),
        "total_clips": total_clips,
        "completed": completed,
        "errors": errors,
        "tracking_success": tracking_success,
        "tracking_failures": tracking_failures,
        "results": results,
    }


# ============================================================
# FULL MATCH ANALYSIS
# ============================================================

def analyze_match(
    video_path: str | Path,
    output_dir: str | Path | None = None,
    model=None,
    clip_duration_sec: float = 1.0,
    confidence: float = 0.30,
    use_ocr: bool = True,
    ocr_gpu: bool = False,
) -> Dict[str, Any]:
    """
    Run the complete cricket match pipeline.

    Steps:

        1. Validate full-match video
        2. Run AutoClipper
        3. Extract delivery clips
        4. Load Transformer
        5. Analyze all deliveries
        6. Run YOLO + Kalman tracking
        7. Save combined JSON
        8. Return final result
    """

    # --------------------------------------------------------
    # Resolve paths
    # --------------------------------------------------------

    video_path = Path(
        video_path
    ).resolve()

    if output_dir is None:

        output_dir = (
            DEFAULT_OUTPUT_DIR
        )

    else:

        output_dir = Path(
            output_dir
        ).resolve()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    clips_dir = (
        output_dir
        / "delivery_clips"
    )

    results_file = (
        output_dir
        / "match_pipeline_results.json"
    )

    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    _validate_file(
        video_path,
        "Match video",
    )

    _validate_models()

    if clip_duration_sec <= 0:

        raise ValueError(
            "clip_duration_sec must be greater than 0."
        )

    if not (
        0.0 < confidence <= 1.0
    ):

        raise ValueError(
            "confidence must be between 0 and 1."
        )

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("COMPLETE CRICKET MATCH PIPELINE")
    print("=" * 70)

    print(
        f"Match video       : "
        f"{video_path}"
    )

    print(
        f"Output directory  : "
        f"{output_dir}"
    )

    print(
        f"Clip duration     : "
        f"{clip_duration_sec:.2f} sec"
    )

    print(
        f"Detection conf.   : "
        f"{confidence:.2f}"
    )

    print("=" * 70)

    # ========================================================
    # STEP 1
    # AUTOCLIPPER
    # ========================================================

    print()
    print(
        "STEP 1/4 - AUTOCLIPPER"
    )

    print(
        "Extracting delivery clips..."
    )

    clipper = create_auto_clipper(
        output_dir=clips_dir,
        confidence=confidence,
        use_ocr=use_ocr,
        ocr_gpu=ocr_gpu,
    )

    generated_clips = (
        clipper.process_match(
            video_path=str(
                video_path
            ),
            clip_duration_sec=(
                clip_duration_sec
            ),
        )
    )

    if generated_clips is None:
        generated_clips = []

    generated_clips = [
        Path(path)
        for path in generated_clips
    ]

    print()
    print(
        f"AutoClipper generated "
        f"{len(generated_clips)} delivery clips."
    )

    if not generated_clips:

        raise RuntimeError(
            "AutoClipper did not generate "
            "any delivery clips."
        )

    # ========================================================
    # STEP 2
    # SHOT CLASSIFICATION
    # ========================================================

    print()
    print(
        "STEP 2/4 - SHOT CLASSIFICATION"
    )

    print(
        "Loading EfficientNet-B0 + Transformer..."
    )

    if model is None:

        model = load_shot_model()

    print(
        "Shot classification model loaded."
    )

    # ========================================================
    # STEP 3
    # DELIVERY ANALYSIS
    # ========================================================

    print()
    print(
        "STEP 3/4 - DELIVERY ANALYSIS"
    )

    analysis = analyze_delivery_clips(
        clips_dir=clips_dir,
        model=model,
    )

    # ========================================================
    # STEP 4
    # SAVE RESULTS
    # ========================================================

    print()
    print(
        "STEP 4/4 - SAVING RESULTS"
    )

    final_result = {
        "pipeline": (
            "Cricket Match Pipeline"
        ),
        "version": "1.0",
        "video_path": str(
            video_path
        ),
        "clips_directory": str(
            clips_dir
        ),
        "total_deliveries": (
            analysis[
                "total_clips"
            ]
        ),
        "completed": (
            analysis[
                "completed"
            ]
        ),
        "errors": (
            analysis[
                "errors"
            ]
        ),
        "tracking_success": (
            analysis[
                "tracking_success"
            ]
        ),
        "tracking_failures": (
            analysis[
                "tracking_failures"
            ]
        ),
        "results": (
            analysis[
                "results"
            ]
        ),
    }

    final_result = _json_safe(
        final_result
    )

    with results_file.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            final_result,
            file,
            indent=2,
            ensure_ascii=False,
        )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)

    print(
        f"Total deliveries : "
        f"{final_result['total_deliveries']}"
    )

    print(
        f"Completed        : "
        f"{final_result['completed']}"
    )

    print(
        f"Errors           : "
        f"{final_result['errors']}"
    )

    print(
        f"Tracking success : "
        f"{final_result['tracking_success']}"
    )

    print(
        f"Tracking failed  : "
        f"{final_result['tracking_failures']}"
    )

    print(
        f"Clips directory  : "
        f"{clips_dir}"
    )

    print(
        f"Results JSON     : "
        f"{results_file}"
    )

    print("=" * 70)

    return final_result


# ============================================================
# COMMAND LINE
# ============================================================

def main() -> None:
    """
    Command-line entry point.
    """

    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Run the complete cricket match pipeline."
        )
    )

    parser.add_argument(
        "video",
        type=str,
        help=(
            "Path to the full cricket "
            "match video."
        ),
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Output directory."
        ),
    )

    parser.add_argument(
        "--clip-duration",
        type=float,
        default=1.0,
        help=(
            "Delivery clip duration "
            "in seconds."
        ),
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.30,
        help=(
            "AutoClipper detection "
            "confidence."
        ),
    )

    parser.add_argument(
        "--no-ocr",
        action="store_true",
        help=(
            "Disable scoreboard OCR."
        ),
    )

    parser.add_argument(
        "--ocr-gpu",
        action="store_true",
        help=(
            "Run EasyOCR on GPU."
        ),
    )

    args = parser.parse_args()

    analyze_match(
        video_path=args.video,
        output_dir=args.output,
        clip_duration_sec=(
            args.clip_duration
        ),
        confidence=args.confidence,
        use_ocr=not args.no_ocr,
        ocr_gpu=args.ocr_gpu,
    )


if __name__ == "__main__":
    main()