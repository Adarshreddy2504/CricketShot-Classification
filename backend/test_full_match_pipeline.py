"""
End-to-end test for the complete cricket match pipeline.
"""

from pathlib import Path
import sys


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT)
    )


# ============================================================
# PIPELINE IMPORT
# ============================================================

from backend.src.pipeline.match_pipeline import (
    analyze_match,
)


# ============================================================
# FULL MATCH VIDEO
# ============================================================

MATCH_VIDEO = Path(
    r"C:\Users\mamid\OneDrive\Desktop\matches\long_match.mp4"
)


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_DIR = (
    PROJECT_ROOT
    / "backend"
    / "full_match_output"
)


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 70)
    print("FULL MATCH PIPELINE TEST")
    print("=" * 70)

    print()
    print("Match video:")
    print(MATCH_VIDEO)

    # --------------------------------------------------------
    # Validate video
    # --------------------------------------------------------

    if not MATCH_VIDEO.exists():

        raise FileNotFoundError(
            f"Match video not found:\n"
            f"{MATCH_VIDEO}"
        )

    if not MATCH_VIDEO.is_file():

        raise ValueError(
            f"Match video is not a file:\n"
            f"{MATCH_VIDEO}"
        )

    file_size_mb = (
        MATCH_VIDEO.stat().st_size
        / (1024 * 1024)
    )

    print()
    print(
        f"File size: "
        f"{file_size_mb:.2f} MB"
    )

    # --------------------------------------------------------
    # Run pipeline
    # --------------------------------------------------------

    result = analyze_match(
        video_path=MATCH_VIDEO,
        output_dir=OUTPUT_DIR,
        clip_duration_sec=1.0,
        confidence=0.30,
        use_ocr=True,
        ocr_gpu=False,
    )

    # --------------------------------------------------------
    # Validate result
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("END-TO-END TEST FINISHED")
    print("=" * 70)

    print(
        f"Total deliveries : "
        f"{result['total_deliveries']}"
    )

    print(
        f"Completed        : "
        f"{result['completed']}"
    )

    print(
        f"Errors           : "
        f"{result['errors']}"
    )

    print(
        f"Tracking success : "
        f"{result['tracking_success']}"
    )

    print(
        f"Tracking failed  : "
        f"{result['tracking_failures']}"
    )

    results_file = (
        OUTPUT_DIR
        / "match_pipeline_results.json"
    )

    print()
    print("Results JSON:")
    print(results_file)

    if not results_file.exists():

        raise RuntimeError(
            "Pipeline completed but "
            "results JSON was not created."
        )

    print()
    print("RESULT JSON EXISTS: YES")
    print()
    print("=" * 70)
    print("TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()