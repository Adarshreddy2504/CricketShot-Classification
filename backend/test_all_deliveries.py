from __future__ import annotations

import json
from pathlib import Path

from src.pipeline.delivery_analyzer import (
    analyze_delivery_clip,
    load_shot_model,
)


CLIPS_DIR = Path(
    r"C:\Users\mamid\OneDrive\Desktop\Cricket"
    r"\backend\test_41_output"
)


def main():
    print("=" * 70)
    print("MULTI-DELIVERY COMBINED TEST")
    print("=" * 70)

    clips = sorted(
        CLIPS_DIR.glob("delivery_*.mp4")
    )

    if not clips:
        raise FileNotFoundError(
            f"No delivery clips found in:\n{CLIPS_DIR}"
        )

    print(f"\nFound {len(clips)} delivery clips.")

    # Load the shot model ONCE.
    print("\nLoading shot model...")
    model = load_shot_model()

    results = []

    for index, clip_path in enumerate(clips, start=1):

        print("\n" + "-" * 70)
        print(
            f"[{index}/{len(clips)}] "
            f"Analyzing {clip_path.name}"
        )
        print("-" * 70)

        try:
            result = analyze_delivery_clip(
                clip_path=clip_path,
                model=model,
                delivery_number=index,
            )

            results.append(result)

            shot = result.get("shot", {})
            ball = result.get("ball", {})

            print(
                f"[SHOT] "
                f"{shot.get('prediction', 'Unknown')} "
                f"({shot.get('confidence', 0.0) * 100:.2f}%)"
            )

            print(
                f"[BALL] "
                f"tracking={ball.get('tracking_success', False)} "
                f"frames={ball.get('frames_processed', 0)} "
                f"detections={ball.get('detections', 0)} "
                f"trajectory={len(ball.get('trajectory', []))}"
            )

        except Exception as exc:
            print(
                f"[ERROR] {clip_path.name}: "
                f"{type(exc).__name__}: {exc}"
            )

            results.append(
                {
                    "delivery": index,
                    "clip": clip_path.name,
                    "error": str(exc),
                }
            )

    # ---------------------------------------------------------------
    # Save results
    # ---------------------------------------------------------------

    output_path = CLIPS_DIR / "multi_delivery_results.json"

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            results,
            f,
            indent=2,
        )

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------

    successful = [
        r for r in results
        if "error" not in r
    ]

    tracking_success = [
        r for r in successful
        if r.get("ball", {}).get(
            "tracking_success",
            False,
        )
    ]

    failed = [
        r for r in results
        if "error" in r
    ]

    print("\n")
    print("=" * 70)
    print("MULTI-DELIVERY SUMMARY")
    print("=" * 70)

    print(f"Total clips       : {len(clips)}")
    print(f"Completed         : {len(successful)}")
    print(f"Errors            : {len(failed)}")
    print(f"Tracking success  : {len(tracking_success)}")

    if successful:
        print("\nShot predictions:")

        for result in successful:
            shot = result.get("shot", {})

            print(
                f"  Delivery "
                f"{result.get('delivery', '?'):>2}: "
                f"{shot.get('prediction', 'Unknown'):<12} "
                f"{shot.get('confidence', 0.0) * 100:6.2f}%"
            )

    print("\nResults saved to:")
    print(output_path)

    print("=" * 70)
    print("TEST FINISHED")
    print("=" * 70)


if __name__ == "__main__":
    main()