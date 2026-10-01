from __future__ import annotations

import json
from pathlib import Path

from src.pipeline.delivery_analyzer import (
    analyze_delivery_clip,
    load_shot_model,
)


CLIP_PATH = Path(
    r"C:\Users\mamid\OneDrive\Desktop\Cricket"
    r"\backend\test_41_output\delivery_001.mp4"
)


def main():

    print()
    print("=" * 70)
    print("SINGLE DELIVERY COMBINED TEST")
    print("=" * 70)

    if not CLIP_PATH.is_file():
        raise FileNotFoundError(
            f"Clip not found:\n{CLIP_PATH}"
        )

    print(
        f"\nClip:\n{CLIP_PATH}"
    )

    model = load_shot_model()

    result = analyze_delivery_clip(
        clip_path=CLIP_PATH,
        model=model,
        delivery_number=1,
    )

    print()
    print("=" * 70)
    print("FINAL COMBINED RESULT")
    print("=" * 70)

    print(
        json.dumps(
            result,
            indent=2,
        )
    )

    print()
    print("=" * 70)
    print("TEST FINISHED")
    print("=" * 70)


if __name__ == "__main__":
    main()