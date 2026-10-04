"""
Kalman trajectory video generator.

Uses the existing:
    BallDetector
    BallTracker

The visualization uses the trajectory returned by the tracker,
including Kalman-tracked positions, instead of drawing only raw
YOLO detections.
"""

from pathlib import Path
from typing import List, Optional, Tuple

import cv2

from ..tracking.detector import BallDetector
from ..tracking.tracker import BallTracker


Point = Tuple[int, int]


def _extract_point(value) -> Optional[Point]:
    """
    Convert a tracker position into an (x, y) integer point.

    Supports:
        (x, y)
        [x, y]
        numpy arrays
        dictionaries containing x/y
    """

    if value is None:
        return None

    # Dictionary form
    if isinstance(value, dict):
        if "x" in value and "y" in value:
            return int(round(float(value["x"]))), int(
                round(float(value["y"]))
            )

        if "center" in value:
            return _extract_point(value["center"])

        return None

    # Tuple/list/numpy-array form
    try:
        if len(value) >= 2:
            return (
                int(round(float(value[0]))),
                int(round(float(value[1]))),
            )
    except (TypeError, ValueError, IndexError):
        pass

    return None


def _get_latest_tracker_point(tracker) -> Optional[Point]:
    """
    Get the newest point from the tracker's trajectory.
    """

    try:
        trajectory = tracker.get_trajectory()
    except AttributeError:
        return None

    if not trajectory:
        return None

    return _extract_point(trajectory[-1])


def generate_trajectory_video(
    input_video: str,
    output_video: str,
    trail_length: int = 40,
) -> dict:
    """
    Generate a video showing the Kalman-tracked ball trajectory.
    """

    input_path = Path(input_video)
    output_path = Path(output_video)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input video not found:\n{input_path}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Open input
    # --------------------------------------------------------

    cap = cv2.VideoCapture(str(input_path))

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open input video:\n{input_path}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 25.0

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    # --------------------------------------------------------
    # Open output
    # --------------------------------------------------------

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height),
    )

    if not writer.isOpened():
        cap.release()

        raise RuntimeError(
            f"Could not create output video:\n{output_path}"
        )

    # --------------------------------------------------------
    # Existing validated detector + tracker
    # --------------------------------------------------------

    detector = BallDetector()
    tracker = BallTracker()

    visual_trail: List[Point] = []

    frames_processed = 0
    detections = 0
    tracked_points = 0

    previous_tracker_point = None

    # --------------------------------------------------------
    # Process video
    # --------------------------------------------------------

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_number = frames_processed

        # ----------------------------------------------------
        # YOLO detection
        # ----------------------------------------------------

        detection = detector.detect(frame)

        if detection is not None:
            detections += 1

        # ----------------------------------------------------
        # Update Kalman tracker
        # ----------------------------------------------------

        tracker.update(
            detection,
            frame_number,
        )

        # ----------------------------------------------------
        # IMPORTANT:
        # Read the position AFTER tracker.update().
        #
        # This allows the visualization to use the tracker's
        # trajectory rather than simply copying YOLO detections.
        # ----------------------------------------------------

        tracker_point = _get_latest_tracker_point(
            tracker
        )

        if tracker_point is not None:

            # Avoid adding exactly the same point repeatedly.
            if (
                previous_tracker_point is None
                or tracker_point != previous_tracker_point
            ):
                visual_trail.append(tracker_point)

                tracked_points += 1

                previous_tracker_point = tracker_point

        # ----------------------------------------------------
        # Keep only recent trajectory
        # ----------------------------------------------------

        if len(visual_trail) > trail_length:
            visual_trail = visual_trail[-trail_length:]

        # ----------------------------------------------------
        # Draw complete tracked trajectory
        # ----------------------------------------------------

        if len(visual_trail) >= 2:

            for i in range(
                1,
                len(visual_trail),
            ):

                previous = visual_trail[i - 1]
                current = visual_trail[i]

                cv2.line(
                    frame,
                    previous,
                    current,
                    (0, 255, 255),
                    3,
                    cv2.LINE_AA,
                )

        # ----------------------------------------------------
        # Draw CURRENT Kalman position
        # ----------------------------------------------------

        if tracker_point is not None:

            x, y = tracker_point

            # Outer circle
            cv2.circle(
                frame,
                (x, y),
                12,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # Ball marker
            cv2.circle(
                frame,
                (x, y),
                7,
                (0, 0, 255),
                -1,
                cv2.LINE_AA,
            )

        # ----------------------------------------------------
        # Information panel
        # ----------------------------------------------------

        panel_height = 125

        cv2.rectangle(
            frame,
            (10, 10),
            (335, panel_height),
            (0, 0, 0),
            -1,
        )

        cv2.putText(
            frame,
            f"Frame: {frame_number}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            f"YOLO detections: {detections}",
            (20, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            f"Tracked points: {tracked_points}",
            (20, 89),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            "KALMAN TRAJECTORY",
            (20, 116),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # ----------------------------------------------------
        # Write frame
        # ----------------------------------------------------

        writer.write(frame)

        frames_processed += 1

    # --------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------

    cap.release()
    writer.release()

    # --------------------------------------------------------
    # Get final tracker trajectory
    # --------------------------------------------------------

    trajectory = tracker.get_trajectory()

    tracking_success = len(trajectory) > 0

    return {
        "input_video": str(input_path),
        "output_video": str(output_path),
        "frames_processed": frames_processed,
        "detections": detections,
        "tracked_points": tracked_points,
        "trajectory_points": len(trajectory),
        "tracking_success": tracking_success,
    }


# ============================================================
# Standalone validation
# ============================================================

if __name__ == "__main__":

    project_root = (
        Path(__file__)
        .resolve()
        .parents[2]
    )

    input_video = (
        project_root
        / "full_match_output"
        / "delivery_clips"
        / "delivery_001.mp4"
    )

    output_video = (
        project_root
        / "visualization_test"
        / "delivery_001_kalman_trajectory.mp4"
    )

    print("=" * 70)
    print("KALMAN TRAJECTORY VIDEO TEST")
    print("=" * 70)

    print(f"\nInput : {input_video}")
    print(f"Output: {output_video}")

    result = generate_trajectory_video(
        input_video=str(input_video),
        output_video=str(output_video),
        trail_length=40,
    )

    print("\n")
    print("=" * 70)
    print("RESULT")
    print("=" * 70)

    print(
        f"Frames processed : "
        f"{result['frames_processed']}"
    )

    print(
        f"YOLO detections  : "
        f"{result['detections']}"
    )

    print(
        f"Tracked points   : "
        f"{result['tracked_points']}"
    )

    print(
        f"Trajectory points: "
        f"{result['trajectory_points']}"
    )

    print(
        f"Tracking success : "
        f"{result['tracking_success']}"
    )

    print(
        f"\nOutput video:\n"
        f"{result['output_video']}"
    )

    if result["tracking_success"]:
        print("\nTEST PASSED")
    else:
        print(
            "\nTEST FAILED: "
            "No Kalman trajectory was generated."
        )