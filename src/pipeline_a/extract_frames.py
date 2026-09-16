# Extract exactly 30 uniformly sampled frames from every delivery clip.
# Frames are converted from OpenCV BGR to RGB and resized to 224x224.

from pathlib import Path
import cv2
import numpy as np


N_FRAMES = 30


def extract_frames(video_path, output_dir):
    video_path = Path(video_path)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"[ERROR] Could not open: {video_path}")
        return False

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        print(f"[ERROR] No frames found: {video_path}")
        cap.release()
        return False

    # Uniformly sample 30 frame positions
    frame_indices = np.linspace(
        0,
        total_frames - 1,
        N_FRAMES,
        dtype=int
    )

    for i, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))

        ret, frame = cap.read()

        if not ret:
            print(f"[WARNING] Failed frame {frame_idx} in {video_path.name}")
            continue

        # Convert BGR → RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize to model input size
        frame = cv2.resize(frame, (224, 224))

        # Save as PNG
        output_path = output_dir / f"frame_{i:02d}.png"

        # Convert RGB → BGR because OpenCV writes BGR
        cv2.imwrite(
            str(output_path),
            cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        )

    cap.release()

    return True


def main():
    project_root = Path(__file__).resolve().parents[2]

    clips_dir = project_root / "data" / "delivery_clips"
    frames_dir = project_root / "data" / "frames"

    clips = sorted(clips_dir.glob("*.mp4"))

    if not clips:
        print(f"[ERROR] No clips found in: {clips_dir}")
        return

    print(f"[+] Found {len(clips)} delivery clips")
    print(f"[+] Output directory: {frames_dir}")

    successful = 0

    for clip in clips:
        clip_name = clip.stem
        clip_output = frames_dir / clip_name

        print(f"[+] Extracting: {clip.name}")

        if extract_frames(clip, clip_output):
            successful += 1

    print("\n" + "=" * 60)
    print("[+] FRAME EXTRACTION COMPLETE")
    print(f"[+] Clips processed: {successful}/{len(clips)}")
    print(f"[+] Frames per clip: {N_FRAMES}")
    print(f"[+] Saved to: {frames_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()