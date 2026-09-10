import sys
import pathlib

# Add the parent directory (Cricket root) to the system path
sys.path.append(str(pathlib.Path(__file__).parent.parent))

from src.segmentation.auto_clipper import AutoClipper

def main():
    # 1. Automatically find a test video in your raw_matches folder
    raw_matches_dir = pathlib.Path("data/raw_matches")
    video_files = list(raw_matches_dir.glob("*.mp4")) + list(raw_matches_dir.glob("*.avi"))

    if not video_files:
        print("[-] No videos found in data/raw_matches/. Please drop a video in there first!")
        return

    test_video = str(video_files[0])
    print(f"[+] Testing Clipper on: {test_video}")

    # 2. Initialize your new Dual-Model AutoClipper
    try:
        clipper = AutoClipper(
            ball_model_path="models/Ball_Detection_Model.pt", 
            bat_model_path="models/Bat_Detection_Model.pt",
            output_dir="data/delivery_clips"
        )
    except Exception as e:
        print(f"[-] Failed to initialize clipper: {e}")
        return

    # 3. Process the match and slice the 1-second clips
    print("[+] Starting the video segmentation process...")
    generated_clips = clipper.process_match(test_video, clip_duration_sec=1.0)

    # 4. Output the results
    print(f"\n[+] Clipper finished successfully! Generated {len(generated_clips)} clips:")
    for clip in generated_clips:
        print(f"  -> {clip}")

if __name__ == "__main__":
    main()