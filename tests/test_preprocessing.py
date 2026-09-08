import os
import cv2
from src.preprocessing.frame_sampler import extract_frames

def main():
    video_path = "data/raw_matches/test.mp4"
    
    # 3. Verify the file exists
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return
        
    # 4. Create an output directory
    output_dir = "outputs/frame_debug/"
    os.makedirs(output_dir, exist_ok=True)
    
    # 5. Call extract_frames
    print(f"Extracting 30 frames from {video_path}...")
    frames = extract_frames(video_path, num_frames=30, target_size=(224, 224))
    
    # 6. Save each one into the output directory
    # 7. Add print statements to show progress
    print(f"Extracted {len(frames)} frames. Saving to {output_dir}...")
    for i, frame in enumerate(frames):
        out_path = os.path.join(output_dir, f"frame_{i:02d}.jpg")
        cv2.imwrite(out_path, frame)
        print(f"Saved {out_path}")
        
    print("Extraction and saving complete!")

if __name__ == "__main__":
    main()
