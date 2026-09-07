import os
import glob
import cv2
from src.preprocessing.frame_sampler import extract_frames

def process_all_deliveries(input_dir="data/delivery_clips", output_dir="data/extracted_data"):
    # Find all .mp4 files inside input_dir
    video_files = glob.glob(os.path.join(input_dir, "*.mp4"))
    
    total_videos = len(video_files)
    print(f"Found {total_videos} video files to process.")
    
    for idx, video_path in enumerate(video_files, 1):
        # Extract the base filename without the extension
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        
        print(f"Processing {idx}/{total_videos}: {base_name}...")
        
        # Create a dedicated output folder path
        shot_frames_dir = os.path.join(output_dir, base_name, "shot_frames")
        os.makedirs(shot_frames_dir, exist_ok=True)
        
        # Call extract_frames to get the 30 padded arrays
        frames = extract_frames(video_path)
        
        if frames is None:
            print(f"Warning: Failed to extract frames from {video_path}")
            continue
            
        # Save those 30 arrays as frame_00.jpg through frame_29.jpg inside the new folder
        for i, frame in enumerate(frames):
            frame_filename = f"frame_{i:02d}.jpg"
            frame_path = os.path.join(shot_frames_dir, frame_filename)
            cv2.imwrite(frame_path, frame)
            
    print("Batch processing complete.")
