import os
from src.pipeline_a.auto_clipper import extract_cricket_deliveries

def main():
    video_path = "data/raw_matches/long_match.mp4"
    output_dir = "data/delivery_clips/"
    
    print("=========================================")
    print(f"Starting segmentation test on video: {video_path}")
    print("=========================================")
    
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        print("Please ensure the video file exists before running.")
        return
        
    print(f"Ensuring output directory exists: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nBeginning extraction process...")
    total_scenes, valid_clips = extract_cricket_deliveries(video_path, output_dir)
    print("\nSegmentation test completed successfully!")
    
    if total_scenes > 0:
        rejection_rate = ((total_scenes - valid_clips) / total_scenes) * 100
    else:
        rejection_rate = 0.0
        
    print("=========================================")
    print("      EXTRACTION METRICS SUMMARY         ")
    print("=========================================")
    print(f"Total Camera Cuts Detected : {total_scenes}")
    print(f"Valid Deliveries Extracted : {valid_clips}")
    print(f"Noise Rejection Rate       : {rejection_rate:.2f}%")
    print("=========================================")

if __name__ == "__main__":
    main()
