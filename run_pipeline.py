import os
import glob
import json
import yaml
import argparse

# Import modules from our project
from src.segmentation.auto_clipper import process_video as extract_cricket_deliveries
from src.pipeline_b.ocr_engine import extract_striker_name

def mock_predict_shot(video_path):
    """
    Mock prediction representing Pipeline A (CricketShotClassifier).
    In a fully integrated setup, this would load model.py, run the clip through 
    the EfficientNet+GRU architecture, and return the predicted class.
    """
    # Hardcoded mock response for now
    return "Cover Drive", 0.95

def run_pipeline(video_path, config_path="config.yaml", output_json="outputs/match_analytics.json"):
    print("="*50)
    print("🏏 CRICKET VIDEO ANALYSIS PIPELINE STARTED 🏏")
    print("="*50)
    
    # 1. Load config
    print(f"\n[*] Loading configuration from {config_path}...")
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print("    [+] Config loaded successfully.")
    except Exception as e:
        print(f"    [-] Failed to load config: {e}")
        return

    # 2 & 3. Accept raw match .mp4 and extract deliveries
    clip_dir = "data/delivery_clips/"
    print(f"\n[*] Step 1: Extracting delivery clips from {video_path}")
    print(f"    [*] Destination: {clip_dir}")
    
    extract_cricket_deliveries(video_path, clip_dir)
    
    # 4. Iterate over every clip in data/delivery_clips/
    print("\n[*] Step 2: Processing extracted clips...")
    
    # Get all mp4 files in clip_dir
    clips = glob.glob(os.path.join(clip_dir, "*.mp4"))
    
    if not clips:
        print("    [-] No clips found in delivery_clips directory. Pipeline aborted.")
        return
        
    print(f"    [+] Found {len(clips)} clips to process.")
    
    results = []
    
    for i, clip_path in enumerate(clips, 1):
        clip_name = os.path.basename(clip_path)
        print(f"\n    ---> Processing Clip {i}/{len(clips)}: {clip_name}")
        
        # 5. OCR pipeline for player name (Pipeline B)
        print("         [~] Running Pipeline B: OCR Striker Extraction...")
        player_name = extract_striker_name(clip_path, config_path)
        if player_name:
            print(f"         [+] Active Striker: {player_name}")
        else:
            print("         [-] Active Striker: Unknown")
            
        # 6. Model prediction (Pipeline A)
        print("         [~] Running Pipeline A: Shot Classification...")
        shot_class, confidence = mock_predict_shot(clip_path)
        print(f"         [+] Predicted Shot: {shot_class} (Confidence: {confidence:.2f})")
        
        # 7. Aggregate results
        results.append({
            "clip_filename": clip_name,
            "player_name": player_name,
            "predicted_shot": shot_class,
            "confidence": float(confidence)
        })
        
    # 8. Save the final list as formatted JSON
    print("\n[*] Step 3: Saving Analytics...")
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"    [+] Results saved to {output_json}")
    
    print("\n" + "="*50)
    print("🏁 PIPELINE COMPLETED SUCCESSFULLY 🏁")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Main Cricket Analysis Pipeline Controller")
    parser.add_argument("video_path", type=str, help="Path to the raw match .mp4 video")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--output", type=str, default="outputs/match_analytics.json", help="Path to output JSON")
    
    args = parser.parse_args()
    run_pipeline(args.video_path, args.config, args.output)
