import os
import cv2
from src.pipeline_b.ocr_engine import ScoreboardOCR

DEMO_DIR = "data/demo_frames"

def main():
    if not os.path.exists(DEMO_DIR):
        os.makedirs(DEMO_DIR)
        print(f"Created directory '{DEMO_DIR}'. Please place sample images (.jpg, .png) inside this folder and run the script again.")
        return

    # Filter for valid image extensions
    valid_extensions = {".jpg", ".jpeg", ".png"}
    image_files = [f for f in os.listdir(DEMO_DIR) if os.path.splitext(f)[1].lower() in valid_extensions]

    if not image_files:
        print(f"No images found in '{DEMO_DIR}'. Please place sample images (.jpg, .png) inside this folder and run the script again.")
        return

    print("Initializing ScoreboardOCR...")
    ocr = ScoreboardOCR()
    
    results = []

    print("\n--- Processing Images ---")
    for filename in image_files:
        img_path = os.path.join(DEMO_DIR, filename)
        frame = cv2.imread(img_path)
        
        if frame is None:
            print(f"[Warning] Failed to read {filename}. Skipping.")
            continue
            
        raw_text = ocr.extract_text(frame)
        batsman = ocr.find_batsman(raw_text)
        
        print(f"[Processing] {filename} -> Active Batsman: {batsman}")
        results.append((filename, batsman))

    # Print summary table
    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)
    print(f"| {'File Name':<20} | {'Detected Batsman':<20} |")
    print("-" * 47)
    for filename, batsman in results:
        batsman_str = str(batsman) if batsman is not None else "None"
        print(f"| {filename:<20} | {batsman_str:<20} |")
    print("-" * 47)

if __name__ == "__main__":
    main()
