import cv2
import yaml
import pytesseract
import re
import os

def load_config(config_path="config.yaml"):
    """Loads configuration parameters from a YAML file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config: {e}")
        return None

def extract_striker_name(video_path, config_path="config.yaml"):
    """
    Extracts the active striker's name from a cricket broadcast scorebug.
    Reads the first frame, crops to the region of interest, applies thresholding,
    and uses OCR & regex to isolate the player's name.
    """
    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        return None
        
    # Load cropping configuration
    config = load_config(config_path)
    if not config:
        bottom_percent = 20 # Fallback
    else:
        bottom_percent = config.get("pipeline_b", {}).get("ocr_bbox", {}).get("bottom_percent", 20)

    # 1. Read first frame using OpenCV
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret or frame is None:
        print(f"Failed to read frame from {video_path}")
        return None
        
    # 2. Crop the frame to the bottom region (Region of Interest)
    height, width = frame.shape[:2]
    crop_y = int(height * (100 - bottom_percent) / 100)
    cropped_frame = frame[crop_y:height, 0:width]
    
    # 3. Convert to grayscale and apply Otsu's thresholding
    gray = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    # 4. Pass the processed image matrix to pytesseract
    try:
        ocr_text = pytesseract.image_to_string(thresh)
    except Exception as e:
        print(f"Tesseract OCR encountered an error: {e}")
        return None
        
    if not ocr_text:
        return None
        
    # 5. Use Regex pattern to isolate the active batsman's name
    # First pattern: looks for a name followed by an asterisk (common indicator for active striker)
    # Examples: "V Kohli *", "S. Smith*"
    asterisk_pattern = r"([A-Za-z\s\.-]+)\*"
    match = re.search(asterisk_pattern, ocr_text)
    
    if match:
        return match.group(1).strip()
        
    # Fallback pattern: look for a capitalized name just before numbers (score data)
    fallback_pattern = r"([A-Z][a-zA-Z\s\.-]+?)\s+\d+"
    fallback_match = re.search(fallback_pattern, ocr_text)
    
    if fallback_match:
        # Return cleaned player name
        return fallback_match.group(1).strip()
        
    return None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="OCR Engine to extract active striker name")
    parser.add_argument("video_path", type=str, help="Path to the cricket clip (.mp4)")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the config file")
    
    args = parser.parse_args()
    
    striker_name = extract_striker_name(args.video_path, args.config)
    
    if striker_name:
        print(f"Active Striker Detected: {striker_name}")
    else:
        print("Could not detect active striker from the scorebug.")
