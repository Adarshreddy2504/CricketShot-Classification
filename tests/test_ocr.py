import cv2
from src.pipeline_b.ocr_engine import ScoreboardOCR

def main():
    print("Initializing OCR Engine...")
    ocr = ScoreboardOCR()
    
    test_image_path = "sample_frame.jpg"
    print(f"Reading test image: {test_image_path}")
    
    frame = cv2.imread(test_image_path)
    
    if frame is None:
        print(f"Failed to load image at {test_image_path}. Please ensure the file exists.")
        return
        
    print("Extracting raw text from the scoreboard region...")
    raw_text = ocr.extract_text(frame)
    print(f"\nRaw text detected:\n{raw_text}\n")
    
    print("Identifying active batsman...")
    batsman = ocr.find_batsman(raw_text)
    
    if batsman:
        print(f"Active Batsman Detected: {batsman}")
    else:
        print("Could not find an active batsman (denoted by '*') in the text.")

if __name__ == "__main__":
    main()
