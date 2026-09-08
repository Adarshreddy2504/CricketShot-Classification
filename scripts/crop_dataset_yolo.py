import os
import cv2
import torch
from ultralytics import YOLO
from tqdm import tqdm
from pathlib import Path

def get_center_crop_box(frame_width, frame_height, crop_width, crop_height):
    x1 = max(0, (frame_width - crop_width) // 2)
    y1 = max(0, (frame_height - crop_height) // 2)
    x2 = min(frame_width, x1 + crop_width)
    y2 = min(frame_height, y1 + crop_height)
    return [x1, y1, x2, y2]

def expand_box(box, margin_ratio, frame_width, frame_height):
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    
    margin_x = int(w * margin_ratio)
    margin_y = int(h * margin_ratio)
    
    nx1 = max(0, x1 - margin_x)
    ny1 = max(0, y1 - margin_y)
    nx2 = min(frame_width, x2 + margin_x)
    ny2 = min(frame_height, y2 + margin_y)
    
    return [int(nx1), int(ny1), int(nx2), int(ny2)]

def process_video(video_path, output_path, model, device):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Failed to open {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps != fps: # Handle NaN or 0
        fps = 30.0
        
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    frames = []
    bboxes = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frames.append(frame)
        
        # Run YOLO detection for class 0 ('person')
        results = model.predict(frame, classes=[0], verbose=False, device=device)
        
        best_box = None
        max_area = -1
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # box format: xyxy
                xyxy = box.xyxy[0].cpu().numpy()
                area = (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])
                if area > max_area:
                    max_area = area
                    best_box = xyxy
                    
        bboxes.append(best_box)
        
    cap.release()
    
    if not frames:
        return
        
    # Forward fill missing boxes
    valid_box = None
    for i in range(len(bboxes)):
        if bboxes[i] is not None:
            valid_box = bboxes[i]
        elif valid_box is not None:
            bboxes[i] = valid_box
            
    # Backward fill if started with None
    valid_box = None
    for i in range(len(bboxes)-1, -1, -1):
        if bboxes[i] is not None:
            valid_box = bboxes[i]
        elif valid_box is not None:
            bboxes[i] = valid_box
            
    # Default fallback if absolutely no person detected in any frame
    default_crop_width = min(224, frame_width)
    default_crop_height = min(224, frame_height)
    center_box = get_center_crop_box(frame_width, frame_height, default_crop_width, default_crop_height)
    
    for i in range(len(bboxes)):
        if bboxes[i] is None:
            bboxes[i] = center_box
            
    # Add margin (15%)
    expanded_bboxes = [expand_box(box, 0.15, frame_width, frame_height) for box in bboxes]
    
    # Calculate maximum dimensions for consistent video writing
    max_w = 0
    max_h = 0
    for box in expanded_bboxes:
        w = box[2] - box[0]
        h = box[3] - box[1]
        max_w = max(max_w, w)
        max_h = max(max_h, h)
        
    max_w = int(max_w) if int(max_w) % 2 == 0 else int(max_w) + 1
    max_h = int(max_h) if int(max_h) % 2 == 0 else int(max_h) + 1
    
    if max_w == 0 or max_h == 0:
        return
        
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (max_w, max_h))
    
    for i, frame in enumerate(frames):
        box = expanded_bboxes[i]
        x1, y1, x2, y2 = box
        cropped = frame[y1:y2, x1:x2]
        
        # Resize to max dimensions to keep video writer happy
        cropped_resized = cv2.resize(cropped, (max_w, max_h))
        out.write(cropped_resized)
        
    out.release()

def main():
    input_base = Path("data/extracted_hf_dataset/cricketshot/train")
    output_base = Path("data/cropped_dataset/train")
    
    if not input_base.exists():
        print(f"Input path {input_base} does not exist.")
        return
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load the lightweight pre-trained model
    model = YOLO("models/yolov8n.pt")
    
    video_files = []
    for cls_dir in input_base.iterdir():
        if cls_dir.is_dir():
            for vid in cls_dir.glob("*.mp4"):
                video_files.append((vid, cls_dir.name))
            for vid in cls_dir.glob("*.avi"):
                video_files.append((vid, cls_dir.name))
                
    print(f"Found {len(video_files)} videos to process.")
    
    for vid_path, cls_name in tqdm(video_files, desc="Processing videos"):
        out_dir = output_base / cls_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / vid_path.name
        
        process_video(vid_path, out_path, model, device)

if __name__ == "__main__":
    main()
