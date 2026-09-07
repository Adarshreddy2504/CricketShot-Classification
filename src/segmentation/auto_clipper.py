import os
import cv2
import numpy as np
from scenedetect import detect, ContentDetector, split_video_ffmpeg
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

def is_live_delivery(frame):
    """
    Checks if a frame is a live delivery by running YOLO inference and applying 
    geometric rules on the detected people bounding boxes.
    """
    if frame is None:
        return False
        
    h, w, _ = frame.shape
    
    results = model(frame, classes=[0], verbose=False)
    boxes = results[0].boxes.xyxy
    
    # Rule A (Population): Total people must be between 4 and 15.
    num_people = len(boxes)
    if num_people < 4 or num_people > 15:
        return False
        
    central_column_count = 0
    top_zone_count = 0
    bottom_zone_count = 0
    
    for box in boxes:
        x1, y1, x2, y2 = box.tolist()
        box_height = y2 - y1
        
        # Rule B (Height Limit): Every person's height < 0.40 * h
        if box_height >= h * 0.40:
            return False
            
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2
        
        # Rule C (Central Column): Between 0.25 * w and 0.75 * w
        if w * 0.25 <= x_center <= w * 0.75:
            central_column_count += 1
            
        # Rule D setup (Two-Zone Crease Anchor)
        if y_center <= h * 0.48:
            top_zone_count += 1
        if y_center >= h * 0.48:
            bottom_zone_count += 1
            
    # Rule C Check: At least 2 people in this central column.
    if central_column_count < 2:
        return False
        
    # Rule D Check:
    if top_zone_count < 1 or bottom_zone_count < 1:
        return False
            
    return True

def extract_cricket_deliveries(video_path, output_dir):
    """
    Finds scenes, filters by duration (2.5-12.0s), evaluates frames with YOLO,
    and exports valid scenes to output_dir.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    print(f"Detecting scenes in {video_path}...")
    print("Note: A 3-hour video may take 20+ minutes to process.")
    
    scenes = detect(video_path, ContentDetector(threshold=27.0), show_progress=True)
    
    valid_scenes = []
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return len(scenes), 0
        
    for scene in scenes:
        start_time, end_time = scene
        duration_sec = end_time.get_seconds() - start_time.get_seconds()
        
        # Scene Duration: 2.5 to 12.0 seconds
        if 2.5 <= duration_sec <= 12.0:
            start_frames = start_time.get_frames()
            end_frames = end_time.get_frames()
            total_frames = end_frames - start_frames
            
            # Early-Shift Polling: 15% and 35%
            frame_indices = [
                int(start_frames + total_frames * 0.15),
                int(start_frames + total_frames * 0.35)
            ]
            
            all_valid = True
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                
                if not ret or not is_live_delivery(frame):
                    all_valid = False
                    break
                    
            if all_valid:
                valid_scenes.append(scene)
                
    cap.release()
    
    print(f"Found {len(valid_scenes)} valid delivery scenes.")
    
    if valid_scenes:
        split_video_ffmpeg(
            video_path, 
            valid_scenes, 
            output_file_template=os.path.join(output_dir, "$VIDEO_NAME-Scene-$SCENE_NUMBER.mp4")
        )
        
    return len(scenes), len(valid_scenes)
