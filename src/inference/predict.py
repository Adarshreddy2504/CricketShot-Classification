import torch
import cv2
import json
import os
import numpy as np
from src.models.efficientnet_gru import CricShotEfficientGRU

# Setup Block
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('data/class_mapping.json', 'r') as f:
    idx_to_class = {v: k for k, v in json.load(f).items()}

model = CricShotEfficientGRU(num_classes=10).to(device)
model.load_state_dict(torch.load('data/cricshot_sota.pth', map_location=device, weights_only=True))
model.eval()

def preprocess_video(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Generate 30 evenly spaced indices
    indices = np.linspace(0, total_frames - 1, 30, dtype=int)
    
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            # Fallback for failed frame reads (rare but possible at the end of some videos)
            frame = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Aspect-ratio-preserving resize
            h, w = frame.shape[:2]
            scale = min(224 / w, 224 / h)
            nw, nh = int(w * scale), int(h * scale)
            resized_frame = cv2.resize(frame, (nw, nh))
            
            # Black canvas padding
            padded_frame = np.zeros((224, 224, 3), dtype=np.uint8)
            x_offset = (224 - nw) // 2
            y_offset = (224 - nh) // 2
            padded_frame[y_offset:y_offset+nh, x_offset:x_offset+nw] = resized_frame
            frame = padded_frame
        frames.append(frame)
    cap.release()
    
    # Stack, transpose to (30, 3, 224, 224), normalize / 255.0, cast to np.float32
    frames = np.array(frames)
    frames = frames.transpose((0, 3, 1, 2))
    frames = (frames / 255.0).astype(np.float32)
    
    # ImageNet Normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
    frames = (frames - mean) / std
    
    # Convert to tensor, add batch dimension, move to device
    return torch.tensor(frames).unsqueeze(0).to(device)

if __name__ == '__main__':
    test_video = "data/delivery_clips/long_match-Scene-002.mp4"
    if not os.path.exists(test_video):
        print("Video not found")
        exit()
        
    tensor = preprocess_video(test_video)
    
    with torch.no_grad():
        outputs = model(tensor)
        
    probs = torch.nn.functional.softmax(outputs, dim=1)
    pred_idx = torch.argmax(probs).item()
    confidence = probs[0][pred_idx].item() * 100
    
    print(f"Predicted shot: {idx_to_class[pred_idx]}")
    print(f"Confidence: {confidence:.2f}%")
