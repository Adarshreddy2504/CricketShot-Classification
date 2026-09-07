import os
import torch
import cv2
import json
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from src.models.efficientnet_gru import CricShotEfficientGRU

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
    test_dir = 'data/extracted_hf_dataset/cricketshot/tiny_train'
    mapping_file = 'data/class_mapping.json'
    weights_file = 'data/cricshot_sota.pth'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(mapping_file, 'r') as f:
        idx_to_class = {v: k for k, v in json.load(f).items()}

    model = CricShotEfficientGRU(num_classes=10).to(device)
    model.load_state_dict(torch.load(weights_file, map_location=device, weights_only=True))
    model.eval()

    correct = 0
    total = 0
    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    if os.path.exists(test_dir):
        for class_name in os.listdir(test_dir):
            if class_name.startswith('.'):
                continue
                
            class_path = os.path.join(test_dir, class_name)
            if not os.path.isdir(class_path):
                continue
                
            videos = [f for f in os.listdir(class_path) if f.endswith(('.mp4', '.avi'))]
            
            for video_name in tqdm(videos, desc=f"Testing {class_name}"):
                video_path = os.path.join(class_path, video_name)
                tensor = preprocess_video(video_path)
                
                with torch.no_grad():
                    outputs = model(tensor)
                    
                pred_idx = torch.argmax(outputs).item()
                predicted_class = idx_to_class[pred_idx]
                
                total += 1
                class_total[class_name] += 1
                
                if predicted_class == class_name:
                    correct += 1
                    class_correct[class_name] += 1

    overall_acc = (correct / total) * 100 if total > 0 else 0

    print("\n========================================")
    print("--- OFFICIAL MODEL EVALUATION REPORT ---")
    print(f"Total Test Videos Analyzed: {total}")
    print(f"Overall Accuracy: {overall_acc:.2f}% ({correct}/{total})")
    print("\n--- PER-CLASS ACCURACY ---")
    for cls_name, cls_tot in class_total.items():
        cls_acc = (class_correct[cls_name] / cls_tot) * 100 if cls_tot > 0 else 0
        print(f"{cls_name}: {cls_acc:.2f}% ({class_correct[cls_name]}/{cls_tot})")
    print("========================================\n")
