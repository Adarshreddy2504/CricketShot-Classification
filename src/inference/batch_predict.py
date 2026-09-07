import torch
import cv2
import json
import os
import csv
import numpy as np
from tqdm import tqdm
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
            # Fallback for failed frame reads
            frame = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (224, 224))
        frames.append(frame)
    cap.release()
    
    # Stack, transpose to (30, 3, 224, 224), normalize / 255.0, cast to np.float32
    frames = np.array(frames)
    frames = frames.transpose((0, 3, 1, 2))
    frames = (frames / 255.0).astype(np.float32)
    
    # Convert to tensor, add batch dimension, move to device
    return torch.tensor(frames).unsqueeze(0).to(device)

if __name__ == '__main__':
    video_dir = 'data/delivery_clips'
    output_csv = 'data/shot_predictions.csv'
    mapping_file = 'data/class_mapping.json'
    weights_file = 'data/cricshot_sota.pth'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(mapping_file, 'r') as f:
        idx_to_class = {v: k for k, v in json.load(f).items()}

    model = CricShotEfficientGRU(num_classes=10).to(device)
    model.load_state_dict(torch.load(weights_file, map_location=device, weights_only=True))
    model.eval()

    video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]

    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Video_File', 'Predicted_Shot', 'Confidence'])
        
        for video_name in tqdm(video_files, desc="Processing Videos"):
            video_path = os.path.join(video_dir, video_name)
            tensor = preprocess_video(video_path)
            
            with torch.no_grad():
                outputs = model(tensor)
            
            probs = torch.nn.functional.softmax(outputs, dim=1)
            pred_idx = torch.argmax(probs).item()
            confidence = probs[0][pred_idx].item() * 100
            
            predicted_class = idx_to_class[pred_idx]
            writer.writerow([video_name, predicted_class, f"{confidence:.2f}%"])

    print(f"Batch prediction complete. Results saved to {output_csv}")
