import cv2
import os
import numpy as np
from tqdm import tqdm

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        cap.release()
        return None

    # Generate 30 evenly spaced indices
    indices = np.linspace(0, total_frames - 1, 30, dtype=int)
    frames_list = []
    
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            # Handle potential read failures gracefully
            continue
            
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Resize to 224x224
        frame = cv2.resize(frame, (224, 224))
        frames_list.append(frame)
        
    cap.release()
    
    # Check if we successfully read frames
    if not frames_list:
        return None
        
    # Ensure exactly 30 frames in case of read failures
    while len(frames_list) < 30:
        frames_list.append(frames_list[-1])
    frames_list = frames_list[:30]

    # Convert list to NumPy array: (30, 224, 224, 3)
    video_tensor = np.array(frames_list)
    
    # Restructure dimensions from (30, 224, 224, 3) to (30, 3, 224, 224)
    video_tensor = np.transpose(video_tensor, (0, 3, 1, 2))
    
    # Normalize pixel values between 0 and 1
    video_tensor = video_tensor.astype(np.float32) / 255.0
    
    return video_tensor

def main():
    input_dir = 'data/delivery_clips/'
    output_dir = 'data/numpy_tensors/'
    
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(input_dir):
        print(f"Input directory '{input_dir}' does not exist.")
        return

    video_files = [f for f in os.listdir(input_dir) if f.endswith('.mp4')]
    
    if not video_files:
        print(f"No .mp4 files found in '{input_dir}'.")
        return
        
    for filename in tqdm(video_files, desc="Converting to Tensors", unit="clip"):
        video_path = os.path.join(input_dir, filename)
        video_tensor = process_video(video_path)
        
        if video_tensor is not None:
            output_filename = filename.replace('.mp4', '.npy')
            output_path = os.path.join(output_dir, output_filename)
            np.save(output_path, video_tensor)

if __name__ == "__main__":
    main()
