import cv2
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class VideoPreprocessor:
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size

    def process_frame(self, frame):
        """Resizes and normalizes a single video frame for model ingestion."""
        if frame is None or frame.size == 0:
            return None
        
        # Resize to match EfficientNetV2 input dimensions
        resized = cv2.resize(frame, self.target_size)
        
        # Normalize pixel values to [0, 1] range
        normalized = resized.astype(np.float32) / 255.0
        return normalized

    def process_clip(self, video_path, num_frames=15):
        """Extracts a uniform sequence of frames from a delivery clip."""
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames <= 0:
            cap.release()
            return []

        # Calculate step size to sample 'num_frames' uniformly across the clip
        step = max(1, total_frames // num_frames)
        processed_frames = []

        for i in range(num_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
            ret, frame = cap.read()
            if not ret:
                break
            
            processed_frame = self.process_frame(frame)
            processed_frames.append(processed_frame)

        cap.release()
        return np.array(processed_frames)