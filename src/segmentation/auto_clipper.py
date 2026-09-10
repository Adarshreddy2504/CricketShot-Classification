import cv2
import pathlib
import logging
from ultralytics import YOLO
from tqdm import tqdm  # The progress bar library

# Suppress standard info spam, only show warnings or errors
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

class AutoClipper:
    def __init__(self, 
                 ball_model_path="models/Ball_Detection_Model.pt", 
                 bat_model_path="models/Bat_Detection_Model.pt",
                 output_dir="data/delivery_clips"):
        """Initializes the AutoClipper using fine-tuned YOLO models."""
        self.output_dir = pathlib.Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            self.ball_model = YOLO(ball_model_path)
            self.bat_model = YOLO(bat_model_path)
        except Exception as e:
            logging.error(f"Failed to load fine-tuned YOLO models: {e}")
            raise

    def process_match(self, video_path, clip_duration_sec=1.0):
        video_path = pathlib.Path(video_path)
        if not video_path.exists():
            logging.error(f"Video path does not exist: {video_path}")
            return []

        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0: fps = 30.0

        clip_frame_length = int(clip_duration_sec * fps)
        generated_clips = []
        
        print(f"\n[+] Processing Match Footage: {video_path.name}")

        frame_idx = 0
        clip_counter = 1
        cooldown_frames = 0 

        # Initialize the dynamic progress bar
        with tqdm(total=total_frames, desc="🎥 Slicing Video", unit="frame", 
                  bar_format="{l_bar}{bar:40}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}") as pbar:
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if cooldown_frames > 0:
                    cooldown_frames -= 1
                    frame_idx += 1
                    pbar.update(1)  # Step the progress bar forward
                    continue

                if frame_idx % 5 == 0:
                    ball_results = self.ball_model(frame, verbose=False)
                    bat_results = self.bat_model(frame, verbose=False)
                    
                    if len(ball_results[0].boxes) > 0 and len(bat_results[0].boxes) > 0:
                        clip_filename = self.output_dir / f"delivery_{clip_counter:03d}.mp4"
                        
                        success = self._extract_clip(str(video_path), frame_idx, clip_frame_length, fps, clip_filename)
                        
                        if success:
                            generated_clips.append(str(clip_filename))
                            clip_counter += 1
                            cooldown_frames = clip_frame_length
                            
                            # Dynamically update the text on the right side of the bar
                            pbar.set_postfix({"Shots Extracted": len(generated_clips)})
                
                frame_idx += 1
                pbar.update(1)  # Step the progress bar forward

        cap.release()
        print(f"\n[+] Success! Generated {len(generated_clips)} one-second shot videos.\n")
        return generated_clips

    def _extract_clip(self, source_video, start_frame, duration_frames, fps, output_path):
        """Slices a specific window of frames and saves it as an MP4 clip."""
        try:
            reader = cv2.VideoCapture(source_video)
            reader.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            width = int(reader.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            
            frames_written = 0
            while frames_written < duration_frames:
                ret, frame = reader.read()
                if not ret: break
                writer.write(frame)
                frames_written += 1
                
            writer.release()
            reader.release()
            return frames_written > 0
        except Exception as e:
            return False