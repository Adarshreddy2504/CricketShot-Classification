import cv2
import pathlib
import logging
import easyocr
from ultralytics import YOLO
from tqdm import tqdm

# =========================================================
# LOGGING
# =========================================================
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# =========================================================
# AUTO CLIPPER
# =========================================================
class AutoClipper:
    def __init__(
        self,
        ball_model_path="models/Ball_Detection_Model.pt",
        bat_model_path="models/Bat_Detection_Model.pt",
        output_dir="data/delivery_clips",
        confidence=0.30,
        use_ocr=True,
        ocr_gpu=False
    ):
        self.output_dir = pathlib.Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.confidence = confidence
        self.use_ocr = use_ocr
        self.ocr_gpu = ocr_gpu

        # 1. Load YOLO Models
        print("[*] Loading Ball YOLO...", flush=True)
        self.ball_model = YOLO(ball_model_path)
        
        print("[*] Loading Bat YOLO...", flush=True)
        self.bat_model = YOLO(bat_model_path)

        # 2. Load EasyOCR (Optional Scoreboard Validation)
        if self.use_ocr:
            print("[*] Loading EasyOCR Scoreboard Validator...", flush=True)
            self.ocr = easyocr.Reader(["en"], gpu=self.ocr_gpu)
            print("[+] All models successfully loaded.", flush=True)
        else:
            print("[+] YOLO models loaded. (OCR Disabled).", flush=True)

    # =====================================================
    # SCOREBOARD PRESENCE CHECK
    # =====================================================
    def scoreboard_present(self, frame):
        height, width = frame.shape[:2]

        # Crop bottom 20% of frame where scoreboard resides
        y_start = int(height * 0.80)
        scoreboard_crop = frame[y_start:height, 0:width]

        try:
            results = self.ocr.readtext(scoreboard_crop, detail=1, paragraph=False)
        except Exception as e:
            logging.error(f"OCR Error: {e}")
            return False

        valid_detections = 0
        for detection in results:
            if len(detection) < 3:
                continue
                
            text = detection[1].strip()
            confidence = float(detection[2])

            if confidence >= 0.30 and len(text) >= 2:
                valid_detections += 1

        return valid_detections >= 1

    # =====================================================
    # PROCESS FULL MATCH
    # =====================================================
    def process_match(self, video_path, clip_duration_sec=1.0):
        video_path = pathlib.Path(video_path)

        if not video_path.exists():
            print(f"[ERROR] Video does not exist: {video_path}", flush=True)
            return []

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print("[ERROR] Could not open video.", flush=True)
            return []

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0: fps = 30.0
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        clip_frame_length = int(clip_duration_sec * fps)
        
        generated_clips = []
        frame_idx = 0
        clip_counter = 1
        cooldown_frames = 0

        with tqdm(total=total_frames, desc="🎥 Slicing Video", unit="frame",
                  bar_format="{l_bar}{bar:40}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}") as pbar:

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break

                if cooldown_frames > 0:
                    cooldown_frames -= 1
                    frame_idx += 1
                    pbar.update(1)
                    continue

                # Process every 5th frame to save compute
                if frame_idx % 5 == 0:
                    ball_results = self.ball_model(frame, conf=self.confidence, verbose=False)
                    bat_results = self.bat_model(frame, conf=self.confidence, verbose=False)

                    ball_detected = len(ball_results[0].boxes) > 0
                    bat_detected = len(bat_results[0].boxes) > 0

                    if ball_detected and bat_detected:
                        
                        # Validate scoreboard presence if OCR is enabled
                        scoreboard_ok = True
                        if self.use_ocr:
                            scoreboard_ok = self.scoreboard_present(frame)

                        if scoreboard_ok:
                            clip_filename = self.output_dir / f"delivery_{clip_counter:03d}.mp4"
                            
                            success = self._extract_clip(
                                source_video=str(video_path),
                                start_frame=frame_idx,
                                duration_frames=clip_frame_length,
                                fps=fps,
                                output_path=clip_filename
                            )

                            if success:
                                generated_clips.append(str(clip_filename))
                                clip_counter += 1
                                cooldown_frames = clip_frame_length
                                pbar.set_postfix({"Extracted": len(generated_clips)})

                frame_idx += 1
                pbar.update(1)

        cap.release()
        print(f"\n[+] Completed. Extracted {len(generated_clips)} clips.", flush=True)
        return generated_clips

    # =====================================================
    # EXTRACT CLIP
    # =====================================================
    def _extract_clip(self, source_video, start_frame, duration_frames, fps, output_path):
        try:
            reader = cv2.VideoCapture(source_video)
            if not reader.isOpened(): return False

            reader.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            width = int(reader.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT))

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
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

        except Exception:
            return False

# =========================================================
# MAIN (For Standalone Testing)
# =========================================================
if __name__ == "__main__":
    print("=== AUTO CLIPPER STARTED ===", flush=True)
    clipper = AutoClipper(use_ocr=True)
    video_path = input("\nEnter full video path: ").strip()
    clips = clipper.process_match(video_path, clip_duration_sec=1.0)