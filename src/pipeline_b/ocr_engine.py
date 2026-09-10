import cv2
import easyocr
import re
import numpy as np
from collections import Counter
import logging

# Configure basic logging for production visibility
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ScoreboardOCR:
    def __init__(self, gpu=True):
        """
        Initializes the ScoreboardOCR engine.
        Uses EasyOCR configured for the English language.
        
        Args:
            gpu (bool): Whether to use GPU acceleration if available.
        """
        try:
            logging.info("Initializing EasyOCR Engine...")
            self.reader = easyocr.Reader(['en'], gpu=gpu)
            logging.info("EasyOCR Engine initialized successfully.")
        except Exception as e:
            logging.error(f"Failed to initialize EasyOCR: {e}")
            raise

    def slice_roi(self, frame, x, y, w, h):
        """
        Crops the Region of Interest (ROI) from the frame to isolate the text.
        
        Args:
            frame (numpy.ndarray): The full video frame.
            x (int): Top-left x-coordinate.
            y (int): Top-left y-coordinate.
            w (int): Width of the ROI.
            h (int): Height of the ROI.
            
        Returns:
            numpy.ndarray: The cropped ROI image, or None if coordinates are invalid.
        """
        if frame is None or frame.size == 0:
            logging.warning("Empty frame provided for ROI slicing.")
            return None

        img_h, img_w = frame.shape[:2]
        
        # Ensure coordinates are within bounds
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(img_w, x + w)
        y2 = min(img_h, y + h)
        
        # Verify if the resulting crop has valid dimensions
        if x2 <= x1 or y2 <= y1:
            logging.warning(f"Invalid ROI coordinates: x={x}, y={y}, w={w}, h={h} for frame {img_w}x{img_h}")
            return None
            
        cropped_roi = frame[y1:y2, x1:x2]
        return cropped_roi

    def preprocess_image(self, roi_img):
        """
        Applies Grayscale, CLAHE, and Otsu's thresholding to prepare the image for optimal OCR.
        
        Args:
            roi_img (numpy.ndarray): The cropped RGB/BGR image.
            
        Returns:
            numpy.ndarray: The preprocessed binary image.
        """
        if roi_img is None or roi_img.size == 0:
            return None

        try:
            # 1. Grayscale Conversion
            gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
            
            # 2. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced_gray = clahe.apply(gray)
            
            # 3. Apply Otsu's Thresholding to binarize (text vs background)
            _, binarized = cv2.threshold(enhanced_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            return binarized
        except Exception as e:
            logging.error(f"Error during image preprocessing: {e}")
            return None

    def process_single_frame(self, frame, x, y, w, h):
        """
        Processes a single frame: extracts ROI, preprocesses, and runs OCR.
        
        Args:
            frame (numpy.ndarray): The raw video frame.
            x, y, w, h (int): Coordinates and dimensions for the ROI.
            
        Returns:
            str: The cleaned extracted text, or None if invalid.
        """
        # Step 1: Slice ROI
        roi = self.slice_roi(frame, x, y, w, h)
        if roi is None:
            return None
            
        # Step 2: Preprocess Image
        processed_roi = self.preprocess_image(roi)
        if processed_roi is None:
            return None
            
        # Step 3: OCR Inference
        try:
            # detail=0 returns a list of strings
            results = self.reader.readtext(processed_roi, detail=0)
        except Exception as e:
            logging.error(f"EasyOCR extraction failed: {e}")
            return None
            
        if not results:
            return None
            
        # Step 4: Text Cleaning
        # Join multiple detected text blocks in the ROI
        raw_text = " ".join(results)
        
        # Keep only alphabetic characters and spaces using Regex
        cleaned_text = re.sub(r'[^a-zA-Z\s]', '', raw_text)
        
        # Normalize whitespace and convert to uppercase for consistency
        cleaned_text = " ".join(cleaned_text.split()).upper()
        
        # Filter out empty or extremely short noise strings
        if len(cleaned_text) < 3:
            return None
            
        return cleaned_text

    def temporal_voting(self, frames, x, y, w, h):
        """
        Processes a sequence of frames and returns the most frequent extracted text.
        Gracefully handles scenarios where the scoreboard is temporarily obscured.
        
        Args:
            frames (list of numpy.ndarray): A list of sequential video frames (e.g., 5 frames).
            x, y, w, h (int): Bounding box coordinates for the nameplate ROI.
            
        Returns:
            str: The agreed upon text string (mode) or None if no valid text was found.
        """
        if not frames or not isinstance(frames, list):
            logging.warning("No frames provided for temporal voting.")
            return None

        candidates = []
        
        # Iterate over all frames in the batch
        for i, frame in enumerate(frames):
            try:
                text = self.process_single_frame(frame, x, y, w, h)
                if text:
                    candidates.append(text)
            except Exception as e:
                logging.error(f"Error processing frame {i} in temporal batch: {e}")
                continue
                
        # If no valid text was extracted across any frames (e.g., scoreboard is missing)
        if not candidates:
            logging.info("Temporal Voting: Scoreboard possibly missing or text unreadable in this sequence.")
            return None
            
        # Find the mode (most frequent item)
        counts = Counter(candidates)
        most_frequent_text, frequency = counts.most_common(1)[0]
        
        logging.debug(f"Temporal Voting Winner: '{most_frequent_text}' (Frequency: {frequency}/{len(frames)})")
        
        return most_frequent_text
