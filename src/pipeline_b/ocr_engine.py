import cv2
import easyocr
import re

class ScoreboardOCR:
    def __init__(self):
        self.reader = easyocr.Reader(['en'], gpu=True)
        
    def get_scoreboard_roi(self, frame):
        height, width = frame.shape[:2]
        crop_y = int(height * 0.8) # Keep the bottom 20%
        cropped_frame = frame[crop_y:height, 0:width]
        return cropped_frame
        
    def extract_text(self, frame):
        roi = self.get_scoreboard_roi(frame)
        text_list = self.reader.readtext(roi, detail=0)
        return text_list
        
    def find_batsman(self, text_list):
        ignore_words = [
            'RUN', 'RATE', 'NEED', 'BALLS', 'OFF', 'TARGET', 'WIN', 'OVERS', 
            'THIS', 'LAST', 'LIO', 'FAL', 'DC', 'MI', 'KKR', 'GT', 'IND', 'AUS',
            'INDIA', 'BANGLADESH', 'DELHI', 'CAPITALS', 'ICNPIALS', 'ICAPHATS', 
            'ICARINTS', 'SPORTS', 'LIVE', 'TATA', 'IPL'
        ]
        valid_names = []
        markers = ['*', '▶', '>', '<', '•', 'P ']
        
        for text in text_list:
            text = text.strip().upper()
            
            if any(char.isdigit() for char in text):
                continue
                
            words = text.split()
            if any(word in ignore_words for word in words):
                continue
                
            if len(text) <= 2:
                continue
                
            marker_found = any(marker in text for marker in markers)
            
            cleaned_string = re.sub(r'[^A-Z\s]', '', text).strip()
            
            if marker_found:
                return cleaned_string
            else:
                valid_names.append(cleaned_string)

        if valid_names:
            return valid_names[0]
        return "Could Not Detect"
