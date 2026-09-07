import cv2
import numpy as np

def resize_with_padding(frame, target_size=(224, 224)):
    """
    Resizes a frame while maintaining aspect ratio and adds black padding.
    """
    h, w = frame.shape[:2]
    tw, th = target_size
    
    scale = min(tw / w, th / h)
    
    nw, nh = int(w * scale), int(h * scale)
    
    resized = cv2.resize(frame, (nw, nh))
    
    padded = np.zeros((th, tw, 3), dtype=np.uint8)
    
    x_offset = (tw - nw) // 2
    y_offset = (th - nh) // 2
    
    padded[y_offset:y_offset+nh, x_offset:x_offset+nw] = resized
    
    return padded
