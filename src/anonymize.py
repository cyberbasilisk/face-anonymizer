"""
Anonymizer:
    This module takes a detected region of interest and returns an obscured version of the same size. 
    Returns a blurred, pixelated or black box censor.
"""
from __future__ import annotations

import cv2

from detect import Box

# Censor options
METHODS = ("blur", "pixelate", "box")


def blur(roi):
    
    k = (max(roi.shape[:2]) // 2) | 1  
    k = max(k, 3)                      
    
    return cv2.GaussianBlur(roi, (k, k), 0)


def pixelate(roi, blocks: int = 10):
    
    h, w = roi.shape[:2]
    
    bw = max(1, blocks)
    bh = max(1, int(blocks * h / w)) 

  
    small = cv2.resize(roi, (bw, bh), interpolation=cv2.INTER_LINEAR)
   
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def box(roi):
    
    return roi * 0



CENSORS = {"blur": blur, "pixelate": pixelate, "box": box}


def anonymize(frame, boxes: list[Box], method: str = "pixelate"):
    
    # Obscure every bounding box in frame using `method` and return the frame.
    
    if method not in CENSORS:
        raise ValueError(f"Unknown method '{method}'. Choose from {METHODS}.")
    
    fn = CENSORS[method]

    for b in boxes:
      
        roi = frame[b.y1:b.y2, b.x1:b.x2]
        
        if roi.size:
            frame[b.y1:b.y2, b.x1:b.x2] = fn(roi)
    
    return frame