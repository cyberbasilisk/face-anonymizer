"""
Detector:
    This module takes a frame (numpy image array from opencv), and 
    returns a list of rectangles, one per detected face.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# Pre-trained model
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
)


@dataclass(frozen=True)
class Box:
    """
    A rectangle in pixel coordinates.
    x1, y1 denote the top-left corner and x2, y2 denote the bottom-right corner.
    """

    x1: int
    y1: int
    x2: int
    y2: int


class FaceDetector:
    
    def __init__( self, model_path: str | Path, min_confidence: float = 0.5, padding: float = 0.25,
                video: bool = False,):

        model_path = Path(model_path)

        # Check for model
        if not model_path.exists():
            
            raise FileNotFoundError(f"Model not found: {model_path}\nRun:\n"
                                    f"  curl -L -o {model_path} {MODEL_URL}")

        # Check if image or video
        mode = vision.RunningMode.VIDEO if video else vision.RunningMode.IMAGE

        # Detector settings
        options = vision.FaceDetectorOptions(base_options = mp_python.BaseOptions(model_asset_path=str(model_path)),
                  running_mode=mode, min_detection_confidence=min_confidence)

        # Initialize detector
        self.detector = vision.FaceDetector.create_from_options(options)
        
        self.video = video
        self.padding = padding

    
    def detect(self, frame_bgr, timestamp_ms: int = 0) -> list[Box]:
        
        """
        Return a padded Box for every face found in the frame.

        frame_bgr:    image as loaded by OpenCV (BGR channel order).
        timestamp_ms: frame time in milliseconds. Only used in video mode.
        """
        h, w = frame_bgr.shape[:2]

        # Convert BGR to RGB
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Detection
        if self.video:
            result = self.detector.detect_for_video(mp_img, timestamp_ms)
        else:
            result = self.detector.detect(mp_img)

        boxes: list[Box] = []
        
        for det in result.detections:
            
            # Bounding_box 
            bb = det.bounding_box

            # Pad bounding box
            pad_x = int(bb.width * self.padding)
            pad_y = int(bb.height * self.padding)

            # Clamp to image
            x1 = max(bb.origin_x - pad_x, 0)
            y1 = max(bb.origin_y - pad_y, 0)
            x2 = min(bb.origin_x + bb.width + pad_x, w)
            y2 = min(bb.origin_y + bb.height + pad_y, h)

            # Handle out of frame coordinates
            if x2 > x1 and y2 > y1:
                boxes.append(Box(x1, y1, x2, y2))
        
        return boxes

    # Clean Up
    def close(self) -> None:
        
        self.detector.close()

    def __enter__(self) -> "FaceDetector":
        
        return self

    def __exit__(self, *exc) -> None:
        
        self.close()