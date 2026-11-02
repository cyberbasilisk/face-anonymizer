"""
Anonymize faces in an image, a video file, or a webcam stream.

Pipeline: 
    read a frame -> detect faces (detect.py) -> obscure them
    (anonymize.py) -> write/show result.
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import cv2

from anonymize import METHODS, anonymize
from detect import FaceDetector

# File extensions
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def parse_args() -> argparse.Namespace:
    
    # Define and parse command-line options
    p = argparse.ArgumentParser(description="Detect and anonymize faces.")
    
    p.add_argument("input", help="image/video path, or webcam index such as 0")
    p.add_argument("-o", "--output", help="output path (default: <name>_anon.<ext>)")
    p.add_argument("-m", "--method", choices=METHODS, default="pixelate")
    
    p.add_argument("--model", default="models/blaze_face_short_range.tflite")
    p.add_argument("--confidence", type=float, default=0.5,
                   help="min detection confidence; lower catches more faces (default 0.5)")
   
    p.add_argument("--padding", type=float, default=0.25,
                   help="extra margin around each face as a fraction of its size")
    
    p.add_argument("--hold", type=int, default=5,
                   help="video: keep a face's region covered for this many frames " \
                        "so a missed detection doesn't flash an unblurred face")
    
    p.add_argument("--show", action="store_true", help="preview window while processing video")
    
    return p.parse_args()


def default_output(path: Path, force_suffix: str | None = None) -> str:
    
    # Build the default output name (photo.jpg becomes photo_anon.jpg)
    suffix = force_suffix or path.suffix
    return str(path.with_name(f"{path.stem}_anon{suffix}"))


# Anonymize a single image file

def process_image(args: argparse.Namespace) -> None:
    
    path = Path(args.input)
    frame = cv2.imread(str(path))

    # imread may return None instead of raising error on failure
    if frame is None:  
        sys.exit(f"Could not read image: {path}")

    # MediaPipe Image mode
    with FaceDetector(args.model, args.confidence, args.padding) as detector:
        boxes = detector.detect(frame)

    # Anonymize
    anonymize(frame, boxes, args.method)

    # Save Output
    out = args.output or default_output(path)
    cv2.imwrite(out, frame)

    print(f"{len(boxes)} face(s) anonymized: {out}")

# Anonymize a video file or a live webcam stream, frame by frame

def process_stream(args: argparse.Namespace, source, is_webcam: bool) -> None:
    
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        sys.exit(f"Could not open video source: {args.input}")

    # Read the source's properties so the output matches it
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0  
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Saving outputs for video and webcam stream
    writer = None
    out = None
    
    if args.output or not is_webcam:
        out = args.output or default_output(Path(args.input), ".mp4")
        writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    # Prepare memory for detection
    recent: deque = deque(maxlen=max(1, args.hold))
    last_ts, idx = -1, 0

    try:
        with FaceDetector(args.model, args.confidence, args.padding, video=True) as detector:
            while True:
                ok, frame = cap.read()
                
                # end of video / webcam disconnected
                if not ok:  
                    break

                # Timestamps for MediaPipe VIDEO mode
                ts = int(time.time() * 1000) if is_webcam else int(idx * 1000 / fps)
                ts = max(ts, last_ts + 1)  
                last_ts = ts

            
                recent.append(detector.detect(frame, ts))
                boxes = [b for frame_boxes in recent for b in frame_boxes]
                anonymize(frame, boxes, args.method)

                if writer:
                    writer.write(frame)
                
                if args.show or is_webcam:
                    cv2.imshow("face-anonymizer (q to quit)", frame)
                    
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                idx += 1
                if idx % 100 == 0:
                    print(f"  processed {idx} frames", end="\r")
    finally:
        # Clean Up
        cap.release()
        
        if writer:
            writer.release()
        
        cv2.destroyAllWindows()

    print(f"Done: {idx} frames" + (f" -> {out}" if out else ""))


def main() -> None:
    args = parse_args()

    # An integer "0" passed means "use webcam 0"
    if args.input.isdigit():
        process_stream(args, int(args.input), is_webcam=True)
        return
    ext = Path(args.input).suffix.lower()

    # Image file
    if ext in IMAGE_EXT:
        process_image(args)
    
    # Video file
    elif ext in VIDEO_EXT:
        process_stream(args, args.input, is_webcam=False)
    
    else:
        sys.exit(f"Unsupported file type '{ext}'. Images: {sorted(IMAGE_EXT)}, "
                 f"Videos: {sorted(VIDEO_EXT)}")


if __name__ == "__main__":
    main()
