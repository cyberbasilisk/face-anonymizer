## Face Detection And Anonymization

A command-line tool that detects faces in images, video files, or a live webcam stream and obscures them with a blur, pixelation, or solid box.

### How it works

1. `main.py` reads a frame from the image, video or webcam.
2. `detect.py` converts it to RGB, runs the MediaPipe face detector, then pads and clamps each box to the frame.
3. `anonymize.py` replaces each boxed region with its blurred, pixelated or blacked-out version.
4. `main.py` writes the frame to the output file and/or shows it in a preview window.

### Setup

```bash
# 1. Clone the repo
git clone https://github.com/YOURNAME/face-anonymizer.git
cd face-anonymizer

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate

# 3. Install dependencies
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

### Usage
Supported formats include `.jpg .jpeg .png .bmp .webp` for image files, and `.mp4 .avi .mov .mkv .webm` for video files.

Run from the project root:

```bash
# Image file
python3 src/main.py data/photo.jpg

# Video file
python3 src/main.py data/clip.mp4 -m blur --show

# Webcam (press q to quit)
python3 src/main.py 0
```

### Options

| Option | Default | Description |
|---|---|---|
| `input` | required | Image path, video path, or a webcam index such as `0` |
| `-o`, `--output` | `<name>_anon.<ext>` | Output file path (the folder must already exist) |
| `-m`, `--method` | `pixelate` | `blur`, `pixelate` or `box` |
| `--model` | `models/blaze_face_short_range.tflite` | Path to the detector model |
| `--confidence` | `0.5` | Minimum detection confidence. Lower catches more faces but adds false positives |
| `--padding` | `0.25` | Extra margin around each face, as a fraction of its size |
| `--hold` | `5` | Video only: frames to keep a region covered after detection |
| `--show` | off | Show a preview window while processing video |

