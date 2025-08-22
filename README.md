# Video Person Search System

*Search hours of footage by **appearance** instead of scrubbing frame‑by‑frame.*

**Video Demo**  
https://glenngriggs.github.io/CCTV-Vision/

---

## Table of Contents

1. [Features](#features)
2. [Performance](#performance)
3. [How It Works](#how-it-works)
4. [Getting Started](#getting-started)
5. [Model Download](#model-download)
6. [Installation Methods](#installation-methods)
7. [Usage Example](#usage-example)
8. [Real‑World Performance](#real-world-performance)
9. [Output](#output)
10. [Technical Notes](#technical-notes)
11. [Configuration](#configuration)
12. [Troubleshooting](#troubleshooting)
13. [Limitations](#limitations)

---

## Features

| Area                     | What it does                                                                                                                 |
| ------------------------ | ---------------------------------------------------------------------------------------------------------------------------- |
| **Detection & Tracking** | YOLO‑v8 spots people in each frame and a simple IoU + Hungarian tracker keeps the same ID as they move.                      |
| **Region Cropping**      | Uses MediaPipe landmarks to grab just the hair, torso, and legs; falls back to fixed box ratios when landmarks are missing.  |
| **Colour Detection**     | Converts the crop to Lab colour‑space and uses ΔE2000 to pick the closest named colour; includes a quick white‑balance pass. |
| **Attribute Model**      | Our trained (with optuna) EfficientNet‑B4 adds gender / clothing heads to back‑up the colour heuristics.                                               |
| **Query CLI**            | Tiny command‑line menu – set any attribute to **N/A** if you don't care.                                                     |
| **Video Export**         | Saves an MP4 with IDs, predictions, and a small HUD overlay.                                                                 |

---

## Performance

Trained on **PA‑100K** & **PETA** pedestrian‑attribute datasets.

| Metric                   | Result      |
| ------------------------ | ----------- |
| Detection Accuracy       | **95.25 %** |
| Attribute Classification | **93.20 %** |

> The attribute model beat our expected 70 % baseline by **+22 pp**.

### Optuna‑selected Hyper‑parameters

```text
Backbone      : EfficientNet‑B4
Learning‑Rate : 5.56 × 10⁻⁴
Batch Size    : 32
Dropout       : 0.698
Epochs        : 48
```

---

## How It Works

1. **Person Detection & Tracking**  
   YOLOv8 finds people each frame → IoU/Hungarian assigns persistent IDs.
2. **Smart Region Extraction**  
   Pose landmarks isolate hair, torso, and legs. Fallback: fixed box ratios.
3. **Color Analysis**  
   Convert crops to Lab → cluster → DeltaE2000 to named colours; auto white‑balance.
4. **Neural Network Fusion**  
   Our pre-trained model handles gender + backs up colour guesses. Fusion picks the most confident source.
5. **Query Engine**  
   CLI lets you select *gender*, *hair‑colour*, *shirt‑colour* – or **N/A**.

---

## Getting Started

### Prerequisites

- **Python 3.9+** (recommended)
- **4GB+ RAM** for CPU processing
- **Optional**: CUDA-compatible GPU for faster processing

---

## Model Download

** IMPORTANT**: Download the trained model before running the system:

**Download Link**: [ULTIMATE_model_score_0.9398_20250716_110939.pt](https://drive.google.com/file/d/1rJOY2qOOnPKr-dUQKbszJOZSxdJ4mJMv/view?usp=drive_link)

1. Download the model file from the link above
2. Place it in the `models/` directory:
   ```
   CCTV-Vision/
   ├── models/
   │   └── ULTIMATE_model_score_0.9398_20250716_110939.pt  ← Place here
   ├── videos/
   └── main.py
   ```

---

## Installation Methods

### Option 1: Conda Environment (Recommended)

**For CPU Processing:**
```bash
# Clone the repository
git clone <your-repo-url>
cd CCTV-Vision

# Create environment from file
conda env create -f environment-cpu.yml

# Activate environment
conda activate cctv_vision

# Run the system
python main.py
```

**For GPU Processing:**
```bash
# Create environment with GPU support
conda env create -f environment.yml  # Uses CUDA

# Activate environment
conda activate cctv_vision

# Run the system
python main.py
```

### Option 2: Pip Installation

**CPU Version:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-cpu.txt
```

**GPU Version:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### Option 3: Manual Installation

Install dependencies manually:

```bash
pip install torch torchvision ultralytics mediapipe opencv-python \
            scikit-image scikit-learn timm numpy scipy pillow
```

---

## Usage Example

```text
GENDER
1. male  2. female  3. N/A
Select: 1

HAIR COLOR
1. black  2. brown  3. blonde  4. gray  5. red  6. N/A
Select: 3

SHIRT COLOR
1. black  2. white  3. red  4. blue  5. green  6. N/A
Select: 4
```

Result → **Find all males with blonde hair wearing blue shirts**.

---

## Real‑World Performance

### CPU Processing
* **2-5 FPS** on modern CPUs (Intel i5/i7, AMD Ryzen 5/7)
* **Memory usage**: ~3GB RAM
* **Best for**: Batch processing, research, systems without GPU

### GPU Processing  
* **8-15 FPS** on single GPU (depends on resolution & pose‑detect toggle)
* **Memory usage**: ~6GB VRAM
* **Best for**: Real-time analysis, large video processing

### Adjustable factors:
- `use_pose` – Pose detection accuracy vs speed
- `attribute_update_interval` – Fewer re‑analyses ⇒ faster
- Video resolution – downscale for speed

The HUD displays live FPS, ETA, active tracks, and match confidence.

---

## Output

* Bounding boxes with persistent **Track IDs**
* Attribute predictions + confidence (**\[MODEL]**, **\[SMART]**, **\[INTEGRATION]**)
* Colour coding: green = match, yellow = partial, grey = non‑match

Example summary:

```
PROCESSING FINISHED
└ Total time         : 127.3 s
└ Unique people      : 47
└ Perfect matches    : 12 (25.5 %)
└ Processing speed   : 4.2 FPS
└ Integration improvements: 156
```

---

## Technical Notes

* **Tracker** – Hungarian assignment on IoU‑predicted motion
* **Colour Space** – Perceptually‑uniform CIE Lab; ΔE2000 distance  
* **Pose Landmarks** – 33‑point MediaPipe skeleton → precise crop masks
* **Model** – EfficientNet‑B4 backbone, multi‑head attribute classifier
* **Safe Loading** – Validates model architecture and handles missing components

---

## Configuration

```python
processor = EnhancedVideoProcessor(
    use_pose=True,                    # MediaPipe on/off
    verbose=True,                     # Console debug info
    attribute_update_interval=5,      # Frames between re‑analysis
    draw_hud=True                     # Overlay statistics on video
)
```

---

## Troubleshooting

### Common Issues

#### **Model Loading Errors**
```
Error: Model file not found
Error: invalid load key, 'v'
```
**Solutions:**
1. **Download the model**: [Get it here](https://drive.google.com/file/d/1rJOY2qOOnPKr-dUQKbszJOZSxdJ4mJMv/view?usp=drive_link)
2. **Check file location**: Place in `models/ULTIMATE_model_score_0.9398_20250716_110939.pt`
3. **Verify download**: File should be ~200MB, not 0 bytes
4. **Re-download if corrupted**: Delete and download again

####  **Video Processing Errors**
```
Error: Cannot open video
Error: moov atom not found
```
**Solutions:**
1. **Check video format**: Use MP4, AVI, or MOV files
2. **Verify file integrity**: Try playing video in media player first
3. **Re-encode if needed**: Use VLC or FFmpeg to convert
4. **Use test video**: Try with a different, known-working video file

####  **Environment Issues**
```
ModuleNotFoundError: No module named 'xyz'
```
**Solutions:**
1. **Use conda environment**: `conda env create -f environment.yml`
2. **Activate environment**: `conda activate cctv_vision`
3. **Reinstall dependencies**: `pip install -r requirements.txt`

####  **Performance Issues**
```
Very slow processing / Out of memory
```
**Solutions:**
1. **Use CPU version**: Install `environment-cpu.yml` for lower memory usage
2. **Reduce video resolution**: Downscale videos before processing
3. **Disable pose detection**: Set `use_pose=False` for speed
4. **Increase update interval**: Use `attribute_update_interval=10`

### MediaPipe Warnings
```
WARNING: Using NORM_RECT without IMAGE_DIMENSIONS...
```
**This is normal** - MediaPipe pose detection works fine, just prefers square images.

### Getting Help

1. **Check file integrity**: Ensure model and video files aren't corrupted
2. **Try debug mode**: Run with `debug_mode=True` for detailed output  
3. **Test with sample data**: Use known-working video files
4. **Check system requirements**: Verify Python 3.9+, sufficient RAM

---

## Limitations

* **Video Quality**: Needs clear, well‑lit footage – struggles with very dark or colour‑shifted scenes
* **Hair Detection**: Fails on heavy occlusion or exotic dye colors  
* **Processing Time**: Scales with both video length and crowd density
* **Model Dependency**: Requires the specific trained model file to function
* **Format Support**: Best with standard MP4/AVI formats; some codecs may cause issues

---

## License

MIT © Glenn Griggs, Shubhanshu Pokharel & Lucas Morris
