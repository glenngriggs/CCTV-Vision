#Video Person Search System

*Search hours of footage by **appearance** instead of scrubbing frame‑by‑frame.*

Video Demo
https://glenngriggs.github.io/CCTV-Vision/

---

## Table of Contents

1. [Features](#features)
2. [Performance](#performance)
3. [How It Works](#how-it-works)
4. [Getting Started](#getting-started)
5. [Usage Example](#usage-example)
6. [Real‑World Performance](#real-world-performance)
7. [Output](#output)
8. [Technical Notes](#technical-notes)
9. [Configuration](#configuration)
10. [Limitations](#limitations)

---

## Features

| Area                     | What it does                                                                                                                 |
| ------------------------ | ---------------------------------------------------------------------------------------------------------------------------- |
| **Detection & Tracking** | YOLO‑v8 spots people in each frame and a simple IoU + Hungarian tracker keeps the same ID as they move.                      |
| **Region Cropping**      | Uses MediaPipe landmarks to grab just the hair, torso, and legs; falls back to fixed box ratios when landmarks are missing.  |
| **Colour Detection**     | Converts the crop to Lab colour‑space and uses ΔE2000 to pick the closest named colour; includes a quick white‑balance pass. |
| **Attribute Model**      | Our trained (with optuna) EfficientNet‑B4 adds gender / clothing heads to back‑up the colour heuristics.                                               |
| **Query CLI**            | Tiny command‑line menu – set any attribute to **N/A** if you don’t care.                                                     |
| **Video Export**         | Saves an MP4 with IDs, predictions, and a small HUD overlay.                                                                 |


## Performance

Trained on **PA‑100K** & **PETA** pedestrian‑attribute datasets.

| Metric                   | Result      |
| ------------------------ | ----------- |
| Detection Accuracy       | **95.25 %** |
| Attribute Classification | **93.20 %** |

> The attribute model beat our expected 70 % baseline by **+22 pp**.

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

### Requirements

Install dependencies (or use the provided `requirements.txt`):

```bash
pip install torch torchvision ultralytics mediapipe opencv-python \
            scikit-image scikit-learn timm optuna
```

### Basic Setup

1. Place the trained model file `ULTIMATE_model_score_0.9398_20250716_110939.pt` next to `main.py`.
2. In Python:

   ```python
   from main import main_system
   main_system()
   ```
3. Choose **“Process video with system”** and follow the prompts.

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

* **8 – 15 FPS** on a single GPU (depends on resolution & pose‑detect toggle).
* Adjustable factors:

  * `use_pose` – Pose detection accuracy vs speed.
  * `attribute_update_interval` – Fewer re‑analyses ⇒ faster.
  * Video resolution – downscale for speed.

The HUD displays live FPS, ETA, active tracks, and match confidence.

---

## Output

* Bounding boxes with persistent **Track IDs**
* Attribute predictions + confidence (**\[MODEL]**, **\[SMART]**, **\[INTEGRATION]**)
* Colour coding: green = match, yellow = partial, grey = non‑match

Example summary:

```
PROCESSING FINISHED
└ Total time         : 127.3 s
└ Unique people      : 47
└ Perfect matches    : 12 (25.5 %)
```

---

## Technical Notes

* **Tracker** – Hungarian assignment on IoU‑predicted motion.
* **Colour Space** – Perceptually‑uniform CIE Lab; ΔE2000 distance.
* **Pose Landmarks** – 33‑point MediaPipe skeleton → precise crop masks.
* **Model** – EfficientNet‑B4 backbone, multi‑head attribute classifier.

---

## Configuration

```python
processor = EnhancedVideoProcessor(
    use_pose=True,            # MediaPipe on/off
    verbose=True,             # Console debug info
    attribute_update_interval=5,  # Frames between re‑analysis
    draw_hud=True             # Overlay statistics on video
)
```

---

## Limitations

* Needs clear, well‑lit footage – struggles with very dark or colour‑shifted scenes.
* Hair colour fails on heavy occlusion or exotic dye.
* Processing time scales with both video length and crowd density.

---

## License

MIT © Glenn Griggs, Shubhanshu Pokharel & Lucas Morris
