# Sleep Detection — Pixel Difference Strategy Research

Research codebase for factory CCTV sleep detection using **pixel-difference strategies** on tracked person crops.

---

## What's in this repo

| Path | Description |
|---|---|
| [`test.py`](test.py) | Main pipeline — YOLO / RT-DETR / RF-DETR inference + MAD-based sleep detection |
| [`pixel_diff_strategy_comparison.py`](pixel_diff_strategy_comparison.py) | **New** — evaluates all 14 pixel-difference strategies on the same inference pass |
| [`streak_analysis.py`](streak_analysis.py) | Post-hoc streak analysis on saved `diff_history` CSVs |
| [`sleep_new_method/sleep_new.py`](sleep_new_method/sleep_new.py) | Production deployment worker (RTSP multi-stream) |
| [`SLEEP_DETECTION_LOGIC_PROOF_AND_ANALYSIS.md`](SLEEP_DETECTION_LOGIC_PROOF_AND_ANALYSIS.md) | Proof & analysis: old rolling-mean logic vs new consecutive-streak logic |
| [`SLEEP_DETECTION_ANALYSIS.md`](SLEEP_DETECTION_ANALYSIS.md) | Detailed per-track and per-video detection analysis |
| [`RFDETR_VS_YOLO_ANALYSIS.md`](RFDETR_VS_YOLO_ANALYSIS.md) | RF-DETR vs YOLO11l tracking coverage comparison (3.0× more person-frames) |
| [`pixel_strategy_results/`](pixel_strategy_results/) | **Experiment results** — all 14 strategy charts, CSVs, and ranking for 3 test videos |
| [`test_video/experiment_1_fps/`](test_video/experiment_1_fps/) | Per-track diff histories and comparison charts from baseline MAD experiments |
| [`test_video/anotated_video_rfdetr/`](test_video/anotated_video_rfdetr/) | RF-DETR annotated video analysis reports |

---

## Key Findings

### Decision Logic: New (Streak) vs Old (Rolling Mean)
The **consecutive-streak logic** (longest unbroken run of frames below threshold) eliminates false positives from the old rolling-mean logic which absorbed motion spikes into averages.

- Old logic at W=100: **97–100% pass rate** (useless as a filter)
- New logic at Streak≥200, Thresh≤8: **56–60% pass rate** (correctly filters active workers)

### Detector Coverage: RF-DETR vs YOLO11l
| Video | YOLO | RF-DETR | Multiplier |
|---|---|---|---|
| Candy Forming M-C 3 | 2,147 frames | 4,101 frames | **1.91×** |
| Kneading Area - Video A | 2,639 frames | 8,648 frames | **3.28×** |
| Kneading Area - Video B | 2,085 frames | 7,847 frames | **3.76×** |
| **TOTAL** | **6,871** | **20,596** | **3.00×** |

RF-DETR's transformer global attention detects partially occluded factory workers that YOLO's CNN feature pyramid misses.

### Pixel-Difference Strategy Ranking (3 test videos, YOLO11m @ 1 fps)
| Rank | Strategy | Median Streak % | vs MAD |
|---|---|---|---|
| 🥇 1 | **pHash** | 35% | +133% |
| 🥈 2 | **MOG2** | 26% | +73% |
| 🥉 3 | SSIM | 19% | +27% |
| 4 | MS-SSIM | 19% | +27% |
| **6** | **MAD ← baseline** | **15%** | — |
| 10–14 | OF / Phase / Histogram | 10–12% | worse |

---

## How to Run

### Install dependencies
```bash
pip install ultralytics opencv-python numpy torch matplotlib scikit-image imagehash pytorch-msssim
# For RF-DETR backend:
pip install rfdetr supervision
```

### Run baseline (MAD) experiment
```bash
python test.py \
    --video test_video \
    --model yolo11m.pt \
    --output-dir test_video/experiment_1_fps \
    --fps 1.0
```

### Run all 14 strategy comparison
```bash
python pixel_diff_strategy_comparison.py \
    --video /path/to/your/videos \
    --model yolo11m.pt \
    --output-dir pixel_strategy_results \
    --fps 1.0
```

### Colab (Google Drive)
```python
from google.colab import drive
drive.mount('/content/drive')

!pip install scikit-image imagehash pytorch-msssim -q

VIDEO_FOLDER = "/content/drive/MyDrive/YOUR_FOLDER"
OUTPUT_DIR   = "/content/drive/MyDrive/pixel_strategy_results"

import subprocess, sys
subprocess.run([
    sys.executable, "pixel_diff_strategy_comparison.py",
    "--video", VIDEO_FOLDER,
    "--model", "yolo11m.pt",
    "--output-dir", OUTPUT_DIR,
    "--fps", "1.0",
])
```

---

## Repository Notes

- **Videos not included** — raw `.MOV` footage excluded via `.gitignore` (too large / proprietary)
- **Model weights not included** — `.pt` / `.engine` files excluded; download from [Ultralytics](https://docs.ultralytics.com/) or [Roboflow](https://github.com/roboflow/rf-detr)
- **All experiment results included** — charts, JSONs, CSVs from `pixel_strategy_results/` and `test_video/experiment_1_fps/` are committed

---

## Architecture

```
Frame from CCTV / video file
        │
        ▼
┌─────────────────────────────────┐
│   Detector + Tracker            │
│   YOLO11 / RT-DETR / RF-DETR   │  ← inference runs ONCE per frame
│   ByteTrack (for RF-DETR)       │
└─────────────────────────────────┘
        │  bounding boxes + track IDs
        ▼
┌─────────────────────────────────┐
│   Reference Locking             │
│   3-frame averaged bbox         │
│   → fixed 32×32 grayscale crop  │
└─────────────────────────────────┘
        │  reference_sig (32×32 uint8)
        ▼
┌─────────────────────────────────┐
│   14 Pixel-Diff Metrics         │
│   (all computed on same crop,   │
│    no extra inference)          │
│   MAD / MSE / pHash / SSIM /    │
│   MOG2 / OF / DCT / ...        │
└─────────────────────────────────┘
        │  diff_history per metric
        ▼
┌─────────────────────────────────┐
│   Streak Decision Logic         │
│   longest consecutive run       │
│   below threshold ≥ N seconds   │
│   → SLEEP ALERT                 │
└─────────────────────────────────┘
```
