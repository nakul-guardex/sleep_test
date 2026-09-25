"""
pixel_diff_strategy_comparison.py
==================================
Evaluates ALL pixel-difference strategies (1-14, plus optional LPIPS #15)
on the same videos and inference pipeline as test.py.

WHAT THIS DOES
--------------
  1. Runs YOLO / RT-DETR / RF-DETR inference ONCE per frame (same as test.py).
  2. For every tracked person, at each frame it:
       - crops the FIXED reference-bbox window (same 32x32 grayscale patch)
       - computes 14 distinct diff metrics between reference crop and current crop
  3. Applies the NEW streak logic (longest consecutive run below threshold)
     to EVERY metric independently, sweeping thresholds.
  4. Produces per-track history JSONs, a master CSV, per-metric streak charts,
     and a final RANKING chart showing which metric best separates
     sleeping vs active on YOUR specific videos.

STRATEGIES TESTED
-----------------
  1  MAD        - Mean Absolute Difference (your current baseline)
  2  MSE        - Mean Squared Error
  3  SAD        - Sum of Absolute Differences (unnormalized MAD)
  4  SSIM       - Structural Similarity Index (requires scikit-image)
  5  MS-SSIM    - Multi-Scale SSIM (requires pytorch-msssim or skimage)
  6  GRAD       - Sobel gradient-map difference
  7  CANNY      - Canny binarised edge difference
  8  OF_MAG     - Dense Optical Flow (Farneback) mean magnitude
  9  OF_SPARSE  - Sparse Optical Flow (Lucas-Kanade) mean point displacement
  10 MOG2       - Background Subtraction foreground pixel ratio (per-track)
  11 DCT        - DCT coefficient L1 difference
  12 PHASE      - Phase-correlation peak offset magnitude
  13 HIST       - Histogram Bhattacharyya distance
  14 PHASH      - Perceptual hash Hamming distance (requires imagehash)
  15 LPIPS      - Learned Perceptual Image Patch Similarity (optional, GPU)

HOW TO RUN (Colab) -- copy-paste these cells in order
------------------------------------------------------

  #### CELL 1 -- Mount Google Drive ####
  from google.colab import drive
  drive.mount('/content/drive')

  #### CELL 2 -- Install extras ####
  !pip install scikit-image imagehash pytorch-msssim -q
  # Optional: RF-DETR backend
  # !pip install rfdetr supervision -q

  #### CELL 3 -- Upload this script ####
  # Either upload via the Files panel or copy from Drive:
  # !cp "/content/drive/MyDrive/path/to/pixel_diff_strategy_comparison.py" .

  #### CELL 4 -- SET YOUR FOLDER PATH HERE, then run ####
  VIDEO_FOLDER = "/content/drive/MyDrive/YOUR_FOLDER_HERE"  # <- edit this
  OUTPUT_DIR   = "/content/drive/MyDrive/pixel_strategy_results"
  MODEL        = "yolo11m.pt"   # or rfdetr-medium / rtdetr-l.pt
  DETECTOR     = "auto"         # auto / yolo / rtdetr / rfdetr

  import subprocess, sys
  result = subprocess.run([
      sys.executable, "pixel_diff_strategy_comparison.py",
      "--video",      VIDEO_FOLDER,
      "--model",      MODEL,
      "--detector",   DETECTOR,
      "--output-dir", OUTPUT_DIR,
      "--fps",        "1.0",
  ], check=False)

OUTPUTS (per video folder)
--------------------------
  track_{id}_all_metrics.json        - raw history for all 14 metrics
  track_{id}_metric_grid.png         - 14-panel signal chart per track
  all_tracks_metrics_summary.csv     - per-track per-metric streak stats
  strategy_ranking_chart.png         - final ranking by stillness separability
  strategy_ranking.json              - machine-readable ranking
"""

import argparse
import csv
import glob
import json
import os
import time
import warnings
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

warnings.filterwarnings("ignore")

# ---- Optional imports (graceful fallback) ----------------------------------
try:
    from skimage.metrics import structural_similarity as ski_ssim
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

try:
    import imagehash
    from PIL import Image as PILImage
    HAS_IMAGEHASH = True
except ImportError:
    HAS_IMAGEHASH = False

try:
    import lpips
    HAS_LPIPS = True
except ImportError:
    HAS_LPIPS = False

try:
    from pytorch_msssim import ms_ssim as _ms_ssim
    HAS_MSSSIM = True
except ImportError:
    HAS_MSSSIM = False


# ---- Pipeline constants (same as test.py) ----------------------------------
LOCK_AVERAGE_FRAMES = 3
RELOCK_THRESHOLD_MAD = 10.0

# ---- Strategy sweep parameters ---------------------------------------------
STREAK_REQUIRED = [10, 20, 30, 60, 100, 150, 200]   # seconds (at 1 fps)

METRIC_KEYS = [
    "MAD", "MSE", "SAD", "SSIM", "MS_SSIM",
    "GRAD", "CANNY", "OF_MAG", "OF_SPARSE",
    "MOG2", "DCT", "PHASE", "HIST", "PHASH",
]
METRIC_LABELS = {
    "MAD":       "#1  MAD (baseline)",
    "MSE":       "#2  MSE",
    "SAD":       "#3  SAD",
    "SSIM":      "#4  1-SSIM",
    "MS_SSIM":   "#5  1-MS-SSIM",
    "GRAD":      "#6  Gradient diff",
    "CANNY":     "#7  Canny edge diff",
    "OF_MAG":    "#8  Optical flow mag",
    "OF_SPARSE": "#9  Sparse OF disp",
    "MOG2":      "#10 MOG2 FG ratio",
    "DCT":       "#11 DCT diff",
    "PHASE":     "#12 Phase shift",
    "HIST":      "#13 Hist Bhattacharyya",
    "PHASH":     "#14 pHash Hamming",
}


# ---- Tracking / locking helpers (identical to test.py) --------------------
class PersonTrack:
    def __init__(self, track_id, lock_avg_frames=LOCK_AVERAGE_FRAMES):
        self.track_id = track_id
        self.reference_sig = None
        self.reference_bbox = None
        self.total_frames = 0
        self.box_buffer = deque(maxlen=lock_avg_frames)
        self.lock_avg_frames = lock_avg_frames
        self.metric_history = {k: [] for k in METRIC_KEYS}
        self.mog2 = cv2.createBackgroundSubtractorMOG2(
            history=200, varThreshold=16, detectShadows=False
        )
        self.ref_corners = None

    def reset_reference(self, sig, bbox):
        self.reference_sig = sig
        self.reference_bbox = bbox
        self.mog2 = cv2.createBackgroundSubtractorMOG2(
            history=200, varThreshold=16, detectShadows=False
        )
        corners = cv2.goodFeaturesToTrack(
            sig, maxCorners=20, qualityLevel=0.3, minDistance=3
        )
        self.ref_corners = corners if corners is not None else np.zeros((1, 1, 2), np.float32)


def get_sig(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)


def averaged_box(box_buffer, w, h):
    boxes = np.array(box_buffer)
    avg = boxes.mean(axis=0)
    x1, y1, x2, y2 = avg
    return (
        int(np.clip(x1, 0, w)), int(np.clip(y1, 0, h)),
        int(np.clip(x2, 0, w)), int(np.clip(y2, 0, h)),
    )


# ---- All 14 metric computations --------------------------------------------
def compute_all_metrics(ref_sig, cur_sig, track):
    m = {}
    ref_f = ref_sig.astype(np.float32)
    cur_f = cur_sig.astype(np.float32)
    diff_abs = np.abs(ref_f - cur_f)

    # 1. MAD
    m["MAD"] = float(diff_abs.mean())

    # 2. MSE
    m["MSE"] = float(np.mean((ref_f - cur_f) ** 2))

    # 3. SAD
    m["SAD"] = float(diff_abs.sum())

    # 4. SSIM (inverted: 0 = identical, higher = more different)
    if HAS_SKIMAGE:
        m["SSIM"] = float(1.0 - ski_ssim(ref_sig, cur_sig, data_range=255))
    else:
        m["SSIM"] = m["MAD"] / 255.0

    # 5. MS-SSIM (inverted)
    if HAS_MSSSIM:
        try:
            t_ref = torch.from_numpy(ref_f).unsqueeze(0).unsqueeze(0) / 255.0
            t_cur = torch.from_numpy(cur_f).unsqueeze(0).unsqueeze(0) / 255.0
            ms_val = _ms_ssim(t_ref, t_cur, data_range=1.0, size_average=True).item()
            m["MS_SSIM"] = float(1.0 - ms_val)
        except Exception:
            m["MS_SSIM"] = m["SSIM"]
    elif HAS_SKIMAGE:
        ref_half = cv2.resize(ref_sig, (16, 16))
        cur_half = cv2.resize(cur_sig, (16, 16))
        s1 = ski_ssim(ref_sig, cur_sig, data_range=255)
        s2 = ski_ssim(ref_half, cur_half, data_range=255)
        m["MS_SSIM"] = float(1.0 - (0.5 * s1 + 0.5 * s2))
    else:
        m["MS_SSIM"] = m["SSIM"]

    # 6. Gradient (Sobel) diff
    grad_ref = cv2.Sobel(ref_sig, cv2.CV_32F, 1, 0) + cv2.Sobel(ref_sig, cv2.CV_32F, 0, 1)
    grad_cur = cv2.Sobel(cur_sig, cv2.CV_32F, 1, 0) + cv2.Sobel(cur_sig, cv2.CV_32F, 0, 1)
    m["GRAD"] = float(np.mean(np.abs(grad_ref - grad_cur)))

    # 7. Canny edge diff
    edges_ref = cv2.Canny(ref_sig, 50, 150).astype(np.float32) / 255.0
    edges_cur = cv2.Canny(cur_sig, 50, 150).astype(np.float32) / 255.0
    m["CANNY"] = float(np.mean(np.abs(edges_ref - edges_cur)))

    # 8. Dense Optical Flow (Farneback) - mean displacement magnitude
    flow = cv2.calcOpticalFlowFarneback(
        ref_sig, cur_sig, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    m["OF_MAG"] = float(mag.mean())

    # 9. Sparse Optical Flow (Lucas-Kanade)
    if track.ref_corners is not None and len(track.ref_corners) > 0:
        p1, st, _ = cv2.calcOpticalFlowPyrLK(
            ref_sig, cur_sig, track.ref_corners, None,
            winSize=(7, 7), maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )
        if p1 is not None and st is not None:
            good = st.ravel() == 1
            if good.sum() > 0:
                disp = np.linalg.norm(
                    p1[good].reshape(-1, 2) - track.ref_corners[good].reshape(-1, 2), axis=1
                )
                m["OF_SPARSE"] = float(disp.mean())
            else:
                m["OF_SPARSE"] = m["OF_MAG"]
        else:
            m["OF_SPARSE"] = m["OF_MAG"]
    else:
        m["OF_SPARSE"] = m["OF_MAG"]

    # 10. MOG2 - foreground pixel ratio (per-track subtractor)
    fg_mask = track.mog2.apply(cur_sig)
    m["MOG2"] = float((fg_mask > 127).sum()) / (32 * 32)

    # 11. DCT coefficient L1 diff
    dct_ref = cv2.dct(ref_f)
    dct_cur = cv2.dct(cur_f)
    m["DCT"] = float(np.mean(np.abs(dct_ref - dct_cur)))

    # 12. Phase correlation peak offset magnitude
    try:
        (dx, dy), _ = cv2.phaseCorrelate(ref_f, cur_f)
        m["PHASE"] = float(np.sqrt(dx ** 2 + dy ** 2))
    except Exception:
        m["PHASE"] = 0.0

    # 13. Histogram Bhattacharyya distance
    hist_ref = cv2.calcHist([ref_sig], [0], None, [64], [0, 256])
    hist_cur = cv2.calcHist([cur_sig], [0], None, [64], [0, 256])
    cv2.normalize(hist_ref, hist_ref)
    cv2.normalize(hist_cur, hist_cur)
    m["HIST"] = float(cv2.compareHist(hist_ref, hist_cur, cv2.HISTCMP_BHATTACHARYYA))

    # 14. Perceptual hash (pHash) Hamming distance
    if HAS_IMAGEHASH:
        try:
            h_ref = imagehash.phash(PILImage.fromarray(ref_sig))
            h_cur = imagehash.phash(PILImage.fromarray(cur_sig))
            m["PHASH"] = float(h_ref - h_cur)
        except Exception:
            m["PHASH"] = m["MAD"] / 10.0
    else:
        m["PHASH"] = m["MAD"] / 10.0

    return m


# ---- Detector / Tracker backends (same as test.py) ------------------------
class BaseTracker:
    def track(self, frame): raise NotImplementedError


class UltralyticsTracker(BaseTracker):
    def __init__(self, model_path, model_type="yolo", conf=0.25, imgsz=1280, device="mps"):
        self.conf = conf; self.imgsz = imgsz; self.device = device
        if model_type == "rtdetr":
            from ultralytics import RTDETR
            self.model = RTDETR(model_path)
        else:
            from ultralytics import YOLO
            self.model = YOLO(model_path)

    def track(self, frame):
        results = self.model.track(
            frame, persist=True, conf=self.conf, imgsz=self.imgsz,
            classes=[0], device=self.device, verbose=False,
        )
        if results and results[0].boxes is not None and results[0].boxes.id is not None:
            return results[0].boxes.xyxy.cpu().numpy(), results[0].boxes.id.cpu().numpy()
        return np.array([]), np.array([])


class RFDETRTracker(BaseTracker):
    def __init__(self, model_name_or_path="rfdetr-medium", conf=0.25, device="mps"):
        self.conf = conf
        try:
            import supervision as sv
            import rfdetr
        except ImportError:
            raise ImportError("RF-DETR needs: pip install rfdetr supervision")
        self.sv = sv
        try:
            self.tracker = sv.ByteTrack(
                track_activation_threshold=max(0.15, conf * 0.8),
                lost_track_buffer=40, minimum_consecutive_frames=1,
            )
        except TypeError:
            self.tracker = sv.ByteTrack(
                track_activation_threshold=max(0.15, conf * 0.8),
                lost_track_buffer=40,
            )
        name = model_name_or_path.lower()
        if "large" in name:   self.model = rfdetr.RFDETRLarge()
        elif "small" in name: self.model = rfdetr.RFDETRSmall()
        elif "nano" in name:  self.model = rfdetr.RFDETRNano()
        elif "base" in name:  self.model = rfdetr.RFDETRBase()
        else:                 self.model = rfdetr.RFDETRMedium()
        self.person_ids = [0]
        try:
            from rfdetr.assets.coco_classes import COCO_CLASSES
            self.person_ids = [
                i for i, c in enumerate(COCO_CLASSES) if c.lower() in ("person", "pedestrian")
            ]
        except Exception:
            self.person_ids = [0, 1]

    def track(self, frame):
        from PIL import Image
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        dets = self.model.predict(Image.fromarray(rgb), threshold=self.conf)
        if len(dets) > 0:
            mask = None
            if "class_name" in dets.data:
                cnames = [str(c).lower() for c in dets.data["class_name"]]
                mask = np.array([c in ("person", "pedestrian") for c in cnames])
            elif dets.class_id is not None:
                mask = np.isin(dets.class_id, self.person_ids)
            if mask is not None and np.any(mask):
                dets = dets[mask]
        if len(dets) > 0:
            tracked = self.tracker.update_with_detections(dets)
            if tracked.tracker_id is not None and len(tracked.tracker_id) > 0:
                return tracked.xyxy, tracked.tracker_id
            return dets.xyxy, np.arange(1, len(dets) + 1)
        return np.array([]), np.array([])


def get_tracker(model_path, detector_type="auto", conf=0.25, imgsz=1280, device="mps"):
    ml, dl = model_path.lower(), detector_type.lower()
    if dl == "rfdetr" or "rfdetr" in ml:
        return RFDETRTracker(model_name_or_path=model_path, conf=conf, device=device)
    elif dl == "rtdetr" or "rtdetr" in ml:
        return UltralyticsTracker(model_path, model_type="rtdetr", conf=conf, imgsz=imgsz, device=device)
    else:
        return UltralyticsTracker(model_path, model_type="yolo", conf=conf, imgsz=imgsz, device=device)


# ---- Streak logic ----------------------------------------------------------
def longest_streak_under(history, threshold):
    best = cur = 0
    start = best_start = best_end = 0
    for i, v in enumerate(history):
        if v <= threshold:
            if cur == 0:
                start = i
            cur += 1
            if cur > best:
                best = cur
                best_start, best_end = start, i
        else:
            cur = 0
    return best, (best_start, best_end)


# ---- Main video processing loop --------------------------------------------
def process_video(
    video_path, model_path, detector_type="auto",
    conf=0.25, imgsz=1280, min_bbox_size=32,
    device="mps", target_fps=1.0, output_dir=".",
):
    tracker_engine = get_tracker(model_path, detector_type, conf, imgsz, device)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w_frame = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_frame = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if target_fps and 0 < target_fps < native_fps:
        frame_step = max(1, int(round(native_fps / target_fps)))
        effective_fps = native_fps / frame_step
    else:
        frame_step = 1
        effective_fps = native_fps

    lock_buf = 1 if frame_step > 5 else LOCK_AVERAGE_FRAMES
    expected = (total_frames + frame_step - 1) // frame_step

    print(f"\n{'='*80}")
    print(f"Video : {os.path.basename(video_path)}")
    print(f"Frames: {total_frames} total -> {expected} processed @ {effective_fps:.1f} fps")
    print(f"Device: {device} | Model: {os.path.basename(model_path)}")
    print(f"{'='*80}")

    tracks = {}
    raw_idx = proc_count = 0
    t0 = t_log = time.time()

    while True:
        if not cap.grab():
            break
        if raw_idx % frame_step != 0:
            raw_idx += 1
            continue
        ret, frame = cap.retrieve()
        raw_idx += 1
        if not ret or frame is None:
            continue
        proc_count += 1

        boxes, track_ids = tracker_engine.track(frame)

        if len(boxes) > 0 and len(track_ids) > 0:
            for box, tid in zip(boxes, track_ids):
                tid = int(tid)
                if tid not in tracks:
                    tracks[tid] = PersonTrack(track_id=tid, lock_avg_frames=lock_buf)
                trk = tracks[tid]

                x1 = max(0, int(box[0])); y1 = max(0, int(box[1]))
                x2 = min(w_frame, int(box[2])); y2 = min(h_frame, int(box[3]))
                if (x2 - x1) < min_bbox_size or (y2 - y1) < min_bbox_size // 2:
                    continue

                trk.box_buffer.append((x1, y1, x2, y2))

                # Reference locking
                if trk.reference_bbox is None:
                    if len(trk.box_buffer) < trk.lock_avg_frames:
                        continue
                    lx1, ly1, lx2, ly2 = averaged_box(trk.box_buffer, w_frame, h_frame)
                    crop = frame[ly1:ly2, lx1:lx2]
                    if crop.size == 0:
                        continue
                    sig = get_sig(crop)
                    trk.reset_reference(sig, (lx1, ly1, lx2, ly2))
                    continue

                # Diff at fixed reference window
                rx1, ry1, rx2, ry2 = trk.reference_bbox
                crop = frame[ry1:ry2, rx1:rx2]
                if crop.size == 0:
                    trk.reference_bbox = None
                    continue

                cur_sig = get_sig(crop)
                metrics = compute_all_metrics(trk.reference_sig, cur_sig, trk)
                for k, v in metrics.items():
                    if k in trk.metric_history:
                        trk.metric_history[k].append(v)
                trk.total_frames += 1

                # Relock on MAD threshold (same as test.py)
                if metrics["MAD"] > RELOCK_THRESHOLD_MAD:
                    lx1, ly1, lx2, ly2 = averaged_box(trk.box_buffer, w_frame, h_frame)
                    new_crop = frame[ly1:ly2, lx1:lx2]
                    if new_crop.size > 0:
                        new_sig = get_sig(new_crop)
                        trk.reset_reference(new_sig, (lx1, ly1, lx2, ly2))

        now = time.time()
        if (now - t_log >= 15.0) or proc_count == expected:
            elapsed = now - t0
            fps_proc = proc_count / elapsed if elapsed > 0 else 0
            eta = (expected - proc_count) / fps_proc if fps_proc > 0 else 0
            print(
                f"[{time.strftime('%H:%M:%S')}] {proc_count}/{expected} "
                f"({100*proc_count/expected:.1f}%) | "
                f"Tracks: {len(tracks)} | "
                f"Speed: {fps_proc:.1f} fps | ETA: {int(eta//60)}m{int(eta%60)}s",
                flush=True,
            )
            t_log = now

    cap.release()
    return tracks, effective_fps


# ---- Analysis & plotting ---------------------------------------------------
def analyze_and_plot(tracks, effective_fps, video_stem, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    time_unit = "sec" if effective_fps <= 2.0 else "frame"
    os.makedirs(output_dir, exist_ok=True)

    summary_rows = []
    all_metric_streaks = {k: [] for k in METRIC_KEYS}

    for tid, trk in tracks.items():
        n = trk.total_frames
        if n < 5:
            continue

        # Save raw histories
        json_path = os.path.join(output_dir, f"track_{tid}_all_metrics.json")
        serializable = {k: [float(x) for x in v] for k, v in trk.metric_history.items()}
        serializable["total_frames"] = n
        serializable["effective_fps"] = effective_fps
        with open(json_path, "w") as f:
            json.dump(serializable, f)

        # Per-track 14-panel chart
        nrows, ncols = 4, 4
        fig = plt.figure(figsize=(22, 16))
        fig.suptitle(
            f"{video_stem}  |  Track {tid}  |  {n} {time_unit}s @ {effective_fps:.1f} fps",
            fontsize=13, fontweight="bold", y=0.99
        )
        gs = gridspec.GridSpec(nrows, ncols, figure=fig, hspace=0.60, wspace=0.38)

        row_info = {}
        for i, key in enumerate(METRIC_KEYS):
            r, c = divmod(i, ncols)
            ax = fig.add_subplot(gs[r, c])
            hist = trk.metric_history[key]
            if not hist:
                ax.set_title(f"{METRIC_LABELS[key]}\n(no data)")
                ax.axis("off")
                continue

            x = np.arange(len(hist))
            arr = np.array(hist)
            auto_thresh = float(np.percentile(arr, 25))
            streak, (ss, se) = longest_streak_under(hist, auto_thresh)

            ax.plot(x, arr, linewidth=0.9, color="#e05252", alpha=0.85)
            ax.fill_between(x, 0, arr, color="#e05252", alpha=0.12)
            if streak > 0:
                ax.axvspan(ss, se, color="#52c48a", alpha=0.28,
                           label=f"streak={streak}")
            ax.axhline(auto_thresh, color="#888", linestyle="--", linewidth=0.7,
                       label=f"p25={auto_thresh:.2f}")
            ax.set_title(METRIC_LABELS[key], fontsize=8, pad=3)
            ax.set_xlabel(time_unit, fontsize=6)
            ax.tick_params(labelsize=6)
            ax.legend(fontsize=5, loc="upper right", framealpha=0.5)

            row_info[key] = {
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "max": float(arr.max()),
                "min": float(arr.min()),
                "p25": float(np.percentile(arr, 25)),
                "streak_at_p25": streak,
            }
            all_metric_streaks[key].append((streak, n))

        chart_path = os.path.join(output_dir, f"track_{tid}_metric_grid.png")
        plt.savefig(chart_path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"    Saved chart: track_{tid}_metric_grid.png")

        # Streak table for all required lengths (using p25 as threshold)
        streak_cols = {}
        for key in METRIC_KEYS:
            hist = trk.metric_history[key]
            if not hist:
                for req in STREAK_REQUIRED:
                    streak_cols[f"{key}_streak{req}"] = False
                continue
            arr = np.array(hist)
            p25 = float(np.percentile(arr, 25))
            best_streak, _ = longest_streak_under(hist, p25)
            for req in STREAK_REQUIRED:
                streak_cols[f"{key}_streak{req}"] = best_streak >= req

        row = {"video": video_stem, "track_id": tid, "total_frames": n}
        for key in METRIC_KEYS:
            info = row_info.get(key, {})
            row[f"{key}_mean"]         = round(info.get("mean", 0), 4)
            row[f"{key}_std"]          = round(info.get("std", 0), 4)
            row[f"{key}_streak_at_p25"]= info.get("streak_at_p25", 0)
        row.update(streak_cols)
        summary_rows.append(row)

    return summary_rows, all_metric_streaks


def plot_strategy_ranking(all_metric_streaks, output_dir, title_label):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ranking = []
    for key in METRIC_KEYS:
        data = all_metric_streaks.get(key, [])
        if not data:
            ranking.append((key, 0.0, 0.0))
            continue
        ratios = [s / max(n, 1) for s, n in data]
        ranking.append((key, float(np.median(ratios)), float(np.mean(ratios))))

    ranking.sort(key=lambda x: -x[1])

    keys_sorted = [r[0] for r in ranking]
    medians     = [r[1] * 100 for r in ranking]
    means       = [r[2] * 100 for r in ranking]
    colors      = ["#3a86ff" if k == "MAD" else "#52c48a" for k in keys_sorted]

    fig, ax = plt.subplots(figsize=(14, 7))
    y = np.arange(len(keys_sorted))
    bars = ax.barh(y, medians, height=0.52, color=colors, alpha=0.85, label="Median")
    ax.scatter(means, y, color="#e05252", zorder=5, s=40, label="Mean", marker="D")
    ax.set_yticks(y)
    ax.set_yticklabels([METRIC_LABELS[k] for k in keys_sorted], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel(
        "Stillness streak as % of track length  (higher = better sustained-stillness detection)",
        fontsize=10
    )
    ax.set_title(
        f"Strategy Ranking -- {title_label}\n"
        f"(streak threshold = 25th percentile of each metric's own distribution)",
        fontsize=12, fontweight="bold"
    )
    ax.legend(fontsize=9)
    ax.bar_label(bars, labels=[f"{v:.1f}%" for v in medians], padding=3, fontsize=8)

    for i, k in enumerate(keys_sorted):
        if k == "MAD":
            ax.annotate("  <- baseline", (medians[i], i),
                        fontsize=8, color="#3a86ff", va="center")

    plt.tight_layout()
    rank_path = os.path.join(output_dir, "strategy_ranking_chart.png")
    plt.savefig(rank_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Ranking chart saved: {rank_path}")

    rank_json = [
        {
            "rank": i + 1,
            "metric": k,
            "label": METRIC_LABELS[k],
            "median_streak_pct": round(med, 2),
            "mean_streak_pct":   round(mn, 2),
        }
        for i, (k, med, mn) in enumerate(ranking)
    ]
    with open(os.path.join(output_dir, "strategy_ranking.json"), "w") as f:
        json.dump(rank_json, f, indent=2)

    print("\n  STRATEGY RANKING (best -> worst sustained-stillness sensitivity):")
    print(f"  {'Rank':<5} {'Metric':<15} {'Median streak %':>17}  {'Mean streak %':>14}  Note")
    print("  " + "-"*70)
    for i, (k, med, mn) in enumerate(ranking):
        flag = "  <- BASELINE" if k == "MAD" else ""
        print(f"  {i+1:<5} {k:<15} {med*100:>14.1f}%  {mn*100:>11.1f}%{flag}")

    return ranking


# ---- Device helper ---------------------------------------------------------
def get_device():
    if torch.cuda.is_available(): return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return "mps"
    return "cpu"


# ---- CLI -------------------------------------------------------------------
def _is_colab():
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def _mount_drive_if_needed():
    """Auto-mount Google Drive when running in Colab (no-op otherwise)."""
    if not _is_colab():
        return
    try:
        from google.colab import drive as _drive
        if not os.path.ismount("/content/drive"):
            print("Mounting Google Drive...")
            _drive.mount("/content/drive")
            print("Drive mounted at /content/drive")
        else:
            print("Google Drive already mounted.")
    except Exception as e:
        print(f"Warning: could not mount Drive automatically: {e}")
        print("Mount manually with:  from google.colab import drive; drive.mount('/content/drive')")


def main():
    # ── Google Drive: mount automatically when running in Colab ──────────
    _mount_drive_if_needed()

    parser = argparse.ArgumentParser(
        description="Pixel-difference strategy comparison for sleep detection.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--video",
        default=None,
        help=(
            "Path to a video file OR a directory containing videos.\n"
            "Google Drive example: /content/drive/MyDrive/YOUR_FOLDER\n"
            "Local example: test_video"
        ),
    )
    parser.add_argument("--model",      default="yolo11m.pt",  help="Model path or name")
    parser.add_argument("--detector",   choices=["auto","yolo","rtdetr","rfdetr"], default="auto")
    parser.add_argument("--device",     default=get_device())
    parser.add_argument("--output-dir", default="pixel_strategy_results")
    parser.add_argument("--fps",        type=float, default=1.0)
    parser.add_argument("--conf",       type=float, default=0.25)
    parser.add_argument("--imgsz",      type=int,   default=1280)
    parser.add_argument("--min-bbox-size", type=int, default=32)
    args = parser.parse_args()

    # ── If no --video given, prompt interactively in Colab ────────────────
    if args.video is None:
        if _is_colab():
            args.video = input(
                "\nPaste your Google Drive video folder path\n"
                "(e.g. /content/drive/MyDrive/Nakul_Guardex/sleep_test/test_video):\n> "
            ).strip()
        else:
            args.video = "test_video"  # local fallback

    print(f"\nVideo source : {args.video}")
    print(f"Output dir   : {args.output_dir}")
    print(f"Model        : {args.model} [{args.detector}]")
    print(f"Device       : {args.device} | FPS: {args.fps}")

    print("\n+- Library availability -----------------------------------------------+")
    print(f"|  scikit-image (SSIM/MS-SSIM) : {'YES' if HAS_SKIMAGE   else 'NO  ->  pip install scikit-image':<45}|")
    print(f"|  imagehash    (pHash #14)    : {'YES' if HAS_IMAGEHASH else 'NO  ->  pip install imagehash Pillow':<45}|")
    print(f"|  pytorch-msssim (MS-SSIM #5) : {'YES' if HAS_MSSSIM   else 'NO  ->  pip install pytorch-msssim':<45}|")
    print(f"|  lpips        (LPIPS #15)    : {'YES' if HAS_LPIPS     else 'NO  ->  pip install lpips (optional)':<45}|")
    print("+----------------------------------------------------------------------+\n")

    # Gather videos
    video_paths = []
    if os.path.isdir(args.video):
        for ext in ("*.MOV","*.mov","*.mp4","*.m4v","*.avi"):
            video_paths.extend(glob.glob(os.path.join(args.video, ext)))
        video_paths.sort()
    elif os.path.isfile(args.video):
        video_paths = [args.video]
    else:
        video_paths = sorted(glob.glob(args.video))

    if not video_paths:
        raise FileNotFoundError(f"No videos found at: {args.video}")

    print(f"Found {len(video_paths)} video(s):")
    for vp in video_paths:
        print(f"  * {os.path.basename(vp)}")

    os.makedirs(args.output_dir, exist_ok=True)
    all_summary_rows = []
    combined_metric_streaks = {k: [] for k in METRIC_KEYS}

    for idx, vp in enumerate(video_paths, 1):
        v_stem = Path(vp).stem
        vdir = os.path.join(args.output_dir, v_stem)
        os.makedirs(vdir, exist_ok=True)

        print(f"\n{'#'*80}")
        print(f"Video {idx}/{len(video_paths)}: {os.path.basename(vp)}")
        print(f"{'#'*80}")

        tracks, eff_fps = process_video(
            video_path=vp,
            model_path=args.model,
            detector_type=args.detector,
            conf=args.conf,
            imgsz=args.imgsz,
            min_bbox_size=args.min_bbox_size,
            device=args.device,
            target_fps=args.fps,
            output_dir=vdir,
        )

        if not tracks:
            print(f"  No tracks found in {os.path.basename(vp)}, skipping.")
            continue

        qualifying = {tid: t for tid, t in tracks.items() if t.total_frames >= 5}
        print(f"\n  Analysing {len(qualifying)} qualifying track(s)...")
        rows, vid_metric_streaks = analyze_and_plot(qualifying, eff_fps, v_stem, vdir)
        all_summary_rows.extend(rows)

        for k in METRIC_KEYS:
            combined_metric_streaks[k].extend(vid_metric_streaks.get(k, []))

        print(f"\n  [Per-video strategy ranking for {v_stem}]")
        plot_strategy_ranking(vid_metric_streaks, vdir, v_stem)

    # Global CSV
    if all_summary_rows:
        csv_path = os.path.join(args.output_dir, "all_tracks_metrics_summary.csv")
        fieldnames = list(all_summary_rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(all_summary_rows)
        print(f"\n  Global CSV saved: {csv_path}")

        print(f"\n{'='*80}")
        print("COMBINED RANKING ACROSS ALL VIDEOS")
        print(f"{'='*80}")
        plot_strategy_ranking(combined_metric_streaks, args.output_dir, "ALL_VIDEOS_COMBINED")

    print(f"\n{'='*80}")
    print(f"DONE.  All outputs in: {args.output_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
