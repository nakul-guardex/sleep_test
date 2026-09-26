"""
live_sleep_benchmark.py
=======================
Live-stream sleep detection benchmark for Google Colab (T4 GPU).

Runs on 5 Mohit cameras simultaneously:
  cam_03 (Ch 301), cam_24 (Ch 2401), cam_29 (Ch 2901),
  cam_38 (Ch 3801), cam_59 (Ch 5901)

Features
--------
- SMOKE TEST: 5-minute pre-flight check (all 5 cams, 1 fps, 1 frame each)
- OLD LOGIC: rolling-mean window decision (intact, unmodified)
- NEW LOGIC: consecutive streak decision (intact, unmodified)
- METRIC BENCHMARK: pHash, MOG2, SSIM (+ MAD baseline) all computed
  in parallel from the same inference pass per frame
- LATENCY BENCHMARK: per-frame inference and per-metric compute latency logged
- NO annotated video — only JPEG snapshots when sleeping is detected
- Output to /content/drive/MyDrive/Nakul_Guardex/sleep_test/live_test/
- All metrics, detections, logs, and analysis written per-session

Usage (inside Colab after Drive is mounted):
  !python live_sleep_benchmark.py --duration 18000
  !python live_sleep_benchmark.py --smoke-only
"""

import argparse
import csv
import json
import logging
import os
import queue
import sys
import threading
import time
import traceback
import warnings
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path

import cv2
import numpy as np
import torch

warnings.filterwarnings("ignore")

# ── Optional imports (graceful fallback) ──────────────────────────────────────
try:
    from skimage.metrics import structural_similarity as ski_ssim
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    print("[WARN] scikit-image not found — SSIM will fallback to MAD/255")

try:
    import imagehash
    from PIL import Image as PILImage
    HAS_IMAGEHASH = True
except ImportError:
    HAS_IMAGEHASH = False
    print("[WARN] imagehash not found — pHash will fallback to 0.0")

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False
    print("[WARN] ultralytics not found — YOLO unavailable")

# ── IST timezone ──────────────────────────────────────────────────────────────
IST = timezone(timedelta(hours=5, minutes=30))


# ── Camera configuration ─────────────────────────────────────────────────────
MOHIT_CAMERAS = [
    {"name": "cam_03", "channel": "301",  "rtsp": "rtsp://admin:mohit%4012345@103.210.29.90:554/Streaming/Channels/301"},
    {"name": "cam_24", "channel": "2401", "rtsp": "rtsp://admin:mohit%4012345@103.210.29.90:554/Streaming/Channels/2401"},
    {"name": "cam_29", "channel": "2901", "rtsp": "rtsp://admin:mohit%4012345@103.210.29.90:554/Streaming/Channels/2901"},
    {"name": "cam_38", "channel": "3801", "rtsp": "rtsp://admin:mohit%4012345@103.210.29.90:554/Streaming/Channels/3801"},
    {"name": "cam_59", "channel": "5901", "rtsp": "rtsp://admin:mohit%4012345@103.210.29.90:554/Streaming/Channels/5901"},
]

# ── Decision logic parameters (1 FPS → 1 frame ≈ 1 second) ──────────────────
# OLD LOGIC — rolling mean windows (seconds) and diff band thresholds
OLD_WINDOWS   = [30, 60, 100, 200, 300]          # rolling window sizes in seconds
OLD_THRESHOLDS = [(0, 3), (0, 5), (0, 8), (0, 10)]  # (min_diff, max_diff) bands

# NEW LOGIC — longest consecutive streak of frames below per-frame threshold
NEW_PER_FRAME_THRESHOLDS = [5, 8, 10]            # per-frame diff thresholds
NEW_REQUIRED_STREAKS     = [30, 60, 100, 200, 300]  # seconds of stillness required

# Primary thresholds for photo-capture triggers
OLD_TRIGGER_WINDOW    = 60   # seconds rolling window
OLD_TRIGGER_THRESH    = (0, 8)  # diff band
NEW_TRIGGER_STREAK    = 60   # seconds streak
NEW_TRIGGER_PER_FRAME = 8    # per-frame diff threshold

# Lock averaging buffer
LOCK_AVG_FRAMES = 3
RELOCK_THRESHOLD = 10.0  # MAD threshold to relock reference

# ── Benchmark metric keys ─────────────────────────────────────────────────────
BENCHMARK_METRICS = ["MAD", "SSIM", "MOG2", "PHASH"]

# ── Output base directory ─────────────────────────────────────────────────────
OUTPUT_BASE = "/content/drive/MyDrive/Nakul_Guardex/sleep_test/live_test"


# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────────────────────────────

def now_ist_str(fmt="%Y-%m-%d_%H-%M-%S"):
    return datetime.now(IST).strftime(fmt)

def now_ist_date_str():
    return datetime.now(IST).strftime("%Y-%m-%d")

def get_sig(crop):
    """32x32 grayscale signature (same as test.py / pixel_diff_strategy_comparison.py)."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)

def averaged_box(box_buffer, w, h):
    """Average recent bboxes (jitter smoothing, same as test.py)."""
    boxes = np.array(box_buffer)
    avg = boxes.mean(axis=0)
    x1, y1, x2, y2 = avg
    return (
        int(np.clip(x1, 0, w)), int(np.clip(y1, 0, h)),
        int(np.clip(x2, 0, w)), int(np.clip(y2, 0, h)),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Decision logic helpers (OLD and NEW — kept fully intact)
# ─────────────────────────────────────────────────────────────────────────────

def old_logic_check(diff_history, window, thresh_pair):
    """
    OLD LOGIC: rolling mean of diff_history over `window` frames must stay
    within thresh_pair = (min_diff, max_diff).
    Returns (passes: bool, coverage_pct: float).
    Identical to old_logic_eval() in test.py.
    """
    n = len(diff_history)
    if n < window:
        return False, 0.0
    arr = np.array(diff_history)
    csum = np.cumsum(np.insert(arr, 0, 0))
    rolling_means = (csum[window:] - csum[:-window]) / window
    tmin, tmax = thresh_pair
    passing = np.where((rolling_means >= tmin) & (rolling_means <= tmax))[0]
    passes = len(passing) > 0
    covered = np.zeros(n, dtype=bool)
    for s in passing:
        covered[s: s + window] = True
    coverage_pct = 100.0 * covered.sum() / n
    return passes, coverage_pct


def new_logic_streak(diff_history, per_frame_threshold):
    """
    NEW LOGIC: longest consecutive streak of frames where diff < per_frame_threshold.
    Returns (longest_streak: int, best_range: tuple).
    Identical to longest_streak_under_threshold() in test.py.
    """
    max_streak, current_streak = 0, 0
    current_start, best_range = None, (0, 0)
    for i, d in enumerate(diff_history):
        if d <= per_frame_threshold:
            if current_streak == 0:
                current_start = i
            current_streak += 1
            if current_streak > max_streak:
                max_streak = current_streak
                best_range = (current_start, i)
        else:
            current_streak = 0
    return max_streak, best_range


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark metric computation (pHash, MOG2, SSIM, MAD)
# ─────────────────────────────────────────────────────────────────────────────

def compute_benchmark_metrics(ref_sig, cur_sig, mog2_subtractor):
    """
    Compute MAD, SSIM, MOG2, pHash on 32x32 grayscale patches.
    Returns dict of metric_name -> float value, plus latency dict in ms.
    """
    metrics = {}
    latencies = {}

    ref_f = ref_sig.astype(np.float32)
    cur_f = cur_sig.astype(np.float32)

    # MAD (baseline — same as old/new logic primary signal)
    t0 = time.perf_counter()
    metrics["MAD"] = float(np.mean(np.abs(ref_f - cur_f)))
    latencies["MAD"] = (time.perf_counter() - t0) * 1000

    # SSIM (inverted: 0 = identical, higher = more different)
    t0 = time.perf_counter()
    if HAS_SKIMAGE:
        try:
            ssim_val = ski_ssim(ref_sig, cur_sig, data_range=255)
            metrics["SSIM"] = float(1.0 - ssim_val)
        except Exception:
            metrics["SSIM"] = metrics["MAD"] / 255.0
    else:
        metrics["SSIM"] = metrics["MAD"] / 255.0
    latencies["SSIM"] = (time.perf_counter() - t0) * 1000

    # MOG2 — per-track foreground pixel ratio
    t0 = time.perf_counter()
    fg_mask = mog2_subtractor.apply(cur_sig)
    metrics["MOG2"] = float((fg_mask > 127).sum()) / (32 * 32)
    latencies["MOG2"] = (time.perf_counter() - t0) * 1000

    # pHash — Hamming distance
    t0 = time.perf_counter()
    if HAS_IMAGEHASH:
        try:
            h_ref = imagehash.phash(PILImage.fromarray(ref_sig))
            h_cur = imagehash.phash(PILImage.fromarray(cur_sig))
            metrics["PHASH"] = float(h_ref - h_cur)
        except Exception:
            metrics["PHASH"] = 0.0
    else:
        metrics["PHASH"] = 0.0
    latencies["PHASH"] = (time.perf_counter() - t0) * 1000

    return metrics, latencies


# ─────────────────────────────────────────────────────────────────────────────
# PersonTrack — tracks one detected person across frames
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PersonTrack:
    track_id: int
    reference_sig: np.ndarray = None
    reference_bbox: tuple = None
    total_frames: int = 0
    # Diff history — primary signal for old/new logic (MAD)
    diff_history: deque = field(default_factory=lambda: deque(maxlen=600))
    bbox_area_history: deque = field(default_factory=lambda: deque(maxlen=600))
    box_buffer: deque = field(default_factory=lambda: deque(maxlen=LOCK_AVG_FRAMES))
    # Per-metric benchmark histories
    metric_histories: dict = field(default_factory=lambda: {m: [] for m in BENCHMARK_METRICS})
    metric_latencies: dict = field(default_factory=lambda: {m: [] for m in BENCHMARK_METRICS})
    # MOG2 subtractor (per-track, fresh on relock)
    mog2: object = None
    # Snapshot for sleeping detection photo
    last_saved_crop: np.ndarray = None
    last_saved_full: np.ndarray = None
    # Trigger state — avoid repeated saves in same sleeping episode
    old_sleeping_triggered: bool = False
    new_sleeping_triggered: bool = False
    frames_since_old_trigger: int = 0
    frames_since_new_trigger: int = 0

    def __post_init__(self):
        self.mog2 = cv2.createBackgroundSubtractorMOG2(
            history=200, varThreshold=16, detectShadows=False
        )

    def reset_reference(self, sig, bbox):
        self.reference_sig = sig
        self.reference_bbox = bbox
        # Reset MOG2 on relock so it re-learns background
        self.mog2 = cv2.createBackgroundSubtractorMOG2(
            history=200, varThreshold=16, detectShadows=False
        )
        # Reset sleeping triggers on significant movement (relock)
        self.old_sleeping_triggered = False
        self.new_sleeping_triggered = False


# ─────────────────────────────────────────────────────────────────────────────
# CameraWorker — one per camera, runs in its own thread
# ─────────────────────────────────────────────────────────────────────────────

class CameraWorker(threading.Thread):
    """
    Processes one RTSP camera stream at ~1 FPS.
    - Runs YOLO inference per frame (shared GPU model via queue or per-thread)
    - Computes old logic, new logic, and benchmark metrics in parallel
    - Saves JPEG snapshots when sleeping detected
    - Writes per-track CSV rows and per-frame latency logs
    """

    def __init__(
        self,
        cam_info: dict,
        session_dir: str,
        model_path: str,
        duration_seconds: int,
        result_queue: queue.Queue,
        smoke_mode: bool = False,
        smoke_frames: int = 1,
        global_csv_lock: threading.Lock = None,
        global_csv_path: str = None,
        latency_csv_lock: threading.Lock = None,
        latency_csv_path: str = None,
    ):
        super().__init__(daemon=True)
        self.cam_info = cam_info
        self.cam_name = cam_info["name"]
        self.rtsp_url = cam_info["rtsp"]
        self.session_dir = session_dir
        self.model_path = model_path
        self.duration_seconds = duration_seconds
        self.result_queue = result_queue
        self.smoke_mode = smoke_mode
        self.smoke_frames = smoke_frames
        self.global_csv_lock = global_csv_lock
        self.global_csv_path = global_csv_path
        self.latency_csv_lock = latency_csv_lock
        self.latency_csv_path = latency_csv_path

        # Per-camera output directory
        self.cam_dir = os.path.join(session_dir, self.cam_name)
        os.makedirs(os.path.join(self.cam_dir, "snapshots"), exist_ok=True)
        os.makedirs(os.path.join(self.cam_dir, "logs"), exist_ok=True)

        # Logger
        self.log = self._setup_logger()
        self.tracks = {}
        self.stop_event = threading.Event()
        self.frames_processed = 0
        self.error_count = 0
        self.status = "INIT"

    def _setup_logger(self):
        log_path = os.path.join(self.cam_dir, "logs", f"{self.cam_name}.log")
        logger = logging.getLogger(f"bench.{self.cam_name}")
        logger.setLevel(logging.DEBUG)
        if not logger.handlers:
            fh = logging.FileHandler(log_path)
            fh.setFormatter(logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            ))
            logger.addHandler(fh)
            # Also echo to console
            ch = logging.StreamHandler(sys.stdout)
            ch.setLevel(logging.INFO)
            ch.setFormatter(logging.Formatter(
                "[%(asctime)s] %(name)s | %(message)s", datefmt="%H:%M:%S"
            ))
            logger.addHandler(ch)
        return logger

    def _open_stream(self):
        """Open RTSP stream with retry logic."""
        cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if cap.isOpened():
            self.log.info(f"Stream opened: {self.rtsp_url}")
        else:
            self.log.error(f"Failed to open stream: {self.rtsp_url}")
        return cap

    def _grab_latest_frame(self, cap):
        """
        Drain buffered frames and return only the latest (live frame).
        Prevents processing stale buffered frames at 1 FPS.
        """
        ret, frame = False, None
        for _ in range(4):  # Drain up to 4 buffered frames
            ret = cap.grab()
            if not ret:
                return False, None
        ret, frame = cap.retrieve()
        return ret, frame

    def _save_sleeping_snapshot(self, track: PersonTrack, frame: np.ndarray,
                                 x1, y1, x2, y2, logic: str, trigger_info: str):
        """Save JPEG crop + full frame snapshot when sleeping is detected."""
        ts = now_ist_str()
        date = now_ist_date_str()
        snap_dir = os.path.join(self.cam_dir, "snapshots", date)
        os.makedirs(snap_dir, exist_ok=True)

        h, w = frame.shape[:2]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        bw, bh = x2 - x1, y2 - y1
        margin = int(max(bw, bh) * 0.5)

        # Crop around person
        ys, ye = max(0, int(cy) - margin), min(h, int(cy) + margin)
        xs, xe = max(0, int(cx) - margin), min(w, int(cx) + margin)
        crop = frame[ys:ye, xs:xe]

        crop_path = os.path.join(
            snap_dir,
            f"{self.cam_name}_SLEEP_{logic}_id{track.track_id}_{ts}_crop.jpg"
        )
        full_path = os.path.join(
            snap_dir,
            f"{self.cam_name}_SLEEP_{logic}_id{track.track_id}_{ts}_full.jpg"
        )

        if crop.size > 0:
            cv2.imwrite(crop_path, crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
        cv2.imwrite(full_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])

        self.log.warning(
            f"🛌 SLEEPING DETECTED [{logic}] track_id={track.track_id} "
            f"| {trigger_info} | crop={crop_path}"
        )
        return crop_path, full_path

    def _append_global_csv(self, row: dict):
        """Thread-safe append to global benchmark CSV."""
        if self.global_csv_lock is None or self.global_csv_path is None:
            return
        fieldnames = [
            "timestamp", "cam_name", "track_id", "total_frames",
            "MAD_mean", "MAD_std", "SSIM_mean", "MOG2_mean", "PHASH_mean",
            "old_logic_triggered", "new_logic_triggered",
            "old_pass_W60_thresh8", "new_streak_thresh8",
            "snapshot_crop", "snapshot_full",
        ]
        with self.global_csv_lock:
            exists = os.path.exists(self.global_csv_path)
            with open(self.global_csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                if not exists:
                    writer.writeheader()
                writer.writerow(row)

    def _append_latency_csv(self, row: dict):
        """Thread-safe append to global latency CSV."""
        if self.latency_csv_lock is None or self.latency_csv_path is None:
            return
        fieldnames = [
            "timestamp", "cam_name", "track_id", "frame_idx",
            "infer_ms", "MAD_ms", "SSIM_ms", "MOG2_ms", "PHASH_ms",
            "total_pipeline_ms",
        ]
        with self.latency_csv_lock:
            exists = os.path.exists(self.latency_csv_path)
            with open(self.latency_csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                if not exists:
                    writer.writeheader()
                writer.writerow(row)

    def run(self):
        """Main camera processing loop."""
        self.status = "STARTING"
        self.log.info(
            f"Camera worker started | smoke={self.smoke_mode} | "
            f"duration={self.duration_seconds}s | model={self.model_path}"
        )

        # ── Load YOLO model ──────────────────────────────────────────────────
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = YOLO(self.model_path)
            self.log.info(f"YOLO model loaded on {device}")
        except Exception as e:
            self.log.error(f"Failed to load YOLO model: {e}")
            self.status = "MODEL_LOAD_FAILED"
            self.result_queue.put({
                "cam": self.cam_name, "status": "FAILED", "reason": str(e)
            })
            return

        cap = None
        start_time = time.time()
        next_frame_time = start_time
        frame_interval = 1.0  # 1 FPS

        while not self.stop_event.is_set():
            elapsed = time.time() - start_time
            if elapsed >= self.duration_seconds:
                self.log.info(f"Duration {self.duration_seconds}s reached. Stopping.")
                break

            # Smoke mode: stop after required frames
            if self.smoke_mode and self.frames_processed >= self.smoke_frames:
                break

            # ── Rate limit to 1 FPS ──────────────────────────────────────────
            now = time.time()
            if now < next_frame_time:
                time.sleep(max(0, next_frame_time - now))
            next_frame_time = time.time() + frame_interval

            frame_pipeline_start = time.perf_counter()

            # ── Open / reconnect stream ──────────────────────────────────────
            try:
                if cap is None or not cap.isOpened():
                    cap = self._open_stream()
                    if not cap.isOpened():
                        self.error_count += 1
                        self.log.warning(f"Stream not open. Retry in 5s. (error #{self.error_count})")
                        time.sleep(5)
                        cap = None
                        continue

                ret, frame = self._grab_latest_frame(cap)
            except Exception as e:
                self.log.error(f"Frame grab error: {e}")
                ret, frame = False, None

            if not ret or frame is None:
                self.log.warning("No frame received. Retrying in 3s.")
                if cap:
                    cap.release()
                cap = None
                time.sleep(3)
                continue

            h_frame, w_frame = frame.shape[:2]

            # ── YOLO inference ───────────────────────────────────────────────
            infer_start = time.perf_counter()
            try:
                results = model.track(
                    frame,
                    persist=True,
                    conf=0.25,
                    imgsz=1280,
                    classes=[0],   # Person only
                    device=device,
                    verbose=False,
                    half=True,     # FP16 on T4
                )
            except Exception as e:
                self.log.error(f"Inference error: {e}")
                self.frames_processed += 1
                continue
            infer_ms = (time.perf_counter() - infer_start) * 1000

            self.frames_processed += 1
            self.status = "RUNNING"
            self.log.debug(f"Frame #{self.frames_processed} | infer={infer_ms:.1f}ms")

            # ── Parse detections ─────────────────────────────────────────────
            current_tids = set()
            if (results and results[0].boxes is not None
                    and results[0].boxes.id is not None):
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy()
            else:
                boxes, track_ids = np.array([]), np.array([])

            frame_latency_rows = []

            # ── Per-detection processing ─────────────────────────────────────
            for box, tid in zip(boxes, track_ids):
                tid = int(tid)
                current_tids.add(tid)

                if tid not in self.tracks:
                    self.tracks[tid] = PersonTrack(track_id=tid)
                track = self.tracks[tid]

                x1, y1, x2, y2 = map(int, box)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w_frame, x2), min(h_frame, y2)

                # Skip tiny detections
                if (x2 - x1) < 32 or (y2 - y1) < 16:
                    continue

                track.box_buffer.append((x1, y1, x2, y2))

                # ── Reference locking (same as test.py) ──────────────────────
                if track.reference_bbox is None:
                    if len(track.box_buffer) < LOCK_AVG_FRAMES:
                        continue
                    lx1, ly1, lx2, ly2 = averaged_box(track.box_buffer, w_frame, h_frame)
                    crop = frame[ly1:ly2, lx1:lx2]
                    if crop.size > 0:
                        sig = get_sig(crop)
                        track.reset_reference(sig, (lx1, ly1, lx2, ly2))
                        self.log.debug(f"Track {tid} reference locked at {track.reference_bbox}")
                    continue

                # ── Diff computation ─────────────────────────────────────────
                rx1, ry1, rx2, ry2 = track.reference_bbox
                ref_crop = frame[ry1:ry2, rx1:rx2]
                if ref_crop.size == 0:
                    track.reference_bbox = None
                    continue

                cur_sig = get_sig(ref_crop)

                # Compute all benchmark metrics (MAD, SSIM, MOG2, PHASH)
                bench_metrics, bench_latencies = compute_benchmark_metrics(
                    track.reference_sig, cur_sig, track.mog2
                )

                mad_val = bench_metrics["MAD"]
                track.diff_history.append(mad_val)
                current_area = (x2 - x1) * (y2 - y1)
                track.bbox_area_history.append(current_area)
                track.total_frames += 1

                # Log per-metric histories
                for m_key in BENCHMARK_METRICS:
                    track.metric_histories[m_key].append(bench_metrics.get(m_key, 0.0))
                    track.metric_latencies[m_key].append(bench_latencies.get(m_key, 0.0))

                # ── Relock on significant motion (same as test.py) ────────────
                if mad_val > RELOCK_THRESHOLD:
                    lx1, ly1, lx2, ly2 = averaged_box(track.box_buffer, w_frame, h_frame)
                    new_crop = frame[ly1:ly2, lx1:lx2]
                    if new_crop.size > 0:
                        new_sig = get_sig(new_crop)
                        track.reset_reference(new_sig, (lx1, ly1, lx2, ly2))
                        self.log.debug(f"Track {tid} RELOCKED (MAD={mad_val:.1f})")

                # ── OLD LOGIC evaluation ──────────────────────────────────────
                diff_list = list(track.diff_history)
                old_passes, old_cov = old_logic_check(
                    diff_list, OLD_TRIGGER_WINDOW, OLD_TRIGGER_THRESH
                )

                # OLD LOGIC sleeping trigger (with 60-frame cooldown)
                track.frames_since_old_trigger += 1
                if old_passes and (not track.old_sleeping_triggered
                                   or track.frames_since_old_trigger > 120):
                    crop_p, full_p = self._save_sleeping_snapshot(
                        track, frame, x1, y1, x2, y2, "OLD",
                        f"window={OLD_TRIGGER_WINDOW}s thresh={OLD_TRIGGER_THRESH} cov={old_cov:.1f}%"
                    )
                    track.old_sleeping_triggered = True
                    track.frames_since_old_trigger = 0
                    self._append_global_csv({
                        "timestamp": now_ist_str("%Y-%m-%d %H:%M:%S"),
                        "cam_name": self.cam_name,
                        "track_id": tid,
                        "total_frames": track.total_frames,
                        "MAD_mean": f"{np.mean(track.metric_histories['MAD']):.3f}",
                        "MAD_std":  f"{np.std(track.metric_histories['MAD']):.3f}",
                        "SSIM_mean": f"{np.mean(track.metric_histories['SSIM']):.4f}" if track.metric_histories['SSIM'] else "N/A",
                        "MOG2_mean": f"{np.mean(track.metric_histories['MOG2']):.4f}" if track.metric_histories['MOG2'] else "N/A",
                        "PHASH_mean": f"{np.mean(track.metric_histories['PHASH']):.2f}" if track.metric_histories['PHASH'] else "N/A",
                        "old_logic_triggered": True,
                        "new_logic_triggered": False,
                        "old_pass_W60_thresh8": True,
                        "new_streak_thresh8": 0,
                        "snapshot_crop": crop_p,
                        "snapshot_full": full_p,
                    })

                # ── NEW LOGIC evaluation ──────────────────────────────────────
                new_streak, _ = new_logic_streak(diff_list, NEW_TRIGGER_PER_FRAME)
                new_passes = new_streak >= NEW_TRIGGER_STREAK

                # NEW LOGIC sleeping trigger (with 60-frame cooldown)
                track.frames_since_new_trigger += 1
                if new_passes and (not track.new_sleeping_triggered
                                   or track.frames_since_new_trigger > 120):
                    crop_p, full_p = self._save_sleeping_snapshot(
                        track, frame, x1, y1, x2, y2, "NEW",
                        f"streak={new_streak}s thresh={NEW_TRIGGER_PER_FRAME}"
                    )
                    track.new_sleeping_triggered = True
                    track.frames_since_new_trigger = 0
                    self._append_global_csv({
                        "timestamp": now_ist_str("%Y-%m-%d %H:%M:%S"),
                        "cam_name": self.cam_name,
                        "track_id": tid,
                        "total_frames": track.total_frames,
                        "MAD_mean": f"{np.mean(track.metric_histories['MAD']):.3f}",
                        "MAD_std":  f"{np.std(track.metric_histories['MAD']):.3f}",
                        "SSIM_mean": f"{np.mean(track.metric_histories['SSIM']):.4f}" if track.metric_histories['SSIM'] else "N/A",
                        "MOG2_mean": f"{np.mean(track.metric_histories['MOG2']):.4f}" if track.metric_histories['MOG2'] else "N/A",
                        "PHASH_mean": f"{np.mean(track.metric_histories['PHASH']):.2f}" if track.metric_histories['PHASH'] else "N/A",
                        "old_logic_triggered": False,
                        "new_logic_triggered": True,
                        "old_pass_W60_thresh8": old_passes,
                        "new_streak_thresh8": new_streak,
                        "snapshot_crop": crop_p,
                        "snapshot_full": full_p,
                    })

                # ── Latency logging (every frame) ─────────────────────────────
                pipeline_ms = (time.perf_counter() - frame_pipeline_start) * 1000
                self._append_latency_csv({
                    "timestamp": now_ist_str("%Y-%m-%d %H:%M:%S"),
                    "cam_name": self.cam_name,
                    "track_id": tid,
                    "frame_idx": self.frames_processed,
                    "infer_ms": f"{infer_ms:.2f}",
                    "MAD_ms":   f"{bench_latencies.get('MAD', 0):.3f}",
                    "SSIM_ms":  f"{bench_latencies.get('SSIM', 0):.3f}",
                    "MOG2_ms":  f"{bench_latencies.get('MOG2', 0):.3f}",
                    "PHASH_ms": f"{bench_latencies.get('PHASH', 0):.3f}",
                    "total_pipeline_ms": f"{pipeline_ms:.2f}",
                })

            # ── Cleanup lost tracks ────────────────────────────────────────────
            for lost_tid in list(self.tracks.keys()):
                if lost_tid not in current_tids:
                    lost_track = self.tracks[lost_tid]
                    if lost_track.total_frames >= 10:
                        self._flush_track_summary(lost_track)
                    del self.tracks[lost_tid]

        # ── Cleanup at end of run ─────────────────────────────────────────────
        if cap:
            cap.release()

        # Flush remaining active tracks
        for tid, track in self.tracks.items():
            if track.total_frames >= 10:
                self._flush_track_summary(track)

        self.status = "DONE"
        self.log.info(
            f"Worker done | frames_processed={self.frames_processed} | "
            f"errors={self.error_count} | tracks={len(self.tracks)}"
        )
        self.result_queue.put({
            "cam": self.cam_name,
            "status": "OK",
            "frames": self.frames_processed,
            "tracks": len(self.tracks),
        })

    def _flush_track_summary(self, track: PersonTrack):
        """Write per-track benchmark JSON summary when track is lost or session ends."""
        if not track.metric_histories.get("MAD"):
            return
        summary = {
            "cam_name": self.cam_name,
            "track_id": track.track_id,
            "total_frames": track.total_frames,
            "metrics": {},
        }
        for m_key in BENCHMARK_METRICS:
            hist = track.metric_histories.get(m_key, [])
            lats = track.metric_latencies.get(m_key, [])
            if hist:
                summary["metrics"][m_key] = {
                    "history": hist[-500:],  # cap to last 500 for file size
                    "mean": float(np.mean(hist)),
                    "std": float(np.std(hist)),
                    "min": float(np.min(hist)),
                    "max": float(np.max(hist)),
                    "mean_latency_ms": float(np.mean(lats)) if lats else 0.0,
                }
        # Old logic sweep results
        diff_list = list(track.diff_history)
        old_sweep = {}
        for w in OLD_WINDOWS:
            for tp in OLD_THRESHOLDS:
                passes, cov = old_logic_check(diff_list, w, tp)
                old_sweep[f"W{w}_T{tp[0]}-{tp[1]}"] = {"passes": passes, "coverage_pct": round(cov, 2)}
        summary["old_logic_sweep"] = old_sweep

        # New logic sweep results
        new_sweep = {}
        for thresh in NEW_PER_FRAME_THRESHOLDS:
            streak, rng = new_logic_streak(diff_list, thresh)
            for req in NEW_REQUIRED_STREAKS:
                new_sweep[f"thresh{thresh}_req{req}"] = {
                    "longest_streak": streak,
                    "passes": streak >= req,
                    "best_range": list(rng),
                }
        summary["new_logic_sweep"] = new_sweep

        out_path = os.path.join(
            self.cam_dir, "logs",
            f"track_{track.track_id}_benchmark.json"
        )
        try:
            with open(out_path, "w") as f:
                json.dump(summary, f, indent=2)
        except Exception as e:
            self.log.error(f"Failed to write track summary: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────

def run_smoke_test(model_path: str, session_dir: str, duration_secs: int = 300):
    """
    Smoke test: runs all 5 cameras for `duration_secs` (default 5 minutes),
    1 frame each. Checks:
      1. RTSP connectivity for each camera
      2. YOLO model loads on GPU
      3. At least 1 frame readable per camera
      4. Write access to Drive output folder
    Returns True if all checks pass, raises RuntimeError on critical failure.
    """
    print("\n" + "=" * 70)
    print("  🔥 SMOKE TEST — 5-minute pre-flight check")
    print(f"  Duration: {duration_secs}s | Cameras: {len(MOHIT_CAMERAS)}")
    print("=" * 70 + "\n")

    results = {}
    errors = []

    # ── Check 1: Drive write access ──────────────────────────────────────────
    test_file = os.path.join(session_dir, "smoke_test_write_check.txt")
    try:
        os.makedirs(session_dir, exist_ok=True)
        with open(test_file, "w") as f:
            f.write(f"Smoke test at {now_ist_str()}\n")
        os.remove(test_file)
        print("✅ [CHECK 1/3] Drive write access: OK")
    except Exception as e:
        err = f"Drive write access FAILED: {e}"
        print(f"❌ [CHECK 1/3] {err}")
        errors.append(err)

    # ── Check 2: YOLO model load ─────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   Device: {device}")
    try:
        model = YOLO(model_path)
        print(f"✅ [CHECK 2/3] YOLO model '{model_path}' loaded on {device}: OK")
        gpu_ok = True
    except Exception as e:
        err = f"YOLO model load FAILED: {e}"
        print(f"❌ [CHECK 2/3] {err}")
        errors.append(err)
        gpu_ok = False

    # ── Check 3: RTSP connectivity + frame grab per camera ───────────────────
    print(f"\n   Checking {len(MOHIT_CAMERAS)} camera streams for {duration_secs}s...")
    result_q = queue.Queue()
    global_csv_lock = threading.Lock()
    global_csv_path = os.path.join(session_dir, "smoke_test_detections.csv")
    latency_csv_lock = threading.Lock()
    latency_csv_path = os.path.join(session_dir, "smoke_test_latency.csv")

    workers = []
    for cam in MOHIT_CAMERAS:
        w = CameraWorker(
            cam_info=cam,
            session_dir=session_dir,
            model_path=model_path,
            duration_seconds=duration_secs,
            result_queue=result_q,
            smoke_mode=False,    # Run for full duration_secs, not just 1 frame
            smoke_frames=1,
            global_csv_lock=global_csv_lock,
            global_csv_path=global_csv_path,
            latency_csv_lock=latency_csv_lock,
            latency_csv_path=latency_csv_path,
        )
        w.start()
        workers.append(w)
        time.sleep(1.0)  # Stagger GPU init

    # Wait for all workers to complete smoke duration
    for w in workers:
        w.join(timeout=duration_secs + 60)

    # Collect results
    cam_results = {}
    while not result_q.empty():
        r = result_q.get_nowait()
        cam_results[r["cam"]] = r

    print("\n  📊 Smoke Test Results:")
    all_ok = True
    for cam in MOHIT_CAMERAS:
        cam_name = cam["name"]
        r = cam_results.get(cam_name, {})
        if r.get("status") == "OK" and r.get("frames", 0) > 0:
            print(f"  ✅ {cam_name}: {r['frames']} frames | {r.get('tracks', 0)} tracks")
            results[cam_name] = "PASS"
        else:
            reason = r.get("reason", "No frames received")
            print(f"  ❌ {cam_name}: FAILED — {reason}")
            results[cam_name] = f"FAIL: {reason}"
            all_ok = False

    # Write smoke test report
    report_path = os.path.join(session_dir, "smoke_test_report.json")
    report = {
        "timestamp": now_ist_str("%Y-%m-%d %H:%M:%S"),
        "duration_secs": duration_secs,
        "drive_write": "OK" if not errors else "FAILED",
        "gpu_model_load": "OK" if gpu_ok else "FAILED",
        "camera_results": results,
        "overall": "PASS" if all_ok and not errors else "FAIL",
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  📄 Smoke test report saved: {report_path}")

    if errors:
        raise RuntimeError(f"Smoke test CRITICAL FAILURES: {errors}")

    if not all_ok:
        print("\n  ⚠️  Some cameras failed smoke test. Check logs before proceeding.")
    else:
        print("\n  ✅ All smoke checks PASSED. Ready for 5-hour run.")

    print("=" * 70 + "\n")
    return all_ok, report


# ─────────────────────────────────────────────────────────────────────────────
# Analysis report generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_analysis_report(session_dir: str):
    """
    Post-run analysis: reads all per-track JSONs and latency CSV,
    generates a comprehensive markdown + JSON analysis report.
    """
    print("\n📊 Generating analysis report...")
    report_lines = [
        "# Live Stream Sleep Detection — Benchmark Analysis Report",
        f"Generated: {now_ist_str('%Y-%m-%d %H:%M:%S')} IST",
        "",
        "## Camera Summary",
        "",
    ]

    all_metric_stats = {m: {"mean": [], "std": [], "lat_ms": []} for m in BENCHMARK_METRICS}
    total_old_triggers = 0
    total_new_triggers = 0

    # Read global CSV
    global_csv = os.path.join(session_dir, "global_benchmark.csv")
    if os.path.exists(global_csv):
        with open(global_csv, "r") as f:
            rows = list(csv.DictReader(f))
        total_old_triggers = sum(1 for r in rows if r.get("old_logic_triggered") == "True")
        total_new_triggers = sum(1 for r in rows if r.get("new_logic_triggered") == "True")
        report_lines.extend([
            f"- Total OLD logic sleep triggers: **{total_old_triggers}**",
            f"- Total NEW logic sleep triggers: **{total_new_triggers}**",
            "",
        ])

    # Per-camera analysis
    for cam in MOHIT_CAMERAS:
        cam_name = cam["name"]
        cam_dir = os.path.join(session_dir, cam_name)
        logs_dir = os.path.join(cam_dir, "logs")

        report_lines.append(f"### {cam_name}")

        # Find all track JSONs
        track_jsons = list(Path(logs_dir).glob("track_*_benchmark.json")) if os.path.isdir(logs_dir) else []
        report_lines.append(f"- Unique tracked persons: **{len(track_jsons)}**")

        for tj_path in track_jsons:
            try:
                with open(tj_path) as f:
                    tj = json.load(f)
            except Exception:
                continue

            tid = tj.get("track_id", "?")
            total_f = tj.get("total_frames", 0)
            report_lines.append(f"  - Track {tid} | Frames: {total_f}")

            for m_key in BENCHMARK_METRICS:
                m_data = tj.get("metrics", {}).get(m_key, {})
                if m_data:
                    mean_v = m_data.get("mean", 0)
                    std_v = m_data.get("std", 0)
                    lat_v = m_data.get("mean_latency_ms", 0)
                    report_lines.append(
                        f"    - {m_key}: mean={mean_v:.3f} std={std_v:.3f} lat={lat_v:.2f}ms"
                    )
                    all_metric_stats[m_key]["mean"].append(mean_v)
                    all_metric_stats[m_key]["std"].append(std_v)
                    all_metric_stats[m_key]["lat_ms"].append(lat_v)

            # Old logic sweep summary
            old_sweep = tj.get("old_logic_sweep", {})
            best_old = [(k, v) for k, v in old_sweep.items() if v.get("passes")]
            if best_old:
                report_lines.append(f"    - OLD logic PASS configs: {[k for k, v in best_old]}")

            # New logic sweep summary
            new_sweep = tj.get("new_logic_sweep", {})
            best_new = [(k, v) for k, v in new_sweep.items() if v.get("passes")]
            if best_new:
                report_lines.append(f"    - NEW logic PASS configs: {[k for k, v in best_new]}")

        report_lines.append("")

    # Metric comparison table
    report_lines.extend([
        "",
        "## Metric Benchmark Summary (all cameras, all tracks)",
        "",
        "| Metric | Mean Value | Std | Avg Latency (ms) |",
        "|--------|-----------|-----|-----------------|",
    ])
    for m_key in BENCHMARK_METRICS:
        stats = all_metric_stats[m_key]
        if stats["mean"]:
            gm = np.mean(stats["mean"])
            gs = np.mean(stats["std"])
            gl = np.mean(stats["lat_ms"])
            report_lines.append(f"| {m_key} | {gm:.4f} | {gs:.4f} | {gl:.2f} |")

    # Latency analysis from CSV
    latency_csv = os.path.join(session_dir, "latency_log.csv")
    if os.path.exists(latency_csv):
        with open(latency_csv, "r") as f:
            lat_rows = list(csv.DictReader(f))
        if lat_rows:
            infer_ms_vals = [float(r["infer_ms"]) for r in lat_rows if r.get("infer_ms")]
            pipe_ms_vals = [float(r["total_pipeline_ms"]) for r in lat_rows if r.get("total_pipeline_ms")]
            report_lines.extend([
                "",
                "## Latency Analysis (all frames across all cameras)",
                "",
                f"- Total frame measurements: **{len(lat_rows)}**",
                f"- YOLO Inference: mean={np.mean(infer_ms_vals):.1f}ms  p50={np.percentile(infer_ms_vals, 50):.1f}ms  p95={np.percentile(infer_ms_vals, 95):.1f}ms",
                f"- Full Pipeline: mean={np.mean(pipe_ms_vals):.1f}ms  p50={np.percentile(pipe_ms_vals, 50):.1f}ms  p95={np.percentile(pipe_ms_vals, 95):.1f}ms",
            ])

    report_lines.extend([
        "",
        "## Key Findings",
        "",
        "- **OLD LOGIC** (rolling mean): triggers on sustained low-motion periods with configurable window/threshold sweep.",
        "- **NEW LOGIC** (consecutive streak): more robust against brief movement interruptions; requires unbroken stillness.",
        "- **pHash**: computationally cheap perceptual hash — good for scene-level changes.",
        "- **MOG2**: adaptive background model — sensitive to local pixel changes even with lighting variation.",
        "- **SSIM**: structural similarity — captures texture/structure changes beyond raw pixel diff.",
        "- **MAD**: baseline fast metric — remains reliable reference signal.",
        "",
        "> All metrics computed in parallel on same inference pass (no extra GPU cost).",
    ])

    report_text = "\n".join(report_lines)
    report_md_path = os.path.join(session_dir, "analysis_report.md")
    with open(report_md_path, "w") as f:
        f.write(report_text)
    print(f"✅ Analysis report saved: {report_md_path}")
    return report_md_path


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--duration", type=int, default=18000,
        help="Main run duration in seconds (default: 18000 = 5 hours)"
    )
    parser.add_argument(
        "--model", type=str, default="yolo11m.pt",
        help="YOLO model path (default: yolo11m.pt)"
    )
    parser.add_argument(
        "--output-dir", type=str, default=OUTPUT_BASE,
        help=f"Output base directory (default: {OUTPUT_BASE})"
    )
    parser.add_argument(
        "--smoke-only", action="store_true",
        help="Run only the 5-minute smoke test and exit"
    )
    parser.add_argument(
        "--smoke-duration", type=int, default=300,
        help="Smoke test duration in seconds (default: 300 = 5 minutes)"
    )
    parser.add_argument(
        "--skip-smoke", action="store_true",
        help="Skip smoke test and jump directly to main run"
    )
    args = parser.parse_args()

    # ── Session directory ─────────────────────────────────────────────────────
    session_ts = now_ist_str("%Y-%m-%d_%H-%M-%S")
    session_dir = os.path.join(args.output_dir, f"session_{session_ts}")
    os.makedirs(session_dir, exist_ok=True)

    # Save session config
    config_path = os.path.join(session_dir, "session_config.json")
    with open(config_path, "w") as f:
        json.dump({
            "session_dir": session_dir,
            "model": args.model,
            "duration_secs": args.duration,
            "cameras": MOHIT_CAMERAS,
            "old_logic": {
                "windows": OLD_WINDOWS,
                "thresholds": [list(t) for t in OLD_THRESHOLDS],
                "trigger_window": OLD_TRIGGER_WINDOW,
                "trigger_thresh": list(OLD_TRIGGER_THRESH),
            },
            "new_logic": {
                "per_frame_thresholds": NEW_PER_FRAME_THRESHOLDS,
                "required_streaks": NEW_REQUIRED_STREAKS,
                "trigger_streak": NEW_TRIGGER_STREAK,
                "trigger_per_frame": NEW_TRIGGER_PER_FRAME,
            },
            "benchmark_metrics": BENCHMARK_METRICS,
            "fps": 1,
        }, f, indent=2)
    print(f"\n📁 Session directory: {session_dir}")
    print(f"📄 Config saved: {config_path}\n")

    # Shared CSVs
    global_csv_lock = threading.Lock()
    global_csv_path = os.path.join(session_dir, "global_benchmark.csv")
    latency_csv_lock = threading.Lock()
    latency_csv_path = os.path.join(session_dir, "latency_log.csv")

    # ── Smoke test ───────────────────────────────────────────────────────────
    if not args.skip_smoke:
        smoke_ok, smoke_report = run_smoke_test(
            model_path=args.model,
            session_dir=session_dir,
            duration_secs=args.smoke_duration,
        )
        if args.smoke_only:
            print("Smoke-only mode. Exiting.")
            return

        if not smoke_ok:
            print("⚠️  Smoke test had failures. Proceeding with caution...")
    else:
        print("⚠️  Smoke test SKIPPED by user flag.")

    # ── Main 5-hour benchmark run ─────────────────────────────────────────────
    print(f"\n🚀 Starting main benchmark run — {args.duration}s ({args.duration//3600}h {(args.duration%3600)//60}m)")
    print(f"   Cameras: {[c['name'] for c in MOHIT_CAMERAS]}")
    print(f"   Metrics: {BENCHMARK_METRICS}")
    print(f"   Output: {session_dir}\n")

    result_q = queue.Queue()
    workers = []

    for cam in MOHIT_CAMERAS:
        w = CameraWorker(
            cam_info=cam,
            session_dir=session_dir,
            model_path=args.model,
            duration_seconds=args.duration,
            result_queue=result_q,
            smoke_mode=False,
            global_csv_lock=global_csv_lock,
            global_csv_path=global_csv_path,
            latency_csv_lock=latency_csv_lock,
            latency_csv_path=latency_csv_path,
        )
        w.start()
        workers.append(w)
        print(f"   ✅ Started worker: {cam['name']}")
        time.sleep(2.0)  # Stagger GPU warm-up per camera

    print(f"\n⏱  {len(workers)} camera workers running. Waiting {args.duration}s...\n")

    # Progress printer every 5 minutes
    run_start = time.time()
    try:
        while True:
            elapsed = time.time() - run_start
            remaining = max(0, args.duration - elapsed)
            all_done = all(not w.is_alive() for w in workers)
            status_str = " | ".join(f"{w.cam_name}:{w.status}[{w.frames_processed}f]" for w in workers)
            print(f"  [{time.strftime('%H:%M:%S')}] Elapsed={elapsed/60:.1f}min Remaining={remaining/60:.1f}min | {status_str}")
            if all_done or remaining <= 0:
                break
            time.sleep(300)  # Print every 5 minutes
    except KeyboardInterrupt:
        print("\n⚠️  KeyboardInterrupt received — stopping workers...")
        for w in workers:
            w.stop_event.set()

    # Wait for all workers to finish
    for w in workers:
        w.join(timeout=60)

    # Collect results
    print("\n📊 Final results:")
    cam_results = {}
    while not result_q.empty():
        r = result_q.get_nowait()
        cam_results[r["cam"]] = r
        print(f"  {r['cam']}: {r['status']} | frames={r.get('frames', 0)} | tracks={r.get('tracks', 0)}")

    # Save final results
    results_path = os.path.join(session_dir, "run_results.json")
    with open(results_path, "w") as f:
        json.dump({
            "session": session_ts,
            "duration_secs": args.duration,
            "cam_results": cam_results,
            "timestamp": now_ist_str("%Y-%m-%d %H:%M:%S"),
        }, f, indent=2)

    # Generate analysis
    report_path = generate_analysis_report(session_dir)
    print(f"\n✅ All done! Session: {session_dir}")
    print(f"📄 Analysis report: {report_path}")


if __name__ == "__main__":
    main()
