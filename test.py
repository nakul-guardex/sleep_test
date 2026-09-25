"""
test.py / sleep_video_test.py

STANDALONE test script for evaluating sleep detection logic.
Mirrors sleep_new.py's tracking/reference/diff logic with 3-frame averaged lock,
evaluating both OLD (rolling mean) and NEW (consecutive streak) decision logic.

Supported Detection & Tracking Backends:
1. YOLO (Ultralytics): e.g. yolo11m.pt, yolo11l.pt
2. RT-DETR (Ultralytics): e.g. rtdetr-l.pt, rtdetr-x.pt
3. RF-DETR (Roboflow): e.g. rfdetr-medium, rfdetr-large via ByteTrack

Features:
- Inference on CUDA (Colab/NVIDIA) or MPS (Apple Silicon GPU)
- Organizes each video's annotated MP4, JSON histories, charts, and CSV/JSON summaries
  into its own dedicated subfolder inside anotated_video/
- Minute-by-minute progress logging with speed (fps) and ETA
"""

import argparse
import csv
import glob
import json
import os
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

# ── Decision-logic sweep parameters ───────────────────────────────────────
OLD_WINDOWS = [100, 200, 250, 300, 500]
OLD_MEAN_THRESHOLDS = [(0, 3), (0, 5), (0, 8), (0, 10)]

NEW_PER_FRAME_THRESHOLDS = [5, 8, 10]
NEW_REQUIRED_STREAKS = [100, 150, 200, 250, 300]

LOCK_AVERAGE_FRAMES = 3  # jitter smoothing fix


def get_decision_params(effective_fps=1.0):
    """
    Returns (windows, required_streaks) tailored to the effective processing FPS.
    At ~1 FPS, 1 frame = ~1 second, so windows/streaks are expressed in seconds.
    At native CCTV FPS (>=5 FPS), windows/streaks are expressed in frame counts.
    """
    if effective_fps <= 2.0:
        # Time-based windows/streaks in seconds (1 frame = 1 second)
        old_windows = [10, 30, 60, 100, 150, 200, 300]
        new_streaks = [10, 20, 30, 60, 100, 150, 200, 300]
    else:
        old_windows = OLD_WINDOWS
        new_streaks = NEW_REQUIRED_STREAKS
    return old_windows, new_streaks


class PersonTrack:
    def __init__(self, track_id, history_maxlen, lock_avg_frames=LOCK_AVERAGE_FRAMES):
        self.track_id = track_id
        self.reference_signature = None
        self.reference_bbox = None
        self.total_frames = 0
        self.diff_history = deque(maxlen=history_maxlen)
        self.bbox_area_history = deque(maxlen=history_maxlen)
        self.lock_avg_frames = lock_avg_frames
        self.box_buffer = deque(maxlen=lock_avg_frames)


def get_appearance_signature(crop):
    """Grayscale and resize whole bbox crop to 32x32."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (32, 32))


def averaged_box(box_buffer, w_frame, h_frame):
    """Average bounding boxes currently in buffer, clipped to frame bounds."""
    boxes = np.array(box_buffer)
    avg = boxes.mean(axis=0)
    x1, y1, x2, y2 = avg
    x1 = int(np.clip(x1, 0, w_frame))
    y1 = int(np.clip(y1, 0, h_frame))
    x2 = int(np.clip(x2, 0, w_frame))
    y2 = int(np.clip(y2, 0, h_frame))
    return x1, y1, x2, y2


# ── Detector & Tracker Backends (YOLO, RT-DETR, RF-DETR) ─────────────────
class BaseTracker:
    def track(self, frame):
        raise NotImplementedError


class UltralyticsTracker(BaseTracker):
    def __init__(self, model_path, model_type="yolo", conf=0.25, imgsz=1280, device="mps"):
        self.conf = conf
        self.imgsz = imgsz
        self.device = device
        if model_type == "rtdetr":
            from ultralytics import RTDETR
            print(f"Loading RT-DETR model '{model_path}' on device '{device}'...", flush=True)
            self.model = RTDETR(model_path)
        else:
            from ultralytics import YOLO
            print(f"Loading YOLO model '{model_path}' on device '{device}'...", flush=True)
            self.model = YOLO(model_path)

    def track(self, frame):
        results = self.model.track(
            frame,
            persist=True,
            conf=self.conf,
            imgsz=self.imgsz,
            classes=[0],  # Person class
            device=self.device,
            verbose=False,
        )
        if results and results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy()
            return boxes, track_ids
        return np.array([]), np.array([])


class RFDETRTracker(BaseTracker):
    """Roboflow RF-DETR detector integrated with ByteTrack tracker."""
    def __init__(self, model_name_or_path="rfdetr-medium", conf=0.25, device="mps"):
        self.conf = conf
        self.device = device
        try:
            import supervision as sv
            import rfdetr
        except ImportError:
            raise ImportError(
                "RF-DETR requires 'rfdetr' and 'supervision'. Run:\n"
                "pip install rfdetr supervision"
            )

        self.sv = sv
        # Lost track buffer gives tolerance to temporary occlusions
        # minimum_consecutive_frames=1 ensures immediate track assignment on frame 1
        try:
            self.tracker = sv.ByteTrack(
                track_activation_threshold=max(0.15, self.conf * 0.8),
                lost_track_buffer=40,
                minimum_consecutive_frames=1,
            )
        except TypeError:
            self.tracker = sv.ByteTrack(
                track_activation_threshold=max(0.15, self.conf * 0.8),
                lost_track_buffer=40,
            )

        name = model_name_or_path.lower()
        print(f"Loading Roboflow RF-DETR model '{model_name_or_path}'...", flush=True)
        if "nano" in name:
            self.model = rfdetr.RFDETRNano()
        elif "small" in name:
            self.model = rfdetr.RFDETRSmall()
        elif "large" in name:
            self.model = rfdetr.RFDETRLarge()
        elif "base" in name:
            self.model = rfdetr.RFDETRBase()
        else:
            self.model = rfdetr.RFDETRMedium()

        # Cache person class id mapping
        self.person_class_ids = [0]
        try:
            from rfdetr.assets.coco_classes import COCO_CLASSES
            self.person_class_ids = [
                i for i, c in enumerate(COCO_CLASSES) if c.lower() in ("person", "pedestrian")
            ]
        except Exception:
            self.person_class_ids = [0, 1]

    def track(self, frame):
        from PIL import Image
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        detections = self.model.predict(pil_img, threshold=self.conf)

        if len(detections) > 0:
            is_person = None
            if "class_name" in detections.data:
                cnames = [str(c).lower() for c in detections.data["class_name"]]
                is_person = np.array([c in ("person", "pedestrian") for c in cnames])
            elif detections.class_id is not None:
                is_person = np.isin(detections.class_id, self.person_class_ids)

            if is_person is not None and np.any(is_person):
                detections = detections[is_person]

        if len(detections) > 0:
            tracked = self.tracker.update_with_detections(detections)
            if tracked.tracker_id is not None and len(tracked.tracker_id) > 0:
                return tracked.xyxy, tracked.tracker_id
            # Fallback if tracker is buffering initial frame
            temp_ids = np.arange(1, len(detections) + 1)
            return detections.xyxy, temp_ids
        return np.array([]), np.array([])


def get_tracker(model_path, detector_type="auto", conf=0.25, imgsz=1280, device="mps"):
    m_lower = model_path.lower()
    d_lower = detector_type.lower()
    if d_lower == "rfdetr" or "rfdetr" in m_lower or "rf-detr" in m_lower:
        return RFDETRTracker(model_name_or_path=model_path, conf=conf, device=device)
    elif d_lower == "rtdetr" or "rtdetr" in m_lower or "rt-detr" in m_lower:
        return UltralyticsTracker(model_path=model_path, model_type="rtdetr", conf=conf, imgsz=imgsz, device=device)
    else:
        return UltralyticsTracker(model_path=model_path, model_type="yolo", conf=conf, imgsz=imgsz, device=device)


# ── Video Processing Pipeline ─────────────────────────────────────────────
def process_video(
    video_path,
    model_path,
    detector_type="auto",
    conf=0.25,
    imgsz=1280,
    min_bbox_size=32,
    relock_threshold=10.0,
    history_maxlen=500,
    device="mps",
    save_annotated=True,
    output_video_path=None,
    target_fps=1.0,
):
    if device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available. Falling back to CPU.", flush=True)
        device = "cpu"
    elif device == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        if torch.cuda.is_available():
            print("Notice: MPS not available. Using CUDA GPU.", flush=True)
            device = "cuda"
        else:
            print("Warning: MPS requested but not available. Falling back to CPU.", flush=True)
            device = "cpu"

    tracker_engine = get_tracker(
        model_path=model_path,
        detector_type=detector_type,
        conf=conf,
        imgsz=imgsz,
        device=device,
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS)
    if native_fps <= 0 or np.isnan(native_fps):
        native_fps = 20.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w_frame = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_frame = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Calculate frame stepping for target FPS (default 1.0 FPS)
    if target_fps is not None and target_fps > 0 and target_fps < native_fps:
        frame_step = max(1, int(round(native_fps / target_fps)))
        effective_fps = native_fps / frame_step
    else:
        frame_step = 1
        effective_fps = native_fps

    lock_buf_len = 1 if frame_step > 5 else LOCK_AVERAGE_FRAMES
    expected_processed = (total_frames + frame_step - 1) // frame_step

    writer = None
    if save_annotated and output_video_path:
        os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_video_path, fourcc, effective_fps, (w_frame, h_frame))
        print(f"Saving annotated video to: {output_video_path} (@ {effective_fps:.1f} fps)", flush=True)

    tracks = {}
    raw_frame_idx = 0
    processed_count = 0
    start_time = time.time()
    last_log_time = start_time
    total_detections_count = 0

    fps_note = f" (sampling 1 frame per {frame_step} frames = ~{effective_fps:.1f} fps)" if frame_step > 1 else ""
    print(
        f"Processing: {os.path.basename(video_path)} ({total_frames} raw frames, {expected_processed} to process{fps_note}, {w_frame}x{h_frame})",
        flush=True,
    )

    while True:
        ret = cap.grab()
        if not ret:
            break

        if raw_frame_idx % frame_step != 0:
            raw_frame_idx += 1
            continue

        ret, frame = cap.retrieve()
        raw_frame_idx += 1
        if not ret or frame is None:
            continue

        processed_count += 1

        annotated_frame = frame.copy() if writer else None

        boxes, track_ids = tracker_engine.track(frame)
        total_detections_count += len(boxes)

        active_track_ids = []
        frame_status = {}

        if len(boxes) > 0 and len(track_ids) > 0:
            for box, tid in zip(boxes, track_ids):
                tid = int(tid)
                active_track_ids.append(tid)

                if tid not in tracks:
                    tracks[tid] = PersonTrack(track_id=tid, history_maxlen=history_maxlen, lock_avg_frames=lock_buf_len)
                track = tracks[tid]

                x1, y1, x2, y2 = map(int, box)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w_frame, x2), min(h_frame, y2)

                # Min bbox filtering
                if (x2 - x1) < min_bbox_size or (y2 - y1) < (min_bbox_size // 2):
                    frame_status[tid] = {"status": "filtered", "box": (x1, y1, x2, y2)}
                    continue

                track.box_buffer.append((x1, y1, x2, y2))

                # Reference locking
                if track.reference_bbox is None:
                    if len(track.box_buffer) < track.lock_avg_frames:
                        frame_status[tid] = {
                            "status": f"buffering ({len(track.box_buffer)}/{track.lock_avg_frames})",
                            "box": (x1, y1, x2, y2),
                        }
                        continue
                    lx1, ly1, lx2, ly2 = averaged_box(track.box_buffer, w_frame, h_frame)
                    crop = frame[ly1:ly2, lx1:lx2]
                    if crop.size > 0:
                        track.reference_signature = get_appearance_signature(crop)
                        track.reference_bbox = (lx1, ly1, lx2, ly2)
                        frame_status[tid] = {
                            "status": "LOCKED",
                            "box": (x1, y1, x2, y2),
                            "ref_box": (lx1, ly1, lx2, ly2),
                            "diff": None,
                        }
                    continue

                # Diff computation
                rx1, ry1, rx2, ry2 = track.reference_bbox
                crop = frame[ry1:ry2, rx1:rx2]

                if crop.size == 0:
                    track.reference_bbox = None
                    frame_status[tid] = {"status": "lost_ref", "box": (x1, y1, x2, y2)}
                    continue

                current_sig = get_appearance_signature(crop)
                diff = float(np.mean(cv2.absdiff(current_sig, track.reference_signature)))
                track.diff_history.append(diff)
                current_area = (x2 - x1) * (y2 - y1)
                track.bbox_area_history.append(current_area)
                track.total_frames += 1

                relocked = False
                if diff > relock_threshold:
                    lx1, ly1, lx2, ly2 = averaged_box(track.box_buffer, w_frame, h_frame)
                    new_crop = frame[ly1:ly2, lx1:lx2]
                    if new_crop.size > 0:
                        track.reference_signature = get_appearance_signature(new_crop)
                        track.reference_bbox = (lx1, ly1, lx2, ly2)
                        relocked = True

                frame_status[tid] = {
                    "status": "RELOCKED" if relocked else "TRACKING",
                    "box": (x1, y1, x2, y2),
                    "ref_box": track.reference_bbox,
                    "diff": diff,
                    "relocked": relocked,
                }

        # Render annotations if video writer active
        if writer is not None:
            header_h = 44
            overlay = annotated_frame.copy()
            cv2.rectangle(overlay, (0, 0), (w_frame, header_h), (25, 25, 25), -1)
            cv2.addWeighted(overlay, 0.75, annotated_frame, 0.25, 0, annotated_frame)

            info_text = (
                f"Sec: {processed_count}/{expected_processed} ({effective_fps:.1f} fps) | Device: {device} | Model: {os.path.basename(model_path)} | "
                f"Active Tracks: {len(active_track_ids)}"
            )
            cv2.putText(
                annotated_frame,
                info_text,
                (16, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            for tid, st in frame_status.items():
                box = st.get("box")
                if not box:
                    continue
                x1, y1, x2, y2 = box
                diff_val = st.get("diff")
                status_lbl = st.get("status", "")
                relocked = st.get("relocked", False)

                if relocked:
                    box_color = (0, 165, 255)  # Orange
                elif status_lbl == "TRACKING":
                    box_color = (0, 230, 0)  # Green
                elif status_lbl == "LOCKED":
                    box_color = (255, 200, 0)  # Cyan
                else:
                    box_color = (180, 180, 180)  # Gray

                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 2)

                ref_box = st.get("ref_box")
                if ref_box:
                    rx1, ry1, rx2, ry2 = ref_box
                    cv2.rectangle(annotated_frame, (rx1, ry1), (rx2, ry2), (0, 255, 255), 1)

                diff_str = f"Diff: {diff_val:.1f}" if diff_val is not None else "Init"
                label_text = f"ID {tid} | {diff_str} | {status_lbl}"
                (txt_w, txt_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                cv2.rectangle(
                    annotated_frame,
                    (x1, max(0, y1 - txt_h - 10)),
                    (x1 + txt_w + 8, y1),
                    box_color,
                    -1,
                )
                cv2.putText(
                    annotated_frame,
                    label_text,
                    (x1 + 4, max(txt_h + 2, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 0, 0),
                    2,
                    cv2.LINE_AA,
                )

            writer.write(annotated_frame)

        now = time.time()
        if (now - last_log_time >= 15.0) or (processed_count == expected_processed):
            elapsed = now - start_time
            fps_calc = processed_count / elapsed if elapsed > 0 else 0
            eta_sec = (expected_processed - processed_count) / fps_calc if fps_calc > 0 else 0
            print(
                f"[{time.strftime('%H:%M:%S')}] Processed {processed_count}/{expected_processed} seconds "
                f"({processed_count*100.0/expected_processed:.1f}%) | Active Tracks: {len(active_track_ids)} | "
                f"Speed: {fps_calc:.1f} fps | ETA: {int(eta_sec//60)}m {int(eta_sec%60)}s",
                flush=True,
            )
            last_log_time = now

    cap.release()
    if writer:
        writer.release()
        print(f"Finished writing video to: {output_video_path}", flush=True)

    print(f"Total detections across all frames: {total_detections_count}, Total unique tracks: {len(tracks)}", flush=True)
    return tracks, effective_fps


def old_logic_eval(diff_history, windows, thresh_pairs):
    n = len(diff_history)
    arr = np.array(diff_history)
    results = {}
    for window in windows:
        if window > n:
            for tmin, tmax in thresh_pairs:
                results[(window, (tmin, tmax))] = {"passes": False, "coverage_pct": 0.0}
            continue
        csum = np.cumsum(np.insert(arr, 0, 0))
        rolling_means = (csum[window:] - csum[:-window]) / window
        for tmin, tmax in thresh_pairs:
            passing_starts = np.where((rolling_means >= tmin) & (rolling_means <= tmax))[0]
            passes = len(passing_starts) > 0
            covered = np.zeros(n, dtype=bool)
            for s in passing_starts:
                covered[s : s + window] = True
            coverage_pct = 100.0 * covered.sum() / n
            results[(window, (tmin, tmax))] = {"passes": passes, "coverage_pct": coverage_pct}
    return results


def longest_streak_under_threshold(diff_history, per_frame_threshold):
    max_streak = 0
    current_streak = 0
    current_start = None
    best_range = (0, 0)
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


def new_logic_eval(diff_history, per_frame_thresholds, required_streaks):
    results = {}
    streak_info = {}
    for thresh in per_frame_thresholds:
        streak, rng = longest_streak_under_threshold(diff_history, thresh)
        streak_info[thresh] = {"longest_streak": streak, "range": rng}
        for req in required_streaks:
            results[(thresh, req)] = {"passes": streak >= req, "longest_streak": streak}
    return results, streak_info


def plot_comparison(track_id, diff_history, new_streak_info, new_best_thresh, output_dir, prefix="", time_unit="Frame"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(diff_history))
    ax.plot(x, diff_history, color="crimson", linewidth=1.2, label="Diff per frame")
    ax.fill_between(x, 0, diff_history, color="crimson", alpha=0.15)

    info = new_streak_info[new_best_thresh]
    start, end = info["range"]
    unit_str = "sec" if "sec" in time_unit.lower() else "frames"
    if info["longest_streak"] > 0:
        ax.axvspan(
            start,
            end,
            color="seagreen",
            alpha=0.25,
            label=f"NEW: longest streak <= {new_best_thresh} ({info['longest_streak']} {unit_str})",
        )

    ax.axhline(y=10, color="gray", linestyle="--", linewidth=0.8, label="Relock threshold (10)")
    ax.set_title(f"{prefix} Track {track_id}: diff signal, OLD vs NEW decision logic")
    ax.set_xlabel(time_unit)
    ax.set_ylabel("Raw diff")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()

    out_path = os.path.join(output_dir, f"track_{track_id}_comparison.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def get_default_device():
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_default_video_dir():
    colab_video_dir = "/content/drive/MyDrive/Nakul_Guardex/sleep_test/test_video"
    if os.path.exists(colab_video_dir) or os.path.exists("/content"):
        return colab_video_dir
    return "test_video"


def get_default_output_dir():
    colab_drive_dir = "/content/drive/MyDrive/Nakul_Guardex/sleep_test/experiment_1_fps"
    if os.path.exists("/content/drive/MyDrive") or os.path.exists("/content"):
        return colab_drive_dir
    return "test_video/anotated_video"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--video",
        default=get_default_video_dir(),
        help="Path to video file or directory containing test videos (default: /content/drive/MyDrive/Nakul_Guardex/sleep_test/test_video on Colab, else test_video)",
    )
    parser.add_argument(
        "--model",
        default="yolo11m.pt",
        help="Model path or name: e.g. yolo11m.pt, rtdetr-l.pt, rfdetr-medium, rfdetr-large (default: yolo11m.pt)",
    )
    parser.add_argument(
        "--detector",
        choices=["auto", "yolo", "rtdetr", "rfdetr"],
        default="auto",
        help="Detector architecture: auto, yolo, rtdetr, rfdetr (default: auto)",
    )
    parser.add_argument(
        "--device",
        default=get_default_device(),
        help="Inference device: cuda, mps, cpu (default: auto-detected)",
    )
    parser.add_argument(
        "--output-dir",
        default=get_default_output_dir(),
        help="Output directory for annotated video folders (default: /content/drive/MyDrive/Nakul_Guardex/sleep_test/experiment_1_fps on Colab, else test_video/anotated_video)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=1.0,
        help="Target processing frame rate in FPS (default: 1.0 for 1 frame per second; set to 0 for native video FPS)",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence (default: 0.25)")
    parser.add_argument("--imgsz", type=int, default=1280, help="Inference image size for YOLO/RT-DETR (default: 1280)")
    parser.add_argument("--min-bbox-size", type=int, default=32, help="Minimum bbox size (default: 32)")
    parser.add_argument("--relock-threshold", type=float, default=10.0, help="Relock diff threshold (default: 10.0)")
    parser.add_argument("--no-cap", action="store_true", help="Disable maxlen=500 track history cap")
    parser.add_argument("--no-video-save", action="store_true", help="Skip saving annotated video files")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    history_maxlen = None if args.no_cap else 500

    # Determine video list
    video_paths = []
    if os.path.isdir(args.video):
        for ext in ("*.MOV", "*.mov", "*.mp4", "*.m4v", "*.avi"):
            video_paths.extend(glob.glob(os.path.join(args.video, ext)))
        video_paths.sort()
    elif os.path.isfile(args.video):
        video_paths = [args.video]
    else:
        matched = sorted(glob.glob(args.video))
        if matched:
            video_paths = matched
        else:
            raise FileNotFoundError(f"No video files found matching '{args.video}'")

    if not video_paths:
        print(f"No videos found in '{args.video}'.", flush=True)
        return

    print(f"Found {len(video_paths)} video(s) to process:", flush=True)
    for vp in video_paths:
        print(f"  - {vp}", flush=True)

    all_video_summaries = {}

    for idx, vp in enumerate(video_paths, 1):
        v_stem = Path(vp).stem
        video_folder = os.path.join(args.output_dir, v_stem)
        os.makedirs(video_folder, exist_ok=True)

        print(
            f"\n{'#'*90}\nProcessing Video {idx}/{len(video_paths)}: {os.path.basename(vp)}\nFolder: {video_folder}\n{'#'*90}",
            flush=True,
        )

        annotated_video_out = None
        if not args.no_video_save:
            annotated_video_out = os.path.join(video_folder, f"{v_stem}_annotated.mp4")

        tracks, effective_fps = process_video(
            video_path=vp,
            model_path=args.model,
            detector_type=args.detector,
            conf=args.conf,
            imgsz=args.imgsz,
            min_bbox_size=args.min_bbox_size,
            relock_threshold=args.relock_threshold,
            history_maxlen=history_maxlen,
            device=args.device,
            save_annotated=(not args.no_video_save),
            output_video_path=annotated_video_out,
            target_fps=args.fps,
        )

        if not tracks:
            print(f"No person tracks found in {vp}.", flush=True)
            continue

        summary_rows = []
        charts_created = 0
        sweep_windows, sweep_streaks = get_decision_params(effective_fps)
        time_unit = "Time (seconds)" if effective_fps <= 2.0 else "Frame"

        for track_id, track in tracks.items():
            diff_history = list(track.diff_history)
            if len(diff_history) < 5:  # lowered minimum to 5 samples
                continue

            json_path = os.path.join(video_folder, f"track_{track_id}_diff_history.json")
            with open(json_path, "w") as f:
                json.dump(diff_history, f)

            old_results = old_logic_eval(diff_history, sweep_windows, OLD_MEAN_THRESHOLDS)
            best_old = None
            for (window, thresh_pair), r in sorted(old_results.items(), key=lambda kv: -kv[1]["coverage_pct"]):
                if r["passes"] and best_old is None:
                    best_old = (window, thresh_pair)

            new_results, streak_info = new_logic_eval(
                diff_history, NEW_PER_FRAME_THRESHOLDS, sweep_streaks
            )
            best_new_thresh = max(NEW_PER_FRAME_THRESHOLDS, key=lambda t: streak_info[t]["longest_streak"])

            chart_path = plot_comparison(
                track_id, diff_history, streak_info, best_new_thresh, video_folder, prefix=v_stem, time_unit=time_unit
            )
            charts_created += 1

            mean_diff = round(float(np.mean(diff_history)), 2)
            spikes_8 = sum(1 for d in diff_history if d > 8.0)
            spikes_10 = sum(1 for d in diff_history if d > 10.0)

            # Old logic passes
            old_pass_30 = old_results.get((30, (0, 8)), {}).get("passes", False)
            old_pass_60 = old_results.get((60, (0, 8)), {}).get("passes", False)
            old_pass_100 = old_results.get((100, (0, 8)), {}).get("passes", False)
            old_pass_200 = old_results.get((200, (0, 8)), {}).get("passes", False)

            # New logic streaks
            streak_5 = streak_info.get(5, {}).get("longest_streak", 0)
            streak_8 = streak_info.get(8, {}).get("longest_streak", 0)
            streak_10 = streak_info.get(10, {}).get("longest_streak", 0)

            new_pass_30 = streak_8 >= 30
            new_pass_60 = streak_8 >= 60
            new_pass_100 = streak_8 >= 100
            new_pass_200 = streak_8 >= 200

            # Direct comparison verdict
            if new_pass_100:
                verdict = "BOTH_PASS (Sustained Stillness >=100s)"
            elif old_pass_100 and not new_pass_100:
                verdict = f"OLD_PASS_NEW_REJECT (Active Worker: {spikes_8} spikes > 8, max streak {streak_8}s)"
            elif old_pass_30 and not new_pass_30:
                verdict = f"OLD_PASS_NEW_REJECT (Active Worker: {spikes_8} spikes > 8, max streak {streak_8}s)"
            elif new_pass_30:
                verdict = "BOTH_PASS (Short Stillness >=30s)"
            else:
                verdict = "BOTH_REJECT (Continuous Movement)"

            summary_rows.append(
                {
                    "video": os.path.basename(vp),
                    "track_id": track_id,
                    "total_seconds": len(diff_history),
                    "mean_diff": mean_diff,
                    "motion_spikes_gt_8": spikes_8,
                    "old_logic_pass_W30": "PASS" if old_pass_30 else "FAIL",
                    "old_logic_pass_W60": "PASS" if old_pass_60 else "FAIL",
                    "old_logic_pass_W100": "PASS" if old_pass_100 else "FAIL",
                    "old_best_pass": best_old,
                    "new_streak_thresh8_sec": streak_8,
                    "new_streak_thresh5_sec": streak_5,
                    "new_logic_pass_streak30": "PASS" if new_pass_30 else "FAIL",
                    "new_logic_pass_streak60": "PASS" if new_pass_60 else "FAIL",
                    "new_logic_pass_streak100": "PASS" if new_pass_100 else "FAIL",
                    "new_logic_pass_streak200": "PASS" if new_pass_200 else "FAIL",
                    "old_vs_new_verdict": verdict,
                    "chart": os.path.basename(chart_path),
                }
            )

        print(f"Generated {charts_created} comparison chart(s) for {os.path.basename(vp)}.", flush=True)
        all_video_summaries[v_stem] = summary_rows

        csv_fields = [
            "video",
            "track_id",
            "total_seconds",
            "mean_diff",
            "motion_spikes_gt_8",
            "old_logic_pass_W30",
            "old_logic_pass_W60",
            "old_logic_pass_W100",
            "old_best_pass",
            "new_streak_thresh8_sec",
            "new_streak_thresh5_sec",
            "new_logic_pass_streak30",
            "new_logic_pass_streak60",
            "new_logic_pass_streak100",
            "new_logic_pass_streak200",
            "old_vs_new_verdict",
            "chart",
        ]

        # Save JSON summary
        summary_path = os.path.join(video_folder, "summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary_rows, f, indent=2)

        # Save CSV summary
        csv_path = os.path.join(video_folder, "summary.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"Saved summary.csv and summary.json to: {video_folder}", flush=True)

    # Master summary
    master_summary_path = os.path.join(args.output_dir, "all_videos_summary.json")
    with open(master_summary_path, "w") as f:
        json.dump(all_video_summaries, f, indent=2)

    master_csv_path = os.path.join(args.output_dir, "all_videos_summary.csv")
    with open(master_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for v_rows in all_video_summaries.values():
            writer.writerows(v_rows)

    print(f"\n\n{'='*90}\nALL VIDEOS COMPLETED\n{'='*90}", flush=True)
    print(f"Results, charts, and CSV/JSON summaries saved to: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()