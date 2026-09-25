import os
import cv2
import time
import logging
import threading
import csv
import psutil
import subprocess
from dataclasses import dataclass, field
from collections import deque
from ultralytics import YOLO
import numpy as np

global_csv_lock = threading.Lock()

@dataclass
class Config:
    video_path: str = ""
    output_dir: str = ""
    stream_name: str = ""
    model_path: str = "yolo11l.engine"
    detection_conf: float = 0.25
    infer_img_size: int = 1280
    use_fp16: bool = True
    min_bbox_size: int = 32
    max_track_frames: int = 500

def get_system_stats():
    cpu_util = psutil.cpu_percent(interval=None)
    try:
        gpu_out = subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"]).decode()
        gpu_util = float(gpu_out.split('\n')[0])
    except Exception:
        gpu_util = 0.0
    return cpu_util, gpu_util

@dataclass
class PersonTrack:
    track_id: int
    reference_signature: np.ndarray = None
    reference_bbox: tuple = None
    total_frames: int = 0
    diff_history: deque = field(default_factory=lambda: deque(maxlen=500))
    bbox_area_history: deque = field(default_factory=lambda: deque(maxlen=500))
    logged: bool = False
    saved_crop: np.ndarray = None
    saved_full: np.ndarray = None

class CameraWorker(threading.Thread):
    def __init__(self, config: Config):
        super().__init__()
        self.daemon = True
        self.config = config
        self.tracks = {}
        
        session_date = time.strftime("%Y-%m-%d")
        log_dir = os.path.join(self.config.output_dir, session_date, "logs")
        os.makedirs(log_dir, exist_ok=True)
        self.log = logging.getLogger(self.config.stream_name)
        self.log.setLevel(logging.INFO)
        fh = logging.FileHandler(os.path.join(log_dir, f"{self.config.stream_name}_new.log"))
        fh.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
        self.log.addHandler(fh)

        os.makedirs(self.config.output_dir, exist_ok=True)

    def get_appearance_signature(self, crop):
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        return cv2.resize(gray, (32, 32))

    def log_and_save_person(self, track):
        now_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        now_date_str = time.strftime("%Y-%m-%d", time.localtime())
        
        out_dir = os.path.join(self.config.output_dir, now_date_str, "tracked_persons")
        os.makedirs(out_dir, exist_ok=True)
        
        crop_path = os.path.join(out_dir, f"{self.config.stream_name}_CROP_id{track.track_id}_{now_str}.jpg")
        full_path = os.path.join(out_dir, f"{self.config.stream_name}_FULL_id{track.track_id}_{now_str}.jpg")
        
        if track.saved_crop is not None:
            cv2.imwrite(crop_path, track.saved_crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
        else:
            crop_path = "None"
            
        if track.saved_full is not None:
            cv2.imwrite(full_path, track.saved_full, [cv2.IMWRITE_JPEG_QUALITY, 90])
        else:
            full_path = "None"
            
        mean_diff = np.mean(track.diff_history) if len(track.diff_history) > 0 else 0.0
        std_diff = np.std(track.diff_history) if len(track.diff_history) > 0 else 0.0
        
        import json
        diff_history_json = json.dumps([float(x) for x in track.diff_history])
        bbox_area_json = json.dumps([float(x) for x in track.bbox_area_history])
        
        global_csv = "/home/ashutosh_sharma/sleep-detection/global_person_metrics.csv"
        with global_csv_lock:
            file_exists = os.path.exists(global_csv)
            with open(global_csv, 'a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["timestamp", "date", "stream_name", "track_id", "total_frames", "mean_diff", "std_diff", "crop_path", "full_path", "diff_history", "bbox_area_history"])
                writer.writerow([
                    time.strftime("%Y-%m-%d %H:%M:%S"),
                    now_date_str,
                    self.config.stream_name,
                    track.track_id,
                    track.total_frames,
                    f"{mean_diff:.2f}",
                    f"{std_diff:.2f}",
                    crop_path,
                    full_path,
                    diff_history_json,
                    bbox_area_json
                ])
        self.log.info(f"Logged track {track.track_id} (frames: {track.total_frames}, mean_diff: {mean_diff:.2f})")

    def run(self):
        import torch
        device = "cuda"
        model = YOLO(self.config.model_path)
        try:
            cap = cv2.VideoCapture(self.config.video_path, cv2.CAP_FFMPEG)
        except Exception as e:
            self.log.error(f"Failed to initialize VideoCapture: {e}")
            cap = None

        while True:
            try:
                if cap is None or not cap.isOpened():
                    cap = cv2.VideoCapture(self.config.video_path, cv2.CAP_FFMPEG)
                ret, frame = cap.read()
            except Exception as e:
                self.log.error(f"OpenCV Error: {e}")
                ret = False

            if not ret:
                if cap is not None:
                    cap.release()
                cap = None
                time.sleep(5)
                continue

            h_frame, w_frame = frame.shape[:2]

            results = model.track(
                frame,
                persist=True,
                conf=self.config.detection_conf,
                imgsz=self.config.infer_img_size,
                classes=[0],
                verbose=False,
                device=device,
                half=self.config.use_fp16
            )

            current_tids = set()
            if results and results[0].boxes and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy()

                for box, tid in zip(boxes, track_ids):
                    tid = int(tid)
                    current_tids.add(tid)

                    if tid not in self.tracks:
                        self.tracks[tid] = PersonTrack(track_id=tid)
                    
                    track = self.tracks[tid]
                    x1, y1, x2, y2 = map(int, box)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w_frame, x2), min(h_frame, y2)
                    
                    if (x2 - x1) < self.config.min_bbox_size or (y2 - y1) < (self.config.min_bbox_size // 2):
                        continue

                    # Reference locking logic
                    if track.reference_bbox is None:
                        crop = frame[y1:y2, x1:x2]
                        if crop.size > 0:
                            track.reference_signature = self.get_appearance_signature(crop)
                            track.reference_bbox = (x1, y1, x2, y2)
                        continue
                    
                    rx1, ry1, rx2, ry2 = track.reference_bbox
                    crop = frame[ry1:ry2, rx1:rx2]
                    
                    if crop.size == 0:
                        track.reference_bbox = None
                        continue
                        
                    current_sig = self.get_appearance_signature(crop)
                    diff = np.mean(cv2.absdiff(current_sig, track.reference_signature))
                    track.diff_history.append(diff)
                    current_area = (x2 - x1) * (y2 - y1)
                    track.bbox_area_history.append(current_area)
                    track.total_frames += 1

                    # Reset reference if moved significantly (optional, kept from original to measure stability from start position)
                    if diff > 10.0:
                        new_crop = frame[y1:y2, x1:x2]
                        if new_crop.size > 0:
                            track.reference_signature = self.get_appearance_signature(new_crop)
                            track.reference_bbox = (x1, y1, x2, y2)

                    # Periodically save snapshot in case track is lost
                    if track.total_frames in [10, 50, 250, 500]:
                        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                        cw, ch = x2 - x1, y2 - y1
                        half_size = int(max(cw, ch) * 0.7)
                        y_s, y_e = max(0, int(cy) - half_size), min(h_frame, int(cy) + half_size)
                        x_s, x_e = max(0, int(cx) - half_size), min(w_frame, int(cx) + half_size)
                        valid_crop = frame[y_s:y_e, x_s:x_e]
                        pad_t, pad_b = max(0, half_size - int(cy)), max(0, int(cy) + half_size - h_frame)
                        pad_l, pad_r = max(0, half_size - int(cx)), max(0, int(cx) + half_size - w_frame)
                        track.saved_crop = cv2.copyMakeBorder(valid_crop, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=[0,0,0])
                        
                        annotated = frame.copy()
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        cv2.putText(annotated, f"ID: {tid} Frames: {track.total_frames}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        track.saved_full = annotated

                    # Log at max frames
                    if not track.logged and track.total_frames >= self.config.max_track_frames:
                        self.log_and_save_person(track)
                        track.logged = True

            # Cleanup lost tracks
            for tid in list(self.tracks.keys()):
                if tid not in current_tids:
                    track = self.tracks[tid]
                    if track.total_frames >= 50 and not track.logged:
                        self.log_and_save_person(track)
                    del self.tracks[tid]

            time.sleep(0.5) # Slight throttle if needed, though YOLO acts as throttle

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("main")

    STREAMS = [
        {"name": "cam_03", "video_path": "rtsp://admin:mohit%4012345@103.210.29.90:554/Streaming/Channels/301", "output_dir": "/home/ashutosh_sharma/sleep-detection/cam_03"},
        {"name": "cam_09", "video_path": "rtsp://admin:mohit%4012345@103.210.29.90:554/Streaming/Channels/901", "output_dir": "/home/ashutosh_sharma/sleep-detection/cam_09"},
        {"name": "cam_11", "video_path": "rtsp://admin:mohit%4012345@103.210.29.90:554/Streaming/Channels/1101", "output_dir": "/home/ashutosh_sharma/sleep-detection/cam_11"},
        {"name": "cam_12", "video_path": "rtsp://admin:mohit%4012345@103.210.29.90:554/Streaming/Channels/1201", "output_dir": "/home/ashutosh_sharma/sleep-detection/cam_12"},
        {"name": "cam_13", "video_path": "rtsp://admin:mohit%4012345@103.210.29.90:554/Streaming/Channels/1301", "output_dir": "/home/ashutosh_sharma/sleep-detection/cam_13"},
        {"name": "cam_22", "video_path": "rtsp://admin:mohit%4012345@103.210.29.90:554/Streaming/Channels/2201", "output_dir": "/home/ashutosh_sharma/sleep-detection/cam_22"},
        {"name": "cam_24", "video_path": "rtsp://admin:mohit%4012345@103.210.29.90:554/Streaming/Channels/2401", "output_dir": "/home/ashutosh_sharma/sleep-detection/cam_24"},
        {"name": "cam_25", "video_path": "rtsp://admin:mohit%4012345@103.210.29.90:554/Streaming/Channels/2501", "output_dir": "/home/ashutosh_sharma/sleep-detection/cam_25"},
        {"name": "cam_29", "video_path": "rtsp://admin:mohit%4012345@103.210.29.90:554/Streaming/Channels/2901", "output_dir": "/home/ashutosh_sharma/sleep-detection/cam_29"},
        {"name": "cam_30", "video_path": "rtsp://admin:mohit%4012345@103.210.29.90:554/Streaming/Channels/3001", "output_dir": "/home/ashutosh_sharma/sleep-detection/cam_30"},
        {"name": "cam_31", "video_path": "rtsp://admin:mohit%4012345@103.210.29.90:554/Streaming/Channels/3101", "output_dir": "/home/ashutosh_sharma/sleep-detection/cam_31"},
        {"name": "cam_37", "video_path": "rtsp://admin:mohit%4012345@103.210.29.90:554/Streaming/Channels/3701", "output_dir": "/home/ashutosh_sharma/sleep-detection/cam_37"},
        {"name": "cam_38", "video_path": "rtsp://admin:mohit%4012345@103.210.29.90:554/Streaming/Channels/3801", "output_dir": "/home/ashutosh_sharma/sleep-detection/cam_38"},
        {"name": "cam_39", "video_path": "rtsp://admin:mohit%4012345@103.210.29.90:554/Streaming/Channels/3901", "output_dir": "/home/ashutosh_sharma/sleep-detection/cam_39"},
        {"name": "cam_41", "video_path": "rtsp://admin:mohit%4012345@103.210.29.90:554/Streaming/Channels/4101", "output_dir": "/home/ashutosh_sharma/sleep-detection/cam_41"},
        {"name": "cam_42", "video_path": "rtsp://admin:mohit%4012345@103.210.29.90:554/Streaming/Channels/4201", "output_dir": "/home/ashutosh_sharma/sleep-detection/cam_42"},
        {"name": "cam_50", "video_path": "rtsp://admin:mohit%4012345@103.210.29.90:554/Streaming/Channels/5001", "output_dir": "/home/ashutosh_sharma/sleep-detection/cam_50"},
        {"name": "cam_51", "video_path": "rtsp://admin:mohit%4012345@103.210.29.90:554/Streaming/Channels/5101", "output_dir": "/home/ashutosh_sharma/sleep-detection/cam_51"},
        {"name": "cam_58", "video_path": "rtsp://admin:mohit%4012345@103.210.29.90:554/Streaming/Channels/5801", "output_dir": "/home/ashutosh_sharma/sleep-detection/cam_58"},
        {"name": "cam_59", "video_path": "rtsp://admin:mohit%4012345@103.210.29.90:554/Streaming/Channels/5901", "output_dir": "/home/ashutosh_sharma/sleep-detection/cam_59"},
        {"name": "cam_61", "video_path": "rtsp://admin:mohit%4012345@103.210.29.90:554/Streaming/Channels/6101", "output_dir": "/home/ashutosh_sharma/sleep-detection/cam_61"},
        {"name": "cam_62", "video_path": "rtsp://admin:mohit%4012345@103.210.29.90:554/Streaming/Channels/6201", "output_dir": "/home/ashutosh_sharma/sleep-detection/cam_62"},
        {"name": "cam_64", "video_path": "rtsp://admin:mohit%4012345@103.210.29.90:554/Streaming/Channels/6401", "output_dir": "/home/ashutosh_sharma/sleep-detection/cam_64"},
    ]

    def upload_worker(streams):
        import subprocess
        from datetime import datetime, timezone, timedelta
        IST = timezone(timedelta(hours=5, minutes=30))
        
        while True:
            time.sleep(3600)  # Wait 1 hour between uploads
            now = datetime.now(IST)
            session_date = now.strftime("%Y-%m-%d")
            logger.info(f"Starting hourly rclone upload for {session_date}...")
            
            remote_base = f"gdrive:{session_date}/fast"
            
            # Global CSV & Dashboard
            global_csv = "/home/ashutosh_sharma/sleep-detection/global_person_metrics.csv"
            if os.path.exists(global_csv):
                subprocess.run(["rclone", "copyto", global_csv, f"{remote_base}/global_person_metrics.csv", "-v"], capture_output=True)
                
            dashboard = "/home/ashutosh_sharma/sleep-detection/dashboard.html"
            if os.path.exists(dashboard):
                subprocess.run(["rclone", "copyto", dashboard, f"{remote_base}/dashboard.html", "-v"], capture_output=True)
            
            for s in streams:
                name = s["name"]
                base_dir = s["output_dir"]
                images_local = f"{base_dir}/{session_date}/tracked_persons"
                images_remote = f"{remote_base}/{name}/tracked_persons"
                
                if os.path.exists(images_local):
                    subprocess.run(["rclone", "copy", images_local, images_remote, "-v"], capture_output=True)
            
            logger.info("Hourly rclone upload completed.")

    uploader = threading.Thread(target=upload_worker, args=(STREAMS,), daemon=True)
    uploader.start()

    workers = []
    for stream_info in STREAMS:
        config = Config(
            video_path=stream_info["video_path"],
            output_dir=stream_info["output_dir"],
            stream_name=stream_info["name"]
        )
        worker = CameraWorker(config)
        worker.start()
        time.sleep(2.0)
        workers.append(worker)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Stopping...")
