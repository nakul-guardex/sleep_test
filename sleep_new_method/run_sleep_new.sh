#!/bin/bash
LOCKFILE="/home/ashutosh_sharma/sleep-detection/sleep_new_method/sleep_new.lock"
exec 200>"$LOCKFILE"
flock -n 200 || { echo "Already running"; exit 1; }

cd /home/ashutosh_sharma/sleep-detection/sleep_new_method

mkdir -p logs
LOGFILE="logs/sleep_new_$(date +%Y-%m-%d).log"

echo "Starting new sleep detection at $(date)" >> "$LOGFILE"
export OPENCV_FFMPEG_CAPTURE_OPTIONS="hwaccel;nvdec"
exec /home/ashutosh_sharma/trt_venv/bin/python3 sleep_new.py >> "$LOGFILE" 2>&1
