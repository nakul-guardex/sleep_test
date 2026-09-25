"""
Streak-based false-positive analysis for sleep detection tracks.

WHAT THIS DOES
---------------
Your dashboard's existing filter is MEAN-based: "does any window of N frames
average below threshold X?" A mean can absorb occasional real-motion spikes
as long as enough quiet frames surround them.

This script instead computes STREAK-based logic: "what's the longest run of
CONSECUTIVE frames that never exceeded the per-frame threshold?" A single
frame over threshold resets the streak to zero, no matter how quiet the
surrounding frames are. This is a fundamentally different aggregation over
the same diff_history data you already have -- no new capture, no code
changes to sleep_new.py needed to test it.

HOW TO RUN
----------
    python3 streak_analysis.py path/to/global_person_metrics.csv

Expects a CSV with at least these columns (matching your existing pipeline):
    track_id, stream_name, total_frames, diff_history

diff_history should be a JSON array string, e.g. "[1.2, 3.4, 15.6, ...]"
(this matches what sleep_new.py already writes).

If your CSV uses different column names, adjust COLUMN NAMES below.
"""

import csv
import json
import sys
from collections import defaultdict

# ── COLUMN NAMES (adjust if your CSV differs) ────────────────────────────
COL_TRACK_ID = "track_id"
COL_STREAM = "stream_name"
COL_TOTAL_FRAMES = "total_frames"
COL_DIFF_HISTORY = "diff_history"

# ── ANALYSIS PARAMETERS ───────────────────────────────────────────────────
MIN_TOTAL_FRAMES = 100                      # only analyze tracks >= this length
PER_FRAME_THRESHOLDS = [5, 8, 10]           # per-frame "did this frame move" cutoffs to test
REQUIRED_STREAK_LENGTHS = [100, 150, 200, 250, 300]  # candidate streak requirements to sweep

# Known labels from your manual visual inspection (sleep_detection_findings.md).
# Key = (track_id, stream_name). Extend this as you label more tracks.
KNOWN_LABELS = {
    ("13", "cam_25"): "FP - Awake but Still",
    ("2", "cam_03"): "FP - Awake but Still",
    ("21", "cam_64"): "FP - Awake but Still",
    ("23", "cam_64"): "FP - Awake but Still",
    ("14", "cam_38"): "FP - Awake but Still",
    ("22", "cam_64"): "FP - Awake but Still",
    ("3", "cam_03"): "FP - Awake but Still",
    ("5", "cam_03"): "FP - Awake but Still",
    ("6", "cam_24"): "FP - Awake but Still",
    ("7", "cam_24"): "FP - Awake but Still",
    ("8", "cam_24"): "FP - Awake but Still",
    ("132", "cam_25"): "FP - Awake but Still",
    ("151", "cam_38"): "FP - Awake but Still",
    ("146", "cam_38"): "FP - Awake but Still",
    ("200", "cam_25"): "FP - Awake but Still",
}


def longest_streak_under_threshold(diff_history, per_frame_threshold):
    """
    Longest run of consecutive frames where diff <= per_frame_threshold.
    A single frame over threshold resets the streak to zero -- this is the
    key difference from a rolling mean, which can absorb occasional spikes.
    """
    max_streak = 0
    current_streak = 0
    for d in diff_history:
        if d <= per_frame_threshold:
            current_streak += 1
            if current_streak > max_streak:
                max_streak = current_streak
        else:
            current_streak = 0
    return max_streak


def spike_count(diff_history, per_frame_threshold):
    return sum(1 for d in diff_history if d > per_frame_threshold)


def load_tracks(csv_path):
    tracks = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing_cols = [c for c in (COL_TRACK_ID, COL_STREAM, COL_TOTAL_FRAMES, COL_DIFF_HISTORY)
                         if c not in reader.fieldnames]
        if missing_cols:
            print(f"ERROR: CSV is missing expected columns: {missing_cols}")
            print(f"Columns found: {reader.fieldnames}")
            print("Edit the COLUMN NAMES section at the top of this script to match your CSV.")
            sys.exit(1)

        for row in reader:
            try:
                total_frames = int(float(row[COL_TOTAL_FRAMES]))
            except (ValueError, TypeError):
                continue
            if total_frames < MIN_TOTAL_FRAMES:
                continue

            raw_history = row.get(COL_DIFF_HISTORY, "")
            if not raw_history:
                continue
            try:
                diff_history = json.loads(raw_history)
            except (json.JSONDecodeError, TypeError):
                # sometimes these get saved with single quotes instead of JSON-valid double quotes
                try:
                    diff_history = json.loads(raw_history.replace("'", '"'))
                except Exception:
                    print(f"WARNING: could not parse diff_history for track {row.get(COL_TRACK_ID)}, skipping")
                    continue

            if not diff_history:
                continue

            tracks.append({
                "track_id": str(row[COL_TRACK_ID]),
                "stream_name": row[COL_STREAM],
                "total_frames": total_frames,
                "diff_history": diff_history,
            })
    return tracks


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 streak_analysis.py path/to/global_person_metrics.csv")
        sys.exit(1)

    csv_path = sys.argv[1]
    tracks = load_tracks(csv_path)
    if not tracks:
        print("No tracks found matching the criteria. Check the CSV path/columns.")
        sys.exit(1)

    print(f"Loaded {len(tracks)} tracks with total_frames >= {MIN_TOTAL_FRAMES}\n")

    # ── Per-track table, one row per (track, threshold) ──────────────────
    results = []
    for t in tracks:
        key = (t["track_id"], t["stream_name"])
        label = KNOWN_LABELS.get(key, "unlabeled")
        for thresh in PER_FRAME_THRESHOLDS:
            streak = longest_streak_under_threshold(t["diff_history"], thresh)
            spikes = spike_count(t["diff_history"], thresh)
            results.append({
                "track_id": t["track_id"],
                "stream_name": t["stream_name"],
                "total_frames": t["total_frames"],
                "per_frame_threshold": thresh,
                "longest_streak": streak,
                "spike_count": spikes,
                "label": label,
            })

    # ── Print full per-track/per-threshold table ──────────────────────────
    print("=" * 110)
    print("PER-TRACK LONGEST STREAK (consecutive frames <= threshold, reset on any spike)")
    print("=" * 110)
    header = f"{'track_id':>8} {'stream':>10} {'frames':>7} {'thresh':>7} {'streak':>7} {'spikes':>7}  label"
    print(header)
    print("-" * len(header))
    for r in sorted(results, key=lambda r: (r["per_frame_threshold"], -r["longest_streak"])):
        print(f"{r['track_id']:>8} {r['stream_name']:>10} {r['total_frames']:>7} "
              f"{r['per_frame_threshold']:>7} {r['longest_streak']:>7} {r['spike_count']:>7}  {r['label']}")

    # ── Sweep: how many tracks pass at each (threshold, required streak) combo ─
    print("\n" + "=" * 110)
    print("SWEEP: how many tracks pass at each (per-frame threshold, required streak length)?")
    print("Broken down by known label, so you can see whether streak logic actually")
    print("separates labeled false positives from everything else.")
    print("=" * 110)

    for thresh in PER_FRAME_THRESHOLDS:
        print(f"\n--- per-frame threshold = {thresh} ---")
        for req_streak in REQUIRED_STREAK_LENGTHS:
            passing = [r for r in results
                       if r["per_frame_threshold"] == thresh and r["longest_streak"] >= req_streak]
            labeled_fp_passing = [r for r in passing if r["label"] != "unlabeled"]
            unlabeled_passing = [r for r in passing if r["label"] == "unlabeled"]
            print(f"  required_streak={req_streak:>4}: "
                  f"{len(passing)} total pass "
                  f"({len(labeled_fp_passing)} are KNOWN false positives, "
                  f"{len(unlabeled_passing)} unlabeled)")
            if labeled_fp_passing:
                names = ", ".join(f"{r['track_id']}({r['stream_name']})" for r in labeled_fp_passing)
                print(f"      known FPs that STILL PASS (streak logic did NOT catch these): {names}")

    # ── Direct check: does streak logic separate id6/id8 from the other 13? ──
    print("\n" + "=" * 110)
    print("DIRECT CHECK: do id6/id8 (cam_24) survive at 200+/300+ while the other")
    print("13 known false positives get correctly excluded?")
    print("=" * 110)
    check_ids = {
        ("6", "cam_24"): "expected to PASS (visually confirmed genuinely still)",
        ("8", "cam_24"): "expected to PASS (visually confirmed genuinely still)",
    }
    other_13 = [k for k in KNOWN_LABELS if k not in check_ids]

    for thresh in PER_FRAME_THRESHOLDS:
        print(f"\n--- per-frame threshold = {thresh} ---")
        for req_streak in [200, 300]:
            print(f"  required_streak={req_streak}:")
            for (tid, stream), note in check_ids.items():
                match = next((r for r in results
                              if r["track_id"] == tid and r["stream_name"] == stream
                              and r["per_frame_threshold"] == thresh), None)
                if match:
                    passed = match["longest_streak"] >= req_streak
                    print(f"    id{tid} ({stream}): streak={match['longest_streak']}, "
                          f"{'PASSES' if passed else 'FAILS'} -- {note}")
            still_passing_fps = [r for r in results
                                  if r["per_frame_threshold"] == thresh
                                  and r["longest_streak"] >= req_streak
                                  and (r["track_id"], r["stream_name"]) in other_13]
            if still_passing_fps:
                names = ", ".join(f"{r['track_id']}({r['stream_name']})" for r in still_passing_fps)
                print(f"    WARNING: these known FPs still pass too: {names}")
            else:
                print(f"    -> None of the other 13 known false positives pass at this setting.")


if __name__ == "__main__":
    main()