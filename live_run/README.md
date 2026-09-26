# Live Run — Sleep Detection Benchmark
**Date**: 26 September 2026 | **Site**: Mohit | **Cameras**: cam_03, cam_24, cam_29, cam_38, cam_59

---

## What is this folder?

This folder contains the complete output of a **5-hour live CCTV benchmark** run to evaluate and improve sleep detection logic on Mohit cameras.

---

## Files

| File / Folder | Description |
|---------------|-------------|
| `EXECUTIVE_SUMMARY.md` | **Start here.** Full benchmark report for management — includes false positive analysis, photos, metric comparison, and recommendations |
| `METRIC_CHOICE_ANALYSIS.md` | Quantitative comparison of all 4 pixel-difference metrics (MAD, SSIM, MOG2, pHash) with false alarm counts at every window/streak setting |
| `METRIC_MATH_EXPLAINER.md` | Mathematical explanation of how each metric works, why each succeeds or fails on live CCTV |
| `Sleep_Detection_Live_Benchmark.ipynb` | Colab notebook used to run the benchmark — includes all analysis cells and outputs |
| `live_sleep_benchmark.py` | Core detection script — multi-threaded, 5 cameras in parallel, all 4 metrics computed per frame |
| `charts/` | Proof charts: old vs new logic signal plots, verdict summary table, sleeping signal reference |
| `false_positive_snapshots/` | 6 CCTV photos saved by OLD logic when it incorrectly flagged workers as sleeping |

---

## Key Results

| Logic | Config | False Alarms (out of 538 tracks) |
|-------|--------|----------------------------------|
| **Old Logic** | Rolling mean W=250s, diff band 0–5 | **6** |
| **New Logic** | Consecutive streak ≥ 250s, diff < 5 | **0** |

**Best metric**: MOG2 — reaches 0 false alarms at W=200s (old logic) and streak=150s (new logic).

---

## False Positive Summary

All 6 false alarms were from **cam_03 (Security Room Inside)**. Workers were using phones, writing, or talking — not sleeping.

| Track | Time | Avg MAD | Motion Spikes | Max Still Streak |
|-------|------|---------|---------------|-----------------|
| 396 | 12:46 | 4.85 | 97 | 40s |
| 1225 | 13:26 | 5.38 | 287 | 46s |
| 1963 | 13:59 | 4.19 | 210 | 120s |
| 3739 | 15:08 | 4.14 | 120 | 142s |
| 4628 | 15:55 | 4.49 | 45 | 132s |
| 4934 | 16:05 | 5.34 | 230 | 118s |

📸 See `false_positive_snapshots/` for the CCTV photos taken at each trigger.

---

## Recommendation

> Use **New Logic (consecutive streak ≥ 120s)** with **MOG2 as the primary signal**.
> This eliminates all false positives while enabling faster detection than the 250s setting.
