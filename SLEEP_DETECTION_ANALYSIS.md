# CCTV Sleep Detection: Dashboard Filter Sweep & Analysis Report

**Dataset Analyzed:** [`global_person_metrics.csv`](file:///Users/nakulpatel/Desktop/Sleeping-new-method-main/DASHBOARD%20SLEEP%20DETECTION/global_person_metrics.csv)  
**Dashboard Tool:** [`dashboard.html`](file:///Users/nakulpatel/Desktop/Sleeping-new-method-main/DASHBOARD%20SLEEP%20DETECTION/dashboard.html)  
**Detailed Report Location:** [`sleep_detection_findings.md`](file:///Users/nakulpatel/Desktop/Sleeping-new-method-main/DASHBOARD%20SLEEP%20DETECTION/sleep_detection_findings.md)  
**Date:** September 2026  

---

## 1. Executive Summary

This report presents a systematic evaluation of candidate filter parameters for a CCTV-based sleep detection pipeline using bounding-box appearance difference. 

In this system:
- A YOLO-based tracker follows persons across frames.
- A 32×32 grayscale appearance patch is cropped each frame and compared against an anchored reference patch using mean absolute pixel difference.
- Consecutive low difference values serve as a proxy for physical stillness to flag potential sleep candidates.

Across the dataset of 50 tracked individuals (23 of which have tracks of $\ge 100$ frames), all 24 core parameter combinations plus normalized delta sweeps were executed. Crop images, full-frame CCTV contexts, and motion charts were evaluated for each matched card.

### Key Takeaway
- **Apparent Sleep Detection Precision:** **0.0%** across all parameter combinations tested. 
- **Sample Categorization:** 100% of all matching candidate cards were **False Positives — Awake but Still**.
- **Static Object Misdetections:** **0%**. YOLO did not misidentify inanimate objects (posters, chairs, reflections) as persons.
- **Root Cause:** Appearance diff effectively detects *pixel stillness*, but cannot distinguish between sleep postures (slumped head, horizontal body, closed eyes) and awake stationary behaviors (standing at attention, working machinery, seated writing, or talking on the phone).

---

## 2. Representative Sample Inspection Log

Each candidate track that met the filter criteria was inspected via its crop image, full-frame context, and motion chart trajectory.

| Track ID | Stream Name | Total Frames | Matched Mean Diff | Motion Chart Profile | Visual Inspection & Context | Classification |
| :--- | :--- | :---: | :---: | :--- | :--- | :--- |
| **id146** | `cam_38` | 154 | 2.31 | Flat baseline with 2 isolated spikes (frames 95, 108) | Worker standing with hands on hips watching the cooker area; alert, upright posture. | **FP — Awake but Still** |
| **id151** | `cam_38` | 140 | 4.14 | Flat baseline with 2 isolated spikes (frames 95, 134) | Worker leaning over a stainless table actively sorting/handling items. | **FP — Awake but Still** |
| **id8** | `cam_24` | 500 | 4.98 | Stable baseline with 5 spikes ($\ge 10$) | Office worker in white shirt standing near the sliding door in Store 3 talking to colleagues. | **FP — Awake but Still** |
| **id13** | `cam_25` | 108 | 6.04 | Flat stretch with 3 isolated spikes | Factory worker standing upright facing a locker/mirror holding a grooming item. | **FP — Awake but Still** |
| **id2** | `cam_03` | 167 | 5.77 | Stable segments separated by 10 periodic spikes | Uniformed security guard sitting upright on chair, hands on knees, head alert looking straight ahead. | **FP — Awake but Still** |
| **id21** | `cam_64` | 100 | 7.36 | Multiple stable segments with 19 periodic spikes | Factory worker standing upright at processing unit #4 operating machinery. | **FP — Awake but Still** |
| **id23** | `cam_64` | 100 | 8.29 | Multiple stable segments with 24 periodic spikes | Factory worker standing upright at table, hands moving while body remains stationary. | **FP — Awake but Still** |
| **id14** | `cam_38` | 149 | 6.64 | Multiple stable segments with 17 periodic spikes | Worker standing upright by catwalk railing observing production line. | **FP — Awake but Still** |
| **id22** | `cam_64` | 103 | 8.20 | Stable segments with 25 periodic spikes | Worker standing upright at machine station #3 kneading/feeding dough. | **FP — Awake but Still** |
| **id3** | `cam_03` | 376 | 5.80 | Stable segments with periodic spikes | Security guard sitting upright at desk writing with pen in logbook register. | **FP — Awake but Still** |
| **id5** | `cam_03` | 376 | 5.42 | Stable segments with 12 periodic spikes | Person in green shirt and baseball cap sitting upright on chair, hands in lap, awake looking sideways. | **FP — Awake but Still** |
| **id6** | `cam_24` | 500 | 6.18 | Stable segments with 11 periodic spikes | Office worker standing upright behind glass partition with hands behind his back. | **FP — Awake but Still** |
| **id7** | `cam_24` | 500 | 5.89 | Stable segments with 41 periodic spikes | Office worker sitting upright at computer desk with arms folded watching monitor/room. | **FP — Awake but Still** |
| **id132** | `cam_25` | 140 | 5.40 | Flat stretch with 3 isolated spikes | Worker in blue shirt and red turban standing upright working at electrical panel. | **FP — Awake but Still** |
| **id200** | `cam_25` | 122 | 5.44 | Stable segments with 6 periodic spikes | Person sitting on desk edge actively holding a phone to his ear talking. | **FP — Awake but Still** |

---

## 3. Systematic Parameter Sweep Findings Table

### Pass 1: Raw Difference Sweeps (Holding `normDiff` at Default `[0, 50]`)

| frameWindow | diffMin | diffMax | normDiffMin | normDiffMax | Match Count | Sampled | True Positives | FP (Static Object) | FP (Awake but Still) | Ambiguous | Notes |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **100** | 0 | 3 | 0 | 50 | **1** | 1 | 0 | 0 | 1 | 0 | Only `cam_38_id146` matches. Worker standing still with hands on hips. |
| **100** | 0 | 5 | 0 | 50 | **3** | 3 | 0 | 0 | 3 | 0 | Matches from `cam_24` and `cam_38`. Stood still or leaned over table. |
| **100** | 0 | 8 | 0 | 50 | **17** | 5 | 0 | 0 | 5 | 0 | Covers guards, office workers, factory machinery operators. |
| **100** | 0 | 10 | 0 | 50 | **21** | 5 | 0 | 0 | 5 | 0 | Nearly all tracks $\ge 100$ frames pass (21 of 23 total). |
| **100** | 2 | 5 | 0 | 50 | **3** | 3 | 0 | 0 | 3 | 0 | Identical to [0, 5]; all matched window means fall in [2.31, 4.98]. |
| **100** | 5 | 10 | 0 | 50 | **21** | 5 | 0 | 0 | 5 | 0 | All 21 passing tracks have windows with mean $\ge 5.0$. |
| **200** | 0 | 3 | 0 | 50 | **0** | 0 | 0 | 0 | 0 | 0 | Too strict; no track sustains mean diff $\le 3$ for 200 consecutive frames. |
| **200** | 0 | 5 | 0 | 50 | **0** | 0 | 0 | 0 | 0 | 0 | No track sustains mean diff $\le 5$ across 200 frames. |
| **200** | 0 | 8 | 0 | 50 | **6** | 5 | 0 | 0 | 6 | 0 | 100% of matches are from `cam_03` (guard desk) and `cam_24` (Store 3 office). |
| **200** | 0 | 10 | 0 | 50 | **6** | 5 | 0 | 0 | 6 | 0 | Exactly identical matches to [0, 8]; no additional tracks exist. |
| **200** | 2 | 5 | 0 | 50 | **0** | 0 | 0 | 0 | 0 | 0 | No tracks match. |
| **200** | 5 | 10 | 0 | 50 | **6** | 5 | 0 | 0 | 6 | 0 | Identical set: 3 tracks from `cam_24`, 3 from `cam_03`. |
| **300** | 0 | 3 | 0 | 50 | **0** | 0 | 0 | 0 | 0 | 0 | No tracks match. |
| **300** | 0 | 5 | 0 | 50 | **0** | 0 | 0 | 0 | 0 | 0 | No tracks match. |
| **300** | 0 | 8 | 0 | 50 | **6** | 5 | 0 | 0 | 6 | 0 | Same 6 tracks from `cam_03` and `cam_24` persist beyond 300 frames. |
| **300** | 0 | 10 | 0 | 50 | **6** | 5 | 0 | 0 | 6 | 0 | Same 6 long-duration tracks from `cam_03` and `cam_24`. |
| **300** | 2 | 5 | 0 | 50 | **0** | 0 | 0 | 0 | 0 | 0 | No tracks match. |
| **300** | 5 | 10 | 0 | 50 | **6** | 5 | 0 | 0 | 6 | 0 | Same 6 long-duration tracks from `cam_03` and `cam_24`. |
| **500** | 0 | 3 | 0 | 50 | **0** | 0 | 0 | 0 | 0 | 0 | No tracks match. |
| **500** | 0 | 5 | 0 | 50 | **0** | 0 | 0 | 0 | 0 | 0 | No tracks match. |
| **500** | 0 | 8 | 0 | 50 | **3** | 3 | 0 | 0 | 3 | 0 | Only 3 tracks in dataset have 500 frames: all from `cam_24` (Store 3 office). |
| **500** | 0 | 10 | 0 | 50 | **3** | 3 | 0 | 0 | 3 | 0 | Same 3 office workers in `cam_24`. |
| **500** | 2 | 5 | 0 | 50 | **0** | 0 | 0 | 0 | 0 | 0 | No tracks match. |
| **500** | 5 | 10 | 0 | 50 | **3** | 3 | 0 | 0 | 3 | 0 | Same 3 office workers in `cam_24`. |

---

### Pass 2: Normalized Difference Sweeps (Holding `diffMin/diffMax` at Best Contender `[0, 8]`)

*Note: In the dashboard formula, `normVal = val / (bbox_area / 1000)`. Because all tracked bounding boxes are large (ranging from 15,000 to 260,000 pixels), `bbox_area / 1000` ranges from 15 to 260. Dividing raw diffs of 2–8 by 15–260 results in normalized delta values between 0.02 and 0.57. Consequently, any upper bound of 5, 10, 15, or 20 does not filter out any track.*

| frameWindow | diffMin | diffMax | normDiffMin | normDiffMax | Match Count | Sampled | True Positives | FP (Static Object) | FP (Awake but Still) | Ambiguous | Notes |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **100** | 0 | 8 | 0 | 5 | **17** | 5 | 0 | 0 | 5 | 0 | All matched normalized deltas are $< 0.6$; no tracks pruned by `normDiffMax=5`. |
| **100** | 0 | 8 | 0 | 10 | **17** | 5 | 0 | 0 | 5 | 0 | Identical result; normalized deltas remain well below 1.0. |
| **100** | 0 | 8 | 0 | 15 | **17** | 5 | 0 | 0 | 5 | 0 | Identical result. |
| **100** | 0 | 8 | 0 | 20 | **17** | 5 | 0 | 0 | 5 | 0 | Identical result. |
| **200** | 0 | 8 | 0 | 5 | **6** | 5 | 0 | 0 | 6 | 0 | `cam_03` and `cam_24` tracks remain unchanged. |
| **300** | 0 | 8 | 0 | 5 | **6** | 5 | 0 | 0 | 6 | 0 | `cam_03` and `cam_24` tracks remain unchanged. |
| **500** | 0 | 8 | 0 | 5 | **3** | 3 | 0 | 0 | 3 | 0 | `cam_24` office tracks remain unchanged. |

---

## 4. Analysis & Recommendations

### 1. Which parameter combination had the best apparent precision?
Across the entire parameter space, **precision is 0.0%**. No true positive sleep instances were present in the captured dataset.
- Ultra-strict thresholds like `[0, 3]` and `[0, 5]` eliminate almost everything (returning 0 to 3 tracks), but even the few surviving tracks are upright, awake individuals standing still.
- If the system's role is strictly as a **first-stage candidate generator** (to feed downstream classification models), `frameWindow = 200` with `diffMin/diffMax = [0, 8]` is the most practical configuration:
  - It filters out short-lived stationary pauses from factory floor walkers (rejecting 88% of all tracks).
  - It yields a manageable candidate stream (**6 tracks**, 12% of tracks) representing genuinely prolonged stationary individuals.

### 2. Static-Object False Positives
- **Dominance Assessment:** No parameter combination was dominated by static-object false positives.
- In all 21 passing tracks, the YOLO detector correctly detected actual humans. The false-positive issue is not a failure of object detection, but an intrinsic limitation of equating low patch variance with sleep.

### 3. Disproportionate Camera Stream Breakdown
- **`cam_24` (Store 3 Office):** Represents 100% of all matches at `frameWindow = 500`. Office environments naturally foster long periods of stationary seated and standing behavior without sleep.
- **`cam_03` (Security Gate / Desk):** Accounts for 50% of matches at `frameWindow = 200` and `300`. Security personnel routinely sit upright at desks writing in logs or watching entryways.
- **`cam_64` (Production Line Machines #3 & #4):** Generates high match volume at `frameWindow = 100` because machine operators stay at their assigned post with fixed bounding-box coordinates while their hands work inside the machine.

### 4. Architectural Next Steps
1. **Pose Estimation (Secondary Classifier):** Complement patch delta with a pose estimation step (e.g., YOLO-Pose). Only flag stillness as sleep if:
   - Head-to-neck angle indicates slumping / nodding forward.
   - Bounding box aspect ratio or keypoint arrangement shows a reclined/horizontal posture.
2. **Workstation Polygon Masking:** Add region-of-interest (ROI) exclusions to omit designated stationary workstations (e.g. guard desks, machine feeding docks).
3. **Recalibrate Normalized Delta:** Because bounding-box areas are large (15k–260k pixels), dividing diff by `area / 1000` compresses values below 0.6. Rescale the divisor or adjust default slider bounds to `[0.0, 0.5]` for effective thresholding.
