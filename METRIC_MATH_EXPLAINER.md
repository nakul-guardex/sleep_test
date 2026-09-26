# Math Behind Each Pixel Difference Strategy
**Reference document for sleep detection benchmark**
*All formulas match exactly what is computed in `live_sleep_benchmark.py` and `pixel_diff_strategy_comparison.py`*

---

## Setup: What every metric operates on

Every metric takes **two 32×32 grayscale patches**:

- **R** = reference patch (locked when person first appears, re-locked if they move)
- **C** = current patch (same bounding box location, current frame)

Both are single-channel uint8 images — 1024 pixel values each, range 0–255.

```
R[i,j] ∈ {0..255}   (i,j) ∈ {0..31} × {0..31}
C[i,j] ∈ {0..255}
N = 32 × 32 = 1024 pixels
```

---

## 1. MAD — Mean Absolute Difference

### Formula
```
MAD = (1/N) × Σ |R[i,j] − C[i,j]|
    = mean of per-pixel absolute differences
```

### In plain language
For every pixel, compute how much brighter or darker it got. Take the average of all 1024 differences.

- Person **completely still**: same pixels, same values → MAD ≈ 0
- Person **shifts arm**: ~50–200 pixels change → MAD = 3–15
- Person **stands up**: most pixels change → MAD = 20–80

### Why it works
- Directly proportional to amount of pixel-level change
- No model to warm up, no state, no parameters
- Linear — doubling the motion approximately doubles MAD

### Why it fails with rolling mean (OLD logic)
Suppose a person moves every 30 seconds (a MAD spike to 30) but sits still the rest of the time (MAD ≈ 1). Over 250 frames:
```
average = (220 × 1 + 30 × 30) / 250 = (220 + 900) / 250 = 4.48
```
→ rolling mean = 4.48 → falls in band 0–5 → **OLD logic says sleeping. Wrong.**

This is exactly what Track 1225 showed: 287 spikes, yet OLD logic triggered.

### Why it works with consecutive streak (NEW logic)
The same person cannot maintain MAD < 5 for 250 *consecutive* frames if they move every 30 seconds. Every spike resets the streak counter to 0.

---

## 2. SSIM — Structural Similarity Index (inverted)

### Formula
SSIM is computed between R and C as:

```
SSIM(R,C) = [2·μR·μC + c1] × [2·σRC + c2]
            ─────────────────────────────────
            [μR² + μC² + c1] × [σR² + σC² + c2]

where:
  μR, μC   = local mean of R and C
  σR², σC² = local variance of R and C
  σRC      = local covariance of R and C
  c1, c2   = stability constants (c1=(0.01×255)², c2=(0.03×255)²)
```

SSIM output ∈ [−1, 1], where 1 = identical.

We **invert** it: `metric = 1 − SSIM(R, C)` so that 0 = identical, higher = more different.

### What each term captures
| Term | Captures |
|------|---------|
| `2·μR·μC / (μR²+μC²)` | **Luminance** — are both patches equally bright? |
| `2·σRC / (σR²+σC²)` | **Structure** — do bright/dark regions align? |
| `σR·σC / ...` | **Contrast** — is the variation similar in both? |

### In plain language
SSIM checks three things simultaneously: brightness match, pattern match, and contrast match. A patch that has the same edges and textures in the same places gets a high score even if absolute pixel values differ slightly.

### Why it works in theory
SSIM is more robust than MAD to uniform brightness shifts — if the whole patch gets 5 units brighter (lighting change), MAD would report a difference but SSIM would not (because structure and contrast are unchanged).

### Why it fails in practice on 32×32 CCTV patches
**Three compounding problems:**

**1. RTSP compression artefacts**
RTSP streams use H.264/H.265 which introduces block artefacts (8×8 or 16×16 pixel blocks). On a 32×32 patch, these artefacts span 2–4 blocks — they appear as structural patterns in C that were not in R, even if the person didn't move.
```
R: clean frame
C: same scene, but compressed differently this keyframe cycle
SSIM sees: different local covariance σRC → reports structural change
MAD sees: ~3 units difference → small, largely ignored
```

**2. 32×32 is too small for SSIM's designed operating range**
SSIM was validated on full-resolution (512×512+) image quality assessment. At 32×32, local statistics (μ, σ) are computed over tiny windows. A single noisy pixel affects a large fraction of the local window, making SSIM jittery.

**3. Result from data**
At W=250 OLD logic: SSIM = 8 false alarms vs MAD = 4. SSIM is noisier despite being a "better" metric by design.

---

## 3. MOG2 — Mixture of Gaussians Background Subtraction

### Algorithm (not a simple formula)
MOG2 models each pixel's intensity as a **mixture of K Gaussian distributions** (K=5 by default).

For each pixel position (i,j), it maintains:
```
{(μk, σk², wk)}  for k = 1..K

where:
  μk  = mean intensity of Gaussian k
  σk² = variance of Gaussian k
  wk  = weight (how often this Gaussian was the best match)
```

**On each new frame**, for every pixel:
1. Check if current intensity C[i,j] matches any existing Gaussian (within 2.5σ)
2. If yes → update that Gaussian's μk, σk², wk (learning rate η=0.005 by default)
3. If no → replace the lowest-weight Gaussian with a new one centred at C[i,j]

**Background selection**: the top Gaussians by weight (until they sum to 0.7) are called "background." Everything else is "foreground."

```
fg_mask[i,j] = 1 if C[i,j] not explained by background Gaussians
             = 0 otherwise

MOG2 metric = Σ fg_mask[i,j] / N = fraction of foreground pixels
```

### Why it is the best metric
**Adaptive model:** When a person is still for ~50–100 frames, MOG2 learns their static appearance. Their pixels join the background distribution. `MOG2 → 0`.

When they move, the new pixel values don't match the learned background → `fg_mask = 1` for those pixels → MOG2 spikes instantly.

**Absorbs slow drift:** Camera vibration, slow lighting changes, and RTSP noise cause small slow shifts in pixel values. MOG2 absorbs these into updated Gaussian means (via η). MAD cannot do this — it always compares to a fixed reference.

```
Example:
Frame 1–200: person still, MOG2 learns their appearance
Frame 200:   background model for person's pixels is stable
Frame 201:   person moves arm → 80 pixels don't match model → MOG2 = 80/1024 = 0.078
Frame 210:   person still again → MOG2 begins falling back to 0
```

### Why it has more false alarms at short streaks (5 at streak=60)
MOG2 starts fresh on every new track (or relock). With `history=200`, it takes ~50–100 frames to learn stable background Gaussians. During warm-up:
```
Frames 1–50: model not yet stable → fg_mask fluctuates → MOG2 > 0 even if still
Frames 50+:  model stable → MOG2 → 0 for still person
```
At streak=60, some tracks are still in warm-up when the 60-frame window starts → brief false alarm. By streak=150, all tracks have passed warm-up.

### Data confirmation
```
OLD W=60:  MOG2=19 vs MAD=71  (3.7× better)
OLD W=150: MOG2=1  vs MAD=15  (15× better)
OLD W=200: MOG2=0  vs MAD=8   (∞ better)
```
The improvement gets larger as window grows — consistent with the warm-up explanation.

---

## 4. pHash — Perceptual Hash Hamming Distance

### Algorithm (step by step)
pHash reduces an image to a 64-bit fingerprint using DCT:

```
Step 1: Resize R to 32×32 grayscale (already done)
Step 2: Compute 2D DCT of the 32×32 patch
         DCT[u,v] = Σ_i Σ_j R[i,j] · cos[π·u·(2i+1)/(2N)] · cos[π·v·(2j+1)/(2N)]
Step 3: Keep only top-left 8×8 block of DCT (64 low-frequency coefficients)
Step 4: Compute mean of these 64 values
Step 5: hash bit[k] = 1 if DCT[k] > mean, else 0
         → 64-bit binary hash

pHash(R) = 64-bit integer
pHash(C) = 64-bit integer

metric = Hamming distance = number of bits that differ
       = popcount(pHash(R) XOR pHash(C))
       ∈ {0, 1, 2, ..., 64}
```

### What it captures
The 8×8 top-left DCT block contains only **low-frequency** spatial components — the broad shapes and gradients in the image. High-frequency details (edges, texture, noise) are discarded.

This is why pHash works well for finding near-duplicate *photographs* — minor cropping, JPEG compression, and colour shifts don't affect low-frequency structure much.

### Why it is the worst metric for this use case

**Problem 1: Discrete output**
```
MAD output: 0.000, 1.234, 2.567, 4.891, ... (continuous)
pHash output: 0, 1, 2, 3, 4, 5, ...          (integer steps)
```
With a rolling mean over 250 frames, continuous MAD averages smoothly. pHash averages integers — a single frame with distance=8 pulls the mean up by 8/250 = 0.032. A single frame with MAD=80 pulls up by 80/250 = 0.32. The discrete nature of pHash makes the rolling mean less smooth and harder to threshold.

**Problem 2: 32×32 is already the reduced resolution**
pHash internally resizes to 32×32 before DCT. Our patch is *already* 32×32. So pHash is computing DCT on a patch that was already resized from a larger bbox — double-resizing loses even more detail and the 8×8 low-frequency block contains very coarse information.

**Problem 3: Low-frequency features are similar across frames**
For a person sitting still in a chair, the low-frequency structure (a blob in the frame, roughly the same brightness distribution) looks similar even when they make small arm movements. pHash may report distance=0 or 1 even for moderate motion — and then spike to 4–8 on a lighting change.

```
Person moves arm:       MAD might be 8.0,  pHash distance might be 0 or 1
Lighting flickers:      MAD might be 3.0,  pHash distance might be 3 or 4
```
This inverted sensitivity (misses motion, flags lighting) makes pHash particularly bad for this application.

**Data confirmation**:
```
OLD W=250: pHash = 12 false alarms  (3× worse than MAD=4, ∞ worse than MOG2=0)
NEW streak=100: pHash = 8           (worse than MAD=6, MOG2=1)
```

---

## Summary: Which math property determines performance

| Property | MAD | SSIM | MOG2 | pHash |
|----------|-----|------|------|-------|
| Output type | Continuous float | Continuous float | Continuous float | **Integer (discrete)** |
| Adapts to slow drift | ❌ Fixed reference | ❌ Fixed reference | **✅ Updates model** | ❌ Fixed reference |
| Sensitive to compression noise | Low | **High** | Low (absorbed) | Medium |
| Warm-up required | ❌ None | ❌ None | **⚠️ ~100 frames** | ❌ None |
| Effective patch resolution | Any | Needs large | Any | **Needs full image** |
| Rolling mean resistant to spikes | ❌ No | ❌ No | **✅ Yes** | ❌ No |

**Bottom line from the data:**
- **MOG2 wins** because it is the only metric that *adapts* — it learns what "still" looks like per-person and produces a near-zero signal only for genuine stillness
- **pHash loses** because it is discrete and low-resolution on 32×32 patches
- **SSIM loses** because it is sensitive to RTSP compression artefacts
- **MAD is a solid baseline** — simple, fast, and interpretable, but vulnerable to rolling-mean averaging of motion bursts

---

*All formulas match the implementation in `pixel_diff_strategy_comparison.py` and `live_sleep_benchmark.py`*
*Benchmark: 538 active tracks, Mohit CCTV, 1 FPS, 32×32 grayscale patches, 2026-09-26*
