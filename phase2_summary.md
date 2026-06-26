# Phase 2 Summary — Signal Preprocessing, Feature Extraction & Correlation Analysis

**Project:** Heart Murmur Detection from PCG Signals
**Phase:** 2 of 6
**Status:** ✅ COMPLETE (14/14 tasks done)
**Notebooks:** `notebooks/02a_preprocessing.ipynb`, `notebooks/02b_feature_correlation.ipynb`
**Figures:** `figures/preprocessing/`, `figures/correlation/`

---

## Task Overview

| Task | What was done | Key result |
|------|--------------|------------|
| 2.0 | Data quality assessment → preprocessing justification table | 6 observations from Phase 1 mapped to preprocessing decisions. Logical chain: observation → problem → solution. |
| 2.1 | Amplitude normalisation | `normalise_amplitude()` in `src/features/normalisation.py`. zero-mean, peak-divide → float64 [-1,1]. RMS varies 6× across recordings before normalisation. |
| 2.2 | Log-spectrogram (STFT) | `compute_log_spectrogram()` in `src/features/spectrogram.py`. Hann 50ms, hop 20ms → (101, T). Freq resolution 20 Hz, feature rate 50 Hz. |
| 2.3 | Frequency cropping (0–800 Hz) | `crop_frequency()`. 101 → 41 bins (59% removed). Boundary inclusive: `freqs[40] == 800.0`. |
| 2.4 | Per-row z-score normalisation | `zscore_per_row()`. Per-row (axis=1), NOT per-column. Each row mean≈0, std≈1. |
| 2.5a | Pipeline visualisation (s6) | 5-panel figure per example: Raw → Normalise → Spectrogram → Crop → Z-score. Murmur systolic energy visible in Panel 5. |
| 2.5b | Metadata feature correlation | Cramér's V / Kruskal-Wallis. All features V/η² < 0.15. Sex not significant (p=0.61). N recordings/patient strongest (η²=0.080). |
| 2.5c | Spectral frequency-band correlation | Mann-Whitney U per bin, Bonferroni corrected. 71/101 bins significant. **30 bins above 800 Hz still significant** — important finding. Peak effect size r≈0.47 at ~140 Hz. |
| 2.5d | Recording-level feature correlation | Kruskal-Wallis + pairwise Mann-Whitney. Duration not significant. Coverage/SNR/N recordings all significant with Unknown lowest. |
| 2.6 | Spectrogram comparison by class (s7) | 3-panel z-scored spectrogram comparison. Murmur systole clearly filled. Unknown has no regular structure. |
| 2.7 | Frequency content by class (s8) | Average power spectrum. Present > Absent 0–1400 Hz. Spike at ~1600 Hz in all 3 — hardware artifact. |
| 2.8 | Parameter documentation (s9) | Full parameter table + DAV terminology mapping in notebook. Direct content for report Methods section. |
| 2.9 | Reusable module | `extract_features()` + `extract_features_batch()` in `src/features/spectrogram.py`. Tested on 10 files: 10/10 success. |
| 2.10 | Pipeline framework diagram (s11) | 4-stage block diagram with Stage 1 expansion and I/O shapes. Saved as `s11_pipeline_diagram.png`. |

---

## Feature Extraction Pipeline — Definitive Parameters

These are the exact parameters used throughout the project. **Do not change** without documented experiment.

| Step | Function | Parameters | Input → Output |
|------|----------|------------|----------------|
| Load | `scipy.io.wavfile.read()` | sr=4000 (verify, raise if different) | `.wav` file → (N,) int16 |
| Normalise | `normalise_amplitude()` | zero-mean, peak-divide | (N,) int16 → (N,) float64 [-1,1] |
| STFT | `compute_log_spectrogram()` | Hann window, nperseg=200, noverlap=120, nfft=200 | (N,) → (101, T) |
| Crop | `crop_frequency()` | max_freq=800, inclusive (freqs <= 800) | (101, T) → (41, T) |
| Z-score | `zscore_per_row()` | axis=1 (per row), std=0 → replace with 1.0 | (41, T) → (41, T) |
| **Full pipeline** | `extract_features()` | all above combined | `.wav` → (41, T), freqs, times |

**Critical numbers:**
- `win_length = 200 samples = 50 ms`
- `hop_length = 80 samples = 20 ms`
- `noverlap = 120 = win_length - hop_length`
- `n_freq_bins = 41` (indices 0–40, frequencies 0–800 Hz at 20 Hz steps)
- `feature_rate = 50 Hz` (one frame every 20 ms)
- `freqs[40] = 800.0 Hz` (verify this if changing parameters)

---

## Module Reference

### `src/features/normalisation.py`
```python
normalise_amplitude(signal: np.ndarray) -> np.ndarray
    # Input:  raw PCG, any dtype (typically int16)
    # Output: float64, range [-1, 1]
    # Edge case: silent signal (peak==0) returns zero array
```

### `src/features/spectrogram.py`
```python
compute_log_spectrogram(signal, sr=4000, win_length_sec=0.050, hop_length_sec=0.020)
    # Returns: S_log (101, T), freqs (101,), times (T,)

crop_frequency(S_log, freqs, max_freq=800.0)
    # Returns: S_cropped (41, T), freqs_cropped (41,)

zscore_per_row(S_cropped)
    # Returns: S_norm (41, T), each row mean≈0 std≈1

extract_features(wav_path, sr=4000, win_length_sec=0.050, hop_length_sec=0.020, max_freq=800.0)
    # Full pipeline in one call
    # Returns: features (41, T), freqs (41,), times (T,)
    # Raises ValueError if loaded sr != expected sr

extract_features_batch(wav_paths, show_progress=True, **kwargs)
    # Returns: list of (features, freqs, times) or None on error
    # Prints error message for failed files, does NOT raise
```

**How to use in Phase 3:**
```python
from pathlib import Path
from src.features.spectrogram import extract_features, extract_features_batch

# Single file
features, freqs, times = extract_features("data/raw/training_data/2530_MV.wav")
# features.shape == (41, T)

# Batch — build paths from recordings.csv
import pandas as pd
recordings = pd.read_csv("data/metadata/recordings.csv")
DATA_ROOT = Path("data/raw/training_data")
wav_paths = [DATA_ROOT / Path(p).name for p in recordings['wav_path']]
results = extract_features_batch(wav_paths, show_progress=True)
```

---

## Correlation Analysis — Key Findings

### 2.5b Metadata Correlation
- **All metadata features have weak association** with murmur label (max V/η² = 0.139)
- Sex: V=0.032, p=0.61 — completely non-predictive
- Age: V=0.139, Pregnancy: V=0.123 — "significant" only due to large N, effect size negligible
- N recordings/patient: η²=0.080 — strongest, but driven by Unknown class (data quality artifact)
- **Conclusion:** Audio-only pipeline is justified. No metadata feature should be added to model input.

### 2.5c Spectral Frequency-Band Correlation
- Present has **higher energy than Absent** across all frequency bins (effect size always positive)
- **Peak discrimination at ~140 Hz** (rank-biserial r ≈ 0.47) — this is where murmur energy is strongest
- Significant discrimination from **0–1400 Hz** (71/101 bins after Bonferroni)
- **30 bins above 800 Hz still significant** — 800 Hz cutoff is conservative, not optimal
- Effect size at 800 Hz: r ≈ 0.28 (60% of peak). Effect size at 1400 Hz: r ≈ 0.10
- **Phase 5 experiment candidate:** Test cutoff at 1000–1200 Hz to see if performance improves
- **Phase 4 XAI connection:** Compare this pre-model importance profile with post-model ablation (Task 4.5)

### 2.5d Recording-Level Correlation
- **Duration:** NOT significant (H=4.26, p=0.12) — same across all classes
- **Annotation coverage:** Highly significant. Unknown median=0.21 vs Present=0.58, Absent=0.53
- **SNR proxy:** Highly significant. Present (9.6 dB) > Absent (8.6 dB) > Unknown (5.9 dB)
  - Note: Present has HIGHEST SNR because murmur energy adds to signal amplitude
- **N recordings:** Unknown median=2 vs Present=4, Absent=4
- **Conclusion:** Unknown class = signal quality failure, not clinical ambiguity. Confirmed statistically.

---

## Files Produced

### Notebooks
| File | Description |
|------|-------------|
| `notebooks/02a_preprocessing.ipynb` | Preprocessing pipeline + visualisation (Sections 1–11) |
| `notebooks/02b_feature_correlation.ipynb` | Correlation analysis (Sections 1–4) |

### Source modules
| File | Description |
|------|-------------|
| `src/features/normalisation.py` | `normalise_amplitude()` |
| `src/features/spectrogram.py` | `compute_log_spectrogram()`, `crop_frequency()`, `zscore_per_row()`, `extract_features()`, `extract_features_batch()` |

### Figures (preprocessing/)
| File | Content |
|------|---------|
| `s2_amplitude_normalisation.png` | Before/after normalisation, 3 examples |
| `s3_spectrogram_full.png` | Full log-spectrogram (0–2000 Hz), 3 examples |
| `s4_frequency_crop.png` | Full vs. cropped spectrogram comparison |
| `s5_zscore_effect.png` | Before/after z-score (diverging colormap) |
| `s6_pipeline_normal.png` | 5-panel pipeline: Normal (2530_MV) |
| `s6_pipeline_murmur.png` | 5-panel pipeline: Murmur (9979_TV) — report main figure |
| `s6_pipeline_unknown.png` | 5-panel pipeline: Unknown (9983_MV) |
| `s7_spectrogram_comparison.png` | 3-class z-scored spectrogram comparison |
| `s8_frequency_content.png` | Average power spectrum by class |
| `s11_pipeline_diagram.png` | Full pipeline framework diagram (Figure 1 in report) |

### Figures (correlation/)
| File | Content |
|------|---------|
| `s1b_metadata_correlation.png` | Metadata association strength bar chart |
| `s2c_spectral_discrimination.png` | Per-frequency-band discrimination (3 panels) |
| `s3d_recording_features.png` | Recording-level features × class violin plots |

---

## Critical Things Phase 3 Must Know

### 1. How to load features
Use `extract_features()` or `extract_features_batch()` from `src/features/spectrogram.py`.
Do NOT re-implement preprocessing — the module is verified and tested.

```python
# Build paths correctly (files are flat in training_data/, no subfolders)
DATA_ROOT = Path("data/raw/training_data")
wav_path = DATA_ROOT / Path(row['wav_path']).name   # from recordings.csv
features, freqs, times = extract_features(wav_path)
```

### 2. Output shape
`extract_features()` returns `(41, T)` — **41 rows (frequency bins), T columns (time frames)**.
RNN input at each timestep = vector of 41 values.
T varies by recording duration: T ≈ duration_seconds × 50.

### 3. Consider saving pre-computed spectrograms
Running `extract_features_batch()` on all 3163 recordings takes ~5–15 minutes.
Recommend saving to `data/processed/spectrograms/` as `.npy` files to avoid recomputing:

```python
import numpy as np
from pathlib import Path

processed_dir = Path("data/processed/spectrograms")
processed_dir.mkdir(parents=True, exist_ok=True)

for i, (result, wav_path) in enumerate(zip(results, wav_paths)):
    if result is not None:
        features, freqs, times = result
        stem = Path(wav_path).stem   # e.g. "2530_MV"
        np.save(processed_dir / f"{stem}.npy", features)
```

### 4. Class imbalance → weighted loss
Present=179, Unknown=68, Absent=695. RNN training must use class-weighted
cross-entropy loss. Weights from paper: inversely proportional to class frequency.
Compute at the **frame level** (not patient level) since each frame has a state label.

### 5. Variable-length sequences → batching strategy
T varies across recordings. Two options for Phase 3:
- **Pack sequences** (recommended): `torch.nn.utils.rnn.pack_padded_sequence`
- **Truncate to fixed length**: simpler but loses information

### 6. Ground-truth segmentation labels
RNN training needs frame-level state labels {S1=1, Systole=2, S2=3, Diastole=4, Murmur=5}.
Load from `.tsv` files in `data/raw/training_data/`.
Murmur state label = approximated from timing annotations (e.g. first 50% of systole
for early-systolic murmur). See paper Section "Recurrent Neural Network" for exact rules.
State 0 (Unannotated) discovered in Phase 1 — must be handled (mask from loss or treat as unlabelled).

---

## Critical Things Phase 4 (XAI) Must Know

### Pre-model frequency importance baseline (Task 2.5c)
The spectral discrimination analysis provides a **data-driven frequency importance profile**:
- Peak importance: ~140 Hz (r≈0.47)
- Importance drops gradually: ~0.28 at 800 Hz, ~0.10 at 1400 Hz

Task 4.5 should compare this against post-model ablation:
```
Pre-model  (Task 2.5c): Rank-biserial r per frequency bin
Post-model (Task 4.5) : Confidence drop when zeroing out each frequency band
```

If consistent → model learned what data suggested.
If divergent → model found higher-level patterns OR is relying on spurious features.

The figure `figures/correlation/s2c_spectral_discrimination.png` should be
referenced directly in the XAI section of the report.

---

## Key Limitations Discovered in Phase 2

These should be explicitly discussed in the report:

1. **800 Hz cutoff is conservative:** 30 frequency bins above 800 Hz are statistically
   discriminative (effect size r=0.10–0.28). The cutoff trades off some discriminative
   content for noise reduction. Phase 5 should test higher cutoffs (1000, 1200 Hz).

2. **Per-row z-score removes absolute energy information:** Z-scoring normalises
   away the fact that Present recordings have ~6 dB higher energy overall.
   The model must learn from temporal patterns, not absolute energy levels.
   This may hurt detection of very loud murmurs (Grade 3) less than quiet ones (Grade 1).

3. **Feature extraction is per-recording, not per-patient:** Each recording is
   processed independently. Patient-level aggregation happens in Stage 4 (classification),
   not in feature extraction.
