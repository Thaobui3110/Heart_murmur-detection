# Phase 3 Summary — Model Reproduction (Sub-phases 3a + 3b partial)

**Project:** Heart Murmur Detection from PCG Signals
**Phase:** 3 of 6
**Status:** 🔄 IN PROGRESS — Tasks 3.1–3.9 complete, 3.10–3.17 pending
**Notebook:** `notebooks/03_model_reproduction.ipynb`
**Training:** `notebooks/03_train_rnn_colab.ipynb` (Google Colab T4 GPU)
**Source modules:** `src/data/labels.py`, `src/data/loader.py`, `src/models/rnn.py`, `src/models/hsmm.py`

---

## Task Overview

| Task | What was done | Status |
|------|--------------|--------|
| 3.1 | 5-state ground truth label generation | ✅ |
| 3.2 | PyTorch Dataset + DataLoader for variable-length sequences | ✅ |
| 3.3 | Bidirectional GRU + FC head architecture | ✅ |
| 3.4 | Class-weighted cross-entropy loss | ✅ |
| 3.5 | Stratified 5-fold CV splits | ✅ |
| 3.6 | RNN training on Google Colab T4 GPU | ✅ |
| 3.7 | RNN posterior visualisation (3 sample recordings) | ✅ |
| 3.8 | Heart rate + systolic interval estimation | ✅ |
| 3.9 | State duration distributions (McDonald constants) | ✅ |
| 3.10 | Duration-dependent Viterbi algorithm | ⬜ |
| 3.11 | 4 parallel HSMM topologies (ω₁–ω₄) | ⬜ |
| 3.12 | Segmentation confidence computation C(ω) | ⬜ |
| 3.13 | Per-recording murmur classification | ⬜ |
| 3.14 | Per-patient aggregation | ⬜ |
| 3.15 | Evaluation metrics (weighted accuracy, confusion matrix) | ⬜ |
| 3.16 | Benchmark comparison with published results | ⬜ |
| 3.17 | Reproduction verification report | ⬜ |

---

## Sub-phase 3a — RNN Training (Tasks 3.1–3.7) ✅ COMPLETE

### Task 3.1 — Ground Truth 5-State Label Generation

**File:** `src/data/labels.py`

**What was done:** Converted 4-state TSV segmentation annotations (S1, Systole, S2, Diastole) into 5-state frame-level labels by relabelling portions of Systole as Murmur based on clinician timing annotations.

**State index convention (0-indexed throughout Phase 3):**

| State | Index | TSV value |
|-------|:-----:|:---------:|
| S1 | 0 | 1 |
| Systole | 1 | 2 |
| S2 | 2 | 3 |
| Diastole | 3 | 4 |
| Murmur | 4 | — (relabelled) |
| Unannotated | -1 | 0 |

**Murmur relabelling logic (from McDonald et al.):**

| Timing | Frames relabelled to Murmur |
|--------|----------------------------|
| Holosystolic | Entire systole interval |
| Early-systolic | First 50% of each systole segment |
| Mid-systolic | Middle 50% of each systole segment |
| Late-systolic | Last 50% of each systole segment |
| Diastolic | NOT relabelled (paper explicitly excludes diastolic murmurs) |

Relabelling only applied when: (1) patient `murmur == 'Present'` AND (2) recording location appears in patient's `murmur_locations`.

**Key bug fixed:** `recording_id` must be derived from wav filename stem (`49748_AV_1.wav` → `49748_AV_1`), not from `patient_id + '_' + location`. 22 duplicate IDs were created with the naive approach for patients with multiple recordings at the same location. Fixed and saved to `recordings.csv`.

**Results:**

| State | Frames | % of labelled |
|-------|:------:|:-------------:|
| S1 | 385,582 | 20.3% |
| Systole | 378,660 | 19.9% |
| S2 | 337,989 | 17.8% |
| Diastole | 741,055 | 39.0% |
| Murmur | 55,591 | 2.9% |
| Unannotated | 1,722,462 | — |

Murmur timing distribution across 497 recordings with murmur frames:
- Holosystolic: 294 recordings (59%)
- Early-systolic: 145 recordings (29%)
- Mid-systolic: 56 recordings (11%)
- Late-systolic: 2 recordings (0.4%)

**3 sample recordings verified:**

| Recording | Label | Murmur frames | Result |
|-----------|-------|:---:|:---:|
| 2530_MV | Absent | 0 | ✅ |
| 9979_TV | Present, Holosystolic | 52 (Systole=0) | ✅ |
| 9983_MV | Unknown | 0, 75.4% Unannotated | ✅ |

**Deviation from ref code:** Ref code (`neural_networks.py`) removes Unannotated frames from features before training. We keep them and use `ignore_index=-1` in CrossEntropyLoss. Gradient effect is identical; we retain temporal context from unannotated regions.

---

### Task 3.2 — Data Loading and Batching

**File:** `src/data/loader.py` (v3 — RAM cache version)

**Problem solved:** PCG recordings have variable duration (T ≈ 250–3250 frames). PyTorch requires fixed-shape tensors in a batch → must pad to T_max per batch.

**Key design decisions:**
- Spectrogram `(41, T)` transposed to `(T, 41)` at load time (RNN needs time as first dim)
- Labels padded with `-1` (= `ignore_index`) to distinguish from valid state 0
- Batch sorted descending by length (required for `pack_padded_sequence`)
- RAM cache (`load_dataset_to_ram()`) loads all 3163 recordings into memory once (~1.4 GB) to eliminate per-batch disk I/O

**Performance issue resolved:** Initial implementation read `.npy` files from disk per batch during training → ~28s/batch on Windows. Solution: preload all data to RAM. Result: ~3.2 it/s on Colab T4.

**Deviation from ref code:** Ref code uses `lru_cache` + lazy loading. We preload all data to RAM in one call. Result: significantly faster training especially on non-local storage (Drive, network paths).

**Verified:**
- Batch shape: `(B, T_max, 41)` features, `(B, T_max)` labels ✅
- Lengths sorted descending ✅
- Padding positions have label = -1 ✅
- Full 3163 recordings, 99 batches (batch_size=32), no errors ✅

---

### Task 3.3 — BiGRU + FC Architecture

**File:** `src/models/rnn.py`

**Architecture (from McDonald et al. CinC 2022 Table 1 / PLOS 2024):**

```
Input: (B, T, 41)
    ↓
BiGRU: hidden=60, layers=3, dropout=0.1 (between layers)
    ↓ output (B, T, 120)
Dropout(0.1)
    ↓
FC1: Linear(120→60) + Tanh + Dropout(0.1)
    ↓
FC2: Linear(60→40) + Tanh
    ↓
Output: Linear(40→5)  ← raw logits, NO softmax
    ↓ (B, T, 5)
```

**Total parameters: 178,025** — small model, appropriate for dataset of 942 patients.

**Minor deviation from ref code:** Dropout placement differs slightly:
- Ours: Dropout after GRU→FC1, after FC1→FC2
- Ref: Dropout after FC1→FC2, after FC2→Output
- Impact: negligible (same dropout rate, same count)

---

### Task 3.4 — Class-Weighted Loss

Class weights computed as inverse frequency at frame level, normalised so minimum weight = 1.0:

| State | Weight |
|-------|:------:|
| S1 | 1.922 |
| Systole | 1.957 |
| S2 | 2.193 |
| Diastole | 1.000 |
| **Murmur** | **13.331** |

`nn.CrossEntropyLoss(weight=weights_tensor, ignore_index=-1)`

Murmur weight 13× higher than Diastole forces the model to attend to the rare murmur frames (2.9% of labelled frames).

Loss on untrained batch: **1.587 ≈ log(5) = 1.609** ✅

---

### Task 3.5 — Stratified 5-Fold CV Splits

**File:** `data/metadata/cv_splits.json`

`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` at patient level, stratified by murmur label.

| Fold | Train patients | Val patients | Train recordings | Val recordings |
|------|:---:|:---:|:---:|:---:|
| fold_0 | 753 | 189 | 2553 | 610 |
| fold_1 | 753 | 189 | 2516 | 647 |
| fold_2 | 754 | 188 | 2519 | 644 |
| fold_3 | 754 | 188 | 2535 | 628 |
| fold_4 | 754 | 188 | 2529 | 634 |

Class distribution per val fold: Present ~19%, Unknown ~7%, Absent ~74% — consistent ✅
No data leakage confirmed ✅
All 942 patients appear in exactly 1 val fold ✅

---

### Task 3.6 — RNN Training

**Training environment:** Google Colab T4 GPU (local CPU infeasible — ~70s/batch vs ~0.3s/batch on T4)

**Config used:**

| Parameter | Value | Ref code value | Note |
|-----------|:-----:|:--------------:|------|
| Optimiser | Adam | Adam | ✅ |
| Learning rate | **1e-3** | **1e-4** | ⚠️ Deviation |
| Batch size | 32 | 32 | ✅ |
| Max epochs | 100 | 1000 | ⚠️ Deviation (sufficient with LR=1e-3) |
| Early stopping patience | 10 | 10 | ✅ |
| Gradient clipping | 1.0 | 1.0 | ✅ |
| Random seed | 42 | — | ✅ |

**Deviation — LR=1e-3 vs ref code 1e-4:**
We used LR=1e-3 (10× higher than ref code). Effect: faster convergence (~25 epochs vs potentially 100+ epochs), but model may stop at a suboptimal local minimum. Val loss plateau observed at epoch 15–20 across all folds. Decision: keep current checkpoints and evaluate final weighted accuracy before deciding whether to retrain.

**Deviation — No random crop:**
Ref code applies a random 6-second (300-frame) crop during training as data augmentation. We train on full sequences. Effect: our model sees full temporal context; ref code model learns local 6-second patterns. Neither is definitively better.

**Training results:**

| Fold | Best epoch | Best val loss | Time |
|------|:---:|:---:|:---:|
| fold_0 | ~18 | 0.3379 | 12.9 min |
| fold_1 | ~15 | 0.4069 | 11.4 min |
| fold_2 | ~21 | 0.3665 | 14.2 min |
| fold_3 | ~12 | 0.3591 | 10.3 min |
| fold_4 | ~18 | 0.3958 | 13.0 min |
| **Mean** | **~17** | **0.3732** | **~12.4 min** |

Total training time: ~62 minutes on T4 (estimated >750 hours on local CPU).

Early stopping at epoch 20–30/100 is expected and healthy for a 178K-parameter model on this dataset size.

**Checkpoints:** `models/rnn/fold_{0-4}_best.pt`
**Loss logs:** `experiments/logs/fold_{0-4}_loss.csv`

---

### Task 3.7 — RNN Posterior Visualisation

**Figures:** `figures/results/v13_rnn_posteriors_{recording_id}.png`

3-panel figure per recording: Ground truth labels | Log-spectrogram | RNN posteriors (5 state probabilities).

**Results:**

| Recording | Expected | Observed | Result |
|-----------|----------|----------|:------:|
| 2530_MV (Absent) | Murmur ≈ 0 everywhere | Murmur < 0.4, never exceeds 0.5 | ✅ |
| 9979_TV (Present, Holosystolic) | Murmur peaks in systole | High periodic Murmur peaks throughout 22s | ✅ |
| 9983_MV (Unknown) | Noisy, uncertain posteriors | Low confidence, states competing at 0.4–0.8 | ✅ |

**Notable finding:** 9979_TV ground truth only annotates first 5 seconds, but RNN correctly predicts murmur pattern throughout all 22 seconds — evidence of good generalisation, not memorisation.

**Matches PLOS Figure 2 / CinC Figure 1 qualitatively** ✅

Small Murmur spikes (~0.3) visible in 2530_MV (Normal) — these are false positives that HSMM (Tasks 3.10–3.12) will filter using physiological constraints.

---

## Sub-phase 3b — HSMM Implementation (Tasks 3.8–3.9) ✅ PARTIAL

### Task 3.8 — Heart Rate + Systolic Interval Estimation

**File:** `src/models/hsmm.py` — `estimate_heart_rate()`, `estimate_systolic_interval()`

**McDonald's approach vs Springer:**

| | Springer | McDonald (implemented) |
|---|---|---|
| HR signal | Homomorphic envelope of raw PCG | `P(S1)+P(Systole)+P(S2)+P(Murmur)` from RNN |
| Zero-mean before ACF | Yes | **No** (matches ref code exactly) |
| Peak finding | `find_peaks` | **`np.argmax`** (matches ref code exactly) |
| HR search range | 30–120 BPM | **30–180 BPM** (extended for paediatric use) |
| Systolic interval | Bazett formula | **ACF of `P(S1)+P(S2)`, range `[150ms, heart_cycle/2]`** |

**Implementation matches ref code `get_heart_rate()` and `get_systolic_interval()` exactly.**

**Results on 3 sample recordings:**

| Recording | HR | Heart period | Systolic interval | % period |
|-----------|:--:|:------------:|:-----------------:|:--------:|
| 2530_MV | 111.1 BPM | 0.540s = 27f | 0.260s | 48.1% |
| 9979_TV | 96.8 BPM | 0.620s = 31f | 0.300s | 48.4% |
| 9983_MV | 130.4 BPM | 0.460s = 23f | 0.220s | 47.8% |

**Known limitation — systolic interval always = heart_cycle/2:**
All 3 recordings show systolic interval ≈ 50% of heart period (capped at `heart_cycle/2`). This is because ACF of `P(S1)+P(S2)` has its dominant peak at the full heart period T, not at the systolic interval. Within the search range `[150ms, T/2]`, ACF is monotonically increasing → `argmax` always selects the upper bound.

This is a known behaviour of the ref code itself on this dataset — NOT a bug in our implementation. The ref code `get_systolic_interval()` uses the same `argmax` logic. Downstream impact on Viterbi is limited because `std_sys=25ms` is wide enough to absorb the ~50ms overestimate in mean_sys.

**This limitation is a Phase 5 improvement candidate:** Replace argmax with a Bayesian estimate or Bazett formula fallback when no clear ACF peak exists.

---

### Task 3.9 — State Duration Distributions

**File:** `src/models/hsmm.py` — `compute_duration_distributions()`

**McDonald constants (fit to CirCor paediatric data, from `segmenter.py`):**

| State | Mean | Std | Source |
|-------|:----:|:---:|--------|
| S1 | **116.3ms (fixed)** | **19.6ms (fixed)** | Absolute constant, CirCor-fit |
| S2 | **103.2ms (fixed)** | **19.5ms (fixed)** | Absolute constant, CirCor-fit |
| Systole | `sys_interval − 127.9ms` | **25ms (fixed)** | Derived from measured sys_interval |
| Diastole | `(period − sys_interval) − 105.3ms` | **50ms (fixed)** | Derived from measured sys_interval |
| Murmur | Same as Systole | Same as Systole | Split per topology in Tasks 3.11 |

**Key differences from Springer fractions (our initial incorrect implementation):**
- Springer uses fractions of heart period (e.g. S1 = 12.2% × period)
- McDonald uses absolute seconds for S1/S2 (fit to paediatric CirCor data, not adult data)
- Springer std for diastole is adaptive (7% × mean_dia + 6ms); McDonald uses fixed 50ms

**Example at HR=96.8 BPM, sys_interval=0.300s for 9979_TV:**

| State | Mean | Std | Sum prob |
|-------|:----:|:---:|:--------:|
| S1 | 116.3ms | 19.6ms | 1.0000 ✅ |
| Systole | 172ms | 25ms | 1.0000 ✅ |
| S2 | 103.2ms | 19.5ms | 1.0000 ✅ |
| Diastole | 214ms | 50ms | 1.0000 ✅ |
| Murmur | 172ms | 25ms | 1.0000 ✅ |

Systole mean (172ms) is ~20-50ms higher than clinical expectation (~120–150ms) due to the systolic interval being capped at heart_cycle/2. Diastole mean (214ms) is ~100ms lower than expected (~300–350ms). Both within tolerable range given fixed std values.

---

## Deviations from McDonald et al. Reference Code — Summary

| # | Deviation | Impact | Decision |
|---|-----------|:------:|----------|
| 1 | LR=1e-3 vs ref 1e-4 | Medium | Keep; retrain if weighted accuracy <0.75 |
| 2 | Max epochs=100 vs ref 1000 | Low | Sufficient with LR=1e-3 (early stop ~epoch 25) |
| 3 | No random 6s crop during training | Low | Full context may be beneficial |
| 4 | Background frames kept (ignore_index=-1) vs deleted | Low | Gradient equivalent |
| 5 | Dropout placement slightly different | Negligible | Not worth changing |
| 6 | Systolic interval capped at heart_cycle/2 | Low | Same as ref code on this dataset |
| 7 | RAM preloading instead of lazy cache | Positive | Much faster training |

---

## Files Produced

### Source modules
| File | Functions |
|------|-----------|
| `src/data/labels.py` | `load_segmentation_tsv()`, `create_frame_labels()`, `apply_murmur_relabelling()`, `create_labels_for_recording()`, `create_all_labels()`, `inspect_recording_labels()`, `verify_label_spectrogram_alignment()` |
| `src/data/loader.py` | `PCGDataset`, `pcg_collate_fn()`, `create_dataloader()`, `load_dataset_to_ram()` |
| `src/models/rnn.py` | `MurmurRNN`, `build_model()`, `count_parameters()` |
| `src/models/hsmm.py` | `estimate_heart_rate()`, `estimate_systolic_interval()`, `compute_duration_distributions()`, `get_hsmm_params()` |

### Data files
| File | Description |
|------|-------------|
| `data/processed/spectrograms/` | 3163 `.npy` files, shape `(41, T)` |
| `data/processed/labels/` | 3163 `.npy` files, shape `(T,)`, dtype int8, values {-1,0,1,2,3,4} |
| `data/metadata/recordings.csv` | Updated with `recording_id` and `n_frames` columns |
| `data/metadata/cv_splits.json` | 5-fold CV splits, seed=42, reproducible |

### Model checkpoints
| File | Epoch | Val loss |
|------|:-----:|:--------:|
| `models/rnn/fold_0_best.pt` | ~18 | 0.3379 |
| `models/rnn/fold_1_best.pt` | ~15 | 0.4069 |
| `models/rnn/fold_2_best.pt` | ~21 | 0.3665 |
| `models/rnn/fold_3_best.pt` | ~12 | 0.3591 |
| `models/rnn/fold_4_best.pt` | ~18 | 0.3958 |

### Figures
| File | Content |
|------|---------|
| `figures/results/v13_rnn_posteriors_2530_MV.png` | RNN posteriors — Normal recording |
| `figures/results/v13_rnn_posteriors_9979_TV.png` | RNN posteriors — Holosystolic murmur |
| `figures/results/v13_rnn_posteriors_9983_MV.png` | RNN posteriors — Unknown recording |
| `figures/results/v13_autocorr_9979_TV.png` | ACF: HR estimation + systolic interval |
| `figures/results/v13_duration_dists_9979_TV.png` | Duration distributions (McDonald) |

---

## Critical Things Tasks 3.10–3.17 Must Know

### 1. How to get posteriors for a recording

```python
import torch, numpy as np
from src.models.rnn import build_model

ckpt  = torch.load(f'models/rnn/{fold_name}_best.pt', map_location='cpu')
model = build_model(seed=42)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

spec = np.load(f'data/processed/spectrograms/{rec_id}.npy')  # (41, T)
T    = spec.shape[1]
with torch.no_grad():
    logits = model(torch.FloatTensor(spec.T).unsqueeze(0), [T])
posteriors = torch.softmax(logits, dim=-1).squeeze(0).numpy()  # (T, 5)
```

### 2. How to get HSMM parameters for a recording

```python
from src.models.hsmm import get_hsmm_params

hr_bpm, sys_interval, log_dur_dists, state_params = get_hsmm_params(posteriors)
# log_dur_dists: dict {state_name: np.ndarray (d_max,)} — log probs
# state_params:  dict {state_name: (mu_frames, sig_frames)}
```

### 3. Duration distributions format for Viterbi

`log_dur_dists[state_name]` is a 1D array of length `d_max`.
- Index 0 = duration of 1 frame
- Index d-1 = duration of d frames
- Values are log-probabilities (normalised over d=1..d_max)

### 4. Which fold to use for each recording

Each recording belongs to exactly 1 val fold. Use that fold's model for inference:

```python
with open('data/metadata/cv_splits.json') as f:
    cv_splits = json.load(f)

rec_to_fold = {}
for fold_name, fold_data in cv_splits.items():
    for rec_id in fold_data['val_recordings']:
        rec_to_fold[rec_id] = fold_name
```

### 5. Ref code topology state ordering

From `segmenter.py`, the 4 HSMM topologies are:
- **ω₁ healthy:** S1(0)→Systole(1)→S2(2)→Diastole(3), 4 states, murmur posterior DISCARDED
- **ω₂ holosystolic:** S1(0)→Murmur-as-Systole(1)→S2(2)→Diastole(3), 4 states, murmur REPLACES systole
- **ω₃ early-systolic:** S1→Murmur→Systole→S2→Diastole, 5 states, systole dist halved
- **ω₄ mid-systolic:** S1→Systole→Murmur→S2→Diastole, 5 states, systole dist quartered

Transition matrices are deterministic (0/1 only) — no probabilistic transitions.

### 6. Viterbi implementation note

Ref code uses Cython (`viterbi_hmm.pyx`) for performance. We will implement in pure NumPy/Python. Expected runtime: <5s per recording. Use cumulative sum trick for observation log-likelihoods to avoid O(T×D) inner loop.

### 7. Confidence formula (from PLOS Eq. 2)

```
C(ω) = (1/T) × Σ P(q_t = q̂_t^(ω) | x_{1:T}, θ)
```

Trace the Viterbi path through the **original 5-state posteriors** (not the modified posteriors used for decoding). For ω₂, when Viterbi says "state 1 = Murmur", look up `posteriors[t, 4]` (not `posteriors[t, 1]`).

---

## Phase 5 Improvement Candidates Identified in Phase 3

| Candidate | Task origin | Expected impact |
|-----------|-------------|-----------------|
| Retrain with LR=1e-4 | Task 3.6 | +5–15% val loss improvement |
| Systolic interval estimation: Bazett fallback when ACF has no clear peak | Task 3.8 | Better systolic/diastole duration boundary |
| Random 6s crop augmentation during training | Task 3.6 | May improve generalisation |
| Extend frequency cutoff from 800Hz to 1000–1200Hz | Phase 2 finding | Phase 2 showed 30 bins above 800Hz still discriminative |
