# Phase 3 Guide — Model Reproduction

**Project:** Heart Murmur Detection from PCG Signals
**Phase:** 3 of 6
**Status:** ⬜ NOT STARTED
**Prerequisite:** Phase 2 COMPLETE (feature extraction pipeline verified)
**Notebook:** `notebooks/03_model_reproduction.ipynb`
**Source modules:** `src/models/`, `src/data/`, `src/evaluation/`
**Estimated total effort:** 25–32 hours

---

## 0. Phase Overview

### 0.1 Goal

Reproduce the full murmur detection pipeline from McDonald et al. (CinC 2022 / PLOS Digital Health 2024) and verify results against published benchmarks. This is the **critical gate** — no improvements (Phase 5) may begin until Phase 3 is verified.

### 0.2 What We Are Building

The pipeline has 4 stages. Stage 1 (feature extraction) is already complete from Phase 2. Phase 3 implements Stages 2–4:

```
Stage 1: Feature Extraction (Phase 2 — DONE)
    Raw PCG → normalise → log-spectrogram → crop 0–800 Hz → z-score
    Output: (41, T) per recording

Stage 2: RNN Segmentation (Sub-phase 3a)
    (41, T) spectrogram → Bidirectional GRU → 5-state posteriors P(q_t | x_{1:T})
    States: {S1=0, Systole=1, S2=2, Diastole=3, Murmur=4}

Stage 3: Parallel HSMM Decoding (Sub-phase 3b)
    5-state posteriors → 4 parallel HSMMs (ω₁–ω₄) → 4 Viterbi paths + confidences

Stage 4: Murmur Classification (Sub-phase 3c)
    Confidences → per-recording decision → per-patient aggregation → {Present, Unknown, Absent}
```

### 0.3 Reproduction Targets (from PLOS paper)

| Metric | Target | Tolerance |
|--------|--------|-----------|
| Weighted accuracy (challenge metric) | 0.798 | ±0.03 |
| Sensitivity — Present | 92.7% | ±3 pp |
| Sensitivity — Unknown | 30.9% | ±5 pp |
| Sensitivity — Absent | 77.6% | ±3 pp |
| AUC-ROC (binary, unknowns removed) | 0.947 | ±0.03 |

Confusion matrix to match (PLOS Table 1):
```
                    True: Present  True: Unknown  True: Absent
Pred: Murmur            166            19            117
Pred: Unknown              1            21             39
Pred: No murmur           12            28            539
```

**Note on discrepancy:** CinC paper reports training weighted accuracy = 0.817; PLOS reports 0.798 on the same data. The PLOS number is the more carefully reported one (journal peer review). Expect our result to fall somewhere in this range.

### 0.4 Sub-phase Structure

| Sub-phase | Tasks | Focus | Est. Hours |
|-----------|-------|-------|------------|
| **3a** — RNN Training | 3.1–3.7 | Ground truth, data loading, model, training, visualisation | 12–15 |
| **3b** — HSMM Implementation | 3.8–3.12 | Heart rate, durations, Viterbi, topologies, confidences | 8–10 |
| **3c** — Classification & Verification | 3.13–3.17 | Decision logic, aggregation, evaluation, comparison, report | 5–7 |

### 0.5 Dependency Graph

```
Phase 2 (DONE)
    │
    ▼
3.1  Create 5-state ground truth ──────────────────┐
    │                                                │
    ▼                                                │
3.2  Data loading / batching ──────────┐             │
    │                                   │             │
    ▼                                   │             │
3.3  BiGRU + FC architecture            │             │
    │                                   │             │
    ▼                                   │             │
3.4  Class-weighted CE loss             │             │
    │                                   │             │
    ▼                                   ▼             │
3.5  5-fold stratified CV split ◄── needs patients.csv + murmur labels
    │
    ▼
3.6  Train RNN ◄── 3.1 + 3.2 + 3.3 + 3.4 + 3.5
    │
    ├──────────────────────────┐
    ▼                          ▼
3.7  Visualise RNN          3.8  Heart rate estimation ◄── needs RNN posteriors
    predictions                │
    (checkpoint)               ▼
                            3.9  State duration distributions
                               │
                               ▼
                            3.10 Duration-dependent Viterbi
                               │
                               ▼
                            3.11 4 HSMM topologies (ω₁–ω₄) ◄── 3.10
                               │
                               ▼
                            3.12 Segmentation confidences C(ω)
                               │
                               ▼
                            3.13 Murmur classification logic
                               │
                               ▼
                            3.14 Per-patient aggregation
                               │
                               ▼
                            3.15 Evaluate (weighted accuracy, confusion matrix)
                               │
                               ▼
                            3.16 Compare against published benchmarks
                               │
                               ▼
                            3.17 Write reproduction verification report
```

### 0.6 Coding Conventions

These conventions are consistent with Phases 1–2 and must be followed throughout Phase 3:

- **Reusable logic in `src/`**, not in the notebook. The notebook calls functions and shows results.
- **Module structure:** `src/models/rnn.py`, `src/models/hsmm.py`, `src/models/viterbi.py`, `src/models/parallel_hsmm.py`, `src/data/splits.py`, `src/data/labels.py`, `src/evaluation/metrics.py`
- **NumPy/SciPy for HSMM**, PyTorch only for RNN. The HSMM is not a neural network.
- **State indexing convention (0-based):** S1=0, Systole=1, S2=2, Diastole=3, Murmur=4. This matches Python's zero-indexing. The `.tsv` files use 1-based (S1=1...Diastole=4); convert on load.
- **Patient-level operations use `patient_id`** as the grouping key, never recording-level IDs alone.
- **All random seeds set to 42** for reproducibility (`torch.manual_seed(42)`, `np.random.seed(42)`, `random.seed(42)`).
- **Config dict** at the top of the notebook captures all hyperparameters in one place.
- **Figures saved to `figures/results/`** with descriptive names.
- **Model checkpoints saved to `models/rnn/`** as `fold{k}_best.pt`.
- **Experiment results saved to `experiments/results/`** as CSV/JSON.

### 0.7 Paper Figures/Tables to Reproduce for Verification

These are the specific figures and tables from the PLOS paper that serve as verification targets. Reproducing them confirms your implementation matches the authors'.

| Paper Element | What It Shows | When to Check | Task |
|---------------|---------------|---------------|------|
| PLOS Fig 2 (CinC Fig 1) | RNN posteriors on a murmur recording (S1, S2, Murmur traces) | After RNN training | 3.7 |
| PLOS Table 1 (CinC Table 3) | 3×3 confusion matrix for murmur detection | After full pipeline | 3.15 |
| PLOS Table 2 | Per-class sensitivity, PPV, F1 | After full pipeline | 3.15 |
| CinC Table 2 | Weighted accuracy: training 0.817 (or PLOS 0.798) | After full pipeline | 3.16 |
| PLOS Fig 4 (CinC Fig 2) | HSMM confidence scatter plot (C^(M−N) vs max confidence) | After 3.12 | 3.16 |
| PLOS Fig 5 | Example recordings with segmentation overlay | After 3.11 | 3.7/3.16 |
| PLOS Fig 6 | ROC curve for murmur detection (binary, unknowns removed) | After full pipeline | 3.16 |

---

## Sub-phase 3a — RNN Training

### Task 3.1: Create Modified Segmentation Ground Truth (5-State with Murmur)

**Objective:** Convert the 4-state segmentation annotations from the dataset into 5-state labels that include a murmur state, matching the paper's training targets.

**Why this matters:** The original `.tsv` files only label {S1, Systole, S2, Diastole}. The paper's key innovation is adding a 5th state (Murmur) by using the clinician's murmur timing annotation to relabel parts of Systole as Murmur. Without this, the RNN cannot learn to detect murmurs — it would only learn normal segmentation.

**Theoretical background:**

The murmur timing annotation (from `Murmur locations` column in patient metadata) tells us *where in the cardiac cycle* the murmur appears. The paper approximates the murmur location within each systolic interval:

- **Early-systolic:** First 50% of each systolic interval → relabel as Murmur
- **Mid-systolic:** Middle 50% of each systolic interval → relabel as Murmur
- **Holosystolic:** Entire systolic interval → relabel as Murmur
- **Late-systolic:** Last 50% of each systolic interval → relabel as Murmur (only 1 case)
- **No murmur / Absent / Unknown:** Leave Systole as-is (no Murmur labels)

Important detail: the murmur timing is specified **per auscultation location** in the patient metadata. A patient might have "Holosystolic" at MV but no murmur at AV. You must match the timing annotation to the correct recording.

**Implementation steps:**

1. Load the `.tsv` segmentation file for each recording. Format: 3 columns (start_time, end_time, state), where state ∈ {0, 1, 2, 3, 4} meaning {Unannotated, S1, Systole, S2, Diastole}.
2. Convert segment boundaries to frame indices using feature_rate = 50 Hz: `frame = int(time_seconds * 50)`.
3. Create a frame-level label array of length T (same T as the spectrogram), initialised to -1 (unlabelled).
4. Fill in the label array using the `.tsv` segments. Map TSV states: 1→0 (S1), 2→1 (Systole), 3→2 (S2), 4→3 (Diastole). State 0 (Unannotated) → leave as -1.
5. For recordings with murmur: look up the murmur timing for this recording's auscultation location from the patient metadata. For each Systole segment, relabel the appropriate portion as Murmur (state index 4).
6. Save label arrays as `.npy` files in `data/processed/labels/`.

**How to find murmur timing per location:**

The patient `.txt` header files contain fields like `#Murmur locations: MV+TV` and `#Systolic murmur timing: Holosystolic`. However, `training_data.csv` already has parsed versions. The key columns are:
- `Murmur`: {Present, Unknown, Absent} — patient-level
- `Murmur locations`: comma-separated list of locations where murmur is heard (e.g. "MV+TV")
- `Systolic murmur timing`: the timing type (Holosystolic, Early-systolic, Mid-systolic, Late-systolic)

**Critical detail:** The murmur timing in the dataset is patient-level, not per-location. That is, if a patient has murmurs at MV and TV, both locations share the same timing annotation. The paper applies the same timing rule to all murmur-positive locations for that patient.

**Murmur relabelling logic (for a single systolic segment spanning frames [s_start, s_end]):**

```
s_len = s_end - s_start  # length of systolic segment in frames

if timing == "Holosystolic":
    labels[s_start : s_end] = MURMUR  # entire systole
elif timing == "Early-systolic":
    mid = s_start + s_len // 2
    labels[s_start : mid] = MURMUR    # first 50%
elif timing == "Mid-systolic":
    quarter = s_len // 4
    labels[s_start + quarter : s_end - quarter] = MURMUR  # middle 50%
elif timing == "Late-systolic":
    mid = s_start + s_len // 2
    labels[mid : s_end] = MURMUR      # last 50%
# remaining frames in systole keep state=SYSTOLE
```

**Input:** `.tsv` segmentation files, `training_data.csv` (or `patients.csv` + `recordings.csv`), feature rate = 50 Hz
**Output:** `data/processed/labels/{recording_id}.npy` — shape (T,), dtype int8, values in {-1, 0, 1, 2, 3, 4}

**File to create:** `src/data/labels.py`
- `load_segmentation_tsv(tsv_path) -> list of (start, end, state)` — raw segment loader
- `create_frame_labels(segments, duration_frames, feature_rate=50) -> np.ndarray` — 4-state labels
- `apply_murmur_relabelling(labels, segments, timing) -> np.ndarray` — add murmur state
- `create_labels_for_recording(recording_id, data_root, metadata, feature_rate=50) -> np.ndarray` — full pipeline for one recording
- `create_all_labels(recordings_df, data_root, patients_df, output_dir) -> dict` — batch processing

**Verification/checkpoint:**
- Spot-check 3 example recordings: 2530_MV (Normal — no Murmur frames), 9979_TV (Present, Holosystolic — all systole→murmur), one Early-systolic patient.
- Print label distribution: count frames per state across all recordings. Murmur should be much rarer than other states (the paper says loss weighting compensates for this).
- Verify that Unannotated frames (state 0 in TSV) map to -1, not to a valid state.
- Verify T of labels matches T of spectrogram for same recording.

**Common pitfalls:**
- **Off-by-one in frame indexing:** `int(time * 50)` vs `round(time * 50)`. Use `int()` (floor) to match the spectrogram's hop alignment.
- **TSV state 0 = Unannotated, not S1.** Phase 1 discovered this. These frames must be masked from the loss (label = -1), not treated as a valid class.
- **Murmur locations field parsing:** The field uses `+` as separator, e.g. `"MV+TV"`. Split on `+` to get a list.
- **Recordings where patient has murmur but that specific location has none:** If recording location is AV but murmur is only at MV+TV, do NOT apply murmur relabelling to the AV recording. Only relabel recordings whose location appears in `Murmur locations`.
- **Empty systole segments after split:** If a systolic segment is only 1 frame long, the 50% split produces an empty murmur region. Handle gracefully (skip or label entire segment).
- **Diastolic murmurs:** Only 5 patients have diastolic timing. The paper explicitly states they do NOT model diastolic murmurs — leave diastole as-is for these patients.

**Estimated time:** 2–3 hours

**Completion criteria:**
- [x] `src/data/labels.py` with all functions
- [x] `data/processed/labels/` populated with .npy files for all 3163 recordings
- [x] Label distribution printed and documented
- [x] 3 spot-checks verified (Normal, Holosystolic, Early-systolic)
- [x] T(labels) == T(spectrogram) verified for 10+ recordings

---

### Task 3.2: Implement Data Loading / Batching for Variable-Length Sequences

**Objective:** Create a PyTorch Dataset and DataLoader that handles variable-length spectrograms + labels, suitable for training the bidirectional GRU.

**Why this matters:** PCG recordings have variable duration (5–65 seconds → T ranges from ~250 to ~3250 frames). PyTorch requires same-length tensors within a batch. You need either padding+packing or a custom batching strategy.

**Implementation steps:**

1. Create a PyTorch Dataset class that loads pre-computed spectrograms and labels from `.npy` files.
2. Implement a `collate_fn` that pads sequences to the maximum length in each batch and records the original lengths.
3. Use `torch.nn.utils.rnn.pack_padded_sequence` / `pad_packed_sequence` in the RNN forward pass (Task 3.3) to avoid computing on padded frames.

**Dataset class design:**

```python
class PCGDataset(torch.utils.data.Dataset):
    def __init__(self, recording_ids, spectrogram_dir, label_dir):
        self.recording_ids = recording_ids
        self.spectrogram_dir = spectrogram_dir
        self.label_dir = label_dir

    def __len__(self):
        return len(self.recording_ids)

    def __getitem__(self, idx):
        rec_id = self.recording_ids[idx]
        features = np.load(spectrogram_dir / f"{rec_id}.npy")  # (41, T)
        labels = np.load(label_dir / f"{rec_id}.npy")          # (T,)
        return {
            'features': torch.FloatTensor(features.T),   # (T, 41) — RNN expects (T, features)
            'labels': torch.LongTensor(labels),           # (T,)
            'length': features.shape[1],                   # original T
            'recording_id': rec_id
        }
```

**Collate function:**

```python
def pcg_collate_fn(batch):
    # Sort by length (descending) — required for pack_padded_sequence
    batch.sort(key=lambda x: x['length'], reverse=True)
    lengths = [x['length'] for x in batch]
    max_len = lengths[0]

    # Pad features and labels
    features_padded = torch.zeros(len(batch), max_len, 41)
    labels_padded = torch.full((len(batch), max_len), -1, dtype=torch.long)  # -1 = ignore

    for i, item in enumerate(batch):
        T = item['length']
        features_padded[i, :T, :] = item['features']
        labels_padded[i, :T] = item['labels']

    return {
        'features': features_padded,    # (B, T_max, 41)
        'labels': labels_padded,         # (B, T_max)
        'lengths': lengths,
        'recording_ids': [x['recording_id'] for x in batch]
    }
```

**Input:** Pre-computed `.npy` spectrograms and labels from Tasks 2.9 and 3.1
**Output:** PyTorch DataLoader yielding padded batches

**File to create:** `src/data/loader.py`
- `class PCGDataset(Dataset)`
- `pcg_collate_fn(batch) -> dict`
- `create_dataloader(recording_ids, spect_dir, label_dir, batch_size, shuffle) -> DataLoader`

**Verification/checkpoint:**
- Load a batch and verify shapes: features `(B, T_max, 41)`, labels `(B, T_max)`, lengths match originals.
- Verify that padded positions have label = -1.
- Verify sorting: `lengths[0] >= lengths[1] >= ... >= lengths[B-1]`.

**Common pitfalls:**
- **Forgetting to transpose the spectrogram.** `extract_features()` returns `(41, T)` but the RNN expects `(T, 41)` — time is the sequence dimension.
- **Not sorting by length.** `pack_padded_sequence` requires descending length order.
- **Label -1 for both Unannotated AND padding.** Both should be ignored in the loss. The cross-entropy `ignore_index=-1` handles this (or use `ignore_index=-100` which is PyTorch's default, and set both padding and unannotated to -100).
- **Batch size too large for memory.** With T up to 3250 and 41 features, a batch of 32 is ~4MB. Should be fine for 32GB RAM. Start with batch_size=16 or 32.

**Estimated time:** 1.5–2 hours

**Completion criteria:**
- [x] `src/data/loader.py` complete
- [x] Batch shape verification passed
- [x] Padding positions verified as ignore-label
- [x] DataLoader produces reproducible batches (seed test)

---

### Task 3.3: Implement Bidirectional GRU + FC Head Architecture

**Objective:** Implement the exact RNN architecture described in the paper: 3-layer BiGRU → 2-layer FC with Tanh → softmax → 5 classes.

**Why this matters:** This is the neural network that converts spectrograms into per-frame state probabilities. The architecture is deliberately small (60 hidden units, 3 layers) to avoid overfitting on the small dataset.

**Theoretical background:**

A bidirectional GRU processes the sequence in both forward and backward directions, then concatenates the outputs at each timestep. This gives the model access to future context (important because knowing what comes after a sound helps identify it — e.g., diastole always follows S2).

Architecture (from PLOS paper):
```
Input: (T, 41)                              ← spectrogram frames
    │
    ▼
3-layer Bidirectional GRU                    ← hidden_size=60, dropout=0.1
    │  Output: (T, 120)                      ← 60 forward + 60 backward
    │
    ▼  Dropout(0.1)
FC layer 1: Linear(120, 60) + Tanh
    │
    ▼  Dropout(0.1)
FC layer 2: Linear(60, 40) + Tanh
    │
    ▼
Output layer: Linear(40, 5) + Softmax       ← 5 state posteriors
    │
    Output: (T, 5)                           ← P(q_t = ξ_i | x_{1:T})
```

**Key hyperparameters (from CinC Table 1 / PLOS):**

| Parameter | Value |
|-----------|-------|
| GRU hidden size | 60 |
| Number of GRU layers | 3 |
| Bidirectional | Yes |
| GRU dropout | 0.1 (between GRU layers, NOT on last layer output) |
| FC hidden sizes | [60, 40] |
| FC activation | Tanh |
| Dropout between GRU→FC and FC→FC | 0.1 |
| Output activation | Softmax (via log_softmax + NLLLoss, or just CrossEntropyLoss) |
| Number of output classes | 5 |

**Implementation steps:**

1. Define a `MurmurRNN` class inheriting from `nn.Module`.
2. Use `nn.GRU(input_size=41, hidden_size=60, num_layers=3, batch_first=True, bidirectional=True, dropout=0.1)`.
3. Add dropout layer after GRU output (before FC).
4. Add FC1: `nn.Linear(120, 60)` + `nn.Tanh()` + `nn.Dropout(0.1)`.
5. Add FC2: `nn.Linear(60, 40)` + `nn.Tanh()`.
6. Add output: `nn.Linear(40, 5)`. Do NOT include softmax here — `nn.CrossEntropyLoss` expects raw logits.
7. In `forward()`, use `pack_padded_sequence` before the GRU and `pad_packed_sequence` after.

**Important implementation detail — dropout placement:**

The paper says "Dropout is applied between both the GRU and fully-connected layers." This means:
- `nn.GRU(..., dropout=0.1)` applies dropout between GRU layers (layers 1→2, 2→3) but NOT after the final GRU layer.
- A separate `nn.Dropout(0.1)` between GRU output and FC1.
- A separate `nn.Dropout(0.1)` between FC1 and FC2.
- No dropout after FC2 (before the output layer).

**Input:** Padded batch `(B, T_max, 41)` + lengths list
**Output:** Logits `(B, T_max, 5)` — raw scores before softmax

**File to create:** `src/models/rnn.py`
- `class MurmurRNN(nn.Module)` with `__init__()` and `forward(features, lengths)`
- `count_parameters(model) -> int` — utility to verify model size

**Verification/checkpoint:**
- Verify parameter count. Expected: ~200k–300k parameters (small model).
- Forward pass on a dummy batch should produce output shape `(B, T_max, 5)`.
- Verify that `torch.softmax(output, dim=-1)` sums to 1.0 at each timestep.
- Apply the model to a single recording and check output shape matches `(1, T, 5)`.

**Common pitfalls:**
- **GRU dropout parameter:** `nn.GRU(dropout=0.1)` only works when `num_layers > 1`. With 3 layers, this is fine.
- **Packed sequence handling:** Make sure to unpack after GRU and before FC layers. The FC layers operate on `(B, T, features)` — they don't need packing.
- **Softmax vs CrossEntropyLoss:** PyTorch's `CrossEntropyLoss` expects raw logits (applies log-softmax internally). If you add softmax in the model, use `NLLLoss` instead. Recommended: output raw logits, use `CrossEntropyLoss`.
- **batch_first=True:** Ensure consistency — both GRU and packed sequences should use `batch_first=True`.

**Estimated time:** 1.5–2 hours

**Completion criteria:**
- [x] `src/models/rnn.py` complete
- [x] Parameter count verified and documented
- [x] Forward pass on dummy batch produces correct shapes
- [x] Model can be saved and loaded with `torch.save`/`torch.load`

---

### Task 3.4: Implement Class-Weighted Cross-Entropy Loss

**Objective:** Compute class weights inversely proportional to frame-level state frequencies and configure the loss function.

**Why this matters:** The 5 states are extremely imbalanced at the frame level. Diastole is the longest cardiac phase and dominates. Murmur is the rarest — it only appears in systolic portions of murmur-positive recordings. Without weighting, the model would learn to never predict Murmur.

**Implementation steps:**

1. Count the total number of frames for each state across ALL training recordings (only counting labelled frames, ignoring -1/unannotated).
2. Compute inverse-frequency weights: `weight[i] = total_labelled_frames / (num_classes * count[i])`.
3. Normalise weights so the minimum weight = 1.0 (optional, for readability).
4. Create `nn.CrossEntropyLoss(weight=weights_tensor, ignore_index=-1)`.

**Expected approximate frame distribution (order of magnitude):**

| State | Approx. % of labelled frames | Weight direction |
|-------|------|------|
| S1 | ~10% | Medium |
| Systole | ~15% (reduced after murmur relabelling) | Medium |
| S2 | ~10% | Medium |
| Diastole | ~55% | Low (most common) |
| Murmur | ~10% (only in Present recordings) | High (rarest) |

The exact numbers will depend on your label generation. Log the actual distribution.

**Input:** All label arrays from Task 3.1
**Output:** `nn.CrossEntropyLoss` configured with weights and `ignore_index`

**File to create:** Logic within the training code (can be a helper function in `src/data/labels.py` or in the notebook).
- `compute_class_weights(label_dir, recording_ids) -> torch.Tensor` — computes weights from label files

**Verification/checkpoint:**
- Print the weight for each class. Murmur should have the highest weight, Diastole the lowest.
- Verify that `ignore_index=-1` correctly excludes unannotated and padded frames from the loss.

**Common pitfalls:**
- **Computing weights at the patient level instead of frame level.** The paper says "inversely weighted to the frequency of each class label in the dataset" — this is frame-level.
- **Including -1 (unannotated) frames in the count.** Only count frames with state ∈ {0,1,2,3,4}.
- **Recomputing weights per fold.** Technically, you should compute weights only on the training fold, not on the validation fold. In practice, the distribution is stable across folds since the stratification is by patient murmur class, not by frame counts. Either approach is acceptable.

**Estimated time:** 0.5–1 hour

**Completion criteria:**
- [x] Class weights computed and documented
- [x] Loss function created with correct `ignore_index`
- [x] Quick sanity test: loss computes on a dummy batch without error

---

### Task 3.5: Implement 5-Fold Stratified Cross-Validation Split

**Objective:** Create patient-level stratified 5-fold CV splits, ensuring no data leakage between folds.

**Why this matters:** The paper uses 5-fold CV stratified by murmur class at the patient level. All recordings from the same patient must be in the same fold. This prevents the model from seeing a patient's AV recording during training and then being tested on their MV recording.

**Implementation steps:**

1. Load `patients.csv` to get the patient_id → murmur_label mapping.
2. Use `sklearn.model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` on patient IDs, stratified by murmur label.
3. For each fold, expand patient-level splits to recording-level using `recordings.csv` (patient_id → list of recording_ids).
4. Save the splits to `data/metadata/cv_splits.json` for reproducibility.

**Split structure:**
```python
{
    "fold_0": {
        "train_patients": [13918, 2530, ...],
        "val_patients": [9979, 9983, ...],
        "train_recordings": ["13918_AV", "13918_PV", "2530_MV", ...],
        "val_recordings": ["9979_TV", "9983_MV", ...]
    },
    "fold_1": { ... },
    ...
}
```

**Input:** `patients.csv`, `recordings.csv`
**Output:** `data/metadata/cv_splits.json`

**File to create:** `src/data/splits.py`
- `create_stratified_cv_splits(patients_df, recordings_df, n_splits=5, seed=42) -> dict`
- `load_cv_splits(path) -> dict`

**Verification/checkpoint:**
- Each fold should have ~188 validation patients and ~754 training patients.
- Murmur class proportions should be similar across all folds (~19% Present, ~7% Unknown, ~74% Absent).
- NO patient appears in both train and val within the same fold.
- All 942 patients appear in exactly one validation fold.

**Common pitfalls:**
- **Stratifying by recording instead of patient.** Must stratify at patient level.
- **Forgetting patients with duplicate recordings.** Patient 49748 has 6 recordings — all 6 must go to the same fold.
- **Different random state across runs.** Fix seed=42 for reproducibility.

**Estimated time:** 1 hour

**Completion criteria:**
- [x] `src/data/splits.py` complete
- [x] `data/metadata/cv_splits.json` saved
- [x] Fold balance verified (per-fold class distribution printed)
- [x] No leakage verified (set intersection test)

---

### Task 3.6: Train RNN, Log Loss Curves

**Objective:** Train the bidirectional GRU on all 5 folds, saving best models and logging training/validation loss curves.

**Why this matters:** This is where you train the core neural network. The loss curves tell you whether the model is converging, overfitting, or underfitting. The paper mentions using Adam optimiser but does not explicitly state the learning rate or number of epochs — you'll need to tune minimally or refer to the reference code.

**Hyperparameters (from paper + reference code):**

| Parameter | Value | Source |
|-----------|-------|--------|
| Optimiser | Adam | Paper |
| Learning rate | 1e-3 (default Adam) | Reference code (start here, reduce if unstable) |
| Batch size | 16–32 | Not specified; 16 is safe for memory |
| Epochs | 50–100 | Reference code uses early stopping; start with 100 |
| Early stopping patience | 10 epochs | Recommended; stop if val loss doesn't improve |
| Gradient clipping | 1.0 (optional) | Recommended for GRU stability |

**Training loop structure:**

```
For each fold k in 0..4:
    1. Create train/val DataLoaders from cv_splits
    2. Initialise model, optimiser, loss function
    3. For each epoch:
        a. Train: forward → loss → backward → step
        b. Validate: forward → loss (no gradient)
        c. Log train_loss, val_loss
        d. If val_loss is best so far → save model checkpoint
        e. If no improvement for `patience` epochs → stop early
    4. Save: models/rnn/fold{k}_best.pt
    5. Save: experiments/logs/fold{k}_loss.csv
```

**What to log per epoch:**
- Training loss (averaged over batches)
- Validation loss (averaged over batches)
- Learning rate (if using a scheduler)
- Epoch time

**Important: handling packed sequences in training loop:**

```python
# Forward pass with packing
packed = pack_padded_sequence(features, lengths, batch_first=True, enforce_sorted=True)
logits = model(features, lengths)  # model handles packing internally

# Loss computation — reshape for CrossEntropyLoss
# logits: (B, T_max, 5), labels: (B, T_max)
loss = criterion(logits.reshape(-1, 5), labels.reshape(-1))
# ignore_index=-1 handles padding and unannotated frames
```

**Input:** Pre-computed spectrograms, labels, CV splits
**Output:** 5 model checkpoints + 5 loss curve CSVs

**Files to create:**
- Training logic in `notebooks/03_model_reproduction.ipynb` (orchestration)
- `experiments/logs/fold{k}_loss.csv` — columns: epoch, train_loss, val_loss
- `models/rnn/fold{k}_best.pt` — saved state dict

**Verification/checkpoint:**
- Loss should decrease over epochs for both train and val.
- Train loss should be lower than val loss (normal), but the gap should not be extreme (>2× would suggest overfitting).
- Training time per fold on CPU: expect 15–40 minutes depending on epochs and batch size. Total for 5 folds: 1.5–3 hours.
- After training, plot loss curves for all 5 folds and verify convergence.

**Common pitfalls:**
- **Forgetting `model.train()` / `model.eval()`.** Dropout behaves differently in train vs eval mode.
- **Not calling `optimiser.zero_grad()` before backward.** Gradients accumulate by default in PyTorch.
- **Out-of-memory on long sequences.** If a batch contains a 65-second recording (T≈3250), memory spikes. Consider limiting batch size to 8 for batches with very long sequences, or set a maximum sequence length (e.g., 2500 frames = 50 seconds, truncating the tail).
- **Validation loss computed on wrong device.** If using CUDA, ensure tensors are on the same device. (For this project, CPU training is expected and this is not an issue.)
- **Not saving the BEST model.** Save the model with the lowest validation loss, not the last epoch's model.

**Estimated time:** 3–4 hours (including training time on CPU)

**Completion criteria:**
- [x] 5 fold models saved to `models/rnn/`
- [x] Loss curves logged and plotted (V24 — training loss curves)
- [x] All folds converge (loss decreasing)
- [x] No extreme overfitting (train/val gap reasonable)
- [x] Training time documented

---

### Task 3.7: Visualise RNN Predictions on Example Recordings

**Objective:** Run the trained RNN on the 3 cross-phase example recordings and plot the posterior probabilities to verify they match the paper's Figure 2.

**Why this matters:** This is the first qualitative check that the RNN is working correctly. The paper's Figure 2 shows that the RNN should confidently distinguish S1, S2, and Murmur. If your RNN's output looks nothing like Figure 2, there is a bug in the pipeline.

**What to plot (for each example recording):**

A 3-panel figure:
1. **Top panel:** Raw PCG waveform (from Phase 1)
2. **Middle panel:** Z-scored spectrogram (from Phase 2)
3. **Bottom panel:** RNN posterior probabilities — 5 lines for S1, Systole, S2, Diastole, Murmur

For the murmur recording (9979_TV), the Murmur posterior should show clear peaks during systole. For the normal recording (2530_MV), the Murmur posterior should be near zero everywhere. For the unknown recording (9983_MV), the posteriors may be noisy/uncertain.

**Implementation steps:**

1. Load the fold-k model where the example recording is in the validation set. (If the recording was in fold 0's validation, load `fold0_best.pt`.)
2. Run the model in eval mode on the recording's spectrogram.
3. Apply softmax to the logits to get posterior probabilities.
4. Plot.

**Input:** Trained model, spectrogram, waveform
**Output:** `figures/results/v13_rnn_posteriors_{recording_id}.png`

**Verification/checkpoint:**
- For the murmur example, the Murmur posterior should peak during systolic intervals.
- S1 and S2 peaks should be sharp and periodic (matching heart beats).
- The posteriors at each timestep should sum to 1.0.
- Compare qualitatively against PLOS Fig 2.

**Common pitfalls:**
- **Using the wrong fold's model.** The example must be from the validation set of the fold whose model you load, otherwise you're evaluating on training data (overly optimistic).
- **Forgetting `model.eval()` and `torch.no_grad()`.** Dropout must be off during inference.
- **Plotting all 5 posteriors is cluttered.** The paper only shows S1, S2, and Murmur in Figure 2 (systole and diastole are omitted for clarity). You can plot all 5 but consider making S1/S2/Murmur prominent and systole/diastole as faint lines.

**Estimated time:** 1–1.5 hours

**Completion criteria:**
- [x] 3 posterior plots saved to `figures/results/`
- [x] Murmur recording shows clear murmur peaks
- [x] Normal recording shows no murmur peaks
- [x] Qualitative match with paper's Figure 2

---

## Sub-phase 3b — HSMM Implementation

### Task 3.8: Implement Heart Rate Estimation via Autocorrelation

**Objective:** Estimate the heart rate of each recording using autocorrelation of the RNN's non-diastolic posteriors.

**Why this matters:** The HSMM needs to know how long each cardiac state (S1, Systole, S2, Diastole) should last for a given recording. These durations scale with heart rate. The paper improves on Springer's envelope-based method by using the RNN posteriors instead — the RNN has already learned to filter noise, producing a cleaner periodic signal.

**Theoretical background:**

The heart sound signal is quasi-periodic. The autocorrelation function of a periodic signal has peaks at integer multiples of the period. By finding the first major peak, we estimate the heart period and thus the heart rate.

The paper's approach:
1. Sum the non-diastolic RNN posteriors: `s(t) = P(S1|t) + P(Systole|t) + P(S2|t) + P(Murmur|t)` = `1 - P(Diastole|t)`.
2. Compute the autocorrelation of `s(t)`.
3. Search for the highest peak in the lag range corresponding to 30–180 bpm. At 50 Hz feature rate:
   - 180 bpm → period = 0.333s → lag = 17 frames
   - 30 bpm → period = 2.0s → lag = 100 frames
   - For the paediatric dataset, ~20% have HR > 120 bpm, so the search range is expanded from Springer's original 30–120 bpm.
4. The peak lag gives the heart period in frames → convert to seconds → convert to BPM.

**Note:** Springer's original method uses the autocorrelation of a smoothed homomorphic envelope of the raw signal. The paper's PLOS S1 Fig compares both approaches and shows the RNN-based method is much smoother. We implement only the RNN-based method.

**Implementation steps:**

1. Compute non-diastolic posterior sum: `s = posteriors[:, 0] + posteriors[:, 1] + posteriors[:, 2] + posteriors[:, 4]` (indices: S1=0, Systole=1, S2=2, Murmur=4).
2. Subtract the mean of `s` (zero-center for better autocorrelation).
3. Compute autocorrelation: `np.correlate(s, s, mode='full')` and take the right half, or use `scipy.signal.correlate`.
4. Normalise the autocorrelation by `acorr[0]` (so peak at lag=0 is 1.0).
5. Search for the highest peak in the lag range [min_lag, max_lag] where:
   - `min_lag = int(60 / max_bpm * feature_rate)` = `int(60/180*50)` = 17
   - `max_lag = int(60 / min_bpm * feature_rate)` = `int(60/30*50)` = 100
6. Use `scipy.signal.find_peaks` or a simple argmax on the slice to find the peak.
7. Convert: `heart_period_sec = peak_lag / feature_rate`, `heart_rate_bpm = 60 / heart_period_sec`.

**Fallback:** If no clear peak is found (autocorrelation is flat — happens for very noisy recordings), default to 80 bpm (typical resting rate).

**Input:** RNN posteriors (T, 5), feature_rate=50
**Output:** Heart rate estimate in BPM (float), heart period in seconds (float)

**File to create:** `src/models/hsmm.py` (start building the HSMM module here)
- `estimate_heart_rate(posteriors, feature_rate=50, min_bpm=30, max_bpm=180) -> (float, float)` — returns (heart_rate_bpm, heart_period_sec)

**Verification/checkpoint:**
- On the 3 example recordings, check that HR estimates are physiologically reasonable (60–160 bpm for paediatric population).
- Plot the autocorrelation function for one example and visually verify the peak corresponds to the heart period.
- Compare a few HR estimates with what you can count from the waveform/segmentation (e.g., count S1 peaks in a 10-second window).

**Common pitfalls:**
- **Autocorrelation normalisation.** Without normalisation, the autocorrelation peak at lag=0 dominates and makes it hard to find periodic peaks. Always normalise by `acorr[0]`.
- **Search range too narrow.** If you use Springer's original 30–120 bpm range, you'll miss paediatric tachycardia (>120 bpm). Use 30–180 bpm.
- **Noisy recordings with no clear peak.** Implement the fallback to default HR.
- **Full autocorrelation vs half.** `np.correlate(s, s, 'full')` returns a symmetric result of length 2T-1. You only need lags ≥ 0 (the second half).

**Estimated time:** 1.5–2 hours

**Completion criteria:**
- [x] `estimate_heart_rate()` function in `src/models/hsmm.py`
- [x] Tested on 3 example recordings with reasonable results
- [x] Fallback mechanism for failed peak detection
- [x] Autocorrelation plot saved for one example

---

### Task 3.9: Implement State Duration Distributions (Springer Method)

**Objective:** Given a heart rate estimate, compute Gaussian duration distributions for each cardiac state, following Springer et al.'s parameterisation.

**Why this matters:** The HSMM uses these duration distributions to constrain the Viterbi decoding. Without them, the Viterbi algorithm would allow physically impossible state sequences (e.g., S1 lasting 2 seconds). The durations scale with heart rate — a faster heart rate means shorter systole/diastole.

**Theoretical background:**

Springer et al. (2016) define the expected duration of each state as a fraction of the heart period, based on clinical literature:

| State | Mean duration (fraction of heart period) | Std deviation (fraction of heart period) |
|-------|------|------|
| S1 | 0.122 | 0.022 |
| Systole | 0.180 | 0.059 |
| S2 | 0.094 | 0.022 |
| Diastole | 1.0 - 0.122 - 0.180 - 0.094 = 0.604 | 0.104 |
| Murmur | Same as Systole (0.180) | Same as Systole (0.059) |

These fractions are multiplied by the heart period (in frames) to get the mean and std in frames. Then a Gaussian N(μ, σ²) is evaluated for each possible duration d = 1, 2, ..., D_max.

**Important:** The Murmur state's duration distribution should match Systole's, since murmurs occur during the systolic phase. For 5-state models (ω₃, ω₄), the Murmur and remaining Systole together span the original systolic interval — their individual durations are shorter. See Task 3.11 for how each topology handles this.

**Implementation steps:**

1. Given heart_period_frames = heart_period_sec × feature_rate:
2. Compute mean and std for each state in frames.
3. Set D_max = max possible duration in frames. A reasonable choice: `D_max = int(heart_period_frames * 2)` or a fixed upper bound like 200 frames (4 seconds).
4. For each state, evaluate the Gaussian PDF at d = 1, 2, ..., D_max. Truncate at d=0 (duration must be ≥ 1).
5. Normalise to sum to 1.0 (it's a probability distribution over durations).

**Input:** Heart period in seconds, feature_rate=50
**Output:** Dict of duration probability arrays, one per state, each of shape (D_max,)

**File to create:** Add to `src/models/hsmm.py`
- `compute_duration_distributions(heart_period_sec, feature_rate=50, d_max=None) -> dict`
  Returns `{0: np.array, 1: np.array, 2: np.array, 3: np.array, 4: np.array}` where key = state index.

**Verification/checkpoint:**
- For a heart rate of 100 bpm (period = 0.6s = 30 frames), S1 mean duration should be ~3.7 frames, Diastole mean ~18 frames.
- Each distribution should be unimodal, positive, and sum to 1.0.
- Plot the duration distributions for one example recording.

**Common pitfalls:**
- **Duration in samples vs frames.** Work in frames (feature_rate=50 Hz), not audio samples (4000 Hz).
- **Gaussian with very small std.** For fast heart rates, S1 might have mean=2 frames and std=0.5 frames. The Gaussian needs to be evaluated carefully and normalised.
- **D_max too small.** If D_max is smaller than a plausible diastole duration, the Viterbi will fail. Use at least `int(heart_period_frames * 1.5)`.
- **Forgetting to normalise.** The truncated Gaussian must sum to 1.0.

**Estimated time:** 1–1.5 hours

**Completion criteria:**
- [x] `compute_duration_distributions()` in `src/models/hsmm.py`
- [x] Tested for 3 different heart rates (e.g., 60, 100, 150 bpm)
- [x] Distributions sum to 1.0
- [x] Duration plot saved for one example

---

### Task 3.10: Implement Duration-Dependent Viterbi Algorithm

**Objective:** Implement the Springer duration-dependent Viterbi algorithm for HSMM decoding.

**Why this matters:** This is the core decoding algorithm. Unlike the standard Viterbi for HMMs (which only considers the previous state), the HSMM Viterbi also considers how long the current state has lasted. This enforces realistic state durations — S1 can't last 500ms and Diastole can't last 10ms.

**Theoretical background:**

The standard HMM Viterbi computes:
```
δ_t(j) = max_i [ δ_{t-1}(i) · a_{ij} · b_j(o_t) ]
```

The HSMM extension (Springer's formulation) replaces this with:
```
δ_t(j) = max_i max_d [ δ_{t-d}(i) · a_{ij} · p_j(d) · ∏_{s=t-d+1}^{t} b_j(o_s) ]
```

Where:
- `δ_t(j)` = probability of the best path ending in state j at time t
- `a_{ij}` = transition probability from state i to state j
- `p_j(d)` = probability of staying in state j for exactly d frames (from Task 3.9)
- `b_j(o_s)` = observation probability at time s for state j = RNN posterior `P(q_s = j | x_{1:T})`

In log domain (for numerical stability):
```
log δ_t(j) = max_i max_d [ log δ_{t-d}(i) + log a_{ij} + log p_j(d) + Σ_{s=t-d+1}^{t} log b_j(o_s) ]
```

The observation sum `Σ log b_j(o_s)` can be computed efficiently using cumulative sums.

**Implementation steps:**

1. Convert all probabilities to log domain.
2. Pre-compute cumulative log-observation sums for each state: `cum_obs[j][t] = Σ_{s=0}^{t} log b_j(o_s)`. Then `Σ_{s=a}^{b} log b_j(o_s) = cum_obs[j][b] - cum_obs[j][a-1]`.
3. Initialise δ and ψ (backtracking) matrices.
4. Forward pass: for each time t, for each state j, find the best (i, d) combination.
5. Backtrack to recover the optimal state sequence.

**Key algorithmic detail — transition matrix structure:**

The cardiac cycle is strictly sequential: S1 → Systole → S2 → Diastole → S1 → ... The transition matrix only allows:
- S1 → Systole (or S1 → Murmur, in ω₃)
- Systole → S2 (or Murmur → S2, in some topologies)
- S2 → Diastole
- Diastole → S1
- Self-transitions are handled implicitly by the duration model, NOT by the transition matrix.

This means `a_{ij}` is either 1.0 (allowed transition) or 0.0 (forbidden), making the transition matrix effectively a deterministic cycle. The actual "staying in a state" is modelled by the duration distribution, not by self-transitions.

**Input:** Log observation probabilities (T, N_states), transition matrix (N_states, N_states), duration distributions (dict of arrays)
**Output:** Optimal state sequence (T,), log-likelihood of the path

**File to create:** `src/models/viterbi.py`
- `duration_dependent_viterbi(log_obs, log_trans, log_durations, d_max) -> (np.ndarray, float)`
  - `log_obs`: (T, N_states) — log RNN posteriors
  - `log_trans`: (N_states, N_states) — log transition probabilities
  - `log_durations`: dict {state_idx: (D_max,)} — log duration probabilities
  - Returns: `(path, log_likelihood)` where path is (T,) int array

**Performance note:** The naive implementation is O(T × N² × D_max). With T≈1000, N=4 or 5, D_max≈200, this is ~400M operations per recording, which is manageable in pure NumPy. If too slow, consider Cython or restricting D_max. The reference code uses Cython (`viterbi_hmm.pyx`), but pure NumPy/Python should work for this project.

**Verification/checkpoint:**
- Run on a simple hand-crafted example (e.g., 4 states, T=20) where you know the correct path.
- Run on one real recording and verify the decoded path makes physiological sense (alternating S1-Systole-S2-Diastole cycles).
- Compare the log-likelihood with the greedy path (argmax at each frame). The Viterbi path should have equal or higher total log-likelihood.

**Common pitfalls:**
- **Numerical underflow.** Always work in log domain. `log(0)` → use `-np.inf` or a very large negative number (e.g., -1e10).
- **Off-by-one in duration indexing.** Duration d=1 means the state lasts exactly 1 frame. Index 0 in the duration array corresponds to d=1. Be explicit about this mapping.
- **Self-transitions in the transition matrix.** The HSMM does NOT use self-transitions. If you accidentally include them, the model will allow states to last indefinitely regardless of the duration distribution. Set all diagonal entries of the transition matrix to 0 (or -inf in log domain).
- **Not handling edge cases at sequence boundaries.** At t=0, there is no "previous state" — initialise δ_0 appropriately (assume the sequence starts in S1, or use a uniform prior).
- **Duration cumulative sum off-by-one.** Be very careful with the indices when computing `Σ_{s=t-d+1}^{t} log b_j(o_s)`.

**Estimated time:** 3–4 hours (this is the hardest single task in the project)

**Completion criteria:**
- [x] `duration_dependent_viterbi()` in `src/models/viterbi.py`
- [x] Tested on hand-crafted example with known answer
- [x] Tested on one real recording with physiologically sensible output
- [x] Execution time measured (should be < 5 seconds per recording)

---

### Task 3.11: Implement 4 HSMM Topologies (ω₁–ω₄)

**Objective:** Configure the 4 parallel HSMM models with their distinct observation mappings and transition matrices.

**Why this matters:** The 4 topologies represent 4 hypotheses about the signal: no murmur, holosystolic murmur, early-systolic murmur, or mid-systolic murmur. Running all 4 in parallel and comparing their confidences is how the algorithm decides whether a murmur is present and what type it is.

**The 4 topologies:**

**ω₁ — Normal (no murmur), 4 states: S1 → Systole → S2 → Diastole**
- Observation probabilities: Use RNN posteriors for {S1, Systole, S2, Diastole}. **Discard the Murmur posterior** and re-normalise the remaining 4 to sum to 1.0.
- Transition matrix: 4×4 cyclic (S1→Sys, Sys→S2, S2→Dia, Dia→S1).
- Duration distributions: 4 states as computed in Task 3.9.

**ω₂ — Holosystolic murmur, 4 states: S1 → Murmur(=Systole) → S2 → Diastole**
- Observation probabilities: Same 5-state RNN posteriors, but **replace the Systole posterior with the Murmur posterior** at every frame. Specifically: use {S1, Murmur, S2, Diastole} as the 4 observation channels. This means the "Systole" state in ω₂'s Viterbi is actually looking at how well the Murmur posterior matches.
- Transition matrix: Same 4×4 cyclic.
- Duration distributions: Same as ω₁ (Systole/Murmur have the same duration distribution).

**ω₃ — Early-systolic murmur, 5 states: S1 → Murmur → Systole → S2 → Diastole**
- Observation probabilities: All 5 RNN posteriors used directly.
- Transition matrix: 5×5, forcing S1→Murmur→Systole→S2→Diastole→S1.
- Duration distributions: Murmur and Systole each get half the normal systolic duration (roughly). More precisely, both use the same Systole duration distribution — the Viterbi will naturally split the systolic interval between them.

**ω₄ — Mid-systolic murmur, 5 states: S1 → Systole → Murmur → S2 → Diastole**
- Observation probabilities: All 5 RNN posteriors used directly.
- Transition matrix: 5×5, forcing S1→Systole→Murmur→S2→Diastole→S1.
- Duration distributions: Same as ω₃.

**Implementation steps:**

1. Create a function or class that, given a topology identifier (ω₁–ω₄) and the RNN posteriors, returns:
   - Modified observation probabilities (remapped/renormalised as needed)
   - Transition matrix
   - Duration distributions (possibly reindexed for the 4-state vs 5-state models)
2. Call `duration_dependent_viterbi()` from Task 3.10 with these parameters.

**Input:** RNN posteriors (T, 5), heart rate estimate, topology identifier
**Output:** Viterbi path (T,), log-likelihood

**File to create:** `src/models/parallel_hsmm.py`
- `class ParallelHSMM` or standalone functions:
  - `prepare_topology(posteriors, heart_period_sec, topology) -> (log_obs, log_trans, log_durations)`
  - `run_parallel_hsmm(posteriors, heart_period_sec) -> dict` — runs all 4 topologies, returns paths + confidences

**Verification/checkpoint:**
- For a murmur recording, ω₂/ω₃/ω₄ should produce a higher-quality path (higher confidence) than ω₁.
- For a normal recording, ω₁ should win.
- All Viterbi paths should be valid cardiac cycles (no impossible transitions).
- Print the decoded paths for one recording under all 4 topologies and verify they differ in how systole is handled.

**Common pitfalls:**
- **Re-normalisation for ω₁ and ω₂.** When discarding or replacing a posterior, the remaining probabilities won't sum to 1.0. You must re-normalise. In log domain: subtract `log(sum(exp(log_probs)))` (log-sum-exp).
- **State index mismatch between 4-state and 5-state models.** The 4-state models (ω₁, ω₂) use state indices 0–3, while the 5-state models (ω₃, ω₄) use 0–4. Make sure duration distributions and transition matrices match the correct state set.
- **ω₂'s "Systole" is actually Murmur.** The Viterbi still calls it "state 1" but the observation probability comes from the Murmur channel. Keep clear comments to avoid confusion.
- **ω₃ vs ω₄ transition matrices.** The only difference is the order of Murmur and Systole after S1. ω₃: S1→Murmur→Systole. ω₄: S1→Systole→Murmur. Get this right.

**Estimated time:** 2–3 hours

**Completion criteria:**
- [x] All 4 topologies implemented and callable
- [x] Tested on 3 example recordings
- [x] Correct topology wins for murmur vs normal recordings
- [x] Paths are physiologically valid

---

### Task 3.12: Compute Segmentation Confidences C(ω)

**Objective:** For each HSMM topology, compute the segmentation confidence by tracing the Viterbi path back through the RNN posteriors.

**Why this matters:** The confidence score C(ω) measures how well the RNN agrees with the HSMM's segmentation. A high confidence means the RNN's predictions align well with the decoded state sequence. This is the key metric used for classification in Stage 4.

**Theoretical background (from Equation 2 in CinC / Equation 2 in PLOS):**

```
C(ω) = (1/T) × Σ_{t=1}^{T} P(q_t = q̂_t^(ω) | x_{1:T}, θ)
```

In words: trace the Viterbi path through the ORIGINAL 5-state RNN posteriors (not the modified ones used for decoding). At each frame t, look up the RNN's posterior probability for the state that the Viterbi path assigned to that frame. Average over all frames.

**Critical subtlety for ω₁ and ω₂:**

For ω₁ (normal, 4 states), the Viterbi path has states {S1=0, Systole=1, S2=2, Diastole=3}. When computing C(ω₁), look up these states in the ORIGINAL 5-state posteriors (not the re-normalised ones used for Viterbi). So if the Viterbi says "state 1" (Systole), look up `posteriors[t, 1]` (Systole posterior) in the original RNN output.

For ω₂ (holosystolic), the Viterbi decoded "state 1" means Murmur (because we swapped the observation). When computing C(ω₂), look up `posteriors[t, 4]` (Murmur posterior) when the path says "state 1". The mapping is: 0→S1(0), 1→Murmur(4), 2→S2(2), 3→Diastole(3).

For ω₃ and ω₄ (5 states), the mapping is direct: state i in the path corresponds to posterior index i.

**Implementation steps:**

1. For each topology, decode the Viterbi path (from Task 3.11).
2. Create a mapping from Viterbi state indices to original RNN posterior indices (differs per topology).
3. For each frame t, look up `original_posteriors[t, mapping[path[t]]]`.
4. Average over all T frames.

**Input:** Viterbi path (T,), original RNN posteriors (T, 5), topology identifier
**Output:** C(ω) — a single float between 0 and 1

**File to create:** Add to `src/models/parallel_hsmm.py`
- `compute_confidence(path, posteriors, topology) -> float`

**Verification/checkpoint:**
- C(ω) should be between 0 and 1 (it's an average of probabilities).
- For a clean recording, C(ω̂) (the best confidence) should be high (>0.7).
- For a noisy recording, C(ω̂) should be lower (<0.6).
- The winning topology ω̂ = argmax C(ω) should match the expected murmur type.

**Common pitfalls:**
- **Using modified posteriors instead of original.** The confidence must trace through the original 5-state RNN posteriors, not the remapped/renormalised ones used for Viterbi decoding.
- **State mapping errors for ω₂.** This is the most error-prone topology because "Viterbi state 1" means Murmur, not Systole.
- **Averaging over padded/unannotated frames.** Only average over frames that were actually part of the signal (ignore padding). In practice, if you're running per-recording (not batched), this isn't an issue.

**Estimated time:** 1 hour

**Completion criteria:**
- [x] `compute_confidence()` in `src/models/parallel_hsmm.py`
- [x] Tested on 3 example recordings
- [x] Confidence values are in [0, 1]
- [x] Higher confidence for correct topology

---

## Sub-phase 3c — Murmur Classification & Verification

### Task 3.13: Implement Murmur Classification Logic

**Objective:** Convert per-recording HSMM confidences into a murmur classification decision (per-recording).

**Why this matters:** This is Stage 4 of the pipeline — the final decision logic that produces the murmur prediction. The paper uses a simple, interpretable rule set based on the HSMM confidences.

**Decision logic (from CinC paper Equations 4–6 / PLOS Equations 4–6):**

```
C^(M) = max(C(ω₂), C(ω₃), C(ω₄))     # best murmur model confidence
C^(N) = C(ω₁)                            # normal model confidence
C^(M-N) = C^(M) - C^(N)                  # murmur likelihood

Per-recording decision:
  if C^(M-N) > 0:  → Murmur detected (ω̂ ∈ {ω₂, ω₃, ω₄})
  else:              → No murmur (ω̂ = ω₁)

Signal quality estimate:
  C^(ω̂) = max(C(ω₁), C(ω₂), C(ω₃), C(ω₄))  # best confidence across all models
```

**Input:** Four confidence values C(ω₁), C(ω₂), C(ω₃), C(ω₄) for one recording
**Output:** Per-recording: murmur detected (bool), murmur type (ω₂/ω₃/ω₄), C^(M-N), C^(ω̂)

**File to create:** Add to `src/models/parallel_hsmm.py` or `src/evaluation/metrics.py`
- `classify_recording(confidences) -> dict` with keys: murmur_detected, best_topology, c_m_n, c_quality

**Verification/checkpoint:**
- On the 3 example recordings, check that the murmur recording is correctly classified as murmur and the normal recording as no murmur.

**Common pitfalls:**
- **Threshold is 0, not 0.5.** C^(M-N) > 0 means murmur, not > 0.5.
- **Confusing C^(ω̂) with C^(M).** C^(ω̂) is the max confidence across ALL 4 models; C^(M) is the max across only the 3 murmur models.

**Estimated time:** 0.5 hour

**Completion criteria:**
- [x] `classify_recording()` function complete
- [x] Tested on 3 examples
- [x] Murmur/Normal correctly distinguished

---

### Task 3.14: Implement Per-Patient Aggregation Rules

**Objective:** Aggregate per-recording decisions into a per-patient prediction: {Present, Unknown, Absent}.

**Why this matters:** Each patient has 1–6 recordings. The final clinical prediction is per-patient. The paper uses a clinically-motivated aggregation rule.

**Aggregation logic (from CinC paper):**

```
For a patient with recordings r₁, r₂, ..., r_n:

1. If ANY recording has murmur detected (C^(M-N) > 0):
   → Patient prediction = "Present"

2. Else if ANY recording has low signal quality (C^(ω̂) < 0.65):
   → Patient prediction = "Unknown"

3. Else:
   → Patient prediction = "Absent"
```

**The threshold 0.65 is for signal quality**, not for murmur likelihood. It determines when the algorithm should say "I'm not confident enough to make a call" rather than "no murmur."

**Input:** List of per-recording classification results for one patient
**Output:** Patient-level prediction ∈ {"Present", "Unknown", "Absent"}

**File to create:** Add to `src/evaluation/metrics.py` or `src/models/parallel_hsmm.py`
- `aggregate_patient_prediction(recording_results) -> str`

**Verification/checkpoint:**
- Patient with murmur in 1/4 recordings → Present (even if other 3 are normal).
- Patient with all normal recordings but one has C^(ω̂) < 0.65 → Unknown.
- Patient with all normal recordings and all C^(ω̂) ≥ 0.65 → Absent.

**Common pitfalls:**
- **Order of precedence.** Murmur detection takes priority over signal quality. If one recording has murmur and another has low quality, the prediction is "Present" (not "Unknown").
- **Threshold value.** The signal quality threshold is 0.65 (from the CinC paper). This was chosen empirically during the challenge.
- **Patients with only Unknown-quality recordings.** If all recordings have C^(ω̂) < 0.65 and none have murmur, the patient is "Unknown."

**Estimated time:** 0.5 hour

**Completion criteria:**
- [x] `aggregate_patient_prediction()` function complete
- [x] Tested with various scenarios (murmur in 1 of 4, all normal, mixed quality)

---

### Task 3.15: Evaluate Murmur Detection (Weighted Accuracy, Confusion Matrix)

**Objective:** Run the full pipeline on all 5 folds (using each fold's validation set) and compute the challenge evaluation metrics.

**Why this matters:** This produces the numbers you compare against the published results. The weighted accuracy is the primary metric.

**Metrics to compute:**

| Metric | Formula/Definition |
|--------|-------------------|
| Confusion matrix | 3×3 (rows = predicted, cols = true) |
| Weighted accuracy | Custom challenge metric (see below) |
| Per-class sensitivity | TP / (TP + FN) for each class |
| Per-class PPV (precision) | TP / (TP + FP) for each class |
| Per-class F1 | 2 × (Sens × PPV) / (Sens + PPV) |
| Macro F1 | Mean of per-class F1 |
| AUC-ROC (binary) | Using C^(M-N) as the score, treating Present vs Absent (Unknown removed) |

**Weighted accuracy formula (PhysioNet Challenge 2022):**

```
W = [[5, 3, 1],    # weight matrix: w[true_class][predicted_class]
     [5, 3, 1],    # rows = true class (Present, Unknown, Absent)
     [1, 1, 1]]    # cols = predicted class (Present, Unknown, Absent)

# Note: the challenge actually uses a cost-based metric.
# Weighted accuracy = Σ w[i][j] × C[i][j] / Σ w[i][j] × N[i]
# where C[i][j] = confusion matrix entries, N[i] = total in class i
# More precisely, it's the sum of correctly-weighted predictions.
```

The exact weighted accuracy formula from the challenge weights Present examples 5× more than Absent. The implementation should follow the official challenge scoring code.

**Full evaluation pipeline:**

```
For each fold k:
    1. Load fold k's trained RNN
    2. For each validation recording:
        a. Extract features (or load pre-computed)
        b. Run RNN → posteriors
        c. Estimate heart rate
        d. Run parallel HSMM → 4 confidences
        e. Classify recording
    3. Aggregate per-patient predictions
    4. Collect (true_label, predicted_label) pairs

Combine all 5 folds' predictions → compute metrics on all 942 patients
```

**Input:** Trained models, pre-computed features, CV splits, ground truth labels
**Output:** Confusion matrix, all metrics, saved to `experiments/results/reproduction_results.json`

**File to create:** `src/evaluation/metrics.py`
- `compute_weighted_accuracy(y_true, y_pred) -> float`
- `compute_all_metrics(y_true, y_pred, confidence_scores) -> dict`
- `evaluate_pipeline(models_dir, splits, features_dir, labels_dir, metadata) -> dict`

**Verification/checkpoint:**
- Confusion matrix should roughly match PLOS Table 1.
- Weighted accuracy should be in the range 0.77–0.83.
- Per-class sensitivities should roughly match PLOS Table 2.

**Common pitfalls:**
- **Using training data for evaluation.** Only evaluate on validation recordings within each fold.
- **Evaluating at recording level instead of patient level.** The challenge metric is per-patient.
- **Wrong weighted accuracy formula.** Use the exact PhysioNet Challenge scoring code as reference.
- **Forgetting to combine folds.** Each patient appears in exactly one validation fold. Combine all 5 folds' predictions to get results on all 942 patients.

**Estimated time:** 2–3 hours

**Completion criteria:**
- [x] All metrics computed and saved
- [x] Confusion matrix printed
- [x] Weighted accuracy computed
- [x] Results saved to `experiments/results/reproduction_results.json`

---

### Task 3.16: Compare All Metrics Against Published Benchmarks

**Objective:** Create a side-by-side comparison table of your results vs. published results and assess whether reproduction is successful.

**What to compare:**

| Metric | Published (PLOS) | Published (CinC) | Your Result | Within Tolerance? |
|--------|:---:|:---:|:---:|:---:|
| Weighted accuracy | 0.798 | 0.817 | ??? | ±0.03 of 0.798 |
| Sensitivity — Present | 92.7% | 92.7% | ??? | ±3 pp |
| Sensitivity — Unknown | 30.9% | 30.9% | ??? | ±5 pp |
| Sensitivity — Absent | 77.6% | 77.6% | ??? | ±3 pp |
| PPV — Present | 55.0% | 55.0% | ??? | ±5 pp |
| PPV — Unknown | 34.4% | 34.4% | ??? | ±5 pp |
| PPV — Absent | 93.1% | 93.1% | ??? | ±3 pp |
| Macro F1 | ~0.621 | — | ??? | ±0.03 |
| AUC-ROC (binary) | 0.947 | — | ??? | ±0.03 |

**Also reproduce these figures:**
- HSMM confidence scatter plot (PLOS Fig 4 / CinC Fig 2) — plot C^(M-N) vs C^(ω̂) for all recordings, coloured by true class.
- Segmentation examples (PLOS Fig 5) — show waveform + decoded segmentation for representative recordings.

**Input:** Results from Task 3.15, published numbers from papers
**Output:**
- `experiments/results/reproduction_comparison.md` — formatted comparison
- `figures/results/v14_hsmm_confidence_scatter.png` — HSMM confidence scatter
- `figures/results/v23_segmentation_examples.png` — segmentation visualisation

**Verification/checkpoint:**
- If within tolerance → reproduction is successful → proceed to Phase 4.
- If outside tolerance → investigate causes before proceeding (see Troubleshooting section below).

**Estimated time:** 1.5–2 hours

**Completion criteria:**
- [x] Comparison table created
- [x] HSMM confidence scatter plot saved
- [x] Segmentation examples saved
- [x] Success/failure assessment documented

---

### Task 3.17: Write Reproduction Verification Report

**Objective:** Document the reproduction attempt with enough detail for the DAV report.

**Content to include:**

1. **Methodology summary** — what was implemented, what was borrowed
2. **Hyperparameter table** — all parameters used, with source (paper/reference code/chosen)
3. **Training details** — epochs, training time, convergence behaviour
4. **Results comparison table** — your numbers vs published
5. **Qualitative verification** — RNN posterior plots, confidence scatter, segmentation examples
6. **Deviations from paper** — any intentional or unintentional differences
7. **Known discrepancies** — where results differ and possible explanations

**Input:** All results from Tasks 3.15–3.16
**Output:** `experiments/results/reproduction_report.md`

**Estimated time:** 1.5–2 hours

**Completion criteria:**
- [x] Report written with all sections
- [x] All figures referenced
- [x] Deviations documented honestly
- [x] Ready to feed into Phase 6 (report writing)

---

## Troubleshooting Guide

If reproduction results are significantly outside tolerance, check these in order:

### RNN Outputs Look Wrong (Task 3.7 fails)
1. **Label generation bug** — verify label distributions match expectations. Check a few TSV files manually.
2. **Feature transposition** — verify spectrogram is (T, 41) not (41, T) when fed to RNN.
3. **Learning rate too high** — loss oscillates instead of decreasing. Try 1e-4.
4. **Missing class weighting** — model predicts mostly Diastole. Verify weights are applied.
5. **Unannotated frames in loss** — verify `ignore_index` is working.

### HSMM Confidences Are All Low (<0.5)
1. **Heart rate estimation failed** — check autocorrelation plots. If HR is wrong, all duration distributions are wrong.
2. **Duration D_max too small** — Viterbi can't find valid paths.
3. **Transition matrix has self-transitions** — remove them.
4. **Observation probabilities are 0** — check for log(0) = -inf issues. Add a small epsilon before taking log.

### Murmur Classification Accuracy Is Too Low
1. **ω₂ posterior swap is incorrect** — the most common topology error. Verify by looking at what ω₂'s Viterbi labels as "Systole" — it should actually be periods where the RNN's Murmur posterior is high.
2. **Confidence state mapping error** — verify C(ω) traces through ORIGINAL posteriors, not modified ones.
3. **Signal quality threshold** — try sweeping from 0.5 to 0.75 to see if 0.65 is optimal for your implementation.
4. **Patient aggregation logic** — verify the priority order (murmur > low quality > absent).

### Results Are Close But Not Matching
1. **Random seed differences** — acceptable source of variance.
2. **Epoch/early-stopping differences** — if you train more/fewer epochs, results shift slightly.
3. **Heart rate search range** — verify you're using 30–180 bpm, not Springer's original 30–120.
4. **Small implementation details** — e.g., how you handle fractional frames during murmur relabelling.

If after systematic debugging the weighted accuracy is still below 0.75, consider the **Fallback Strategy** from the roadmap (use reference code outputs for the HSMM stage, focus on analysis and explainability).

---

## Final Checklist

### Sub-phase 3a — RNN Training
- [ ] 3.1 Create 5-state ground truth labels
- [ ] 3.2 Implement data loading / batching
- [ ] 3.3 Implement BiGRU + FC architecture
- [ ] 3.4 Implement class-weighted cross-entropy loss
- [ ] 3.5 Implement 5-fold stratified CV splits
- [ ] 3.6 Train RNN on all 5 folds, save checkpoints + loss curves
- [ ] 3.7 Visualise RNN predictions, compare with paper Fig 2

### Sub-phase 3b — HSMM Implementation
- [ ] 3.8 Implement heart rate estimation (autocorrelation)
- [ ] 3.9 Implement state duration distributions (Springer)
- [ ] 3.10 Implement duration-dependent Viterbi algorithm
- [ ] 3.11 Implement 4 HSMM topologies (ω₁–ω₄)
- [ ] 3.12 Compute segmentation confidences C(ω)

### Sub-phase 3c — Classification & Verification
- [ ] 3.13 Implement per-recording murmur classification logic
- [ ] 3.14 Implement per-patient aggregation rules
- [ ] 3.15 Evaluate full pipeline (weighted accuracy, confusion matrix, all metrics)
- [ ] 3.16 Compare against published benchmarks + produce confidence scatter plot
- [ ] 3.17 Write reproduction verification report

### Files to Create
| File | Purpose |
|------|---------|
| `src/data/labels.py` | Ground truth label generation (5-state with murmur) |
| `src/data/loader.py` | PyTorch Dataset + DataLoader for variable-length sequences |
| `src/data/splits.py` | 5-fold stratified CV split generation |
| `src/models/rnn.py` | MurmurRNN (BiGRU + FC head) |
| `src/models/hsmm.py` | Heart rate estimation + duration distributions |
| `src/models/viterbi.py` | Duration-dependent Viterbi algorithm |
| `src/models/parallel_hsmm.py` | 4 parallel HSMM topologies + confidence computation |
| `src/evaluation/metrics.py` | Weighted accuracy + all evaluation metrics |
| `data/processed/labels/` | Pre-computed frame-level label .npy files |
| `data/processed/spectrograms/` | Pre-computed spectrogram .npy files (from Phase 2) |
| `data/metadata/cv_splits.json` | Reproducible CV fold assignments |
| `models/rnn/fold{0-4}_best.pt` | Best model checkpoint per fold |
| `experiments/logs/fold{0-4}_loss.csv` | Training/validation loss per epoch |
| `experiments/results/reproduction_results.json` | Quantitative results |
| `experiments/results/reproduction_comparison.md` | Side-by-side comparison with published numbers |
| `experiments/results/reproduction_report.md` | Full reproduction report |

### Figures to Produce
| Figure | Corresponds to | Task |
|--------|---------------|------|
| `figures/results/v13_rnn_posteriors_*.png` | PLOS Fig 2 | 3.7 |
| `figures/results/v14_hsmm_confidence_scatter.png` | PLOS Fig 4 / CinC Fig 2 | 3.16 |
| `figures/results/v23_segmentation_examples.png` | PLOS Fig 5 | 3.16 |
| `figures/results/v24_training_loss_curves.png` | N/A (standard ML practice) | 3.6 |

---

## Quick Reference — Key Equations

**RNN output (per frame):**
```
P(q_t = ξ_i | x_{1:T}, θ)    where ξ_i ∈ {S1, Systole, S2, Diastole, Murmur}
```

**HSMM confidence (Eq. 2):**
```
C(ω) = (1/T) × Σ_{t=1}^{T} P(q_t = q̂_t^(ω) | x_{1:T}, θ)
```

**Model selection (Eq. 3):**
```
ω̂ = argmax_ω C(ω)
```

**Murmur likelihood (Eq. 4–6):**
```
C^(M) = max(C(ω₂), C(ω₃), C(ω₄))
C^(N) = C(ω₁)
C^(M−N) = C^(M) − C^(N)
```

**Patient aggregation:**
```
if any recording has ω̂ ∈ {ω₂, ω₃, ω₄}  →  "Present"
elif any recording has C^(ω̂) < 0.65      →  "Unknown"
else                                        →  "Absent"
```
