# Phase 2 Guide — Signal Preprocessing, Feature Extraction & Correlation Analysis

**Project:** Heart Murmur Detection from PCG Signals
**Phase:** 2 of 6
**Status:** ⬜ NOT STARTED
**Prerequisites:** Phase 0 ✅, Phase 1 ✅
**Estimated Effort:** 16–20 hours
**Notebooks:** `02a_preprocessing.ipynb`, `02b_feature_correlation.ipynb`
**Figures output:** `figures/preprocessing/`, `figures/correlation/`

---

## 1. Phase Objectives

Phase 2 has **four interconnected goals**, ordered to follow the supervisor's required DAV narrative:

1. **Data Quality Assessment** — Summarise Phase 1 observations that motivate each preprocessing step. This creates the logical chain: *observation → problem → solution*.
2. **Preprocessing & Feature Transformation** — Implement the paper's Stage 1 feature extraction exactly, framing each step with DAV terminology (transformation / selection / normalisation). Visualise Input → Output at every stage.
3. **Feature Correlation Analysis** — Before any model training, analyse which features actually discriminate between murmur classes. This is a required DAV step that was not in the original paper.
4. **Feature Selection Justification** — Connect the correlation results back to the preprocessing decisions. For example: "spectral bands above 800 Hz show no class discrimination → the 800 Hz cropping is justified."

**Why this order matters:** The supervisor wants to see that preprocessing decisions are not arbitrary — they are driven by data observations (from EDA) and validated by statistical analysis (correlation). This is the difference between "we cropped at 800 Hz because the paper said so" and "we cropped at 800 Hz because our spectral analysis confirms no discriminative content exists above that frequency."

---

## 2. Prerequisites Check

Before starting, verify the following from Phase 1:

| Prerequisite | Location | Used for |
|---|---|---|
| `patients.csv` | `data/metadata/patients.csv` | Metadata correlation (Task 2.5b) |
| `recordings.csv` | `data/metadata/recordings.csv` | Recording-level correlation (Task 2.5d) |
| 3 cross-phase example recordings | 2530_MV (Normal), 9979_TV (Murmur), 9983_MV (Unknown) | Visualisation of every preprocessing step |
| `training_data.csv` | `data/raw/training_data/training_data.csv` | Authoritative metadata source |
| Raw .wav files | `data/raw/training_data/{pid}/` | Audio input for all processing |
| `src/data/parse_metadata.py` | `src/data/` | Metadata loading utility |
| `src/visualisation/style.py` | `src/visualisation/` | Consistent plot styling |

**Environment:** conda `heart-murmur`, Python 3.10. Packages needed: `numpy`, `scipy`, `librosa`, `pandas`, `matplotlib`, `seaborn`. All already installed.

---

## 3. The Paper's Feature Extraction Pipeline — Technical Reference

From the PLOS Digital Health paper (Section "Feature Extraction") and CinC 2022 (Section 2.1):

```
Raw PCG (4000 Hz, int16)
    │
    ▼ Step A: Amplitude normalisation
    │   x_norm = (x - mean(x)) / max(|x - mean(x)|)
    │   Output: float64, range [-1, 1]
    │
    ▼ Step B: Log-spectrogram (STFT)
    │   Window: Hann, length 50 ms (200 samples at 4000 Hz)
    │   Hop:    20 ms (80 samples)
    │   → Frequency resolution: 4000/200 = 20 Hz per bin
    │   → Time resolution (feature rate): 1/0.020 = 50 frames per second
    │   → Full spectrogram shape: (n_fft/2 + 1) × T = 101 freq bins × T time frames
    │   → Frequency range: 0 to 2000 Hz (Nyquist)
    │   Apply log: S_log = log(|STFT|² + ε)  where ε avoids log(0)
    │
    ▼ Step C: Frequency cropping (0–800 Hz)
    │   Keep only bins 0–40 (indices 0 to 40 inclusive, at 20 Hz resolution)
    │   → Cropped shape: 41 freq bins × T time frames
    │   Rationale: "remove higher frequencies that contain no heart sound information"
    │              and "reduces the risk of the RNN learning to overfit to irrelevant
    │              high-frequency noise such as speech and background sounds"
    │
    ▼ Step D: Per-row z-score normalisation
        For each frequency row f:
            S_norm[f, :] = (S_log[f, :] - mean(S_log[f, :])) / std(S_log[f, :])
        Rationale: "Murmurs commonly contain much less time-frequency energy than
                    S1 and S2 sounds, and this normalisation reduces the dynamic range"
        → Final output shape: 41 × T (ready for RNN input)
```

**Key numbers to get right:**
- Sampling rate: 4000 Hz (fixed by the Littmann 3200 device)
- Window length: 200 samples = 50 ms
- Hop length: 80 samples = 20 ms
- n_fft: 200 (same as window length — no zero-padding)
- Frequency bins after cropping: 41 (0 Hz, 20 Hz, 40 Hz, ..., 800 Hz)
- Feature rate: 50 Hz (one spectrogram frame every 20 ms)

---

## 4. Task-by-Task Detailed Guidance

---

### Task 2.0 — Data Quality Assessment → Preprocessing Justification Table

**Objective:** Create a structured table that maps every relevant Phase 1 observation to a preprocessing decision. This is the logical bridge between EDA and preprocessing that the supervisor explicitly requires.

**Why it matters:** The supervisor said: *"phải có nhận xét về dữ liệu... sau đấy sẽ diễn giải với những cái dữ liệu như vậy thì sẽ áp dụng những bước tiền xử lý như nào."* This task is that bridge — it proves your preprocessing is not arbitrary.

**What to produce:**

A table with columns: `Data Observation (Phase 1)` → `Problem It Causes` → `Preprocessing Step` → `DAV Category`

Expected rows:

| Data Observation (Phase 1) | Problem It Causes | Preprocessing Step | DAV Category |
|---|---|---|---|
| Recording amplitudes vary wildly (RMS ranges from ~1000 to ~7000 across examples; int16 clipping detected) | Models sensitive to absolute amplitude scale; comparisons across recordings meaningless without normalisation | Zero-mean + peak normalisation → amplitude range [-1, 1] | Feature Normalisation |
| Heart sounds are acoustic signals with time-varying frequency content; raw waveform doesn't expose frequency structure | Time-domain waveform alone can't distinguish S1/S2/murmur by their spectral signatures | STFT with Hann window → log-spectrogram (time-frequency representation) | Feature Transformation |
| Heart sound energy concentrated below ~800 Hz; higher frequencies contain speech, ambient noise, stethoscope artifacts | RNN may overfit to irrelevant high-frequency noise patterns that don't generalise | Crop spectrogram to 0–800 Hz (discard bins above 800 Hz) | Feature Selection |
| Murmurs have much lower time-frequency energy than S1/S2 sounds; dynamic range makes quiet murmurs nearly invisible in spectrogram | RNN may focus on loud S1/S2 and ignore subtle murmur content | Per-frequency-row z-score normalisation to equalise dynamic range | Feature Normalisation |
| Variable recording durations (5s–65s) | Fixed-length input required for batched training | Variable-length sequences handled by RNN (no truncation/padding at feature level) | (Addressed in Phase 3) |

**How to implement:** This is a documentation task, not a coding task. Write it as a Markdown table at the top of notebook `02a_preprocessing.ipynb`. It becomes the opening of the "Preprocessing" section in your final report.

**Pitfalls:**
- Don't just list steps from the paper. The point is to connect each step to a *data observation you actually made* in Phase 1.
- Each row should be traceable: "In Task 1.8 we saw that amplitude varies 6× across recordings → this motivates normalisation."

---

### Task 2.1 — Feature Transformation: Amplitude Normalisation

**Objective:** Implement and visualise the first preprocessing step: zero-mean, peak-amplitude normalisation.

**Why it matters:** Raw PCG signals are stored as int16 values with vastly different amplitudes across recordings. The spectrogram and all subsequent processing assume normalised input. This is also the simplest step — a good warm-up that produces a clear before/after visualisation.

**Algorithm:**
```python
def normalise_amplitude(signal):
    """Zero-mean, peak-normalise a PCG signal."""
    signal = signal.astype(np.float64)
    signal = signal - np.mean(signal)
    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = signal / peak
    return signal
```

**Input:** Raw PCG waveform, int16, variable length, sampling rate 4000 Hz.
**Output:** Float64 waveform, range [-1, 1], same length.

**Visualisation (part of V9):** For the 3 cross-phase examples, show:
- Top panel: raw waveform with y-axis in int16 units
- Bottom panel: normalised waveform with y-axis in [-1, 1]
- Annotate the peak amplitude of each raw signal to show the scale difference

**Code location:** `src/features/normalisation.py`

**Pitfalls:**
- Make sure to convert to float64 *before* subtracting mean (int16 overflow risk).
- Handle the edge case of a zero-signal (all silence) — peak would be 0, avoid division by zero.
- `librosa.load()` already returns float32 normalised audio. If you use librosa to load, you're loading a *pre-normalised* signal — which means you need to decide: use librosa's normalisation (which divides by max of int16 range = 32768) or implement the paper's normalisation yourself. **Recommendation:** Load with `scipy.io.wavfile.read()` to get raw int16, then normalise yourself, so you control exactly what happens and can show the before/after.

---

### Task 2.2 — Feature Transformation: Log-Spectrogram

**Objective:** Compute the STFT-based log-spectrogram, the core feature representation for the entire pipeline.

**Why it matters:** The spectrogram converts the 1D time-domain signal into a 2D time-frequency representation. This is the fundamental *feature transformation* of the project — everything downstream (RNN, HSMM) operates on this representation. The paper chose spectrogram over alternatives (MFCC, homomorphic envelope, wavelets) because it provides "an interpretable 2D visualisation of the time-frequency energy in the recording" (PLOS paper).

**Algorithm:**
```python
import numpy as np
from scipy.signal import stft

def compute_log_spectrogram(signal, sr=4000, win_length_sec=0.050, hop_length_sec=0.020):
    """
    Compute log-spectrogram following McDonald et al.

    Parameters:
        signal: normalised PCG waveform (float64)
        sr: sampling rate (4000 Hz)
        win_length_sec: window length in seconds (0.050)
        hop_length_sec: hop length in seconds (0.020)

    Returns:
        S_log: log power spectrogram, shape (n_freq, n_frames)
        freqs: frequency axis in Hz
        times: time axis in seconds
    """
    win_length = int(win_length_sec * sr)   # 200 samples
    hop_length = int(hop_length_sec * sr)   # 80 samples

    freqs, times, Zxx = stft(signal, fs=sr, window='hann',
                              nperseg=win_length, noverlap=win_length - hop_length,
                              nfft=win_length)

    # Power spectrogram + log
    S_power = np.abs(Zxx) ** 2
    S_log = np.log(S_power + 1e-10)  # epsilon to avoid log(0)

    return S_log, freqs, times
```

**Input:** Normalised PCG waveform, float64, range [-1, 1].
**Output:** Log-spectrogram, shape (101, T) where T = floor((N - 200) / 80) + 1. Frequency axis 0–2000 Hz.

**Key parameter choices to explain in the report:**
- **50 ms window:** "The time duration of S1 and S2 sounds is approximately 100 ms, so a 50 ms window was chosen to enable precise identification of the major heart sounds whilst maintaining an effective frequency resolution of 20 Hz" (PLOS paper).
- **20 ms hop:** Gives 50 frames/second, which is sufficient temporal resolution for heart sound segmentation.
- **No zero-padding:** n_fft = win_length = 200. This means frequency resolution = sr / n_fft = 4000/200 = 20 Hz.
- **Log transform:** Compresses dynamic range, makes quiet murmur energy more visible relative to loud S1/S2.

**Visualisation (part of V9):** Show the full spectrogram (0–2000 Hz) for the 3 cross-phase examples before cropping. Use `pcolormesh` with frequency on y-axis, time on x-axis, log power as colour.

**Alternative implementations:**
- `scipy.signal.stft` — recommended, gives explicit control over all parameters
- `librosa.stft` — also fine, but default parameters differ (n_fft=2048), so you must override everything explicitly
- `np.fft.rfft` on sliding windows — manual but fully transparent

**Recommendation:** Use `scipy.signal.stft` for clarity.

**Pitfalls:**
- `librosa.stft` uses `center=True` by default (pads signal), which changes output length. Set `center=False` to match the paper's behaviour, or use scipy.
- The `noverlap` parameter in scipy is `win_length - hop_length = 200 - 80 = 120`, not `hop_length` directly.
- The epsilon value (1e-10) for log should be small enough to not distort the spectrogram but large enough to prevent -inf. The exact value doesn't matter much; 1e-10 or 1e-8 are both fine.
- **Check dimensions:** For a 21.5s recording at 4000 Hz: N = 86000 samples → T = floor((86000-200)/80) + 1 = 1073 frames. Spectrogram shape: (101, 1073).

---

### Task 2.3 — Feature Selection: Frequency Cropping (0–800 Hz)

**Objective:** Crop the spectrogram to retain only the 0–800 Hz frequency range.

**Why it matters:** This is a *feature selection* step — we're deliberately discarding features (frequency bins above 800 Hz) that we believe carry no discriminative information and may contain harmful noise. In DAV terms, this reduces dimensionality from 101 to 41 frequency bins while preserving all clinically relevant content.

**Algorithm:**
```python
def crop_frequency(S_log, freqs, max_freq=800):
    """Crop spectrogram to 0-max_freq Hz."""
    freq_mask = freqs <= max_freq
    return S_log[freq_mask, :], freqs[freq_mask]
```

**Input:** Full log-spectrogram (101 × T), frequency axis 0–2000 Hz.
**Output:** Cropped log-spectrogram (41 × T), frequency axis 0–800 Hz.

**This step will be validated by Task 2.5c (spectral correlation analysis).** The correlation analysis should confirm that frequency bands above 800 Hz show no statistically significant difference between Present and Absent classes. If it does, that's strong justification. If it doesn't — if some high-frequency band *is* discriminative — that's an interesting finding worth discussing.

**Visualisation (part of V9):** Show full spectrogram (0–2000 Hz) next to cropped spectrogram (0–800 Hz). Draw a horizontal line at 800 Hz on the full spectrogram to show where the cut happens. Annotate: "Discarded: frequencies 820–2000 Hz (60 bins, 59% of spectrogram). Retained: frequencies 0–800 Hz (41 bins, 41% of spectrogram)."

**Pitfalls:**
- With 20 Hz resolution, 800 Hz corresponds to bin index 40 (0-indexed). So bins 0–40 inclusive = 41 bins. Double-check with `freqs[40]` = 800.0.
- The boundary is inclusive: we keep 800 Hz.

---

### Task 2.4 — Feature Normalisation: Per-Row Z-Score

**Objective:** Z-score normalise each frequency row independently across the time axis.

**Why it matters:** S1 and S2 heart sounds have much higher energy than murmurs in the spectrogram. Without normalisation, the RNN would see murmur-related frequency bins as near-zero compared to the dominant S1/S2 bins. Per-row z-scoring ensures every frequency bin has zero mean and unit variance *over time*, making the RNN equally sensitive to all frequency bands. This is the most important preprocessing step for murmur detection — it's what makes quiet murmurs "visible" to the model.

**Algorithm:**
```python
def zscore_per_row(S_cropped):
    """Z-score normalise each frequency row independently."""
    means = S_cropped.mean(axis=1, keepdims=True)
    stds = S_cropped.std(axis=1, keepdims=True)
    stds[stds == 0] = 1.0  # avoid division by zero for silent frequency bins
    return (S_cropped - means) / stds
```

**Input:** Cropped log-spectrogram (41 × T).
**Output:** Normalised spectrogram (41 × T), each row has mean≈0 and std≈1.

**Visualisation (V12 — Z-score effect):** Side-by-side comparison:
- Left: cropped log-spectrogram (before z-score) — S1/S2 dominate visually
- Right: z-scored spectrogram — all frequency bands at comparable scale, murmur regions more visible
- Use the murmur example (9979_TV) where the effect is most dramatic.

**Pitfalls:**
- Normalise per row (frequency axis), NOT per column (time axis). Each frequency bin is normalised independently across all time frames of that recording.
- Handle constant rows (std=0) which can occur in very high-frequency bins with no energy — replace std with 1.0 to produce a zero row.
- This normalisation is *per recording*, not across the dataset. Each recording is normalised independently.

---

### Task 2.5a — Visualise Preprocessing Pipeline (V9)

**Objective:** Create the flagship preprocessing visualisation showing all 4 steps applied sequentially, with intermediate results visible at each stage.

**Why it matters:** This directly satisfies the supervisor's requirement: *"đầu vào là cái gì, làm cái gì, đầu ra là cái gì, minh họa được ra... phải show ra được kết quả."* This figure becomes one of the most important in the report.

**What to produce — V9 (multi-panel figure):**

For each of the 3 cross-phase examples, create a figure with 5 panels stacked vertically:

1. **Raw waveform** — Input to the pipeline. Title: "Raw PCG (int16)". x-axis: time (s), y-axis: amplitude.
2. **Normalised waveform** — After Task 2.1. Title: "After Amplitude Normalisation". x-axis: time (s), y-axis: amplitude [-1, 1].
3. **Full log-spectrogram (0–2000 Hz)** — After Task 2.2. Title: "Log-Spectrogram (Feature Transformation)". x-axis: time (s), y-axis: frequency (Hz). Draw a horizontal dashed line at 800 Hz.
4. **Cropped log-spectrogram (0–800 Hz)** — After Task 2.3. Title: "After Frequency Cropping (Feature Selection: 0–800 Hz)".
5. **Z-scored spectrogram** — After Task 2.4. Title: "After Per-Row Z-Score (Feature Normalisation)". This is the final RNN input.

**Layout:** One tall figure per example (5 panels). Show all 3 in the notebook; pick the murmur example for the report's main figure.

**Annotations on each panel:** Brief text label describing what happened (e.g., "Hann window, 50 ms, hop 20 ms" on the spectrogram panel).

**Save to:** `figures/preprocessing/v9_pipeline_normal.png`, `v9_pipeline_murmur.png`, `v9_pipeline_unknown.png`

---

### Task 2.5b — Feature Correlation: Metadata

**Objective:** Compute and visualise the statistical association between patient metadata features and the murmur label (3-class: Present / Unknown / Absent).

**Why it matters:** This is the first of three required feature correlation analyses (a DAV step missing from the original paper). Even though Phase 1 already showed that demographics are not strong predictors (murmur prevalence ~19% across all groups), the supervisor wants to see a formal correlation analysis that proves it quantitatively. This also justifies the paper's decision to build an audio-only pipeline rather than a metadata-heavy model.

**Features to analyse:**

| Feature | Type | Correlation Method |
|---|---|---|
| Age category (Neonate/Infant/Child/Adolescent) | Ordinal / Categorical | Cramér's V vs. Murmur label |
| Sex (Male/Female) | Binary | Cramér's V or Chi-squared test |
| Height (cm) | Continuous (with missing values) | Point-biserial correlation (binary: Present vs. Absent) or Kruskal-Wallis (3-class) |
| Weight (kg) | Continuous (with missing values) | Same as height |
| Pregnancy status | Binary | Cramér's V |
| Number of recordings per patient | Discrete (1–6) | Point-biserial or Kruskal-Wallis |
| Campaign (CC2014 / CC2015) | Binary | Cramér's V |

**How to compute Cramér's V:**
```python
from scipy.stats import chi2_contingency

def cramers_v(x, y):
    """Compute Cramér's V for two categorical variables."""
    contingency = pd.crosstab(x, y)
    chi2, p, dof, expected = chi2_contingency(contingency)
    n = contingency.sum().sum()
    k = min(contingency.shape) - 1
    return np.sqrt(chi2 / (n * k)) if k > 0 else 0.0
```

**Visualisation (V11b):** Annotated heatmap showing the association strength (Cramér's V or equivalent) between each metadata feature and the murmur label. Annotate each cell with the value and p-value. Use a diverging colourmap — but most values should be near 0 (weak association), confirming the Phase 1 finding.

**Expected outcome:** All correlations should be weak (V < 0.1–0.15), confirming that metadata features have negligible discriminative power for murmur detection. The strongest correlation might be "number of recordings" vs. Unknown (already found in Phase 1: patients with 1 recording have 25.8% Unknown rate vs. 2.2% for 4 recordings).

**Report narrative:** "We performed formal correlation analysis between all available patient metadata features and the murmur classification label. As Table/Figure X shows, all associations are weak (Cramér's V < 0.15), confirming the Phase 1 finding that demographics are not predictive of murmur status. This validates the baseline paper's design decision to rely exclusively on the PCG audio signal for murmur detection."

**Save to:** `figures/correlation/v11b_metadata_correlation.png`

**Pitfalls:**
- Handle missing values: height and weight have 8–12% missing. Drop NaN rows for those specific correlations (document this).
- Cramér's V ranges from 0 to 1. For the chi-squared test to be valid, expected cell counts should generally be ≥ 5. Check this.
- Don't run this on the Unknown class alone (N=68 is small). Run on the full 3-class label.

---

### Task 2.5c — Feature Correlation: Spectral Frequency Bands

**Objective:** Analyse which frequency bands in the spectrogram have statistically different energy distributions between Present and Absent classes. This is the most important correlation analysis — it validates the 800 Hz feature selection decision and previews the XAI frequency-band analysis in Phase 4.

**Why it matters:** The paper crops at 800 Hz and claims frequencies above contain "no heart sound information." Our job is to verify this claim with data. Additionally, the supervisor's required DAV step "feature correlation" demands we show which features (here: frequency bands) are associated with the target variable before model training.

**Method:**

1. Compute the log-spectrogram (Steps A–B, before cropping) for all recordings in the dataset (or a representative sample — see note below).
2. For each of the 101 frequency bins (0 Hz to 2000 Hz in 20 Hz steps):
   - Extract the mean energy across time for each recording: `mean_energy[f] = mean(S_log[f, :])`
   - Split by patient-level murmur label: Present vs. Absent
   - Run a Mann-Whitney U test (non-parametric, doesn't assume normality)
   - Record the U statistic, p-value, and effect size (e.g., rank-biserial correlation or Cohen's d)
3. Apply Bonferroni correction for multiple comparisons (101 tests).

**Implementation note on scale:** Computing spectrograms for all 3163 recordings takes time. Two options:
- **Full dataset:** Process all recordings, aggregate at patient level. More rigorous. May take 5–15 minutes depending on implementation.
- **Sampled:** Use 100–200 recordings stratified by class. Faster, still statistically valid for this purpose.
- **Recommendation:** Full dataset. It's a one-time computation that you'll need for Phase 3 anyway, so build the feature extraction pipeline to handle all recordings now.

**Visualisation (V11c) — Per-frequency-band discrimination plot:**

A figure with two aligned panels:

**Panel 1 (top):** Mean spectral energy by class, per frequency band.
- x-axis: frequency (Hz), 0–2000
- y-axis: mean log energy
- Two lines: Present (red) and Absent (blue), with shaded confidence intervals (±1 SE or IQR)
- Vertical dashed line at 800 Hz

**Panel 2 (bottom):** Statistical significance per band.
- x-axis: frequency (Hz), 0–2000
- y-axis: -log10(p-value) from Mann-Whitney U test
- Horizontal dashed line at -log10(0.05/101) for Bonferroni threshold
- Colour bars green (significant) vs. grey (not significant)

**Expected outcome:** Significant differences concentrated in the 50–400 Hz range (where murmur energy overlaps with systolic content). Differences above 800 Hz should be minimal or non-significant, validating the cropping decision.

**Save to:** `figures/correlation/v11c_spectral_discrimination.png`

**Pitfalls:**
- Analyse at the *patient* level, not recording level, to avoid pseudo-replication (one patient with 4 recordings counted 4×). Either average across a patient's recordings first, or use a mixed-effects approach (overkill for DAV — just average).
- Exclude Unknown patients from this analysis — they're a data quality class, not a clinical class. The comparison should be Present vs. Absent.
- The Mann-Whitney U test is appropriate because spectral energy distributions are unlikely to be normal.
- Bonferroni correction is conservative — if some bands above 800 Hz are significant even after correction, that's a noteworthy finding (potential limitation of the 800 Hz cutoff).

---

### Task 2.5d — Feature Correlation: Recording-Level Features

**Objective:** Analyse the relationship between recording-level features and murmur class. These features are already computed from Phase 1 — this task just formalises the analysis with statistical tests and proper visualisation.

**Why it matters:** Completes the feature correlation picture. Some of these features (annotation coverage, SNR) are indirect signal quality indicators that connect to the Unknown class analysis in Phase 4.

**Features to analyse:**

| Feature | Source | Test |
|---|---|---|
| Recording duration (seconds) | `recordings.csv` or re-computed | Kruskal-Wallis (3-class) |
| Annotation coverage (%) | Phase 1 Task 1.10 | Kruskal-Wallis |
| SNR proxy (dB) | Phase 1 Task 1.10 | Kruskal-Wallis |
| Number of recordings per patient | `recordings.csv` grouped | Kruskal-Wallis |
| Number of annotated cardiac cycles | From .tsv segmentation files | Kruskal-Wallis |

**Visualisation (V11d):** Grouped box plots or violin plots, one subplot per feature, grouped by murmur class (Present / Unknown / Absent). Annotate with Kruskal-Wallis p-value and pairwise post-hoc results (Dunn's test) where significant.

**Expected outcomes:**
- Duration: no significant difference across classes (confirmed in Phase 1)
- Annotation coverage: significant difference — Unknown ≪ Present ≈ Absent
- SNR proxy: significant difference — Unknown < Absent < Present
- Number of recordings: significant — Unknown concentrates in low-recording patients
- Number of cardiac cycles: probably follows annotation coverage pattern

**Save to:** `figures/correlation/v11d_recording_features.png`

**Pitfalls:**
- Some of these analyses overlap with Phase 1 findings. That's OK — Phase 1 was exploratory (visual), this is confirmatory (statistical). The report should reference both.
- For patient-level features (number of recordings), each patient is one data point. For recording-level features (duration, annotation coverage), either aggregate per patient or note the unit of analysis.

---

### Task 2.6 — Compare Spectrograms: Normal vs. Murmur vs. Noisy (V10)

**Objective:** Produce the V10 visualisation — a 3-panel comparison of normalised spectrograms (after full preprocessing) for the 3 cross-phase examples.

**Why it matters:** This is the "what does processed data look like" figure. It shows the reader what the RNN will actually see as input and highlights the visual differences between classes that the model must learn to distinguish.

**What to produce:** 3 spectrograms side by side (or stacked), all with the same colour scale:
- Normal (2530_MV): regular pattern, clear S1/S2, quiet systole/diastole
- Murmur (9979_TV): additional energy filling systole between S1 and S2
- Unknown (9983_MV): irregular, possibly fragmented, low overall energy

**Use the fully preprocessed (z-scored, cropped) spectrograms** — this shows what the RNN sees.

**Annotations:** Mark S1, systole, S2, diastole regions on the murmur example (use segmentation .tsv if available, or mark approximate boundaries). Highlight the murmur energy band.

**Save to:** `figures/preprocessing/v10_spectrogram_comparison.png`

---

### Task 2.7 — Frequency Content Differences Between Classes (V11)

**Objective:** Produce the V11 visualisation — average power spectrum by class, showing where murmur energy is concentrated.

**Why it matters:** This is the simpler, more intuitive version of Task 2.5c. While 2.5c does formal statistical testing, V11 shows the overall spectral profile as a line plot that's easy to interpret. Both belong in the report but serve different purposes.

**Method:**
1. For all recordings (or a large sample), compute the mean spectral energy per frequency bin (before z-scoring, after log-spectrogram and cropping).
2. Average across recordings within each class (Present / Absent).
3. Plot as two overlapping line plots.

**What to produce — V11:**
- x-axis: frequency (Hz), 0–2000 (full range, to show what's above 800 Hz)
- y-axis: mean log spectral energy
- Lines: Present (red), Absent (blue), optionally Unknown (grey dashed)
- Vertical line at 800 Hz
- Shade the region where Present > Absent (murmur energy band)

**Save to:** `figures/preprocessing/v11_frequency_content.png`

**Difference from V11c:** V11 is a simple visual comparison (average energy curves). V11c adds formal statistical testing per band with significance markers. V11 goes in the preprocessing section of the report; V11c goes in the feature correlation section.

---

### Task 2.8 — Document Feature Extraction Parameters and DAV Justification

**Objective:** Create a comprehensive parameter table and a written narrative explaining *why* each parameter was chosen, framed in DAV terminology.

**What to produce:** A summary section in the notebook (which becomes the Methods section in the report) containing:

**Parameter Table:**

| Parameter | Value | Unit | Justification |
|---|---|---|---|
| Sampling rate | 4000 | Hz | Fixed by Littmann 3200 stethoscope |
| Normalisation | Zero-mean, peak-divide | — | Removes DC offset and amplitude variation between recordings |
| STFT window | Hann | — | Smooth tapering; standard for spectral analysis |
| Window length | 50 | ms (200 samples) | Matches ~half of S1/S2 duration (~100 ms); balances time and frequency resolution |
| Hop length | 20 | ms (80 samples) | 50 Hz feature rate; sufficient for heart sound event tracking |
| Frequency resolution | 20 | Hz | sr / n_fft = 4000/200 |
| Feature rate | 50 | Hz | 1 / hop_length_sec |
| Frequency range | 0–800 | Hz | Validated by spectral correlation analysis (Task 2.5c) |
| Frequency bins | 41 | — | (800/20) + 1 bins after cropping |
| Row normalisation | Per-row z-score | — | Equalises dynamic range across frequency bins; makes murmurs visible |

**DAV Terminology Mapping:**

| Preprocessing Step | DAV Category | Dimensionality Effect |
|---|---|---|
| Amplitude normalisation | Feature Normalisation | No change (1D → 1D) |
| Log-spectrogram (STFT) | Feature Transformation | 1D waveform → 2D time-frequency (N samples → 101 × T) |
| Frequency cropping | Feature Selection | Reduces from 101 to 41 frequency bins (59% reduction) |
| Per-row z-score | Feature Normalisation | No dimensionality change (41 × T → 41 × T) |

---

### Task 2.9 — Package into Reusable Feature Extraction Module

**Objective:** Combine all preprocessing steps into a single, clean, reusable function that will be called in Phase 3 for all recordings.

**Code location:** `src/features/spectrogram.py`

**Interface:**
```python
def extract_features(wav_path, sr=4000, win_length_sec=0.050, hop_length_sec=0.020,
                     max_freq=800):
    """
    Full feature extraction pipeline: load → normalise → spectrogram → crop → z-score.

    Parameters:
        wav_path: path to .wav file
        sr: sampling rate (must be 4000 for this dataset)
        win_length_sec: STFT window length in seconds
        hop_length_sec: STFT hop length in seconds
        max_freq: maximum frequency to retain (Hz)

    Returns:
        features: numpy array, shape (n_freq_bins, n_frames)
        times: time axis in seconds
        freqs: frequency axis in Hz
    """
```

**Also add a batch processing function:**
```python
def extract_features_batch(wav_paths, **kwargs):
    """Extract features for a list of recordings. Returns list of (features, times, freqs)."""
```

**Testing:** Run on the 3 cross-phase examples and verify:
- Output shape: (41, T) where T is consistent with duration / hop_length
- Output values: check mean ≈ 0, std ≈ 1 for each row (after z-score)
- Visual inspection: compare output spectrograms with those from Task 2.5a

---

### Task 2.10 — Pipeline Framework Diagram (V-pipe)

**Objective:** Create a visual diagram of the full pipeline showing every stage from raw audio to final murmur classification, with input/output data shapes annotated.

**Why it matters:** The supervisor explicitly requires a pipeline framework diagram. This becomes Figure 1 of the report and provides the reader with an overview before diving into details.

**What to produce:**

A block diagram showing the 4 main stages:

```
Raw PCG ──▶ Stage 1: Feature Extraction ──▶ Stage 2: RNN (Bi-GRU) ──▶ Stage 3: Parallel HSMM ──▶ Stage 4: Classification
(4kHz,int16)   (41 × T spectrogram)        (5-state posteriors)       (4 segmentations +          (Present/Unknown/Absent)
                                                                        4 confidences)
```

Inside the Feature Extraction box, expand the 4 sub-steps:
```
Raw (N,) → Normalise → STFT (101, T) → Crop (41, T) → Z-score (41, T)
```

**How to create:**
- Option 1: Matplotlib/manual drawing — full control, but tedious
- Option 2: Draw in a vector tool (draw.io, Figma) — faster for diagrams
- Option 3: Create in the notebook using matplotlib boxes/arrows — keeps everything in code
- **Recommendation:** Use whatever is fastest. This diagram is about clarity, not code elegance. A clean draw.io export as PNG/SVG is perfectly fine.

**Save to:** `figures/preprocessing/v_pipeline_framework.png` (or `.svg`)

---

## 5. Notebook Organisation

### Notebook 02a: `02a_preprocessing.ipynb`

```
Section 1: Data Quality Assessment (Task 2.0)
    - Load Phase 1 findings
    - Data Quality → Preprocessing Justification table

Section 2: Amplitude Normalisation (Task 2.1)
    - Implementation
    - Before/after visualisation (3 examples)

Section 3: Log-Spectrogram (Task 2.2)
    - Implementation
    - Full spectrogram visualisation (3 examples)
    - Discussion: why spectrogram? why these parameters?

Section 4: Frequency Cropping (Task 2.3)
    - Implementation
    - Full vs. cropped comparison

Section 5: Per-Row Z-Score (Task 2.4)
    - Implementation
    - Before/after z-score comparison (V12)

Section 6: Complete Pipeline Visualisation (Task 2.5a — V9)
    - 5-panel figure for each of the 3 examples

Section 7: Spectrogram Comparison by Class (Task 2.6 — V10)
    - 3-panel figure: Normal vs. Murmur vs. Unknown

Section 8: Frequency Content by Class (Task 2.7 — V11)
    - Average power spectrum plot

Section 9: Parameter Summary & DAV Justification (Task 2.8)
    - Parameter table
    - DAV terminology mapping

Section 10: Reusable Module (Task 2.9)
    - Package code into src/features/
    - Verification tests
```

### Notebook 02b: `02b_feature_correlation.ipynb`

```
Section 1: Metadata Feature Correlation (Task 2.5b)
    - Cramér's V / point-biserial calculations
    - V11b: Metadata correlation heatmap

Section 2: Spectral Frequency-Band Correlation (Task 2.5c)
    - Compute spectrograms for full dataset (or large sample)
    - Per-band Mann-Whitney U tests
    - V11c: Frequency-band discrimination plot

Section 3: Recording-Level Feature Correlation (Task 2.5d)
    - Load Phase 1 quality metrics
    - Kruskal-Wallis tests
    - V11d: Box/violin plots

Section 4: Synthesis — What Does Correlation Tell Us?
    - Connect correlation findings to preprocessing decisions
    - "Feature selection is justified because..."
    - Preview: how this connects to Phase 4 XAI (ablation comparison)
```

---

## 6. Visualisation Checklist

| V# | Description | Task | Save path |
|---|---|---|---|
| V9 | Preprocessing pipeline — 5 panels per example (×3 examples) | 2.5a | `figures/preprocessing/v9_pipeline_{normal,murmur,unknown}.png` |
| V10 | Spectrogram comparison by class | 2.6 | `figures/preprocessing/v10_spectrogram_comparison.png` |
| V11 | Frequency energy distribution by class | 2.7 | `figures/preprocessing/v11_frequency_content.png` |
| V11b | Metadata feature correlation heatmap | 2.5b | `figures/correlation/v11b_metadata_correlation.png` |
| V11c | Per-frequency-band discrimination plot | 2.5c | `figures/correlation/v11c_spectral_discrimination.png` |
| V11d | Recording-level features × class | 2.5d | `figures/correlation/v11d_recording_features.png` |
| V12 | Z-score before/after effect | 2.4 | `figures/preprocessing/v12_zscore_effect.png` |
| V-pipe | Pipeline framework diagram | 2.10 | `figures/preprocessing/v_pipeline_framework.png` |

**Total: 8 visualisations (some with multiple sub-figures).**

---

## 7. Common Pitfalls for Phase 2

| # | Pitfall | How to avoid |
|---|---|---|
| 1 | Using `librosa.load()` default behaviour — it resamples to 22050 Hz and normalises to [-1, 1] automatically | Use `scipy.io.wavfile.read()` for raw int16 access, or `librosa.load(sr=4000, mono=True)` with explicit sr |
| 2 | STFT parameter confusion between libraries | Stick to one library (scipy recommended). Double-check: noverlap = nperseg - hop_length |
| 3 | Off-by-one in frequency cropping | Verify `freqs[40] == 800.0` before cropping. Use `<=` not `<` for the boundary. |
| 4 | Z-scoring the wrong axis | Z-score along axis=1 (time), independently for each frequency row. NOT axis=0. |
| 5 | Computing spectrograms for all 3163 recordings is slow | Use vectorised operations. Expect 5–15 minutes. Consider saving processed spectrograms to `data/processed/` as .npy files for reuse in Phase 3. |
| 6 | Correlation analysis on recordings instead of patients | Aggregate features at the patient level first (mean across recordings) to avoid pseudo-replication. |
| 7 | Forgetting to exclude Unknown from Present-vs-Absent analysis | Unknown is a data quality class. Spectral correlation (Task 2.5c) should compare Present vs. Absent only. |
| 8 | Not handling missing values in metadata correlation | Height, weight have 8–12% NaN. Drop NaN per feature, document N used. |
| 9 | Making the pipeline framework diagram too complex | Keep it clean: 4 main stages, one expansion of Stage 1. Don't try to show every HSMM topology — that's Phase 3's diagram. |
| 10 | Spending too much time on diagram aesthetics | Clarity > beauty. A clean draw.io or matplotlib diagram is sufficient. Don't burn hours on this. |

---

## 8. Recommended Task Order

The tasks have dependencies. Here's the optimal execution order:

```
Task 2.0  (Justification table — documentation, no code needed)
   │
   ▼
Tasks 2.1 → 2.2 → 2.3 → 2.4  (Sequential preprocessing steps)
   │
   ▼
Task 2.5a  (Pipeline visualisation — needs 2.1–2.4 done)
   │
   ├──▶ Task 2.6 (Spectrogram comparison — uses outputs from 2.1–2.4)
   ├──▶ Task 2.7 (Frequency content — uses outputs from 2.2, before 2.3)
   ├──▶ Task 2.5b (Metadata correlation — independent, uses Phase 1 data)
   ├──▶ Task 2.5c (Spectral correlation — needs spectrograms from 2.2 for all recordings)
   └──▶ Task 2.5d (Recording-level correlation — mostly uses Phase 1 data)
   │
   ▼
Task 2.8  (Parameter documentation — after all analysis is done)
   │
   ▼
Task 2.9  (Package module — after all code is tested)
   │
   ▼
Task 2.10 (Pipeline diagram — can be done anytime, but best after understanding full pipeline)
```

**Tasks 2.5b, 2.5c, 2.5d, 2.6, 2.7 can be done in parallel** — they're independent analyses that all build on the preprocessing foundation from 2.1–2.4.

---

## 9. Files to Produce

| File | Type | Description |
|---|---|---|
| `notebooks/02a_preprocessing.ipynb` | Notebook | Preprocessing implementation + visualisation |
| `notebooks/02b_feature_correlation.ipynb` | Notebook | Correlation analysis (metadata, spectral, recording-level) |
| `src/features/spectrogram.py` | Module | Reusable feature extraction pipeline |
| `src/features/normalisation.py` | Module | Amplitude normalisation function |
| `figures/preprocessing/v9_*.png` | Figures | Pipeline step visualisations (×3) |
| `figures/preprocessing/v10_*.png` | Figure | Class comparison spectrograms |
| `figures/preprocessing/v11_*.png` | Figure | Frequency content by class |
| `figures/preprocessing/v12_*.png` | Figure | Z-score before/after |
| `figures/preprocessing/v_pipeline_*.png` | Figure | Pipeline framework diagram |
| `figures/correlation/v11b_*.png` | Figure | Metadata correlation heatmap |
| `figures/correlation/v11c_*.png` | Figure | Spectral band discrimination |
| `figures/correlation/v11d_*.png` | Figure | Recording-level feature × class |

**Optional but recommended:**

| `data/processed/spectrograms/` | Directory | Pre-computed spectrograms as .npy for Phase 3 reuse |

---

## 10. Connection to Later Phases

| Phase 2 Output | Used In | How |
|---|---|---|
| `src/features/spectrogram.py` | Phase 3 | RNN training data preparation — extract features for all 3163 recordings |
| V11c (spectral discrimination) | Phase 4, Task 4.5 | Compare pre-model correlation with post-model ablation importance |
| Justification table (Task 2.0) | Phase 6 | Direct content for report Preprocessing section |
| Pipeline diagram (Task 2.10) | Phase 6 | Report Figure 1 |
| Correlation findings | Phase 6 | Report Feature Correlation section (6.6) |
| Pre-computed spectrograms | Phase 3 | Load from `data/processed/` instead of recomputing |

---

## Progress Tracker

| Task | Status | Notes |
|---|---|---|
| 2.0 Data quality → preprocessing justification | ⬜ | |
| 2.1 Amplitude normalisation | ⬜ | |
| 2.2 Log-spectrogram | ⬜ | |
| 2.3 Frequency cropping | ⬜ | |
| 2.4 Per-row z-score | ⬜ | |
| 2.5a Pipeline visualisation (V9) | ⬜ | |
| 2.5b Metadata correlation (V11b) | ⬜ | |
| 2.5c Spectral correlation (V11c) | ⬜ | |
| 2.5d Recording-level correlation (V11d) | ⬜ | |
| 2.6 Spectrogram comparison (V10) | ⬜ | |
| 2.7 Frequency content (V11) | ⬜ | |
| 2.8 Parameter documentation | ⬜ | |
| 2.9 Reusable module | ⬜ | |
| 2.10 Pipeline diagram (V-pipe) | ⬜ | |
