# Heart Murmur Detection from PCG Signals — Complete Project Roadmap

**Project Type:** University DAV Course Project (Data Analysis & Visualization)
**Baseline:** McDonald et al. (2022/2024) — PhysioNet Challenge 2022 Winner
**Dataset:** CirCor DigiScope (PhysioNet Challenge 2022)
**Code Repository:** https://github.com/am2234/parallel-hsmm-murmur
**Development Machine:** Windows 64-bit, Intel Core Ultra 5 125H, 32GB RAM, Intel Arc Graphics, ~361GB free storage
**Development Strategy:** Local (CPU-only PyTorch — sufficient for the small GRU model)

---

## 1. Project Requirements Analysis

### 1.1 The Machine Learning Problem

**Inputs:**
- Phonocardiogram (PCG) audio recordings at 4000 Hz from a Littmann 3200 electronic stethoscope
- Recordings from up to 4 auscultation locations per patient (aortic, pulmonic, tricuspid, mitral)
- Patient metadata: age, sex, weight, height, pregnancy status (used only as potential auxiliary features for murmur detection, not for outcome prediction)

**Output — Single Task (Project Scope):**
**Murmur Detection (3-class):** Per-patient classification as *Murmur Present*, *Murmur Absent*, or *Unknown*

**Evaluation Metric:**
- Weighted accuracy (weights: Present=5, Unknown=3, Absent=1) — penalises missed murmurs

> **Scope decision:** This DAV project focuses exclusively on the **murmur detection task**. The clinical outcome prediction task (CatBoost-based, binary Normal/Abnormal) described in the baseline papers is explicitly **out of scope** — not reproduced, not improved upon. This keeps the project focused, allows deeper explainability and visualization work on a single task, and matches the DAV course's emphasis on depth of analysis over breadth of tasks. The papers are still read in full since the murmur detection pipeline and the outcome pipeline share Stages 1–3 (feature extraction, RNN, parallel HSMM) — only CatBoost (Stage 4's separate outcome branch) is excluded.

### 1.2 DAV Course Requirements Mapping

| Course Requirement | Where It Maps in This Project |
|---|---|
| Problem definition (inputs/outputs) | Section 1.1 above; murmur detection task formulation |
| Data preprocessing & discussion | Data quality assessment (Task 2.0) → justified preprocessing: amplitude normalisation, spectrogram extraction, frequency cropping, z-score normalisation. Each step framed as Input → Algorithm → Output with intermediate visualisation. |
| Feature correlation | Phase 2 Tasks 2.5b–2.5d: metadata correlation heatmap, per-frequency-band discrimination analysis (Present vs. Absent), recording-level feature × class analysis |
| Feature extraction / feature transformation | Feature transformation: raw PCG → log-spectrogram (time-frequency representation). Feature selection: frequency cropping 0–800 Hz (justified by spectral correlation analysis). Dimensionality reduction: RNN compresses 41-dim spectrogram → 5-state posteriors. |
| Feature influence on predictions / Explainability (XAI) | Dedicated Section 5A — confidence-score interpretation, segmentation overlay explanations, per-frequency-band importance (ablation + correlation comparison), error case studies |
| Data analysis & visualization emphasis | EDA of dataset, signal visualisation, segmentation plots, ROC curves, confusion matrices, explainability visuals |
| Pipeline framework | Pipeline framework diagram (Task 2.10) with input/output shapes annotated at each stage — becomes Figure 1 in report |

### 1.3 Supervisor's Requirements Mapping

| Supervisor Requirement | Strategy |
|---|---|
| Study best-performing solutions | Analyse CinC 2022 conference paper + PLOS Digital Health journal extension |
| Reproduce the methodology | Re-implement the murmur detection pipeline (Stages 1–4) using the public GitHub code as reference |
| Analyse strengths and limitations | Systematic evaluation documented in Phases 3–4 |
| Interpret how features influence predictions (explainability) | Dedicated **Section 5A** — global/local/error-based explanation levels, concrete techniques, deliverables checklist |
| Propose improvements | Justified experiments in Phase 5, scoped to murmur detection |
| Attempt to outperform | Phase 6 — only after verified reproduction |
| Pipeline framework diagram with Input → Algorithm → Output per step | Task 2.10: annotated pipeline diagram (V-pipe). Report Section 6.4 walks through each step with intermediate results. |
| Data observations → Preprocessing justification (logical chain) | Task 2.0: structured table mapping EDA findings to preprocessing decisions. Report Section 6.5 follows "observation → problem → solution" pattern. |
| Feature correlation analysis | Tasks 2.5b–2.5d: metadata, spectral, and recording-level correlation analysis with visualisations (V11b, V11c, V11d). Report Section 6.6. |
| Feature selection / feature transformation | Phase 2 tasks reframed with DAV terminology: transformation (spectrogram), selection (frequency cropping), normalisation (z-score). Each justified by correlation analysis. |
| Visualise and explain each step with intermediate results | Every Phase 2 task produces Input → Output visualisation. V9 shows full 4-stage intermediate chain. |

---

## 2. Dataset Analysis

### 2.1 Overview

| Property | Value |
|---|---|
| Source | CirCor DigiScope dataset, paediatric population, Brazil (2014–2015) |
| Total patients | 1568 encounters from 1452 unique patients |
| Total recordings | 5268 PCGs |
| Public training set | 942 encounters (60%) |
| Validation set | ~157 encounters (10%, hidden) |
| Test set | ~469 encounters (30%, hidden) |
| Recording device | Littmann 3200, fixed 4000 Hz sampling rate |
| Auscultation locations | Aortic (AV), Pulmonic (PV), Tricuspid (TV), Mitral (MV) |
| Recording duration | Variable (seconds to tens of seconds) |

### 2.2 Labels and Annotations

**Per-Patient Labels:**
- Murmur status: Present (179), Unknown (68), Absent (695) in training set — **this project's target label**
- Demographics: age, sex, weight, height, pregnancy status
- *(Clinical outcome label exists in the dataset but is out of scope for this project — not used)*

**Per-Recording Annotations:**
- Heart sound segmentation labels (S1, systole, S2, diastole boundaries)
- Murmur location (which auscultation sites)
- Murmur timing (early-systolic, mid-systolic, late-systolic, holosystolic)
- Murmur grade (1=quiet, 2=moderate, 3=loud)
- Murmur shape, pitch, quality characteristics

### 2.3 Critical Dataset Characteristics

- **Class imbalance:** Only ~19% of patients have murmurs; Unknown class is very small (7.2%) — this is the central challenge for the murmur detection task
- **Almost exclusively systolic:** 304 systolic murmurs vs. only 10 diastolic — diastolic detection not feasible
- **Single annotator:** All labels from one clinician — inherent subjectivity
- **No pathological vs. innocent distinction:** Murmur labels don't differentiate between clinically significant and benign murmurs
- **Pregnant subset:** 110 patients self-reported pregnancy, but ages not specified
- **Small dataset:** Deep learning approaches struggled to generalise; feature engineering was crucial
- *(Note: validation/test prevalence mismatch was a known issue for the clinical outcome task in the original challenge — not directly relevant since that task is out of scope here)*

### 2.4 File Structure (per patient)

Each patient folder contains:
- `{patient_id}.txt` — metadata header file (demographics, labels)
- `{patient_id}_{location}.wav` — PCG audio file(s) for each auscultation location
- `{patient_id}_{location}.tsv` — segmentation annotations (S1/systole/S2/diastole boundaries)
- `{patient_id}_{location}.hea` — WFDB header file

---

## 3. Baseline Paper Analysis

### 3.1 Conference Paper (CinC 2022)

**Title:** "Detection of Heart Murmurs in Phonocardiograms with Parallel Hidden Semi-Markov Models"
**Authors:** McDonald, Gales, Agarwal (University of Cambridge)
**Team:** CUED Acoustics

**Key Results:**
- Murmur detection: Weighted accuracy 0.776 (test), ranked 2nd/40 (0.004 below 1st; 1st was disqualified → won 1st prize) — **this project's reproduction target**
- *(Clinical outcome: Cost score 11,144 (test), ranked 1st/40 — out of scope for this project)*

**Core Innovation:** Instead of a single HSMM that assumes only healthy heart sounds, the algorithm runs 4 parallel HSMMs each making different assumptions about what the signal contains. The confidence comparison between these models simultaneously produces both a segmentation and a murmur classification.

### 3.2 Journal Extension (PLOS Digital Health, 2024)

The journal paper adds substantial content beyond the conference paper:
- Detailed mathematical formulations for all algorithm stages
- Extended literature review and clinical context
- In-depth cross-validation results with confusion matrices, ROC curves, reliability diagrams
- CatBoost feature analysis and clinical outcome discussion
- Analysis of the PhysioNet 2022 challenge results across all teams
- Discussion of dataset limitations and future directions

### 3.3 The 4-Stage Algorithm Pipeline

**Stage 1 — Feature Extraction:**
- Amplitude normalisation (zero-mean, peak-normalised)
- Log-spectrogram: Hann window, 50 ms length, 20 ms step → 20 Hz frequency resolution, 50 Hz feature rate
- Crop to 0–800 Hz (above this is noise, not heart sounds)
- Per-frequency-row z-score normalisation (reduces dynamic range between murmurs and S1/S2)

**Stage 2 — RNN Prediction:**
- 3-layer bidirectional GRU (hidden size 60)
- 2-layer fully connected head (sizes 60→40) with Tanh activations
- Softmax output → 5-state posteriors: {S1, S2, systole, diastole, murmur}
- Trained with cross-entropy loss, inversely weighted by class frequency
- Dropout 0.1, Adam optimiser, 5-fold stratified cross-validation
- Murmur ground-truth: approximate from clinician timing labels (e.g. early-systolic → first 50% of each systole)

**Stage 3 — Parallel HSMM Segmentation:**
- 4 HSMMs with different topologies:
  - ω₁: Normal (4 states, murmur posterior discarded)
  - ω₂: Holosystolic murmur (murmur replaces systole)
  - ω₃: Early-systolic murmur (5 states: S1→murmur→systole→S2→diastole)
  - ω₄: Mid-systolic murmur (5 states: S1→systole→murmur→S2→diastole)
- State durations estimated from heart rate (autocorrelation of non-diastolic RNN posteriors)
- Springer duration-dependent Viterbi algorithm for decoding
- Per-model segmentation confidence: C(ω) = mean RNN posterior along Viterbi path

**Stage 4 — Classification (murmur detection — in scope):**
- Murmur likelihood: C(M−N) = max(C(ω₂), C(ω₃), C(ω₄)) − C(ω₁)
- Signal quality estimate: C(ω̂) = max confidence across all 4 models
- Per-patient aggregation: if any recording → murmur, predict "Present"; if low confidence → "Unknown"; else "Absent"

**Clinical Outcome Prediction — OUT OF SCOPE for this project:**
- *(The original papers add a separate CatBoost gradient boosted decision tree at this point, using per-location murmur likelihood, signal quality, age, pregnancy status, and recording count to predict a binary Normal/Abnormal clinical outcome. This project does not reproduce or improve this branch — documented here only for completeness of understanding the source papers.)*

### 3.4 Reported Benchmark Numbers (Training Cross-Validation) — Murmur Task

| Metric | Murmur Task (this project's target) |
|---|---|
| Micro-averaged accuracy | 0.771 |
| Challenge metric (weighted accuracy) | 0.798 |
| Macro F1 | 0.621 |
| AUC-ROC (binary, unknowns removed) | 0.947 |
| Sensitivity (Present class) | 92.7% |
| Specificity (Absent class) | 77.6% |

*(Clinical outcome metrics from the original papers are omitted here — out of scope.)*

### 3.5 Strengths Identified from the Papers

1. **Lightweight & interpretable** — no end-to-end black box; each component has a clear role
2. **Robust to noise** — HSMM enforces physically valid state transitions; noise alone can't cause false positives
3. **Dual output** — simultaneous segmentation and classification from the same model
4. **Signal quality estimation** — built-in mechanism to flag unreliable recordings
5. **Minimal overfitting** — training and test scores closely matched, unlike most competing teams
6. **Clinically meaningful** — murmur localisation (timing + location) provides interpretable output
7. **Domain-informed design** — feature extraction choices grounded in cardiac acoustics

### 3.6 Limitations Identified from the Papers (Murmur Detection Focus)

1. **No diastolic murmur detection** — only 10 diastolic examples; murmur state only models systolic
2. **Approximate murmur labels** — ground-truth murmur localisation is heuristic (e.g. "first 50% of systole")
3. **Single annotator** — inherent label noise from subjective clinician assessment
4. **No pathological vs. innocent murmur distinction** — dataset limitation, not method limitation
5. **RNN observes frames independently from HSMM perspective** — while the RNN models temporal context, the HSMM treats each frame's posterior independently
6. **No data augmentation** — small dataset, no augmentation strategies reported
7. **Heart rate estimation** — autocorrelation can fail on noisy/arrhythmic recordings
8. **Low PPV on "Unknown" class** — only 30.9% sensitivity for Unknown, suggesting this class is hard to characterise from confidence scores alone
9. **Limited explainability beyond confidence scores** — the original papers show the HSMM confidence scatter plot as their main interpretability tool but do not systematically analyse *which spectral/temporal features* drive high vs. low confidence — **this is a gap this project's Explainability section (5A) directly addresses**

*(Items related solely to the clinical outcome task — e.g. CatBoost feature set limitations, outcome specificity — are omitted as out of scope.)*

---

## 4. What Must Be Reproduced

### 4.1 Mandatory Reproduction Targets

These must be working before any improvement attempts. **Scope: murmur detection pipeline only (Stages 1–4 of the algorithm). The clinical outcome / CatBoost branch is not reproduced.**

| Component | Target | Verification |
|---|---|---|
| Feature extraction pipeline | Log-spectrogram with exact parameters | Visual comparison with paper's Fig 2 |
| RNN training | 5-fold CV, same hyperparameters | Murmur weighted accuracy ≈ 0.798 on training |
| Parallel HSMM decoding | 4 parallel models with correct topologies | Segmentation examples matching paper's Fig 5 |
| Murmur classification | C(M−N) thresholding | Confusion matrix matching Table 1 |

### 4.2 Acceptable Reproduction Tolerance

Since you cannot access the hidden test set, your reproduction target is the cross-validated training set result. A reproduction is considered successful if:
- Murmur weighted accuracy is within ±0.03 of 0.798
- Per-class sensitivities match the reported values within ±3 percentage points

### 4.3 Key Code Reference

The authors' repository (https://github.com/am2234/parallel-hsmm-murmur) contains:
- Training and inference scripts
- HSMM implementation
- RNN model definition
- Results generation scripts
- Pre-computed model outputs on the training set

**Important:** You should study this code to understand the implementation, but your reproduction should be your own implementation with documented understanding — not a blind copy-paste.

---

## 5. Potential Improvements

**Scope: all improvements below target the murmur detection pipeline (RNN + parallel HSMM + classification threshold). Nothing here touches a clinical outcome model, since that task is out of scope.**

These are ranked by feasibility, expected impact, and alignment with DAV requirements.

### 5.1 High Feasibility / High DAV Value

| # | Improvement | Rationale | Risk |
|---|---|---|---|
| 1 | **Threshold optimisation study** — systematic analysis of the C(M−N) decision threshold for murmur classification | Paper uses a fixed threshold (C(M−N)=0) chosen for the weighted-accuracy metric; exploring the sensitivity/specificity trade-off space is excellent for DAV and directly explainable via the confidence scatter plot | Low |
| 2 | **Data augmentation** — time stretching, noise injection, pitch shifting on PCG signals before RNN training | Not used in baseline; could improve RNN robustness given small dataset, especially for the under-represented Unknown class | Low–Medium |
| 3 | **Per-murmur-grade and per-location error analysis** — systematic breakdown of where the murmur classifier fails | Paper reports grade-level sensitivity (87.5% grade 1 → 100% grade 2/3) but doesn't deeply analyse *why* quiet murmurs are missed; this connects directly to the Explainability section | Low |
| 4 | **"Unknown" class deep-dive** — investigate what signal characteristics drive the C(ω̂) signal-quality threshold and why Unknown sensitivity is only 30.9% | Paper itself flags this as a weak point of the murmur task; addressing it is a natural, scoped improvement | Low |

### 5.2 Medium Feasibility / Good DAV Value

| # | Improvement | Rationale | Risk |
|---|---|---|---|
| 5 | **RNN architecture swap** — replace GRU with Transformer encoder or 1D-CNN for the segmentation/murmur model | Modern architectures may capture temporal dependencies better | Medium — need to re-tune |
| 6 | **Mel-spectrogram or MFCC features** — alternative to linear log-spectrogram as RNN input | Standard in audio ML; could capture perceptually relevant information differently than the linear log-spectrogram | Low–Medium |
| 7 | **Auxiliary metadata as RNN/HSMM input** — incorporate age, weight at the murmur-classification stage (not just as a downstream outcome feature) | Murmur acoustics are known to vary with patient age/size; testing this directly in the murmur task (rather than only in the now-excluded outcome model) is a novel, in-scope contribution | Medium |

### 5.3 Lower Feasibility / High Risk

| # | Improvement | Rationale | Risk |
|---|---|---|---|
| 8 | **End-to-end model** — bypass HSMM entirely, classify murmur directly from RNN output | Contradicts the paper's core insight that HSMMs prevent overfitting | High — likely worse on small data |
| 9 | **Pre-trained audio models** — fine-tune AudioSet/wav2vec for PCG murmur classification | Transfer learning from speech/music domains | High — domain gap; overfitting risk |
| 10 | **Diastolic murmur modelling** — add ω₅/ω₆ HSMM topologies for diastolic murmurs | Only 10 examples; statistically meaningless | Very High — not enough data |

### 5.4 Recommended Improvement Strategy

Start with improvements #1 and #4 (threshold optimisation + Unknown-class analysis). These are:
- Low risk — they don't break the reproduction
- High DAV value — they generate rich visualisations and directly support the Explainability section
- Scientifically meaningful — they address acknowledged limitations of the murmur task specifically
- Experimentally verifiable — clear before/after comparison

Then attempt #2 (augmentation) and #3 (error analysis) if time permits.

---

## 5A. Explainability & Feature Interpretation Plan (XAI)

This section directly addresses your supervisor's requirement to **interpret how features influence predictions** and the DAV course's emphasis on explainability — not just accuracy. It is treated as a first-class deliverable, not an afterthought tacked onto Phase 4.

### 5A.1 Why Explainability Matters Here

A clinical screening tool that cannot explain *why* it flagged a patient is far less useful to a clinician than one that can point to evidence. The baseline papers already lean toward interpretability by design (HSMM segmentation over an end-to-end black box), which gives you a strong foundation — your job is to make that interpretability *visible and analysed*, not just claimed.

### 5A.2 Three Levels of Explanation

**Level 1 — Global: What does the model attend to, in general?**
- Which frequency bands in the spectrogram carry the most discriminative information between murmur and normal signals?
- Which heart-cycle phase (systole vs. diastole vs. transition regions) most influences the murmur posterior?
- Across the whole training set, what does the distribution of C(M−N) confidence look like, and where is the decision boundary positioned relative to it?

**Level 2 — Local: Why was *this specific* recording classified this way?**
- For any single recording, overlay the RNN's 5-state posterior probabilities directly on the waveform and spectrogram — this is inherently interpretable because the model's internal belief about S1/S2/systole/diastole/murmur is visible frame-by-frame, not hidden in a black box.
- Show the Viterbi-decoded segmentation path for all 4 parallel HSMMs side-by-side for one recording, so a reader can see *why* model ω₃ (early-systolic) won over ω₁ (normal) for a given confidence score.
- Highlight the specific time window the algorithm identifies as the murmur region.

**Level 3 — Error-based: Why did the model get specific cases wrong?**
- For false negatives (missed murmurs): are they concentrated in low murmur-grade cases, specific auscultation locations, or noisy recordings?
- For false positives: do they correlate with low signal quality C(ω̂), or with specific non-murmur sounds (breathing, talking, movement artefacts) that resemble murmur spectral content?
- Case-study format: pick 3–5 representative misclassified recordings and walk through the full pipeline output (spectrogram → RNN posteriors → HSMM segmentation → confidence score) to narrate exactly where the breakdown occurred.

### 5A.3 Feature Influence Analysis — Concrete Techniques

| Technique | What it explains | Applies to |
|---|---|---|
| HSMM confidence scatter plot (C(M) vs. C(N)) | Global separability between murmur/normal; the model's overall decision geometry | Level 1 |
| Per-frequency-band energy comparison (murmur vs. normal, averaged across dataset) | Which frequencies the spectrogram-based RNN is implicitly relying on | Level 1 |
| RNN posterior probability traces over time, overlaid on waveform | Frame-by-frame "attention" — fully interpretable since these are direct model outputs | Level 2 |
| Parallel HSMM path comparison (4 Viterbi paths side-by-side) | Why a specific murmur timing class was chosen over "normal" | Level 2 |
| Confidence vs. murmur grade scatter/box plot | Whether the model's confidence is well-calibrated to clinical severity | Level 1/3 |
| Permutation-style ablation on spectrogram frequency bands (zero out a band, measure confidence shift) | Causal feature importance — does removing this frequency range change the prediction? | Level 1 |
| Confusion matrix stratified by murmur grade / location / signal quality | Structured error analysis | Level 3 |
| Case studies of misclassified recordings (full pipeline walkthrough) | Concrete, narrated local explanations | Level 3 |

**Note on SHAP:** SHAP is well-suited to tabular models like CatBoost, which is now out of scope. For this project's RNN+HSMM pipeline, the more appropriate and more clinically meaningful explainability tools are the ones above — since the model's intermediate outputs (segmentation states, per-model confidences) are *already* interpretable representations, unlike a black-box embedding. This is one of the paper's own stated advantages and is worth stating explicitly in your report.

### 5A.4 Where This Lives in the Roadmap

- **Phase 4 (Analysis & Explainability)** is the primary home for this work — Level 1 and Level 2 explanations, confidence analysis, error case studies
- **Phase 5 (Improvements)** ties back into this — e.g., the threshold optimisation study (#1) is itself an explainability-driven improvement, since it's grounded in understanding the confidence distribution from 5A.3
- **Phase 6 (Report)** — a dedicated "Explainability & Feature Analysis" section is required, separate from the raw results section, to satisfy the supervisor's explicit requirement

### 5A.5 Explainability Deliverables Checklist

- [ ] Global confidence scatter plot with decision boundary annotated
- [ ] Per-frequency-band importance analysis (murmur vs. normal)
- [ ] At least 5 local case-study walkthroughs (mix of correct and incorrect predictions)
- [ ] Parallel HSMM path comparison figure for at least one ambiguous case
- [ ] Confidence-vs-grade calibration plot
- [ ] Stratified error analysis (by grade, location, signal quality)
- [ ] Written narrative connecting each visualization to a clinical or methodological insight (a figure alone is not an explanation — the report must say what it means)

---

## 6. Milestone-Based Project Plan

### Phase 0: Foundation (Week 1)
**Goal:** Set up environment, understand dataset, understand papers thoroughly.

**Deliverables:**
- Working Python environment with all dependencies
- Downloaded and verified dataset
- Annotated reading notes for both papers
- This project roadmap document

**Required Knowledge:** Python environment management, basic signal processing concepts, reading ML papers
**Estimated Effort:** 8–12 hours
**Expected Outputs:** Environment ready, dataset on disk, paper summaries

---

### Phase 1: Exploratory Data Analysis (Weeks 1–2)
**Goal:** Deeply understand the data before writing any ML code. This is the foundation of your DAV grade.

**Deliverables:**
- Dataset statistics report
- Class distribution visualisations
- Signal quality assessment
- Demographic analysis
- Recording-level and patient-level analysis
- At least 8–10 publication-quality visualisations

**Required Knowledge:** pandas, matplotlib/seaborn, librosa/scipy for audio, basic statistics
**Estimated Effort:** 15–20 hours
**Expected Outputs:** EDA notebook, figures folder, data quality report

---

### Phase 2: Signal Preprocessing, Feature Extraction & Correlation Analysis (Weeks 2–3)
**Goal:** Implement the feature extraction pipeline exactly as described in the paper, frame each step using DAV terminology (feature transformation / selection / normalisation), justify every preprocessing decision with data observations from Phase 1, and analyse feature correlations before model training. Visualise every step with intermediate results.

**DAV Narrative Flow (supervisor's required pattern):**
1. **Data Quality Assessment** — summarise data observations from Phase 1 that motivate preprocessing
2. **Preprocessing & Feature Transformation** — for each step: Input → Algorithm → Output → Visualise intermediate result
3. **Feature Correlation Analysis** — analyse which features discriminate between classes *before* model training
4. **Feature Selection Justification** — connect correlation results back to preprocessing decisions (e.g., "frequency bands above 800 Hz show no class discrimination → cropping is justified")

**Deliverables:**
- Data quality assessment → preprocessing justification table (observation → problem → solution)
- Feature transformation: amplitude normalisation module
- Feature transformation: log-spectrogram computation module
- Feature selection: frequency cropping (0–800 Hz), justified by spectral analysis
- Feature normalisation: per-row z-score normalisation
- Visualisation of each preprocessing step on example recordings (intermediate results shown)
- Feature correlation analysis: metadata features × murmur label, spectral frequency bands × class, recording-level features × class
- Comparison of normal vs. murmur vs. noisy spectrograms
- Pipeline framework diagram (input/output annotated at every stage)
- Reusable feature extraction module

**Required Knowledge:** DSP fundamentals (STFT, windowing, spectrograms), NumPy, SciPy, correlation analysis (Cramér's V, point-biserial, Mann-Whitney U)
**Estimated Effort:** 16–20 hours (expanded from 12–15 to include correlation analysis)
**Expected Outputs:** Preprocessing pipeline, feature extraction notebook, correlation analysis notebook, spectrogram visualisations, pipeline diagram

---

### Phase 3: Model Reproduction (Weeks 3–5)
**Goal:** Reproduce the full pipeline and verify against published numbers.

**Sub-phase 3a — RNN Training (Week 3–4):**
- Implement bidirectional GRU with FC head
- Create ground-truth segmentation labels (including murmur approximation)
- Train with 5-fold stratified CV
- Verify RNN predictions visually against paper's figures

**Sub-phase 3b — HSMM Implementation (Week 4):**
- Implement heart rate estimation (autocorrelation method)
- Implement Springer's duration-dependent Viterbi algorithm
- Implement 4 parallel HSMM topologies
- Compute segmentation confidences

**Sub-phase 3c — Murmur Classification & Verification (Week 4–5):**
- Implement murmur classification from HSMM confidences
- Run full 5-fold CV evaluation
- Compare results against published murmur-task benchmarks

**Required Knowledge:** PyTorch, HMM/HSMM theory, Viterbi algorithm, cross-validation
**Estimated Effort:** 25–32 hours (largest phase; reduced from original estimate since CatBoost/outcome branch is out of scope)
**Expected Outputs:** Trained RNN + HSMM models, reproduction results table, reproduction verification report

---

### Phase 4: Analysis & Explainability (Weeks 5–6)
**Goal:** Deep analysis of the reproduced murmur detection model — this is where you earn DAV marks. Follow the full plan in **Section 5A** for explainability work.

**Deliverables:**
- Confusion matrix (murmur task)
- ROC curve with AUC
- Per-class sensitivity/specificity breakdown
- HSMM confidence scatter plot (reproducing paper's Fig 4) — global explainability
- Segmentation quality examples (reproducing paper's Fig 5) — local explainability
- Per-frequency-band importance analysis
- Parallel HSMM path comparison for ambiguous cases
- Error analysis: false positive / false negative case studies (Level 3 explainability)
- Reliability diagram
- Per-murmur-grade and per-location performance breakdown

**Required Knowledge:** scikit-learn metrics, matplotlib advanced, error analysis methodology, basic signal/frequency interpretation
**Estimated Effort:** 18–22 hours (expanded to accommodate explainability depth)
**Expected Outputs:** Analysis notebook, explainability report (Section 5A checklist), 12+ additional visualisations

---

### Phase 5: Improvements (Weeks 6–7)
**Goal:** Implement and evaluate justified improvements to the murmur detection pipeline.

**Deliverables:**
- Threshold optimisation analysis (sensitivity/specificity trade-off)
- Per-grade and per-location error analysis (ties into Section 5A)
- Data augmentation experiments (if time permits)
- RNN input feature variant comparison — e.g. Mel-spectrogram vs. log-spectrogram (if time permits)
- Every improvement backed by before/after comparison

**Required Knowledge:** Feature engineering, hyperparameter search, ablation studies
**Estimated Effort:** 15–20 hours
**Expected Outputs:** Improvement experiments notebook, results comparison table, improvement justification report

---

### Phase 6: Report & Presentation (Week 7–8)
**Goal:** Produce a professional DAV report and presentation following the supervisor's required pattern.

**Report Structure (follows supervisor's required DAV flow):**
1. Introduction — problem definition, motivation, project scope
2. Literature Review — baseline paper analysis, related work
3. Dataset Description — data quality observations, class imbalance, signal characteristics
4. Pipeline Framework — diagram + per-step Input → Algorithm → Output description
5. Preprocessing & Feature Transformation — data observations → justified preprocessing, intermediate results, DAV terminology
6. Feature Correlation Analysis — metadata, spectral, recording-level correlation findings; justification for feature selection decisions
7. Model Methodology — RNN training, parallel HSMM, classification logic (reproduction)
8. Results — reproduction metrics + improvement experiments, intermediate results at each pipeline stage
9. Explainability & Feature Interpretation (XAI) — Section 5A deliverables, correlation-vs-ablation comparison
10. Discussion — strengths, limitations, improvement justification
11. Conclusion

**Deliverables:**
- Written report (structure above)
- Presentation slides
- All code cleaned and documented
- Reproducibility instructions

**Required Knowledge:** Academic writing, LaTeX/Word, presentation design
**Estimated Effort:** 15–20 hours
**Expected Outputs:** Final report, slide deck, code repository

---

## 7. Task Breakdown Structure

### Phase 0 Tasks
- [ ] 0.1 Install Python 3.9+, PyTorch, NumPy, SciPy, pandas, matplotlib, seaborn, librosa (CatBoost/SHAP not required — clinical outcome task out of scope)
- [ ] 0.2 Download CirCor dataset from PhysioNet
- [ ] 0.3 Clone baseline code repository (reference only)
- [ ] 0.4 Read CinC 2022 conference paper — annotate
- [ ] 0.5 Read PLOS Digital Health journal paper — annotate
- [ ] 0.6 Read PhysioNet Challenge 2022 description page
- [ ] 0.7 Create project folder structure

### Phase 1 Tasks
- [ ] 1.1 Parse all patient metadata files into a DataFrame
- [ ] 1.2 Compute dataset-level statistics (patients, recordings, locations, durations)
- [ ] 1.3 Visualise murmur class distribution
- [ ] 1.4 Visualise demographic distributions (age, sex, weight, height)
- [ ] 1.5 Analyse recordings per patient (how many locations per patient?)
- [ ] 1.6 Analyse recording durations distribution
- [ ] 1.7 Analyse murmur characteristics (timing, grade, location, shape)
- [ ] 1.8 Load and plot raw PCG waveforms (normal, murmur, noisy)
- [ ] 1.9 Visualise segmentation annotations on waveforms
- [ ] 1.10 Compute and visualise signal quality proxies (SNR estimates, amplitude statistics)
- [ ] 1.11 Create EDA summary notebook

### Phase 2 Tasks
- [ ] 2.0 Data quality assessment → preprocessing justification table (map Phase 1 observations to preprocessing steps: e.g., "variable amplitude → normalisation", "high-freq noise → 800 Hz cropping", "dynamic range disparity → z-score")
- [ ] 2.1 Feature Transformation: Amplitude normalisation (zero-mean, peak-normalised)
- [ ] 2.2 Feature Transformation: Log-spectrogram (STFT with Hann window, 50 ms, 20 ms step → time-frequency representation)
- [ ] 2.3 Feature Selection: Frequency cropping (0–800 Hz) — justified by spectral analysis showing no discriminative content above 800 Hz
- [ ] 2.4 Feature Normalisation: Per-row z-score normalisation (reduces dynamic range between murmurs and S1/S2)
- [ ] 2.5a Visualise each preprocessing step sequentially on example recordings (Input → Output at each stage, showing intermediate results)
- [ ] 2.5b Feature Correlation — Metadata: compute correlation (point-biserial / Cramér's V) between patient metadata features (age, sex, weight, height, pregnancy, recording count) and murmur label; visualise as annotated heatmap (V11b)
- [ ] 2.5c Feature Correlation — Spectral: compute per-frequency-band mean energy for Present vs. Absent; Mann-Whitney U test per band; visualise as frequency-band discrimination plot (V11c); compare against 800 Hz cutoff decision
- [ ] 2.5d Feature Correlation — Recording-level: correlate duration, annotation coverage, SNR proxy, number of recordings/patient with murmur class; visualise as grouped box/violin plots (V11d)
- [ ] 2.6 Compare spectrograms: normal vs. murmur vs. noisy
- [ ] 2.7 Visualise frequency content differences between classes (average power spectrum by class)
- [ ] 2.8 Document feature extraction parameters and choices (with DAV justification: which step is transformation, which is selection, why)
- [ ] 2.9 Package into reusable feature extraction module
- [ ] 2.10 Create pipeline framework diagram: Raw PCG → [Preprocessing] → [Feature Extraction] → [RNN] → [Parallel HSMM] → [Classification], with input/output shapes and data dimensions annotated at each stage — becomes Figure 1 in report

### Phase 3 Tasks
- [ ] 3.1 Create modified segmentation ground-truth (5-state with murmur)
- [ ] 3.2 Implement data loading / batching for variable-length sequences
- [ ] 3.3 Implement bidirectional GRU + FC head architecture
- [ ] 3.4 Implement class-weighted cross-entropy loss
- [ ] 3.5 Implement 5-fold stratified cross-validation split
- [ ] 3.6 Train RNN, log loss curves
- [ ] 3.7 Visualise RNN predictions on example recordings
- [ ] 3.8 Implement heart rate estimation via autocorrelation
- [ ] 3.9 Implement state duration distributions (Springer method)
- [ ] 3.10 Implement duration-dependent Viterbi algorithm
- [ ] 3.11 Implement 4 HSMM topologies (ω₁–ω₄)
- [ ] 3.12 Compute segmentation confidences C(ω)
- [ ] 3.13 Implement murmur classification logic
- [ ] 3.14 Implement per-patient aggregation rules
- [ ] 3.15 Evaluate murmur detection (weighted accuracy, confusion matrix)
- [ ] 3.16 Compare all metrics against published murmur-task benchmarks
- [ ] 3.17 Write reproduction verification report
- *(CatBoost / clinical outcome tasks removed — out of scope for this project)*

### Phase 4 Tasks
- [ ] 4.1 Generate full confusion matrix (murmur task)
- [ ] 4.2 Plot ROC curve with AUC value
- [ ] 4.3 Reproduce HSMM confidence scatter plot (Fig 4 from journal) — global explainability
- [ ] 4.4 Create segmentation visualisation examples — local explainability
- [ ] 4.5 Per-frequency-band importance analysis (murmur vs. normal spectral energy) — **compare post-model frequency importance (ablation) against pre-model frequency correlation (from Task 2.5c): do the model's important frequencies match the statistically discriminative ones? This strengthens the XAI narrative by showing consistency between data-driven and model-driven feature importance.**
- [ ] 4.6 Parallel HSMM path comparison figure (4 Viterbi paths side-by-side, ≥1 ambiguous case)
- [ ] 4.7 Analyse false positives and false negatives
- [ ] 4.8 Build 5 local case-study walkthroughs (mix of correct/incorrect predictions)
- [ ] 4.9 Per-murmur-grade performance analysis
- [ ] 4.10 Per-auscultation-location performance analysis
- [ ] 4.11 Create reliability/calibration diagram for murmur confidence
- [ ] 4.12 Analyse "Unknown" class — what makes signals unclassifiable?
- [ ] 4.13 Write strengths/limitations analysis
- [ ] 4.14 Write explainability narrative connecting each figure to an insight (Section 5A.5 checklist)

### Phase 5 Tasks
- [ ] 5.1 Threshold optimisation: sweep C(M−N) threshold, plot sensitivity/specificity trade-offs
- [ ] 5.2 Stratified analysis: which grades/locations benefit most from threshold changes?
- [ ] 5.3 (Optional) Implement data augmentation for RNN training (noise, time-stretch, pitch-shift)
- [ ] 5.4 (Optional) Train RNN with Mel-spectrogram input, compare to log-spectrogram baseline
- [ ] 5.5 (Optional) Test auxiliary metadata (age, weight) as additional RNN/HSMM input
- [ ] 5.6 Create before/after comparison tables
- [ ] 5.7 Statistical significance testing (if applicable)
- [ ] 5.8 Write improvement justification report

### Phase 6 Tasks
- [ ] 6.1 Write report: Introduction (problem definition, motivation, project scope)
- [ ] 6.2 Write report: Literature Review
- [ ] 6.3 Write report: Dataset Description (data quality observations, class imbalance, signal characteristics)
- [ ] 6.4 Write report: Pipeline Framework (diagram + per-step Input → Algorithm → Output description)
- [ ] 6.5 Write report: Preprocessing & Feature Transformation (data observations → justified preprocessing steps, intermediate results visualised, DAV terminology: transformation/selection/normalisation)
- [ ] 6.6 Write report: Feature Correlation Analysis (metadata correlation, spectral band discrimination, recording-level features; findings that justify feature selection decisions)
- [ ] 6.7 Write report: Model Methodology (RNN training, parallel HSMM, classification logic — reproduction)
- [ ] 6.8 Write report: Results (reproduction metrics + improvement experiments, intermediate results at each pipeline stage)
- [ ] 6.9 Write report: Explainability & Feature Interpretation (XAI) (Section 5A deliverables: global/local/error-based explanations, correlation-vs-ablation comparison)
- [ ] 6.10 Write report: Discussion (strengths, limitations, correlation→ablation consistency, improvement justification)
- [ ] 6.11 Write report: Conclusion
- [ ] 6.12 Compile all figures at publication quality
- [ ] 6.13 Create presentation slides
- [ ] 6.14 Clean and document all code
- [ ] 6.15 Write README with reproducibility instructions
- [ ] 6.16 Final review and submission

---

## 8. Recommended Folder Structure

```
heart-murmur-detection/
│
├── README.md                          # Project overview, setup, reproducibility
├── requirements.txt                   # Python dependencies
├── environment.yml                    # Conda environment (alternative)
│
├── data/
│   ├── raw/                           # Downloaded CirCor dataset (gitignored)
│   │   ├── training_data/
│   │   └── ...
│   ├── processed/                     # Extracted features, spectrograms
│   └── metadata/                      # Parsed patient DataFrames, splits
│
├── notebooks/
│   ├── 01_eda.ipynb                   # Exploratory Data Analysis
│   ├── 02a_preprocessing.ipynb        # Feature extraction walkthrough (transformation, selection, normalisation)
│   ├── 02b_feature_correlation.ipynb  # Feature correlation analysis (metadata, spectral, recording-level)
│   ├── 03_model_reproduction.ipynb    # Training and evaluation
│   ├── 04_analysis_explainability.ipynb  # Model analysis, explainability, errors
│   ├── 05_improvements.ipynb          # Improvement experiments
│   └── 06_figures.ipynb               # Final publication-quality figures
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py                  # Data loading utilities
│   │   ├── parser.py                  # Metadata parsing
│   │   └── splits.py                  # Cross-validation splits
│   ├── features/
│   │   ├── __init__.py
│   │   ├── spectrogram.py             # Log-spectrogram extraction
│   │   └── normalisation.py           # Signal normalisation
│   ├── models/
│   │   ├── __init__.py
│   │   ├── rnn.py                     # Bidirectional GRU model
│   │   ├── hsmm.py                    # HSMM implementation
│   │   ├── viterbi.py                 # Duration-dependent Viterbi
│   │   └── parallel_hsmm.py           # 4 parallel HSMM wrapper
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py                 # Challenge metrics (weighted accuracy)
│   │   └── analysis.py                # Error analysis utilities
│   └── visualisation/
│       ├── __init__.py
│       ├── signals.py                 # Waveform and spectrogram plotting
│       ├── results.py                 # Confusion matrices, ROC, etc.
│       └── explainability.py          # Confidence scatter, HSMM path comparison, frequency-band importance plots
│
├── experiments/
│   ├── logs/                          # Training logs
│   ├── configs/                       # Experiment configurations
│   └── results/                       # Saved metrics, comparison tables
│
├── models/                            # Saved model checkpoints
│   └── rnn/                           # RNN checkpoints only — no CatBoost (out of scope)
│
├── figures/                           # Final publication-quality figures
│   ├── eda/
│   ├── preprocessing/
│   ├── correlation/                   # Feature correlation analysis figures
│   ├── results/
│   └── improvements/
│
├── report/
│   ├── report.pdf                     # Final report
│   ├── report.tex / report.docx       # Report source
│   └── references.bib                 # Bibliography
│
└── presentation/
    └── slides.pptx / slides.pdf       # Presentation
```

---

## 9. Experiment Tracking Plan

### 9.1 What to Track

For every experiment, record:

**Configuration:**
- Experiment ID (e.g., `exp_001_baseline_rnn`)
- Date and time
- Changed parameters vs. baseline
- Random seeds used

**Training:**
- Loss curves (train + validation per fold)
- Training time
- Convergence epoch

**Results:**
- Per-fold metrics (all 5 folds)
- Mean ± standard deviation across folds
- Per-class breakdown (sensitivity, PPV, F1)
- Challenge metrics (weighted accuracy, cost score)

**Artefacts:**
- Model checkpoints (best per fold)
- Predictions on validation fold
- Confusion matrices
- Figures

### 9.2 Tracking Strategy

For a university project, a simple approach is best:

1. **Experiments spreadsheet** — a single CSV/Excel tracking all experiments with columns for key metrics
2. **Config files** — one YAML/JSON per experiment in `experiments/configs/`
3. **Structured naming** — `exp_{NNN}_{short_description}`
4. **Git commits** — tag each major experiment

Do NOT use MLflow/Weights & Biases unless you already know them — learning a tracking tool is not worth the time investment for this project.

### 9.3 Key Experiments to Run

**Scope: murmur detection task only.**

| Exp ID | Description | Phase |
|---|---|---|
| exp_001 | Baseline reproduction — exact paper parameters | 3 |
| exp_002 | Reproduction verification — compare vs. published murmur-task benchmarks | 3 |
| exp_003 | Per-frequency-band ablation (explainability) | 4 |
| exp_004 | Threshold sweep — murmur task sensitivity/specificity trade-off | 5 |
| exp_005 | Data augmentation — noise injection | 5 |
| exp_006 | Data augmentation — time stretching | 5 |
| exp_007 | Mel-spectrogram vs. log-spectrogram input comparison | 5 |
| exp_008 | Auxiliary metadata (age/weight) as RNN/HSMM input | 5 |

---

## 10. Visualization Plan (DAV Requirements)

This is critical for your grade. Every visualisation should tell a story.

### 10.1 EDA Visualisations (Phase 1)

| # | Visualisation | Purpose | Plot Type |
|---|---|---|---|
| V1 | Murmur class distribution | Show class imbalance | Bar chart with counts + percentages |
| V2 | Age distribution by murmur class | Explore demographic patterns | Overlapping histograms or violin plots |
| V3 | Recordings per patient distribution | Understand data structure | Histogram |
| V4 | Recording duration distribution | Characterise signal length variability | Histogram with class overlay |
| V5 | Murmur location heatmap | Which auscultation sites have murmurs? | Heatmap or bar chart |
| V6 | Murmur grade vs. timing cross-tabulation | Understand murmur characteristics | Stacked bar or heatmap |
| V7 | Raw PCG waveforms | Show what the data looks like | Time-series plot (3 examples: normal, murmur, noisy) |
| V8 | Annotated segmentation overlay | Show ground-truth labels on signal | Waveform with colored regions |

### 10.2 Preprocessing & Feature Correlation Visualisations (Phase 2)

| # | Visualisation | Purpose | Plot Type |
|---|---|---|---|
| V9 | Preprocessing pipeline steps | Show transformation at each stage (Input → Output) | 4-panel figure (raw → normalised → spectrogram → z-scored) |
| V10 | Spectrogram comparison by class | Visual differences between classes | 3-panel spectrograms |
| V11 | Frequency energy distribution | Justify 800 Hz cutoff | Average power spectrum by class |
| V11b | Metadata feature correlation heatmap | Show (lack of) correlation between demographics and murmur label | Annotated heatmap (Cramér's V / point-biserial) |
| V11c | Per-frequency-band discrimination plot | Which spectral bands distinguish Present vs. Absent? Justify feature selection | Bar chart or heatmap with significance markers |
| V11d | Recording-level feature × class analysis | Correlate duration, coverage, SNR with murmur class | Grouped box plots or violin plots |
| V12 | Z-score effect | Show dynamic range reduction | Before/after spectrogram comparison |
| V-pipe | Pipeline framework diagram | Full pipeline overview with I/O shapes at each stage — Figure 1 in report | Annotated block diagram |

### 10.3 Model Results & Explainability Visualisations (Phase 4)

| # | Visualisation | Purpose | Plot Type |
|---|---|---|---|
| V13 | RNN state predictions | Show 5-state output on example signals | Stacked area or line plot below waveform |
| V14 | HSMM confidence scatter | Core result — murmur vs. normal separation (global explainability) | 2D scatter with class colours (reproduce Fig 4) |
| V15 | Confusion matrix — murmur task | Classification performance | Annotated heatmap |
| V16 | ROC curve — murmur task (binary) | Discrimination ability | ROC with AUC annotation |
| V17 | Reliability diagram | Calibration assessment | Calibration plot |
| V18 | Per-grade sensitivity | Performance vs. murmur severity | Bar chart |
| V19 | Per-frequency-band importance | Which frequencies drive murmur detection (explainability) | Bar chart or spectral overlay |
| V20 | Parallel HSMM path comparison | Why one murmur-timing model wins over another (local explainability) | Multi-panel state-sequence plot, 4 paths side-by-side |
| V21 | Local case-study walkthroughs | Full pipeline explanation for individual recordings | Multi-panel (waveform + spectrogram + RNN posteriors + HSMM path + confidence) |
| V22 | Confidence vs. murmur grade | Calibration of confidence to clinical severity (explainability) | Scatter or box plot |
| V23 | Segmentation examples | Show algorithm output | Multi-panel (waveform + RNN + HSMM segmentation) |
| V24 | Training loss curves | Convergence and overfitting check | Line plot per fold |

### 10.4 Improvement Visualisations (Phase 5)

| # | Visualisation | Purpose | Plot Type |
|---|---|---|---|
| V25 | Sensitivity/specificity trade-off | Threshold analysis | Dual-axis or parameterised curve |
| V26 | Before/after ROC comparison | Improvement verification | Overlaid ROC curves |
| V27 | Stratified improvement breakdown | Which grades/locations benefited from each improvement | Grouped bar chart |
| V28 | Final results comparison table | Summary of all experiments | Formatted table (for report) |

---

## 11. Risk Assessment and Mitigation Plan

### 11.1 Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| **HSMM implementation is complex** — Viterbi with duration constraints is non-trivial | High | High | Study Springer's original paper and the baseline code carefully. Start with a simpler 4-state HSMM before adding murmur states. Budget extra time. |
| **Reproduction doesn't match published numbers** | Medium | High | Accept ±5% tolerance. Document deviations honestly. Common causes: random seed, implementation details, label processing differences. |
| **RNN training instability / overfitting** | Medium | Medium | Use exact hyperparameters from paper. Monitor per-fold validation loss. Apply early stopping if needed. |
| **Computational resources insufficient** | Low | Medium | The RNN is small (3-layer GRU, hidden=60). Should train on any modern laptop GPU or even CPU in reasonable time. |
| **Dataset download or format issues** | Low | Low | Download early. Verify file counts and structure match documentation. |

### 11.2 Project Management Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| **Time underestimation for HSMM phase** | High | High | Start Phase 3 early. The HSMM is the hardest part. If stuck >5 days, use the authors' code as closer reference (document this). |
| **Spending too much time on improvements without verified reproduction** | Medium | High | Hard rule: NO improvements until Phase 3 is verified. Improvements without a working baseline are scientifically meaningless. |
| **Over-engineering improvements** | Medium | Medium | Each improvement should be implementable in <1 day. If it takes longer, it's too complex for this project. |
| **Visualisation quality too low for DAV** | Medium | High | Start EDA early (Phase 1). Use consistent styling. Set up a plotting style/theme once and reuse. |
| **Report writing takes too long** | Medium | Medium | Write incrementally — document each phase as you complete it. Don't leave everything for the final week. |

### 11.3 Fallback Strategy

If the full HSMM reproduction proves too difficult:

**Fallback Option A (Partial Reproduction):**
- Reproduce the feature extraction and RNN training fully
- For the HSMM, use the authors' pre-computed outputs from their `final_model/` directory
- Focus your original contribution on the Explainability (Section 5A) and error-analysis phases instead
- Document clearly what was reproduced vs. what was borrowed

**Fallback Option B (Simplified Pipeline):**
- Use Springer's original (simpler) HSMM segmentation as a baseline
- Focus on a traditional feature extraction → classifier pipeline
- This is still a valid DAV project but loses the "reproducing the winner" angle

Both fallbacks still allow you to meet all DAV course requirements.

---

## 12. Summary: Critical Path and Dependencies

```
Phase 0 (Foundation)
    │
    ▼
Phase 1 (EDA) ──────────────────────────────────────── [Can write report intro]
    │
    ▼
Phase 2 (Feature Extraction) ───────────────────────── [Can write report methods §1]
    │
    ▼
Phase 3 (Reproduction) ← CRITICAL GATE ──────────────── [Must verify before proceeding]
    │
    ├── Verified? YES ──▶ Phase 4 (Analysis) ──▶ Phase 5 (Improvements)
    │                                                       │
    └── Verified? NO ───▶ Debug / Fallback                  │
                                                            ▼
                                                    Phase 6 (Report + Presentation)
```

**The single most important rule: Phase 3 must be verified before Phase 5 begins.**

---

## Progress Tracker

> **Scope change log:**
> - **v1 (initial):** Project scope narrowed to murmur detection only. Explainability & Feature Interpretation Plan (Section 5A) added.
> - **v2 (2026-06-22):** Roadmap revised to align with supervisor's DAV report requirements. Changes: (1) Added feature correlation analysis tasks (2.5b–2.5d) with visualisations V11b–V11d — this was a missing DAV step. (2) Reframed Phase 2 tasks with DAV terminology (feature transformation / selection / normalisation). (3) Added Task 2.0: data quality assessment → preprocessing justification table. (4) Added Task 2.10: pipeline framework diagram. (5) Updated Phase 6 report structure to follow supervisor's required pattern (Input → Algorithm → Output per step, data observations → justified preprocessing, feature correlation → selection justification). (6) Connected Phase 4 ablation analysis to Phase 2 correlation analysis for XAI narrative consistency. (7) Phase 1 marked COMPLETE with task-level details.

### Phase 0: Foundation — Status: ✅ COMPLETE (7/7 tasks done)
Started: 2026-06-21 — Completed: 2026-06-21
Project path: `D:\1 Onedrive\1 Lectures\3.2 DAV\papers\heart-murmur-detection\`
Environment: conda `heart-murmur`, Python 3.10.20, CPU-only PyTorch

| Task | Status | Notes |
|---|---|---|
| 0.1 Install Python environment | ✅ Done | Conda env `heart-murmur` created, Python 3.10.20, torch (CPU), numpy, scipy, pandas, librosa, catboost, shap, jupyterlab all installed and verified. *(Note: catboost/shap were installed before the project scope was narrowed to murmur detection only — harmless to keep, just unused.)* |
| 0.2 Download CirCor dataset | ✅ Done | Extracted to `data/raw/training_data/` + LICENSE, RECORDS, SHA256SUMS, training_data.csv |
| 0.3 Clone baseline code (reference) | ✅ Done | github.com/am2234/parallel-hsmm-murmur cloned for reference |
| 0.4 Read CinC 2022 paper | ✅ Done | Summary + literature review table in phase0_summary.md |
| 0.5 Read PLOS Digital Health paper | ✅ Done | Summary + method comparison table in phase0_summary.md |
| 0.6 Read PhysioNet Challenge page | ✅ Done | Task definitions, metrics, data split documented |
| 0.7 Create folder structure | ✅ Done | Full project skeleton created under heart-murmur-detection/ |

**See `phase0_summary.md` for full reading summaries, literature review table, and method comparison table.**

### Phase 1: EDA — Status: ✅ COMPLETE (11/11 tasks done)
Started: 2026-06-21 — Completed: 2026-06-22
See `phase1_summary_eng.md` for detailed findings.

| Task | Status | Notes |
|---|---|---|
| 1.1 Parse metadata | ✅ Done | `patients.csv` (942×24), `recordings.csv` (3163×8). Duplicate-location naming handled. |
| 1.2 Dataset statistics | ✅ Done | 942 patients, 3163 recordings, median 4 rec/patient, median 21.5s duration, 8–12% missing demographics. |
| 1.3 Murmur class distribution (V1) | ✅ Done | Absent 73.8%, Present 19.0%, Unknown 7.2%. |
| 1.4 Demographic distributions (V2) | ✅ Done | Paediatric population. Murmur prevalence stable ~19% across age/sex. |
| 1.5 Recordings per patient (V3) | ✅ Done | 62.4% have 4 recordings. Unknown concentrates in low-recording patients. |
| 1.6 Recording durations (V4) | ✅ Done | Median ~21s. Duration nearly identical across classes. |
| 1.7 Murmur characteristics (V5, V6) | ✅ Done | Near-exclusively systolic. Grade 1 = 58%. Holosystolic 57%. |
| 1.8 Raw PCG waveforms (V7) | ✅ Done | 3 examples: Normal (2530_MV), Murmur (9979_TV), Unknown (9983_MV). |
| 1.9 Segmentation annotations (V8) | ✅ Done | State 0 (Unannotated) discovered — large gaps in Unknown recordings. |
| 1.10 Signal quality proxies | ✅ Done | Unknown median coverage 24%, SNR 5.65 dB. Confirms Unknown = signal quality issue. |
| 1.11 EDA summary notebook | ✅ Done | Narrative flow, key findings, example IDs for cross-phase tracking. |
### Phase 2: Feature Extraction — Status: ⬜ NOT STARTED
### Phase 3: Reproduction — Status: ⬜ NOT STARTED
### Phase 4: Analysis — Status: ⬜ NOT STARTED
### Phase 5: Improvements — Status: ⬜ NOT STARTED
### Phase 6: Report — Status: ⬜ NOT STARTED

---

## Appendix: Key References

1. McDonald A, Gales MJF, Agarwal A. "Detection of Heart Murmurs in Phonocardiograms with Parallel Hidden Semi-Markov Models." CinC 2022. DOI: 10.22489/CinC.2022.020
2. McDonald A, Gales MJF, Agarwal A. "A recurrent neural network and parallel hidden Markov model algorithm to segment and detect heart murmurs in phonocardiograms." PLOS Digital Health 3(11): e0000436, 2024.
3. Reyna MA et al. "Heart Murmur Detection from Phonocardiogram Recordings: The George B. Moody PhysioNet Challenge 2022." PhysioNet, 2023.
4. Oliveira J et al. "The CirCor DigiScope dataset: from murmur detection to murmur classification." IEEE JBHI, 2021.
5. Springer DB, Tarassenko L, Clifford GD. "Logistic Regression-HSMM-Based Heart Sound Segmentation." IEEE TBME, 2016.
6. Clifford GD et al. "Recent advances in heart sound analysis." Physiological Measurement, 2017.
7. Authors' code: https://github.com/am2234/parallel-hsmm-murmur
