# Phase 0 Summary — Project Setup Recap

**Project location:** `D:\1 Onedrive\1 Lectures\3.2 DAV\papers\heart-murmur-detection\`
**Environment name:** `heart-murmur` (conda, Python 3.10.20, CPU-only PyTorch)

> **Project scope:** This project reproduces and improves the **murmur detection task only** (3-class: Present/Unknown/Absent). The clinical outcome prediction task (CatBoost-based, binary Normal/Abnormal) from the baseline papers is **out of scope** — not reproduced, not improved. Explainability work (Section 5A of the main roadmap) replaces it as a first-class deliverable.

---

## How to Open and Run the Project

```bash
conda activate heart-murmur
cd /d "D:\1 Onedrive\1 Lectures\3.2 DAV\papers\heart-murmur-detection"
jupyter lab
```

Quick check if needed:
```bash
python --version          # Python 3.10.20
python -c "import torch; print(torch.__version__)"   # CPU build
```

---

## Phase 0 — What Was Done

| Task | Status |
|---|---|
| 0.1 Install Python environment | ✅ Done |
| 0.2 Download CirCor dataset | ✅ Done |
| 0.3 Clone baseline code (reference) | ✅ Done |
| 0.7 Create folder structure | ✅ Done |
| 0.4–0.6 Read papers + challenge page | ✅ Done (see summaries below) |

**Phase 0 is complete. Next: Phase 1 — Exploratory Data Analysis.**

---

## Project Folder Structure — What Each Folder Is For

The guiding principle: **each folder separates one type of artefact**, so that as the project grows across 6 phases, you can always find what you need in seconds instead of digging through a flat pile of files.

```
heart-murmur-detection/
│
├── data/
│   ├── raw/              ← The CirCor dataset, exactly as downloaded. NEVER edit these files.
│   │                        If you ever break something downstream, raw/ is your guaranteed
│   │                        clean restart point — you never need to re-download.
│   ├── processed/         ← Derived data: extracted spectrograms, computed features.
│   │                        Safe to delete entirely and regenerate from raw/ at any time.
│   └── metadata/          ← Parsed patient DataFrames, train/val/test fold splits.
│                             Keeps "what data goes in which fold" separate from the audio itself.
│
├── notebooks/             ← One notebook per roadmap phase, numbered in execution order:
│   ├── 01_eda.ipynb                      (Phase 1)
│   ├── 02_preprocessing.ipynb            (Phase 2)
│   ├── 03_model_reproduction.ipynb       (Phase 3)
│   ├── 04_analysis_explainability.ipynb  (Phase 4)
│   ├── 05_improvements.ipynb             (Phase 5)
│   └── 06_figures.ipynb                  (final figure polishing for the report)
│   This is your main visible "story" — a reader opens these in order and sees the whole project.
│
├── src/                   ← Reusable Python code, NOT duplicated across notebooks.
│   ├── data/               loading, parsing metadata, building CV splits
│   ├── features/            spectrogram extraction, normalisation
│   ├── models/               RNN, HSMM, Viterbi decoder, parallel HSMM wrapper
│   ├── evaluation/           murmur-task metrics (weighted accuracy), error analysis
│   └── visualisation/        plotting + explainability functions reused across notebooks
│   Why this exists: if you write a spectrogram function inside notebook 02, notebook 03 can't
│   reuse it without copy-pasting — which risks two diverging versions of the "same" function.
│   Defining it once in src/ and importing it everywhere keeps a single source of truth.
│
├── experiments/            ← The evidence trail for "every improvement must be justified
│   ├── configs/                experimentally" (your supervisor's hard requirement).
│   ├── logs/                Each experiment gets a config (what changed) and logged results
│   └── results/              (numbers), separate from the heavy model files and the images.
│
├── models/                 ← Saved RNN model checkpoints (.pt). No CatBoost — out of scope.
│   └── rnn/                  Kept separate because these are large binary files — easy to
│                              exclude from version control and won't clutter your code folders.
│
├── figures/                ← Final exported images, organised by the phase that produced them.
│   ├── eda/                  When writing the report, you go straight to figures/results/
│   ├── preprocessing/         instead of hunting through dozens of mixed images.
│   ├── results/
│   └── improvements/
│
├── report/                 ← The actual deliverable document — kept separate from working
│                              files so it's obvious what to hand in.
│
└── presentation/            ← Slide deck — same reasoning as report/.
```

### The short version, if you only remember one sentence per folder:
- `data/` = raw vs. derived data, never mix them
- `notebooks/` = the readable story, one file per phase
- `src/` = logic you don't want to copy-paste
- `experiments/` = numeric proof your improvements actually helped
- `models/` = trained weights, kept out of the way
- `figures/` = what goes into the report, pre-sorted
- `report/` & `presentation/` = what you actually submit

---

## Reading Summaries

> Note: the summaries below cover both tasks described in the papers (murmur detection + clinical outcome) for completeness of understanding — the two tasks share Stages 1–3 of the algorithm (feature extraction, RNN, parallel HSMM). Only the murmur detection task is reproduced/improved in this project; clinical outcome (CatBoost) content is marked accordingly.

### 1. PhysioNet Challenge 2022 (George B. Moody PhysioNet Challenge)

**What it is:** A competition organising 40 teams to build algorithms for two tasks on the same dataset — murmur detection and clinical outcome prediction — using PCG recordings from the CirCor DigiScope dataset (paediatric population, Brazil).

**Tasks and metrics:**
- *Murmur detection* (3-class: Present / Unknown / Absent) — judged by **weighted accuracy**, with weights 5 / 3 / 1 respectively. This heavily penalises missing a real murmur (false negative), reflecting the clinical cost of a missed diagnosis. **This is the task this project targets.**
- *Clinical outcome* (binary: Normal / Abnormal) — judged by a **custom cost function** modelling the financial/clinical cost of a screening pathway (lower is better). Also penalises false negatives heavily. *(Out of scope for this project.)*

**Data split:** 60% training (public, ~942 patients), 10% validation (hidden, repeated scoring during the competition), 30% test (hidden, one-time final scoring).

**Key finding from the challenge itself:** Rankings shifted significantly between validation and test sets. The abnormal-outcome prevalence differed substantially between validation (0.383) and test (0.507), meaning teams that over-tuned to the validation set were penalised on test. About half the teams (19/40) scored *worse than a random classifier* on the cost metric — a strong signal that overfitting to a small dataset was the dominant failure mode of the challenge, not algorithm sophistication.

**Why this matters for your project:** This is exactly why your reproduction target is the **cross-validated training set**, not an attempt to simulate the hidden test set — the cross-validation numbers are what the winning team itself reports as representative, and chasing the (inaccessible) test set would not be methodologically sound.

---

### 2. Top-Ranked Paper — CinC 2022 Conference Paper

**Citation:** McDonald, Gales, Agarwal. "Detection of Heart Murmurs in Phonocardiograms with Parallel Hidden Semi-Markov Models." *Computing in Cardiology* 2022.

**Result:** Murmur detection ranked 2nd/40 (0.776 weighted accuracy on test, only 0.004 below 1st — won 1st prize as the top-ranked team was disqualified for code issues) — **this project's reproduction target**. *(Clinical outcome ranked 1st/40 with cost score 11,144 on test — out of scope.)*

**Core method, in one sentence:** A bidirectional GRU converts a normalised log-spectrogram into 5-state posteriors (S1, S2, systole, diastole, murmur), and 4 parallel HSMMs — each assuming a different murmur timing pattern — decode this into both a heart-sound segmentation and a murmur confidence score. *(In the original papers, this confidence score then feeds a CatBoost model for clinical outcome — that branch is not part of this project.)*

**Why "parallel" HSMMs is the key idea:** Traditional segmentation HSMMs assume a healthy 4-state cycle (S1→systole→S2→diastole) and break when a loud murmur masks S1/S2. Instead of one model, this paper runs 4 HSMMs simultaneously — one normal, three with different murmur placements — and compares their confidences. The model with the best-fitting confidence determines both the segmentation **and** the murmur classification in a single step, rather than segmenting first and classifying second.

**Reported training cross-validation number (murmur task, your target):** weighted accuracy 0.798 — see Method Comparison Table below.

---

### 3. Journal Extension — PLOS Digital Health (2024)

**Citation:** McDonald, Gales, Agarwal. "A recurrent neural network and parallel hidden Markov model algorithm to segment and detect heart murmurs in phonocardiograms." *PLOS Digital Health* 3(11): e0000436, 2024.

**Relationship to the conference paper:** Same core algorithm, same authors, same competition entry — but expanded roughly 5x in length with full mathematical detail, deeper evaluation, and broader discussion. This is the primary reference for your reproduction; the conference paper is a good first read for orientation.

**What it adds beyond the conference paper:**
- Full equations for HSMM confidence scores: C(M) = max(C(ω₂), C(ω₃), C(ω₄)), C(N) = C(ω₁), and the final murmur likelihood C(M−N) = C(M) − C(N) — **core to this project's reproduction**
- *(Detailed CatBoost feature description for clinical outcome — out of scope)*
- Confusion matrices, ROC curves, reliability diagrams, and per-murmur-grade sensitivity breakdown (87.5% sensitivity for grade 1 quiet murmurs, rising to 100% for grades 2–3) — **all murmur-task, in scope**
- A dedicated section analysing the **entire PhysioNet 2022 challenge** results across all 40 teams, not just their own entry
- An explicit discussion of dataset and method limitations (see below)

**Limitations the authors themselves identify, filtered to murmur-detection relevance** (directly feeds your "propose improvements" and Explainability sections):
- The murmur detector only targets *audible* abnormal sounds — some signals may have inaudible time-frequency features the algorithm misses
- No distinction between pathological and innocent (benign) murmurs in the dataset's labels
- Diastolic murmurs (only 10 examples) are not modelled at all — insufficient data
- Sensitivity to the "Unknown" class is weak (only 30.9%) — flagged in the paper as a hard case
- **The paper's main interpretability tool is the HSMM confidence scatter plot, but it does not systematically analyse *which spectral/temporal features* drive high vs. low confidence — this is the gap this project's Explainability section (5A in the main roadmap) directly addresses**

*(Outcome-only limitations — weak specificity 31.3%, limited CatBoost feature set, custom cost function comparability — omitted as out of scope.)*

---

## Literature Review Table

| Source | Type | Year | Core Contribution | Dataset Used | Reported Result |
|---|---|---|---|---|---|
| PhysioNet Challenge 2022 (Reyna et al.) | Competition / benchmark definition | 2022 | Defines the two-task problem (murmur detection **[in scope]** + clinical outcome **[out of scope]**), the CirCor dataset split, and the weighted-accuracy / cost-function metrics | CirCor DigiScope (this project's dataset) | N/A — defines the evaluation framework, not a method |
| McDonald, Gales, Agarwal — CinC 2022 | Conference paper | 2022 | Introduces parallel HSMM + RNN segmentation-classification algorithm; winning challenge entry | CirCor DigiScope | Murmur: 0.776 weighted acc. (test), 2nd/40 **[reproduction target]**. *(Outcome: cost 11,144, 1st/40 — out of scope)* |
| McDonald, Gales, Agarwal — PLOS Digital Health | Journal extension | 2024 | Same algorithm, full mathematical derivation, extended cross-validation analysis, challenge-wide result analysis, explicit limitations discussion | CirCor DigiScope | Murmur: 0.798 weighted acc. (train CV) **[reproduction target]**. *(Outcome: cost 10,565 — out of scope)* |
| Springer et al. (referenced in both papers) | Prior segmentation method (2016 challenge baseline) | 2016 | Logistic regression + HSMM segmentation assuming healthy 4-state cycle; provided as the 2016 challenge baseline tool | PhysioNet 2016 dataset | State-of-the-art at the time; fails on strong-murmur recordings — the gap this 2022 paper's "parallel" HSMM design addresses |

---

## Method Comparison Table — Reported Numbers Across Splits (Murmur Task)

This table is your direct reproduction target reference. Your cross-validated training results should land close to the **Training** column. *(Outcome-task numbers from the original papers are omitted — out of scope.)*

| Metric | Training (public, your reproduction target) | Validation (hidden) | Test (hidden, final) |
|---|---|---|---|
| Murmur weighted accuracy | 0.817 (CinC) / 0.798 (journal CV) | 0.758 | 0.776 |
| Murmur ranking | — | — | 2nd / 40 |
| Murmur sensitivity (Present class) | 92.7% | — | — |
| Murmur PPV (Present class) | 55.0% | — | — |
| Murmur sensitivity (Absent class) | 77.6% | — | — |
| Murmur PPV (Absent class) | 93.1% | — | — |
| Murmur sensitivity (Unknown class) | 30.9% | — | — |
| Macro F1 (murmur task) | 0.621 | — | — |
| AUC-ROC (binary, unknowns removed) | 0.947 | — | — |

**Reading note:** the journal paper's cross-validated training number (0.798) differs slightly from the conference paper's single training-set number (0.817) because cross-validation averages performance across 5 held-out folds rather than evaluating on the same data the model was fitted to — this is the more honest and more appropriate number for you to target.

---

## Explainability (XAI) Approach — Not SHAP/LIME

A natural question: since the supervisor requires "interpret how features influence predictions," why isn't this project using SHAP or LIME?

**Why not SHAP/LIME:** Both are *post-hoc* explainability methods — designed to approximate feature importance for a black-box model that takes a flat feature vector in and produces a single score out (SHAP uses Shapley values from game theory; LIME fits a local linear approximation). They fit tabular models like CatBoost very well. CatBoost is out of this project's scope, so there is no black-box tabular model to apply them to.

**What this project uses instead — interpretability by design:** The RNN + parallel HSMM pipeline produces clinically meaningful intermediate outputs at every stage — the RNN's per-frame state posteriors and the HSMM's decoded segmentation path are not hidden embeddings, they are direct, visualisable answers to "what does the model think is happening right now." Explaining the model means visualising these intermediate representations directly, rather than approximating them after the fact. This is sometimes called *intrinsic* or *glass-box* interpretability, as opposed to the *post-hoc/black-box* interpretability that SHAP and LIME represent.

**Concrete techniques used (see Section 5A of the main roadmap for full detail):**
- RNN posterior traces overlaid on the waveform — direct visualisation of model belief over time
- Parallel HSMM path comparison — structural explanation from the model's own probabilistic decoding, not an approximation
- Per-frequency-band ablation (occlude a frequency range, measure confidence shift) — an occlusion-based feature importance technique, conceptually the same family as SHAP/LIME but applied directly in the signal domain instead of a flat feature vector
- Confidence-vs-grade calibration analysis — assesses whether output probabilities are trustworthy, a separate but related strand of explainable/trustworthy ML

**One-line summary for the report:** *"This project uses interpretability-by-design rather than post-hoc explainability (SHAP/LIME), since the RNN+HSMM architecture's intermediate outputs are already clinically interpretable representations; frequency-band ablation is used as a complementary occlusion-based importance technique."*

---

## Next Steps

Phase 0 is fully complete. Moving to **Phase 1: Exploratory Data Analysis** — parsing patient metadata, visualising the murmur class distribution, and inspecting raw PCG signals.
