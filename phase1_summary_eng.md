# Phase 1 Summary — Exploratory Data Analysis

**Project:** Heart Murmur Detection from PCG Signals
**Dataset:** CirCor DigiScope (PhysioNet Challenge 2022)
**Completed:** Phase 1 — 11/11 tasks done
**Artifacts:** `notebooks/01_eda.ipynb`, `data/metadata/`, `figures/eda/`

---

## Task Overview

| Task | What was done | Key result |
|------|--------------|------------|
| 1.1 | Parse metadata into 2 clean DataFrames | `patients.csv` (942×24), `recordings.csv` (3163×8). Discovered duplicate-location patients (e.g. 49748 with AV×2, PV×2) — naming convention `{id}_{loc}_{n}.wav` handled correctly. |
| 1.2 | Compute dataset-level statistics | 942 patients, 3163 recordings, median 4 recordings/patient, median duration 21.5s, missing data 8–12% for demographics. 2 collection campaigns (CC2014: 385, CC2015: 557). |
| 1.3 | Visualise murmur class distribution (V1) | Severe class imbalance: Absent 73.8%, Present 19.0%, Unknown 7.2%. Justifies weighted accuracy metric. |
| 1.4 | Visualise demographic distributions (V2) | Paediatric population (76.5% Child). Murmur prevalence stable ~19% across age/sex — demographics not strong predictors. |
| 1.5 | Analyse recordings per patient (V3) | 62.4% have 4 recordings. Unknown class concentrates in patients with fewer recordings (25.8% at 1 rec vs 2.2% at 4 rec). |
| 1.6 | Analyse recording durations (V4) | Median ~21s (15–42 cardiac cycles). Duration nearly identical across 3 classes — not a confounding variable. |
| 1.7 | Analyse murmur characteristics (V5, V6) | Near-exclusively systolic (178/179). Grade 1 = 58% of murmurs. 3 dominant timings: Holosystolic 57%, Early-systolic 33%, Mid-systolic 10%. |
| 1.8 | Plot raw PCG waveforms (V7) | 3 representative examples selected: Normal (2530_MV), Murmur (9979_TV), Unknown (9983_MV). Visual differences stark — murmur fills systole, unknown is irregular. |
| 1.9 | Overlay segmentation annotations (V8) | Ground-truth S1/Systole/S2/Diastole labels visible on waveforms. State 0 (Unannotated) discovered — large gaps especially in Unknown recordings. |
| 1.10 | Compute signal quality proxies | Annotation coverage: Unknown median 24% vs Present 60%. SNR proxy: Unknown median 5.7 dB vs Present 9.5 dB. Confirms Unknown = signal quality issue. |
| 1.11 | Finalise EDA notebook | Narrative flow with interpretation after every plot. Key findings summary. Example IDs documented for cross-phase tracking. |

---

## Detailed Key Findings

### 1. Dataset Scale and Structure

The CirCor DigiScope dataset contains **942 patient encounters** with a total of **3163 PCG recordings**, collected from paediatric patients at screening clinics in Paraíba, Brazil, across two campaigns (CC2014 with 385 patients and CC2015 with 557 patients). Recordings were made using a Littmann 3200 electronic stethoscope at a fixed sampling rate of 4000 Hz. *(Task 1.2)*

Each patient has recordings from **up to 5 auscultation locations**: the four standard cardiac valve positions (Aortic/AV, Pulmonic/PV, Tricuspid/TV, Mitral/MV) plus an occasional non-standard position (Phc). 62.4% of patients were recorded at all 4 standard locations (median = 4 recordings per patient), while 36.3% have only 1–3 recordings — meaning the model's patient-level aggregation logic must handle incomplete location sets. A small number of patients (13) have 5 or 6 recordings because the same location was recorded multiple times (e.g. patient 49748 has AV×2, PV×2, TV, MV). *(Tasks 1.1, 1.2, 1.5)*

Recording durations range from 5.15s to 64.51s, with a median of ~21.5s. At a typical paediatric heart rate of 80–120 bpm, this provides ~28–42 cardiac cycles per recording — more than sufficient for the RNN and HSMM to identify repeating patterns. Duration distributions are nearly identical across the three murmur classes (mean ~22–23s for all), confirming that duration is not a confounding variable for classification. *(Task 1.6)*

---

### 2. Class Imbalance — The Central Challenge

The murmur label distribution is severely imbalanced *(Task 1.3)*:

- **Absent:** 695 patients (73.8%) — no murmur detected
- **Present:** 179 patients (19.0%) — murmur confirmed by expert annotator
- **Unknown:** 68 patients (7.2%) — annotator could not determine

The ~4:1 ratio between Absent and Present means a naive "always predict Absent" classifier would achieve 73.8% standard accuracy — yet would miss every murmur. This is why the PhysioNet Challenge uses **weighted accuracy** with asymmetric weights (Present=5, Unknown=3, Absent=1): missing a murmur is penalised 5× more heavily than missing an Absent case. This also justifies the class-weighted cross-entropy loss used during RNN training in the baseline paper.

---

### 3. Demographics — Not a Strong Predictor

The dataset is almost entirely paediatric: **Child (664, 76.5%)** dominates, followed by Infant (126), Adolescent (72), and Neonate (6). No Young Adult patients exist. 74 patients (7.9%) have missing age information. Note that age categories here come from the dataset's own labels — they are not derived from numeric ages. *(Task 1.4)*

Sex is nearly balanced (Female 486, Male 456). The critical finding is that **murmur prevalence is stable at ~19% across all age groups and both sexes** — Female 18.9% vs Male 19.1%, and no age group deviates meaningfully from the overall 19% rate. Height and weight distributions similarly show no class-separating pattern. *(Task 1.4)*

**Implication for model design:** Demographics provide minimal discriminative power for murmur detection. The baseline paper's decision to rely primarily on the PCG audio signal (through the RNN+HSMM pipeline) rather than patient metadata is well-justified by this data. Metadata features like age, sex, weight, and height are used only in the CatBoost clinical outcome branch, which is out of scope for this project.

---

### 4. Unknown Class = Signal Quality, Not Clinical Ambiguity

One of the most important findings from the EDA is that the Unknown class does not represent genuine clinical uncertainty about whether a murmur exists. Instead, it reflects **poor signal quality** that prevents the expert annotator from making any determination. Three independent lines of evidence support this conclusion:

**Evidence 1 — Number of recordings (Task 1.5):** Unknown prevalence drops sharply as the number of recordings increases: 25.8% of patients with just 1 recording are Unknown, versus only 2.2% of patients with 4 recordings. Patients with fewer recordings tend to be those where the clinical encounter was difficult (uncooperative child, limited time), resulting in both fewer and lower-quality recordings.

**Evidence 2 — Annotation coverage (Task 1.10):** The segmentation annotations (.tsv files) reveal that Unknown recordings have a median annotation coverage of only 24.1% — meaning ~75% of the recording is Unannotated (state 0) because the annotator could not identify S1/S2 boundaries. Present recordings have 59.6% median coverage, and Absent recordings have 52.0%. Visually, Unknown recording 9983_MV shows huge Unannotated gaps in the first 8 seconds. *(Task 1.9)*

**Evidence 3 — SNR proxy (Task 1.10):** The heart sound signal-to-noise ratio is lowest for Unknown (median 5.65 dB) compared to Present (9.50 dB) and Absent (8.66 dB). Heart sounds in Unknown recordings are harder to distinguish from background noise.

**Evidence 4 — Age distribution (Task 1.4):** Unknown is most prevalent in the youngest patients — Neonates (16.7%) and Infants (19.8%) vs Children (5.6%) and Adolescents (4.2%). Younger children are harder to record (movement, crying, smaller body → weaker signal).

**Implication for the pipeline:** The baseline model assigns Unknown when the maximum HSMM confidence C_ω̂ falls below 0.65. This threshold-based approach effectively learns the same pattern discovered above: when signal quality is too low for confident segmentation, the model abstains rather than guessing. Understanding this mechanism is important for Phase 5 (threshold optimisation as an improvement path).

---

### 5. Murmur Characteristics — What the Model Needs to Detect

Analysis restricted to the 179 Present patients reveals several important patterns *(Task 1.7)*:

**Almost exclusively systolic:** 178 of 179 murmurs are systolic, with only 5 patients having any diastolic murmur annotation. This means the entire detection task is essentially systolic murmur detection — diastolic murmur detection is not feasible with this dataset.

**Three dominant timing patterns:**
- **Holosystolic (101 cases, 57%):** Murmur fills the entire systolic interval between S1 and S2. Waveform shows continuous high-energy signal throughout systole (visible in example 9979_TV). Modelled by HSMM topology ω₂, where the murmur state replaces the normal systole state entirely.
- **Early-systolic (59 cases, 33%):** Murmur appears immediately after S1, then transitions to normal systole before S2. Modelled by ω₃ (5-state: S1 → Murmur → Systole → S2 → Diastole).
- **Mid-systolic (17 cases, 10%):** Normal systole first, then murmur appears before S2. Modelled by ω₄ (5-state: S1 → Systole → Murmur → S2 → Diastole).
- **Late-systolic (1 case, 0.6%):** Too rare for any model to learn. No dedicated HSMM topology, which is appropriate.

The match between the data's timing distribution and the paper's 4 HSMM topologies (ω₁ normal + ω₂–ω₄ for the 3 common murmur timings) is a key design validation.

**Grade distribution — Grade 1 is the hard problem:**
- Grade 1 (quiet, I/VI): 104 cases (58%)
- Grade 2 (moderate, II/VI): 28 cases (16%)
- Grade 3 (loud, III/VI): 46 cases (26%)

Over half of all murmurs are Grade 1 — the quietest and hardest to detect. The baseline paper reports that all 13 misclassified Present patients on the training set were Grade 1. This makes Grade 1 sensitivity the primary improvement target for Phase 5.

**Murmurs are widespread, not localised:** Mean 2.8 murmur locations per patient (median 3), and 70 of 179 patients (39%) have murmurs audible at all 4 standard locations. This justifies the pipeline's aggregation rule: if ANY recording from a patient is classified as murmur → the patient is labelled "Murmur Present." A localised aggregation strategy (requiring multiple locations) would miss the 34 patients (19%) with murmur at only 1 location.

**Other characteristics:** Plateau shape dominates (62%), Low pitch is most common (49%), and Harsh quality (54%) slightly edges out Blowing (44%). Musical quality is rare (4 cases). These characteristics are less directly relevant to the model architecture but useful for report context.

---

### 6. Signal Characteristics — What Raw Data Looks Like

Visual inspection of 3 representative waveforms *(Tasks 1.8, 1.9)* reveals:

**Normal heart sound (2530_MV):** Low amplitude (RMS=1165), clean signal with S1 and S2 appearing as distinct spikes separated by quiet intervals. Diastole is nearly silent. Segmentation annotations are dense and regular — 44 cardiac cycles in 26.7s (~99 bpm). Occasional amplitude spikes from stethoscope movement artifacts.

**Murmur heart sound (9979_TV):** Much higher amplitude (RMS=7038, 6× normal). The defining visual feature is that systole is NOT quiet — instead, continuous high-energy signal fills the interval between S1 and S2. This is the signature of holosystolic murmur visible even without any signal processing. Annotations are regular but sparser than Normal.

**Unknown heart sound (9983_MV):** Low amplitude (RMS=1067, similar to Normal) but irregular pattern — alternating between near-silence and sudden energy bursts. Segmentation annotations are extremely sparse (coverage ~24%), with most of the recording as Unannotated. The annotator could not reliably identify cardiac cycle boundaries.

**Amplitude saturation:** Both Normal and Murmur examples hit the int16 maximum (±32768), indicating clipping/saturation in the recording hardware. This is a known issue requiring amplitude normalisation before spectrogram extraction (addressed in Phase 2).

---

### 7. Cross-Phase Example Recordings

Three recordings were selected for tracking across all project phases. The same examples will be used in Phase 2 (preprocessing visualisation), Phase 3 (RNN output inspection), and Phase 4 (case study walkthroughs).

| ID | Label | Patient | Location | Why chosen |
|----|-------|---------|----------|------------|
| 2530_MV | Normal | 2530 | MV | Clean signal, full 4 locations, dense annotations, typical Absent case |
| 9979_TV | Murmur | 9979 | TV | Grade 3 Holosystolic, most audible at TV, high RMS, clear murmur pattern |
| 9983_MV | Unknown | 9983 | MV | Has 4 recordings but still Unknown, sparse annotations, irregular signal |

---

### 8. Implications for Next Phases

| Finding | Phase affected | How |
|---------|---------------|-----|
| Amplitude clipping | Phase 2 | Amplitude normalisation required before spectrogram |
| Class imbalance (73.8/19/7.2%) | Phase 3 | Class-weighted cross-entropy loss for RNN training |
| Almost all systolic murmurs | Phase 3 | Only 3 murmur HSMM topologies needed (ω₂–ω₄) |
| Grade 1 = 58% of Present | Phase 4, 5 | Primary error analysis target; improvement focus |
| Unknown = signal quality | Phase 4, 5 | Confidence threshold analysis; potential improvement via threshold tuning |
| Variable annotation density | Phase 3 | RNN training must handle recordings with many Unannotated frames |
| Murmurs widespread across locations | Phase 3 | "Any location = Present" aggregation rule validated |
| Demographics not predictive | Phase 3 | Confirms audio-only pipeline is correct approach |

---

## Deliverables Produced

| File | Description |
|------|-------------|
| `notebooks/01_eda.ipynb` | Complete EDA notebook with 11 tasks, all plots, interpretations |
| `src/data/parse_metadata.py` | Reusable metadata parsing (handles duplicate-location patients) |
| `src/visualisation/style.py` | Project-wide color constants + plot style setup |
| `data/metadata/patients.csv` | 942 patients × 24 columns |
| `data/metadata/recordings.csv` | 3163 recordings × 8 columns (incl. quality metrics) |
| `figures/eda/v1_murmur_class_distribution.png` | Class imbalance bar chart |
| `figures/eda/v2a_age_by_murmur.png` | Age × murmur grouped bar |
| `figures/eda/v2b_sex_by_murmur.png` | Sex × murmur grouped bar |
| `figures/eda/v2c_height_weight_by_murmur.png` | Height/Weight histograms by class |
| `figures/eda/v3_recordings_per_patient.png` | Recordings per patient distribution |
| `figures/eda/v4_recording_durations.png` | Duration distribution by class |
| `figures/eda/v5_murmur_locations.png` | Murmur location bar chart |
| `figures/eda/v6_timing_grade_crosstab.png` | Timing × Grade cross-tabulation |
| `figures/eda/v7_raw_waveforms.png` | 3 example raw waveforms |
| `figures/eda/v8_segmentation_overlay.png` | 3 examples with segmentation overlay |
| `figures/eda/v_signal_quality.png` | Annotation coverage + SNR boxplots |
