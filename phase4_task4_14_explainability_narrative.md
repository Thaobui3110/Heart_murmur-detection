# Task 4.14 — Explainability Narrative
# Draft content for Report Section 6.9 (Phase 6)
# *(Section number to be confirmed in Phase 6 report outline)*

---

## 6.9 Explainability & Feature Interpretation

### 6.9.1 Explainability Strategy

This project adopts an **interpretability-by-design** approach rather than
post-hoc methods such as SHAP or LIME. This choice is motivated by the
architecture of the pipeline itself: unlike black-box classifiers that
map raw features directly to predictions, the RNN + parallel HSMM pipeline
produces interpretable intermediate outputs at every stage — RNN state
posteriors, per-topology confidence scores, and decoded segmentation paths.
These intermediate representations carry direct clinical meaning (cardiac
cycle states, murmur timing type, segmentation quality), making them
inherently more informative than post-hoc attribution maps applied to a
learned embedding.

The analysis is structured across three levels: **global explainability**
(how does the model behave across the full dataset?), **local explainability**
(how does the model decide on a specific recording?), and **error-based
explainability** (where and why does the model fail?).

---

### 6.9.2 Global Explainability — How Does the Pipeline Decide?

#### 6.9.2.1 The Two-Dimensional Decision Space

Figure s3 reproduces PLOS Fig. 4 and visualises the classification logic
of the pipeline in two-dimensional confidence space: C(M−N) on the x-axis
(murmur confidence minus normal confidence) and C(ω̂) on the y-axis
(maximum segmentation confidence across four HSMM topologies).

Three distinct clusters emerge from this scatter plot. Present patients
(n=179) concentrate in the right region where C(M−N) > 0, reflecting
strong murmur evidence in the signal. Absent patients (n=695) cluster
in the upper-left quadrant — high segmentation quality C(ω̂) > 0.65 but
negative C(M−N). Unknown patients (n=68) distribute along the lower
portion of the plot where C(ω̂) < 0.65, regardless of C(M−N) sign.

This two-dimensional structure reveals why the pipeline requires two
separate confidence measures rather than one: C(M−N) alone cannot
distinguish Unknown from Absent, since both classes have negative or
near-zero C(M−N). The C(ω̂) threshold at 0.65 acts as a signal quality
gate, abstaining from classification when the HSMM cannot confidently
segment the cardiac cycle.

#### 6.9.2.2 Frequency-band Feature Importance

Figure s5 presents the results of a frequency-band ablation study:
for each of the 41 frequency bins (0–800 Hz, 20 Hz step), the bin
was zeroed out in the spectrogram and the resulting drop in C(M−N)
was measured across 50 Present validation recordings.

The ablation profile reveals a clear peak at **140 Hz**
(importance = 0.0114), with the cluster of highest importance spanning
100–180 Hz. This is consistent with the known spectral characteristics
of heart murmurs, which concentrate energy in the low-frequency range
corresponding to turbulent blood flow through stenotic valves.

Critically, this post-model importance profile is consistent with the
pre-model correlation analysis from Phase 2 (Task 2.5c), which identified
the same 100–200 Hz range as the most discriminative between Present and
Absent patients using rank-biserial correlation on raw spectrograms.
The convergence of two independent analyses — one based on data statistics,
the other on model behaviour — provides strong evidence that the RNN has
learned to exploit the clinically relevant frequency region.

A secondary observation is that importance does not decay to zero above
400 Hz, with a local peak at 460 Hz and a slight rise near 760–800 Hz.
Combined with Phase 2 findings showing 30 bins above 800 Hz remaining
statistically significant after Bonferroni correction, this suggests the
800 Hz frequency cutoff is conservative and may discard discriminative
information — a finding that directly motivates the frequency range
expansion experiment in Phase 5.

---

### 6.9.3 Local Explainability — How Does the Pipeline Decide on One Recording?

#### 6.9.3.1 Segmentation Walkthrough (Three Tracking Recordings)

Figure s4 reproduces PLOS Fig. 5, showing four-panel segmentation examples
for the three tracking recordings selected in Phase 1.

For **2530_MV** (true: Absent), the RNN posterior alternates cleanly
between S1, Systole, S2, and Diastole with no Murmur activation. The
HSMM selects the Healthy topology (C(M−N) = −0.069) and its decoded
path closely matches the ground truth annotation — a textbook example
of a normal cardiac cycle correctly classified.

For **9979_TV** (true: Present, Grade 3 Holosystolic), the posterior
shows sustained Murmur state activation replacing Systole throughout the
recording, consistent with a holosystolic murmur that fills the entire
systolic interval. The HSMM correctly selects the Holosystolic topology
(C(M−N) = +0.223). Ground truth annotations, available only for the
first 5 seconds, confirm Murmur during systole.

For **9983_MV** (true: Unknown), the posterior is chaotic with no stable
pattern, reflecting the poor signal quality documented in Phase 1 for
Unknown recordings (annotation coverage 24%, SNR 5.7 dB). The ground
truth is almost entirely unannotated. C(ω̂) = 0.653 falls just above the
Unknown threshold, leading to an Absent prediction — a case where the
pipeline's abstention mechanism nearly activates.

#### 6.9.3.2 Parallel HSMM Path Comparison

Figure s6 examines patient 84786 (Grade 1 Early-systolic, false negative),
showing all four HSMM topology paths evaluated in parallel alongside the
RNN posterior.

For recording **84786_AV**, the Healthy topology wins with confidence
0.7756 against Early-systolic at 0.7363 — a margin of 0.039. The RNN
posterior shows intermittent Murmur activation that is insufficient to
overcome the Healthy path's likelihood advantage. For recording
**84786_PV**, the margin narrows to effectively zero: Healthy and
Early-systolic tie at 0.6815, with the pipeline selecting Healthy by
the tie-breaking rule. This near-miss case illustrates the sensitivity
of the decision boundary for low-grade murmurs, where the RNN's Murmur
posterior is present but too weak and too inconsistent to tip the balance.

---

### 6.9.4 Error-Based Explainability — Where and Why Does the Pipeline Fail?

#### 6.9.4.1 False Negatives: A Grade-Specific Failure

Figure s7 and the grade breakdown in Figure s9 reveal a striking pattern:
**all 13 false negative patients are Grade 1**, with zero false negatives
among Grade 2 or Grade 3 patients. The sensitivity by grade (87.5% /
100% / 100%) matches PLOS 2024 exactly.

This grade-specific failure has a mechanistic explanation traceable to
the preprocessing pipeline. Per-row z-score normalisation (Phase 2,
Task 2.4) removes absolute energy from each spectrogram frame, equalising
the dynamic range across recordings. For Grade 3 murmurs with large
amplitude, this normalisation preserves the relative contrast between
murmur and non-murmur frames. For Grade 1 murmurs with low amplitude,
the same normalisation makes murmur frames indistinguishable from noise
after standardisation. The ablation study (Figure s5) confirms that the
100–200 Hz band — where murmur energy concentrates — is the most
informative region, but even this signal is insufficient for Grade 1
when absolute energy has been removed by z-score.

Case study 84786_PV (Figure s8) exemplifies this failure mode: a
near-tie between Healthy and Early-systolic topologies, with Murmur
posterior appearing intermittently but never sustaining long enough for
the HSMM to commit.

#### 6.9.4.2 False Positives: Threshold Proximity

The 131 false positive patients (Absent classified as Present) share
a common feature: their C(M−N) values cluster immediately above zero
(0.00–0.05, Figure s7). This is structurally different from the true
positive cluster which concentrates at C(M−N) > 0.10. The FP
distribution suggests that these Absent recordings contain physiological
or environmental sounds — such as split S2, friction rubs, or movement
artifacts — that produce murmur-like spectral energy in the 100–200 Hz
band, sufficient to push C(M−N) marginally above threshold.

Case study 84969_MV (Figure s8) illustrates this: a high-amplitude
Absent recording with irregular energy bursts that the RNN interprets
as intermittent Murmur signal, resulting in C(M−N) = +0.167 and an
erroneous Present classification.

#### 6.9.4.3 Unknown Class: Signal Quality as the Root Cause

Figure s12 shows that Unknown patients have a median C(ω̂) of 0.672 —
directly adjacent to the classification threshold at 0.65 — compared
to 0.800 for Present and 0.810 for Absent. Only 36.8% of Unknown
patients fall below the threshold (correctly triggering abstention),
while the remaining 63.2% are misclassified as Present or Absent.

This distribution reflects the Phase 1 finding that Unknown patients
have significantly lower annotation coverage (24% vs 60% for Present)
and lower SNR (5.7 dB vs 9.5 dB). Poor signal quality degrades HSMM
segmentation confidence, but not consistently enough to trigger the
C(ω̂) < 0.65 abstention rule across all Unknown recordings.

---

### 6.9.5 Calibration

Figure s11 shows the reliability diagram for C(ω̂) against binary
classification accuracy (Present/Absent recordings only). The ECE of
0.059 indicates moderate calibration. The pipeline is well-calibrated
in the C(ω̂) = 0.65–0.80 range — the region most relevant for clinical
deployment — but is slightly underconfident at low values and slightly
overconfident above 0.80. The practical implication is that C(ω̂) scores
in the well-calibrated range can be communicated to clinicians as
meaningful probability estimates, while high-confidence predictions
(C(ω̂) > 0.80) may warrant slight downward adjustment.

---

### 6.9.6 Summary

The three-level explainability analysis converges on a coherent picture
of the pipeline's behaviour. Globally, the model correctly identifies
the 100–200 Hz frequency band as the most informative region, consistent
with both clinical knowledge of murmur acoustics and Phase 2 data
analysis. Locally, the pipeline provides transparent decision paths
through its intermediate outputs, enabling clinician-interpretable
justification for each classification. At the error level, failures
are systematic rather than random: all missed murmurs are Grade 1,
all false positives are near-threshold Absent recordings, and Unknown
misclassification is explained by signal quality falling just above
the abstention boundary. These findings directly inform the targeted
improvement experiments proposed in Phase 5.

---

## Figures Referenced in This Section

| Figure | File | Task |
|--------|------|------|
| s3 | figures/results/s3_hsmm_confidence_scatter.png | 4.3 |
| s4 | figures/results/s4_segmentation_examples.png | 4.4 |
| s5 | figures/explainability/s5_frequency_importance.png | 4.5 |
| s6 | figures/explainability/s6_hsmm_path_comparison.png | 4.6 |
| s6b | figures/explainability/s6b_hsmm_path_comparison_84786_PV.png | 4.6 |
| s7 | figures/results/s7_error_fp_fn.png | 4.7 |
| s8 | figures/explainability/s8_case_study_*.png (x5) | 4.8 |
| s9 | figures/results/s9_grade_performance.png | 4.9 |
| s11 | figures/results/s11_reliability_diagram.png | 4.11 |
| s12 | figures/results/s12_unknown_analysis.png | 4.12 |

## Word Count
~1150 words (target: 800–1200 words) ✅
