# Tóm Tắt Phase 4 — Analysis & Explainability

**Dự án:** Phát hiện tiếng thổi tim từ tín hiệu PCG
**Phase:** 4 / 6
**Trạng thái:** HOÀN THÀNH (14/14 tasks)
**Notebook chính:** notebooks/04_analysis_explainability.ipynb
**Ngày hoàn thành:** 2026-06-28

---

## Tiến Độ Tổng Quan

| Task | Nội dung | Trạng thái | File output chính |
|------|----------|------------|-------------------|
| 4.1 | Confusion Matrix | DONE | figures/results/s1_confusion_matrix.png |
| 4.2 | ROC + Per-class Metrics | DONE | s2_roc_curve.png, s2b_per_class_metrics.png |
| 4.3 | HSMM Confidence Scatter (PLOS Fig.4) | DONE | s3_hsmm_confidence_scatter.png |
| 4.4 | Segmentation Examples (PLOS Fig.5) | DONE | s4_segmentation_examples.png |
| 4.5 | Frequency Ablation Importance | DONE | s5_frequency_importance.png |
| 4.6 | Parallel HSMM Path Comparison | DONE | s6_hsmm_path_comparison.png, s6b_...png |
| 4.7 | Error Analysis FP/FN | DONE | s7_error_fp_fn.png |
| 4.8 | Case Study Walkthroughs | DONE | s8_case_study_{id}.png x5 |
| 4.9 | Performance theo Murmur Grade | DONE | s9_grade_performance.png |
| 4.10 | Performance theo Location | DONE | s10_location_performance.png |
| 4.11 | Reliability Diagram | DONE | s11_reliability_diagram.png |
| 4.12 | Unknown Class Analysis | DONE | s12_unknown_analysis.png |
| 4.13 | Strengths/Limitations | DONE | experiments/results/phase4_strengths_limitations.md |
| 4.14 | Explainability Narrative | DONE | phase4_task4_14_explainability_narrative.md (EN+VI) |

---

## Files Code Đã Tạo

### src/evaluation/__init__.py
File trống, tạo mới để Python nhận src/evaluation/ là package.

### src/evaluation/metrics.py
Hàm: compute_confusion_matrix, compute_weighted_accuracy, compute_per_class_metrics
Constants: MURMUR_WEIGHTS = {Present:5, Unknown:3, Absent:1}, CLASSES

### src/evaluation/analysis.py
Hàm: identify_errors(df_pat) → (fn_df, fp_df)

### src/visualisation/explainability.py
Hàm mới (file tạo mới hoàn toàn):
- plot_confusion_matrix — Task 4.1
- plot_roc_curve — Task 4.2
- plot_per_class_metrics — Task 4.2
- plot_confidence_scatter — Task 4.3
- plot_reliability_diagram — Task 4.11
- plot_frequency_importance — Task 4.5
- _plot_state_bar — helper dùng trong 4.4, 4.6, 4.8

QUAN TRỌNG: STATE_COLORS dùng integer keys {0:'#CCCCCC', 1:'#3498DB', 2:'#E74C3C', 3:'#2ECC71', 4:'#F1C40F'}

### Helper functions trong notebook (không đưa vào src/)
- get_posterior(rec_id, fold_name) → (T,5) posterior
- get_waveform(rec_id) → (signal_float, sr)
- get_gt_labels(rec_id, n_frames) → labels array
- get_all_topology_paths(rec_id, fold_name) → dict 4 topologies
- plot_segmentation_example(rec_id, fold_name, ax_array) → 4-panel
- plot_parallel_hsmm(rec_id, fold_name, result, save_path) → 6-row

---

## Kết Quả Chính

### Metrics (Task 4.1, 4.2)
Weighted Accuracy = 0.7726 (target 0.773) OK
AUC-ROC = 0.9523 (PLOS: 0.947) VUOT

Per-class:
  Present:  Sensitivity=0.927, Specificity=0.797, PPV=0.517, F1=0.664
  Unknown:  Sensitivity=0.191, Specificity=0.945, PPV=0.213, F1=0.202
  Absent:   Sensitivity=0.744, Specificity=0.826, PPV=0.923, F1=0.824

### Frequency Ablation (Task 4.5)
Peak importance: 140 Hz (0.0114)
Top 5: 140Hz, 160Hz, 120Hz, 180Hz, 460Hz
Nhat quan voi Phase 2 Task 2.5c (peak 140 Hz)
Secondary importance 400-800 Hz → evidence cho Phase 5 mo rong frequency range

### Error Analysis (Task 4.7, 4.9)
FN: 13 patients, 100% Grade 1, 10/13 Early-systolic
FP: 131 patients, C(M-N) tap trung 0.00-0.05

### Grade Performance (Task 4.9)
Grade 1: 87.5% (104 patients) -- khop PLOS 2024
Grade 2: 100% (28 patients)   -- khop PLOS 2024
Grade 3: 100% (46 patients)   -- khop PLOS 2024

### Parallel HSMM (Task 4.6)
Patient 84786 Grade 1 Early-systolic:
  84786_AV: Healthy 0.7756 vs Early-systolic 0.7363 (margin 0.039)
  84786_PV: Healthy 0.6815 vs Early-systolic 0.6815 (TIE -- tie-breaking chon Healthy)

### Calibration (Task 4.11)
ECE = 0.0593, well-calibrated o vung C(w) = 0.65-0.80

### Unknown Analysis (Task 4.12)
Unknown median C(w) = 0.672 (ngay sat nguong 0.65)
36.8% Unknown duoi nguong vs 10% Present/Absent

---

## Thong Tin Ky Thuat

### Ten cot thuc te (khac voi guide)
true_label       → true_murmur        (patient_results.csv)
predicted_label  → pred_murmur        (patient_results.csv)
murmur_grade     → systolic_murmur_grading (float 1.0/2.0/3.0) (patients.csv)
murmur_timing    → systolic_murmur_timing  (patients.csv)
location: phai join df_rec voi df_recordings

### Fold mapping ban ghi quan trong
2530_MV  → fold_2  (Tracking Absent)
9979_TV  → fold_0  (Tracking Present Grade 3 Holo)
9983_MV  → fold_4  (Tracking Unknown)
84786_AV, 84786_PV → fold_3 (FN patient Grade 1)
84969_MV → fold_3  (FP patient Absent→Present)

### Models da load trong notebook
fold_0, fold_1, fold_2, fold_3, fold_4 tat ca da load vao dict models

### API parallel_hsmm.py (khong co class)
run_parallel_hsmm(posteriors_5, feature_rate=50)
  → dict: healthy_states, healthy_conf, murmur_states, murmur_conf,
          murmur_model, all_confs, log_dur_dists, heart_rate_bpm, d_max
segment_healthy / segment_holosystolic / segment_early_systolic / segment_mid_systolic
  → (states, conf, _)

---

## Van De Gap Phai & Huong Giai Quyet

### Van de 1: Viterbi qua cham (Task 4.5)
Nguyen nhan: hsmm_viterbi() la pure Python loop, ~2.5s/recording tren Colab
Giai phap ap dung: giam subset xuong 10 recordings/fold = 50 tong, chay 86 phut
Huong fix sau Phase 4: vectorise bang NumPy broadcasting (uoc tinh 8x speedup)
QUAN TRONG: chi vectorise sau khi Phase 4 hoan toan xong tranh inconsistency voi Phase 3

### Van de 2: Khong tao duoc Figure s5b (Task 4.5)
Nguyen nhan: spectrograms da luu la AFTER z-score, mean ~0, khong tinh duoc correlation co nghia
Giai phap: skip s5b, narrative van du dua tren hai peak doc lap (Phase 2 vs Phase 4 deu ra 140 Hz)
Huong fix: save intermediate spectrograms TRUOC z-score neu can figure cho final report

---

## Files Output Day Du

### figures/results/
s1_confusion_matrix.png
s2_roc_curve.png
s2b_per_class_metrics.png
s3_hsmm_confidence_scatter.png
s4_segmentation_examples.png
s7_error_fp_fn.png
s9_grade_performance.png
s10_location_performance.png
s11_reliability_diagram.png
s12_unknown_analysis.png

### figures/explainability/
s5_frequency_importance.png
s6_hsmm_path_comparison.png (84786_AV)
s6b_hsmm_path_comparison_84786_PV.png (tie case)
s8_case_study_2530_MV.png
s8_case_study_9979_TV.png
s8_case_study_9983_MV.png
s8_case_study_84786_PV.png
s8_case_study_84969_MV.png

### experiments/results/
ablation_importance.npy  -- shape (41,), mean delta C(M-N) per bin
ablation_delta_cmn.npy   -- shape (41, 50), delta per bin per recording
phase4_strengths_limitations.md

### report/drafts/ (hoac luu o project root)
phase4_task4_14_explainability_narrative.md    -- tieng Anh (~1150 words)
phase4_task4_14_explainability_narrative_vi.md -- tieng Viet (~1100 words)

---

## Strengths & Limitations Tom Tat (Task 4.13)

DIEM MANH:
1. Sensitivity Present = 92.7% (khop PLOS 2024)
2. AUC = 0.952 (vuot PLOS 0.947)
3. Pipeline co kha nang giai thich tot qua intermediate outputs
4. Tu dong nhan dien timing type tien tho tim
5. ECE = 0.059, calibrated tot o vung C(w) = 0.65-0.80

HAN CHE:
1. Sensitivity Unknown = 19.1% (vs 30.9% PLOS) -- LR=1e-3 thay vi 1e-4
   → Phase 5: LR=1e-4, threshold tuning C(w) tu 0.65 xuong 0.55
2. 100% FN la Grade 1 -- z-score removes absolute energy
   → Phase 5: data augmentation hoac global normalization
3. FP rate 18.8% tu Absent -- C(M-N) ngay sat nguong 0
   → Phase 5: tang nguong len 0.02-0.05
4. 800 Hz cutoff conservative -- importance khong ve 0 o 400-800 Hz
   → Phase 5: mo rong len 1000-1200 Hz
5. Viterbi bottleneck -- ~2.5s/recording, khong dung GPU
   → Phase 5: vectorise NumPy

---

## Buoc Tiep Theo: Phase 5 — Improvements

Cac experiment duoc de xuat dua tren Phase 4 findings:
1. LR=1e-4 (fix Unknown sensitivity)
2. Threshold sweep C(M-N) (giam FP)
3. Frequency range mo rong 0-1200 Hz (Phase 2 + Task 4.5 evidence)
4. Vectorise Viterbi (prerequisite cho moi experiment khac)
5. Data augmentation cho Grade 1 (fix FN)
