# Heart Murmur Detection — Cấu trúc dự án

```
heart-murmur-detection/
│
├── data/
│   ├── raw/
│   │   ├── training_data/              # 942 bệnh nhân × (AV/PV/TV/MV) × 3 định dạng
│   │   │   ├── {patient_id}.txt        # Metadata thô từng bệnh nhân (942 files)
│   │   │   ├── {patient_id}_{loc}.wav  # Tín hiệu PCG ghi âm (3 163 files)
│   │   │   ├── {patient_id}_{loc}.hea  # Header waveform (3 163 files)
│   │   │   └── {patient_id}_{loc}.tsv  # Nhãn phân đoạn S1/Systole/S2/Diastole (3 163 files)
│   │   │       # loc ∈ {AV, PV, TV, MV, Phc}
│   │   │       # Bệnh nhân nhiều bản ghi cùng vị trí: {pid}_{loc}_1.wav, _2.wav, ...
│   │   ├── training_data.csv           # CSV tổng hợp tất cả bệnh nhân (nguồn gốc CirCor)
│   │   ├── RECORDS                     # Danh sách tên bản ghi
│   │   ├── LICENSE.txt
│   │   └── SHA256SUMS.txt
│   │
│   ├── metadata/                       # Metadata đã parse và làm sạch
│   │   ├── patients.csv                # 1 dòng/bệnh nhân (942 dòng)
│   │   │                               # Cột quan trọng: patient_id, murmur, systolic_murmur_grading
│   │   │                               # (float 1.0/2.0/3.0), systolic_murmur_timing, murmur_locations
│   │   ├── recordings.csv             # 1 dòng/file .wav (~3163 dòng)
│   │   │                               # Cột: recording_id (stem wav), patient_id, location,
│   │   │                               # duration_seconds, n_frames
│   │   │                               # Lưu ý: recording_id = stem file wav, KHÔNG phải patient_id+'_'+loc
│   │   │                               # (quan trọng với bệnh nhân có nhiều bản ghi cùng vị trí)
│   │   └── cv_splits.json             # 5-fold stratified CV splits (seed=42, cấp patient)
│   │                                   # Present~19%, Unknown~7%, Absent~74% mỗi fold
│   │                                   # Mỗi fold: 753-754 BN train, 188-189 BN val
│   │
│   └── processed/                      # Dữ liệu đã tiền xử lý (tạo bởi notebook 02a)
│       ├── spectrograms/               # 3163 file .npy — log-spectrogram đã z-score
│       │   └── {recording_id}.npy      # shape (41, T), float32
│       │                               # 0-800 Hz, 50 Hz feature rate, STFT 50ms/hop20ms
│       └── labels/                     # 3163 file .npy — nhãn 5 trạng thái per-frame
│           └── {recording_id}.npy      # shape (T,), int8
│                                       # S1=0, Systole=1, S2=2, Diastole=3, Murmur=4, Unannotated=-1
│                                       # Murmur phân bố: Holo 59%, Early 29%, Mid 11%, Late 0.4%
│
├── src/                                # Source code dự án
│   ├── __init__.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── parse_metadata.py           # Load/parse metadata từ training_data → DataFrames
│   │   │   # load_patient_df()    — DataFrame 942 bệnh nhân
│   │   │   # load_recording_df()  — DataFrame 3163 bản ghi, xử lý multi-recording per location
│   │   │   # save_metadata()      — xuất ra data/metadata/
│   │   │
│   │   ├── labels.py                   # TSV thô → nhãn 5 lớp per-frame (Task 3.1)
│   │   │   # load_labels(recording_id) — đọc .tsv → (T,) array, S1=0...Murmur=4, Unannotated=-1
│   │   │   # Murmur relabelling theo timing của bác sĩ:
│   │   │   #   Holosystolic → 100% khoảng tâm thu thành Murmur
│   │   │   #   Early-systolic → 50% đầu mỗi đoạn tâm thu
│   │   │   #   Mid-systolic → 50% giữa mỗi đoạn tâm thu
│   │   │   #   Late-systolic → 50% cuối mỗi đoạn tâm thu
│   │   │   # Chỉ áp dụng khi murmur=='Present' VÀ location trong murmur_locations
│   │   │
│   │   └── loader.py                   # PyTorch Dataset + DataLoader (Task 3.2)
│   │       # PCGDataset          — serve (T,41) features + (T,) labels, hỗ trợ RAM cache
│   │       # load_dataset_to_ram() — preload toàn bộ 3163 bản ghi vào RAM (~1.4 GB)
│   │       #                         giải quyết I/O bottleneck OneDrive (~28s/batch → 0.3s/batch)
│   │       # pcg_collate_fn()    — sort desc theo độ dài, pad features=0, labels=-1
│   │       #                       → (B, T_max, 41) batch
│   │
│   ├── features/
│   │   ├── normalisation.py            # Chuẩn hoá biên độ PCG
│   │   │   # normalise_amplitude() — zero-mean, peak-norm: x = (x-mean)/max(|x-mean|)
│   │   └── spectrogram.py              # Biến đổi tín hiệu PCG → feature matrix (41, T)
│   │       # compute_log_spectrogram() — STFT Hann 50ms/hop20ms → log(|X|²+ε), shape (101, T)
│   │       # crop_frequency()          — giữ 0–800 Hz: (101,T) → (41,T), loại 59% bins nhiễu
│   │       # zscore_per_row()          — normalize từng freq bin theo time: mean≈0, std≈1
│   │
│   ├── models/
│   │   ├── rnn.py                      # Kiến trúc Bidirectional GRU (Task 3.3)
│   │   │   # MurmurRNN: GRU(in=41, hidden=60, layers=3, bidirectional)
│   │   │   #            + Dropout(0.1) + FC(120→60, Tanh, Dropout) + FC(60→40, Tanh) + FC(40→5)
│   │   │   # Đầu vào: (B, T, 41) → Đầu ra: (B, T, 5) logits (không có softmax)
│   │   │   # Tổng tham số: 178 025 — nhỏ gọn, phù hợp dataset 942 BN
│   │   │   # build_model(seed=42) — khởi tạo + seed torch/numpy/random
│   │   │
│   │   ├── hsmm.py                     # Ước lượng nhịp tim + phân phối thời gian (Tasks 3.8–3.9)
│   │   │   # estimate_heart_rate(posteriors) — ACF của S1+Sys+S2+Mur (= 1-P(Dia))
│   │   │   #                                   no zero-mean, argmax trực tiếp → BPM
│   │   │   # estimate_systolic_interval(posteriors, hr_bpm)
│   │   │   #   ACF của P(S1)+P(S2), search [150ms, heart_cycle/2], trả về lag trực tiếp
│   │   │   # compute_duration_distributions(hr_bpm, sys_interval)
│   │   │   #   Hằng số McDonald (fit từ CirCor nhi đồng):
│   │   │   #   S1=116.3ms±19.6ms, S2=103.2ms±19.5ms (cố định, không phụ thuộc nhịp tim)
│   │   │   #   Systole = (sys_interval - 127.9ms) ± 25ms
│   │   │   #   Diastole = (heart_period - sys_interval - 105.3ms) ± 50ms
│   │   │   # get_hsmm_params(posteriors) — wrapper: posteriors→HR→sys_interval→log_dur_dists
│   │   │
│   │   ├── viterbi.py                  # HSMM Viterbi thuần NumPy (Task 3.10)
│   │   │   # hsmm_viterbi(posteriors, duration_matrix, max_duration, transition_matrix)
│   │   │   #   Khớp interface viterbi_hmm.pyx của McDonald (Cython)
│   │   │   #   Dùng cumulative log-sum tránh O(T×N×D) loop bên trong
│   │   │   #   ~2.5s/recording T≈1100 frames (bottleneck — Phase 5 sẽ vectorise)
│   │   │   # build_duration_matrix(log_dur_dists, state_order, d_max)
│   │   │
│   │   └── parallel_hsmm.py            # 4 HSMM topologies song song + confidence (Tasks 3.11–3.12)
│   │       # ω₁ segment_healthy()       — 4-state S1→Sys→S2→Dia, bỏ posterior Murmur
│   │       # ω₂ segment_holosystolic()  — 4-state, Murmur thay thế channel Systole
│   │       # ω₃ segment_early_systolic()— 5-state S1→Mur→Sys→S2→Dia, phân bố Systole ÷2
│   │       # ω₄ segment_mid_systolic()  — 5-state S1→Sys↔Mur→S2→Dia, phân bố Systole ÷4
│   │       # compute_confidence(obs, states) — mean posterior tại vị trí Viterbi path
│   │       # run_parallel_hsmm(posteriors_5) — chạy cả 4, trả về dict:
│   │       #   healthy_conf, murmur_conf, murmur_model ('Holosystolic'/'Early-systolic'/'Mid-systolic')
│   │       #   all_confs (dict 4 topology), log_dur_dists, heart_rate_bpm, d_max
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py                  # Hàm tính chỉ số đánh giá (Task 4.1, 4.2)
│   │   │   # MURMUR_WEIGHTS = {Present:5, Unknown:3, Absent:1}
│   │   │   # compute_confusion_matrix(y_true, y_pred) → DataFrame
│   │   │   # compute_weighted_accuracy(y_true, y_pred) → float
│   │   │   # compute_per_class_metrics(y_true, y_pred) → dict (sensitivity, specificity, PPV, F1)
│   │   └── analysis.py                 # Phân tích lỗi (Task 4.7)
│   │       # identify_errors(df_pat) → (fn_df, fp_df)
│   │       #   fn_df = False Negative: Present thực tế → Absent dự đoán (13 BN, 100% Grade 1)
│   │       #   fp_df = False Positive: Absent thực tế → Present dự đoán (131 BN)
│   │
│   └── visualisation/
│       ├── __init__.py
│       ├── style.py                    # Style/màu sắc nhất quán toàn dự án
│       │   # MURMUR_COLORS     — Present/Unknown/Absent
│       │   # STATE_COLORS      — {0:S1, 1:Systole, 2:S2, 3:Diastole, 4:Murmur} (integer keys)
│       │   # LOCATION_COLORS   — AV/PV/TV/MV/Phc
│       │   # setup_style()     — cấu hình matplotlib toàn cục
│       └── explainability.py           # Hàm vẽ cho Phase 4 (Tasks 4.1–4.12)
│           # plot_confusion_matrix     — Task 4.1
│           # plot_roc_curve            — Task 4.2
│           # plot_per_class_metrics    — Task 4.2
│           # plot_confidence_scatter   — Task 4.3 (tái tạo PLOS Figure 4)
│           # plot_reliability_diagram  — Task 4.11
│           # plot_frequency_importance — Task 4.5 (ablation tần số)
│           # _plot_state_bar           — helper dùng trong 4.4, 4.6, 4.8
│
├── notebooks/                          # Jupyter notebooks
│   ├── 01_eda.ipynb                    # Phase 1: Exploratory Data Analysis (11 hình)
│   ├── 02a_preprocessing.ipynb         # Phase 2: Tiền xử lý → spectrograms + labels .npy
│   ├── 02b_feature_correlation.ipynb   # Phase 2: Phân tích tương quan đặc trưng
│   ├── 03_model_reproduction.ipynb     # Phase 3 (local): Verify RNN, HSMM, Tasks 3.1–3.12
│   │   # KHÔNG train — chỉ kiểm tra kiến trúc và visualise posteriors/duration/Viterbi
│   ├── 03_train_rnn_colab.ipynb        # Phase 3 (Colab T4): Train 5-fold BiGRU (Task 3.6)
│   │   # lr=1e-3, batch=32, patience=10, max_epochs=100, seed=42
│   │   # Tổng ~62 phút, checkpoint → models/rnn/
│   ├── 03_hsmm_inference_colab.ipynb   # Phase 3 (Colab): HSMM inference + evaluation toàn bộ
│   │   # Tasks 3.13–3.17: run_parallel_hsmm() 3163 recordings, patient aggregation,
│   │   # weighted accuracy, confusion matrix, AUC-ROC, reproduction report
│   ├── 04_analysis_explainability.ipynb.ipynb  # Phase 4 (local): Analysis & Explainability
│   │   # Tasks 4.1–4.14: confusion matrix, ROC, HSMM scatter, segmentation examples,
│   │   # frequency ablation, parallel HSMM comparison, error analysis, case studies,
│   │   # grade/location performance, reliability diagram, Unknown analysis, narrative
│   ├── 04_task45.ipynb                 # Phase 4 (local): Tasks 4.5 ablation bổ sung
│   │   # Frequency importance ablation — 50 recordings, 41 bins, ~86 phút Colab
│   └── data/processed/spectrograms/   # Copy spectrograms dùng trong Colab
│
├── figures/
│   ├── eda/                            # Phase 1 — Biểu đồ EDA (notebook 01)
│   │   ├── v1_murmur_class_distribution.png  # Phân bố 3 lớp: Present/Unknown/Absent
│   │   ├── v2a_age_by_murmur.png             # Phân bố tuổi theo lớp murmur
│   │   ├── v2b_sex_by_murmur.png             # Phân bố giới tính
│   │   ├── v2c_height_weight_by_murmur.png   # Phân bố chiều cao/cân nặng
│   │   ├── v3_recordings_per_patient.png     # Số bản ghi/BN (có BN có nhiều bản ghi)
│   │   ├── v4_recording_durations.png        # Phân bố độ dài bản ghi (giây)
│   │   ├── v5_murmur_locations.png           # Vị trí nghe tim của tiếng thổi (AV/PV/TV/MV)
│   │   ├── v6_timing_grade_crosstab.png      # Crosstab: timing × grade của murmur
│   │   ├── v7_raw_waveforms.png              # Tín hiệu PCG thô (3 mẫu: Present/Unknown/Absent)
│   │   ├── v8_segmentation_overlay.png       # Waveform + nhãn phân đoạn chồng lên nhau
│   │   └── v_signal_quality.png              # Chất lượng tín hiệu theo lớp
│   │
│   ├── preprocessing/                  # Phase 2 — Biểu đồ tiền xử lý (notebook 02a)
│   │   ├── s2_normalisation.png        # Before/after chuẩn hoá biên độ (3×2 grid)
│   │   ├── s3_spectrogram_full.png     # Log-spectrogram 0–2000 Hz, đường 800 Hz cyan
│   │   ├── s4_frequency_crop.png       # So sánh 101 bins vs 41 bins sau crop
│   │   ├── s5_zscore_effect.png        # Effect of per-row z-score (murmur bins nổi bật)
│   │   ├── s6_pipeline_murmur.png      # Full pipeline end-to-end — mẫu Present
│   │   ├── s6_pipeline_normal.png      # Full pipeline end-to-end — mẫu Absent
│   │   ├── s6_pipeline_unknown.png     # Full pipeline end-to-end — mẫu Unknown
│   │   ├── s7_spectrogram_comparison.png  # So sánh spectrogram 3 class (RNN input thực tế)
│   │   ├── s8_frequency_content.png       # Mean spectral energy theo freq bin (0–2000 Hz)
│   │   └── s11_pipeline_diagram.png       # Sơ đồ tổng thể data pipeline (Phase 4)
│   │
│   ├── correlation/                    # Phase 2 — Phân tích tương quan (notebook 02b)
│   │   ├── s1b_metadata_correlation.png    # Cramér's V / η² của metadata vs murmur — tất cả < 0.15
│   │   │                                   # Justify audio-only pipeline (metadata không phân biệt được)
│   │   ├── s2c_spectral_discrimination.png # Mean energy Present vs Absent, Mann-Whitney U + Bonferroni
│   │   │                                   # 71/101 bins significant; peak 140 Hz nhất quán Phase 4
│   │   └── s3d_recording_features.png      # Kruskal-Wallis: duration/annotation coverage/SNR
│   │                                       # Unknown có annotation coverage thấp hơn rõ rệt
│   │
│   ├── results/                        # Phase 3 & 4 — Kết quả mô hình
│   │   │                               # ── Phase 3 (task 3.7, 3.8, 3.9, 3.16) ──
│   │   ├── v13_rnn_posteriors_2530_MV.png     # Task 3.7: Posterior RNN — Absent (bình thường)
│   │   │                                       # 3 panel: ground truth | log-spec | posterior 5 trạng thái
│   │   ├── v13_rnn_posteriors_9979_TV.png     # Task 3.7: Posterior RNN — Present Holosystolic
│   │   │                                       # Đỉnh Murmur đều đặn 22 giây dù GT chỉ có 5 giây đầu
│   │   ├── v13_rnn_posteriors_9983_MV.png     # Task 3.7: Posterior RNN — Unknown
│   │   │                                       # Confidence thấp, các trạng thái cạnh tranh 0.4-0.8
│   │   ├── v13_autocorr_9979_TV.png           # Task 3.8: ACF ước lượng HR và khoảng tâm thu
│   │   ├── v13_duration_dists_9979_TV.png      # Task 3.9: Phân bố thời lượng (Springer fractions)
│   │   ├── v13_duration_dists_mcdonald_9979_TV.png  # Task 3.9: Phân bố thời lượng (McDonald constants)
│   │   ├── v14_hsmm_confidence_scatter.png    # Task 3.16: Tái tạo PLOS Figure 4
│   │   │                                       # C(M-N) vs C(ω̂): Present dương, Absent âm, Unknown phân tán
│   │   ├── v23_segmentation_examples.png      # Task 3.16: Tái tạo PLOS Figure 5
│   │   │                                       # Waveform + phân đoạn + posterior 3 bản ghi mẫu
│   │   │
│   │   │                               # ── Phase 4 (tasks 4.1–4.12) ──
│   │   ├── s1_confusion_matrix.png            # Task 4.1: Ma trận nhầm lẫn 3×3
│   │   │                                       # Present: 166 đúng / 12 FN / 1 FN-Unknown
│   │   ├── s2_roc_curve.png                   # Task 4.2: ROC curve, AUC=0.952 (PLOS: 0.947)
│   │   ├── s2b_per_class_metrics.png          # Task 4.2: Sensitivity/Specificity/PPV/F1 per class
│   │   │                                       # Present 92.7% / Unknown 19.1% / Absent 74.4%
│   │   ├── s3_hsmm_confidence_scatter.png     # Task 4.3: HSMM scatter (phiên bản cải tiến)
│   │   ├── s4_segmentation_examples.png       # Task 4.4: Segmentation examples (phiên bản cải tiến)
│   │   ├── s7_error_fp_fn.png                 # Task 4.7: Phân tích lỗi FP/FN
│   │   │                                       # FN: 13 BN, 100% Grade 1, 10/13 Early-systolic
│   │   │                                       # FP: 131 BN, C(M-N) tập trung 0.00-0.05
│   │   ├── s9_grade_performance.png           # Task 4.9: Sensitivity theo grade murmur
│   │   │                                       # Grade 1: 87.5%, Grade 2+3: 100% (khớp PLOS)
│   │   ├── s10_location_performance.png       # Task 4.10: Sensitivity theo vị trí nghe tim
│   │   ├── s11_reliability_diagram.png        # Task 4.11: Reliability diagram, ECE=0.059
│   │   │                                       # Well-calibrated vùng C(ω)=0.65-0.80
│   │   └── s12_unknown_analysis.png           # Task 4.12: Phân tích lớp Unknown
│   │                                           # Median C(ω̂)=0.672 (sát ngưỡng 0.65), 36.8% dưới ngưỡng
│   │
│   ├── explainability/                 # Phase 4 — Explainability analysis
│   │   ├── s5_frequency_importance.png        # Task 4.5: Ablation tần số
│   │   │                                       # Peak 140 Hz (importance=0.0114), top 5: 140/160/120/180/460 Hz
│   │   │                                       # Nhất quán với Phase 2 Task 2.5c (peak 140 Hz)
│   │   ├── s5b_importance_vs_correlation.png  # Task 4.5: Importance vs spectral correlation
│   │   │                                       # (skipped per summary — correlation không ý nghĩa do z-score)
│   │   ├── s6_hsmm_path_comparison.png        # Task 4.6: So sánh 4 topology — BN 84786_AV
│   │   │                                       # Healthy 0.7756 vs Early-systolic 0.7363 (margin 0.039)
│   │   ├── s6b_hsmm_path_comparison_84786_PV.png  # Task 4.6: Tie case — 84786_PV
│   │   │                                           # Healthy=Early-systolic=0.6815 → tie-breaking chọn Healthy
│   │   ├── s8_case_study_2530_MV.png         # Task 4.8: Case study — Absent (True Negative)
│   │   ├── s8_case_study_9979_TV.png         # Task 4.8: Case study — Present Grade 3 Holosystolic (TP)
│   │   ├── s8_case_study_9983_MV.png         # Task 4.8: Case study — Unknown (khó phân loại)
│   │   ├── s8_case_study_84786_PV.png        # Task 4.8: Case study — FN Grade 1 Early-systolic
│   │   └── s8_case_study_84969_MV.png        # Task 4.8: Case study — FP Absent→Present
│   │
│   └── improvements/                   # Phase 5 — (chưa có)
│
├── models/
│   ├── rnn/                            # Checkpoint BiGRU 5-fold (Task 3.6)
│   │   ├── fold_0_best.pt              # Val loss=0.338 (best epoch ~18)
│   │   ├── fold_1_best.pt              # Val loss=0.407 (best epoch ~15)
│   │   ├── fold_2_best.pt              # Val loss=0.367 (best epoch ~21)
│   │   ├── fold_3_best.pt              # Val loss=0.359 (best epoch ~12)
│   │   └── fold_4_best.pt              # Val loss=0.396 (best epoch ~18)
│   └── catboost/                       # (Trống — outcome classifier chưa train)
│
├── experiments/
│   ├── configs/                        # (Trống)
│   ├── logs/                           # (Trống — fold_{0-4}_loss.csv nếu có)
│   └── results/                        # Kết quả inference + đánh giá
│       ├── recording_results.csv       # C(M-N), C(ω̂), murmur_model per 3163 bản ghi
│       │                               # Cột: recording_id, fold, c_mn, c_hat, murmur_model
│       ├── patient_results.csv         # pred_murmur, true_murmur per 942 bệnh nhân
│       │                               # Cột: patient_id, true_murmur, pred_murmur, c_mn_max, c_hat_min
│       ├── reproduction_metrics.json   # Tất cả chỉ số đánh giá Phase 3
│       │                               # weighted_accuracy=0.7726, AUC=0.9523, n_patients=942
│       ├── roc_curve.png               # Đường cong ROC (từ Phase 3 notebook inference)
│       ├── reproduction_comparison.md  # So sánh đầy đủ với PLOS 2024 và CinC 2022
│       ├── phase3_reproduction_report.md  # Báo cáo kiểm chứng tái tạo (Task 3.17) — 9 phần
│       ├── phase4_strengths_limitations.md # Điểm mạnh/hạn chế (Task 4.13)
│       ├── ablation_importance.npy     # shape (41,) — mean Δ C(M-N) khi mask từng freq bin
│       └── ablation_delta_cmn.npy      # shape (41, 50) — Δ C(M-N) per bin per recording
│
├── reference_code/                     # Code tham khảo từ McDonald et al. (git submodule)
│   ├── src/
│   │   ├── segmenter.py                # NGUỒN THAM CHIẾU CHÍNH cho Phase 3
│   │   │                               # double_duration_viterbi(), segment_*(), get_heart_rate()
│   │   │                               # get_systolic_interval(), get_duration_distributions()
│   │   │                               # compute_segmentation_confidence()
│   │   ├── neural_networks.py          # Kiến trúc BiGRU gốc (tham chiếu cho rnn.py)
│   │   └── decision_tree.py            # CatBoost outcome classifier (chưa triển khai)
│   ├── results/                        # Scripts tạo figure cho bài báo PLOS 2024
│   │   ├── heart_rate_estimate.py      # Tham chiếu: signal S1+Sys+S2+Mur cho ACF HR
│   │   ├── murmur_results.py
│   │   ├── confidences_2D.py
│   │   ├── outcome_scores.py
│   │   ├── outcomes_roc.py
│   │   ├── nn_predictions.py
│   │   ├── reliability_diagram.py
│   │   ├── sample_recordings.py
│   │   ├── utils.py
│   │   └── official_outcome_scores.tsv
│   ├── final_model/
│   │   ├── settings.json
│   │   ├── recordings.csv
│   │   ├── tree_inputs.csv
│   │   ├── outcome_predictions.csv
│   │   ├── 50260_MV_posteriors.csv
│   │   └── 85203_AV_posterior.csv
│   ├── viterbi_hmm.pyx                 # Cython HSMM Viterbi (tham chiếu cho viterbi.py)
│   ├── team_code.py
│   ├── train_model.py
│   ├── train_model_cued.py
│   ├── train_model_hparams.py
│   ├── run_model.py
│   ├── evaluate_model.py
│   ├── helper_code.py
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── README.md
│   └── LICENSE.md
│
├── report/                             # (Trống — dành cho báo cáo)
├── presentation/                       # (Trống — dành cho slide thuyết trình)
└── project_structure.md                # File này
```

---

## Tóm tắt dữ liệu

| Mục | Số lượng |
|-----|----------|
| Bệnh nhân | 942 |
| File `.wav` (bản ghi PCG) | 3 163 |
| Spectrogram `.npy` (41, T) | 3 163 |
| Label `.npy` (T,) | 3 163 |
| Model checkpoint | 5 (5-fold) |

**Vị trí nghe tim:** AV (Aortic Valve), PV (Pulmonary Valve), TV (Tricuspid Valve), MV (Mitral Valve), Phc

**Nhãn murmur:** `Present` (179 BN) / `Unknown` (68 BN) / `Absent` (695 BN)

**Nhãn 5 lớp:** S1=0 · Systole=1 · S2=2 · Diastole=3 · Murmur=4 · Unannotated=-1

**Trọng số class loss:** Murmur=13.3× · S2=2.2× · Systole=2.0× · S1=1.9× · Diastole=1.0×

---

## Luồng dữ liệu (Data Pipeline)

```
data/raw/training_data/
        │
        ▼  src/data/parse_metadata.py
data/metadata/patients.csv + recordings.csv + cv_splits.json
        │
        ▼  notebooks/02a_preprocessing.ipynb
        ▼  normalisation.py → spectrogram.py [STFT → log → crop → z-score]
        ▼  src/data/labels.py [TSV → 5-state per-frame labels]
data/processed/spectrograms/ (41,T) + data/processed/labels/ (T,)
        │
        ▼  notebooks/03_train_rnn_colab.ipynb  [Colab T4 GPU, ~62 phút]
        ▼  src/models/rnn.py — BiGRU(41,60,3L) + FC(→5)
        ▼  src/data/loader.py — RAM preload → pcg_collate_fn
models/rnn/fold_{0..4}_best.pt
        │
        ▼  notebooks/03_hsmm_inference_colab.ipynb  [Colab]
        ▼  src/models/hsmm.py       [HR + sys_interval → duration dists]
        ▼  src/models/viterbi.py    [HSMM Viterbi NumPy]
        ▼  src/models/parallel_hsmm.py  [4 topologies → C(ω)]
experiments/results/recording_results.csv + patient_results.csv + reproduction_metrics.json
        │
        ▼  notebooks/04_analysis_explainability.ipynb  [local]
        ▼  src/evaluation/metrics.py + analysis.py
        ▼  src/visualisation/explainability.py
figures/results/ + figures/explainability/
experiments/results/ablation_*.npy + phase4_strengths_limitations.md
```

---

## Kết quả Phase 3 — So sánh với PLOS 2024

| Chỉ số | Của ta | PLOS 2024 | Chênh | Trong dung sai? |
|--------|:------:|:---------:|:-----:|:---------------:|
| Weighted accuracy | **0.773** | 0.798 | -0.025 | ✅ (±0.03) |
| Sensitivity Present | **92.7%** | 92.7% | 0% | ✅ khớp hoàn hảo |
| Sensitivity Unknown | **19.1%** | 30.9% | -11.8% | ❌ (LR=1e-3 thay vì 1e-4) |
| Sensitivity Absent | **74.4%** | 77.6% | -3.2% | ✅ (±5%) |
| AUC-ROC | **0.952** | 0.947 | +0.005 | ✅ vượt mục tiêu |
| ECE (calibration) | **0.059** | — | — | ✅ well-calibrated |

**6/9 chỉ số trong dung sai. Sai lệch tập trung ở lớp Unknown — nguyên nhân: LR=1e-3 thay vì 1e-4.**

---

## Tiến độ

| Phase | Nội dung | Trạng thái |
|-------|----------|-----------|
| Phase 1 | EDA — phân tích dữ liệu thô | ✅ Hoàn thành |
| Phase 2 | Tiền xử lý — spectrogram + label .npy | ✅ Hoàn thành |
| Phase 3 | Model reproduction — BiGRU + HSMM inference (17 tasks) | ✅ Hoàn thành |
| Phase 4 | Analysis & Explainability (14 tasks) | ✅ Hoàn thành |
| Phase 5 | Improvements — LR=1e-4, threshold tuning, vectorise Viterbi | ⬜ Chưa bắt đầu |
| Phase 6 | Final report + presentation | ⬜ Chưa bắt đầu |

---

## Quy ước đặt tên file bản ghi

```
Bệnh nhân 1 bản ghi/vị trí:     {patient_id}_{loc}.wav       # 13918_AV.wav
Bệnh nhân nhiều bản ghi/vị trí: {patient_id}_{loc}_1.wav     # 50260_MV_1.wav
recording_id = stem(wav file)    # KHÔNG phải patient_id+'_'+location
```

---

## Bản ghi mẫu quan trọng

| recording_id | Lớp | Grade | Timing | Fold | Vai trò |
|---|---|---|---|---|---|
| 2530_MV | Absent | — | — | fold_2 | Mẫu bình thường tracking |
| 9979_TV | Present | 3 | Holosystolic | fold_0 | Mẫu tiếng thổi rõ |
| 9983_MV | Unknown | — | — | fold_4 | Mẫu khó phân loại |
| 84786_AV, 84786_PV | Present | 1 | Early-systolic | fold_3 | FN patient (Grade 1 khó) |
| 84969_MV | Absent | — | — | fold_3 | FP patient |
