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
│   │   │       # Nếu bệnh nhân có nhiều bản ghi cùng vị trí: {pid}_{loc}_1.wav, _2.wav, ...
│   │   ├── training_data.csv           # CSV tổng hợp tất cả bệnh nhân (nguồn gốc từ CirCor)
│   │   ├── RECORDS                     # Danh sách tên bản ghi
│   │   ├── LICENSE.txt
│   │   └── SHA256SUMS.txt
│   │
│   ├── metadata/                       # Metadata đã được parse và làm sạch
│   │   ├── patients.csv                # 1 dòng/bệnh nhân (942 dòng), output của parse_metadata.py
│   │   └── recordings.csv             # 1 dòng/file .wav (~3 163 dòng), kèm duration_seconds
│   │
│   └── processed/                      # (Trống — dành cho dữ liệu sau tiền xử lý)
│
├── src/                                # Source code dự án
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── parse_metadata.py           # Load/parse metadata từ training_data → DataFrames
│   │       # load_patient_df()    — DataFrame bệnh nhân, 942 dòng
│   │       # load_recording_df()  — DataFrame bản ghi, xử lý multi-recording per location
│   │       # save_metadata()      — Xuất ra data/metadata/
│   │
│   ├── features/
│   │   ├── normalisation.py            # normalise_amplitude(): zero-mean, peak-norm PCG
│   │   │   # x_norm = (x - mean(x)) / max(|x - mean(x)|)
│   │   └── spectrogram.py              # 3 bước biến đổi tín hiệu → feature matrix (41, T)
│   │       # compute_log_spectrogram() — STFT Hann 50ms/hop20ms → log(|X|²+ε), shape (101, T)
│   │       # crop_frequency()          — giữ 0–800 Hz: (101, T) → (41, T), bỏ 59% bins noise
│   │       # zscore_per_row()          — normalize từng freq bin theo time axis → mean≈0, std≈1
│   │
│   ├── visualisation/
│   │   ├── __init__.py
│   │   └── style.py                    # Style/màu sắc nhất quán toàn dự án
│   │       # MURMUR_COLORS     — Present/Unknown/Absent
│   │       # STATE_COLORS      — S1/Systole/S2/Diastole/Unannotated
│   │       # LOCATION_COLORS   — AV/PV/TV/MV/Phc
│   │       # setup_style()     — Cấu hình matplotlib toàn cục
│   │
│   ├── evaluation/                     # (Trống — dành cho metrics đánh giá)
│   └── modelsmkdir                     # (Artifact tạo nhầm, có thể xóa)
│
├── notebooks/                          # Jupyter notebooks phân tích
│   ├── 01_eda.ipynb                    # Exploratory Data Analysis
│   ├── 02a_preprocessing.ipynb         # Tiền xử lý tín hiệu
│   └── 02b_feature_correlation.ipynb   # Phân tích tương quan đặc trưng
│
├── figures/                            # Hình ảnh xuất ra từ notebooks
│   ├── eda/                            # Biểu đồ EDA
│   │   ├── v1_murmur_class_distribution.png
│   │   ├── v2a_age_by_murmur.png
│   │   ├── v2b_sex_by_murmur.png
│   │   ├── v2c_height_weight_by_murmur.png
│   │   ├── v3_recordings_per_patient.png
│   │   ├── v4_recording_durations.png
│   │   ├── v5_murmur_locations.png
│   │   ├── v6_timing_grade_crosstab.png
│   │   ├── v7_raw_waveforms.png
│   │   ├── v8_segmentation_overlay.png
│   │   └── v_signal_quality.png
│   ├── correlation/                    # Biểu đồ từ notebook 02b_feature_correlation.ipynb
│   │   ├── s1b_metadata_correlation.png    # Section 1: Cramér's V / η² của metadata features vs murmur label — tất cả < 0.15, justify audio-only pipeline
│   │   ├── s2c_spectral_discrimination.png # Section 2: 3-panel — mean energy theo freq bin (Present vs Absent), Mann-Whitney U + Bonferroni, 71/101 bins significant
│   │   └── s3d_recording_features.png      # Section 3: Kruskal-Wallis của duration/annotation coverage/SNR/n_recordings — Unknown có annotation coverage thấp hơn rõ
│   │
│   ├── preprocessing/                  # Biểu đồ từ notebook 02a_preprocessing.ipynb
│   │   ├── s2_normalisation.png        # Section 2: Before/After amplitude normalisation (3×2 grid)
│   │   ├── s3_spectrogram_full.png     # Section 3: Log-spectrogram 0–2000 Hz, đường 800 Hz cyan
│   │   ├── s4_frequency_crop.png       # Section 4: So sánh (101 bins) vs (41 bins) sau crop
│   │   ├── s5_zscore_effect.png        # Section 5: Effect of per-row z-score, murmur bins nổi bật
│   │   ├── s6_pipeline_murmur.png      # Section 6: Full pipeline end-to-end — mẫu Present
│   │   ├── s6_pipeline_normal.png      # Section 6: Full pipeline end-to-end — mẫu Absent
│   │   ├── s6_pipeline_unknown.png     # Section 6: Full pipeline end-to-end — mẫu Unknown
│   │   ├── s7_spectrogram_comparison.png  # Section 7: So sánh spectrogram đã xử lý đủ (z-scored, 0–800 Hz) giữa 3 class — đây là RNN input thực tế
│   │   └── s8_frequency_content.png       # Section 8: Mean spectral energy theo freq bin (0–2000 Hz) của 3 class trước z-score, phân tích phân bố năng lượng
│   ├── improvements/                   # (Trống)
│   └── results/                        # (Trống — biểu đồ kết quả mô hình)
│
├── models/                             # Checkpoint / artifact mô hình đã train
│   ├── catboost/
│   └── rnn/
│
├── experiments/                        # Quản lý thí nghiệm
│   ├── configs/                        # File cấu hình hyperparameter
│   ├── logs/                           # Log training
│   └── results/                        # Kết quả định lượng
│
├── reference_code/                     # Code tham khảo từ CirCor challenge (git submodule)
│   ├── src/
│   │   ├── decision_tree.py            # Mô hình cây quyết định
│   │   ├── neural_networks.py          # Mô hình mạng nơ-ron
│   │   └── segmenter.py                # Phân đoạn S1/S2 (Viterbi HMM)
│   ├── results/
│   │   ├── murmur_results.py
│   │   ├── outcome_scores.py
│   │   ├── nn_predictions.py
│   │   ├── heart_rate_estimate.py
│   │   ├── confidences_2D.py
│   │   ├── outcomes_roc.py
│   │   ├── reliability_diagram.py
│   │   ├── sample_recordings.py
│   │   ├── utils.py
│   │   └── official_outcome_scores.tsv
│   ├── final_model/                    # Mô hình cuối đã lưu
│   │   ├── settings.json
│   │   ├── recordings.csv
│   │   ├── tree_inputs.csv
│   │   ├── outcome_predictions.csv
│   │   ├── 50260_MV_posteriors.csv
│   │   └── 85203_AV_posterior.csv
│   ├── team_code.py                    # Entry point train/predict của nhóm challenge
│   ├── train_model.py
│   ├── train_model_cued.py
│   ├── train_model_hparams.py
│   ├── run_model.py
│   ├── evaluate_model.py
│   ├── helper_code.py
│   ├── viterbi_hmm.pyx                 # Cython: HMM Viterbi decoder
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── README.md
│   └── LICENSE.md
│
├── report/                             # (Trống — dành cho báo cáo)
├── presentation/                       # (Trống — dành cho slide thuyết trình)
│
└── 0].to_string())                     # (Artifact tạo nhầm, có thể xóa)
```

---

## Tóm tắt dữ liệu

| Mục | Số lượng |
|-----|----------|
| Bệnh nhân | 942 |
| File `.wav` (bản ghi PCG) | 3 163 |
| File `.hea` (header) | 3 163 |
| File `.tsv` (nhãn phân đoạn) | 3 163 |
| Tổng file trong `training_data/` | 10 431 |

**Vị trí nghe tim (auscultation locations):** AV (Aortic Valve), PV (Pulmonary Valve), TV (Tricuspid Valve), MV (Mitral Valve), Phc (Phocardiogram)

**Nhãn murmur:** `Present` / `Unknown` / `Absent`

**Nhãn phân đoạn (TSV):** `0` Unannotated · `1` S1 · `2` Systole · `3` S2 · `4` Diastole

---

## Luồng dữ liệu (Data Pipeline)

```
data/raw/training_data/          (dữ liệu thô CirCor DigiScope)
        │
        ▼  src/data/parse_metadata.py
data/metadata/patients.csv       (942 × N cột bệnh nhân, đã làm sạch)
data/metadata/recordings.csv     (~3163 × 6 cột, kèm duration)
        │
        ▼  src/features/normalisation.py → normalise_amplitude()
        ▼  src/features/spectrogram.py  → compute_log_spectrogram()
        ▼                                  crop_frequency()  [101→41 bins]
        ▼                                  zscore_per_row()
data/processed/                  (features/spectrogram đã tiền xử lý — Phase 3)
        │
        ▼  notebooks/02a_preprocessing.ipynb
models/                          (checkpoint mô hình)
        │
        ▼  experiments/
experiments/results/             (kết quả đánh giá)
figures/results/                 (biểu đồ kết quả)
```

---

## Quy ước đặt tên file bản ghi

```
Bệnh nhân có 1 bản ghi mỗi vị trí:
    {patient_id}_{loc}.wav          # vd: 13918_AV.wav

Bệnh nhân có nhiều bản ghi cùng vị trí:
    {patient_id}_{loc}_1.wav        # vd: 50260_MV_1.wav
    {patient_id}_{loc}_2.wav        # vd: 50260_MV_2.wav
```
