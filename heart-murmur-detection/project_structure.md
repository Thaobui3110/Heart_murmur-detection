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
│   │   ├── patients.csv                # 1 dòng/bệnh nhân (942 dòng)
│   │   ├── recordings.csv             # 1 dòng/file .wav (~3 163 dòng), kèm duration_seconds
│   │   └── cv_splits.json             # 5-fold stratified CV splits theo patient_id (seed=42)
│   │
│   └── processed/                      # Dữ liệu đã tiền xử lý (tạo bởi notebook 02a)
│       ├── spectrograms/               # Log-spectrogram (41, T) mỗi bản ghi — 3 163 file .npy
│       │   └── {recording_id}.npy      # shape (41, T), float32
│       └── labels/                     # Nhãn 5 lớp mỗi bản ghi — 3 163 file .npy
│           └── {recording_id}.npy      # shape (T,), int8; S1=0 Sys=1 S2=2 Dia=3 Mur=4 Unannotated=-1
│
├── src/                                # Source code dự án
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── parse_metadata.py           # Load/parse metadata từ training_data → DataFrames
│   │   │   # load_patient_df()    — DataFrame bệnh nhân, 942 dòng
│   │   │   # load_recording_df()  — DataFrame bản ghi, xử lý multi-recording per location
│   │   │   # save_metadata()      — Xuất ra data/metadata/
│   │   ├── labels.py                   # TSV → nhãn 5 lớp per-frame
│   │   │   # load_labels()        — đọc .tsv, interpolate → (T,) array per recording
│   │   │   # Murmur relabelling theo timing: Holosystolic(0,1), Early(0,0.5), Mid(0.25,0.75), Late(0.5,1)
│   │   │   # Location filter: chỉ gán Murmur tại vị trí nghe thấy murmur
│   │   └── loader.py                   # PyTorch Dataset + DataLoader utilities
│   │       # PCGDataset          — serve (T,41) features + (T,) labels, hỗ trợ RAM cache
│   │       # load_dataset_to_ram() — preload toàn bộ 3163 bản ghi vào RAM (~1.2 GB)
│   │       # pcg_collate_fn()    — sort desc, pad features=0, labels=-1 → (B, T_max, 41)
│   │
│   ├── features/
│   │   ├── normalisation.py            # normalise_amplitude(): zero-mean, peak-norm PCG
│   │   │   # x_norm = (x - mean(x)) / max(|x - mean(x)|)
│   │   └── spectrogram.py              # 3 bước biến đổi tín hiệu → feature matrix (41, T)
│   │       # compute_log_spectrogram() — STFT Hann 50ms/hop20ms → log(|X|²+ε), shape (101, T)
│   │       # crop_frequency()          — giữ 0–800 Hz: (101, T) → (41, T), bỏ 59% bins noise
│   │       # zscore_per_row()          — normalize từng freq bin theo time axis → mean≈0, std≈1
│   │
│   ├── models/
│   │   ├── rnn.py                      # Kiến trúc Bidirectional GRU (Task 3.5)
│   │   │   # MurmurRNN: GRU(41,60,3-layer,bidir) + Dropout(0.1) + FC(120→60→40→5)
│   │   │   # build_model(seed=42)  — khởi tạo + seed
│   │   │   # Input: (B, T, 41) → Output: (B, T, 5) logits (softmax → posteriors)
│   │   ├── hsmm.py                     # Ước lượng nhịp tim + phân phối thời gian (Tasks 3.8–3.9)
│   │   │   # estimate_heart_rate()      — ACF của S1+Sys+S2+Mur, argmax, no zero-mean
│   │   │   # estimate_systolic_interval() — ACF của P(S1)+P(S2), search [150ms, T/2]
│   │   │   # compute_duration_distributions() — Gaussian per state (McDonald absolute-sec constants)
│   │   │   # get_hsmm_params()         — wrapper: posteriors → HR → sys_interval → log_dur_dists
│   │   │   #   S1=0.116s, S2=0.103s fixed; Systole/Diastole derived from measured intervals
│   │   ├── viterbi.py                  # HSMM Viterbi thuần NumPy (Task 3.10)
│   │   │   # hsmm_viterbi(posteriors, duration_matrix, max_duration, transition_matrix)
│   │   │   # build_duration_matrix()   — log_dur_dists dict → (D_max, N) array
│   │   │   # Vectorised over states; outer loop over time
│   │   └── parallel_hsmm.py            # 4 HSMM topologies song song + confidence (Tasks 3.11–3.12)
│   │       # segment_healthy()         — ω₁: 4-state, S1→Sys→S2→Dia
│   │       # segment_holosystolic()    — ω₂: 4-state, Murmur replaces Systole channel
│   │       # segment_early_systolic()  — ω₃: 5-state, S1→Mur→Sys→S2→Dia, Systole halved
│   │       # segment_mid_systolic()    — ω₄: 5-state, S1→Sys↔Mur→S2→Dia, Systole quartered
│   │       # run_parallel_hsmm()       — chạy cả 4, trả về confs + best murmur model
│   │
│   ├── evaluation/                     # (Trống — dành cho metrics đánh giá)
│   └── visualisation/
│       ├── __init__.py
│       └── style.py                    # Style/màu sắc nhất quán toàn dự án
│           # MURMUR_COLORS     — Present/Unknown/Absent
│           # STATE_COLORS      — S1/Systole/S2/Diastole/Unannotated
│           # LOCATION_COLORS   — AV/PV/TV/MV/Phc
│           # setup_style()     — Cấu hình matplotlib toàn cục
│
├── notebooks/                          # Jupyter notebooks phân tích
│   ├── 01_eda.ipynb                    # Phase 1: Exploratory Data Analysis
│   ├── 02a_preprocessing.ipynb         # Phase 2: Tiền xử lý tín hiệu → spectrograms + labels .npy
│   ├── 02b_feature_correlation.ipynb   # Phase 2: Phân tích tương quan đặc trưng
│   ├── 03_model_reproduction.ipynb     # Phase 3 (local): Kiểm tra RNN architecture, HSMM, inference
│   │   # Tasks 3.1–3.9, 3.11–3.12 (không train, chỉ verify và visualise)
│   ├── 03_train_rnn_colab.ipynb        # Phase 3 (Colab T4): Train 5-fold BiGRU
│   │   # Task 3.6 — lr=1e-3, batch=32, patience=10, seed=42
│   │   # Results: fold val_loss ≈ 0.34–0.41; checkpoints saved to models/rnn/
│   ├── 04_hsmm_inference_colab.ipynb   # Phase 3 (Colab): HSMM inference + evaluation toàn bộ
│   │   # Tasks 3.13–3.15 — run_parallel_hsmm() trên 3163 recordings, patient aggregation, metrics
│   └── data/                           # Symlink/copy dữ liệu dùng trong Colab
│       └── processed/spectrograms/     # Copy spectrograms cho notebook 04
│
├── figures/                            # Hình ảnh xuất ra từ notebooks
│   ├── eda/                            # Biểu đồ EDA (notebook 01)
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
│   ├── preprocessing/                  # Biểu đồ từ notebook 02a
│   │   ├── s2_normalisation.png        # Before/After amplitude normalisation (3×2 grid)
│   │   ├── s3_spectrogram_full.png     # Log-spectrogram 0–2000 Hz, đường 800 Hz cyan
│   │   ├── s4_frequency_crop.png       # So sánh (101 bins) vs (41 bins) sau crop
│   │   ├── s5_zscore_effect.png        # Effect of per-row z-score, murmur bins nổi bật
│   │   ├── s6_pipeline_murmur.png      # Full pipeline end-to-end — mẫu Present
│   │   ├── s6_pipeline_normal.png      # Full pipeline end-to-end — mẫu Absent
│   │   ├── s6_pipeline_unknown.png     # Full pipeline end-to-end — mẫu Unknown
│   │   ├── s7_spectrogram_comparison.png  # So sánh spectrogram 3 class — RNN input thực tế
│   │   └── s8_frequency_content.png       # Mean spectral energy theo freq bin (0–2000 Hz)
│   ├── correlation/                    # Biểu đồ từ notebook 02b
│   │   ├── s1b_metadata_correlation.png    # Cramér's V / η² metadata vs murmur — tất cả < 0.15
│   │   ├── s2c_spectral_discrimination.png # Mean energy (Present vs Absent), Mann-Whitney U
│   │   └── s3d_recording_features.png      # Kruskal-Wallis: duration/annotation/SNR/n_recordings
│   ├── results/                        # Biểu đồ từ Phase 3 (notebook 03 + 04)
│   │   ├── v13_rnn_posteriors_2530_MV.png     # RNN posterior xác suất — bản ghi mẫu
│   │   ├── v13_rnn_posteriors_9979_TV.png
│   │   ├── v13_rnn_posteriors_9983_MV.png
│   │   ├── v13_autocorr_9979_TV.png           # ACF heart rate estimation
│   │   ├── v13_duration_dists_9979_TV.png      # Duration distributions (Springer fractions)
│   │   └── v13_duration_dists_mcdonald_9979_TV.png  # Duration distributions (McDonald absolute)
│   └── improvements/                   # (Trống)
│
├── models/                             # Checkpoint mô hình đã train
│   ├── rnn/
│   │   ├── fold_0_best.pt              # Best val checkpoint fold 0 (val_loss=0.338)
│   │   ├── fold_1_best.pt              # Best val checkpoint fold 1 (val_loss=0.407)
│   │   ├── fold_2_best.pt              # Best val checkpoint fold 2 (val_loss=0.367)
│   │   ├── fold_3_best.pt              # Best val checkpoint fold 3 (val_loss=0.359)
│   │   └── fold_4_best.pt              # Best val checkpoint fold 4 (val_loss=0.396)
│   └── catboost/                       # (Trống — outcome classifier chưa train)
│
├── experiments/                        # Quản lý thí nghiệm
│   ├── configs/                        # (Trống — file cấu hình hyperparameter)
│   ├── logs/                           # (Trống — log training)
│   └── results/                        # Kết quả inference + đánh giá (output của notebook 04)
│       ├── recording_results.csv       # c_mn, c_hat, murmur_model per bản ghi
│       ├── patient_results.csv         # pred (Present/Unknown/Absent), true label per bệnh nhân
│       ├── reproduction_metrics.json   # weighted_accuracy + n_patients (kết quả cuối)
│       └── roc_curve.png               # AUC-ROC curve
│
├── reference_code/                     # Code tham khảo từ McDonald et al. (git submodule)
│   ├── src/
│   │   ├── segmenter.py                # double_duration_viterbi(), segment_*(), get_heart_rate()
│   │   ├── neural_networks.py          # Kiến trúc BiGRU gốc
│   │   └── decision_tree.py            # CatBoost outcome classifier
│   ├── results/                        # Scripts tạo figure cho paper
│   │   ├── murmur_results.py
│   │   ├── heart_rate_estimate.py
│   │   ├── confidences_2D.py
│   │   ├── outcome_scores.py
│   │   ├── outcomes_roc.py
│   │   ├── nn_predictions.py
│   │   ├── reliability_diagram.py
│   │   ├── sample_recordings.py
│   │   ├── utils.py
│   │   └── official_outcome_scores.tsv
│   ├── final_model/                    # Mô hình cuối của nhóm challenge
│   │   ├── settings.json
│   │   ├── recordings.csv
│   │   ├── tree_inputs.csv
│   │   ├── outcome_predictions.csv
│   │   ├── 50260_MV_posteriors.csv
│   │   └── 85203_AV_posterior.csv
│   ├── viterbi_hmm.pyx                 # Cython: HSMM Viterbi decoder (tham chiếu cho viterbi.py)
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
| File `.hea` (header) | 3 163 |
| File `.tsv` (nhãn phân đoạn) | 3 163 |
| Tổng file trong `training_data/` | 10 431 |
| Spectrogram `.npy` (processed) | 3 163 |
| Label `.npy` (processed) | 3 163 |

**Vị trí nghe tim (auscultation locations):** AV (Aortic Valve), PV (Pulmonary Valve), TV (Tricuspid Valve), MV (Mitral Valve), Phc (Phocardiogram)

**Nhãn murmur:** `Present` / `Unknown` / `Absent`

**Nhãn phân đoạn (TSV):** `0` Unannotated · `1` S1 · `2` Systole · `3` S2 · `4` Diastole

**Nhãn 5 lớp (labels.py):** S1=0 · Systole=1 · S2=2 · Diastole=3 · Murmur=4 · Unannotated=-1

---

## Luồng dữ liệu (Data Pipeline)

```
data/raw/training_data/          (dữ liệu thô CirCor DigiScope)
        │
        ▼  src/data/parse_metadata.py
data/metadata/patients.csv       (942 × N cột bệnh nhân, đã làm sạch)
data/metadata/recordings.csv     (~3163 × 6 cột, kèm duration)
data/metadata/cv_splits.json     (5-fold stratified splits, patient level)
        │
        ▼  notebooks/02a_preprocessing.ipynb
        ▼  src/features/normalisation.py  → normalise_amplitude()
        ▼  src/features/spectrogram.py    → compute_log_spectrogram()
        ▼                                    crop_frequency()  [101→41 bins]
        ▼                                    zscore_per_row()
        ▼  src/data/labels.py             → load_labels()  [TSV → 5-state]
data/processed/spectrograms/     (41, T) float32 .npy per recording
data/processed/labels/           (T,) int8 .npy per recording
        │
        ▼  notebooks/03_train_rnn_colab.ipynb  (Colab T4 GPU)
        ▼  src/models/rnn.py  — MurmurRNN: BiGRU(41,60,3L) + FC(→5)
        ▼  src/data/loader.py — PCGDataset + pcg_collate_fn
models/rnn/fold_{0..4}_best.pt   (5 checkpoint, 5-fold CV)
        │
        ▼  notebooks/04_hsmm_inference_colab.ipynb  (Colab)
        ▼  src/models/hsmm.py         — HR + systolic interval estimation
        ▼  src/models/viterbi.py      — HSMM Viterbi (pure NumPy)
        ▼  src/models/parallel_hsmm.py — 4 topologies + confidence C(ω)
experiments/results/
    recording_results.csv        (c_mn, c_hat per recording)
    patient_results.csv          (Present/Unknown/Absent per patient)
    reproduction_metrics.json    (weighted_accuracy)
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

---

## Tiến độ

| Phase | Nội dung | Trạng thái |
|-------|----------|-----------|
| Phase 1 | EDA — phân tích dữ liệu thô | ✅ Hoàn thành |
| Phase 2 | Tiền xử lý — spectrogram + label .npy | ✅ Hoàn thành |
| Phase 3 | Model reproduction — BiGRU + HSMM inference | ✅ Hoàn thành |
| Phase 4 | Cải tiến — LR schedule, augmentation, v.v. | ⬜ Chưa bắt đầu |
