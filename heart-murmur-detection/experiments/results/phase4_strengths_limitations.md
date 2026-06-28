# Phase 4 — Strengths & Limitations Analysis

## Điểm Mạnh

**1. Sensitivity Present xuất sắc (92.7%)**
Pipeline bỏ sót chỉ 13/179 bệnh nhân có tiếng thổi tim — khớp hoàn hảo
với PLOS 2024 (delta = 0.000, Task 4.2). Trong bối cảnh lâm sàng, đây là
chỉ số quan trọng nhất: không bỏ sót bệnh nhân cần được chẩn đoán.

**2. AUC-ROC vượt paper gốc (0.952 vs 0.947)**
Khả năng phân tách Present/không-Present ở mức threshold-free cao hơn
báo cáo của tác giả +0.005 (Task 4.2). Cho thấy reproduction thành công
và pipeline tổng quát hóa tốt trên cross-validation.

**3. Khả năng giải thích tốt qua intermediate outputs**
Pipeline không phải black-box — mỗi bước tạo ra output có ý nghĩa lâm sàng:
- C(M−N) giải thích quyết định murmur/normal (Task 4.3, Fig s3)
- Posterior RNN theo thời gian cho thấy "mô hình nghĩ gì" (Task 4.4, Fig s4)
- 4 topology paths giải thích timing detection (Task 4.6, Fig s6)

**4. Tự động nhận diện timing type tiếng thổi**
Parallel HSMM chọn đúng Holosystolic cho 9979_TV (Task 4.4) và
Early-systolic cho 84786 (Task 4.6) — không cần label timing khi inference.
Đây là lợi thế so với pipeline single-topology.

**5. Calibration chấp nhận được (ECE = 0.059)**
Ở vùng C(ω̂) > 0.65 (phần lớn recordings), mô hình calibrated tốt.
Có thể tin tưởng confidence score cho clinical decision support (Task 4.11).

---

## Hạn Chế

**1. Sensitivity Unknown thấp (19.1% vs 30.9% PLOS)**
55/68 bệnh nhân Unknown bị phân loại sai (Task 4.2, 4.12).
Nguyên nhân: (a) Unknown là class khó về định nghĩa, (b) LR=1e-3 thay vì
1e-4 làm RNN chưa hội tụ tối ưu cho class thiểu số này.
→ Phase 5: thử LR=1e-4 và threshold tuning C(ω̂) từ 0.65 xuống 0.55.

**2. 100% False Negative là Grade 1 — pipeline mù với murmur nhẹ**
13/13 FN có systolic_murmur_grading = 1.0, 10/13 là Early-systolic (Task 4.7, 4.9).
Nguyên nhân: z-score per-row normalize away absolute energy — Grade 1
amplitude thấp bị normalize giống noise.
→ Phase 5: data augmentation hoặc global normalization thay z-score.

**3. False Positive cao từ Absent (131/695 = 18.8%)**
C(M−N) của FP tập trung ngay sát threshold 0 (Task 4.7).
→ Phase 5: tăng threshold lên 0.02–0.05.

**4. 800 Hz frequency cutoff conservative**
Ablation peak importance không về 0 ở 400–800 Hz, secondary peak tại 460 Hz
(Task 4.5). Phase 2 xác nhận 30 bins trên 800 Hz vẫn significant.
→ Phase 5: mở rộng frequency range lên 1000–1200 Hz.

**5. Tốc độ Viterbi giới hạn khả năng experiment**
~2.5s/recording, không dùng GPU (Task 4.5).
→ Phase 5: vectorise hsmm_viterbi() bằng NumPy broadcasting.

---

## Tóm Tắt

| | Điểm | Evidence | Phase 5 Action |
|---|------|----------|----------------|
| ✅ | Sensitivity Present = 92.7% | Task 4.2 | — |
| ✅ | AUC = 0.952 > PLOS 0.947 | Task 4.2 | — |
| ✅ | Interpretable pipeline | Tasks 4.3, 4.4, 4.6 | — |
| ✅ | Auto timing detection | Task 4.6 | — |
| ✅ | ECE = 0.059 | Task 4.11 | — |
| ❌ | Unknown sensitivity 19.1% | Task 4.2, 4.12 | LR=1e-4, threshold tuning |
| ❌ | 100% FN là Grade 1 | Task 4.7, 4.9 | Data augmentation |
| ❌ | FP rate 18.8% từ Absent | Task 4.7 | Threshold C(M−N) > 0.02 |
| ❌ | 800 Hz cutoff conservative | Task 4.5, Phase 2 | Mở rộng lên 1200 Hz |
| ❌ | Viterbi bottleneck | Task 4.5 | Vectorise NumPy |
