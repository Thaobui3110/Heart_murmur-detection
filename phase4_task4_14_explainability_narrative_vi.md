# Task 4.14 — Explainability Narrative (Tiếng Việt)
# Nháp nội dung cho Mục 6.9 của Báo Cáo (Phase 6)
# *(Số mục sẽ được xác nhận khi viết outline báo cáo trong Phase 6)*

---

## 6.9 Giải Thích Mô Hình & Phân Tích Đặc Trưng

### 6.9.1 Chiến Lược Explainability

Dự án này sử dụng phương pháp **interpretability-by-design** (khả năng giải
thích được tích hợp vào kiến trúc) thay vì các phương pháp post-hoc như
SHAP hay LIME. Lựa chọn này xuất phát từ bản chất kiến trúc của pipeline:
khác với các mô hình phân loại dạng hộp đen ánh xạ trực tiếp đặc trưng
thô sang dự đoán, pipeline RNN + parallel HSMM tạo ra các đầu ra trung
gian có ý nghĩa ở mỗi bước — posterior RNN theo từng trạng thái, điểm
confidence từng topology, và đường giải mã phân đoạn. Những biểu diễn
trung gian này mang ý nghĩa lâm sàng trực tiếp (các trạng thái chu kỳ tim,
loại timing tiếng thổi, chất lượng phân đoạn), khiến chúng mang tính
giải thích tốt hơn so với bản đồ attribution post-hoc áp dụng lên
một embedding đã học.

Phân tích được tổ chức theo ba cấp độ: **explainability toàn cục**
(mô hình hoạt động như thế nào trên toàn bộ dataset?), **explainability
cục bộ** (mô hình quyết định như thế nào trên một bản ghi cụ thể?),
và **explainability dựa trên lỗi** (mô hình sai ở đâu và tại sao?).

---

### 6.9.2 Explainability Toàn Cục — Pipeline Quyết Định Như Thế Nào?

#### 6.9.2.1 Không Gian Quyết Định Hai Chiều

Hình s3 tái tạo PLOS Fig. 4, visualize logic phân loại của pipeline
trong không gian confidence hai chiều: C(M−N) trên trục x (confidence
murmur trừ confidence normal) và C(ω̂) trên trục y (confidence phân
đoạn tối đa qua bốn topology HSMM).

Ba cụm riêng biệt xuất hiện từ scatter plot này. Bệnh nhân Present
(n=179) tập trung ở vùng bên phải nơi C(M−N) > 0, phản ánh bằng chứng
tiếng thổi rõ ràng trong tín hiệu. Bệnh nhân Absent (n=695) cụm ở
góc phần tư trên-trái — C(ω̂) > 0.65 cao nhưng C(M−N) âm. Bệnh nhân
Unknown (n=68) phân bố dọc theo phần dưới của biểu đồ nơi C(ω̂) < 0.65,
bất kể dấu của C(M−N).

Cấu trúc hai chiều này giải thích tại sao pipeline cần hai độ đo
confidence riêng biệt thay vì một: C(M−N) đơn độc không thể phân biệt
Unknown với Absent, vì cả hai lớp đều có C(M−N) âm hoặc gần bằng 0.
Ngưỡng C(ω̂) = 0.65 hoạt động như một cổng chất lượng tín hiệu, từ
chối phân loại khi HSMM không thể phân đoạn chu kỳ tim với độ tin cậy
đủ cao.

#### 6.9.2.2 Tầm Quan Trọng Của Từng Dải Tần

Hình s5 trình bày kết quả của nghiên cứu frequency-band ablation:
với mỗi trong 41 bin tần số (0–800 Hz, bước 20 Hz), bin đó được
zeroing out trong spectrogram và mức giảm C(M−N) kết quả được đo
trên 50 bản ghi validation Present.

Profile ablation cho thấy đỉnh rõ ràng tại **140 Hz**
(importance = 0.0114), với cụm importance cao nhất trải rộng 100–180 Hz.
Điều này nhất quán với các đặc trưng phổ tần đã biết của tiếng thổi tim,
tập trung năng lượng trong dải tần thấp tương ứng với dòng máu chảy
rối qua van tim bị hẹp.

Điều quan trọng là profile importance post-model này nhất quán với
phân tích correlation pre-model từ Phase 2 (Task 2.5c), xác định cùng
dải 100–200 Hz là dải phân biệt nhất giữa bệnh nhân Present và Absent
sử dụng rank-biserial correlation trên spectrogram thô. Sự hội tụ của
hai phân tích độc lập — một dựa trên thống kê dữ liệu, một dựa trên
hành vi mô hình — cung cấp bằng chứng mạnh rằng RNN đã học cách khai
thác vùng tần số có liên quan lâm sàng.

Một quan sát thứ yếu là importance không giảm về 0 phía trên 400 Hz,
với đỉnh cục bộ tại 460 Hz và tăng nhẹ gần 760–800 Hz. Kết hợp với
kết quả Phase 2 cho thấy 30 bins trên 800 Hz vẫn có ý nghĩa thống kê
sau hiệu chỉnh Bonferroni, điều này gợi ý ngưỡng cắt 800 Hz là
conservative và có thể bỏ qua thông tin phân biệt — phát hiện này
trực tiếp thúc đẩy thí nghiệm mở rộng dải tần trong Phase 5.

---

### 6.9.3 Explainability Cục Bộ — Pipeline Quyết Định Thế Nào Trên Từng Bản Ghi?

#### 6.9.3.1 Walkthrough Phân Đoạn (Ba Bản Ghi Tracking)

Hình s4 tái tạo PLOS Fig. 5, hiển thị ví dụ phân đoạn 4 panel cho
ba bản ghi tracking được chọn từ Phase 1.

Với **2530_MV** (nhãn thực: Absent), posterior RNN luân phiên đều đặn
giữa S1, Systole, S2 và Diastole mà không có kích hoạt Murmur. HSMM
chọn topology Healthy (C(M−N) = −0.069) và đường giải mã của nó khớp
chặt với annotation ground truth — ví dụ điển hình về chu kỳ tim bình
thường được phân loại đúng.

Với **9979_TV** (nhãn thực: Present, Grade 3 Holosystolic), posterior
cho thấy kích hoạt trạng thái Murmur liên tục thay thế Systole xuyên
suốt bản ghi, nhất quán với tiếng thổi holosystolic lấp đầy toàn bộ
khoảng systole. HSMM chọn đúng topology Holosystolic (C(M−N) = +0.223).
Annotation ground truth, chỉ có trong 5 giây đầu, xác nhận Murmur
trong systole.

Với **9983_MV** (nhãn thực: Unknown), posterior hỗn loạn không có
pattern ổn định, phản ánh chất lượng tín hiệu kém được ghi nhận trong
Phase 1 cho bản ghi Unknown (annotation coverage 24%, SNR 5.7 dB).
Ground truth hầu như hoàn toàn không có annotation. C(ω̂) = 0.653 nằm
ngay trên ngưỡng Unknown, dẫn đến dự đoán Absent — trường hợp cơ chế
abstention của pipeline gần như được kích hoạt.

#### 6.9.3.2 So Sánh Paths Parallel HSMM

Hình s6 khảo sát bệnh nhân 84786 (Grade 1 Early-systolic, false negative),
hiển thị tất cả bốn topology paths HSMM được đánh giá song song cùng
với posterior RNN.

Với bản ghi **84786_AV**, topology Healthy thắng với confidence 0.7756
so với Early-systolic ở 0.7363 — margin 0.039. Posterior RNN cho thấy
kích hoạt Murmur không liên tục, không đủ để vượt qua lợi thế likelihood
của Healthy path. Với bản ghi **84786_PV**, margin thu hẹp đến gần bằng
không: Healthy và Early-systolic tie tại 0.6815, pipeline chọn Healthy
theo quy tắc tie-breaking. Case gần-miss này minh họa độ nhạy cảm của
ranh giới quyết định với tiếng thổi grade thấp, nơi posterior Murmur
của RNN có mặt nhưng quá yếu và không nhất quán để đủ nghiêng cán cân.

---

### 6.9.4 Explainability Dựa Trên Lỗi — Pipeline Sai Ở Đâu Và Tại Sao?

#### 6.9.4.1 False Negative: Thất Bại Theo Grade

Hình s7 và phân tích grade trong Hình s9 tiết lộ pattern nổi bật:
**tất cả 13 bệnh nhân false negative là Grade 1**, không có false
negative nào trong Grade 2 hay Grade 3. Sensitivity theo grade (87.5%
/ 100% / 100%) khớp hoàn hảo với PLOS 2024.

Thất bại theo grade này có giải thích cơ chế có thể truy nguyên về
pipeline tiền xử lý. Z-score normalisation per-row (Phase 2, Task 2.4)
loại bỏ năng lượng tuyệt đối khỏi mỗi frame spectrogram, cân bằng
dynamic range qua các bản ghi. Với tiếng thổi Grade 3 có biên độ lớn,
normalisation này bảo tồn tương phản tương đối giữa các frame murmur
và non-murmur. Với tiếng thổi Grade 1 có biên độ thấp, cùng normalisation
đó khiến các frame murmur không phân biệt được với noise sau chuẩn hóa.
Nghiên cứu ablation (Hình s5) xác nhận dải 100–200 Hz là vùng thông
tin nhất, nhưng ngay cả signal này cũng không đủ cho Grade 1 khi năng
lượng tuyệt đối đã bị loại bỏ bởi z-score.

Case study 84786_PV (Hình s8) điển hình cho failure mode này: near-tie
giữa topology Healthy và Early-systolic, với posterior Murmur xuất hiện
ngắt quãng nhưng không bao giờ duy trì đủ lâu để HSMM commit.

#### 6.9.4.2 False Positive: Gần Ngưỡng Quyết Định

131 bệnh nhân false positive (Absent được phân loại là Present) có
điểm chung: giá trị C(M−N) của họ tập trung ngay trên 0 (0.00–0.05,
Hình s7). Điều này khác biệt về cấu trúc so với cụm true positive
tập trung tại C(M−N) > 0.10. Phân bố FP gợi ý những bản ghi Absent
này chứa các âm thanh sinh lý hoặc môi trường — như split S2, friction
rub, hoặc artifact chuyển động — tạo ra năng lượng phổ tần giống murmur
trong dải 100–200 Hz, đủ để đẩy C(M−N) vượt ngưỡng một chút.

Case study 84969_MV (Hình s8) minh họa điều này: bản ghi Absent
có biên độ cao với các burst năng lượng không đều mà RNN diễn giải
là Murmur signal ngắt quãng, dẫn đến C(M−N) = +0.167 và phân loại
Present sai.

#### 6.9.4.3 Lớp Unknown: Chất Lượng Tín Hiệu Là Nguyên Nhân Gốc

Hình s12 cho thấy bệnh nhân Unknown có median C(ω̂) là 0.672 — ngay
cạnh ngưỡng phân loại 0.65 — so với 0.800 cho Present và 0.810 cho
Absent. Chỉ 36.8% bệnh nhân Unknown nằm dưới ngưỡng (kích hoạt đúng
cơ chế abstention), trong khi 63.2% còn lại bị phân loại sai thành
Present hoặc Absent.

Phân bố này phản ánh phát hiện Phase 1 rằng bệnh nhân Unknown có
annotation coverage thấp hơn đáng kể (24% so với 60% cho Present)
và SNR thấp hơn (5.7 dB so với 9.5 dB). Chất lượng tín hiệu kém
làm giảm confidence phân đoạn HSMM, nhưng không đủ nhất quán để
kích hoạt quy tắc abstention C(ω̂) < 0.65 trên tất cả bản ghi Unknown.

---

### 6.9.5 Calibration

Hình s11 hiển thị reliability diagram cho C(ω̂) so với binary
classification accuracy (chỉ bản ghi Present/Absent). ECE = 0.059
cho thấy calibration ở mức trung bình. Pipeline được calibrated tốt
trong dải C(ω̂) = 0.65–0.80 — vùng liên quan nhất cho triển khai lâm
sàng — nhưng hơi underconfident ở giá trị thấp và hơi overconfident
trên 0.80. Hàm ý thực tế là điểm C(ω̂) trong dải được calibrated tốt
có thể được truyền đạt đến bác sĩ lâm sàng như ước tính xác suất có
ý nghĩa, trong khi các dự đoán có confidence cao (C(ω̂) > 0.80) có
thể cần điều chỉnh giảm nhẹ.

---

### 6.9.6 Tóm Tắt

Phân tích explainability ba cấp độ hội tụ về một bức tranh nhất quán
về hành vi của pipeline. Ở cấp toàn cục, mô hình xác định đúng dải
tần 100–200 Hz là vùng thông tin nhất, nhất quán với cả kiến thức lâm
sàng về âm học tiếng thổi tim và phân tích dữ liệu Phase 2. Ở cấp
cục bộ, pipeline cung cấp các đường quyết định minh bạch thông qua
các đầu ra trung gian, cho phép giải thích có thể diễn giải được với
bác sĩ lâm sàng cho từng phân loại. Ở cấp lỗi, các thất bại mang
tính hệ thống chứ không ngẫu nhiên: tất cả tiếng thổi bị bỏ sót là
Grade 1, tất cả false positive là bản ghi Absent gần ngưỡng, và phân
loại sai Unknown được giải thích bằng chất lượng tín hiệu nằm ngay
trên ranh giới abstention. Những phát hiện này trực tiếp định hướng
các thí nghiệm cải tiến có mục tiêu được đề xuất trong Phase 5.

---

## Danh Sách Hình Được Tham Chiếu

| Hình | File | Task |
|------|------|------|
| s3 | figures/results/s3_hsmm_confidence_scatter.png | 4.3 |
| s4 | figures/results/s4_segmentation_examples.png | 4.4 |
| s5 | figures/explainability/s5_frequency_importance.png | 4.5 |
| s6 | figures/explainability/s6_hsmm_path_comparison.png | 4.6 |
| s6b | figures/explainability/s6b_hsmm_path_comparison_84786_PV.png | 4.6 |
| s7 | figures/results/s7_error_fp_fn.png | 4.7 |
| s8 | figures/explainability/s8_case_study_*.png (×5) | 4.8 |
| s9 | figures/results/s9_grade_performance.png | 4.9 |
| s11 | figures/results/s11_reliability_diagram.png | 4.11 |
| s12 | figures/results/s12_unknown_analysis.png | 4.12 |

## Số Từ
~1100 từ tiếng Việt (target: 800–1200 từ) ✅
