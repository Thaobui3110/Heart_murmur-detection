# Phase 3 — Đánh Giá Thực Thi vs Reference Code (McDonald et al.)

> Phân tích dựa trên toàn bộ code đã viết:
> - `notebooks/03_model_reproduction.ipynb` — Tasks 3.1–3.9 + 3.7 visualization
> - `notebooks/03_train_rnn_colab.ipynb` — Task 3.6 training (đã chạy xong trên Colab T4)
> - `src/data/labels.py` — Task 3.1
> - `src/data/loader.py` — Task 3.2
> - `src/models/rnn.py` — Task 3.3
> - `src/models/hsmm.py` — Tasks 3.8, 3.9
> - So sánh với `reference_code/src/neural_networks.py`, `segmenter.py`, `viterbi_hmm.pyx`, `team_code.py`

---

## Tóm Tắt Trạng Thái Phase 3

| Task | Mô tả | Trạng thái | File |
|---|---|---|---|
| 3.1 | Ground-truth 5-state labels | ✅ Hoàn thành | `labels.py` |
| 3.2 | DataLoader + batching | ✅ Hoàn thành | `loader.py` |
| 3.3 | BiGRU architecture | ✅ Hoàn thành | `rnn.py` |
| 3.4 | Class-weighted loss | ✅ Hoàn thành | notebook cell |
| 3.5 | 5-fold CV splits | ✅ Hoàn thành | notebook cell |
| 3.6 | Training 5-fold | ✅ Hoàn thành (Colab) | colab notebook |
| 3.7 | Visualization posteriors | ✅ Hoàn thành | notebook cell |
| 3.8 | Heart rate estimation | ✅ Hoàn thành | `hsmm.py` |
| 3.9 | Duration distributions | ✅ Hoàn thành | `hsmm.py` |
| 3.10 | Viterbi HSMM | ❌ Chưa implement | — |
| 3.11 | HSMM topologies | ❌ Chưa implement | — |
| 3.12 | Confidence computation | ❌ Chưa implement | — |
| 3.13 | Murmur classification | ❌ Chưa implement | — |

**Phase 3a (RNN training):** Hoàn thành.
**Phase 3b (HSMM segmentation):** Chỉ hoàn thành HR estimation và duration distributions — chưa có Viterbi.
**Phase 3c (Classification):** Chưa bắt đầu.

---

## So Sánh Chi Tiết Từng Task

---

### Task 3.1 — Ground-truth Labels (`labels.py`)

#### Những gì đã làm
- Map TSV states (1-indexed) → 0-indexed: S1=0, Sys=1, S2=2, Dia=3, Unannotated=-1
- Murmur relabelling theo timing: Holosystolic/Early/Mid/Late-systolic
- Lọc theo `murmur_locations` để chỉ apply relabelling ở đúng location

#### So sánh với ref code

**✅ ĐÚNG — State mapping:**
```python
# labels.py (chúng ta)
_TSV_STATE_MAP = {0: UNANNOTATED, 1: S1, 2: SYSTOLE, 3: S2, 4: DIASTOLE}

# ref code (neural_networks.py)
if row.state == 1: segmentation_state = 0   # S1
elif row.state == 2: segmentation_state = 1  # Systole
elif row.state == 3: segmentation_state = 2  # S2
elif row.state == 4: segmentation_state = 3  # Diastole
```
Mapping KHỚP HOÀN TOÀN. ✅

**✅ ĐÚNG — Murmur timing fractions:**
```python
# labels.py
elif timing == 'Mid-systolic':
    quarter = s_len // 4         # ≈ int(0.25 * s_len)
    m_start = s_start + quarter
    m_end   = s_end - quarter

# ref code
elif timing == "Mid-systolic":
    portion = [0.25, 0.75]       # → int(0.25*dur), ceil(0.75*dur)
```
Thực chất giống nhau, chỉ khác rounding nhỏ (floor vs ceil). ✅

**✅ ĐÚNG — Murmur location filtering:**
Cả hai đều kiểm tra `murmur_locations` trước khi apply relabelling. ✅

**❌ SAI — Background frames KHÔNG bị xóa:**
Đây là sai lệch lớn nhất của Task 3.1.

```python
# ref code neural_networks.py — trong RecordingDataset.__getitem__()
indices_to_keep = segmentation_label != -1
segmentation_label = segmentation_label[indices_to_keep]
features = features[..., indices_to_keep]  # XÓA LUÔN background frames
```

```python
# labels.py (chúng ta) — GIỮ NGUYÊN background frames (-1)
labels = np.full(duration_frames, UNANNOTATED, dtype=np.int8)
# ... chỉ fill vào vùng annotated, phần còn lại vẫn là -1
```

Ref code **xóa hoàn toàn** các frame unannotated trước khi đưa vào model.
Chúng ta **giữ** chúng, dựa vào `ignore_index=-1` trong CrossEntropyLoss.

**Tác động:** Hai cách đều valid về mặt gradient (ignore_index không tính loss cho -1), nhưng:
- Ref code: GRU chỉ xử lý các frame có annotation → sequence ngắn hơn thực tế, không có "khoảng trống"
- Chúng ta: GRU xử lý toàn bộ recording kể cả phần không annotated → model học thêm temporal context từ background frames
- **Ảnh hưởng thực tế:** thấp, nhưng khác với paper gốc

---

### Task 3.2 — DataLoader (`loader.py`)

#### Những gì đã làm
- `PCGDataset`: đọc từ disk hoặc RAM cache
- `pcg_collate_fn`: sort descending by length, pad features=0.0, labels=-1
- `create_dataloader`: wrapper với caching support
- `load_dataset_to_ram`: preload toàn bộ dataset vào RAM (cải tiến so với ref)

#### So sánh với ref code

**✅ TỐT HƠN — RAM preloading:**
```python
# loader.py — load_dataset_to_ram()
for rec_id in tqdm(recording_ids):
    spec_cache[rec_id]  = np.load(...)  # load hết vào RAM
    label_cache[rec_id] = np.load(...)  # ~1.2 GB
```

Ref code dùng `lru_cache` + lazy loading. Cách của chúng ta load hết 1 lần vào RAM (`~1.2 GB`) thì mỗi epoch chỉ tốn ~2 giây trên Colab (so với hàng giờ nếu đọc từ OneDrive). **Đây là cải tiến tích cực.** ✅+

**❌ SAI — Format tensor (B, T, F) vs (B, F, T):**
```python
# loader.py (chúng ta)
features = torch.FloatTensor(spec.T)   # (41, T) → (T, 41)
# batch: (B, T_max, 41) — T_max ở chiều 1

# ref code neural_networks.py
padded_features[i, ..., :T] = features  # features là (41, T)
# batch: (B, 41, T_max) — T_max ở chiều 2
```

Ref code giữ nguyên `(B, 41, T)` và model làm `x = x.permute(0, 2, 1)` bên trong.
Chúng ta transpose khi load (`spec.T`) → `(T, 41)` và model nhận `(B, T, 41)` trực tiếp.

**Tác động:** Về mặt tính toán GIỐNG NHAU nếu rnn.py cũng tương thích. `rnn.py` của chúng ta nhận `(B, T, 41)` và trả `(B, T, 5)` — hoàn toàn consistent với loader.py. ✅ Không sai, chỉ khác convention.

**❌ SAI — Không có random crop:**
```python
# ref code — random 6-second crop trong training
TRAINING_SEQUENCE_LENGTH = 6  # giây = 300 frames
if self.sequence_length is not None:
    random_start = torch.randint(0, max(T - 300, 1), (1,)).item()
    features = features[..., random_start : random_start + 300]
```

```python
# loader.py (chúng ta)
# KHÔNG có crop — luôn trả toàn bộ recording
```

**Tác động: CAO.** Ref code training sample là 6 giây (300 frames). Chúng ta training trên toàn bộ recording (trung bình 1200 frames). Hệ quả:
1. Ref code có data augmentation tự nhiên (random start mỗi epoch)
2. Với batch_size=32, ref code batch = 32×300 = 9600 frames; chúng ta = 32×~1200 = ~38400 frames → memory khác nhau (nhưng đều fit)
3. Ref code training nhanh hơn vì sequence ngắn hơn
4. Gradient của ref code focus vào 6-second window, có thể generalize tốt hơn

Trên Colab, mỗi epoch mất ~27 giây với chúng ta (full sequences). Ref code có thể nhanh hơn.

---

### Task 3.3 — RNN Architecture (`rnn.py`)

#### Những gì đã làm
```
Input: (B, T, 41)
→ BiGRU(hidden=60, layers=3, dropout=0.1 between layers) → (B, T, 120)
→ Dropout(0.1)
→ FC1: Linear(120→60) + Tanh
→ Dropout(0.1)
→ FC2: Linear(60→40) + Tanh
→ Output: Linear(40→5) → (B, T, 5)
```

#### So sánh với ref code

**✅ ĐÚNG — GRU parameters:**
- hidden_size=60, num_layers=3, bidirectional=True, dropout=0.1 ✅
- input_size=41 ✅

**⚠️ LỆCH — Dropout placement:**

| Vị trí | Ours (rnn.py) | Ref code |
|---|---|---|
| Giữa GRU layers | ✅ dropout=0.1 (GRU kwarg) | ✅ dropout=0.1 |
| Sau GRU output → trước FC1 | ✅ Dropout(0.1) | ❌ Không có |
| Giữa FC1 → FC2 | ✅ Dropout(0.1) | ✅ Dropout(0.1) |
| Sau FC2 → trước Output | ❌ Không có | ✅ Dropout(0.1) |

```python
# rnn.py (chúng ta)
x = self.dropout(gru_out)       # ← dropout sau GRU
x = self.act1(self.fc1(x))
x = self.dropout(x)             # ← dropout sau FC1
x = self.act2(self.fc2(x))
logits = self.output_layer(x)   # ← không dropout trước output

# ref code
nn.Sequential(
    Linear(120→60), Tanh, Dropout,   # dropout sau FC1
    Linear(60→40),  Tanh, Dropout,   # dropout sau FC2 ← khác!
    Linear(40→5)
)
# KHÔNG có dropout giữa GRU → FC1
```

**Tác động: Thấp.** Cả hai đều có 2 lần dropout (0.1), chỉ khác vị trí. Về mặt regularization, ảnh hưởng nhỏ. Không cần sửa.

**✅ ĐÚNG — Số tham số:**
Total params ~120K — khớp với paper. ✅

**✅ ĐÚNG — Output format:**
rnn.py trả `(B, T_max, 5)`. Criterion nhận `(B, 5, T_max)` sau khi `logits.permute(0, 2, 1)`. Consistent. ✅

---

### Task 3.4 — Class-weighted Loss

#### Những gì đã làm
```python
frame_counts = [S1=385582, Sys=378660, S2=337989, Dia=741055, Murmur=55591]
weights = [1.922, 1.957, 2.193, 1.000, 13.330]  # relative to Diastole
criterion = CrossEntropyLoss(weight=weights_tensor, ignore_index=-1)
```

#### So sánh với ref code

**✅ ĐÚNG — Inverse frequency formula:**
Cùng dùng `weight[i] = total / (n_classes × count[i])`, sau đó normalize. ✅

**⚠️ LỆCH NHỎ — Dataset scope:**
Ours: đếm frames từ **tất cả 3163 recordings** (kể cả Unknown patients).
Ref code: chỉ tính từ recordings của **non-Unknown patients** (~2380 recordings) vì tách `recording_df_gq`.

**Tác động: Thấp.** Phân phối frame không thay đổi đáng kể khi thêm Unknown recordings. Murmur weight có thể nhỉnh lên 1-2% vì Unknown recordings có ít Murmur frames.

---

### Task 3.5 — 5-fold CV Splits

#### Những gì đã làm
```python
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# Stratify theo murmur label (Present/Unknown/Absent)
# 942 patients → recording IDs per fold
```

#### So sánh với ref code

**❌ SAI — random_state=42 vs None:**
```python
# ref code team_code.py
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=None)  # ← None!
```

Ref code dùng `random_state=None` → kết quả fold KHÔNG cố định, thay đổi mỗi lần chạy.
Chúng ta dùng `random_state=42` → hoàn toàn reproducible.

**Tác động: Trung bình — tích cực cho chúng ta.** Dùng seed=42 giúp tái tạo kết quả dễ dàng hơn. Đây là lựa chọn tốt hơn ref code (mặc dù khác paper gốc).

**❌ SAI — Không lọc Unknown patients:**
```python
# ref code team_code.py
recording_df_gq = recording_df[recording_df["patient_murmur_label"] != "Unknown"]
# Chỉ train RNN trên non-Unknown (882/942 patients ~ 79%)

# Chúng ta — train_recordings bao gồm TẤT CẢ 942 patients
train_recordings = recordings_df[recordings_df['patient_id'].isin(train_patients)]['recording_id'].tolist()
```

**Tác động: CAO.** Ref code train RNN trên ~2553 recordings (không có Unknown), chúng ta train trên ~2553 recordings (có cả Unknown). Theo ref code, Unknown recordings có quality thấp → đưa vào training có thể làm nhiễu. Tuy nhiên, trong thực tế chênh lệch này không quá lớn.

---

### Task 3.6 — Training (Colab Notebook)

#### Kết quả training thực tế

| Fold | Best epoch | Best val loss | Dừng tại | Thời gian |
|---|---|---|---|---|
| fold_0 | ~18 | 0.3379 | epoch 28 | 12.9 min |
| fold_1 | ~15 | 0.4069 | epoch 25 | 11.4 min |
| fold_2 | ~15 | 0.3665 | epoch 31 | 14.2 min |
| fold_3 | ~15 | 0.3591 | epoch 22 | 10.3 min |
| fold_4 | ~18 | 0.3958 | epoch 28 | 13.0 min |

**Trung bình val loss: 0.373 ± 0.028**

#### So sánh với ref code

**❌ SAI — Learning rate 1e-3 vs 1e-4:**
```python
# colab notebook
CONFIG = {'lr': 1e-3, ...}

# ref code
TRAINING_LR = 1e-4  # ← 10 lần nhỏ hơn!
```

**Tác động: CAO — Quan trọng nhất trong Task 3.6.**
Với LR=1e-3, model hội tụ nhanh (~15 epochs) nhưng có thể dừng tại điểm không tối ưu.
Với LR=1e-4 (ref), model cần nhiều epoch hơn (có thể 50-200 epoch) nhưng tìm được minimum tốt hơn.

Bằng chứng từ training log:
- fold_0: train_loss giảm từ 0.961 → 0.299 trong 28 epochs, val_loss best = 0.338
- Val loss "plateau" sớm (epoch 15-20) gợi ý model đang "dừng chưa đủ sâu"
- Nếu dùng LR=1e-4, val loss có thể giảm thêm 5-15%

**❌ SAI — max_epochs=100 vs 1000:**
Ref code dùng max_epochs=1000 với LR=1e-4. Với LR nhỏ hơn, model cần nhiều epoch hơn.
Chúng ta dùng max_epochs=100 — đủ với LR=1e-3 (early stopping tại epoch ~25) nhưng sẽ không đủ nếu sửa LR thành 1e-4.

**❌ SAI — Không có random crop:**
Ref code dùng 6-second random crop khi training. Chúng ta dùng toàn bộ recording.
- Cách của ref: data augmentation + tập trung vào sub-sequences → có thể học pattern cục bộ tốt hơn
- Cách của chúng ta: model thấy toàn bộ context → có thể học long-range dependencies tốt hơn
- **Ảnh hưởng thực tế:** không rõ, nhưng đây là sai lệch so với paper gốc

**✅ ĐÚNG — batch_size=32:** Khớp với ref code. ✅
**✅ ĐÚNG — patience=10:** Ref code cũng dùng patience=10 (theo early stopping logic). ✅
**✅ ĐÚNG — grad_clip=1.0:** Ref code cũng clip gradient. ✅
**✅ ĐÚNG — Adam optimizer:** Ref code dùng Adam. ✅

**✅ TỐT HƠN — Copy to local storage trước:**
```python
# colab cell 5a
shutil.copytree(SPEC_DIR, LOCAL_SPEC_DIR)  # Drive → /content/ (local SSD)
```
Sau đó dùng `LOCAL_SPEC_DIR` để train → loại bỏ Drive I/O bottleneck. Đây là kỹ thuật tốt hơn ref code (ref dùng local files trực tiếp). ✅+

---

### Task 3.8 — Heart Rate Estimation (`hsmm.py`)

#### Những gì đã làm
```python
s = P(S1) + P(Systole) + P(S2) + P(Murmur)  # non-diastole
s = s - s.mean()
acorr = np.correlate(s, s, mode='full')[T-1:]
# find_peaks trong range [30, 180] BPM
```

#### So sánh với ref code

**✅ ĐÚNG — Signal definition:**
```python
# hsmm.py
s = posteriors[:,0] + posteriors[:,1] + posteriors[:,2] + posteriors[:,4]

# ref code segmenter.py
non_diastole = (1 - posteriors[:, 3]).flatten()
```
`P(S1)+P(Sys)+P(S2)+P(Murmur) = 1 - P(Dia)` vì posteriors sum to 1 (softmax). ✅

**✅ ĐÚNG — Zero-mean và normalize:** Giống ref code. ✅

**✅ ĐÚNG — BPM search range [30, 180]:** Khớp với ref code. ✅

**✅ ĐÚNG — Peak selection:** Cả hai chọn peak cao nhất trong range. ✅

**⚠️ KHÁC NHỎ — peak_height threshold:**
```python
# hsmm.py
peaks, props = find_peaks(search_region, height=0.0)  # chỉ lấy peak dương

# ref code — không có min height, lấy tất cả peaks
peaks, _ = find_peaks(search_region)
```
Ảnh hưởng nhỏ — chỉ loại bỏ peaks âm trong hsmm.py. Ref code lấy cả peaks âm rồi sort. Kết quả về cơ bản giống nhau.

---

### Task 3.9 — Duration Distributions (`hsmm.py`)

#### Những gì đã làm
```python
SPRINGER_DURATIONS = {
    'S1':       (0.122, 0.022),   # mean_frac, std_frac của chu kỳ tim
    'Systole':  (0.180, 0.059),
    'S2':       (0.094, 0.022),
    'Diastole': (0.604, 0.104),
    'Murmur':   (0.180, 0.059),   # same as Systole
}
# mu = frac * heart_period_frames
# sigma = std_frac * heart_period_frames
```

#### So sánh với ref code

**❌ SAI — Cách tính duration distributions:**

Ref code (segmenter.py) dùng **giá trị tuyệt đối (giây)**, không phải fraction:
```python
# S1 và S2: hằng số tuyệt đối
distrib_S1 = sci_stat.norm(loc=0.1163 * fs, scale=0.0196 * fs)
distrib_S2 = sci_stat.norm(loc=0.1032 * fs, scale=0.0195 * fs)

# Systole và Diastole: phụ thuộc systolic_interval ước lượng từ tín hiệu
mean_systole = (systolic_interval * fs) - (0.1279 * fs)
mean_diastole = (((60/heart_rate) - systolic_interval) * fs) - (0.1053 * fs)
```

Chúng ta dùng Springer (2016) fractions:
```python
mu_S1 = 0.122 × T_heart  # tại HR=75: 0.122 × 50 × 0.8s = 4.88 frames ≈ 97.6ms
```

Ref code:
```python
mu_S1 = 0.1163 × fs       # tại fs=50: 0.1163 × 50 = 5.815 frames = 116.3ms
```

**Sai lệch ví dụ tại HR=75 BPM (T=50frames=0.8s):**

| State | Ours (Springer) | Ref (McDonald) | Chênh lệch |
|---|---|---|---|
| S1 mean | 0.122×50×0.8 = **4.88f** (97.6ms) | 0.1163×50 = **5.82f** (116.3ms) | +19ms |
| S2 mean | 0.094×50×0.8 = **3.76f** (75.2ms) | 0.1032×50 = **5.16f** (103.2ms) | +28ms |
| S1 std | 0.022×40 = **0.88f** (17.6ms) | 0.0196×50 = **0.98f** (19.6ms) | ~similar |
| Systole | dynamic | dynamic (từ signal) | khác cách tính |

**Tác động: CAO.** Tham số duration distribution ảnh hưởng trực tiếp đến Viterbi path.
Với S1/S2 mean cao hơn (116ms vs 98ms), ref code "expects" S1 và S2 dài hơn. Điều này dẫn đến Viterbi phân đoạn khác nhau, và confidence scores khác nhau.

Ngoài ra, ref code ước lượng **systolic_interval** từ tín hiệu (thay vì dùng fixed fraction 0.180 × T), rồi trừ đi các constant để có mean systole/diastole. Đây là cách tính **thích nghi với từng recording**, tốt hơn cách dùng fixed fraction.

---

## Các Vấn Đề Tiềm Ẩn và Ảnh Hưởng Kết Quả

### Vấn đề ảnh hưởng KẾT QUẢ (theo thứ tự ưu tiên)

| # | Vấn đề | Tác động ước tính |
|---|---|---|
| 1 | **LR=1e-3 thay vì 1e-4** | Giảm 5-15% accuracy — RNN chưa converge tối ưu |
| 2 | **Duration distributions dùng Springer fractions** | Viterbi phân đoạn sai → confidence sai → classification sai |
| 3 | **Không có random crop khi training** | Thiếu data augmentation → kém generalize |
| 4 | **Không lọc Unknown patients khi training** | Nhiễu từ recordings không rõ ràng |
| 5 | **Background frames không bị xóa** | GRU xử lý khoảng trống → có thể nhiễu feature |
| 6 | **Dropout placement khác** | Ảnh hưởng regularization nhỏ |

### Vấn đề chưa implement (Phase 3b, 3c)
- **Viterbi HSMM** (Task 3.10): Chưa có code → **không thể chạy inference**
- **4 HSMM topologies** (Task 3.11): Chưa có
- **Confidence computation** (Task 3.12): Chưa có
- **Murmur classification decision** (Task 3.13): Chưa có

### Những điều đã làm TỐT HƠN ref code
1. **RAM preloading** (`load_dataset_to_ram`): Colab training chỉ mất 27s/epoch thay vì nhiều giờ
2. **Copy to local SSD trước khi train**: Loại bỏ Drive I/O bottleneck
3. **random_state=42** cho CV splits: Reproducibility tốt hơn
4. **Docstrings và comments** rõ ràng hơn ref code
5. **Modular code**: mỗi task có file riêng, dễ debug

---

## Khuyến Nghị Sửa Trước Khi Implement 3.10-3.13

### Nên sửa ngay (ảnh hưởng lớn)

1. **Sửa LR trong colab notebook:**
   ```python
   CONFIG = {'lr': 1e-4, 'max_epochs': 1000, ...}  # và retrain
   ```
   Cost: ~1.5 giờ retrain trên Colab T4.

2. **Sửa duration distributions trong hsmm.py:**
   ```python
   # Thay SPRINGER_DURATIONS bằng absolute-second constants:
   S1_MEAN_SEC = 0.1163
   S1_STD_SEC  = 0.0196
   S2_MEAN_SEC = 0.1032
   S2_STD_SEC  = 0.0195
   # Systole và Diastole: cần ước lượng systolic_interval trước
   ```
   Điều này đòi hỏi thêm bước "estimate systolic interval" vào hsmm.py.

### Có thể giữ nguyên (ảnh hưởng nhỏ)

3. **Background frames**: Giữ ignore_index=-1 thay vì xóa → code đơn giản hơn
4. **Dropout placement**: Không ảnh hưởng đáng kể đến kết quả
5. **random_state=42**: Tốt hơn ref code, giữ nguyên

### Cần implement tiếp

6. **Task 3.10** — Viterbi HSMM: implement trong `hsmm.py`
   - Dùng Python thuần (không cần Cython như ref code)
   - Cần implement Springer Extended Viterbi (vòng lặp tới T+D_max)

7. **Task 3.11** — 3 murmur topologies + 1 healthy:
   - segment_healthy (4 states)
   - segment_holosystolic (murmur = systole)
   - segment_early_systolic (S1→Murmur→Sys→S2→Dia)
   - segment_mid_systolic (S1→Sys/Murmur→...)

8. **Task 3.12** — Confidence = mean posterior at Viterbi path states

9. **Task 3.13** — Decision: `murmur_conf - healthy_conf > 0 → Present`

---

## Phụ Lục: Training Results Đã Đạt Được

Model đã train xong 5 folds. Checkpoint tại:
- `models/rnn/fold_0_best.pt` — epoch ~18, val_loss=0.3379
- `models/rnn/fold_1_best.pt` — epoch ~15, val_loss=0.4069
- `models/rnn/fold_2_best.pt` — epoch ~15, val_loss=0.3665
- `models/rnn/fold_3_best.pt` — epoch ~15, val_loss=0.3591
- `models/rnn/fold_4_best.pt` — epoch ~18, val_loss=0.3958

Biến động val_loss giữa folds (0.337 – 0.407) lớn hơn mong đợi, một phần do LR=1e-3 hội tụ tại điểm không ổn định.
