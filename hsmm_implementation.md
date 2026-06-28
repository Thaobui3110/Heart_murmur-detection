# Triển Khai HSMM trong Reference Code (McDonald et al., CinC 2022)

> Dựa trên phân tích trực tiếp từ:
> `reference_code/src/segmenter.py`, `reference_code/src/viterbi_hmm.pyx`,
> `reference_code/src/neural_networks.py`, `reference_code/team_code.py`

---

## 1. Kiến Trúc Tổng Thể (3 Giai Đoạn)

```
Tín hiệu PCG (.wav)
      │
      ▼
┌─────────────────────────────────┐
│  GIAI ĐOẠN 1: BiGRU             │  neural_networks.py
│  Input : (B, 41, T)             │
│  Output: (B, 5, T)  posteriors  │  P(S1|x), P(Sys|x), P(S2|x), P(Dia|x), P(Mur|x)
└─────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────┐
│  GIAI ĐOẠN 2: HSMM              │  segmenter.py + viterbi_hmm.pyx
│  Input : posteriors (T, 5)      │
│  Output: confidence_healthy     │
│          confidence_murmur      │
│          murmur_timing          │
└─────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────┐
│  GIAI ĐOẠN 3: Decision Tree     │  decision_tree.py
│  Input : conf_diff, biometrics  │
│  Output: Present/Unknown/Absent │
└─────────────────────────────────┘
```

---

## 2. HSMM Là Gì và Tại Sao Dùng

HMM thông thường giả định xác suất chuyển trạng thái không đổi ở mỗi bước thời gian, tức là thời gian ở mỗi trạng thái tuân theo phân phối hình học. Điều này không đúng với tim — S1 kéo dài khoảng 116ms, diastole khoảng 400ms, không ngẫu nhiên.

**HSMM (Hidden Semi-Markov Model)** khắc phục bằng cách mô hình hoá rõ ràng phân phối thời gian lưu trú tại mỗi trạng thái:

```
P(sequence) = ∏ P(observations | state s) × P(duration d | state s) × P(transition s→s')
```

---

## 3. Ước Lượng Nhịp Tim và Khoảng Systole

Trước khi chạy Viterbi, code ước lượng hai tham số quan trọng từ chính tín hiệu để tính phân phối thời gian động.

### 3.1 Ước Lượng Nhịp Tim (`get_heart_rate`, dòng 68–84)

```python
def get_heart_rate(posterior, fs, min_val=30, max_val=150, states=[1]):
    systolic_posterior = np.sum(posterior[:, states], axis=1)
    acf = np.correlate(systolic_posterior, systolic_posterior, mode="full")
    acf = acf[len(acf) // 2:]
    acf = acf / acf[0]
    min_index = round((60 / max_val) * fs)
    max_index = round((60 / min_val) * fs)
    rel_peak_loc = np.argmax(acf[min_index : max_index + 1])
    absolute_peak_loc = min_index + rel_peak_loc
    return 60 / (absolute_peak_loc / fs)
```

**Cơ chế:** Tính autocorrelation của posterior tổng hợp các trạng thái `[S1, Systole, S2, Murmur]` (chỉ số 0,1,2,4). Đỉnh autocorrelation trong khoảng [30, 180] bpm tương ứng với chu kỳ tim.

Tại inference, `states=[0,1,2,4]` — bao gồm cả murmur vì murmur xảy ra trong systole và đóng góp vào tín hiệu nhịp tim.

### 3.2 Ước Lượng Khoảng Systole (`get_systolic_interval`, dòng 22–47)

```python
mhs_posterior = np.sum(posterior[:, [0, 2]], axis=1)  # S1 + S2
mhs_acf = np.correlate(mhs_posterior, mhs_posterior, mode="full")
```

**Cơ chế:** Dùng autocorrelation của `S1 + S2` (hai tiếng tim chính) để tìm khoảng cách giữa chúng, từ đó suy ra khoảng systole. Giới hạn: min 150ms, max = nửa chu kỳ tim.

---

## 4. Phân Phối Thời Gian Lưu Trú (`get_duration_distributions`, dòng 50–65)

Sau khi có nhịp tim và khoảng systole, code tính 4 phân phối Gaussian cho 4 trạng thái:

```python
def get_duration_distributions(heart_rate, systolic_interval, fs):
    distrib_S1  = sci_stat.norm(loc=0.1163 * fs, scale=0.0196 * fs)
    distrib_S2  = sci_stat.norm(loc=0.1032 * fs, scale=0.0195 * fs)

    mean_sys = (systolic_interval * fs) - (0.1279 * fs)
    mean_sys = max(mean_sys, 0.07 * fs)
    std_sys  = 0.025 * fs

    mean_dia = (((60 / heart_rate) - systolic_interval) * fs) - (0.1053 * fs)
    mean_dia = max(mean_dia, 0.1 * fs)
    std_diastole = 0.050 * fs

    distrib_sys = sci_stat.norm(loc=mean_sys, scale=std_sys)
    distrib_dia = sci_stat.norm(loc=mean_dia, scale=std_diastole)

    return distrib_S1, distrib_sys, distrib_S2, distrib_dia
```

| Trạng thái | Mean | Std | Loại |
|---|---|---|---|
| S1 | 116.3 ms | 19.6 ms | Hằng số tuyệt đối |
| S2 | 103.2 ms | 19.5 ms | Hằng số tuyệt đối |
| Systole | `systolic_interval − 127.9 ms` | 25.0 ms | **Động** theo từng bệnh nhân |
| Diastole | `(T_heart − systolic_interval) − 105.3 ms` | 50.0 ms | **Động** theo từng bệnh nhân |

> **Lưu ý:** S1 và S2 dùng hằng số tuyệt đối (giây), KHÔNG phải phần trăm chu kỳ tim như bảng Springer (2016). Systole và Diastole được trừ đi S1/S2 tương ứng vì phân phối mô hình hoá phần "bên trong" của khoảng đó.

Khoảng thời gian tối đa cho Viterbi:
```python
max_duration = int((60 / heart_rate) * fs * max_duration_factor)
```
Mặc định `max_duration_factor = 1`, tức một chu kỳ tim đầy đủ.

---

## 5. Thuật Toán Viterbi HSMM (`viterbi_hmm.pyx`)

Được implement bằng **Cython** để tối ưu tốc độ. Đây là phiên bản mở rộng của Springer (2010).

### 5.1 Công Thức Đệ Quy

Với mỗi thời điểm `t`, trạng thái `s`, và thời gian lưu trú `d`:

```
delta[t, s] = max over (d, i):
    delta[t-d, i]                     # log prob tốt nhất đến t-d ở trạng thái i
  + log A[i, s]                       # log prob chuyển từ i sang s
  + Σ_{τ=t-d}^{t-1} log P(x_τ | s)   # log prob quan sát trong khoảng d bước
  + log p_s(d)                        # log prob thời gian lưu trú d ở trạng thái s
```

### 5.2 Vòng Lặp Chính (Cython)

```python
for t in range(1, T + max_duration):       # mở rộng qua T (Springer extension)
    for s in range(N):                     # N = số trạng thái
        for d in range(1, max_duration+1): # mọi thời gian lưu trú có thể
            start_t = max(0, min(t - d, T - 1))
            end_t   = min(t, T)

            # Tìm trạng thái trước tốt nhất
            delta_max = max over i: delta[start_t, i] + log(A[i, s])

            # Tích lũy log-likelihood quan sát
            product_obs = Σ log(posteriors[τ, s]) for τ in [start_t, end_t)

            # Kết hợp: prev + obs + duration
            delta_this = delta_max + product_obs + log(durations[d-1, s])

            if delta_this > delta[t, s]:
                delta[t, s] = delta_this
                psi[t, s] = i_max           # trạng thái trước tốt nhất
                psi_duration[t, s] = d      # thời gian lưu trú tốt nhất
```

**Độ phức tạp:** O(T × N × D) trong đó D = max_duration (một chu kỳ tim ≈ 2400 frames tại 4000Hz).

### 5.3 Springer Extended Viterbi

Viterbi thông thường kết thúc tại `t = T`. Springer mở rộng vòng lặp đến `t = T + max_duration` để tránh trường hợp bị cắt giữa trạng thái:

```python
# Tìm trạng thái tốt nhất trong vùng [T, T+max_duration]
for t in range(T, T + max_duration):
    for s in range(N):
        if delta[t, s] > max_delta_after:
            current_state = s
            end_time = t
```

### 5.4 Backtracking

```python
while t > 0:
    d = psi_duration[t, current_state]     # thời gian lưu trú được chọn
    for i in range(max(0, t-d), t):
        states[i] = current_state          # gán trạng thái cho toàn bộ khoảng
    t = max(0, t - d)
    current_state = psi[t, current_state]  # lùi về trạng thái trước
```

---

## 6. Ma Trận Chuyển Trạng Thái

### Mô Hình Healthy (4 trạng thái)

```python
transition_matrix = np.array([
    [0, 1, 0, 0],   # S1 → Systole
    [0, 0, 1, 0],   # Systole → S2
    [0, 0, 0, 1],   # S2 → Diastole
    [1, 0, 0, 0],   # Diastole → S1
])
```

Chu kỳ cố định S1→Systole→S2→Diastole→S1. Xác suất murmur (channel 4) bị gộp vào kênh systole:
```python
new_posteriors[:, 1] = posteriors[:, 1]  # healthy: dùng systole posterior
```

### Mô Hình Holosystolic Murmur (4 trạng thái, thay systole bằng murmur)

```python
new_posteriors[:, 1] = posteriors[:, 4]  # dùng murmur posterior thay systole
# Ma trận giống healthy: S1→Murmur→S2→Diastole→S1
```

Murmur chiếm toàn bộ systole.

### Mô Hình Early-Systolic Murmur (5 trạng thái)

```python
transition_matrix = np.array([
    [0, 0, 0, 0, 1],   # S1 → Murmur (đầu tiên)
    [0, 0, 1, 0, 0],   # Systole → S2
    [0, 0, 0, 1, 0],   # S2 → Diastole
    [1, 0, 0, 0, 0],   # Diastole → S1
    [0, 1, 0, 0, 0],   # Murmur → Systole (phần còn lại)
])
```

Duration murmur và systole mỗi loại = nửa khoảng systole bình thường:
```python
distrib_sys_half = sci_stat.norm(orig_sys_distrib.mean() / 2,
                                  orig_sys_distrib.std()  / 2)
```

### Mô Hình Mid-Systolic Murmur (5 trạng thái)

```python
transition_matrix = np.array([
    [0, 1, 0, 0, 1],   # S1 → Systole hoặc Murmur
    [0, 0, 1, 0, 1],   # Systole → S2 hoặc Murmur
    [0, 0, 0, 1, 0],   # S2 → Diastole
    [1, 0, 0, 0, 0],   # Diastole → S1
    [0, 1, 0, 0, 0],   # Murmur → Systole
])
# Systole nhỏ hơn 4 lần, murmur bằng nửa systole bình thường
```

---

## 7. `double_duration_viterbi` — Điểm Vào Chính

```python
def double_duration_viterbi(posteriors, fs, ...):
    # Bước 1: Ước lượng nhịp tim từ tín hiệu
    heart_rate = get_heart_rate(posteriors, fs, min_hr, max_hr, states=[0,1,2,4])

    # Bước 2: Ước lượng khoảng systole từ ACF
    systolic_interval = get_systolic_interval(posteriors, fs, heart_rate, min_systole)

    # Bước 3: Tạo phân phối thời gian động
    duration_distributions = get_duration_distributions(heart_rate, systolic_interval, fs)

    # Bước 4: Chạy HSMM Viterbi cho mô hình healthy
    healthy_seg, healthy_conf = segment_healthy_signal(...)

    # Bước 5: Chạy HSMM Viterbi cho 3 mô hình murmur — chọn cái có confidence cao nhất
    for model in ["Holosystolic", "Early-systolic", "Mid-systolic"]:
        seg, conf = murmur_functions[model](posteriors, duration_distributions, ...)
        if conf > best_murmur_confidence:
            best = (seg, conf, model)

    return healthy_seg, best_murmur_states, healthy_conf, best_murmur_confidence, best_murmur_model
```

Gọi hàm này: 4 lần Viterbi trên mỗi recording (1 healthy + 3 murmur models).

---

## 8. Tính Confidence và Ra Quyết Định

### Confidence Score

```python
def compute_segmentation_confidence(posteriors, segmentation):
    return posteriors[np.arange(len(posteriors)), segmentation].mean()
```

Confidence = trung bình posterior tại mỗi frame theo trạng thái được gán. Giá trị nằm trong [0, 1].

### Ra Quyết Định (`decide_murmur_outcome`, team_code.py)

```python
def decide_murmur_outcome(features):
    conf_differences = [v for k,v in features.items() if k.startswith("conf_difference_")]
    signal_quals     = [v for k,v in features.items() if k.startswith("signal_qual_")]

    if np.nanmax(conf_differences) > 0:
        return "Present"       # ít nhất 1 vị trí: murmur_conf > healthy_conf
    elif np.nanmin(signal_quals) < 0.65:
        return "Unknown"       # ít nhất 1 vị trí có chất lượng tín hiệu kém
    else:
        return "Absent"
```

Với:
- `conf_difference = murmur_conf − healthy_conf` (tính trên mỗi vị trí nghe tim: AV, MV, PV, TV)
- `signal_qual = max(murmur_conf, healthy_conf)` — đo chất lượng tín hiệu tổng thể

---

## 9. Nhãn Murmur Trong Training (Label Generation)

Nhãn được tạo trong `RecordingDataset.__getitem__` (neural_networks.py) theo logic:

```
TSV state 0 (background) → label -1  (ignored in CrossEntropyLoss)
TSV state 1 (S1)         → label  0
TSV state 2 (Systole)    → label  1  (hoặc 4 nếu có murmur)
TSV state 3 (S2)         → label  2
TSV state 4 (Diastole)   → label  3
Phần systole có murmur   → label  4
```

Phần murmur trong systole được tính theo thời điểm (timing):
```python
if row.state == 2 and label == "Present":
    if timing == "Holosystolic":   portion = [0,    1  ]  # 100% systole
    if timing == "Early-systolic": portion = [0,    0.5]  # 50% đầu
    if timing == "Mid-systolic":   portion = [0.25, 0.75] # 50% giữa
    if timing == "Late-systolic":  portion = [0.5,  1  ]  # 50% cuối
```

---

## 10. Luồng Dữ Liệu Đầy Đủ Khi Inference

```
recording.wav
    │
    ▼ calculate_features()
spectrogram (41, T)     ← log-spectrogram, z-normalised
    │
    ▼ 5 fold BiGRU models
posteriors × 5 folds    ← softmax, shape (T, 5)
    │
    ▼ mean across folds
posteriors (T, 5)       ← ensemble predictions
    │
    ├─▶ get_heart_rate()           → heart_rate (bpm)
    ├─▶ get_systolic_interval()    → systolic_interval (s)
    ├─▶ get_duration_distributions() → 4 Normal distributions
    │
    ├─▶ segment_healthy_signal()   → healthy_conf
    ├─▶ segment_holosystolic_murmur() ┐
    ├─▶ segment_early_systolic_murmur() ┤→ best_murmur_conf
    └─▶ segment_mid_systolic_murmur() ┘
    │
    ▼
conf_difference = best_murmur_conf - healthy_conf
    │
    ▼ decide_murmur_outcome()
"Present" / "Unknown" / "Absent"
```

---

## 11. Tóm Tắt Các Tham Số Quan Trọng

| Tham số | Giá trị | Nguồn |
|---|---|---|
| `fs` features | 50 Hz | 1/hop_length (20ms) |
| `min_hr` | 30 bpm | `double_duration_viterbi` |
| `max_hr` | 180 bpm | `double_duration_viterbi` |
| `min_systole` | 150 ms | `double_duration_viterbi` |
| S1 duration mean | 116.3 ms | `get_duration_distributions` |
| S2 duration mean | 103.2 ms | `get_duration_distributions` |
| Signal qual threshold | 0.65 | `decide_murmur_outcome` |
| Murmur models tested | 3 | Holosystolic, Early, Mid |
| max_duration | 1 × T_heart | HSMM Viterbi window |
