# CNN Lite từng lớp: float và int8

Tài liệu học tập về mô hình **Lite** đóng băng: `results/2026-10-15_581b4c74_final_A5_svdb_n25_lite_e3/`
(`model.pt` cho float, `lite_int8.npz` cho int8). Mọi số liệu lấy từ checkpoint và code
(`python/models/cnn1d.py`, `python/quant/int8.py`); các ví dụ ở §6 sinh bằng `scripts/trace_beat.py` trên golden vectors.
Bối cảnh toàn hệ thống: [`system_overview.md`](system_overview.md).

## 1. Kiến trúc và kích thước

```
z (1×360) ─ block1: Conv(1→16, k7) BN ReLU MaxPool2 ─ (16×180)
          ─ block2: Conv(16→32, k5) BN ReLU MaxPool2 ─ (32×90)
          ─ block3: Conv(32→64, k3) BN ReLU MaxPool2 ─ (64×45)
          ─ block4: Conv(64→64, k3) BN ReLU          ─ (64×45)
          ─ GAP ─ (64) ─ concat RR chuẩn hoá (4) ─ (68) ─ FC1 68→32, ReLU ─ FC2 32→4 ─ logit
```

Conv dùng padding "same" (k//2 số 0 mỗi bên, chiều dài không đổi), không có bias (BN đóng vai bias). Dropout 0.2
sau mỗi MaxPool và 0.5 trước FC2 **chỉ có tác dụng khi huấn luyện**; khi suy luận là phép đồng nhất nên không xuất
hiện trong bản int8.

| lớp | vào → ra (shape) | tham số float | MAC / nhịp | int8 triển khai |
|---|---|---|---|---|
| conv0 + BN0 | (1, 360) → (16, 360) | 112 + 32 (γ, β) | 40 320 | w 16×1×7 int8, bias 16 int32 |
| pool0 | (16, 360) → (16, 180) | 0 | 0 (so sánh) | max của cặp |
| conv1 + BN1 | (16, 180) → (32, 180) | 2 560 + 64 | 460 800 | w 32×16×5, bias 32 |
| pool1 | → (32, 90) | 0 | 0 | |
| conv2 + BN2 | (32, 90) → (64, 90) | 6 144 + 128 | 552 960 | w 64×32×3, bias 64 |
| pool2 | → (64, 45) | 0 | 0 | |
| conv3 + BN3 | (64, 45) → (64, 45) | 12 288 + 128 | 552 960 | w 64×64×3, bias 64 |
| GAP | (64, 45) → (64) | 0 | 2 880 phép cộng | tổng int32 |
| FC1 | (64 + 4) → 32 | 2 176 + 32 | 2 176 | w 32×64 + 32×4 int8, bias 32 int32 |
| FC2 | 32 → 4 | 128 + 4 | 128 | w 4×32 int8, bias 4 int32 |
| **tổng** | | **23 796** tham số học | **1 609 344** | **23 408 B** int8 + **212** bias int32 |

- MAC của conv = `L_out · C_out · C_in · k`, tính cả các tap rơi vào phần đệm 0. Nếu C bỏ qua các tap đó thì còn 1 593 792 MAC.
- Hai buffer `rr_mean`, `rr_std` (4 + 4 số) không phải tham số học, và được gộp vào FC1 (§4.3).
- 4 conv chiếm 99.9 % MAC. Block 2–4 mỗi block khoảng 0.5 M MAC; block 1 rẻ vì chỉ có 1 kênh vào.

Trường nhìn (receptive field) của một vị trí ở block 4 trên tín hiệu vào: k7 (7) → pool (8) → k5 (+4·2 = 16) →
pool (18) → k3 (+2·4 = 26) → pool (30) → k3 (+2·8 = 46) mẫu ≈ 128 ms. Như vậy mỗi vị trí đầu ra "nhìn" khoảng
một phức bộ QRS. GAP sau đó gom 45 vị trí phủ toàn cửa sổ 1 s.

## 2. Từng lớp: phép toán float

**Conv1d (cross-correlation, "same")**, đầu ra kênh o tại vị trí t:

  `y[o, t] = Σ_c Σ_{j=0}^{k−1} W[o, c, j] · x[c, t + j − k//2]`  (x ngoài [0, L) coi là 0)

**BatchNorm (chế độ eval)**, dùng thống kê chạy (running) học được khi huấn luyện:

  `BN(y)[o] = γ_o · (y[o] − μ_o) / sqrt(σ²_o + ε) + β_o`,  ε = 1e−5

**ReLU**: `max(0, ·)`. **MaxPool(2)**: `p[o, t] = max(h[o, 2t], h[o, 2t+1])` (chiều dài chẵn: 360 → 180 → 90 → 45).

**GAP**: `g[o] = (1/45) Σ_t h4[o, t]`.

**FC1**: `u = ReLU(W1 · [g ; r̂] + b1)` với `r̂ = (rr − μ_rr) / σ_rr` (z-score RR). **FC2**: `logit = W2 · u + b2`.

Huấn luyện: focal loss (γ = 2) + trọng số lớp sqrt-inverse, AdamW. Đầu ra là 4 logit; softmax chỉ nằm trong hàm loss.

## 3. Ba loại chuẩn hoá

| | z-score đầu vào | BatchNorm | z-score RR |
|---|---|---|---|
| chuẩn hoá cái gì | từng cửa sổ nhịp (360 mẫu) | từng kênh feature map | từng đặc trưng RR |
| μ, σ lấy từ đâu | **chính nhịp đó**, tính lúc chạy | thống kê chạy trên tập train, cố định khi suy luận | inner-train (μ, σ lưu trong checkpoint) |
| mục đích | bỏ biên độ và đường nền khác nhau giữa bệnh nhân/điện cực; mạng chỉ thấy **hình dạng** | ổn định huấn luyện; khi suy luận chỉ là phép affine theo kênh | đưa 4 RR (đơn vị khác nhau: giây, tỉ số) về cùng thang trước khi trộn với GAP |
| giá trị thật | kẹp ±8 σ; ε = 1e−6 | ε = 1e−5; γ ∈ [0.857, 1.211], σ² ∈ [0.0083, 3.398] (4 block) | μ = (0.737, 0.774, 0.758, 0.971), σ = (0.215, 0.212, 0.176, 0.160) (rr_prev s, rr_next s, rr_avg s, ratio) |
| trên board | **tính thật** bằng số nguyên (`zscore_int`: tổng, isqrt, chia làm tròn) | **gộp vào conv** (không tốn phép tính) | **gộp vào FC1** (không tốn phép tính) |

Vì sao hai loại sau gộp được: lúc suy luận cả hai đều là hàm **affine cố định** (nhân hằng số rồi cộng hằng số) nằm
liền trước/sau một phép tuyến tính, nên có thể nhân thẳng vào trọng số và bias. z-score đầu vào thì phụ thuộc vào chính
dữ liệu (μ, σ của từng nhịp), nên không gộp được và phải tính trên board.

## 4. Bản int8 khác bản float ở đâu

Quy ước: số nguyên `q` với số mũ `e` biểu diễn giá trị `q · 2^−e`. Mọi scale là **luỹ thừa của 2**, nên đổi scale chỉ
là dịch bit. Làm tròn khi dịch phải: `rshift_round(acc, s) = (acc + 2^(s−1)) >> s` (dịch số học). Biểu thức này phải
giữ y nguyên trong C.

### 4.1 Gộp BN vào conv (`fold_bn`)

`W'[o] = W[o] · γ_o / sqrt(σ²_o + ε)`,  `b'[o] = β_o − μ_o · γ_o / sqrt(σ²_o + ε)`

Conv không bias + BN trở thành **một conv có bias**. Kết quả là toán học tương đương (float64), chưa lượng tử.

### 4.2 Lượng tử conv (`quantize_model`, `conv_int`)

| ký hiệu | ý nghĩa | giá trị thật (Lite cuối) |
|---|---|---|
| `a_in`, `a_out` | số mũ kích hoạt vào/ra (theo tensor), chọn sao cho `max|x| · 2^a ≤ 127` trên tập hiệu chỉnh | đầu vào 4; conv0 ra 2; conv1–3 ra 3 |
| `k_c` | số mũ trọng số **theo từng kênh ra** c | conv0 5..8, conv1 8..10, conv2 7..9, conv3 8..10 |
| `w_q` | `clamp(round(W' · 2^k_c), ±127)` int8 | |
| `b_q` | `round(b' · 2^(k_c + a_in))` int32 (cùng scale với accumulator) | conv0 −1004..129, …, conv3 −14157..11152 |

Phép tính mỗi kênh:

  `acc = Σ w_q · x_q + b_q`  (int32, scale 2^−(k_c + a_in))
  `y_q = clamp(rshift_round(acc, k_c + a_in − a_out), 0, 127)`

Lượng dịch `s = k_c + a_in − a_out` (thật): conv0 7..10, conv1 7..9, conv2 7..9, conv3 8..10. **ReLU và kẹp gộp
làm một**: `clamp(·, 0, 127)`. Sau ReLU giá trị không âm, nên kích hoạt thực chất là 7 bit không dấu.

**MaxPool trên int8 là chính xác.** Phép requantize là hàm đơn điệu không giảm, nên `max` trước hay sau requantize cho
cùng kết quả. Phép so sánh số nguyên không gây sai số.

### 4.3 GAP và FC1: gộp 1/45 và z-score RR

- GAP trên board là **tổng** `S[o] = Σ_t y_q[o, t]` (int32, tối đa 45·127 = 5 715). Hệ số 1/45 được nhân vào trọng số
  FC1 (`w1_gap = W1[:, :64] / 45`), nên không cần phép chia.
- z-score RR gộp vào FC1: `W1r' = W1r / σ_rr`, `b1' = b1 − W1r · (μ_rr / σ_rr)`. Board đưa thẳng RR (đã lượng tử)
  vào FC1.
- RR int8 với số mũ riêng từng đặc trưng: `rr_exp = (6, 6, 6, 5)`.
- FC1 có **hai khối tích luỹ với số mũ riêng**, vì trọng số GAP sau khi chia 45 rất nhỏ. Nếu dùng chung một số mũ,
  chúng sẽ làm tròn về 0. Cụ thể:
  - khối GAP: `acc_g = Σ wg_q · S`, số mũ Eg = a_gap + eg_o ∈ {17, 18};
  - khối RR: `acc_r = Σ wr_q · rr_q`, số mũ Er ∈ {10..14};
  - căn về E = min(Eg, Er) ∈ {10..14}: `acc = rshift_round(acc_g, Eg − E) + rshift_round(acc_r, Er − E) + b_q`;
  - `h_q = clamp(rshift_round(acc, E − a5), 0, 127)` với a5 = 4 (dịch 6..10). ReLU cũng là kẹp.

### 4.4 FC2 và logit

Trọng số FC2 có một số mũ chung cho cả tensor (k2 = 7), đầu vào a5 = 4:

  `logit = Σ w_q · h_q + b_q`  (int32, scale chung 2^−11; b_q = (568, 194, −166, 49))

Không requantize về int8. Logit giữ nguyên int32 và **cùng một scale** cho cả 4 lớp, nên so sánh trực tiếp được.

### 4.5 Đầu vào

`z_q` = z-score số nguyên × 16, kẹp ±127 (§3.6 của `system_overview.md`). Bản float tương đương là
`clamp(floor(z · 16 + 0.5), ±127)`.

### 4.6 Sai khác thực tế

- Cổng lượng tử (đầu vào float, mô hình int8): macro-F1 OOF thay đổi −0.0007.
- Cổng pipeline số nguyên (iii) (cả tiền xử lý lẫn CNN đều số nguyên): float 0.5089 → 0.5096, tức −0.0007.
- Trên DS1, (iii) đổi argmax so với float ở 0.8–1.4 % số nhịp (397–704 nhịp tuỳ seed). Hầu hết là nhịp N, nhưng có
  sự chuyển qua lại giữa các lớp (`pipeline_int.log`).
- Ngân sách int32: `_check_int32` raise lỗi nếu bất kỳ accumulator nào vượt 2^31 − 1. Chưa bản ghi nào gặp lỗi này.

## 5. Vì sao board dùng argmax, không dùng softmax

`softmax(z)_i = e^{z_i} / Σ_j e^{z_j}` là hàm **đơn điệu tăng** theo từng `z_i` khi các `z_j` khác giữ nguyên, và
mẫu số giống nhau cho mọi i. Vì vậy `argmax softmax(z) = argmax z`: nhãn không đổi.

Lợi ích trên board:
- không cần `exp` hay phép chia (cần float hoặc bảng tra);
- logit đã ở **cùng scale 2^−11**, nên so sánh 4 số int32 là đủ.

Xác suất chỉ cần cho người xem. Host (`live_monitor`) tự tính `softmax(logit / 2^11)` từ logit int32 nhận qua UART.

Lưu ý cho C: khi hoà (hai logit bằng nhau) thì NumPy `argmax` chọn **chỉ số nhỏ nhất**, nên C phải giữ đúng quy ước này
để trùng golden.

## 6. Bốn nhịp ví dụ (golden vectors, một nhịp đầu tiên mỗi lớp)

Lệnh: `python scripts/trace_beat.py --first-of N|S|V|F --out docs/img`. Mỗi lần chạy, script tính lại toàn bộ CNN số
nguyên từ `z_q`/`rr_q` của golden và **khẳng định trùng golden từng bit** (cả 4 nhịp: YES).

Đọc hình:
- **Hàng trên**: cửa sổ đã lọc (mV), với z_q chồng lên theo trục phải; bên phải là 4 logit.
- **Hai hàng dưới**: heatmap đầu ra 4 block, kênh × vị trí, trước pooling. Thang màu theo max của chính block đó.
- **Cột hẹp**: 64 tổng GAP.
- **Khung chữ**: RR, rr_q, FC1.

| lớp thật | đoạn / nhịp | RR (rr_prev, rr_next mẫu; ratio) | logit / 2^11 (N, S, V, F) | dự đoán |
|---|---|---|---|---|
| N | rec101_93600 / 0 | 340, 359; 1.000 (n8 = 1: nhịp đầu đoạn) | 1.707, 0.145, −2.282, −1.964 | **N** ✓ |
| S | rec101_93600 / 121 | 194, 426; 0.570 | −1.141, 2.492, 0.695, −5.069 | **S** ✓ |
| V | rec118_237600 / 100 | 160, 386; 0.607 | −1.280, 2.122, 1.786, −4.892 | **S** ✗ |
| F | rec201_522000 / 11 | 436, 439; 1.334 | 1.421, −4.342, 0.208, −0.596 | **N** ✗ |

Hai nhịp sai là số thật, không chọn lọc. Đây đúng là hai kiểu lỗi mà confusion matrix cũng cho thấy:
- **V → S**: nhịp V này đến rất sớm (ratio 0.61), giống "nhịp sớm" của S. Logit S và V gần nhau (2.12 so với 1.79).
- **F → N**: F (nhịp hoà) có RR bình thường và hình dạng lai giữa N và V. Mô hình Lite có F-recall bằng 0 trên DS2.

### N — rec101_93600, nhịp 0
![N](img/trace_rec101_93600_beat0.png)

### S — rec101_93600, nhịp 121
![S](img/trace_rec101_93600_beat121.png)

### V — rec118_237600, nhịp 100 (dự đoán S)
![V](img/trace_rec118_237600_beat100.png)

Trong cửa sổ này nhịp trước chỉ cách 160 mẫu (444 ms), nên phức bộ của nó nằm trọn trong nửa trái cửa sổ (−450 ms).
Các kênh block 1 phản ứng mạnh ở cả hai phức bộ.

### F — rec201_522000, nhịp 11 (dự đoán N)
![F](img/trace_rec201_522000_beat11.png)

### Ví dụ đầu ra văn bản (nhịp S)

```
golden segment rec101_93600, beat 121 (detection 122), R = sample 41698 of the segment (record 101, sample 135298)
label S, predicted S; recomputed int_forward_trace == golden: YES
stage    shape         scale              min         max  first values  | meaning
window   (360,)        2^-12          -227176     1229665  -98114 -127086 -150714 -167630 -177139 -180667  | filt[R+2-180 : R+2+180], ADC counts x 2^12 (int32)
z_q      (360,)        2^-4               -16         106  -5 -8 -10 -11 -12 -12  | integer z-score, clip +-127 (= +-8 sd)
conv0    (16, 360)     2^-2                 0          26  0 0 0 0 0 0  | block 1: conv k=7 + folded BN, ReLU, clamp 0..127
pool0    (16, 180)     2^-2                 0          26  0 0 0 0 0 0  | MaxPool(2)
conv1    (32, 180)     2^-3                 0          43  0 0 2 4 4 2  | block 2: conv k=5 + folded BN, ReLU, clamp 0..127
pool1    (32, 90)      2^-3                 0          43  0 4 4 2 2 3  | MaxPool(2)
conv2    (64, 90)      2^-3                 0          48  0 0 0 0 0 0  | block 3: conv k=3 + folded BN, ReLU, clamp 0..127
pool2    (64, 45)      2^-3                 0          48  0 0 0 0 0 0  | MaxPool(2)
conv3    (64, 45)      2^-3                 0          46  0 0 0 0 0 0  | block 4: conv k=3 + folded BN, ReLU, clamp 0..127
gap      (64,)         2^-3                15         280  98 166 81 70 63 215  | sum over 45 positions (1/45 folded in FC1)
rr_parts (4,)          2^-0                 8        2723  194 426 2723 8  | rr_prev, rr_next, sum8, n8 (samples, RR clipped 72..1080)
rr_q     (4,)          -                   18          76  34 76 61 18  | int8 RR features, exponents [6, 6, 6, 5] (rr_prev, rr_next, rr_avg, ratio)
fc1      (32,)         2^-4                 0          34  0 28 0 0 23 33  | FC1 + ReLU, clamp 0..127
logits   (4,)          2^-11           -10382        5103  -2337 5103 1423 -10382  | FC2, int32, common scale
logits / 2^11 = [-1.141, 2.492, 0.695, -5.069] -> softmax (display only) [0.022, 0.838, 0.139, 0.0] -> argmax S
```

Kiểm tra tay vài con số:
- `rr_q[0] = round(194 · 64 / 360) = round(34.49) = 34`, tức rr_prev = 34/64 = 0.53 s.
- `ratio = 194 · 8 / 2723 = 0.570`, nên `rr_q[3] = round(0.570 · 32) = 18`.
- Kích hoạt lớn nhất chỉ 26–54 trên 127 vì số mũ được hiệu chỉnh theo **max trên cả tập hiệu chỉnh**; một nhịp bình
  thường chỉ dùng một phần dải.
