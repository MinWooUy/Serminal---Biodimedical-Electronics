# Giao thức PC ↔ FPGA theo ô (tile) — hợp đồng hiện hành

> **Nguồn gốc là RTL và host của nhóm FPGA** trong submodule `Serminal---Biodimedical-Electronics`, nhánh `dev`, commit **`4ad6c8b`**
> (13/11/2026). File này mô tả đúng hành vi của RTL đó; chỗ nào README của nhóm FPGA khác RTL thì **theo RTL** và ghi ở §10.
> Từ 14/11/2026 đây là hợp đồng chính thức (quyết định của người dùng, xem `NHATKY.md`). Giao thức v2 (`docs/proto_v2.md`,
> BEAT_IN / RESULT) chỉ còn là tài liệu cũ: nhóm FPGA không cài đặt nó. Không bao giờ trộn v1, v2 và giao thức ô trên một đường truyền.
>
> Tệp RTL gốc (đường dẫn trong submodule): `rtl/cnn1d/comm/protocol/packet_rx.v`, `packet_tx.v`, `host_command_controller.v`,
> `rtl/cnn1d/top/ecg_accel_top.v`, `rtl/cnn1d/post/conv1d_post_process.v`, `rtl/cnn1d/de10_lite_top.v`; host của họ: `host/ecg_host.py`.

## 1. Vai trò

- **FPGA** là một bộ tính **một ô (tile)** của phép Conv1D / tích vô hướng: tối đa `ROWS = 7` vị trí (hàng) × `COLS = 7` kênh ra (cột).
  Mỗi ô: tích luỹ int32 của `K` cặp int8, cộng bias (nếu bật nạp trước), rồi hậu xử lý: dịch phải có làm tròn → ReLU (tuỳ chọn) → kẹp ±127.
- **PC** làm mọi thứ còn lại: tiền xử lý (lọc, Pan-Tompkins, cửa sổ, z-score, RR), chia từng lớp CNN thành ô, nạp dữ liệu / trọng số / bias,
  ghép kết quả, và các phép phần cứng chưa có: MaxPool, GAP, ghép ba lần dịch của FC1, argmax (§9).

## 2. Lớp vật lý

UART 115 200 8N1, không bắt tay phần cứng, mức logic 3.3 V. Bo DE10-Lite: `de10_lite_top` (xung `MAX10_CLK1_50` = 50 MHz, reset = `~KEY0`),
tham số `POST_ENABLE = 1`, `FEEDER_DEPTH = 1024`; mặc định của `ecg_accel_top`: `MEM_DEPTH = 1024`, `MAX_PAYLOAD = 256`,
`ARRAY_ROWS = ARRAY_COLS = 7`, `RX_TIMEOUT_CLKS = CLK_FREQ/10` (100 ms).

## 3. Khung

```
A5 5A | CMD u8 | SEQ u8 | LEN u16 LE | PAYLOAD (LEN byte) | CRC u16 LE
```

- CRC-16/CCITT-FALSE (đa thức 0x1021, khởi tạo 0xFFFF, không đảo bit, XorOut 0) trên **CMD, SEQ, LEN, PAYLOAD**; "123456789" → 0x29B1.
  Khác v2 ở **byte SEQ** sau CMD — hai giao thức không tương thích byte với nhau.
- Số nhiều byte đều little-endian. `LEN ≤ MAX_PAYLOAD = 256` theo cả hai chiều.
- Ví dụ (đã tính bằng `python/link/proto_v2.crc16`, cùng thuật toán CRC):
  - `GET_STATUS`, SEQ 0: `A5 5A 05 00 00 00 85 38`
  - `RUN_TILE`, SEQ 1: `A5 5A 04 01 00 00 01 79`
  - `ACK` cho GET_STATUS, SEQ 0, cờ 0x01: `A5 5A 80 00 03 00 05 00 01 34 4F`

## 4. Lệnh (PC → FPGA)

| CMD | Tên | PAYLOAD | LEN hợp lệ | Kiểm tra của RTL → NACK |
|---|---|---|---|---|
| `01` | WRITE_IFMAP | `base u16`, rồi 1–254 byte int8 | 3…256 | `base + n > 1024` → 2 |
| `02` | WRITE_WEIGHT | `base u16`, rồi 1–254 byte int8 | 3…256 | `base + n > 1024` → 2 |
| `03` | CONFIG_TILE | `K u16, step u16, rows u8, cols u8, x_base u16, w_base u16` [+ `preload u8`] | 10 hoặc 11 | xem dưới → 2 |
| `04` | RUN_TILE | rỗng | 0 | LEN ≠ 0 → 2; chưa sẵn sàng → 3 |
| `05` | GET_STATUS | rỗng | 0 | LEN ≠ 0 → 2 |
| `06` | WRITE_BIAS | `base u16`, rồi 1–7 int32 LE | 6, 10, …, 30 (LEN ≡ 2 mod 4) | `base + số bias > 7` → 2 |
| `07` | CONFIG_POST | `shift i8 × 7` (một mỗi cột), rồi `relu u8` ∈ {0, 1} | **đúng 8** | LEN ≠ 8 hoặc relu > 1 → 2 |
| khác | — | — | — | → 4 |

**CONFIG_TILE** bị từ chối (mã 2) nếu: `preload > 1`; `K = 0` hoặc `K > FEEDER_DEPTH (1024)`; `step = 0`; `rows ∉ 1…7`; `cols ∉ 1…7`;
`x_base + (rows − 1)·step + K > 1024`; `w_base + cols·K > 1024`. Chấp nhận thì đặt `configured = 1`; `preload = 1` bật cộng bias.
CONFIG_TILE dài 10 byte = **tắt** nạp bias.

**RUN_TILE** trả NACK 3 nếu: chưa CONFIG_TILE; ô trước còn chạy; đang bật preload mà bias của một cột đang dùng (`< cols`) chưa từng ghi;
`POST_ENABLE = 1` mà chưa CONFIG_POST; hoặc một địa chỉ cần đọc chưa có dữ liệu hợp lệ (§6). RUN không có ACK sơ bộ: trả thẳng RESULT hoặc NACK.

## 5. Trả lời (FPGA → PC) — SEQ luôn chép lại SEQ của lệnh

| CMD | Tên | PAYLOAD |
|---|---|---|
| `80` | ACK | `[cmd gốc, 00]`; với GET_STATUS thêm 1 byte cờ: bit0 đã cấu hình, bit1 bank ifmap tạm "bẩn", bit2 bank trọng số tạm "bẩn", bit3 bật preload bias, bit4 đã CONFIG_POST |
| `81` | NACK | `[cmd gốc, mã]`: **1** CRC sai hoặc LEN > 256 · **2** sai độ dài / cấu hình / địa chỉ · **3** chưa chạy được / thiếu dữ liệu · **4** CMD lạ |
| `82` | RESULT | `raw int32 × 49` (196 B) theo hàng-trước `(hàng·7 + cột)·4`, rồi (khi `POST_ENABLE = 1`) `post int8 × 49` (49 B): **245 B** trên bo DE10-Lite |

Ô không dùng (hàng ≥ rows hoặc cột ≥ cols): `post = 0`. Với NACK mã 1, `cmd gốc` và SEQ lấy từ header nhận được (có thể đã hỏng).

## 6. Bộ nhớ và vòng đời bank

- Hai bộ nhớ dữ liệu, mỗi bộ **1 024 byte**, kiểu ping-pong: bank **tạm** (WRITE ghi vào) và bank **hoạt động** (ô đọc từ đó).
  Mỗi byte có một bit "hợp lệ" riêng ở mỗi bank.
- WRITE ghi vào bank tạm, đánh dấu từng byte đã ghi và đặt cờ "bẩn". Có thể ghi nhiều đoạn, địa chỉ bất kỳ.
- RUN kiểm từng địa chỉ cần đọc (ifmap: `x_base + r·step + k`, r < rows, k < K; trọng số: `w_base + c·K + k`, c < cols):
  bank nào bẩn thì kiểm ở bank tạm, không bẩn thì kiểm ở bank hoạt động. Thiếu một byte → NACK 3, **không đổi bank nào**.
- Đạt thì bank bẩn **đổi vai** (tạm ↔ hoạt động), bit hợp lệ của bank tạm mới bị xoá, rồi mới chạy ô.
  Hệ quả: sau khi ghi lại một phần, bank tạm phải chứa **đủ mọi byte** ô kế tiếp cần, không "vá" lên bank hoạt động được.
  RUN lại mà không ghi gì thì dùng lại cả hai bank hoạt động; chỉ ghi ifmap thì trọng số giữ nguyên (và ngược lại).
- **Bias**: 7 thanh ghi int32, một mỗi cột, **không** theo ping-pong; ghi là có hiệu lực ngay, giữ tới khi reset.
- Reset xoá cấu hình, mọi bit hợp lệ và cờ bias; phải nạp lại từ đầu.

## 7. Phép tính của một ô

Với hàng `r` (vị trí) và cột `c` (kênh ra):

```
raw[r][c]  = (preload ? bias[c] : 0) + Σ_{k<K} X[x_base + r·step + k] · W[w_base + c·K + k]      (int32)
post[r][c] = clamp( relu( round_shift(raw[r][c], shift[c]) ), −127, 127 )
round_shift(a, s) = (a + 2^(s−1)) >> s   khi 1 ≤ s ≤ 31   (RTL: ((a >> (s−1)) + 1) >> 1, tương đương)
                  = 0                     khi s ≥ 32
                  = a                     khi s = 0
                  = a << −s               khi −7 ≤ s ≤ −1
                  = sign(a) · 127         khi s ≤ −8
```

`round_shift` với 0 ≤ s ≤ 31 trùng `rshift_round` của `python/quant/int8.py`; với ReLU bật, kẹp ±127 + ReLU = `clip(·, 0, 127)` của mô hình.
Nhóm FPGA đã kiểm conv0 trên bo: 5 760 / 5 760 giá trị trùng golden.

**Bố cục cho Conv1D nhiều kênh** (ifmap theo thời gian-trước `X[t][ci]`): `step = stride·Cin = Cin`, `K = k·Cin`, và trọng số của kênh ra `co`
phải xếp **`[k][ci]`** (khác thứ tự `co→ci→k` của các tệp `.hex` v2). Đệm "same": thêm `(k−1)/2·Cin` số 0 ở mỗi đầu ifmap.

## 8. Thời gian và quy tắc dừng-chờ

- **Chỉ một lệnh đang chờ**: gửi lệnh, đợi trọn ACK / NACK / RESULT rồi mới gửi lệnh sau. Byte tới khi FPGA đang xử lý hoặc đang phát sẽ **bị bỏ**
  (bộ nhận chỉ bật ở trạng thái rảnh). Không có hàng đợi, không RTS/CTS.
- Gói nhận dở bị huỷ sau 100 ms không có byte. Lỗi khung UART và gói dở bị bỏ **im lặng** (không NACK).
- Sau một gói có LEN > 256 (NACK 1 ngay sau header), phần thân còn lại bị đọc như rác: **ngừng gửi ít nhất 100 ms** rồi mới đồng bộ lại.
- SEQ chỉ để ghép trả lời với lệnh (tăng mod 256), **không** dùng chống trùng. Nhóm FPGA quy định **không tự gửi lại** — đặc biệt không gửi lại RUN khi mất RESULT.
- Host của nhóm FPGA dùng thời gian chờ trả lời 2 s, không gửi lại; host của mình theo cùng quy tắc (câu hỏi §10.3).

## 9. Ánh xạ CNN Lite đóng băng lên ô

| Lớp | Phép | Chạy ở | Ghi chú |
|---|---|---|---|
| conv0 1→16, k7, L 360 | ô, preload bias, post (shift theo kênh, ReLU) | FPGA | K = 7, step 1; ifmap 366 B, trọng số 112 B |
| MaxPool 2 | max cặp | **PC** | `maxpool1d2.v` có trong RTL nhưng chưa nối vào top |
| conv1 16→32, k5, L 180 | ô | FPGA | K = 80; ifmap 184·16 = 2 944 B > 1 024 → nạp theo đoạn thời gian |
| conv2 32→64, k3, L 90 | ô | FPGA | K = 96 |
| conv3 64→64, k3, L 45 | ô | FPGA | K = 192 → `cols·192 ≤ 1 024` nên **tối đa 5 cột** mỗi ô |
| GAP | tổng theo thời gian (≈ 14 bit) | **PC** | không vừa đầu vào int8 của ô |
| FC1 phần RR (4 → 32) | ô, không preload, chỉ đọc `raw` | FPGA | rows = 1, K = 4 |
| FC1 phần GAP + ghép 3 lần dịch + bias + ReLU | **PC** | | ba lần `rshift_round` khác nhau (`Eg−E`, `Er−E`, `E−a_out`), post chỉ có một |
| FC2 32 → 4 | ô, preload bias, đọc `raw` = logit | FPGA | rows = 1, K = 32, cols = 4 |
| argmax | **PC** | | so sánh `>` chặt như `int8.py` |

**Đo ở B2** (`python/link/tile_sched.py`, `scripts/tile_plan.py`, kết quả `results/2026-11-14_5b3e043c_tile_plan/`): cả 1 306 nhịp golden
khớp từng bit ở mọi lớp. Mỗi nhịp: **513 ô, 1 681 giao dịch, 259 573 byte trên dây ≈ 22,5 s ở 115 200 baud** (chỉ thời gian truyền;
chưa tính thời gian tính trên FPGA và độ trễ USB-UART mỗi lượt: 1 681 × 1…16 ms = thêm 1,7…27 s). Riêng các khung RESULT đã là
129 789 B ≈ 11,3 s (mỗi ô trả 196 B raw + 49 B post, lớp conv chỉ cần phần post). Cả 1 306 nhịp ≥ 8,2 giờ.

| Lớp | Kênh / lần nạp trọng số | Vị trí / lần nạp ifmap | Ô | Byte | Giây @115 200 |
|---|---|---|---|---|---|
| conv0 | 16 | 360 | 156 | 45 980 | 4,0 |
| conv1 | 7 | 56 | 130 | 53 804 | 4,7 |
| conv2 | 7 | 28 | 130 | 70 796 | 6,2 |
| conv3 | 5 | 14 | 91 | 86 850 | 7,5 |
| FC1 (RR) | 32 | 1 | 5 | 1 617 | 0,1 |
| FC2 | 4 | 1 | 1 | 526 | 0,05 |

Giả định (cùng chuỗi lệnh): RESULT chỉ trả post → 13,8 s; 460 800 baud → 5,6 s; 921 600 baud → 2,8 s; cả hai → 1,7 s mỗi nhịp.

## 10. Chỗ khác nhau / câu hỏi gửi nhóm FPGA

Câu hỏi mở được theo dõi riêng trong repo dự án (không nằm trong gói bàn giao). Chỗ khác README ↔ RTL đáng nhớ nhất:
README ghi `CONFIG_POST` 4 byte và RESULT 36 / 45 byte (ô 3×3); RTL ô 7×7 đòi **8 byte** và trả **196 / 245 byte**. Tài liệu này theo RTL.
