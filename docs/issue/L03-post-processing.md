# L03. Kết quả tile bị xóa, chưa có hậu xử lý

Ghi chú cho giới hạn L03 trong [hw-accelerator-limitations.md](hw-accelerator-limitations.md). Cập nhật: 17/10/2026 (bản 2: PSM 2 chiều). Liên quan: L07 (vector unit), L06 (AGU dạng descriptor), L10 (valid/ready), L13 (bus quá rộng).

**Kết luận:** accelerator hiện dừng ở tổng tích lũy INT32 của một tile (`o_tile`). Chưa có khối nào cộng bias, requant, ReLU, MaxPool, cộng GAP hay ghi kết quả về SRAM, nên **không chạy nối được hai lớp conv**. Kết quả của tile n lại bị xóa khi tile n + 1 bắt đầu compute.

Thiết kế chọn: **PSM 2 chiều** dùng chính chuỗi shadow scan có sẵn trong mảng PE.

- **Chiều ghi (preload):** trước khi tính, bơm bias vào shadow register. MAC khởi tạo accumulator từ shadow thay cho hằng số 0, nên bias được cộng mà không tốn bộ cộng nào.
- **Chiều đọc (drain):** sau khi tính, dịch shadow ra thành **stream theo hàng**. Mỗi chu kỳ ra một kênh, ở mọi vị trí của tile. Shadow chính là thanh ghi chốt, nên không cần chốt 512 bit và không cần bus `o_tile` rộng.

Các khối cần làm:

1. **PSM 2 chiều**, cùng hai sửa đổi nhỏ ở PE và ở FSM của core.
2. **`post_proc`**: vector unit SIMD rộng ROWS làn, gồm requant → clamp → pool → GAP. Không còn tầng bias.
3. **`output_writer`**: ghi int8 vào bank đang ghi của ping-pong theo layout `X[t][c]`, có padding cho lớp sau.
4. **`conv1d_layer_top`**: nối các khối để chạy trọn **một lớp** và so với golden.

Việc chạy nhiều lớp liên tiếp (net sequencer, ROM cấu hình, swap bank giữa các lớp) nằm ngoài L03.

## 1. Hiện trạng

### Chuỗi shadow scan

Mỗi PE có một shadow register 32 bit. Shadow của các PE trong **một hàng** nối thành thanh ghi dịch, đi từ cột 0 sang cột cuối ([pe_array.v](../../rtl/cnn1d/core/pe_array.v)):

```
hàng r:   0 ──> [shadow c0] ──> [shadow c1] ──> [shadow c2] ──> endpoint = cột (active_cols-1)
          ^                                                     (scan_endpoint_mux)
          đầu vào cột 0 đang nối cứng bằng 0
```

| Hoạt động | Cơ chế | File |
| --- | --- | --- |
| Khởi tạo accumulator | `acc_value = acc_init_q ? 0 : o_mac`, tức luôn bắt đầu từ 0 | [MAC.v](../../rtl/cnn1d/pe/MAC.v) |
| Capture | **Mỗi PE tự capture** accumulator vào shadow, 2 chu kỳ sau khi cờ `last` của chính nó tới (lệch nhau theo độ trễ systolic) | [pe_top.v:65-70](../../rtl/cnn1d/pe/pe_top.v) |
| Drain | FSM core ở `S_DRAIN` bật `scan_en` trong `active_cols` chu kỳ. PSM nhận từ endpoint cột ac−1, ac−2, ..., 0 và gom thành `o_tile` | [cnn_control_fsm.v](../../rtl/cnn1d/core/cnn_control_fsm.v), [partial_sum_manager.v](../../rtl/cnn1d/core/partial_sum_manager.v) |
| Xóa kết quả | `S_LOAD` của core phát `psm_clr`, xóa `o_tile` của tile trước | [partial_sum_manager.v:94](../../rtl/cnn1d/core/partial_sum_manager.v) |

Như vậy **chiều đọc đã có**. Chiều ghi đã có sẵn đầu vào (cột 0, đang nối cứng bằng 0), chỉ thiếu đường đưa shadow vào accumulator.

### Đường ra của kết quả và các chỗ thiếu

| Chỗ | File | Ghi chú |
| --- | --- | --- |
| Kết quả chỉ đi ra cổng | [conv1d_tile_top.v:85](../../rtl/cnn1d/conv1d/conv1d_tile_top.v) (`o_tile`) | Bus ROWS × COLS × 32 bit (288 bit với 3×3; sẽ là 8.192 bit với 16×16) |
| Cổng ghi SRAM ifmap chỉ nối ra ngoài | [conv1d_tile_top.v:68-71](../../rtl/cnn1d/conv1d/conv1d_tile_top.v) | Hiện testbench/host dùng để nạp dữ liệu |
| Vị trí tile đã có, nhưng chưa ai dùng | [conv1d_layer_controller.v:66-67](../../rtl/cnn1d/conv1d/conv1d_layer_controller.v) (`o_out_row_base`, `o_out_col_base`) | |
| Layer controller chưa nối với tile top | — | Mới chỉ test riêng (95/95 PASS) |

### Thứ tự tile trong một lớp

`conv1d_layer_controller` lặp **Cout ở vòng trong, L ở vòng ngoài**: chạy hết mọi nhóm Cout của một nhóm vị trí rồi mới sang nhóm vị trí sau.

## 2. Phép tính cần làm (phải trùng golden từng bit)

Theo [cnn_explained.md §4](../cnn_explained.md) và `python/quant/int8.py`. Với mỗi phần tử ra ở vị trí t, kênh c:

```latex
\begin{aligned}
\mathrm{acc}[t][c] &= b_q[c] + \textstyle\sum_k w_q \cdot x_q \quad (\text{int32; bias đã nằm sẵn nhờ preload}) \\
y &= \mathrm{clamp}\big((\mathrm{acc} + 2^{s_c-1}) \ggg s_c,\; 0,\; 127\big) \\
p[t'] &= \max\big(y[2t'],\, y[2t'+1]\big) \quad (\text{nếu pool}) \\
S[c] &= \textstyle\sum_t y[t] \quad (\text{nếu GAP})
\end{aligned}
```

- Preload cho ra đúng `Σ + b_q`, giống hệt golden: phép cộng số nguyên có tính giao hoán, nên cộng bias trước hay sau đều như nhau.
- `>>>` là **dịch phải số học**. Làm tròn kiểu "cộng nửa rồi dịch", nên giá trị đúng bằng nửa luôn được làm tròn lên (về phía +∞), kể cả với số âm. Ví dụ: −3 với s = 1 cho (−3 + 1) >>> 1 = −1.
- ReLU gộp chung vào clamp(0, 127). MaxPool lấy trên giá trị **sau requant**; trùng golden vì requant là hàm đơn điệu không giảm.

| Lớp | Cout | out_len | Shift theo kênh | Bias | Pool | GAP | Padding cho lớp sau | So với golden |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| conv0 | 16 | 360 | 7..10 | −1004..129 | có | — | p = 2 | `beat_pool0` [16, 180] |
| conv1 | 32 | 180 | 7..9 | | có | — | p = 1 | `beat_pool1` [32, 90] |
| conv2 | 64 | 90 | 7..9 | | có | — | p = 1 | `beat_pool2` [64, 45] |
| conv3 | 64 | 45 | 8..10 | −14157..11152 | — | có | (không ghi) | `beat_gap` [64], int32 |

Đầu ra trước pool cũng có trong golden (`beat_conv0..3`), dùng để tìm lỗi ở tầng nào khi so sai.

## 3. Kiến trúc đề xuất

```
                         layer_cfg (cbase, pool_en, gap_en, p_next, Cout, obase, ...)
                                    |
conv1d_layer_controller --(row_base, col_base, active_rows/cols)------------------------+
          | tile_start                                                                  |
          v                                                                             v
 bias RAM --[PSM ghi]--> cột 0 ─> PE ─> PE ─> PE ─> endpoint --[PSM đọc]--> post_proc (SIMD ROWS làn)
 cbase+col_base+c         (preload, bơm cho mọi hàng)              (drain: 1 kênh × ROWS vị trí / chu kỳ)
                                                                                        |
                                                                   requant -> clamp -> pool -> GAP S[c]
                                                                                        |
                                                                    (t, c, y) ─> output_writer ─> ifmap ping-pong
                                                                                  (padding 0)      (mux với host)
```

### Trình tự cho mỗi tile

Ký hiệu ac = `active_cols`:

```
core FSM: S_IDLE -> S_LOAD -----> S_PRE_KICK -----> S_PRE_WAIT ----------> S_COMPUTE -------------> S_DRAIN ------------> S_DONE
                    (psm_clr:     (xung             (chờ PSM dịch đủ ac     (feeder_start;          (cho phép drain;
                     PSM chốt      preload_start)    lần, preload_done      lần MAC đầu ở mỗi PE:   PSM dịch ac lần,
                     ac, ar)                         -> feeder_start)       acc = shadow + w*x)     mỗi lần bắt tay)
```

- **PSM tự đếm số lần dịch, ở cả hai chiều.** FSM của core tạo `scan_en` qua thanh ghi, nên nó dịch lố thêm 1–2 lần sau khi đã đủ. Lúc drain thì không sao vì dữ liệu đã được chụp, nhưng lúc preload chỉ cần lố một lần là bias lệch sang cột bên cạnh. Vì vậy FSM chỉ phát xung `preload_start` rồi chờ `preload_done`. `scan_en` đưa vào mảng PE là `drain_shift | preload_shift`, cả hai đều do PSM tạo.
- **Thứ tự bơm:** giá trị bơm vào trước đi xa nhất. Vì vậy bơm `bias[base + ac − 1]` đầu tiên và `bias[base]` sau cùng, với base = cbase + col_base. Sau ac lần dịch, cột c giữ `bias[base + c]`.
- **Phát cho mọi hàng:** bias chỉ phụ thuộc kênh (cột), nên mỗi lần dịch chỉ đọc **một từ 32 bit** và đưa vào cột 0 của mọi hàng.
- **PRELOAD xong rồi mới phát `feeder_start`.** Có thể cho PRELOAD chạy chồng lên vài chu kỳ đầu, vì PE(r, c) chỉ nhận phần tử đầu sau khoảng r + c chu kỳ lệch, nhưng chỉ được vài chu kỳ nên không đáng làm phức tạp.
- **Tương thích ngược:** chế độ ZERO (bơm 0) cho đúng hành vi cũ. Đã kiểm chứng: regression và `tb_conv1d_tile_corner` PASS y nguyên (mục 6).

## 4. Quyết định thiết kế

### 4.1 Shadow là thanh ghi chốt; drain thành stream

| Cách | Trạng thái |
| --- | --- |
| ~~Chốt `o_tile` vào thanh ghi 512 bit trong `post_proc`~~ | **Bỏ** (là đề xuất của bản 1) |
| **Shadow giữ kết quả; drain ra post_proc từng lần dịch, có valid/ready** | Chọn |

- Sau khi capture, kết quả nằm sẵn trong shadow. Accumulator đã rảnh để tính tiếp, và shadow chỉ bị ghi đè khi chính PE đó capture lần sau. Vì vậy shadow **đã đóng vai thanh ghi chốt**.
- **Backpressure:** PSM chỉ chụp và dịch khi `i_drain_ready` = 1 (`drain_fire = drain_active && i_drain_ready`). Khi `post_proc` chưa nhận, shadow tự giữ giá trị và chuỗi dừng lại an toàn. Đã kiểm chứng với `ready` bật/tắt ngẫu nhiên khoảng 40% thời gian.
- Bỏ bus `o_tile`: chỉ còn ROWS × 32 bit mỗi chu kỳ ở đầu ra (128 bit với 4×4). Sửa được một phần L13. Có thể giữ `o_tile` sau một tham số trong lúc chuyển đổi, để các testbench cũ vẫn chạy.

### 4.2 Shadow dùng chung cho hai chiều

- Phải **drain xong tile n rồi mới preload tile n + 1**. Hiện tại FSM chạy nối tiếp nên điều này tự đúng.
- Gộp drain và preload vào cùng các lần dịch (kết quả ra ở endpoint đúng lúc bias mới vào ở cột 0) thì tiết kiệm được ac chu kỳ. Nhưng khi đó PSM phải biết trước `col_base` của tile sau. Không làm, vì chỉ được 4 chu kỳ trên khoảng 13·K chu kỳ mỗi tile (khoảng 0,03%).
- **Tile đầu tiên** không có drain trước nó, nên PRELOAD nằm ở đầu **mọi** tile.
- **Kể cả ở chế độ ZERO, PRELOAD vẫn phải dịch đủ ac lần.** Khi accumulator khởi tạo từ shadow, shadow của các cột active phải sạch. Nhưng khi drain dịch chính xác ac lần, các cột từ ac trở đi vẫn giữ giá trị cũ. Ví dụ tile trước có ac = 1 chỉ dịch 1 lần, nên cột 1 và 2 còn giá trị của tile trước; nếu tile sau có ac = 3 thì sẽ cộng nhầm. Đã xác nhận bằng thí nghiệm: bỏ việc dịch preload trong bản drain chính xác gây **62 lỗi ở chế độ ZERO**. Ở bản trung gian, lỗi này bị che đi vì FSM dịch lố khi drain và vô tình xóa sạch shadow.
- **Ràng buộc với L02 bước 2 (stream):** sau này nếu cho tile n + 1 compute trong lúc tile n đang drain, hai việc này sẽ tranh nhau shadow. Lúc đó phải gộp drain với preload như trên, hoặc thêm bank shadow thứ hai. Cần ghi nhớ khi làm L02 bước 2 và L04.

### 4.3 MaxPool: cặp pool nằm sẵn trong một lần drain

Mỗi lần dịch, endpoint của **mọi hàng** cùng đưa ra **một kênh**. Các hàng là các vị trí `row_base .. row_base + ROWS − 1`. Vậy một lần drain là một vector **ROWS vị trí liên tiếp của cùng kênh c**:

```
lần dịch j (kênh c = col_base + ac - 1 - j):   [ y(t0,c), y(t0+1,c), y(t0+2,c), y(t0+3,c) ]   (ROWS = 4)
pool:                                          [ max(y0,y1),         max(y2,y3)          ]
```

| ARRAY_ROWS | Cặp pool | Hệ quả |
| --- | --- | --- |
| **Chẵn (4)** | Luôn nằm trong một vector, vì `row_base` là bội của ROWS | Pool chỉ là ROWS/2 phép max song song, không cần lưu gì giữa các lần |
| Lẻ (3, hiện tại) | Vị trí cuối của tile ghép với vị trí đầu của nhóm sau. Nhóm sau chỉ tới **sau khi đã chạy hết mọi Cout** | Phải lưu một giá trị chờ cho **mọi** Cout (64 byte với ECG), hoặc đọc–so sánh–ghi lại trên SRAM |

**Đề xuất:** chuyển sang ARRAY 4×4 như spec v2, và thêm assert "pool_en thì ARRAY_ROWS chẵn và out_len chẵn". Các lớp ECG có out_len 360/180/90, đều chẵn. Tile cuối của conv2 có 90 − 22·4 = 2 hàng, vẫn chẵn.

### 4.4 `post_proc` là SIMD ROWS làn, shift đọc một lần cho mỗi vector

Mọi làn trong một vector cùng kênh c, nên dùng **chung một giá trị `shift[c]`**: mỗi lần drain chỉ đọc bảng shift một lần.

- **Thông lượng:** ac vector cho mỗi tile, tức 4 chu kỳ với 4×4.
- **Requant:** chỉ dịch bit bây giờ, nhưng đặt sau tham số `REQUANT = "SHIFT" | "M0_SHIFT"`, để sau này đổi sang kiểu `(a · M0) >> shift` của TFLite chỉ phải thay một tầng.
- **Độ rộng trung gian:** tính `acc + 2^(s−1)` ở ACC_W + 1 bit. Với ECG, |acc| ≤ 192·127·127 + 14.157 ≈ 3,1 triệu, rất xa biên, nhưng nên làm tổng quát.

### 4.5 Ghi ra SRAM: writer là nút thắt nhỏ

Layout channel-last `X[t][c]` đặt các vị trí của cùng một kênh cách nhau Cout ô trong bộ nhớ. Một vector drain (ROWS hoặc ROWS/2 phần tử) vì vậy rơi vào các địa chỉ **không liền nhau**. Với SRAM 1 byte, writer phải ghi 1 phần tử mỗi chu kỳ:

| | Chu kỳ ghi mỗi tile (4×4) | So với compute một tile |
| --- | --- | --- |
| Có pool | ac · ROWS/2 = 8 | ≥ K + ~17 chu kỳ, cộng thời gian nạp |
| Không pool | ac · ROWS = 16 | |

Đặt một FIFO nhỏ (4–8 vector) giữa `post_proc` và writer. Khi FIFO đầy, backpressure dừng `scan_en`. Khi làm L04 (SRAM nhiều bank, rộng hơn), writer ghi song song được.

### 4.6 Tranh chấp cổng ghi SRAM

Cổng ghi ifmap (`i_ifmap_mem_wr_*`) hiện chỉ có host hoặc testbench dùng. Trong `conv1d_layer_top`, thêm một bộ mux: khi lớp đang chạy thì writer được ghi, còn lại thì host được ghi. Host chỉ nạp `z_q` trước lớp đầu tiên nên hai bên không bao giờ ghi cùng lúc. Trong lúc chạy một lớp, compute đọc **bank đang đọc** (đầu vào lớp n), còn writer ghi **bank đang ghi** (đầu vào lớp n + 1). Swap bank sau mỗi lớp là việc của sequencer.

### 4.7 Hướng NPU: nạp lại tổng riêng phần

Cùng cổng ghi đó, nếu đưa vào **một giá trị riêng cho mỗi hàng** (thay vì phát chung bias), sẽ nạp được tổng riêng phần của lần tính trước: C = A₁B₁ + A₂B₂. Đây là cách **chia K thành nhiều đoạn**, bắt buộc với CNN2D (K = 3×3×256 = 2304). Cách này không phụ thuộc độ sâu FIFO. Việc này cần cổng ghi rộng ROWS × 32 bit và một bộ nhớ int32 (accumulator SRAM). Nằm ngoài ECG, nhưng interface PSM ở mục 5 nên chừa sẵn chế độ này.

## 5. Interface

### Sửa đổi ở PE và mảng PE

| Chỗ | Hiện tại | Sau khi sửa |
| --- | --- | --- |
| `MAC.v` | `acc_value = acc_init_q ? 32'b0 : o_mac` | `acc_value = acc_init_q ? i_init : o_mac`, với `i_init` = shadow của chính PE đó (cổng `o_pe` của `ProcessingElement`) |
| `pe_array.v`, cột 0 | `local_scan_in = 0` | `local_scan_in = i_preload[row*ACC_W +: ACC_W]` (cổng mới ROWS × ACC_W) |

### FSM của core (`cnn_control_fsm`)

| Trạng thái | Việc |
| --- | --- |
| `S_LOAD` | Chỉ phát `psm_clr` (PSM chốt ac, ar). **Không phát `feeder_start` nữa** |
| **`S_PRE_KICK`** (mới, 3'd5) | Phát xung `o_preload_start` |
| **`S_PRE_WAIT`** (mới, 3'd6) | Chờ `i_preload_done`, rồi phát `feeder_start` và sang `S_COMPUTE` |
| `S_COMPUTE` | Như cũ |
| `S_DRAIN` | Như cũ: bật `o_psm_capture_en` suốt trạng thái này. Từ bước 2, tín hiệu đó chỉ còn nghĩa là "cho phép drain"; `o_scan_en` của FSM không còn nối vào mảng PE |

### PSM 2 chiều (bản đã làm thử)

| Cổng | Hướng | Ý nghĩa |
| --- | --- | --- |
| `i_clr`, `i_active_rows`, `i_active_cols` | vào | Như cũ: chốt kích thước tile |
| `i_preload_start` | vào | Xung từ FSM, đến sau `i_clr` |
| `i_preload_mode` | vào | 0 = ZERO (hành vi cũ), 1 = BIAS (phát chung cho mọi hàng). PSUM để dành cho NPU, khi đó cổng này mở rộng thành 2 bit |
| `o_bias_rd_en`, `o_bias_rd_idx` → `i_bias_rd_data` | ra/vào | Đọc bias RAM. `o_bias_rd_idx` là **chỉ số cột trong tile** (ac − 1 xuống 0); bên ngoài cộng thêm base = cbase + col_base. RAM trễ 1 chu kỳ |
| `o_preload_shift`, `o_preload_vec[ROWS*ACC_W]` | ra | Dịch chuỗi một lần, kèm giá trị cho cột 0 của mọi hàng. Ngoài lúc preload thì vec luôn bằng 0 |
| `o_preload_done` | ra | Xung, ngay sau lần dịch cuối |
| `i_capture_en` | vào | Từ FSM, bật suốt `S_DRAIN`: cho phép drain |
| `i_drain_ready` → `o_drain_valid`, `o_drain_idx`, `o_drain_vec[ROWS*ACC_W]` | stream | Một lần bắt tay = cột `o_drain_idx` (kênh col_base + idx) ở ROWS vị trí. Thứ tự idx: ac − 1 xuống 0 |
| `o_drain_shift` | ra | Dịch chuỗi đúng một lần cho mỗi lần bắt tay |
| `o_tile`, `o_tile_valid`, `o_capture_done` | ra | Vẫn được giữ trong lúc chuyển đổi (các testbench cũ dùng); bỏ khi `post_proc` dùng stream |

### `post_proc` (SIMD ROWS làn)

| Cổng | Hướng | Ý nghĩa |
| --- | --- | --- |
| `i_valid`, `o_ready`, `i_c`, `i_vec[ROWS*ACC_W]`, `i_row_base`, `i_active_rows` | stream vào | Từ PSM đọc |
| `i_pool_en`, `i_gap_en`, `i_clamp_lo`, `i_clamp_hi`, `i_cbase` | vào | Cấu hình lớp |
| `o_shift_addr` → `i_shift` | ra/vào | Một lần đọc cho mỗi vector |
| `o_valid`, `i_ready`, `o_c`, `o_t0`, `o_n`, `o_vec[ROWS*8]` | stream ra | n phần tử int8 ở các vị trí t0..t0 + n − 1 của kênh c. n = active_rows, hoặc active_rows/2 nếu pool |
| `i_gap_clr`, `o_gap[COUT_MAX*GAP_W]` | | Xóa và đọc tổng GAP. Với ECG, GAP_W = 13 bit là đủ (45·127 = 5.715) |

### `output_writer`

| Cổng | Hướng | Ý nghĩa |
| --- | --- | --- |
| `i_valid`, `o_ready`, `i_c`, `i_t0`, `i_n`, `i_vec` | stream vào | Từ `post_proc` (qua FIFO nhỏ) |
| `i_obase`, `i_cout`, `i_out_len`, `i_p_next` | vào | Cấu hình. `out_len` là chiều dài **sau** pool |
| `i_pad_start`, `o_pad_done` | vào/ra | Ghi 0 vào 2·p_next·Cout ô padding |
| `o_wr_en`, `o_wr_addr`, `o_wr_data` | ra | Mỗi chu kỳ ghi một phần tử tại `obase + (t + p_next)·Cout + c` |

## 6. Hướng dẫn từng bước

Bước 0–2 **đã được làm thử và kiểm chứng ngày 17/10/2026** trên bản sao của commit `e24f0bb`. Kết quả nằm trong 2 file patch, áp lần lượt:

| Patch | Nội dung | File bị sửa |
| --- | --- | --- |
| [`patches/L03-step0-1-psm-preload.patch`](patches/L03-step0-1-psm-preload.patch) | Bước 0 + 1: preload (ZERO/BIAS) | `MAC.v`, `ProcessingElement.v`, `pe_array.v`, `cnn_control_fsm.v`, `partial_sum_manager.v`, `cnn_compute_core.v`, `conv1d_tile_top.v`, `tb_conv1d_tile_top.v`, `tb_conv1d_tile_corner.v` |
| [`patches/L03-step2-psm-drain-stream.patch`](patches/L03-step2-psm-drain-stream.patch) | Bước 2: drain thành stream có backpressure | `partial_sum_manager.v`, `cnn_compute_core.v`, `conv1d_tile_top.v`, `tb_conv1d_tile_top.v`, `tb_conv1d_tile_corner.v` |

Áp patch bước 0–1 (đứng ở gốc repo, chạy trong Git Bash):

```bash
git apply docs/issue/patches/L03-step0-1-psm-preload.patch
```

Chạy regression trước khi áp tiếp:

```bash
rtl/cnn1d/sim/run_tb.sh all
```

Áp patch bước 2:

```bash
git apply docs/issue/patches/L03-step2-psm-drain-stream.patch
```

Lệnh `git apply` có thể in cảnh báo "trailing whitespace". Đó chỉ là khoảng trắng cuối dòng, giống style sẵn có trong repo, không ảnh hưởng gì.

Nếu muốn tự sửa tay để hiểu từng chỗ, làm theo các mục dưới đây. Code giống hệt nội dung trong patch.

### Bước 0. Preload với bias = 0 (không được đổi hành vi)

**0.1 `pe/MAC.v`**: thêm cổng khởi tạo và thay hằng số 0.

```verilog
	input wire[7:0] i_a, i_b,
	input wire[31:0] i_init,     // giá trị khởi tạo accumulator (shadow của PE)
...
	// Mux 2:1
	assign acc_value = acc_init_q ? i_init : o_mac;      // trước: acc_init_q ? 32'b0 : o_mac
```

**0.2 `pe/ProcessingElement.v`**: nối shadow của chính PE vào `i_init`. `o_pe` là đầu ra của `shadow_register`.

```verilog
	MAC mac(
		...
		.i_a(i_weight), .i_b(i_image),
		.i_init(o_pe),
		.o_mac(mac_value)
	);
```

**0.3 `core/pe_array.v`**: thêm cổng `i_preload`, cột 0 lấy từ cổng này.

```verilog
  // Preload: giá trị bơm vào cột 0 của từng hàng khi dịch chuỗi shadow
  input wire [ARRAY_ROWS*ACC_W-1:0] i_preload,
...
				if(col == 0) begin: GEN_SCAN_FIRST
					assign local_scan_in = i_preload[row*ACC_W +: ACC_W];   // trước: {ACC_W{1'b0}}
```

**0.4 `core/cnn_control_fsm.v`**: thêm 2 cổng và 2 trạng thái.

```verilog
	// Preload (PSM tự đếm số lần dịch)
	output reg o_preload_start,
	input wire i_preload_done,
...
	localparam S_PRE_KICK = 3'd5; // phát xung preload_start cho PSM
	localparam S_PRE_WAIT = 3'd6; // chờ PSM dịch xong ac lần
...
	// thêm vào CẢ nhánh reset VÀ khối gán mặc định đầu mỗi chu kỳ:
	o_preload_start <= 1'b0;
...
				S_LOAD: begin
					busy <= 1'b1;
					// psm_clr: PSM chốt active_rows/cols của tile mới
					o_psm_clr <= 1'b1;
					state <= S_PRE_KICK;          // trước: phát cả feeder_start ở đây rồi sang S_COMPUTE
				end
				S_PRE_KICK: begin
					busy <= 1'b1;
					o_preload_start <= 1'b1;
					state <= S_PRE_WAIT;
				end
				S_PRE_WAIT: begin
					busy <= 1'b1;
					// Shadow đã chứa giá trị khởi tạo -> mới cho feeder chạy
					if(i_preload_done) begin
						o_feeder_start <= 1'b1;
						state <= S_COMPUTE;
					end
				end
```

**0.5 `core/partial_sum_manager.v`**: thêm cổng và khối preload. Đây là pipeline 2 tầng: tầng đọc phát địa chỉ, tầng dịch nhận dữ liệu bias đã về rồi dịch.

```verilog
	// thêm vào danh sách cổng (sau o_capture_done):
	input wire i_preload_start,                      // xung, sau i_clr
	input wire i_preload_mode,                       // 0: ZERO, 1: BIAS (phát chung cho mọi hàng)
	output wire o_bias_rd_en,                        // đọc bias RAM
	output wire [COL_W-1:0] o_bias_rd_idx,           // cột trong tile: ac-1 .. 0
	input wire [ACC_W-1:0] i_bias_rd_data,           // có ở chu kỳ sau o_bias_rd_en
	output wire o_preload_shift,                     // dịch chuỗi shadow 1 lần
	output wire [ARRAY_ROWS*ACC_W-1:0] o_preload_vec,
	output reg o_preload_done                        // xung, sau lần dịch cuối
...
	// đặt sau khai báo active_rows_q / active_cols_q:
	reg pre_busy, pre_shift_q, pre_last_q;
	reg [COL_W-1:0] pre_idx;

	assign o_bias_rd_en = pre_busy & i_preload_mode;
	assign o_bias_rd_idx = active_cols_q - 1'b1 - pre_idx;
	assign o_preload_shift = pre_shift_q;
	// Ngoài lúc preload luôn bơm 0
	assign o_preload_vec = (pre_shift_q && i_preload_mode) ? {ARRAY_ROWS{i_bias_rd_data}} : {(ARRAY_ROWS*ACC_W){1'b0}};

	always @(posedge i_clk or posedge i_reset) begin
		if(i_reset) begin
			pre_busy <= 1'b0;
			pre_idx <= {COL_W{1'b0}};
			pre_shift_q <= 1'b0;
			pre_last_q <= 1'b0;
			o_preload_done <= 1'b0;
		end else begin
			pre_shift_q <= pre_busy;
			pre_last_q <= pre_busy && (pre_idx == active_cols_q - 1'b1);
			o_preload_done <= pre_last_q;
			if(i_preload_start && !pre_busy) begin
				pre_busy <= 1'b1;
				pre_idx <= {COL_W{1'b0}};
			end else if(pre_busy) begin
				if(pre_idx == active_cols_q - 1'b1) pre_busy <= 1'b0;
				else pre_idx <= pre_idx + 1'b1;
			end
		end
	end
```

Giản đồ thời gian với ac = 3, b[i] là bias của cột i:

```
chu kỳ:          t0     t1     t2     t3     t4
pre_busy:        1      1      1      0      0
o_bias_rd_idx:   2      1      0      -      -      (địa chỉ phát ở chu kỳ n -> dữ liệu về ở n+1)
pre_shift_q:     0      1      1      1      0      (dịch ở sườn cuối chu kỳ)
o_preload_vec:   0      b[2]   b[1]   b[0]   0
o_preload_done:  0      0      0      0      1      (FSM thấy -> feeder_start)
sau t3: cột 0 = b[0], cột 1 = b[1], cột 2 = b[2]
```

**0.6 `core/cnn_compute_core.v`**: thêm cổng, nối FSM ↔ PSM ↔ mảng PE.

```verilog
	// thêm vào danh sách cổng:
	input wire i_preload_mode,
	output wire o_bias_rd_en,
	output wire [COL_W-1:0] o_bias_rd_idx,
	input wire [ACC_W-1:0] i_bias_rd_data
...
	wire preload_start, preload_done, preload_shift;
	wire [ARRAY_ROWS*ACC_W-1:0] preload_vec;
...
	// cnn_control_fsm:
		.o_preload_start(preload_start),
		.i_preload_done(preload_done),
	// pe_array:
		.i_scan_en(scan_en | preload_shift),      // bước 2 đổi thành drain_shift | preload_shift
		.i_preload(preload_vec),
	// partial_sum_manager:
		.i_preload_start(preload_start),
		.i_preload_mode(i_preload_mode),
		.o_bias_rd_en(o_bias_rd_en),
		.o_bias_rd_idx(o_bias_rd_idx),
		.i_bias_rd_data(i_bias_rd_data),
		.o_preload_shift(preload_shift),
		.o_preload_vec(preload_vec),
		.o_preload_done(preload_done)
```

**0.7 `conv1d/conv1d_tile_top.v`**: chuyển tiếp 4 cổng ra ngoài. Bias RAM nằm ngoài tile top; địa chỉ = (cbase + col_base) + `o_bias_rd_idx`.

```verilog
	input wire i_preload_mode,
	output wire o_bias_rd_en,
	output wire [COL_W-1:0] o_bias_rd_idx,
	input wire [ACC_W-1:0] i_bias_rd_data
...
	// u_compute_core:
		.i_preload_mode(i_preload_mode),
		.o_bias_rd_en(o_bias_rd_en),
		.o_bias_rd_idx(o_bias_rd_idx),
		.i_bias_rd_data(i_bias_rd_data)
```

**0.8 Testbench.** Mọi testbench có instance `conv1d_tile_top` phải nối cổng mới. Một cổng vào để trống sẽ trôi ở `z` và làm hỏng kết quả: `tb_conv1d_tile_corner` báo 88 lỗi khi chưa nối.

```verilog
	// tb_conv1d_tile_top.v
		.i_preload_mode(1'b0),
		.i_bias_rd_data(32'd0)
```

**Xong khi:** với `i_preload_mode = 0`, `run_tb.sh all` PASS y nguyên và `tb_conv1d_tile_corner` PASS 36/36. *Kết quả đã đạt: 10/10 testbench và 36/36.*

### Bước 1. Bật preload bias và kiểm chứng

Phần RTL đã có ở bước 0; chỉ cần đặt `i_preload_mode = 1` và nối bias RAM. Phần việc chính là testbench. Bản trong patch đã mở rộng `tb_conv1d_tile_corner` như sau:

- **3 cấu hình chạy song song:** DEPTH 16 + ZERO, DEPTH 16 + BIAS, DEPTH 1024 + BIAS.
- **Mô hình bias RAM** trễ 1 chu kỳ, với `bias_base` đóng vai cbase + col_base:

```verilog
	always @(posedge i_clk) begin
		if (o_bias_rd_en) begin
			bias_q <= bias_mem[bias_base + o_bias_rd_idx];
			bias_reads = bias_reads + 1;
		end
	end
```

- **Mô hình tham chiếu cộng bias:** `if (PRELOAD_BIAS && r < ar && c < ac) acc = bias_mem[bias_base + c];` trước khi cộng dồn tích.
- Mỗi tile sinh `bias_base` ngẫu nhiên và bias theo mode: −2^20, +2^20, 0, hoặc ngẫu nhiên ±2^20.
- **C18:** reset giữa lúc preload (khi core ở `S_PRE_WAIT`), rồi chạy tile tiếp.
- **C19:** chế độ BIAS phải có đọc bias RAM; chế độ ZERO thì không được đọc lần nào.

*Kết quả đã đạt:* PASS 59/59 trên 3 cấu hình (157 và 103 lần đọc bias). Cài lỗi cố ý (chỉ chạy các cấu hình DEPTH 16):

| Lỗi cố ý | ZERO | BIAS |
| --- | --- | --- |
| Bơm sai thứ tự (cột 0 trước) | 0 lỗi | **240 lỗi** |
| Thiếu một lần dịch | **FAIL** | **FAIL** |
| Không dịch khi preload | 0 lỗi ở bản bước 1 / **62 lỗi** ở bản bước 2 (xem 4.2) | **367 lỗi** |
| Quên phát cho hàng 1 | 0 lỗi | **125 lỗi** |
| MAC vẫn khởi tạo bằng 0 | 0 lỗi | **367 lỗi** |

Cấu hình ZERO không lỗi ở các dòng đó là đúng: khi bias bằng 0 thì các lỗi này không làm thay đổi kết quả.

### Bước 2. Drain thành stream có backpressure

FSM giữ nguyên. `o_psm_capture_en` (bật suốt `S_DRAIN`) trở thành tín hiệu "cho phép drain", và PSM tự quyết định thời điểm chụp và dịch.

**2.1 `core/partial_sum_manager.v`**: thêm cổng stream, đặt điều kiện chụp là bắt tay.

```verilog
	// thêm vào danh sách cổng:
	input wire i_drain_ready,
	output wire o_drain_valid,
	output wire [COL_W-1:0] o_drain_idx,            // cột trong tile: ac-1 .. 0
	output wire [ARRAY_ROWS*ACC_W-1:0] o_drain_vec,
	output wire o_drain_shift                       // dịch chuỗi shadow 1 lần
...
	// đặt sau scan_endpoint_mux:
	wire drain_active = i_capture_en && !o_tile_valid && (active_rows_q != {ROW_W{1'b0}}) && (active_cols_q != {COL_W{1'b0}});
	wire drain_fire = drain_active && i_drain_ready;
	assign o_drain_valid = drain_active;
	assign o_drain_idx = active_cols_q - 1'b1 - cap_cnt;
	assign o_drain_vec = scan_pe;
	assign o_drain_shift = drain_fire;
...
	// trong khối capture, thay điều kiện cũ:
	//   else if(i_capture_en && !o_tile_valid && (active_rows_q != 0) && (active_cols_q != 0)) begin
	else if(drain_fire) begin
```

**2.2 `core/cnn_compute_core.v`**: chuyển tiếp stream; `scan_en` của mảng PE lấy hoàn toàn từ PSM.

```verilog
	input wire i_drain_ready,
	output wire o_drain_valid,
	output wire [COL_W-1:0] o_drain_idx,
	output wire [ARRAY_ROWS*ACC_W-1:0] o_drain_vec
...
	wire drain_shift;
	// pe_array:
		.i_scan_en(drain_shift | preload_shift),   // PSM đếm chính xác cả hai chiều
	// partial_sum_manager:
		.i_drain_ready(i_drain_ready),
		.o_drain_valid(o_drain_valid),
		.o_drain_idx(o_drain_idx),
		.o_drain_vec(o_drain_vec),
		.o_drain_shift(drain_shift)
```

Sau bước này, `scan_en` do FSM tạo không còn nối vào đâu (Quartus sẽ cảnh báo tín hiệu thừa). Có thể giữ lại để debug, hoặc xóa khi dọn code.

**2.3 `conv1d/conv1d_tile_top.v`**: chuyển tiếp 4 cổng stream. Đặt `i_drain_ready = 1` cố định thì hành vi giống hệt bản cũ.

**2.4 Testbench:**
- `tb_conv1d_tile_top.v` nối `.i_drain_ready(1'b1)`.
- `tb_conv1d_tile_corner.v` có thêm tham số `DRAIN_BP`. Khi bật, `i_drain_ready` ngẫu nhiên khoảng 40% thời gian (bật ở 2 cấu hình BIAS).
- **Bộ thu stream** ghép các lần bắt tay thành ma trận rồi so với kỳ vọng, và kiểm tra **số lần bắt tay đúng bằng ac**:

```verilog
	always @(posedge i_clk) begin
		if (o_drain_valid && i_drain_ready) begin
			for (str_r = 0; str_r < ARRAY_ROWS; str_r = str_r + 1)
				str_c[str_r * ARRAY_COLS + o_drain_idx] = o_drain_vec[str_r * ACC_W +: ACC_W];
			str_cnt = str_cnt + 1;
		end
	end
```

*Kết quả đã đạt:* regression 11/11 PASS, `tb_conv1d_tile_corner` 59/59 với backpressure. Cài lỗi cố ý "chuỗi vẫn dịch khi `ready = 0`" thì cấu hình có backpressure báo 618 lỗi.

`o_tile` vẫn được PSM ghép đầy đủ ở bước này để các testbench cũ chạy được. Bỏ nó khi `post_proc` (bước 3) đã nhận stream.

### Bước 3. `post_proc`

1. Viết `rtl/cnn1d/vec/post_proc.v` và `tb/tb_post_proc.v`. Mô hình tham chiếu cài đúng công thức ở mục 2. Các trường hợp biên:
   - acc âm và dương; acc vừa đúng ±0,5 sau khi dịch (kiểm tra hướng làm tròn); s = 0 và s lớn nhất;
   - y tính ra đúng 127, 128 (bị kẹp), 0, −1 (bị kẹp);
   - hai giá trị bằng nhau trong cặp pool;
   - `active_rows` < ROWS (tile cuối);
   - backpressure từ writer;
   - GAP cộng qua nhiều tile, rồi `i_gap_clr`.
2. Cài lỗi cố ý: đổi `>>>` thành `>>`, bỏ phần làm tròn, ghép cặp pool sai.

### Bước 4. `output_writer`

Kiểm tra:
- địa chỉ đúng công thức ở mục 5;
- đủ 2·p_next·Cout ô padding bằng 0;
- không ghi lố `obase + (out_len + 2·p_next)·Cout`;
- đúng thứ tự khi `i_valid` ngắt quãng.

### Bước 5. `conv1d_layer_top`: một lớp hoàn chỉnh, so với golden

1. Nối layer controller, tile top, PSM 2 chiều, `post_proc`, FIFO, `output_writer` và mux cổng ghi.
2. Cần dữ liệu golden ở layout của accelerator. Script `export_hex.py` (cổng 1 trong spec v2, thuộc phần ML) phải xuất:
   - weight của từng lớp đã transpose sang `W[c][j][i]` (golden lưu `[Cout][Cin][k]`);
   - bias int32 và shift của từng kênh;
   - đầu vào từng lớp ở layout `X[t][i]` có padding, lấy từ `beat_z_q` (conv0) hoặc `beat_pool*` của lớp trước (golden lưu `(C, L)`, phải transpose);
   - đầu ra kỳ vọng: `beat_pool0..2` (đã transpose và pad theo p_next), `beat_gap`.
   Golden nằm ở `export/golden/<đoạn>_<tên>.bin`; kiểu và shape ghi trong `export/golden/manifest.json`.
3. `tb/tb_conv1d_layer_top.v`, cấu hình ARRAY 4×4, `FEEDER_DEPTH = 1024`, SRAM 4096:
   - conv0 → so cả bank đầu ra với `beat_pool0` (đã pad);
   - conv1, conv2: dùng `beat_pool` của lớp trước làm đầu vào, so với `beat_pool` của lớp này;
   - conv3 → so S[64] với `beat_gap`;
   - ít nhất 10 beat thuộc nhiều lớp N/S/V/F.
4. Xong khi: cả 4 lớp trùng golden 0 sai khác trên mọi beat đã thử. Kết quả này là cổng 3 trong spec v2, chỉ còn thiếu phần sequencer.

## 7. Danh sách kiểm chứng

- [ ] Bước 0: preload với bias = 0, regression và `tb_conv1d_tile_corner` PASS y nguyên. *(Đã làm thử: 10/10 và 36/36. Còn phải áp patch vào repo.)*
- [ ] Preload bias: đúng thứ tự, phát đủ cho mọi hàng, tile thiếu cột, bias âm và lớn, reset giữa lúc preload. *(Đã làm thử: 59/59, bắt được 5/5 lỗi cố ý.)*
- [ ] PRELOAD vẫn dịch đủ ac lần ở chế độ ZERO (không bỏ qua để "tối ưu").
- [ ] Drain stream: backpressure dài vẫn đúng; số lần bắt tay đúng bằng ac. *(Đã làm thử: 59/59 với ready ngẫu nhiên.)*
- [ ] Xóa `o_tile` và `scan_en` thừa của FSM sau khi `post_proc` dùng stream.
- [ ] `post_proc`: làm tròn đúng với số âm; clamp ở 0 và 127; pool khi hai giá trị bằng nhau; tile thiếu hàng; backpressure.
- [ ] Assert: `pool_en` thì ARRAY_ROWS chẵn và out_len chẵn.
- [ ] `output_writer`: địa chỉ, padding bằng 0, không ghi lố.
- [ ] `conv1d_layer_top`: conv0..conv2 trùng `beat_pool*`, conv3 trùng `beat_gap`, ≥ 10 beat.
- [ ] Tổng hợp: thêm khoảng 32 LE mỗi PE cho mux khởi tạo; Fmax không tụt (đường MAC: mux → CLA 32 bit → accumulator).
