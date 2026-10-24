# L02. Loader không có backpressure, K bị giới hạn bởi độ sâu FIFO

Ghi chú cho giới hạn L02 trong [hw-accelerator-limitations.md](hw-accelerator-limitations.md). Cập nhật: 17/10/2026. Cần đọc [L01-ring-buffer.md](L01-ring-buffer.md) trước.

**Kết luận:** mỗi tile hiện chạy theo kiểu "nạp xong rồi mới tính", và loader ghi vào FIFO mà không kiểm tra cờ full. Khi K = k·Cin lớn hơn độ sâu FIFO, các lần ghi thừa bị bỏ và **tile treo ở trạng thái COMPUTE**, không có cờ báo lỗi nào. Mô hình ECG cần K tới 192, trong khi FIFO hiện chỉ sâu 16.

Sửa theo 2 bước:

- **Bước 1 (cho ECG):** FIFO sâu 1024, loader có backpressure, và chặn cấu hình K > DEPTH bằng cờ lỗi.
- **Bước 2 (stream, mở đường cho CNN2D):** loader nạp xen kẽ giữa các lane, compute chạy song song với load, FIFO nhỏ lại.

## 1. Luồng hiện tại

```
conv1d_tile_top (mỗi tile)
  S_LOAD_START -> S_LOAD_WAIT ---------------------> S_CORE_START -> S_COMPUTE -> S_DONE
                  (chờ loader ifmap VÀ weight xong)    (bắt đầu feeder + PE)

conv1d_ifmap_loader (mỗi phần tử, ~3 chu kỳ)
  S_REQ (đọc SRAM) -> S_WAIT (chờ rd_valid) -> S_WRITE (ghi vào FIFO của lane row_cnt_q)

Thứ tự AGU sinh địa chỉ: hàng ở vòng ngoài, k ở vòng trong
  hàng 0: k = 0..K-1 -> hàng 1: k = 0..K-1 -> ... -> hàng R-1

ifmap_feeder / weight_feeder: lockstep
  chỉ đọc khi MỌI lane đang dùng đều có dữ liệu, đếm k từ 0 tới K-1 để sinh cờ first/last
```

| Chỗ | File |
| --- | --- |
| Tile chờ nạp xong mới tính | [conv1d_tile_top.v:497](../../rtl/cnn1d/conv1d/conv1d_tile_top.v) (`S_LOAD_WAIT`) |
| Loader ghi mà không xét cờ full | [conv1d_ifmap_loader.v:148](../../rtl/cnn1d/conv1d/conv1d_ifmap_loader.v), [conv1d_weight_loader.v:267](../../rtl/cnn1d/conv1d/conv1d_weight_loader.v) (`S_WRITE`) |
| AGU luôn tăng địa chỉ ở `S_WRITE` | [conv1d_ifmap_loader.v:64](../../rtl/cnn1d/conv1d/conv1d_ifmap_loader.v), [conv1d_weight_loader.v:101](../../rtl/cnn1d/conv1d/conv1d_weight_loader.v) (`agu_advance = (state_q == S_WRITE)`) |
| Thứ tự hàng ngoài, k trong | [conv1d_address_generating.v:129](../../rtl/cnn1d/conv1d/conv1d_address_generating.v) |
| Cờ full đã có dây nhưng chưa nối vào loader | [conv1d_ifmap_feeder_top.v:53, 125](../../rtl/cnn1d/conv1d/conv1d_ifmap_feeder_top.v) (`feeder_mem_full`) |
| Feeder chạy lockstep | [ifmap_feeder.v:48-58](../../rtl/cnn1d/feeder/ifmap_feeder.v) (`all_ready`) |

## 2. Chuyện gì xảy ra khi K > DEPTH

Thí nghiệm ngày 17/10/2026: lấy RTL đã commit (FIFO cũ, DEPTH = 16), sửa test đầu tiên của `tb_conv1d_tile_top` chỉ ở tham số K:

| K | Kết quả | Giải thích |
| --- | --- | --- |
| 3 | Tile chạy xong | Bình thường |
| 16 | Tile chạy xong, giá trị ra là `x` | FIFO vừa đủ chứa. Giá trị `x` do testbench chỉ nạp dữ liệu cho K = 3, nên phần đọc thêm là ô SRAM chưa khởi tạo. Đây không phải lỗi của DUT |
| 20 | **Treo ở COMPUTE** cho tới khi testbench báo `Timeout waiting for o_tile_valid` | Xem bên dưới |

Cơ chế treo khi K = 20, DEPTH = 16:

1. Loader ghi 20 phần tử vào lane 0. FIFO nhận 16 phần tử, bỏ qua 4 lần ghi còn lại, và loader không hề biết.
2. Loader báo xong, tile chuyển sang COMPUTE. Feeder đếm k từ 0 tới 19.
3. Đọc hết 16 phần tử thì lane rỗng, `all_ready = 0`, feeder đứng chờ. Không ai ghi thêm nữa, nên chờ mãi.

Như vậy lỗi không làm sai kết quả mà làm **treo cả accelerator**. Khi tích hợp vào SoC, triệu chứng sẽ là `busy` không bao giờ hạ, rất khó lần ra nguyên nhân.

## 3. Những gì đã sẵn sàng cho stream

Không cần sửa mảng PE. Hai điều kiện để dữ liệu chảy ngắt quãng mà vẫn đúng đã có sẵn:

- **PE chịu được chu kỳ trống:** `compute_en = act_valid & wgt_valid`, và cờ valid được đẩy theo dữ liệu qua từng hop ([pe_top.v](../../rtl/cnn1d/pe/pe_top.v)). Ở chu kỳ không có dữ liệu, PE đơn giản là không cộng.
- **Feeder lockstep:** mọi lane ifmap và weight cùng dừng và cùng chạy. Độ lệch giữa các hàng (`feed_delay`) là cố định, nên các cặp activation/weight vẫn gặp nhau đúng PE.

Chỉ cần sửa **loader** và **FSM của tile**.

## 4. Vì sao stream đòi phải đổi thứ tự nạp

Nếu chỉ thêm backpressure và cho compute chạy song song, nhưng FIFO nhỏ hơn K, thì hệ thống kẹt cứng (deadlock):

```
DEPTH = 32, K = 192, R = 4 hàng, thứ tự hiện tại (hàng ngoài, k trong):

  loader: ghi lane 0 -> 32 phần tử -> lane 0 đầy -> loader dừng chờ lane 0
  feeder: chờ cả 4 lane có dữ liệu -> lane 1, 2, 3 rỗng -> không đọc
  => loader chờ feeder, feeder chờ loader: deadlock
```

Cách sửa là **nạp xen kẽ** (k ở vòng ngoài, hàng ở vòng trong):

```
thứ tự xen kẽ:  (k=0: hàng 0,1,2,3) -> (k=1: hàng 0,1,2,3) -> ...
địa chỉ:        addr = base + hàng·window_step + k
```

Khi đó mọi lane đầy lên đều nhau, feeder có thể đọc ngay từ chu kỳ thứ R, và FIFO chỉ cần đủ để che độ trễ.

## 5. Bước 1: đủ cho ECG (vẫn nạp xong rồi mới tính)

### 1a. Hoàn tất L01: dùng `ring_buffer` trong `feed_lane`

`feed_lane.v` đã đổi sang `ring_buffer` và bỏ `CNT_W`. Tuy nhiên các tầng phía trên vẫn truyền `CNT_W` xuống, nên **hiện các testbench có dùng `feed_lane` không load được design** (`vsim-3584: Module parameter 'CNT_W' not found for override`). Cần bỏ `CNT_W` ở:

| File | Dòng |
| --- | --- |
| `feeder/ifmap_feeder.v` | 4 (khai báo), 117 (truyền xuống) |
| `feeder/weight_feeder.v` | 4, 118 |
| `conv1d/conv1d_ifmap_feeder_top.v` | 8, 97 |
| `conv1d/conv1d_weight_feeder_top.v` | 9, 120 |
| `conv1d/conv1d_tile_top.v` | 33, 300, 352 |
| `tb/tb_feed_lane.v`, `tb/tb_conv1d_tile_top.v`, `tb/tb_conv1d_weight_feeder_top.v` | 5/17, 18/119, 16/227 |

Xong khi `rtl/cnn1d/sim/run_tb.sh all` báo tất cả PASS (xem [sim-flow.md](../sim-flow.md)).

### 1b. Backpressure cho loader

Loader chỉ ghi khi lane đích còn chỗ; lane đầy thì đứng yên ở `S_WRITE`. Phác thảo cho `conv1d_ifmap_loader` (weight loader làm y hệt với `col_cnt_q`):

```verilog
// cổng mới
input wire [ARRAY_ROWS-1:0] i_lane_full,     // từ ifmap_feeder.o_mem_full

wire wr_ok = ~i_lane_full[row_cnt_q];

assign agu_advance = (state_q == S_WRITE) && wr_ok;   // chỉ tăng địa chỉ khi ghi được

always @(*) begin
	o_ifm_wr_en   = {ARRAY_ROWS{1'b0}};
	o_ifm_wr_data = {ARRAY_ROWS*DATA_W{1'b0}};
	if (state_q == S_WRITE && wr_ok) begin
		o_ifm_wr_en[row_cnt_q] = 1'b1;
		o_ifm_wr_data[row_cnt_q*DATA_W +: DATA_W] = rd_data_q;
	end
end

// trong FSM:
S_WRITE: begin
	if (wr_ok) begin
		// giữ nguyên logic cũ: k_last / window_last / row_cnt_q++
	end
	// không ghi được: đứng yên ở S_WRITE, rd_data_q giữ giá trị
end
```

Sau đó nối `feeder_mem_full` vào `i_lane_full` trong `conv1d_ifmap_feeder_top`, và nối `o_mem_full` của `weight_feeder` vào loader weight trong `conv1d_weight_feeder_top`.

`mem_full` của `ring_buffer` là tổ hợp từ hai con trỏ, nên chặn được ngay trong cùng chu kỳ. `ring_buffer` cũng tự chặn ghi khi đầy, nên đây là hai lớp bảo vệ khớp nhau.

**Lưu ý:** ở bước 1 tile vẫn nạp xong mới tính. Vì vậy nếu K > DEPTH, backpressure chỉ chuyển chỗ treo từ feeder sang loader: loader đứng chờ lane bớt đầy, mà feeder chưa được chạy. Backpressure là điều kiện cần cho bước 2, chưa phải lời giải cho K lớn. Ở bước 1, K lớn được giải quyết bằng 1c.

### 1c. FIFO sâu 1024 và chặn cấu hình sai

- Đặt `FEEDER_DEPTH = 1024` ở `conv1d_tile_top`. Trên MAX 10, mỗi lane chiếm trọn 1 M9K (1024 × 8) dù độ sâu là 256 hay 1024, nên không tốn thêm (xem L01, mục 8).
- Khi nhận `i_start`, nếu `i_k_len > FEEDER_DEPTH` thì **không chạy**: bật cờ `o_cfg_err` và chuyển thẳng sang DONE. Phần mềm hoặc sequencer đọc được cờ này, thay vì để accelerator treo.
- Trong mô phỏng, thêm một assert in `[FAIL]` khi gặp trường hợp này, để testbench bắt được ngay.

### 1d. Test với K lớn

Mở rộng `tb_conv1d_tile_top` (hoặc viết `tb_conv1d_tile_k`), theo mẫu của [`tb_ring_buffer.v`](../../rtl/cnn1d/tb/tb_ring_buffer.v):

- K = 7, 80, 96, 192 (đúng các lớp của ECG) và một giá trị K = DEPTH.
- Ifmap và weight ngẫu nhiên trong dải int8. Kết quả kỳ vọng tính ngay trong testbench: `C[r][c] = Σ_k X[base + r·step + k] · W[c·K + k]`.
- Thử các tile thiếu hàng/cột (`active_rows`, `active_cols` < kích thước array).
- Test âm: K = DEPTH + 1 phải bật `o_cfg_err` và trả `done`, không được treo.

**Bước 1 xong khi:** regression PASS, có test K = 192 PASS, và test K = DEPTH + 1 trả `o_cfg_err`.

## 6. Bước 2: stream thật (làm cùng L04)

| # | Việc | Ghi chú |
| --- | --- | --- |
| 2a | AGU sinh địa chỉ theo thứ tự xen kẽ: `for k { for hàng { addr = base + hàng·step + k } }` | Weight cũng vậy, với cột thay cho hàng. Nên làm luôn thành AGU dạng descriptor (L06): `count[d]`, `stride[d]` cho từng chiều |
| 2b | `conv1d_tile_top`: khởi động core ngay sau khi loader bắt đầu, bỏ trạng thái chờ `S_LOAD_WAIT`. Tile chỉ xong khi cả loader lẫn core đều xong | Cờ first/last vẫn do bộ đếm k của feeder sinh ra, không phụ thuộc loader |
| 2c | Hạ `FEEDER_DEPTH` xuống 32–64, bỏ giới hạn K ≤ DEPTH | FIFO nhỏ có thể dùng MLAB hoặc thanh ghi, để dành M9K cho scratchpad |
| 2d | Test: K = 192 với DEPTH = 32 phải PASS | Đây là test chứng minh không còn deadlock |

**Về tốc độ:** bước 2 chỉ nhanh hơn khoảng 8%. Loader vẫn tốn khoảng 3 chu kỳ cho mỗi phần tử, nên lượng dữ liệu cấp vào vẫn là nút thắt; chạy chồng chỉ tiết kiệm được K + ~17 chu kỳ mỗi tile. Lợi ích thật là K không còn bị giới hạn và FIFO nhỏ lại. Muốn nhanh thật thì cần L04 (loader pipeline 1 phần tử/chu kỳ, đọc SRAM 64 bit), và L04 dựa trên chính thứ tự xen kẽ của 2a.

## 7. Danh sách kiểm chứng

- [ ] 1a: bỏ `CNT_W` ở mọi tầng; `run_tb.sh all` PASS.
- [ ] 1b: loader ifmap và weight có `i_lane_full`; ghi chỉ xảy ra khi lane chưa đầy; AGU chỉ tăng khi ghi được.
- [ ] 1c: `FEEDER_DEPTH = 1024`; K > DEPTH bật `o_cfg_err`, không treo.
- [ ] 1d: test K = 7, 80, 96, 192 và K = DEPTH PASS với dữ liệu ngẫu nhiên; test âm K = DEPTH + 1 PASS.
- [ ] Tổng hợp: FIFO mỗi lane dùng 1 M9K.
- [ ] (Bước 2) K = 192 với DEPTH = 32 PASS, không deadlock.
