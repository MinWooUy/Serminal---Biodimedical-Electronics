# Giới hạn của accelerator Conv1D và phương án đề xuất

Cập nhật: 17/10/2026. Phạm vi: `rtl/cnn1d/` (lõi copy từ `AI_Chip`).

Lõi tính toán (PE, systolic array, PSM) dùng lại được. Các giới hạn chủ yếu nằm ở ba lớp bao quanh lõi:

- **Đường cấp dữ liệu:** FIFO, loader, băng thông bộ nhớ.
- **Sinh địa chỉ:** AGU hiện chỉ đúng cho 1D.
- **Điều khiển:** cấu hình qua cổng top-level, chưa có descriptor hay MMIO.

Mục tiêu lâu dài của thiết kế là mở rộng lên CNN2D, tái cấu hình được và dùng lại làm NPU. Vì vậy mỗi phương án dưới đây phải giải quyết được bài ECG trước mắt, đồng thời không khóa đường mở rộng về sau.

Ký hiệu ưu tiên:

- **P0**: chặn bài ECG, làm trong 3 tuần.
- **P1**: làm trong 3 tuần nếu viết theo hướng tổng quát ngay từ đầu.
- **P2**: để sau ECG, phục vụ định hướng NPU.

## Tổng hợp

| ID | Giới hạn | Nhóm | Ưu tiên | Trạng thái |
| --- | --- | --- | --- | --- |
| L01 | FIFO kiểu thanh ghi dịch, không đọc và ghi cùng chu kỳ | Đường dữ liệu | P0 | Mở |
| L02 | Loader không kiểm tra cờ full; K bị giới hạn bởi độ sâu FIFO | Đường dữ liệu | P0 | Mở |
| L03 | Kết quả tile bị xóa ở tile sau; chưa có khối hậu xử lý | Đường dữ liệu | P0 | Mở |
| L04 | Băng thông bộ nhớ 1 byte/chu kỳ; load và compute chạy nối tiếp | Hiệu năng | P2 | Mở |
| L05 | Không tái dùng dữ liệu giữa các tile | Hiệu năng | P2 | Mở |
| L06 | AGU chỉ đúng cho 1D; chưa có padding, dilation, depthwise | Tổng quát hóa | P1 | Mở |
| L07 | Chỉ có phép GEMM; chưa có vector unit | Tổng quát hóa | P0/P1 | Mở |
| L08 | Tham số độ rộng dữ liệu khai báo nhưng không được dùng thật | Tổng quát hóa | P1 | Mở |
| L09 | Cấu hình qua cổng; chưa có MMIO hay descriptor | Điều khiển | P2 | Mở |
| L10 | Interface tự đặt, không theo chuẩn valid/ready | Điều khiển | P1/P2 | Mở |
| L11 | Multiplier Wallace dựng bằng LE, không dùng DSP | Hiện thực | P1 | Mở |
| L12 | Kiểu reset lẫn lộn; reset trên bộ nhớ cản việc map vào RAM | Hiện thực | P1 | Mở |
| L13 | Bus kết quả và cổng top-level quá rộng | Hiện thực | P2 | Mở |
| L14 | Testbench chọn tay, kích thước nhỏ, chưa so với golden | Kiểm chứng | P1 | Mở |

## Chi tiết

### L01. FIFO kiểu thanh ghi dịch

- **Bằng chứng:** [FIFO_param.v:23-40](../../rtl/cnn1d/feeder/FIFO_param.v)
  - Mỗi lần đọc, toàn bộ mảng dịch đi một ô (`mem[i] <= mem[i+1]`).
  - Khi reset, mọi ô `mem` bị xóa về 0.
  - Nhánh ghi được ưu tiên hơn nhánh đọc (`if (wr_en) … else if (rd_en)`).
- **Hậu quả:**
  - Không map được vào M9K. Với độ sâu 256 và 8 lane, FIFO tốn khoảng 16k flip-flop cộng thêm mux.
  - Ring buffer không tăng lượng dữ liệu mỗi chu kỳ (mỗi cổng vẫn 1 đọc, 1 ghi). Nó chỉ cho đọc và ghi chạy đồng thời. Muốn tăng thông lượng thì xem L04.
  - Không đọc và ghi được trong cùng một chu kỳ, nên không stream được dữ liệu.
- **Đề xuất:**
  - Viết lại thành FIFO vòng tròn: con trỏ đọc/ghi, RAM 2 cổng, đọc và ghi cùng chu kỳ.
  - Interface valid/ready. Không reset mảng nhớ, chỉ reset con trỏ.
  - Giữ nguyên port của `feed_lane` để không phải sửa các khối phía trên.
  - FIFO này dùng lại được cho mọi khối sau này.
  - Giải thích chi tiết và RTL đề xuất: [L01-ring-buffer.md](L01-ring-buffer.md).

### L02. Loader không kiểm tra cờ full, K bị giới hạn

- **Bằng chứng:**
  - [conv1d_ifmap_loader.v:148](../../rtl/cnn1d/conv1d/conv1d_ifmap_loader.v) và [conv1d_weight_loader.v:267](../../rtl/cnn1d/conv1d/conv1d_weight_loader.v): trạng thái `S_WRITE` ghi vào FIFO mà không xét cờ full.
  - [conv1d_tile_top.v:497](../../rtl/cnn1d/conv1d/conv1d_tile_top.v): tile phải load xong (`S_LOAD_WAIT`) mới bắt đầu compute.
- **Hậu quả:**
  - Mỗi lần nạp một tile, cần K ≤ `FEEDER_DEPTH` (hiện là 16).
  - Model ECG cần K tới 192. Khi K vượt độ sâu FIFO, các lần ghi thừa bị bỏ và **tile treo ở COMPUTE**, không có cờ lỗi nào. Đã xác nhận bằng mô phỏng ngày 17/10 với K = 20, DEPTH = 16.
  - Các testbench hiện tại chỉ thử K = 3.
- **Đề xuất:**
  - Loader phải dừng khi FIFO full (backpressure). Sau khi có L01, cho compute bắt đầu ngay khi FIFO có dữ liệu, không chờ load xong.
  - Trước mắt, khi vẫn nạp hết rồi mới tính, FIFO phải chứa trọn K (K lớn nhất của ECG là 192). Đặt `FEEDER_DEPTH = 1024`: trên MAX 10, mỗi lane chiếm trọn 1 M9K (1024 × 8) bất kể độ sâu là 256 hay 1024, nên không tốn thêm. Thêm assert `K ≤ DEPTH`.
  - Muốn stream với FIFO nhỏ hơn K thì phải **đổi thứ tự nạp của loader**. Hiện loader nạp hết K phần tử của hàng 0 rồi mới sang hàng 1, trong khi feeder chỉ đọc khi mọi lane đều có dữ liệu. Nếu DEPTH < K: lane 0 đầy, loader dừng chờ, lane 1 không bao giờ có dữ liệu, và hệ thống kẹt cứng (deadlock). Cần nạp xen kẽ: k ở vòng ngoài, hàng ở vòng trong. PE đã chịu được chu kỳ trống, vì `compute_en = act_valid & wgt_valid` và cờ valid đi theo dữ liệu qua từng hop ([pe_top.v](../../rtl/cnn1d/pe/pe_top.v)).
  - Về lâu dài, độ sâu không nên phụ thuộc K, vì CNN2D có K = kh·kw·Cin (ví dụ 3×3×256 = 2304). Khi đã stream có backpressure, FIFO chỉ cần che độ trễ của loader, độ lệch giữa các hàng (ROWS − 1) và khoảng 2 burst. Hạ `FEEDER_DEPTH` xuống 32–64 và để M9K dành cho scratchpad.
  - Viết test tile với K = 7, 80, 96 và 192.
  - Giải thích chi tiết và hướng dẫn 2 bước: [L02-loader-backpressure.md](L02-loader-backpressure.md).

### L03. Kết quả tile bị xóa, chưa có hậu xử lý

- **Bằng chứng:**
  - [partial_sum_manager.v:94](../../rtl/cnn1d/core/partial_sum_manager.v): `i_clr` của tile sau xóa `o_tile`.
  - [conv1d_tile_top.v:85](../../rtl/cnn1d/conv1d/conv1d_tile_top.v): kết quả chỉ được đưa ra cổng, không ghi về SRAM.
- **Hậu quả:** không nối được nhiều lớp với nhau.
- **Đề xuất:**
  - **PSM 2 chiều** trên chuỗi shadow scan có sẵn:
    - *Chiều ghi (preload):* trước khi tính, bơm bias vào cột 0. MAC khởi tạo accumulator từ shadow thay cho hằng số 0, nên bias được cộng miễn phí. Bơm 0 thì giữ đúng hành vi cũ.
    - *Chiều đọc (drain):* dịch shadow ra thành stream có valid/ready, mỗi chu kỳ một kênh × ROWS vị trí. Shadow chính là thanh ghi chốt, nên không cần chốt 512 bit và bỏ được bus `o_tile` (một phần L13).
    - Cùng cổng ghi này về sau sẽ nạp lại tổng riêng phần khi chia K (cần cho CNN2D).
  - `post_proc` là vector unit SIMD ROWS làn (requant → clamp → pool → GAP), không còn tầng bias. Thêm `output_writer` ghi vào bank đang ghi của ping-pong, rồi swap sau mỗi lớp.
  - MaxPool cần ARRAY_ROWS chẵn (4×4). Khi đó cặp pool nằm sẵn trong một vector drain. Với 3 hàng, cặp pool vắt qua tile, mà layer controller chạy hết mọi Cout rồi mới sang nhóm vị trí sau.
  - Giải thích chi tiết, interface và hướng dẫn 6 bước (bắt đầu bằng preload bias = 0 để chứng minh không đổi hành vi): [L03-post-processing.md](L03-post-processing.md).

### L04. Băng thông bộ nhớ thấp, load và compute nối tiếp

- **Bằng chứng:**
  - `tensor_sram` chỉ có cổng rộng 1 byte. Loader tốn khoảng 3 chu kỳ/phần tử (S_REQ → S_WAIT → S_WRITE).
  - Đo trên mô phỏng ngày 17/10: một tile tốn khoảng 10·K + 20 chu kỳ.
  - `BUS_ELEMENTS` trong `cnn_pkg` của `AI_Chip` có khai báo nhưng không được dùng.
- **Hậu quả:**
  - PE làm việc khoảng 10% thời gian.
  - Array cần (rows + cols) byte mỗi chu kỳ, nên tăng lên 8×8 hay 16×16 gần như không nhanh thêm.
  - Với ECG vẫn đủ: khoảng 27 ms/beat ở 50 MHz.
- **Đề xuất (P2):**
  - Scratchpad nhiều bank, rộng (rows + cols) byte/chu kỳ.
  - Loader dạng pipeline, mỗi chu kỳ đưa ra 1 phần tử cho mỗi lane.
  - DMA kèm double buffer để nạp tile sau trong khi tile hiện tại đang tính.

### L05. Không tái dùng dữ liệu giữa các tile

- **Bằng chứng:** [conv1d_layer_controller.v](../../rtl/cnn1d/conv1d/conv1d_layer_controller.v)
  - Mỗi tile nạp lại cả ifmap lẫn weight.
  - Ifmap bị đọc lại ceil(Cout / COLS) lần; weight bị đọc lại ceil(L / ROWS) lần.
  - Các cửa sổ chồng nhau (khi k > stride) cũng bị đọc lại.
- **Hậu quả:** lưu lượng bộ nhớ gấp nhiều lần mức cần. Với CNN2D (kernel 3×3, nhiều kênh), mức lãng phí còn tăng mạnh.
- **Đề xuất (P2):**
  - Chọn dataflow có giữ dữ liệu: weight-stationary hoặc giữ ifmap trong scratchpad trong suốt một nhóm Cout.
  - Dùng line buffer hoặc window generator để tái dùng phần chồng giữa các cửa sổ (spec v1 đã có khối này).

### L06. AGU chỉ đúng cho 1D

- **Bằng chứng:** [conv1d_address_generating.v:39-44](../../rtl/cnn1d/conv1d/conv1d_address_generating.v)
  - Công thức `addr = base + out_pos·stride·Cin + k`.
  - Công thức này giả định cửa sổ K = k·Cin byte nằm liền nhau trong bộ nhớ. Điều đó chỉ đúng với conv 1D lưu theo channel-last.
- **Hậu quả:**
  - Với 2D, một cửa sổ gồm ky đoạn rời nhau, mỗi đoạn dài kx·Cin byte.
  - Chưa có padding, dilation, depthwise hay grouped conv.
- **Đề xuất (P1):**
  - Viết AGU dạng descriptor: `base`, cùng `count[d]` và `stride[d]` cho 3–4 vòng lặp lồng nhau (cách làm của Gemmini và TVM VTA).
  - Dùng trước cho `output_writer` của bài ECG (ghi có padding), sau đó thay AGU của loader.
  - Với conv 1D hiện tại, descriptor chỉ cần 2 chiều. Với CNN2D, chỉ cần thêm chiều, không phải viết lại AGU.

### L07. Chỉ có phép GEMM, chưa có vector unit

- **Bằng chứng:** cả `rtl/cnn1d/` chưa có khối nào làm bias, requant, activation, pool hay cộng element-wise.
- **Hậu quả:**
  - ECG không chạy được: cần bias, phép dịch theo kênh, clamp, MaxPool, GAP.
  - Mạng có nhánh residual (ResNet, MobileNet) cũng không chạy được.
- **Đề xuất:**
  - Viết `post_proc` thành **vector unit** có interface stream, tách khỏi systolic array. Các bước xử lý theo thứ tự:
    1. Cộng bias INT32.
    2. Requant `(acc + 2^(s−1)) >>> s`. Để sẵn chỗ cho dạng M0 × shift (lượng tử kiểu TFLite).
    3. Clamp/ReLU.
    4. Pool (max hoặc avg).
    5. Cộng dồn GAP.
    6. Về sau thêm cộng element-wise.
  - Mỗi bước bật/tắt bằng cờ trong descriptor.

### L08. Tham số độ rộng không được dùng thật

- **Bằng chứng:**
  - [ProcessingElement.v:2-3](../../rtl/cnn1d/pe/ProcessingElement.v) và [pe_top.v:2-3](../../rtl/cnn1d/pe/pe_top.v) khai báo `DATA_W` và `ACC_W`.
  - Nhưng [MAC.v:5-7](../../rtl/cnn1d/pe/MAC.v), [Accumulator.v:5](../../rtl/cnn1d/pe/Accumulator.v), [shadow_register.v:6-7](../../rtl/cnn1d/pe/shadow_register.v), `WeightRegister.v` và `ImageRegister.v` đều cứng 8/32 bit.
  - `cnn_pkg` trong `AI_Chip` ghi `ACC_W = 16`, mâu thuẫn với 32 bit đang dùng.
- **Hậu quả:**
  - Đổi sang INT4 hay INT16 phải sửa tay nhiều file.
  - Chưa hỗ trợ zero-point (uint8 bất đối xứng).
- **Đề xuất (P1):**
  - Truyền `DATA_W` và `ACC_W` xuống tới tận MAC.
  - Gom mọi tham số vào một package duy nhất và bỏ `cnn_pkg` cũ.
  - Với zero-point: thêm bước trừ zero-point ở đầu vào PE, hoặc bù sẵn vào bias (cách làm của TFLite).

### L09. Cấu hình qua cổng, chưa có MMIO hay descriptor

- **Bằng chứng:**
  - Cấu hình (`i_k_len`, `i_window_step`, `i_active_rows`, `i_out_len`, …) đi thẳng vào cổng của `conv1d_tile_top` và `conv1d_layer_controller`.
  - Có 3 FSM lồng nhau (layer, tile, core) và handshake ở mỗi tile.
- **Hậu quả:** phần mềm hay compiler không có cách lập trình accelerator; mỗi model mới đòi sửa RTL.
- **Đề xuất:**
  - Bài ECG: ROM cấu hình 4 lớp cùng `net_sequencer`. Định dạng mỗi bản ghi giống một descriptor.
  - P2: thêm hàng đợi lệnh `LOAD`, `CONV`, `VEC`, `STORE` kèm register map AXI-Lite/MMIO cho RISC-V (đúng hướng của spec v1).

### L10. Interface tự đặt

- **Bằng chứng:** mỗi khối dùng tên tín hiệu và giao thức riêng (`i_feed_step`, `o_ready`, `i_start_tile`, …). Không có chuẩn valid/ready thống nhất.
- **Hậu quả:** khó thay từng khối và khó gắn vào SoC.
- **Đề xuất:**
  - Các khối mới (FIFO, vector unit, writer) dùng valid/ready kiểu AXI-Stream ngay từ đầu.
  - P2: thêm AXI-Lite slave cho register map và AXI master cho DMA.

### L11. Multiplier Wallace dựng bằng LE

- **Bằng chứng:**
  - [MAC.v:19](../../rtl/cnn1d/pe/MAC.v) dùng `wallace_tree_multiplier`.
  - Tổng hợp trên Cyclone IV (03/10/2026): 4.767 LE cho 9 PE, 0/132 DSP.
- **Hậu quả:** khoảng 530 LE/PE. Array 16×16 không vừa các FPGA cỡ nhỏ như DE10-Lite.
- **Đề xuất (P1):**
  - Bọc multiplier trong một wrapper có tham số `MUL_IMPL`: `"dsp"` dùng toán tử `*`, `"wallace"` dùng bản custom (cho ASIC hoặc nghiên cứu).
  - Thư mục `pe/multiplier/` của `AI_Chip` đã có sẵn khung `multiplier_generic`, có thể dùng lại.

### L12. Kiểu reset lẫn lộn

- **Bằng chứng:**
  - Trong `rtl/cnn1d/` có 14 khối reset bất đồng bộ và 22 khối reset đồng bộ. `sa_processing_element` của `AI_Chip` dùng reset mức thấp.
  - [tensor_sram.v:48](../../rtl/cnn1d/mem/tensor_sram.v) reset thanh ghi đọc; `FIFO_param` reset cả mảng nhớ.
- **Hậu quả:** cản Quartus map vào RAM, khó port sang ASIC, dễ lỗi khi ghép các khối.
- **Đề xuất (P1):**
  - Thống nhất một kiểu reset (ví dụ async-assert / sync-deassert, mức cao) cho toàn bộ thiết kế.
  - Không reset mảng nhớ.
  - Kiểm tra trong báo cáo fitter rằng bộ nhớ đã được map vào M9K.

### L13. Bus và cổng quá rộng

- **Bằng chứng:**
  - `o_tile` rộng ROWS × COLS × 32 bit: 288 bit với 3×3; sẽ là 8.192 bit với 16×16.
  - `cnn_accelerator_top` cũ kéo ra 369 chân, phần lớn để debug.
- **Hậu quả:** không mở rộng được, khó route.
- **Đề xuất (P2):**
  - Đưa kết quả ra từng hàng hoặc từng cột qua stream rộng COLS × 32 bit.
  - Đưa debug vào register map và bộ đếm, thay vì kéo ra chân.

### L14. Kiểm chứng còn mỏng

- **Bằng chứng:**
  - Testbench là các trường hợp chọn tay, kích thước nhỏ (K = 3, array 3×3).
  - 4 testbench trong [tb/stale/](../../rtl/cnn1d/tb/stale/) đã lệch interface hiện tại.
  - Chưa có test ngẫu nhiên, chưa quét tham số, chưa có CI.
- **Hậu quả:** đổi tham số (sang 4×4, đổi độ sâu FIFO) rồi không biết thiết kế có vỡ chỗ nào không.
- **Đề xuất (P1):**
  - Viết testbench sinh vector ngẫu nhiên, so với golden Python (`python/quant/int8.py`) qua file `.hex`.
  - Quét tham số: ARRAY 3×3 và 4×4; K từ 1 đến 192.
  - Sửa hoặc xóa các testbench trong `tb/stale/`.
  - `sim/run_tb.sh all` phải chạy sạch trước mỗi lần commit.

## Lộ trình

**Trong 3 tuần (ECG, viết theo hướng tái dùng được):**

1. L01 + L02: FIFO vòng tròn, loader có backpressure, test K lớn.
2. L07 + L03: vector unit `post_proc` có interface stream, cộng với `output_writer`.
3. L06: AGU dạng descriptor, dùng trước cho writer.
4. L08 + L11 + L12: truyền tham số thật, wrapper multiplier, thống nhất kiểu reset.
5. L14: testbench so với golden cho mọi khối mới.

**Sau ECG (định hướng NPU):**

1. L04 + L05: scratchpad nhiều bank, DMA, double buffer, dataflow có giữ dữ liệu.
2. L09 + L10: hàng đợi lệnh, MMIO/AXI-Lite, gắn RISC-V.
3. L06 mở rộng: CNN2D, padding, depthwise.
4. L13: bus kết quả dạng stream; debug qua bộ đếm và register map.

**Không làm trong 3 tuần:** AXI đầy đủ, compiler, CNN2D thật. Các interface ở trên chỉ cần thiết kế sao cho sau này cắm thêm được.

## Tham khảo

- Gemmini: <https://github.com/ucb-bar/gemmini> (scratchpad, DMA, systolic array, ISA dạng lệnh).
- TVM VTA: <https://github.com/apache/tvm-vta> (descriptor, hàng đợi lệnh, runtime).
- FINN: <https://github.com/Xilinx/finn> (dataflow, lượng tử hóa, đóng gói weight).
- Đặc tả hệ thống v2 (Claude Docs) và `docs/spec.pdf` (spec v1).
