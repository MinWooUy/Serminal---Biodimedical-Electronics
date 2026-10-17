# Ring buffer (FIFO vòng tròn)

Ghi chú cho giới hạn L01 trong [hw-accelerator-limitations.md](hw-accelerator-limitations.md). Cập nhật: 17/10/2026.

Ring buffer là một FIFO mà **dữ liệu nằm yên trong RAM, chỉ có hai con trỏ chạy vòng quanh**:

- con trỏ ghi chỉ tới ô sẽ được ghi tiếp theo;
- con trỏ đọc chỉ tới ô sẽ được đọc tiếp theo.

Không có phép dịch dữ liệu nào. Nhờ vậy ring buffer map được vào RAM (M9K) và đọc/ghi được trong cùng một chu kỳ. Đây là cách sửa cho `FIFO_param` hiện tại.

## 1. Vì sao không dùng FIFO dịch

`FIFO_param` hiện tại ([FIFO_param.v:23-40](../../rtl/cnn1d/feeder/FIFO_param.v)) luôn đọc ở `mem[0]`. Mỗi lần đọc, nó dịch cả mảng lên một ô:

```
trước khi đọc:  mem = [A][B][C][D]   rd_data <= A
sau khi đọc:    mem = [B][C][D][ ]   (mọi ô đều được ghi lại)
```

| | FIFO dịch (`FIFO_param`) | Ring buffer |
|---|---|---|
| Mỗi lần đọc | Ghi lại **toàn bộ** DEPTH ô | Chỉ tăng con trỏ đọc |
| Hiện thực | Flip-flop + mux, khoảng DEPTH × DATA_W FF | 1 khối RAM (M9K) |
| Đọc và ghi cùng chu kỳ | Không: nhánh ghi được ưu tiên, nhánh đọc bị bỏ qua | Có, hai con trỏ chạy độc lập |
| Tài nguyên với DEPTH = 256, 8 lane | khoảng 16k FF | 8 M9K |
| Reset | Xóa mọi ô nhớ | Chỉ reset con trỏ |

## 2. Cấu trúc

```
            wr_ptr (ghi tiếp vào đây)
               |
    +----+----+----+----+----+----+----+----+
    |    | B  | C  | D  |    |    |    |    |   mem[0..7]
    +----+----+----+----+----+----+----+----+
          |
        rd_ptr (đọc tiếp ở đây)

    Con trỏ tới ô cuối thì quay về ô 0 (vòng tròn).
```

- **RAM 2 cổng (simple dual-port):** một cổng ghi tại `wr_ptr`, một cổng đọc tại `rd_ptr`.
- **DEPTH là lũy thừa của 2.** Khi đó con trỏ tự quay vòng chỉ bằng phép cộng, bỏ bit tràn, không cần so sánh rồi đặt lại về 0.
- **Ghi** khi `wr_en` và FIFO chưa đầy: `mem[wr_ptr] <= data`, rồi `wr_ptr++`.
- **Đọc** khi `rd_en` và FIFO chưa rỗng: `rd_data <= mem[rd_ptr]`, rồi `rd_ptr++`.

## 3. Phân biệt đầy và rỗng: thêm 1 bit vào con trỏ

Nếu con trỏ chỉ có log2(DEPTH) bit, thì `wr_ptr == rd_ptr` vừa có thể là rỗng vừa có thể là đầy (con trỏ ghi đã chạy trọn một vòng). Cách chuẩn để tách hai trường hợp: **cho con trỏ rộng thêm 1 bit**. Bit cao nhất đánh dấu con trỏ đang ở vòng chẵn hay vòng lẻ.

- **Rỗng:** hai con trỏ bằng nhau ở mọi bit.
- **Đầy:** khác nhau ở bit cao nhất, bằng nhau ở các bit còn lại (con trỏ ghi đi trước con trỏ đọc đúng một vòng).
- **Số phần tử:** `count = wr_ptr − rd_ptr` (phép trừ modulo, rộng log2(DEPTH) + 1 bit).

Ví dụ với DEPTH = 4. Mỗi con trỏ rộng 3 bit; 2 bit thấp là địa chỉ, bit cao là vòng.

| Bước | Thao tác | wr_ptr | rd_ptr | count | Trạng thái |
| --- | --- | --- | --- | --- | --- |
| 0 | Reset | 000 | 000 | 0 | rỗng |
| 1 | Ghi A, B, C, D | 100 | 000 | 4 | **đầy** (2 bit thấp bằng nhau, bit cao khác nhau) |
| 2 | Đọc A | 100 | 001 | 3 | |
| 3 | Ghi E (vào ô 00, quay vòng) và đọc B **trong cùng chu kỳ** | 101 | 010 | 3 | count không đổi |
| 4 | Ghi F (vào ô 01) | 110 | 010 | 4 | **đầy** |
| 5 | Ghi G | 110 | 010 | 4 | bị chặn vì đang đầy |
| 6 | Đọc C, D, E, F | 110 | 110 | 0 | **rỗng** (bằng nhau ở mọi bit) |

Khi FIFO đầy, RTL ở mục 7 chặn lệnh ghi kể cả khi trong cùng chu kỳ có lệnh đọc. Cách này có thể mất một chu kỳ ghi, nhưng tránh được việc ghi đè lên ô đang được đọc ra.

## 4. Đọc và ghi cùng một chu kỳ

Hai con trỏ được cập nhật độc lập, nên trong một chu kỳ có 4 khả năng:

| wr_en | rd_en | Kết quả |
| --- | --- | --- |
| 0 | 0 | Giữ nguyên |
| 1 | 0 | count + 1 (nếu chưa đầy) |
| 0 | 1 | count − 1 (nếu chưa rỗng) |
| 1 | 1 | Ghi và đọc ở hai ô khác nhau, count không đổi |

Khả năng cuối là thứ FIFO dịch không làm được, và là điều kiện để **stream**: loader đẩy vào 1 phần tử/chu kỳ, PE lấy ra 1 phần tử/chu kỳ, cả hai chạy liên tục.

Trường hợp đọc và ghi cùng một ô trong một chu kỳ chỉ có thể xảy ra khi FIFO đang rỗng. Khi rỗng thì lệnh đọc bị chặn, nên trường hợp này không phát sinh. Dù vậy vẫn nên đặt thuộc tính read-during-write của RAM là `"OLD_DATA"` hoặc `"DONT_CARE"` trong Quartus cho chắc.

## 5. Độ trễ đọc: chuẩn hay first-word fall-through

M9K có cổng đọc đồng bộ: đưa địa chỉ ở chu kỳ n thì có dữ liệu ở chu kỳ n + 1.

| Kiểu | Hành vi | Ghi chú |
| --- | --- | --- |
| **Chuẩn** | `rd_en` ở chu kỳ n, `rd_data` có ở chu kỳ n + 1 | Giống `FIFO_param` hiện tại (`rd_data <= mem[0]`), nên thay thẳng vào `feed_lane` được |
| First-word fall-through (FWFT) | `rd_data` luôn sẵn phần tử đầu tiên; `rd_en` chỉ để "lấy đi" | Cần thêm thanh ghi prefetch. Hợp với interface valid/ready |

Với L01, dùng kiểu **chuẩn** để không phải sửa `feed_lane`. Khi chuyển các khối sang valid/ready (L10), bọc thêm một tầng FWFT ở ngoài.

## 6. Map vào M9K

Để Quartus suy ra M9K thay vì flip-flop:

- **Không reset mảng nhớ.** Chỉ reset `wr_ptr` và `rd_ptr`.
- Ghi và đọc mảng trong **một khối `always @(posedge clk)` không có reset**.
- Thanh ghi đầu ra `rd_data` cũng không reset (hoặc chỉ dùng clear bất đồng bộ, M9K có hỗ trợ).
- Một cổng ghi và một cổng đọc, cùng một clock.
- Xem trong báo cáo fitter mục "RAM Summary" để chắc FIFO đã nằm trong M9K.

Mỗi khối M9K chứa 9 kbit (1024 × 8 hoặc 1024 × 9). Một lane chiếm trọn ít nhất 1 khối, nên trên MAX 10 độ sâu 256 hay 1024 tốn số M9K như nhau.

## 7. RTL đề xuất

Bản phác dưới đây chưa được biên dịch hay mô phỏng. Port giữ giống `FIFO_param` (`wr_en`, `rd_en`, `wr_data`, `rd_data`, `mem_full`, `mem_empty`) và có thêm `count`.

```verilog
module ring_fifo #(
	parameter DATA_W = 8,
	parameter DEPTH  = 1024,                // lũy thừa của 2
	parameter AW     = $clog2(DEPTH)
)(
	input  wire              clk, reset,
	input  wire              wr_en, rd_en,
	input  wire [DATA_W-1:0] wr_data,
	output reg  [DATA_W-1:0] rd_data,
	output wire              mem_full, mem_empty,
	output wire [AW:0]       count
);
	reg [DATA_W-1:0] mem [0:DEPTH-1];       // không reset -> map vào M9K
	reg [AW:0] wr_ptr, rd_ptr;              // thêm 1 bit đánh dấu vòng

	wire do_wr = wr_en & ~mem_full;         // đầy thì chặn ghi, kể cả khi đang đọc
	wire do_rd = rd_en & ~mem_empty;

	assign count     = wr_ptr - rd_ptr;
	assign mem_empty = (wr_ptr == rd_ptr);
	assign mem_full  = (wr_ptr[AW] != rd_ptr[AW]) && (wr_ptr[AW-1:0] == rd_ptr[AW-1:0]);

	always @(posedge clk) begin             // đường dữ liệu: không reset
		if (do_wr) mem[wr_ptr[AW-1:0]] <= wr_data;
		if (do_rd) rd_data <= mem[rd_ptr[AW-1:0]];
	end

	always @(posedge clk or posedge reset) begin
		if (reset) begin
			wr_ptr <= 0;
			rd_ptr <= 0;
		end else begin
			if (do_wr) wr_ptr <= wr_ptr + 1'b1; // hai con trỏ độc lập
			if (do_rd) rd_ptr <= rd_ptr + 1'b1;
		end
	end
endmodule
```

Cho loader (L02): dùng `count` hoặc thêm `almost_full = (count >= DEPTH − N)`. N là số lệnh ghi có thể còn đang trên đường đi trong pipeline của loader, để loader dừng kịp trước khi FIFO đầy.

## 8. Chọn độ sâu

Độ sâu phụ thuộc vào cách chạy:

| Cách chạy | Yêu cầu | Độ sâu |
| --- | --- | --- |
| Nạp hết rồi mới tính (hiện tại) | Mỗi lane chứa trọn K (ECG: K lớn nhất là 192) | 256 là đủ; **1024** tốn cùng số M9K và cho K tới 1024 |
| Stream có backpressure (sau khi sửa L02) | Chỉ cần che độ trễ: SRAM và pipeline loader (vài chu kỳ), độ lệch giữa các hàng (ROWS − 1), khoảng 2 burst | **32–64**, K không còn bị giới hạn |

Với CNN2D, K = kh·kw·Cin (ví dụ 3×3×256 = 2304), nên kiểu "nạp hết rồi mới tính" không mở rộng được. Stream là hướng bắt buộc.

## 9. Ring buffer không tăng thông lượng

Mỗi cổng RAM vẫn chỉ 1 lần đọc và 1 lần ghi mỗi chu kỳ. Ring buffer chỉ giúp hai phía chạy **đồng thời** ở tốc độ đó. Muốn có nhiều dữ liệu hơn mỗi chu kỳ (L04), cần thêm một trong các cách sau:

- **Ghi rộng, đọc hẹp:** M9K hỗ trợ hai cổng có độ rộng khác nhau. Ví dụ ghi 64 bit (8 phần tử int8), đọc 8 bit. Khi đó `wr_ptr` tăng 8 mỗi lần ghi, và điều kiện đầy đổi thành "còn ít hơn 8 ô trống".
- **Loader pipeline:** phát một lệnh đọc mỗi chu kỳ, thay cho vòng REQ → WAIT → WRITE hiện tại.
- **Chia bank:** nhiều RAM đặt song song, mỗi lane hoặc nhóm lane đọc một bank riêng.

Ví dụ với array 4×4: loader đọc 64 bit (8 phần tử k liên tiếp của một hàng) mỗi chu kỳ và xoay vòng giữa 4 lane ifmap. Lượng cấp vào là 8 B/chu kỳ, trong khi 4 lane chỉ tiêu thụ 4 B/chu kỳ.

## 10. Ring buffer ở các tầng khác của NPU

Cùng một ý tưởng hai con trỏ quay vòng xuất hiện ở ba tầng:

| Tầng | Ô chứa gì | Ở đâu trong lộ trình |
| --- | --- | --- |
| FIFO feeder | 1 phần tử int8 | L01 (bây giờ) |
| Line buffer / window generator | 1 hàng của feature map; cửa sổ trượt bằng cách xoay con trỏ hàng thay vì chép lại | L05, CNN2D |
| Hàng đợi lệnh (giống ring của NIC) | 1 descriptor: `LOAD`/`CONV`/`VEC`/`STORE`, địa chỉ, kích thước. CPU ghi đầu ring, accelerator đọc cuối ring | L09, khi gắn RISC-V (cách làm của Gemmini, TVM VTA) |

Ở tầng thứ ba, hai con trỏ nằm ở hai phía khác nhau (CPU và accelerator), nên cần thanh ghi head/tail qua MMIO và một ngắt báo xong. Ring buffer của NIC thường được minh họa theo đúng mô hình này.

## 11. Danh sách kiểm chứng

- [ ] Reset xong: `mem_empty = 1`, `mem_full = 0`, `count = 0`.
- [ ] Ghi đến đầy (DEPTH phần tử): `mem_full = 1`. Ghi thêm thì bị bỏ qua, không đè lên dữ liệu cũ.
- [ ] Đọc đến rỗng: dữ liệu ra đúng thứ tự. Đọc thêm thì bị bỏ qua.
- [ ] Đọc và ghi đồng thời liên tục trong ≥ 3 × DEPTH chu kỳ: dữ liệu đúng thứ tự, `count` ổn định, con trỏ quay qua nhiều vòng.
- [ ] Ghi và đọc ngẫu nhiên, so với một mô hình hàng đợi trong testbench (scoreboard).
- [ ] Thay vào `feed_lane`, chạy lại `sim/run_tb.sh all`: mọi testbench PASS.
- [ ] Tổng hợp: báo cáo fitter cho thấy mỗi lane dùng 1 M9K, không phải FF.
