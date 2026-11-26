// tb_ring_buffer.v
// Kiểm chứng ring_buffer bằng một mô hình hàng đợi (scoreboard) chạy song song với DUT.
// Mỗi chu kỳ, checker so count / mem_full / mem_empty với mô hình, và so rd_data
// ở chu kỳ ngay sau mỗi lần đọc được chấp nhận. Các test chỉ tạo kích thích.
//
// Chạy 2 cấu hình song song: DEPTH = 4 (quay vòng nhanh) và DEPTH = 16.
`timescale 1ns/1ps

module tb_ring_buffer;
	wire done4, done16;
	wire [31:0] err4, err16;

	tb_ring_buffer_unit #(.DEPTH(4),  .SEED(1)) u_d4  (.o_done(done4),  .o_errors(err4));
	tb_ring_buffer_unit #(.DEPTH(16), .SEED(7)) u_d16 (.o_done(done16), .o_errors(err16));

	initial begin
		wait(done4 && done16);
		#20;
		$display("==============================================");
		$display(" RING BUFFER TEST SUMMARY");
		$display("   DEPTH=4  : %0d errors", err4);
		$display("   DEPTH=16 : %0d errors", err16);
		if (err4 == 0 && err16 == 0)
			$display(" RESULT : ALL TESTS PASSED");
		else
			$display(" RESULT : FAILED");
		$display("==============================================");
		$finish;
	end

	initial begin
		#2000000;
		$display("[FAIL] TIMEOUT tb_ring_buffer");
		$finish;
	end
endmodule

module tb_ring_buffer_unit #(
	parameter DATA_W = 8,
	parameter DEPTH  = 16,
	parameter SEED   = 1
)(
	output reg        o_done,
	output reg [31:0] o_errors
);
	localparam AW = $clog2(DEPTH);

	reg clk = 1'b0;
	always #5 clk = ~clk;

	reg reset;
	reg wr_en, rd_en;
	reg [DATA_W-1:0] wr_data;

	wire [DATA_W-1:0] rd_data;
	wire mem_full, mem_empty;
	wire [AW:0] count;

	ring_buffer #(
		.DATA_W(DATA_W),
		.DEPTH(DEPTH)
	) dut (
		.clk(clk), .reset(reset),
		.wr_en(wr_en), .rd_en(rd_en),
		.wr_data(wr_data),
		.mem_full(mem_full), .mem_empty(mem_empty),
		.rd_data(rd_data),
		.count(count)
	);

	// =====================================================
	// Mô hình tham chiếu: hàng đợi vòng tròn viết bằng integer
	// =====================================================
	reg [DATA_W-1:0] q [0:DEPTH-1];
	integer q_head, q_tail, q_count;
	reg [DATA_W-1:0] exp_rd;
	reg exp_rd_valid;

	// Đầy thì chặn ghi kể cả khi đang đọc (đúng như đặc tả của DUT)
	wire m_wr = wr_en && (q_count < DEPTH);
	wire m_rd = rd_en && (q_count > 0);

	always @(posedge clk) begin
		if (reset) begin
			q_head <= 0;
			q_tail <= 0;
			q_count <= 0;
			exp_rd_valid <= 1'b0;
		end else begin
			exp_rd_valid <= m_rd;
			if (m_rd) begin
				exp_rd <= q[q_head];
				q_head <= (q_head + 1) % DEPTH;
			end
			if (m_wr) begin
				q[q_tail] <= wr_data;
				q_tail <= (q_tail + 1) % DEPTH;
			end
			q_count <= q_count + (m_wr ? 1 : 0) - (m_rd ? 1 : 0);
		end
	end

	// =====================================================
	// Checker: so DUT với mô hình ở mỗi sườn xuống
	// =====================================================
	integer errors;
	integer reads_checked;

	always @(negedge clk) begin
		if (!reset) begin
			if (count !== q_count[AW:0]) begin
				errors = errors + 1;
				$display("[FAIL] D=%0d t=%0t count=%0d, mong doi %0d", DEPTH, $time, count, q_count);
			end
			if (mem_full !== (q_count == DEPTH)) begin
				errors = errors + 1;
				$display("[FAIL] D=%0d t=%0t mem_full=%b, mong doi %b", DEPTH, $time, mem_full, (q_count == DEPTH));
			end
			if (mem_empty !== (q_count == 0)) begin
				errors = errors + 1;
				$display("[FAIL] D=%0d t=%0t mem_empty=%b, mong doi %b", DEPTH, $time, mem_empty, (q_count == 0));
			end
			if (exp_rd_valid) begin
				reads_checked = reads_checked + 1;
				if (rd_data !== exp_rd) begin
					errors = errors + 1;
					$display("[FAIL] D=%0d t=%0t rd_data=%h, mong doi %h", DEPTH, $time, rd_data, exp_rd);
				end
			end
		end
	end

	// =====================================================
	// Kích thích (đổi tín hiệu ở sườn xuống)
	// =====================================================
	reg [DATA_W-1:0] next_val;
	integer i, rnd;
	integer err_before;

	task step(input w, input r);
		begin
			wr_en = w;
			rd_en = r;
			if (w) begin
				wr_data = next_val;
				next_val = next_val + 1'b1;
			end
			@(negedge clk);
		end
	endtask

	task idle(input integer n);
		integer k;
		begin
			for (k = 0; k < n; k = k + 1) step(1'b0, 1'b0);
		end
	endtask

	task report(input [8*40-1:0] name);
		begin
			if (errors == err_before)
				$display("[PASS] D=%0d %0s", DEPTH, name);
			else
				$display("[FAIL] D=%0d %0s (%0d loi)", DEPTH, name, errors - err_before);
			err_before = errors;
		end
	endtask

	initial begin
		o_done = 1'b0;
		errors = 0;
		err_before = 0;
		reads_checked = 0;
		next_val = 8'h01;
		rnd = SEED;
		wr_en = 1'b0;
		rd_en = 1'b0;
		wr_data = 0;

		// T1: reset
		reset = 1'b1;
		repeat (3) @(negedge clk);
		reset = 1'b0;
		@(negedge clk);
		if (mem_empty !== 1'b1 || mem_full !== 1'b0 || count !== 0) begin
			errors = errors + 1;
			$display("[FAIL] D=%0d sau reset: empty=%b full=%b count=%0d", DEPTH, mem_empty, mem_full, count);
		end
		report("T1 reset");

		// T2: ghi đến đầy, rồi ghi thêm 3 lần (phải bị bỏ qua)
		for (i = 0; i < DEPTH + 3; i = i + 1) step(1'b1, 1'b0);
		if (mem_full !== 1'b1) begin
			errors = errors + 1;
			$display("[FAIL] D=%0d khong bao full sau %0d lan ghi", DEPTH, DEPTH);
		end
		report("T2 ghi den day, ghi thua bi chan");

		// T3: đọc đến rỗng, rồi đọc thêm 3 lần (phải bị bỏ qua)
		for (i = 0; i < DEPTH + 3; i = i + 1) step(1'b0, 1'b1);
		idle(1);
		if (mem_empty !== 1'b1) begin
			errors = errors + 1;
			$display("[FAIL] D=%0d khong bao empty sau khi doc het", DEPTH);
		end
		report("T3 doc den rong, doc thua bi chan");

		// T4: đọc và ghi đồng thời liên tục, con trỏ quay qua nhiều vòng
		for (i = 0; i < DEPTH / 2; i = i + 1) step(1'b1, 1'b0);
		for (i = 0; i < 3 * DEPTH; i = i + 1) step(1'b1, 1'b1);
		for (i = 0; i < DEPTH / 2; i = i + 1) step(1'b0, 1'b1);
		idle(1);
		report("T4 doc ghi dong thoi, quay nhieu vong");

		// T5: đang đầy mà đọc và ghi cùng lúc: chỉ đọc được, ghi bị chặn
		for (i = 0; i < DEPTH; i = i + 1) step(1'b1, 1'b0);
		step(1'b1, 1'b1);
		if (count !== DEPTH - 1) begin
			errors = errors + 1;
			$display("[FAIL] D=%0d full + doc/ghi dong thoi: count=%0d, mong doi %0d", DEPTH, count, DEPTH - 1);
		end
		for (i = 0; i < DEPTH; i = i + 1) step(1'b0, 1'b1);
		idle(1);
		report("T5 day + doc/ghi dong thoi");

		// T6: ngẫu nhiên 4000 chu kỳ
		for (i = 0; i < 4000; i = i + 1) begin
			rnd = $random(rnd);
			step(rnd[0], rnd[1] | rnd[2]);      // đọc nhiều hơn ghi một chút
		end
		for (i = 0; i < 4000; i = i + 1) begin
			rnd = $random(rnd);
			step(rnd[0] | rnd[1], rnd[2]);      // ghi nhiều hơn đọc: hay chạm đầy
		end
		for (i = 0; i < DEPTH; i = i + 1) step(1'b0, 1'b1);
		idle(2);
		report("T6 ngau nhien 8000 chu ky");

		// T7: reset giữa chừng xóa trạng thái
		for (i = 0; i < DEPTH / 2 + 1; i = i + 1) step(1'b1, 1'b0);
		wr_en = 1'b0;
		rd_en = 1'b0;
		reset = 1'b1;
		@(negedge clk);
		reset = 1'b0;
		@(negedge clk);
		if (mem_empty !== 1'b1 || count !== 0) begin
			errors = errors + 1;
			$display("[FAIL] D=%0d reset giua chung khong xoa: count=%0d", DEPTH, count);
		end
		report("T7 reset giua chung");

		$display("D=%0d: da kiem %0d lan doc, %0d loi", DEPTH, reads_checked, errors);
		o_errors = errors;
		o_done = 1'b1;
	end
endmodule
