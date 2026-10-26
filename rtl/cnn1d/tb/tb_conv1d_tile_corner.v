// tb_conv1d_tile_corner.v
// Kiểm chứng conv1d_tile_top ở các trường hợp biên, so với mô hình tham chiếu.
//
// Mô hình giữ bản sao của CẢ HAI bank ping-pong (ifmap và weight), nên tính được
// kết quả kỳ vọng cho mọi cấu hình:
//   C[r][c] = sum_k X[ibase + r*step + k] * W[wbase + c*K + k]   (r < active_rows, c < active_cols)
//   C[r][c] = 0                                                  (hàng/cột không dùng)
// Ô nhớ chưa từng ghi mang giá trị x, nên nếu DUT đọc ra ngoài vùng hợp lệ thì
// kết quả kỳ vọng thành x và bị báo lỗi.
//
// Hai cấu hình chạy song song:
//   u_d16  : FEEDER_DEPTH = 16   -> biên K = 16 / 17 nhanh, nhiều test ngẫu nhiên
//   u_d1024: FEEDER_DEPTH = 1024 -> cấu hình thật, kích thước ECG, K = 1024 / 1025
`timescale 1ns/1ps

module tb_conv1d_tile_corner;
	wire done16, done16b, done1024;
	wire [31:0] err16, err16b, err1024;

	tb_conv1d_tile_corner_unit #(.FEEDER_DEPTH(16),   .MEM_DEPTH(256),  .N_RANDOM(40), .SEED(11), .PRELOAD_BIAS(0))
		u_d16   (.o_done(done16),   .o_errors(err16));
	tb_conv1d_tile_corner_unit #(.FEEDER_DEPTH(16),   .MEM_DEPTH(256),  .N_RANDOM(40), .SEED(11), .PRELOAD_BIAS(1), .DRAIN_BP(1))
		u_d16b  (.o_done(done16b),  .o_errors(err16b));
	tb_conv1d_tile_corner_unit #(.FEEDER_DEPTH(1024), .MEM_DEPTH(4096), .N_RANDOM(6),  .SEED(23), .PRELOAD_BIAS(1), .DRAIN_BP(1))
		u_d1024 (.o_done(done1024), .o_errors(err1024));

	initial begin
		wait(done16 && done16b && done1024);
		#20;
		$display("==============================================");
		$display(" CONV1D TILE CORNER TEST SUMMARY");
		$display("   FEEDER_DEPTH=16   : %0d loi", err16);
		$display("   FEEDER_DEPTH=16b,	preload bias, 	drain backpressure  : %0d loi", err16b);
		$display("   FEEDER_DEPTH=1024,	preload bias, 	drain backpressure  : %0d loi", err1024);
		if (err16 == 0 && err16b == 0 && err1024 == 0)
			$display(" RESULT : ALL TESTS PASSED");
		else
			$display(" RESULT : FAILED");
		$display("==============================================");
		$finish;
	end

	initial begin
		#20000000;   // 20 ms
		$display("[FAIL] TIMEOUT tb_conv1d_tile_corner");
		$finish;
	end
endmodule

module tb_conv1d_tile_corner_unit #(
	parameter FEEDER_DEPTH = 16,
	parameter MEM_DEPTH    = 256,
	parameter N_RANDOM     = 20,
	parameter SEED         = 1,
	parameter PRELOAD_BIAS = 0,
	parameter DRAIN_BP     = 0 // 1: i_drain_ready bật tắt ngẫu nhiên
)(
	output reg        o_done,
	output reg [31:0] o_errors
);
	localparam DATA_W     = 8;
	localparam ACC_W      = 32;
	localparam CFG_W      = 16;
	localparam ADDR_W     = $clog2(MEM_DEPTH);
	localparam ARRAY_ROWS = 3;
	localparam ARRAY_COLS = 3;
	localparam ROW_W      = $clog2(ARRAY_ROWS + 1);
	localparam COL_W      = $clog2(ARRAY_COLS + 1);
	localparam N_OUT      = ARRAY_ROWS * ARRAY_COLS;

	// =====================================================
	// DUT
	// =====================================================
	reg i_clk = 1'b0;
	always #5 i_clk = ~i_clk;

	reg i_reset, i_start;
	reg [CFG_W-1:0]  i_k_len;
	reg [ADDR_W-1:0] i_window_step, i_ifmap_base_addr, i_wgt_base_addr;
	reg [ROW_W-1:0]  i_active_rows;
	reg [COL_W-1:0]  i_active_cols;

	reg i_ifmap_mem_wr_en, i_ifmap_mem_swap;
	reg [ADDR_W-1:0] i_ifmap_mem_wr_addr;
	reg signed [DATA_W-1:0] i_ifmap_mem_wr_data;
	reg i_wgt_mem_wr_en, i_wgt_mem_swap;
	reg [ADDR_W-1:0] i_wgt_mem_wr_addr;
	reg signed [DATA_W-1:0] i_wgt_mem_wr_data;

	wire [N_OUT*ACC_W-1:0] o_tile;
	wire o_tile_valid, o_busy, o_done_dut, o_cfg_err;
	wire [2:0] o_state_dbg, o_core_state_dbg, o_ifmap_loader_state, o_wgt_loader_state;
	wire o_ifmap_read_bank, o_ifmap_write_bank, o_wgt_read_bank, o_wgt_write_bank;

	wire o_bias_rd_en;
	wire [COL_W-1:0] o_bias_rd_idx;
	reg i_drain_ready;
	wire o_drain_valid;
	wire [COL_W-1:0] o_drain_idx;
	wire [ARRAY_ROWS*ACC_W-1:0] o_drain_vec;

	// Bộ thu stream drain: ghép lại thành ma trận để so với kỳ vọng
	reg signed [ACC_W-1:0] str_c [0:ARRAY_ROWS*ARRAY_COLS-1];
	integer str_cnt, str_r, bp_seed;
	always @(posedge i_clk) begin
		if (o_drain_valid && i_drain_ready) begin
			for (str_r = 0; str_r < ARRAY_ROWS; str_r = str_r + 1)
				str_c[str_r * ARRAY_COLS + o_drain_idx] = o_drain_vec[str_r * ACC_W +: ACC_W];
			str_cnt = str_cnt + 1;
		end
	end
	// Backpressure: ready ngẫu nhiên khoảng 40% chu kỳ
	always @(negedge i_clk) begin
		bp_seed = $random(bp_seed);
		i_drain_ready = DRAIN_BP ? (bp_seed[2:0] < 3) : 1'b1;
	end

	// BISA RAM: địa chỉ = bias_base + idx (bias_base đóng vai cbase + col_base), đọc trễ 1 chu kỳ
	localparam BIAS_DEPTH = 64;
	reg signed [ACC_W-1:0] bias_mem [0:BIAS_DEPTH-1];
	reg signed [ACC_W-1:0] bias_q;

	integer bias_base;
	integer bias_reads;

	always @(posedge i_clk) begin
		if(o_bias_rd_en) begin
			bias_q <= bias_mem[bias_base + o_bias_rd_idx];
			bias_reads = bias_reads + 1;	
		end
	end

	conv1d_tile_top #(
		.DATA_W(DATA_W), .ACC_W(ACC_W),
		.ADDR_W(ADDR_W), .CFG_W(CFG_W),
		.IFMAP_MEM_DEPTH(MEM_DEPTH), .WGT_MEM_DEPTH(MEM_DEPTH),
		.FEEDER_DEPTH(FEEDER_DEPTH), .K_W(16),
		.ARRAY_ROWS(ARRAY_ROWS), .ARRAY_COLS(ARRAY_COLS),
		.ROW_W(ROW_W), .COL_W(COL_W)
	) dut (
		.i_clk(i_clk), .i_reset(i_reset),
		.i_start(i_start), .i_k_len(i_k_len), .i_window_step(i_window_step),
		.i_active_rows(i_active_rows), .i_active_cols(i_active_cols),
		.i_ifmap_base_addr(i_ifmap_base_addr), .i_wgt_base_addr(i_wgt_base_addr),
		.i_ifmap_mem_wr_en(i_ifmap_mem_wr_en), .i_ifmap_mem_wr_addr(i_ifmap_mem_wr_addr),
		.i_ifmap_mem_wr_data(i_ifmap_mem_wr_data), .i_ifmap_mem_swap(i_ifmap_mem_swap),
		.i_wgt_mem_wr_en(i_wgt_mem_wr_en), .i_wgt_mem_wr_addr(i_wgt_mem_wr_addr),
		.i_wgt_mem_wr_data(i_wgt_mem_wr_data), .i_wgt_mem_swap(i_wgt_mem_swap),
		.i_preload_mode(PRELOAD_BIAS[0]), .i_bias_rd_data(bias_q),
		.o_bias_rd_en(o_bias_rd_en), .o_bias_rd_idx(o_bias_rd_idx),
		.i_drain_ready(i_drain_ready), .o_drain_valid(o_drain_valid),
		.o_drain_idx(o_drain_idx), .o_drain_vec(o_drain_vec),
		.o_tile(o_tile), .o_tile_valid(o_tile_valid),
		.o_busy(o_busy), .o_done(o_done_dut),
		.o_state_dbg(o_state_dbg), .o_core_state_dbg(o_core_state_dbg),
		.o_ifmap_loader_state(o_ifmap_loader_state), .o_wgt_loader_state(o_wgt_loader_state),
		.o_ifmap_read_bank(o_ifmap_read_bank), .o_ifmap_write_bank(o_ifmap_write_bank),
		.o_wgt_read_bank(o_wgt_read_bank), .o_wgt_write_bank(o_wgt_write_bank),
		.o_cfg_err(o_cfg_err)
	);

	// =====================================================
	// Mô hình: bản sao 2 bank cho ifmap và weight
	// =====================================================
	reg signed [DATA_W-1:0] xb0 [0:MEM_DEPTH-1];
	reg signed [DATA_W-1:0] xb1 [0:MEM_DEPTH-1];
	reg signed [DATA_W-1:0] wb0 [0:MEM_DEPTH-1];
	reg signed [DATA_W-1:0] wb1 [0:MEM_DEPTH-1];
	reg x_rd_bank, w_rd_bank;     // bank DUT đang đọc (theo mô hình)

	integer errors, err_before, prints;
	integer seed;
	reg signed [ACC_W-1:0] exp_c [0:N_OUT-1];
	reg exp_has_x;

	// ---------------- tiện ích ----------------
	task fail(input [8*120-1:0] msg);
		begin
			errors = errors + 1;
			if (prints < 30) $display("[FAIL] D=%0d t=%0t %0s", FEEDER_DEPTH, $time, msg);
			prints = prints + 1;
		end
	endtask

	task report(input [8*100-1:0] name);
		begin
			if (errors == err_before) $display("[PASS] D=%0d %0s", FEEDER_DEPTH, name);
			else                      $display("[FAIL] D=%0d %0s (%0d loi)", FEEDER_DEPTH, name, errors - err_before);
			err_before = errors;
		end
	endtask

	function signed [DATA_W-1:0] rnd8(input integer dummy);
		integer r;
		begin
			r = $random(seed);
			rnd8 = r[7:0];
		end
	endfunction

	function integer rnd_range(input integer lo, input integer hi);   // [lo, hi]
		integer r;
		begin
			r = $random(seed);
			if (r < 0) r = -r;
			rnd_range = lo + (r % (hi - lo + 1));
		end
	endfunction

	// mode: 0 ngẫu nhiên, 1 x=-128 w=-128, 2 x=-128 w=127, 3 x=127 w=127, 4 x=+/-1 xen kẽ, w ngẫu nhiên
	function signed [DATA_W-1:0] gen_x(input integer mode, input integer a);
		begin
			case (mode)
				1, 2:    gen_x = -8'sd128;
				3:       gen_x = 8'sd127;
				4:       gen_x = a[0] ? -8'sd1 : 8'sd1;
				default: gen_x = rnd8(0);
			endcase
		end
	endfunction

	function signed [DATA_W-1:0] gen_w(input integer mode);
		begin
			case (mode)
				1:       gen_w = -8'sd128;
				2, 3:    gen_w = 8'sd127;
				default: gen_w = rnd8(0);
			endcase
		end
	endfunction

	// ---------------- ghi bộ nhớ (1 phần tử / chu kỳ) ----------------
	task put_x(input integer a, input signed [DATA_W-1:0] v);
		begin
			i_ifmap_mem_wr_en = 1'b1; i_ifmap_mem_wr_addr = a; i_ifmap_mem_wr_data = v;
			if (x_rd_bank) xb0[a] = v; else xb1[a] = v;     // ghi vào bank đang KHÔNG đọc
			@(negedge i_clk);
		end
	endtask

	task put_w(input integer a, input signed [DATA_W-1:0] v);
		begin
			i_wgt_mem_wr_en = 1'b1; i_wgt_mem_wr_addr = a; i_wgt_mem_wr_data = v;
			if (w_rd_bank) wb0[a] = v; else wb1[a] = v;
			@(negedge i_clk);
		end
	endtask

	// Ghi vùng dữ liệu của tile (luôn phủ đủ ARRAY_ROWS hàng / ARRAY_COLS cột và thêm
	// 4 ô rác ở cuối, để phát hiện DUT dùng hàng/cột không active hoặc đọc lố), rồi swap.
	task fill_and_swap(input integer ib, input integer wb, input integer k, input integer step, input integer mode);
		integer a, x_end, w_end;
		begin
			if (o_ifmap_write_bank !== ~x_rd_bank || o_wgt_write_bank !== ~w_rd_bank)
				fail("write bank cua DUT khac mo hinh");
			x_end = ib + (ARRAY_ROWS - 1) * step + k + 4;
			w_end = wb + ARRAY_COLS * k + 4;
			if (x_end > MEM_DEPTH) x_end = MEM_DEPTH;
			if (w_end > MEM_DEPTH) w_end = MEM_DEPTH;
			for (a = ib; a < x_end; a = a + 1) put_x(a, gen_x(mode, a));
			i_ifmap_mem_wr_en = 1'b0;
			for (a = wb; a < w_end; a = a + 1) put_w(a, gen_w(mode));
			i_wgt_mem_wr_en = 1'b0;
			i_ifmap_mem_swap = 1'b1; i_wgt_mem_swap = 1'b1;
			@(negedge i_clk);
			i_ifmap_mem_swap = 1'b0; i_wgt_mem_swap = 1'b0;
			x_rd_bank = ~x_rd_bank;
			w_rd_bank = ~w_rd_bank;
			// bias: base ngẫu nhiên, giá trị theo mode (1: -2^30, 3: +2^30, 4: 0, còn lại ngẫu nhiên ±2^20)
			bias_base = rnd_range(0, BIAS_DEPTH - ARRAY_COLS);
			for(a = 0; a < BIAS_DEPTH; a = a + 1) begin
				case(mode)
					1: bias_mem[a] = -(1 << 20);
					2: bias_mem[a] = (1 << 20);
					4: bias_mem[a] = 0;
					default: bias_mem[a] = rnd_range(0, (1 << 21)) - (1 << 20);
				endcase
			end
			if (o_ifmap_read_bank !== x_rd_bank || o_wgt_read_bank !== w_rd_bank)
				fail("read bank cua DUT khac mo hinh sau swap");
		end
	endtask

	task compute_expected(input integer k, input integer step, input integer ar, input integer ac,
	                      input integer ib, input integer wb);
		integer r, c, j, xa, wa;
		reg signed [DATA_W-1:0] xv, wv;
		reg signed [ACC_W-1:0] acc;
		begin
			exp_has_x = 1'b0;
			for (r = 0; r < ARRAY_ROWS; r = r + 1)
				for (c = 0; c < ARRAY_COLS; c = c + 1) begin
					acc = 0;
					if (PRELOAD_BIAS && r < ar && c < ac) acc = bias_mem[bias_base + c];
					if (r < ar && c < ac)
						for (j = 0; j < k; j = j + 1) begin
							xa = ib + r * step + j;
							wa = wb + c * k + j;
							xv = x_rd_bank ? xb1[xa] : xb0[xa];
							wv = w_rd_bank ? wb1[wa] : wb0[wa];
							if ((^xv === 1'bx) || (^wv === 1'bx)) exp_has_x = 1'b1;
							acc = acc + xv * wv;
						end
					exp_c[r * ARRAY_COLS + c] = acc;
				end
		end
	endtask

	// ---------------- chạy một tile ----------------
	// hold: số chu kỳ giữ i_start = 1 (1 = xung; lớn = giữ qua cả lúc done)
	task run_tile(input integer k, input integer step, input integer ar, input integer ac,
	              input integer ib, input integer wb, input integer expect_err, input integer hold);
		integer cyc, tmo, idx;
		reg saw_load, saw_busy;
		reg signed [ACC_W-1:0] act;
		begin
			if (!expect_err) begin
				compute_expected(k, step, ar, ac, ib, wb);
				if (exp_has_x) fail("mo hinh doc o nho chua ghi (cau hinh test sai)");
			end
			tmo = 12 * (k + 8) * ARRAY_ROWS + 2000;
			i_k_len = k; i_window_step = step;
			i_active_rows = ar; i_active_cols = ac;
			i_ifmap_base_addr = ib; i_wgt_base_addr = wb;
			str_cnt = 0;
			for(idx = 0; idx < N_OUT; idx = idx + 1) str_c[idx] = 0;
			i_start = 1'b1;
			cyc = 0; saw_load = 1'b0; saw_busy = 1'b0;
			while (o_done_dut !== 1'b1 && cyc < tmo) begin
				@(negedge i_clk);
				cyc = cyc + 1;
				if (cyc >= hold) i_start = 1'b0;
				if (o_ifmap_loader_state != 0 || o_wgt_loader_state != 0) saw_load = 1'b1;
				if (o_busy) saw_busy = 1'b1;
			end

			if (cyc >= tmo) begin
				fail("timeout: tile khong bao done (treo)");
			end else if (expect_err) begin
				if (o_cfg_err !== 1'b1) fail("cau hinh sai nhung o_cfg_err = 0");
				if (saw_load)           fail("cau hinh sai nhung loader van chay");
				if (saw_busy)           fail("cau hinh sai nhung o_busy = 1");
			end else begin
				if (o_cfg_err !== 1'b0)    fail("cau hinh hop le nhung o_cfg_err = 1");
				if (o_tile_valid !== 1'b1) fail("done nhung o_tile_valid = 0");
				if (str_cnt != ac) fail("so lan bat tay drain khac active_cols");
				for (idx = 0; idx < N_OUT; idx = idx + 1)
					if ((idx / ARRAY_COLS) < ar && (idx % ARRAY_COLS) < ac && str_c[idx] !== exp_c[idx]) begin
						errors = errors + 1;
						if (prints < 30) $display("[FAIL] D=%0d t=%0t stream C[%0d][%0d]=%0d, mong doi %0d", FEEDER_DEPTH, $time, idx / ARRAY_COLS, idx % ARRAY_COLS, str_c[idx], exp_c[idx]);
						prints = prints + 1;
					end
				for (idx = 0; idx < N_OUT; idx = idx + 1) begin
					act = o_tile[idx * ACC_W +: ACC_W];
					if (act !== exp_c[idx]) begin
						errors = errors + 1;
						if (prints < 30)
							$display("[FAIL] D=%0d t=%0t K=%0d step=%0d ar=%0d ac=%0d C[%0d][%0d]=%0d, mong doi %0d",
							         FEEDER_DEPTH, $time, k, step, ar, ac, idx / ARRAY_COLS, idx % ARRAY_COLS, act, exp_c[idx]);
						prints = prints + 1;
					end
				end
			end

			// nhả start (nếu còn giữ) -> DUT phải về IDLE và không chạy tile thứ hai
			i_start = 1'b0;
			cyc = 0;
			while (o_done_dut === 1'b1 && cyc < 10) begin @(negedge i_clk); cyc = cyc + 1; end
			if (o_done_dut === 1'b1) fail("o_done khong ha sau khi nha start");
			repeat (5) begin
				@(negedge i_clk);
				if (o_busy || o_ifmap_loader_state != 0) fail("DUT tu chay tiep sau khi tile xong");
			end
		end
	endtask

	// fill + run gộp
	task tile(input integer k, input integer step, input integer ar, input integer ac,
	          input integer ib, input integer wb, input integer mode);
		begin
			fill_and_swap(ib, wb, k, step, mode);
			run_tile(k, step, ar, ac, ib, wb, 0, 1);
		end
	endtask

	task do_reset;
		begin
			i_reset = 1'b1;
			i_start = 1'b0;
			i_ifmap_mem_wr_en = 1'b0; i_ifmap_mem_swap = 1'b0;
			i_wgt_mem_wr_en = 1'b0;   i_wgt_mem_swap = 1'b0;
			repeat (3) @(negedge i_clk);
			i_reset = 1'b0;
			x_rd_bank = 1'b0;     // ping-pong reset về bank 0
			w_rd_bank = 1'b0;
			@(negedge i_clk);
		end
	endtask

	// =====================================================
	// Kịch bản
	// =====================================================
	integer i, k, st, ar, ac, ib, wb, km, mode;

	initial begin
		o_done = 1'b0;
		errors = 0; err_before = 0; prints = 0;
		seed = SEED;
		bp_seed = SEED + 1000;
		str_cnt = 0;
		bias_base = 0;
		bias_reads = 0;
		bias_q = 0;
		i_k_len = 0; i_window_step = 0; i_active_rows = 0; i_active_cols = 0;
		i_ifmap_base_addr = 0; i_wgt_base_addr = 0;
		i_ifmap_mem_wr_addr = 0; i_ifmap_mem_wr_data = 0;
		i_wgt_mem_wr_addr = 0;   i_wgt_mem_wr_data = 0;
		@(negedge i_clk);
		do_reset;
		if (o_cfg_err !== 1'b0) fail("o_cfg_err khong ve 0 sau reset");
		report("C00 reset: o_cfg_err = 0");

		tile(1, 1, 3, 3, 0, 0, 0);
		report("C01 K = 1 (nho nhat)");

		tile(FEEDER_DEPTH, 1, 3, 3, 0, 0, 0);
		report("C02 K = DEPTH (FIFO vua day)");

		run_tile(FEEDER_DEPTH + 1, 1, 3, 3, 0, 0, 1, 1);
		report("C03 K = DEPTH+1 -> o_cfg_err, khong treo");

		tile(5, 1, 3, 3, 0, 0, 0);
		if (o_cfg_err !== 1'b0) fail("o_cfg_err khong xoa sau tile hop le");
		report("C04 tile hop le ngay sau tile loi");

		run_tile(0, 1, 3, 3, 0, 0, 1, 1);
		report("C05 K = 0 -> o_cfg_err");

		run_tile(16'hFFFF, 1, 3, 3, 0, 0, 1, 1);
		report("C06 K = 65535 -> o_cfg_err");

		for (ar = 1; ar <= ARRAY_ROWS; ar = ar + 1)
			for (ac = 1; ac <= ARRAY_COLS; ac = ac + 1)
				tile(5, 2, ar, ac, 0, 0, 0);
		report("C07 active_rows x active_cols: 9 to hop");

		tile(6, 0, 3, 3, 0, 0, 0);
		report("C08 step = 0 (moi hang cung cua so)");
		tile(6, 6, 3, 3, 0, 0, 0);
		report("C09 step = K (cua so lien ke, khong chong)");
		tile(4, 9, 3, 3, 0, 0, 0);
		report("C10 step > K (cua so cach quang)");

		km = FEEDER_DEPTH;
		for (mode = 1; mode <= 4; mode = mode + 1) tile(km, 1, 3, 3, 0, 0, mode);
		report("C11 gia tri bien: -128*-128, -128*127, 127*127, +/-1 voi K = DEPTH");

		tile(7, 3, 3, 3, 37, 13, 0);
		tile(7, 3, 2, 3, MEM_DEPTH / 4 + 1, MEM_DEPTH / 8 + 3, 0);
		report("C12 base ifmap/weight khac 0, le");

		tile(3, 1, 3, 3, 0, 0, 0);
		fill_and_swap(0, 0, 4, 1, 0);
		run_tile(4, 1, 3, 3, 0, 0, 0, 1);
		run_tile(4, 1, 3, 3, 0, 0, 0, 1);      // chạy lại cùng bank, không ghi mới
		run_tile(FEEDER_DEPTH + 1, 1, 3, 3, 0, 0, 1, 1);
		run_tile(4, 1, 3, 3, 0, 0, 0, 1);      // dữ liệu bank vẫn còn sau tile lỗi
		report("C13 nhieu tile lien tiep, chay lai cung bank, xen tile loi");

		fill_and_swap(0, 0, 8, 1, 0);
		run_tile(8, 1, 3, 3, 0, 0, 0, 100000); // giữ start qua cả lúc done
		report("C14 giu i_start muc 1 suot tile: chi chay 1 lan");

		// reset giữa chừng: trong lúc load, rồi trong lúc compute
		fill_and_swap(0, 0, FEEDER_DEPTH, 1, 0);
		i_k_len = FEEDER_DEPTH; i_window_step = 1; i_active_rows = 3; i_active_cols = 3;
		i_ifmap_base_addr = 0; i_wgt_base_addr = 0;
		i_start = 1'b1; @(negedge i_clk); i_start = 1'b0;
		repeat (10) @(negedge i_clk);
		do_reset;
		if (o_busy || o_done_dut || o_ifmap_loader_state != 0) fail("reset trong luc load: DUT chua ve IDLE");
		tile(5, 1, 3, 3, 0, 0, 0);
		fill_and_swap(0, 0, 6, 1, 0);
		i_k_len = 6; i_start = 1'b1; @(negedge i_clk); i_start = 1'b0;
		while (o_state_dbg != 3'd4) @(negedge i_clk);   // S_COMPUTE
		repeat (2) @(negedge i_clk);
		do_reset;
		tile(5, 1, 3, 3, 0, 0, 0);
		report("C15 reset giua luc load / compute roi chay tiep");

		fill_and_swap(0, 0, 6, 1, 0);
		i_k_len = 6; i_window_step = 1; i_active_rows = 3; i_active_cols = 3;
		i_start = 1'b1; @(negedge i_clk); i_start = 1'b0;
		while (o_core_state_dbg != 3'd6) @(negedge i_clk);   // core S_PRE_WAIT
		do_reset;
		tile(5, 1, 3, 3, 0, 0, 0);
		report("C18 reset giua luc preload roi chay tiep");
		if (PRELOAD_BIAS) begin
			if (bias_reads == 0) fail("che do bias nhung chua doc bias RAM lan nao");
			report("C19 che do bias: co doc bias RAM");
		end else begin
			if (bias_reads != 0) fail("che do 0 nhung van doc bias RAM");
			report("C19 che do 0: khong doc bias RAM");
		end

		if (FEEDER_DEPTH >= 192) begin
			tile(7, 1, 3, 3, 0, 0, 0);
			tile(80, 16, 3, 3, 0, 0, 0);
			tile(96, 32, 3, 3, 0, 0, 0);
			tile(192, 64, 3, 3, 0, 0, 0);
			tile(192, 64, 1, 2, 0, 0, 0);
			report("C16 kich thuoc ECG: K = 7/80/96/192, step = Cin");
		end

		for (i = 0; i < N_RANDOM; i = i + 1) begin
			k  = rnd_range(1, FEEDER_DEPTH);
			st = rnd_range(0, (k < 64) ? k + 4 : 64);
			ar = rnd_range(1, ARRAY_ROWS);
			ac = rnd_range(1, ARRAY_COLS);
			ib = rnd_range(0, MEM_DEPTH - (ARRAY_ROWS - 1) * st - k - 5);
			wb = rnd_range(0, MEM_DEPTH - ARRAY_COLS * k - 5);
			if (ib < 0) ib = 0;
			if (wb < 0) wb = 0;
			tile(k, st, ar, ac, ib, wb, 0);
		end
		report("C17 cau hinh ngau nhien");

		$display("D=%0d bias=%0d: %0d loi, %0d lan doc bias", FEEDER_DEPTH, PRELOAD_BIAS, errors, bias_reads);
		o_errors = errors;
		o_done = 1'b1;
	end
endmodule
