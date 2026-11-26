`timescale 1ns/1ps
// Ghi moc thoi gian (chu ky) cua TUNG O trong mot layer, de tach delta = tau - nap - tinh thanh cac thanh phan.
//
// Kich thich va golden giong tb_conv1d_layer_top (conv0..conv3 noi tiep, so tung byte voi golden), them
// mot monitor chi DOC tin hieu phan cung (khong sua RTL) qua tham chieu phan cap. Moi o in mot dong
//     TILE,<layer>,<idx>,<K>,<ru>,<cu>,<14 moc chu ky>,<n_wr>
// Phan tich: python scripts/tile_timing_report.py rtl/cnn1d/sim/out/tb_tile_timing.log
//
// Chay (Git Bash, o goc repo):
//   rtl/cnn1d/sim/run_tb.sh tb_tile_timing                                   # 7x7, chi conv0 (~21 k chu ky)
//   rtl/cnn1d/sim/run_tb.sh --param LAST_LAYER=3 tb_tile_timing              # ca conv0..conv3 (~364 k chu ky, lau)
//   rtl/cnn1d/sim/run_tb.sh --param ARRAY_ROWS=3 --param ARRAY_COLS=3 tb_tile_timing
//
// Moc (so chu ky tinh tu 0 luc bat dau mo phong; moc = chu ky dau tien tin hieu o muc nay SAU khi o bat dau):
//   t_start    o_tile_start cua layer controller (xung phat o)          [conv1d_layer_top.ctrl_tile_start]
//   t_ldst     tile FSM vao S_LOAD_START (hai loader duoc cap xung start)
//   t_loaded   tile FSM vao S_CORE_START (ca hai loader xong, bat dau cap xung cho loi)
//   t_cload    Core FSM vao S_LOAD  (psm_clr)
//   t_ckick    Core FSM vao S_PRE_KICK (preload_start)
//   t_cwait    Core FSM vao S_PRE_WAIT (cho preload_done)
//   t_ccomp    Core FSM vao S_COMPUTE  (feeder_start da phat = preload xong)
//   t_cdrain   Core FSM vao S_DRAIN (sa_tile_done: moi PE da nhan last)
//   t_cdone    Core FSM vao S_DONE  (psm_capture_done)
//   t_post     post_fire: o_tile_valid canh len, post-process bat dau
//   t_postv    o_post_valid: post-process tra ket qua
//   t_wstart   fmap_writer nhan i_start
//   t_wdone    fmap_writer bao o_done
//   t_commit   tile_commit: controller duoc phat o ke tiep
//   n_wr       so byte fmap_writer ghi (wr_en hoac gap_en) trong o
module tb_tile_timing;
	parameter ARRAY_ROWS = 7;
	parameter ARRAY_COLS = 7;
	parameter LAST_LAYER = 0;   // 0..3: chay den layer nay (3 = ca 4 layer, rat lau)

	reg clk = 0, reset = 1;
	always #5 clk = ~clk;

	reg x_wr_en = 0, x_swap = 0, w_wr_en = 0, w_swap = 0;
	reg [15:0] x_wr_addr = 0, w_wr_addr = 0;
	reg signed [7:0] x_wr_data = 0, w_wr_data = 0;
	reg bias_wr_en = 0, shift_wr_en = 0;
	reg [15:0] bias_wr_addr = 0, shift_wr_addr = 0;
	reg [31:0] bias_wr_data = 0;
	reg [7:0] shift_wr_data = 0;

	reg layer_start = 0;
	reg [15:0] out_len = 0, cout = 0, k_len = 0, window_step = 0, pad_next = 0;
	reg pool_en = 0, gap_en = 0;
	reg [15:0] gap_rd_idx = 0;
	wire [15:0] gap_rd_data;
	wire busy, done, err;

	conv1d_layer_top #(
		.DATA_W(8), .ACC_W(32),
		.ARRAY_ROWS(ARRAY_ROWS), .ARRAY_COLS(ARRAY_COLS)
	) dut (
		.i_clk(clk), .i_reset(reset),
		.i_x_wr_en(x_wr_en), .i_x_wr_addr(x_wr_addr), .i_x_wr_data(x_wr_data), .i_x_swap(x_swap),
		.i_w_wr_en(w_wr_en), .i_w_wr_addr(w_wr_addr), .i_w_wr_data(w_wr_data), .i_w_swap(w_swap),
		.i_bias_wr_en(bias_wr_en), .i_bias_wr_addr(bias_wr_addr), .i_bias_wr_data(bias_wr_data),
		.i_shift_wr_en(shift_wr_en), .i_shift_wr_addr(shift_wr_addr), .i_shift_wr_data(shift_wr_data),
		.i_layer_start(layer_start), .i_out_len(out_len), .i_cout(cout), .i_k_len(k_len),
		.i_window_step(window_step), .i_pool_en(pool_en), .i_gap_en(gap_en), .i_pad_next(pad_next),
		.i_flags(4'd0), .i_w_base(16'd0), .i_b_base(16'd0),
		.i_gap_rd_idx(gap_rd_idx), .o_gap_rd_data(gap_rd_data),
		.i_dbg_rd_en(1'b0), .i_dbg_rd_addr(16'd0), .o_dbg_rd_data(), .o_dbg_rd_valid(),
		.o_busy(busy), .o_done(done), .o_err(err)
	);

	// ------------------------------------------------------------------ monitor
	integer cyc = 0;
	always @(posedge clk) cyc <= cyc + 1;

	// Gia tri state cua cac FSM (khop localparam trong RTL)
	localparam T_LOAD_START = 3'd1, T_CORE_START = 3'd3, T_DONE = 3'd5;
	localparam C_LOAD = 3'd1, C_COMPUTE = 3'd2, C_DRAIN = 3'd3, C_DONE = 3'd4, C_PRE_KICK = 3'd5, C_PRE_WAIT = 3'd6;

	wire [2:0] tile_st = dut.u_tile.state_q;
	wire [2:0] core_st = dut.u_tile.u_compute_core.u_controller.state;

	integer layer;                 // layer dang chay (tb gan)
	integer in_tile;               // 1 giua t_start va t_commit
	integer tile_idx, tiles_seen, tiles_bad;
	integer tk, tru, tcu, n_wr;
	integer t_start, t_ldst, t_loaded, t_cload, t_ckick, t_cwait, t_ccomp, t_cdrain, t_cdone;
	integer t_post, t_postv, t_wstart, t_wdone, t_commit;

	initial begin
		in_tile = 0; tile_idx = 0; tiles_seen = 0; tiles_bad = 0; n_wr = 0;
	end

	task clear_stamps;
		begin
			t_start = -1; t_ldst = -1; t_loaded = -1; t_cload = -1; t_ckick = -1; t_cwait = -1; t_ccomp = -1;
			t_cdrain = -1; t_cdone = -1; t_post = -1; t_postv = -1; t_wstart = -1; t_wdone = -1; t_commit = -1;
			n_wr = 0;
		end
	endtask

	// moc phai tang dan va du 13 moc; sai thi tinh vao tiles_bad
	function ordered;
		input integer dummy;
		begin
			ordered = (t_start >= 0) && (t_ldst >= t_start) && (t_loaded >= t_ldst) && (t_cload >= t_loaded) &&
			          (t_ckick >= t_cload) && (t_cwait >= t_ckick) && (t_ccomp >= t_cwait) && (t_cdrain >= t_ccomp) &&
			          (t_cdone >= t_cdrain) && (t_post >= t_ccomp) && (t_postv >= t_post) && (t_wstart >= t_postv) &&
			          (t_wdone >= t_wstart) && (t_commit >= t_wdone) && (n_wr > 0);
		end
	endfunction

	always @(posedge clk) begin
		if (!reset) begin
			// bat dau o: xoa moc, ghi K, ru, cu cua o
			if (dut.ctrl_tile_start) begin
				clear_stamps;
				in_tile = 1;
				t_start = cyc;
				tk = dut.ctrl_k_len; tru = dut.ctrl_rows; tcu = dut.ctrl_cols;
			end
			if (in_tile) begin
				if (t_ldst   < 0 && tile_st == T_LOAD_START) t_ldst   = cyc;
				if (t_loaded < 0 && tile_st == T_CORE_START) t_loaded = cyc;
				if (t_cload  < 0 && core_st == C_LOAD)       t_cload  = cyc;
				if (t_ckick  < 0 && core_st == C_PRE_KICK)   t_ckick  = cyc;
				if (t_cwait  < 0 && core_st == C_PRE_WAIT)   t_cwait  = cyc;
				if (t_ccomp  < 0 && core_st == C_COMPUTE)    t_ccomp  = cyc;
				if (t_cdrain < 0 && core_st == C_DRAIN)      t_cdrain = cyc;
				if (t_cdone  < 0 && core_st == C_DONE)       t_cdone  = cyc;
				if (t_post   < 0 && dut.u_tile.g_post.post_fire) t_post = cyc;
				if (t_postv  < 0 && dut.u_tile.o_post_valid)     t_postv = cyc;
				if (t_wstart < 0 && dut.writer_start)        t_wstart = cyc;
				if (t_wdone  < 0 && dut.writer_done)         t_wdone  = cyc;
				if (dut.wr_en_w || dut.gap_en_w)             n_wr = n_wr + 1;
				if (dut.tile_commit) begin
					t_commit = cyc;
					tiles_seen = tiles_seen + 1;
					if (!ordered(0)) begin
						tiles_bad = tiles_bad + 1;
						if (tiles_bad < 10) $display("[FAIL] layer %0d o %0d: moc khong du hoac khong tang dan", layer, tile_idx);
					end
					$display("TILE,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d",
						layer, tile_idx, tk, tru, tcu,
						t_start, t_ldst, t_loaded, t_cload, t_ckick, t_cwait, t_ccomp, t_cdrain, t_cdone,
						t_post, t_postv, t_wstart, t_wdone, t_commit, n_wr);
					tile_idx = tile_idx + 1;
					in_tile = 0;
				end
			end
		end
	end

	// ------------------------------------------------------------------ kich thich + golden
	reg [7:0]  xmem  [0:511];
	reg [7:0]  wmem  [0:16383];
	reg [31:0] bmem  [0:63];
	reg [7:0]  smem  [0:63];
	reg [7:0]  emem  [0:4095];
	reg [15:0] gmem  [0:63];

	integer i, errors, nw, ne, cyc_start, cyc_total, layer_cycles, exp_tiles, rows_eff;
	integer n_cout;
	reg [7:0] got;
	reg match_board;

	task load_layer_params;
		input integer l;
		begin
			case (l)
				0: begin out_len = 360; cout = 16; k_len = 7;   window_step = 1;  pool_en = 1; gap_en = 0; pad_next = 2; nw = 112;   ne = 184*16; end
				1: begin out_len = 180; cout = 32; k_len = 80;  window_step = 16; pool_en = 1; gap_en = 0; pad_next = 1; nw = 2560;  ne = 92*32;  end
				2: begin out_len = 90;  cout = 64; k_len = 96;  window_step = 32; pool_en = 1; gap_en = 0; pad_next = 1; nw = 6144;  ne = 47*64;  end
				default: begin out_len = 45; cout = 64; k_len = 192; window_step = 64; pool_en = 0; gap_en = 1; pad_next = 0; nw = 12288; ne = 0; end
			endcase
		end
	endtask

	// so chu ky board/mo phong 7x7 (docs/issue/L04): conv0..conv3
	function integer board_cycles;
		input integer l;
		begin
			case (l)
				0: board_cycles = 21429;
				1: board_cycles = 104379;
				2: board_cycles = 124603;
				default: board_cycles = 113321;
			endcase
		end
	endfunction

	initial begin
		errors = 0; cyc_total = 0;
		layer = 0;
		clear_stamps;
		repeat (4) @(negedge clk);
		reset = 0;
		repeat (2) @(negedge clk);

		$display("TILE_HEADER,layer,idx,K,ru,cu,t_start,t_ldst,t_loaded,t_cload,t_ckick,t_cwait,t_ccomp,t_cdrain,t_cdone,t_post,t_postv,t_wstart,t_wdone,t_commit,n_wr");
		$display("CONFIG,%0d,%0d", ARRAY_ROWS, ARRAY_COLS);

		// input layer 0: 366 byte, vao bank ghi roi swap
		$readmemh("sim/hex/x0.hex", xmem);
		for (i = 0; i < 366; i = i + 1) begin
			@(negedge clk); x_wr_en = 1; x_wr_addr = i; x_wr_data = xmem[i];
		end
		@(negedge clk); x_wr_en = 0;
		@(negedge clk); x_swap = 1;
		@(negedge clk); x_swap = 0;

		for (layer = 0; layer <= LAST_LAYER; layer = layer + 1) begin
			load_layer_params(layer);
			case (layer)
				0: begin $readmemh("sim/hex/w0.hex", wmem); $readmemh("sim/hex/b0.hex", bmem); $readmemh("sim/hex/s0.hex", smem); $readmemh("sim/hex/exp0.hex", emem); end
				1: begin $readmemh("sim/hex/w1.hex", wmem); $readmemh("sim/hex/b1.hex", bmem); $readmemh("sim/hex/s1.hex", smem); $readmemh("sim/hex/exp1.hex", emem); end
				2: begin $readmemh("sim/hex/w2.hex", wmem); $readmemh("sim/hex/b2.hex", bmem); $readmemh("sim/hex/s2.hex", smem); $readmemh("sim/hex/exp2.hex", emem); end
				3: begin $readmemh("sim/hex/w3.hex", wmem); $readmemh("sim/hex/b3.hex", bmem); $readmemh("sim/hex/s3.hex", smem); $readmemh("sim/hex/gap.hex", gmem); end
			endcase
			n_cout = cout;

			// weight [Cout][K] -> bank ghi, swap
			for (i = 0; i < nw; i = i + 1) begin
				@(negedge clk); w_wr_en = 1; w_wr_addr = i; w_wr_data = wmem[i];
			end
			@(negedge clk); w_wr_en = 0;
			@(negedge clk); w_swap = 1;
			@(negedge clk); w_swap = 0;

			// bias + shift theo kenh
			for (i = 0; i < n_cout; i = i + 1) begin
				@(negedge clk);
				bias_wr_en = 1; bias_wr_addr = i; bias_wr_data = bmem[i];
				shift_wr_en = 1; shift_wr_addr = i; shift_wr_data = smem[i];
			end
			@(negedge clk); bias_wr_en = 0; shift_wr_en = 0;

			// chay layer; dem o de doi chieu voi cong thuc
			tile_idx = 0; tiles_seen = 0; tiles_bad = 0;
			cyc_start = $time / 10;
			@(negedge clk); layer_start = 1;
			@(negedge clk); layer_start = 0;
			wait (done);
			@(negedge clk);
			layer_cycles = $time / 10 - cyc_start;
			cyc_total = cyc_total + layer_cycles;
			$display("LAYER,%0d,%0d,%0d", layer, layer_cycles, tiles_seen);
			$display("layer %0d xong sau %0d chu ky, %0d o, err=%0d", layer, layer_cycles, tiles_seen, err);
			if (err) begin errors = errors + 1; $display("[FAIL] layer %0d: err=1 (overrun)", layer); end

			// so o phai bang ceil(T / rows) * ceil(Cout / C)
			rows_eff = pool_en ? (ARRAY_ROWS & ~1) : ARRAY_ROWS;
			exp_tiles = ((out_len + rows_eff - 1) / rows_eff) * ((cout + ARRAY_COLS - 1) / ARRAY_COLS);
			if (tiles_seen != exp_tiles) begin
				errors = errors + 1;
				$display("[FAIL] layer %0d: so o = %0d, cong thuc = %0d", layer, tiles_seen, exp_tiles);
			end else $display("[PASS] layer %0d: so o = %0d khop ceil(T/R)*ceil(Cout/C)", layer, tiles_seen);
			if (tiles_bad != 0) errors = errors + 1;
			else $display("[PASS] layer %0d: %0d o deu du 14 moc va tang dan", layer, tiles_seen);

			// 7x7: so chu ky phai trung so board/mo phong da ghi (L04)
			if (ARRAY_ROWS == 7 && ARRAY_COLS == 7) begin
				if (layer_cycles != board_cycles(layer)) begin
					errors = errors + 1;
					$display("[FAIL] layer %0d: %0d chu ky, board/L04 = %0d", layer, layer_cycles, board_cycles(layer));
				end else $display("[PASS] layer %0d: %0d chu ky trung so board", layer, layer_cycles);
			end

			if (layer < 3) begin
				// output da swap sang bank DOC cua ifmap
				for (i = 0; i < ne; i = i + 1) begin
					if (dut.u_tile.u_ifmap_mem.read_bank_q == 1'b0) got = dut.u_tile.u_ifmap_mem.u_bank0.mem[i];
					else got = dut.u_tile.u_ifmap_mem.u_bank1.mem[i];
					if (got !== emem[i]) begin
						errors = errors + 1;
						if (errors < 20)
							$display("[FAIL] layer %0d fmap[%0d] (t=%0d c=%0d): got=%0d want=%0d",
								layer, i, i / n_cout, i % n_cout, $signed(got), $signed(emem[i]));
					end
				end
				$display("layer %0d: so sanh %0d byte output (da pad)", layer, ne);
			end else begin
				for (i = 0; i < 64; i = i + 1) begin
					gap_rd_idx = i; #1;
					if (gap_rd_data !== gmem[i]) begin
						errors = errors + 1;
						if (errors < 20) $display("[FAIL] gap[%0d]: got=%0d want=%0d", i, gap_rd_data, gmem[i]);
					end
				end
				$display("layer 3: so sanh GAP 64 kenh");
			end
		end

		if (errors == 0) begin
			$display("[PASS] tb_tile_timing %0dx%0d: layer 0..%0d khop golden, tong %0d chu ky", ARRAY_ROWS, ARRAY_COLS, LAST_LAYER, cyc_total);
			$display("RESULT : ALL TESTS PASSED");
		end else begin
			$display("[FAIL] tb_tile_timing: %0d loi", errors);
			$display("RESULT : FAILED");
		end
		$finish;
	end

	initial begin
		#2000000000;
		$display("[FAIL] TIMEOUT");
		$finish;
	end
endmodule
