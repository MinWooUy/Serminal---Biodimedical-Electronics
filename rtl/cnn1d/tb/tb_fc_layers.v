`timescale 1ns/1ps
// FC1 + FC2 cua CNN Lite tren conv1d_layer_top (dung lai mang PE), so voi golden tung bit:
//   FC1: out_len 3, cout 32, K 68, co row_combine | gap_in  -> h (32 int8) o bank DOC [0..31]
//   FC2: out_len 1, cout 4,  K 32, co out_int32 | argmax    -> 4 logit int32 LE [0..15] + nhan [16]
// gap_mem va so kenh GAP duoc nap thang tu testbench (thay cho conv3) de chay nhanh.
// Vector: python scripts/rtl/gen_fc_hex.py (4 nhip N/S/V/F). Chay tu rtl/cnn1d.
module tb_fc_layers;
	parameter ARRAY_ROWS = 3;
	parameter ARRAY_COLS = 3;
	parameter FIRST_BEAT = 0;   // chay tu nhip nay (0..3)
	parameter WATCH = 0;        // 1: in trang thai moi 5000 chu ky (go roi khi treo)
	localparam NB = 4, COUT_MAX = 64;

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
	reg [3:0] flags = 0;
	wire [15:0] gap_rd_data;
	wire busy, done, err;

	conv1d_layer_top #(
		.DATA_W(8), .ACC_W(32), .COUT_MAX(COUT_MAX),
		.ARRAY_ROWS(ARRAY_ROWS), .ARRAY_COLS(ARRAY_COLS)
	) dut (
		.i_clk(clk), .i_reset(reset),
		.i_x_wr_en(x_wr_en), .i_x_wr_addr(x_wr_addr), .i_x_wr_data(x_wr_data), .i_x_swap(x_swap),
		.i_w_wr_en(w_wr_en), .i_w_wr_addr(w_wr_addr), .i_w_wr_data(w_wr_data), .i_w_swap(w_swap),
		.i_bias_wr_en(bias_wr_en), .i_bias_wr_addr(bias_wr_addr), .i_bias_wr_data(bias_wr_data),
		.i_shift_wr_en(shift_wr_en), .i_shift_wr_addr(shift_wr_addr), .i_shift_wr_data(shift_wr_data),
		.i_layer_start(layer_start), .i_out_len(out_len), .i_cout(cout), .i_k_len(k_len),
		.i_window_step(window_step), .i_pool_en(pool_en), .i_gap_en(gap_en), .i_pad_next(pad_next),
		.i_flags(flags),
		.i_gap_rd_idx(16'd0), .o_gap_rd_data(gap_rd_data),
		.i_dbg_rd_en(1'b0), .i_dbg_rd_addr(16'd0), .o_dbg_rd_data(), .o_dbg_rd_valid(),
		.o_busy(busy), .o_done(done), .o_err(err)
	);

	reg [7:0]  w1 [0:2175];
	reg [31:0] b1 [0:31];
	reg [7:0]  so [0:31];
	reg [7:0]  sg [0:31];
	reg [7:0]  w2 [0:127];
	reg [31:0] b2 [0:3];
	reg [15:0] gmem [0:NB*64-1];
	reg [7:0]  rmem [0:NB*4-1];
	reg [7:0]  hmem [0:NB*32-1];
	reg [31:0] lmem [0:NB*4-1];
	reg [7:0]  pmem [0:NB-1];

	integer i, beat, errors, cyc0, cyc_fc1, cyc_fc2;
	reg [31:0] got32;

	function [7:0] rd_bank; // bank DOC cua ifmap
		input integer a;
		begin
			if (dut.u_tile.u_ifmap_mem.read_bank_q == 1'b0) rd_bank = dut.u_tile.u_ifmap_mem.u_bank0.mem[a];
			else rd_bank = dut.u_tile.u_ifmap_mem.u_bank1.mem[a];
		end
	endfunction

	task write_weights; // [Cout][K] vao bank ghi roi swap
		input integer n;
		input integer which; // 1 = FC1, 2 = FC2
		begin
			for (i = 0; i < n; i = i + 1) begin
				@(negedge clk); w_wr_en = 1; w_wr_addr = i; w_wr_data = (which == 1) ? w1[i] : w2[i];
			end
			@(negedge clk); w_wr_en = 0;
			@(negedge clk); w_swap = 1;
			@(negedge clk); w_swap = 0;
		end
	endtask

	task run_layer;
		input [15:0] l_out_len, l_cout, l_k;
		input [3:0] l_flags;
		output integer cycles;
		begin
			out_len = l_out_len; cout = l_cout; k_len = l_k; window_step = l_k;
			pool_en = 0; gap_en = 0; pad_next = 0; flags = l_flags;
			cyc0 = $time / 10;
			@(negedge clk); layer_start = 1;
			@(negedge clk); layer_start = 0;
			wait (done);
			@(negedge clk);
			cycles = $time / 10 - cyc0;
			if (err) begin errors = errors + 1; $display("[FAIL] fmap_writer overrun"); end
		end
	endtask

	initial begin
		$readmemh("sim/hex/fc_w1.hex", w1);   $readmemh("sim/hex/fc_b1.hex", b1);
		$readmemh("sim/hex/fc_so.hex", so);   $readmemh("sim/hex/fc_sg.hex", sg);
		$readmemh("sim/hex/fc_w2.hex", w2);   $readmemh("sim/hex/fc_b2.hex", b2);
		$readmemh("sim/hex/fc_gap.hex", gmem); $readmemh("sim/hex/fc_rr.hex", rmem);
		$readmemh("sim/hex/fc_h.hex", hmem);  $readmemh("sim/hex/fc_logits.hex", lmem);
		$readmemh("sim/hex/fc_pred.hex", pmem);
		errors = 0;
		repeat (4) @(negedge clk);
		reset = 0;
		repeat (2) @(negedge clk);
		dut.gap_cout_q = 64; // nhu sau conv3 (layer GAP 64 kenh)

		for (beat = FIRST_BEAT; beat < NB; beat = beat + 1) begin
			// "conv3" vua xong: GAP 64 kenh; host ghi 4 byte RR vao hang 2 (dia chi 2*68 + 64)
			for (i = 0; i < 64; i = i + 1) dut.gap_mem[i] = gmem[beat*64 + i];
			for (i = 0; i < 4; i = i + 1) begin
				@(negedge clk); x_wr_en = 1; x_wr_addr = 200 + i; x_wr_data = rmem[beat*4 + i];
			end
			@(negedge clk); x_wr_en = 0;

			// ---- FC1 ----
			write_weights(32*68, 1);
			for (i = 0; i < 32; i = i + 1) begin
				@(negedge clk);
				bias_wr_en = 1; bias_wr_addr = i; bias_wr_data = b1[i];
				shift_wr_en = 1; shift_wr_addr = i; shift_wr_data = so[i];
			end
			for (i = 0; i < 32; i = i + 1) begin
				@(negedge clk); bias_wr_en = 0;
				shift_wr_en = 1; shift_wr_addr = COUT_MAX + i; shift_wr_data = sg[i];
			end
			@(negedge clk); bias_wr_en = 0; shift_wr_en = 0;
			run_layer(3, 32, 68, 4'b1100, cyc_fc1);
			for (i = 0; i < 32; i = i + 1)
				if (rd_bank(i) !== hmem[beat*32 + i]) begin
					errors = errors + 1;
					if (errors < 20) $display("[FAIL] beat %0d h[%0d]: got=%0d want=%0d", beat, i, rd_bank(i), hmem[beat*32 + i]);
				end

			// ---- FC2 ----
			write_weights(4*32, 2);
			for (i = 0; i < 4; i = i + 1) begin
				@(negedge clk); bias_wr_en = 1; bias_wr_addr = i; bias_wr_data = b2[i];
			end
			@(negedge clk); bias_wr_en = 0;
			run_layer(1, 4, 32, 4'b0011, cyc_fc2);
			for (i = 0; i < 4; i = i + 1) begin
				got32 = {rd_bank(4*i + 3), rd_bank(4*i + 2), rd_bank(4*i + 1), rd_bank(4*i)};
				if (got32 !== lmem[beat*4 + i]) begin
					errors = errors + 1;
					$display("[FAIL] beat %0d logit[%0d]: got=%0d want=%0d", beat, i, $signed(got32), $signed(lmem[beat*4 + i]));
				end
			end
			if (rd_bank(16) !== pmem[beat]) begin
				errors = errors + 1;
				$display("[FAIL] beat %0d pred: got=%0d want=%0d", beat, rd_bank(16), pmem[beat]);
			end
			$display("beat %0d: pred %0d (golden %0d), FC1 %0d chu ky, FC2 %0d chu ky", beat, rd_bank(16), pmem[beat], cyc_fc1, cyc_fc2);
		end

		if (errors == 0) $display("PASS tb_fc_layers %0dx%0d: FC1 + FC2 + argmax khop golden %0d nhip", ARRAY_ROWS, ARRAY_COLS, NB - FIRST_BEAT);
		else $display("[FAIL] tb_fc_layers: %0d loi", errors);
		$finish;
	end

	always @(posedge clk)
		if (WATCH && ($time / 10) % 5000 == 0)
			$display("[watch] t=%0d beat=%0d lstate=%0d xs=%0d ctrl=%0d tile=%0d commit_pending=%0d busy=%0d",
				$time / 10, beat, dut.lstate, dut.xs, dut.ctrl_dbg, dut.u_tile.state_q, dut.commit_pending, busy);

	initial begin
		#20000000;
		$display("[FAIL] tb_fc_layers: timeout");
		$finish;
	end
endmodule
