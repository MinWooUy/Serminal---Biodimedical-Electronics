`timescale 1ns/1ps
// 4 layer Conv Lite chay noi tiep tren conv1d_layer_top, so voi golden:
//   conv0+pool0 -> exp0, conv1+pool1 -> exp1, conv2+pool2 -> exp2, conv3+GAP -> gap
// Chay tu rtl/cnn1d, hex trong sim/hex (xem gen_layer_hex.py).
module tb_conv1d_layer_top;
	parameter ARRAY_ROWS = 3;
	parameter ARRAY_COLS = 3;
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
		.i_gap_rd_idx(gap_rd_idx), .o_gap_rd_data(gap_rd_data),
		.o_busy(busy), .o_done(done), .o_err(err)
	);

	reg [7:0]  xmem  [0:511];
	reg [7:0]  wmem  [0:16383];
	reg [31:0] bmem  [0:63];
	reg [7:0]  smem  [0:63];
	reg [7:0]  emem  [0:4095];
	reg [15:0] gmem  [0:63];

	integer i, errors, layer, nw, ne, cyc_start, cyc_total;
	integer n_cout, n_k;
	reg [7:0] got;

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

	initial begin
		errors = 0; cyc_total = 0;
		repeat (4) @(negedge clk);
		reset = 0;
		repeat (2) @(negedge clk);

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

			// chay layer
			cyc_start = $time / 10;
			@(negedge clk); layer_start = 1;
			@(negedge clk); layer_start = 0;
			wait (done);
			@(negedge clk);
			cyc_total = cyc_total + ($time / 10 - cyc_start);
			$display("layer %0d xong sau %0d chu ky, err=%0d", layer, $time / 10 - cyc_start, err);
			if (err) errors = errors + 1;

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

		if (errors == 0) $display("PASS tb_conv1d_layer_top %0dx%0d: layer 0..%0d khop golden, tong %0d chu ky", ARRAY_ROWS, ARRAY_COLS, LAST_LAYER, cyc_total);
		else $display("[FAIL] tb_conv1d_layer_top: %0d loi", errors);
		$finish;
	end

	initial begin
		#2000000000;
		$display("[FAIL] TIMEOUT");
		$finish;
	end
endmodule
