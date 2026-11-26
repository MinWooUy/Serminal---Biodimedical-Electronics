`timescale 1ns/1ps

module tb_conv1d_layer_controller;

	// =====================================================
	// PARAMETERS
	// =====================================================
	localparam ADDR_W = 16;
	localparam CFG_W = 16;
	
	localparam ARRAY_ROWS = 3;
	localparam ARRAY_COLS = 3;
	
	localparam ROW_W = (ARRAY_ROWS <= 1) ? 1 : $clog2(ARRAY_ROWS + 1);
	localparam COL_W = (ARRAY_COLS <= 1) ? 1 : $clog2(ARRAY_COLS + 1);
	
	
	// =====================================================
	// CLOCK / RESET
	// =====================================================
	reg i_clk;
	reg i_reset;
	
	
	// =====================================================
	// LAYER CONTROL
	// =====================================================
	reg i_start;
	
	reg [CFG_W-1:0] i_out_len;
	reg [CFG_W-1:0] i_out_channels;
	reg [CFG_W-1:0] i_k_len;
	reg [ADDR_W-1:0] i_window_step;
	
	reg [ADDR_W-1:0] i_ifmap_base_addr;
	reg [ADDR_W-1:0] i_wgt_base_addr;
	
	
	// =====================================================
	// TILE STATUS
	// =====================================================
	reg i_tile_done;
	
	
	// =====================================================
	// TILE CONTROL OUTPUT
	// =====================================================
	wire o_tile_start;
	
	wire [ADDR_W-1:0] o_ifmap_base_addr;
	wire [ADDR_W-1:0] o_wgt_base_addr;
	
	wire [ROW_W-1:0] o_active_rows;
	wire [COL_W-1:0] o_active_cols;
	
	wire [CFG_W-1:0] o_out_row_base;
	wire [CFG_W-1:0] o_out_col_base;
	
	wire [CFG_W-1:0] o_k_len;
	wire [ADDR_W-1:0] o_window_step;
	
	
	// =====================================================
	// STATUS
	// =====================================================
	wire o_busy;
	wire o_done;
	wire [2:0] o_state_dbg;
	
	
	// =====================================================
	// TEST VARIABLES
	// =====================================================
	integer errors;
	integer tile_count;
	integer timeout_cnt;
	
	
	// =====================================================
	// DUT
	// =====================================================
	conv1d_layer_controller #(
		.ADDR_W(ADDR_W),
		.CFG_W(CFG_W),
		
		.ARRAY_ROWS(ARRAY_ROWS),
		.ARRAY_COLS(ARRAY_COLS),
		
		.ROW_W(ROW_W),
		.COL_W(COL_W)
	) dut(
		.i_clk(i_clk),
		.i_reset(i_reset),
		
		.i_start(i_start),
		
		.i_out_len(i_out_len),
		.i_out_channels(i_out_channels),
		.i_k_len(i_k_len),
		.i_window_step(i_window_step),
		
		.i_ifmap_base_addr(i_ifmap_base_addr),
		.i_wgt_base_addr(i_wgt_base_addr),
		
		.i_tile_done(i_tile_done),
		
		.o_tile_start(o_tile_start),
		
		.o_ifmap_base_addr(o_ifmap_base_addr),
		.o_wgt_base_addr(o_wgt_base_addr),
		
		.o_active_rows(o_active_rows),
		.o_active_cols(o_active_cols),
		
		.o_out_row_base(o_out_row_base),
		.o_out_col_base(o_out_col_base),
		
		.o_k_len(o_k_len),
		.o_window_step(o_window_step),
		
		.o_busy(o_busy),
		.o_done(o_done),
		.o_state_dbg(o_state_dbg)
	);
	
	
	// =====================================================
	// CLOCK
	// =====================================================
	initial begin
		i_clk = 1'b0;
		forever #5 i_clk = ~i_clk;
	end
	
	
	// =====================================================
	// RESET
	// =====================================================
	task reset_dut;
	begin
		i_reset = 1'b1;
		
		i_start = 1'b0;
		i_tile_done = 1'b0;
		
		i_out_len = {CFG_W{1'b0}};
		i_out_channels = {CFG_W{1'b0}};
		i_k_len = {CFG_W{1'b0}};
		i_window_step = {ADDR_W{1'b0}};
		
		i_ifmap_base_addr = {ADDR_W{1'b0}};
		i_wgt_base_addr = {ADDR_W{1'b0}};
		
		repeat(4) @(posedge i_clk);
		
		@(negedge i_clk);
		i_reset = 1'b0;
	end
	endtask
	
	
	// =====================================================
	// START LAYER
	// =====================================================
	task start_layer;
		input integer out_len;
		input integer out_channels;
		input integer k_len;
		input integer window_step;
		input integer ifmap_base;
		input integer wgt_base;
	begin
		@(negedge i_clk);
		
		i_out_len = out_len;
		i_out_channels = out_channels;
		i_k_len = k_len;
		i_window_step = window_step;
		
		i_ifmap_base_addr = ifmap_base;
		i_wgt_base_addr = wgt_base;
		
		i_start = 1'b1;
		
		@(posedge i_clk);
		@(negedge i_clk);
		
		i_start = 1'b0;
	end
	endtask
	
	
	// =====================================================
	// WAIT TILE START
	// =====================================================
	task wait_tile_start;
	begin
		timeout_cnt = 0;
		
		while((o_tile_start !== 1'b1) && (timeout_cnt < 100)) begin
			@(posedge i_clk);
			timeout_cnt = timeout_cnt + 1;
		end
		
		if(timeout_cnt >= 100) begin
			$display("[FAIL] Timeout waiting for o_tile_start");
			errors = errors + 1;
		end
	end
	endtask
	
	
	// =====================================================
	// SEND TILE DONE
	// =====================================================
	task send_tile_done;
	begin
		// Giả lập conv1d_tile_top chạy vài cycle
		repeat(3) @(posedge i_clk);
		
		@(negedge i_clk);
		i_tile_done = 1'b1;
		
		@(posedge i_clk);
		@(negedge i_clk);
		
		i_tile_done = 1'b0;
	end
	endtask
	
	
	// =====================================================
	// CHECK CURRENT TILE CONFIG
	// =====================================================
	task check_tile;
		input integer expected_tile;
		input integer expected_row_base;
		input integer expected_col_base;
		input integer expected_active_rows;
		input integer expected_active_cols;
		input integer expected_ifmap_base;
		input integer expected_wgt_base;
		input integer expected_k_len;
		input integer expected_window_step;
	begin
		$display("");
		$display("Checking tile %0d...", expected_tile);
		
		if(o_out_row_base !== expected_row_base) begin
			$display("[FAIL] row_base = %0d, expected = %0d", o_out_row_base, expected_row_base);
			errors = errors + 1;
		end else begin
			$display("[PASS] row_base = %0d", o_out_row_base);
		end
		
		if(o_out_col_base !== expected_col_base) begin
			$display("[FAIL] col_base = %0d, expected = %0d", o_out_col_base, expected_col_base);
			errors = errors + 1;
		end else begin
			$display("[PASS] col_base = %0d", o_out_col_base);
		end
		
		if(o_active_rows !== expected_active_rows) begin
			$display("[FAIL] active_rows = %0d, expected = %0d", o_active_rows, expected_active_rows);
			errors = errors + 1;
		end else begin
			$display("[PASS] active_rows = %0d", o_active_rows);
		end
		
		if(o_active_cols !== expected_active_cols) begin
			$display("[FAIL] active_cols = %0d, expected = %0d", o_active_cols, expected_active_cols);
			errors = errors + 1;
		end else begin
			$display("[PASS] active_cols = %0d", o_active_cols);
		end
		
		if(o_ifmap_base_addr !== expected_ifmap_base) begin
			$display("[FAIL] ifmap_base = %0d, expected = %0d", o_ifmap_base_addr, expected_ifmap_base);
			errors = errors + 1;
		end else begin
			$display("[PASS] ifmap_base = %0d", o_ifmap_base_addr);
		end
		
		if(o_wgt_base_addr !== expected_wgt_base) begin
			$display("[FAIL] wgt_base = %0d, expected = %0d", o_wgt_base_addr, expected_wgt_base);
			errors = errors + 1;
		end else begin
			$display("[PASS] wgt_base = %0d", o_wgt_base_addr);
		end
		
		if(o_k_len !== expected_k_len) begin
			$display("[FAIL] k_len = %0d, expected = %0d", o_k_len, expected_k_len);
			errors = errors + 1;
		end else begin
			$display("[PASS] k_len = %0d", o_k_len);
		end
		
		if(o_window_step !== expected_window_step) begin
			$display("[FAIL] window_step = %0d, expected = %0d", o_window_step, expected_window_step);
			errors = errors + 1;
		end else begin
			$display("[PASS] window_step = %0d", o_window_step);
		end
		
		tile_count = tile_count + 1;
	end
	endtask
	
	
	// =====================================================
	// WAIT LAYER DONE
	// =====================================================
	task wait_layer_done;
	begin
		timeout_cnt = 0;
		
		while((o_done !== 1'b1) && (timeout_cnt < 100)) begin
			@(posedge i_clk);
			timeout_cnt = timeout_cnt + 1;
		end
		
		if(timeout_cnt >= 100) begin
			$display("[FAIL] Timeout waiting for layer done");
			errors = errors + 1;
		end else begin
			$display("[PASS] Layer controller DONE");
		end
	end
	endtask
	
	
	// =====================================================
	// TC1
	//
	// Lout = 8
	// Cout = 5
	// K = 7
	// window_step = 2
	//
	// IFMAP base = 10
	// WGT base = 100
	//
	// Expected:
	//
	// Tile0: row=0 col=0 ar=3 ac=3 IFM=10 W=100
	// Tile1: row=0 col=3 ar=3 ac=2 IFM=10 W=121
	// Tile2: row=3 col=0 ar=3 ac=3 IFM=16 W=100
	// Tile3: row=3 col=3 ar=3 ac=2 IFM=16 W=121
	// Tile4: row=6 col=0 ar=2 ac=3 IFM=22 W=100
	// Tile5: row=6 col=3 ar=2 ac=2 IFM=22 W=121
	// =====================================================
	task test_partial_layer;
	begin
		$display("");
		$display("================================================");
		$display(" TC1 : LOUT=8 COUT=5 K=7 STEP=2");
		$display("================================================");
		
		reset_dut();
		tile_count = 0;
		
		start_layer(8, 5, 7, 2, 10, 100);
		
		wait_tile_start();
		check_tile(0, 0, 0, 3, 3, 10, 100, 7, 2);
		send_tile_done();
		
		wait_tile_start();
		check_tile(1, 0, 3, 3, 2, 10, 121, 7, 2);
		send_tile_done();
		
		wait_tile_start();
		check_tile(2, 3, 0, 3, 3, 16, 100, 7, 2);
		send_tile_done();
		
		wait_tile_start();
		check_tile(3, 3, 3, 3, 2, 16, 121, 7, 2);
		send_tile_done();
		
		wait_tile_start();
		check_tile(4, 6, 0, 2, 3, 22, 100, 7, 2);
		send_tile_done();
		
		wait_tile_start();
		check_tile(5, 6, 3, 2, 2, 22, 121, 7, 2);
		send_tile_done();
		
		wait_layer_done();
		
		if(tile_count != 6) begin
			$display("[FAIL] tile_count = %0d, expected = 6", tile_count);
			errors = errors + 1;
		end else begin
			$display("[PASS] tile_count = 6");
		end
		
		repeat(3) @(posedge i_clk);
	end
	endtask
	
	
	// =====================================================
	// TC2
	//
	// Lout = 6
	// Cout = 6
	//
	// 4 full 3x3 tiles
	// =====================================================
	task test_exact_layer;
	begin
		$display("");
		$display("================================================");
		$display(" TC2 : LOUT=6 COUT=6");
		$display("================================================");
		
		reset_dut();
		tile_count = 0;
		
		start_layer(6, 6, 3, 1, 0, 0);
		
		wait_tile_start();
		check_tile(0, 0, 0, 3, 3, 0, 0, 3, 1);
		send_tile_done();
		
		wait_tile_start();
		check_tile(1, 0, 3, 3, 3, 0, 9, 3, 1);
		send_tile_done();
		
		wait_tile_start();
		check_tile(2, 3, 0, 3, 3, 3, 0, 3, 1);
		send_tile_done();
		
		wait_tile_start();
		check_tile(3, 3, 3, 3, 3, 3, 9, 3, 1);
		send_tile_done();
		
		wait_layer_done();
		
		if(tile_count != 4) begin
			$display("[FAIL] tile_count = %0d, expected = 4", tile_count);
			errors = errors + 1;
		end else begin
			$display("[PASS] tile_count = 4");
		end
		
		repeat(3) @(posedge i_clk);
	end
	endtask
	
	
	// =====================================================
	// TC3
	//
	// Lout = 1
	// Cout = 1
	//
	// 1 tile 1x1
	// =====================================================
	task test_corner_layer;
	begin
		$display("");
		$display("================================================");
		$display(" TC3 : LOUT=1 COUT=1");
		$display("================================================");
		
		reset_dut();
		tile_count = 0;
		
		start_layer(1, 1, 5, 1, 20, 40);
		
		wait_tile_start();
		check_tile(0, 0, 0, 1, 1, 20, 40, 5, 1);
		send_tile_done();
		
		wait_layer_done();
		
		if(tile_count != 1) begin
			$display("[FAIL] tile_count = %0d, expected = 1", tile_count);
			errors = errors + 1;
		end else begin
			$display("[PASS] tile_count = 1");
		end
		
		repeat(3) @(posedge i_clk);
	end
	endtask
	
	
	// =====================================================
	// DEBUG MONITOR
	// =====================================================
	always @(posedge i_clk) begin
		if(!i_reset) begin
			$display("T=%0t state=%0d start=%b tile_start=%b tile_done=%b done=%b | row=%0d col=%0d ar=%0d ac=%0d | ifm=%0d wgt=%0d",
				$time,
				o_state_dbg,
				i_start,
				o_tile_start,
				i_tile_done,
				o_done,
				o_out_row_base,
				o_out_col_base,
				o_active_rows,
				o_active_cols,
				o_ifmap_base_addr,
				o_wgt_base_addr
			);
		end
	end
	
	
	// =====================================================
	// MAIN
	// =====================================================
	initial begin
		errors = 0;
		tile_count = 0;
		
		i_reset = 1'b0;
		i_start = 1'b0;
		i_tile_done = 1'b0;
		
		i_out_len = {CFG_W{1'b0}};
		i_out_channels = {CFG_W{1'b0}};
		i_k_len = {CFG_W{1'b0}};
		i_window_step = {ADDR_W{1'b0}};
		
		i_ifmap_base_addr = {ADDR_W{1'b0}};
		i_wgt_base_addr = {ADDR_W{1'b0}};
		
		test_partial_layer();
		test_exact_layer();
		test_corner_layer();
		
		$display("");
		$display("================================================");
		$display(" CONV1D LAYER CONTROLLER TEST SUMMARY");
		$display("================================================");
		
		if(errors == 0) begin
			$display(" RESULT : ALL TESTS PASSED");
		end else begin
			$display(" RESULT : FAILED (%0d errors)", errors);
		end
		
		$display("================================================");
		
		repeat(5) @(posedge i_clk);
		$finish;
	end
	
	
	// =====================================================
	// GLOBAL TIMEOUT
	// =====================================================
	initial begin
		#500000;
		
		$display("");
		$display("[TIMEOUT] tb_conv1d_layer_controller");
		
		$finish;
	end

endmodule