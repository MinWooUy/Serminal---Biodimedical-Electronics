`timescale 1ns/1ps

module tb_cnn_compute_core;

	// ---------
	// Parameters
	// ---------
	localparam DATA_W = 8;
	localparam ACC_W = 32;
	localparam ARRAY_ROWS = 3;
	localparam ARRAY_COLS = 3;
	localparam COL_W = (ARRAY_COLS <= 1) ? 1 : $clog2(ARRAY_COLS + 1);


	// ---------
	// Clock / Reset
	// ---------
	reg i_clk;
	reg i_reset;


	// ---------
	// Control
	// ---------
	reg i_start;
	reg [COL_W-1:0] i_active_cols;

	wire o_feeder_start;


	// ---------
	// IFMAP stream
	// ---------
	reg [ARRAY_ROWS*DATA_W-1:0] i_ifm_data;

	reg [ARRAY_ROWS-1:0] i_ifm_valid;
	reg [ARRAY_ROWS-1:0] i_ifm_first;
	reg [ARRAY_ROWS-1:0] i_ifm_last;


	// ---------
	// Weight stream
	// ---------
	reg [ARRAY_COLS*DATA_W-1:0] i_wgt_data;

	reg [ARRAY_COLS-1:0] i_wgt_valid;
	reg [ARRAY_COLS-1:0] i_wgt_first;
	reg [ARRAY_COLS-1:0] i_wgt_last;


	// ---------
	// Result
	// ---------
	wire [ARRAY_ROWS*ARRAY_COLS*ACC_W-1:0] o_tile;

	wire o_tile_valid;

	wire o_busy;
	wire o_done;

	wire [2:0] o_state_dbg;


	// ---------
	// Test variables
	// ---------
	integer errors;
	integer timeout_cnt;


	// ---------
	// DUT
	// ---------
	cnn_compute_core #(
		.DATA_W(DATA_W),
		.ACC_W(ACC_W),
		.ARRAY_ROWS(ARRAY_ROWS),
		.ARRAY_COLS(ARRAY_COLS),
		.COL_W(COL_W)
	) dut(
		.i_clk(i_clk),
		.i_reset(i_reset),

		.i_start(i_start),
		.i_active_cols(i_active_cols),

		// IFMAP stream
		.i_ifm_data(i_ifm_data),
		.i_ifm_valid(i_ifm_valid),
		.i_ifm_first(i_ifm_first),
		.i_ifm_last(i_ifm_last),

		// Weight stream
		.i_wgt_data(i_wgt_data),
		.i_wgt_valid(i_wgt_valid),
		.i_wgt_first(i_wgt_first),
		.i_wgt_last(i_wgt_last),

		.o_feeder_start(o_feeder_start),

		.o_tile(o_tile),
		.o_tile_valid(o_tile_valid),

		.o_busy(o_busy),
		.o_done(o_done),

		.o_state_dbg(o_state_dbg)
	);


	// ---------
	// Clock
	// ---------
	initial begin
		i_clk = 1'b0;

		forever #5 i_clk = ~i_clk;
	end


	// ---------
	// Clear input streams
	// ---------
	task clear_stream;
	begin
		i_ifm_data = {ARRAY_ROWS*DATA_W{1'b0}};

		i_ifm_valid = {ARRAY_ROWS{1'b0}};
		i_ifm_first = {ARRAY_ROWS{1'b0}};
		i_ifm_last = {ARRAY_ROWS{1'b0}};


		i_wgt_data = {ARRAY_COLS*DATA_W{1'b0}};

		i_wgt_valid = {ARRAY_COLS{1'b0}};
		i_wgt_first = {ARRAY_COLS{1'b0}};
		i_wgt_last = {ARRAY_COLS{1'b0}};
	end
	endtask


	// ---------
	// Reset
	// ---------
	task reset_dut;
	begin
		i_reset = 1'b1;

		i_start = 1'b0;

		i_active_cols = 0;

		clear_stream();


		repeat(4)
			@(posedge i_clk);


		@(negedge i_clk);

		i_reset = 1'b0;
	end
	endtask


	// ---------
	// Start tile
	// ---------
	task start_tile;

		input integer active_cols;

	begin
		@(negedge i_clk);

		i_active_cols = active_cols;

		i_start = 1'b1;


		@(posedge i_clk);


		@(negedge i_clk);

		i_start = 1'b0;
	end
	endtask


	// =========================================================
	// Drive full 3x3 skewed streams
	//
	// A:
	//
	// row0 = 1 2 3
	// row1 = 2 3 4
	// row2 = 3 4 5
	//
	// B columns:
	//
	// col0 =  1  0 -1
	// col1 =  1  1  1
	// col2 = -1  2 -1
	//
	// Boundary timing:
	//
	// cycle      IFMAP                   WEIGHT
	//
	// 0          r0:k0                   c0:k0
	//
	// 1          r0:k1 r1:k0             c0:k1 c1:k0
	//
	// 2          r0:k2 r1:k1 r2:k0       c0:k2 c1:k1 c2:k0
	//
	// 3                r1:k2 r2:k1             c1:k2 c2:k1
	//
	// 4                      r2:k2                   c2:k2
	// =========================================================

	task drive_stream_active_cols_3;
	begin

		// Wait until controller tells external feeders
		// to begin issuing the tile.
		wait(o_feeder_start == 1'b1);


		// =====================================================
		// Cycle 0
		// =====================================================

		@(negedge i_clk);

		clear_stream();


		// IFMAP row0 = k0 = 1
		i_ifm_data[0*DATA_W +: DATA_W] = 8'sd1;

		i_ifm_valid[0] = 1'b1;
		i_ifm_first[0] = 1'b1;


		// Weight col0 = k0 = 1
		i_wgt_data[0*DATA_W +: DATA_W] = 8'sd1;

		i_wgt_valid[0] = 1'b1;
		i_wgt_first[0] = 1'b1;


		// =====================================================
		// Cycle 1
		// =====================================================

		@(negedge i_clk);

		clear_stream();


		// row0 k1 = 2
		i_ifm_data[0*DATA_W +: DATA_W] = 8'sd2;

		i_ifm_valid[0] = 1'b1;


		// row1 k0 = 2
		i_ifm_data[1*DATA_W +: DATA_W] = 8'sd2;

		i_ifm_valid[1] = 1'b1;
		i_ifm_first[1] = 1'b1;


		// col0 k1 = 0
		i_wgt_data[0*DATA_W +: DATA_W] = 8'sd0;

		i_wgt_valid[0] = 1'b1;


		// col1 k0 = 1
		i_wgt_data[1*DATA_W +: DATA_W] = 8'sd1;

		i_wgt_valid[1] = 1'b1;
		i_wgt_first[1] = 1'b1;


		// =====================================================
		// Cycle 2
		// =====================================================

		@(negedge i_clk);

		clear_stream();


		// row0 k2 = 3
		i_ifm_data[0*DATA_W +: DATA_W] = 8'sd3;

		i_ifm_valid[0] = 1'b1;
		i_ifm_last[0] = 1'b1;


		// row1 k1 = 3
		i_ifm_data[1*DATA_W +: DATA_W] = 8'sd3;

		i_ifm_valid[1] = 1'b1;


		// row2 k0 = 3
		i_ifm_data[2*DATA_W +: DATA_W] = 8'sd3;

		i_ifm_valid[2] = 1'b1;
		i_ifm_first[2] = 1'b1;


		// col0 k2 = -1
		i_wgt_data[0*DATA_W +: DATA_W] = -8'sd1;

		i_wgt_valid[0] = 1'b1;
		i_wgt_last[0] = 1'b1;


		// col1 k1 = 1
		i_wgt_data[1*DATA_W +: DATA_W] = 8'sd1;

		i_wgt_valid[1] = 1'b1;


		// col2 k0 = -1
		i_wgt_data[2*DATA_W +: DATA_W] = -8'sd1;

		i_wgt_valid[2] = 1'b1;
		i_wgt_first[2] = 1'b1;


		// =====================================================
		// Cycle 3
		// =====================================================

		@(negedge i_clk);

		clear_stream();


		// row1 k2 = 4
		i_ifm_data[1*DATA_W +: DATA_W] = 8'sd4;

		i_ifm_valid[1] = 1'b1;
		i_ifm_last[1] = 1'b1;


		// row2 k1 = 4
		i_ifm_data[2*DATA_W +: DATA_W] = 8'sd4;

		i_ifm_valid[2] = 1'b1;


		// col1 k2 = 1
		i_wgt_data[1*DATA_W +: DATA_W] = 8'sd1;

		i_wgt_valid[1] = 1'b1;
		i_wgt_last[1] = 1'b1;


		// col2 k1 = 2
		i_wgt_data[2*DATA_W +: DATA_W] = 8'sd2;

		i_wgt_valid[2] = 1'b1;


		// =====================================================
		// Cycle 4
		// =====================================================

		@(negedge i_clk);

		clear_stream();


		// row2 k2 = 5
		i_ifm_data[2*DATA_W +: DATA_W] = 8'sd5;

		i_ifm_valid[2] = 1'b1;
		i_ifm_last[2] = 1'b1;


		// col2 k2 = -1
		i_wgt_data[2*DATA_W +: DATA_W] = -8'sd1;

		i_wgt_valid[2] = 1'b1;
		i_wgt_last[2] = 1'b1;


		// =====================================================
		// Return bus to idle
		// =====================================================

		@(negedge i_clk);

		clear_stream();

	end
	endtask


	// =========================================================
	// Drive active_cols = 2
	//
	// Same IFMAP.
	//
	// Only:
	//
	// col0 = 1 0 -1
	// col1 = 1 1 1
	//
	// col2 completely inactive.
	// =========================================================

	task drive_stream_active_cols_2;
	begin

		wait(o_feeder_start == 1'b1);


		// =====================================================
		// Cycle 0
		// =====================================================

		@(negedge i_clk);

		clear_stream();


		i_ifm_data[0*DATA_W +: DATA_W] = 8'sd1;
		i_ifm_valid[0] = 1'b1;
		i_ifm_first[0] = 1'b1;


		i_wgt_data[0*DATA_W +: DATA_W] = 8'sd1;
		i_wgt_valid[0] = 1'b1;
		i_wgt_first[0] = 1'b1;


		// =====================================================
		// Cycle 1
		// =====================================================

		@(negedge i_clk);

		clear_stream();


		i_ifm_data[0*DATA_W +: DATA_W] = 8'sd2;
		i_ifm_valid[0] = 1'b1;


		i_ifm_data[1*DATA_W +: DATA_W] = 8'sd2;
		i_ifm_valid[1] = 1'b1;
		i_ifm_first[1] = 1'b1;


		i_wgt_data[0*DATA_W +: DATA_W] = 8'sd0;
		i_wgt_valid[0] = 1'b1;


		i_wgt_data[1*DATA_W +: DATA_W] = 8'sd1;
		i_wgt_valid[1] = 1'b1;
		i_wgt_first[1] = 1'b1;


		// =====================================================
		// Cycle 2
		// =====================================================

		@(negedge i_clk);

		clear_stream();


		i_ifm_data[0*DATA_W +: DATA_W] = 8'sd3;
		i_ifm_valid[0] = 1'b1;
		i_ifm_last[0] = 1'b1;


		i_ifm_data[1*DATA_W +: DATA_W] = 8'sd3;
		i_ifm_valid[1] = 1'b1;


		i_ifm_data[2*DATA_W +: DATA_W] = 8'sd3;
		i_ifm_valid[2] = 1'b1;
		i_ifm_first[2] = 1'b1;


		i_wgt_data[0*DATA_W +: DATA_W] = -8'sd1;
		i_wgt_valid[0] = 1'b1;
		i_wgt_last[0] = 1'b1;


		i_wgt_data[1*DATA_W +: DATA_W] = 8'sd1;
		i_wgt_valid[1] = 1'b1;


		// col2 remains invalid


		// =====================================================
		// Cycle 3
		// =====================================================

		@(negedge i_clk);

		clear_stream();


		i_ifm_data[1*DATA_W +: DATA_W] = 8'sd4;
		i_ifm_valid[1] = 1'b1;
		i_ifm_last[1] = 1'b1;


		i_ifm_data[2*DATA_W +: DATA_W] = 8'sd4;
		i_ifm_valid[2] = 1'b1;


		i_wgt_data[1*DATA_W +: DATA_W] = 8'sd1;
		i_wgt_valid[1] = 1'b1;
		i_wgt_last[1] = 1'b1;


		// =====================================================
		// Cycle 4
		// =====================================================

		@(negedge i_clk);

		clear_stream();


		i_ifm_data[2*DATA_W +: DATA_W] = 8'sd5;
		i_ifm_valid[2] = 1'b1;
		i_ifm_last[2] = 1'b1;


		// =====================================================
		// Idle
		// =====================================================

		@(negedge i_clk);

		clear_stream();

	end
	endtask


	// ---------
	// Check one result
	//
	// Tile is assumed row-major:
	//
	// index = row*ARRAY_COLS + col
	// ---------
	task check_tile;

		input integer row;
		input integer col;
		input integer expected;

		integer index;
		reg signed [ACC_W-1:0] actual;

	begin
		index = row*ARRAY_COLS + col;

		actual =
			$signed(
				o_tile[
					index*ACC_W
					+: ACC_W
				]
			);


		if(actual !== expected) begin

			$display(
				"[FAIL] C[%0d][%0d] = %0d, expected = %0d",
				row,
				col,
				actual,
				expected
			);

			errors = errors + 1;

		end else begin

			$display(
				"[PASS] C[%0d][%0d] = %0d",
				row,
				col,
				actual
			);

		end

	end
	endtask


	// =========================================================
	// TC1
	// active_cols = 3
	// =========================================================

	task test_active_cols_3;

	begin
		$display("");
		$display("============================================");
		$display(" TC1 : CNN COMPUTE CORE - ACTIVE_COLS = 3");
		$display("============================================");


		reset_dut();


		start_tile(3);


		drive_stream_active_cols_3();


		// Wait result
		timeout_cnt = 0;

		while(
			(o_tile_valid !== 1'b1) &&
			(timeout_cnt < 200)
		) begin

			@(posedge i_clk);

			timeout_cnt = timeout_cnt + 1;

		end


		if(timeout_cnt >= 200) begin

			$display(
				"[FAIL] Timeout waiting for o_tile_valid"
			);

			errors = errors + 1;

		end else begin

			#1;

			$display("");
			$display("Checking output tile...");


			// Row 0
			check_tile(0,0,-2);
			check_tile(0,1, 6);
			check_tile(0,2, 0);


			// Row 1
			check_tile(1,0,-2);
			check_tile(1,1, 9);
			check_tile(1,2, 0);


			// Row 2
			check_tile(2,0,-2);
			check_tile(2,1,12);
			check_tile(2,2, 0);

		end


		// Wait controller done
		timeout_cnt = 0;

		while(
			(o_done !== 1'b1) &&
			(timeout_cnt < 100)
		) begin

			@(posedge i_clk);

			timeout_cnt = timeout_cnt + 1;

		end


		if(timeout_cnt >= 100) begin

			$display(
				"[FAIL] Timeout waiting for o_done"
			);

			errors = errors + 1;

		end else begin

			$display(
				"[PASS] Controller DONE"
			);

		end


		// Release start protocol completely
		repeat(3)
			@(posedge i_clk);

	end
	endtask


	// =========================================================
	// TC2
	// active_cols = 2
	//
	// Expected active tile:
	//
	// [-2  6]
	// [-2  9]
	// [-2 12]
	//
	// Physical col2 should remain cleared in PSM.
	// =========================================================

	task test_active_cols_2;

	begin
		$display("");
		$display("============================================");
		$display(" TC2 : CNN COMPUTE CORE - ACTIVE_COLS = 2");
		$display("============================================");


		reset_dut();


		start_tile(2);


		drive_stream_active_cols_2();


		timeout_cnt = 0;


		while(
			(o_tile_valid !== 1'b1) &&
			(timeout_cnt < 200)
		) begin

			@(posedge i_clk);

			timeout_cnt = timeout_cnt + 1;

		end


		if(timeout_cnt >= 200) begin

			$display(
				"[FAIL] Timeout waiting for o_tile_valid"
			);

			errors = errors + 1;

		end else begin

			#1;

			$display("");
			$display("Checking active columns...");


			check_tile(0,0,-2);
			check_tile(0,1, 6);

			check_tile(1,0,-2);
			check_tile(1,1, 9);

			check_tile(2,0,-2);
			check_tile(2,1,12);


			// Inactive physical column
			check_tile(0,2,0);
			check_tile(1,2,0);
			check_tile(2,2,0);

		end


		timeout_cnt = 0;


		while(
			(o_done !== 1'b1) &&
			(timeout_cnt < 100)
		) begin

			@(posedge i_clk);

			timeout_cnt = timeout_cnt + 1;

		end


		if(timeout_cnt >= 100) begin

			$display(
				"[FAIL] Timeout waiting for o_done"
			);

			errors = errors + 1;

		end else begin

			$display(
				"[PASS] Controller DONE"
			);

		end


		repeat(3)
			@(posedge i_clk);

	end
	endtask


	// ---------
	// Debug monitor
	// ---------
	always @(posedge i_clk) begin

		if(!i_reset) begin

			$display(
				"T=%0t state=%0d start=%b feeder_start=%b busy=%b done=%b tile_valid=%b | act_v=%b wgt_v=%b",
				$time,
				o_state_dbg,

				i_start,
				o_feeder_start,

				o_busy,
				o_done,

				o_tile_valid,

				i_ifm_valid,
				i_wgt_valid
			);

		end

	end


	// ---------
	// Main
	// ---------
	initial begin

		errors = 0;

		i_reset = 1'b0;

		i_start = 1'b0;

		i_active_cols = 0;

		clear_stream();


		// ---------
		// Run tests
		// ---------
		test_active_cols_3();

		test_active_cols_2();


		// ---------
		// Summary
		// ---------
		$display("");
		$display("============================================");
		$display(" CNN COMPUTE CORE TEST SUMMARY");
		$display("============================================");


		if(errors == 0) begin

			$display(
				" RESULT : ALL TESTS PASSED"
			);

		end else begin

			$display(
				" RESULT : FAILED (%0d errors)",
				errors
			);

		end


		$display("============================================");


		repeat(5)
			@(posedge i_clk);

		$finish;

	end


	// ---------
	// Global timeout
	// ---------
	initial begin

		#100000;

		$display("");
		$display(
			"[TIMEOUT] tb_cnn_compute_core"
		);

		$finish;

	end

endmodule