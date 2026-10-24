`timescale 1ns/1ps

module tb_conv1d_tile_top;

	// =====================================================
	// PARAMETERS
	// =====================================================
	localparam DATA_W = 8;
	localparam ACC_W = 32;
	
	localparam ADDR_W = 6;
	localparam CFG_W = 16;
	
	localparam IFMAP_MEM_DEPTH = 64;
	localparam WGT_MEM_DEPTH = 64;
	
	localparam FEEDER_DEPTH = 16;
	localparam CNT_W = 5;
	localparam K_W = 16;
	
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
	// TILE CONTROL
	// =====================================================
	reg i_start;
	
	reg [CFG_W-1:0] i_k_len;
	reg [ADDR_W-1:0] i_window_step;
	
	reg [ROW_W-1:0] i_active_rows;
	reg [COL_W-1:0] i_active_cols;
	
	reg [ADDR_W-1:0] i_ifmap_base_addr;
	reg [ADDR_W-1:0] i_wgt_base_addr;
	
	
	// =====================================================
	// IFMAP SRAM WRITE PORT
	// =====================================================
	reg i_ifmap_mem_wr_en;
	reg [ADDR_W-1:0] i_ifmap_mem_wr_addr;
	reg signed [DATA_W-1:0] i_ifmap_mem_wr_data;
	reg i_ifmap_mem_swap;
	
	
	// =====================================================
	// WEIGHT SRAM WRITE PORT
	// =====================================================
	reg i_wgt_mem_wr_en;
	reg [ADDR_W-1:0] i_wgt_mem_wr_addr;
	reg signed [DATA_W-1:0] i_wgt_mem_wr_data;
	reg i_wgt_mem_swap;
	
	
	// =====================================================
	// RESULT
	// =====================================================
	wire [ARRAY_ROWS*ARRAY_COLS*ACC_W-1:0] o_tile;
	wire o_tile_valid;
    wire [71:0] post_tile;
    wire post_valid;
    integer post_checks=0;
    integer pr,pc;
    reg signed [63:0] post_expected;
	
	
	// =====================================================
	// STATUS
	// =====================================================
	wire o_busy;
	wire o_done;
	
	wire [2:0] o_state_dbg;
	wire [2:0] o_core_state_dbg;
	
	wire [2:0] o_ifmap_loader_state;
	wire [2:0] o_wgt_loader_state;
	
	
	// =====================================================
	// MEMORY DEBUG
	// =====================================================
	wire o_ifmap_read_bank;
	wire o_ifmap_write_bank;
	
	wire o_wgt_read_bank;
	wire o_wgt_write_bank;
	
	
	// =====================================================
	// TEST VARIABLES
	// =====================================================
	integer errors;
	integer timeout_cnt;
	
	
	// =====================================================
	// DUT
	// =====================================================
	conv1d_tile_top #(
		.POST_ENABLE(1),
		.DATA_W(DATA_W),
		.ACC_W(ACC_W),
		
		.ADDR_W(ADDR_W),
		.CFG_W(CFG_W),
		
		.IFMAP_MEM_DEPTH(IFMAP_MEM_DEPTH),
		.WGT_MEM_DEPTH(WGT_MEM_DEPTH),
		
		.FEEDER_DEPTH(FEEDER_DEPTH),
		.CNT_W(CNT_W),
		.K_W(K_W),
		
		.ARRAY_ROWS(ARRAY_ROWS),
		.ARRAY_COLS(ARRAY_COLS),
		
		.ROW_W(ROW_W),
		.COL_W(COL_W)
	) dut(
		.i_clk(i_clk),
		.i_reset(i_reset),
		
		// Tile control
		.i_start(i_start),
		.i_k_len(i_k_len),
		.i_window_step(i_window_step),
		.i_active_rows(i_active_rows),
		.i_active_cols(i_active_cols),
		.i_ifmap_base_addr(i_ifmap_base_addr),
		.i_wgt_base_addr(i_wgt_base_addr),
		
		// IFMAP SRAM write
		.i_ifmap_mem_wr_en(i_ifmap_mem_wr_en),
		.i_ifmap_mem_wr_addr(i_ifmap_mem_wr_addr),
		.i_ifmap_mem_wr_data(i_ifmap_mem_wr_data),
		.i_ifmap_mem_swap(i_ifmap_mem_swap),
		
		// Weight SRAM write
		.i_wgt_mem_wr_en(i_wgt_mem_wr_en),
		.i_wgt_mem_wr_addr(i_wgt_mem_wr_addr),
		.i_wgt_mem_wr_data(i_wgt_mem_wr_data),
		.i_wgt_mem_swap(i_wgt_mem_swap),
		
		// Result
		.o_tile(o_tile),
		.o_tile_valid(o_tile_valid),
        .i_post_bias({3{32'd2}}),.i_post_shift({3{8'd1}}),.i_post_relu(1'b1),
        .o_post_tile(post_tile),.o_post_valid(post_valid),
		
		// Status
		.o_busy(o_busy),
		.o_done(o_done),
		.o_state_dbg(o_state_dbg),
		.o_core_state_dbg(o_core_state_dbg),
		.o_ifmap_loader_state(o_ifmap_loader_state),
		.o_wgt_loader_state(o_wgt_loader_state),
		
		// Memory debug
		.o_ifmap_read_bank(o_ifmap_read_bank),
		.o_ifmap_write_bank(o_ifmap_write_bank),
		.o_wgt_read_bank(o_wgt_read_bank),
		.o_wgt_write_bank(o_wgt_write_bank)
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
		
		i_k_len = {CFG_W{1'b0}};
		i_window_step = {ADDR_W{1'b0}};
		
		i_active_rows = {ROW_W{1'b0}};
		i_active_cols = {COL_W{1'b0}};
		
		i_ifmap_base_addr = {ADDR_W{1'b0}};
		i_wgt_base_addr = {ADDR_W{1'b0}};
		
		i_ifmap_mem_wr_en = 1'b0;
		i_ifmap_mem_wr_addr = {ADDR_W{1'b0}};
		i_ifmap_mem_wr_data = {DATA_W{1'b0}};
		i_ifmap_mem_swap = 1'b0;
		
		i_wgt_mem_wr_en = 1'b0;
		i_wgt_mem_wr_addr = {ADDR_W{1'b0}};
		i_wgt_mem_wr_data = {DATA_W{1'b0}};
		i_wgt_mem_swap = 1'b0;
		
		repeat(4) @(posedge i_clk);
		
		@(negedge i_clk);
		i_reset = 1'b0;
	end
	endtask
	
	
	// =====================================================
	// WRITE IFMAP SRAM
	// =====================================================
	task write_ifmap;
		input integer addr;
		input integer value;
	begin
		@(negedge i_clk);
		
		i_ifmap_mem_wr_en = 1'b1;
		i_ifmap_mem_wr_addr = addr;
		i_ifmap_mem_wr_data = value;
		
		@(posedge i_clk);
		@(negedge i_clk);
		
		i_ifmap_mem_wr_en = 1'b0;
	end
	endtask
	
	
	// =====================================================
	// WRITE WEIGHT SRAM
	// =====================================================
	task write_weight;
		input integer addr;
		input integer value;
	begin
		@(negedge i_clk);
		
		i_wgt_mem_wr_en = 1'b1;
		i_wgt_mem_wr_addr = addr;
		i_wgt_mem_wr_data = value;
		
		@(posedge i_clk);
		@(negedge i_clk);
		
		i_wgt_mem_wr_en = 1'b0;
	end
	endtask
	
	
	// =====================================================
	// SWAP IFMAP BANK
	// =====================================================
	task swap_ifmap;
	begin
		@(negedge i_clk);
		
		i_ifmap_mem_wr_en = 1'b0;
		i_ifmap_mem_swap = 1'b1;
		
		@(posedge i_clk);
		@(negedge i_clk);
		
		i_ifmap_mem_swap = 1'b0;
	end
	endtask
	
	
	// =====================================================
	// SWAP WEIGHT BANK
	// =====================================================
	task swap_weight;
	begin
		@(negedge i_clk);
		
		i_wgt_mem_wr_en = 1'b0;
		i_wgt_mem_swap = 1'b1;
		
		@(posedge i_clk);
		@(negedge i_clk);
		
		i_wgt_mem_swap = 1'b0;
	end
	endtask
	
	
	// =====================================================
	// START TILE
	// =====================================================
	task start_tile;
		input integer k_len;
		input integer window_step;
		input integer active_rows;
		input integer active_cols;
		input integer ifmap_base;
		input integer wgt_base;
	begin
		@(negedge i_clk);
		
		i_k_len = k_len;
		i_window_step = window_step;
		i_active_rows = active_rows;
		i_active_cols = active_cols;
		i_ifmap_base_addr = ifmap_base;
		i_wgt_base_addr = wgt_base;
		
		i_start = 1'b1;
		
		@(posedge i_clk);
		@(negedge i_clk);
		
		i_start = 1'b0;
	end
	endtask
	
	
	// =====================================================
	// LOAD COMMON IFMAP
	//
	// X = [1 2 3 4 5]
	//
	// Hardware có thể tạo:
	//
	// row0 = [1 2 3]
	// row1 = [2 3 4]
	// row2 = [3 4 5]
	//
	// active_rows quyết định dùng bao nhiêu window đầu.
	// =====================================================
	task load_ifmap;
	begin
		write_ifmap(0, 1);
		write_ifmap(1, 2);
		write_ifmap(2, 3);
		write_ifmap(3, 4);
		write_ifmap(4, 5);
	end
	endtask
	
	
	// =====================================================
	// LOAD COMMON WEIGHTS
	//
	// W0 = [ 1  0 -1]
	// W1 = [ 1  1  1]
	// W2 = [-1  2 -1]
	//
	// Packed Cout-major:
	//
	// addr 0..2 -> W0
	// addr 3..5 -> W1
	// addr 6..8 -> W2
	// =====================================================
	task load_weights;
	begin
		// W0
		write_weight(0,  1);
		write_weight(1,  0);
		write_weight(2, -1);
		
		// W1
		write_weight(3,  1);
		write_weight(4,  1);
		write_weight(5,  1);
		
		// W2
		write_weight(6, -1);
		write_weight(7,  2);
		write_weight(8, -1);
	end
	endtask
	
	
	// =====================================================
	// LOAD WEIGHTS WITH GARBAGE COL2
	//
	// Dùng cho active_cols = 2.
	//
	// W2 chứa garbage để kiểm tra hardware thật sự
	// không sử dụng inactive column.
	// =====================================================
	task load_weights_col2_garbage;
	begin
		// W0
		write_weight(0,  1);
		write_weight(1,  0);
		write_weight(2, -1);
		
		// W1
		write_weight(3,  1);
		write_weight(4,  1);
		write_weight(5,  1);
		
		// Inactive W2 garbage
		write_weight(6,  99);
		write_weight(7, -88);
		write_weight(8,  77);
	end
	endtask
	
	
	// =====================================================
	// CHECK ONE TILE VALUE
	//
	// Physical tile layout:
	//
	// index = row*ARRAY_COLS + col
	// =====================================================
	task check_tile;
		input integer row;
		input integer col;
		input integer expected;
		
		integer index;
		reg signed [ACC_W-1:0] actual;
	begin
		index = row*ARRAY_COLS + col;
		actual = $signed(o_tile[index*ACC_W +: ACC_W]);
		
		if(actual !== expected) begin
			$display("[FAIL] C[%0d][%0d] = %0d, expected = %0d", row, col, actual, expected);
			errors = errors + 1;
		end else begin
			$display("[PASS] C[%0d][%0d] = %0d", row, col, actual);
		end
	end
	endtask
	
	
	// =====================================================
	// WAIT TILE VALID
	// =====================================================
	task wait_tile_valid;
	begin
		timeout_cnt = 0;
		
		while((o_tile_valid !== 1'b1) && (timeout_cnt < 500)) begin
			@(posedge i_clk);
			timeout_cnt = timeout_cnt + 1;
		end
		
		if(timeout_cnt >= 500) begin
			$display("[FAIL] Timeout waiting for o_tile_valid");
			errors = errors + 1;
		end
	end
	endtask
	
	
	// =====================================================
	// WAIT DONE
	// =====================================================
	task wait_done;
	begin
		timeout_cnt = 0;
		
		while((o_done !== 1'b1) && (timeout_cnt < 200)) begin
			@(posedge i_clk);
			timeout_cnt = timeout_cnt + 1;
		end
		
		if(timeout_cnt >= 200) begin
			$display("[FAIL] Timeout waiting for o_done");
			errors = errors + 1;
		end else begin
			$display("[PASS] Tile controller DONE");
		end
	end
	endtask
	
	
	// =====================================================
	// TC1
	//
	// active_rows = 3
	// active_cols = 3
	//
	// Expected:
	//
	// [-2   6   0]
	// [-2   9   0]
	// [-2  12   0]
	// =====================================================
	task test_full_3x3;
	begin
		$display("");
		$display("==============================================");
		$display(" TC1 : ACTIVE_ROWS = 3, ACTIVE_COLS = 3");
		$display("==============================================");
		
		reset_dut();
		
		load_ifmap();
		load_weights();
		
		swap_ifmap();
		swap_weight();
		
		start_tile(3, 1, 3, 3, 0, 0);
		
		wait_tile_valid();
		
		if(o_tile_valid) begin
			#1;
			
			check_tile(0,0,-2);
			check_tile(0,1, 6);
			check_tile(0,2, 0);
			
			check_tile(1,0,-2);
			check_tile(1,1, 9);
			check_tile(1,2, 0);
			
			check_tile(2,0,-2);
			check_tile(2,1,12);
			check_tile(2,2, 0);
		end
		
		wait_done();
		
		repeat(3) @(posedge i_clk);
	end
	endtask
	
	
	// =====================================================
	// TC2
	//
	// active_rows = 2
	// active_cols = 3
	//
	// Loader chỉ tạo:
	//
	// row0 = [1 2 3]
	// row1 = [2 3 4]
	//
	// row2 inactive
	//
	// Expected:
	//
	// [-2   6   0]
	// [-2   9   0]
	// [ 0   0   0]
	// =====================================================
	task test_partial_rows_2x3;
	begin
		$display("");
		$display("==============================================");
		$display(" TC2 : ACTIVE_ROWS = 2, ACTIVE_COLS = 3");
		$display("==============================================");
		
		reset_dut();
		
		load_ifmap();
		load_weights();
		
		swap_ifmap();
		swap_weight();
		
		start_tile(3, 1, 2, 3, 0, 0);
		
		wait_tile_valid();
		
		if(o_tile_valid) begin
			#1;
			
			check_tile(0,0,-2);
			check_tile(0,1, 6);
			check_tile(0,2, 0);
			
			check_tile(1,0,-2);
			check_tile(1,1, 9);
			check_tile(1,2, 0);
			
			check_tile(2,0, 0);
			check_tile(2,1, 0);
			check_tile(2,2, 0);
		end
		
		wait_done();
		
		repeat(3) @(posedge i_clk);
	end
	endtask
	
	
	// =====================================================
	// TC3
	//
	// active_rows = 2
	// active_cols = 2
	//
	// Row2 inactive.
	// Col2 inactive và chứa garbage.
	//
	// Expected:
	//
	// [-2   6   0]
	// [-2   9   0]
	// [ 0   0   0]
	// =====================================================
	task test_partial_2x2;
	begin
		$display("");
		$display("==============================================");
		$display(" TC3 : ACTIVE_ROWS = 2, ACTIVE_COLS = 2");
		$display("==============================================");
		
		reset_dut();
		
		load_ifmap();
		load_weights_col2_garbage();
		
		swap_ifmap();
		swap_weight();
		
		start_tile(3, 1, 2, 2, 0, 0);
		
		wait_tile_valid();
		
		if(o_tile_valid) begin
			#1;
			
			check_tile(0,0,-2);
			check_tile(0,1, 6);
			check_tile(0,2, 0);
			
			check_tile(1,0,-2);
			check_tile(1,1, 9);
			check_tile(1,2, 0);
			
			check_tile(2,0, 0);
			check_tile(2,1, 0);
			check_tile(2,2, 0);
		end
		
		wait_done();
		
		repeat(3) @(posedge i_clk);
	end
	endtask
	
	
	// =====================================================
	// TC4
	//
	// active_rows = 1
	// active_cols = 1
	//
	// Chỉ compute:
	//
	// [1 2 3] dot [1 0 -1] = -2
	//
	// Expected:
	//
	// [-2   0   0]
	// [ 0   0   0]
	// [ 0   0   0]
	// =====================================================
	task test_corner_1x1;
	begin
		$display("");
		$display("==============================================");
		$display(" TC4 : ACTIVE_ROWS = 1, ACTIVE_COLS = 1");
		$display("==============================================");
		
		reset_dut();
		
		load_ifmap();
		
		// W0 active
		write_weight(0,  1);
		write_weight(1,  0);
		write_weight(2, -1);
		
		// Garbage sau active kernel
		write_weight(3,  77);
		write_weight(4, -66);
		write_weight(5,  55);
		write_weight(6, -44);
		write_weight(7,  33);
		write_weight(8, -22);
		
		swap_ifmap();
		swap_weight();
		
		start_tile(3, 1, 1, 1, 0, 0);
		
		wait_tile_valid();
		
		if(o_tile_valid) begin
			#1;
			
			check_tile(0,0,-2);
			check_tile(0,1, 0);
			check_tile(0,2, 0);
			
			check_tile(1,0, 0);
			check_tile(1,1, 0);
			check_tile(1,2, 0);
			
			check_tile(2,0, 0);
			check_tile(2,1, 0);
			check_tile(2,2, 0);
		end
		
		wait_done();
		
		repeat(3) @(posedge i_clk);
	end
	endtask
	
	
	// =====================================================
	// DEBUG MONITOR
	// =====================================================
	always @(posedge i_clk) begin
		if(!i_reset) begin
			$display("T=%0t tile=%0d core=%0d busy=%b done=%b valid=%b | ar=%0d ac=%0d | ifm_loader=%0d wgt_loader=%0d",
				$time,
				o_state_dbg,
				o_core_state_dbg,
				o_busy,
				o_done,
				o_tile_valid,
				i_active_rows,
				i_active_cols,
				o_ifmap_loader_state,
				o_wgt_loader_state
			);
		end
	end
	
	
	// =====================================================
	// MAIN
	// =====================================================
	initial begin
		errors = 0;
		
		i_reset = 1'b0;
		i_start = 1'b0;
		
		i_k_len = {CFG_W{1'b0}};
		i_window_step = {ADDR_W{1'b0}};
		
		i_active_rows = {ROW_W{1'b0}};
		i_active_cols = {COL_W{1'b0}};
		
		i_ifmap_base_addr = {ADDR_W{1'b0}};
		i_wgt_base_addr = {ADDR_W{1'b0}};
		
		i_ifmap_mem_wr_en = 1'b0;
		i_ifmap_mem_wr_addr = {ADDR_W{1'b0}};
		i_ifmap_mem_wr_data = {DATA_W{1'b0}};
		i_ifmap_mem_swap = 1'b0;
		
		i_wgt_mem_wr_en = 1'b0;
		i_wgt_mem_wr_addr = {ADDR_W{1'b0}};
		i_wgt_mem_wr_data = {DATA_W{1'b0}};
		i_wgt_mem_swap = 1'b0;
		
		// Run tests
		test_full_3x3();
		test_partial_rows_2x3();
		test_partial_2x2();
		test_corner_1x1();
		
		// Summary
		$display("");
		$display("==============================================");
		$display(" CONV1D TILE ACTIVE ROW/COL TEST SUMMARY");
		$display("==============================================");
		
		if(errors == 0) begin
			if(post_checks!=4) $fatal(1,"Expected one post output for each of 4 tiles, got %d",post_checks);
            $display(" RESULT : ALL TESTS PASSED (including post processing)");
		end else begin
			$display(" RESULT : FAILED (%0d errors)", errors);
		end
		
		$display("==============================================");
		
		repeat(5) @(posedge i_clk);
		$finish;
	end
	
	
	// =====================================================
	// GLOBAL TIMEOUT
	// =====================================================
	initial begin
		#1000000;
		
		$display("");
		$display("[TIMEOUT] tb_conv1d_tile_top");
		
		$finish;
	end

    always @(negedge i_clk) if(post_valid) begin
        post_checks=post_checks+1;
        for(pr=0;pr<3;pr=pr+1) for(pc=0;pc<3;pc=pc+1) begin
            post_expected=$signed(o_tile[(pr*3+pc)*32 +: 32]);
            post_expected=(post_expected+3) >>> 1;
            if(post_expected<0) post_expected=0;
            if(post_expected>127) post_expected=127;
            if(pr>=i_active_rows || pc>=i_active_cols) post_expected=0;
            if($signed(post_tile[(pr*3+pc)*8 +: 8])!==post_expected)
                $fatal(1,"Integrated post mismatch");
        end
    end
endmodule
