// conv1d_tile_top
// │
// ├── tensor_pingpong_buffer        <- raw X
// │
// ├── conv1d_ifmap_feeder_top
// │   ├── conv1d_ifmap_loader
// │   │   └── AGU
// │   └── ifmap_feeder
// │
// ├── tensor_pingpong_buffer        <- packed W
// │
// ├── conv1d_weight_feeder_top
// │   ├── conv1d_weight_loader
// │   │   └── AGU
// │   └── weight_feeder
// │
// └── cnn_compute_core
//     ├── cnn_control_fsm
//     ├── pe_array
//     └── partial_sum_manager

module conv1d_tile_top #(
	parameter DATA_W = 8,
	parameter ACC_W = 32,
	
	parameter ADDR_W = 16,
	parameter CFG_W = 16,
	
	parameter IFMAP_MEM_DEPTH = 1024,
	parameter WGT_MEM_DEPTH = 1024,
	
	parameter FEEDER_DEPTH = 1024,
	parameter K_W = 16,
	
	parameter ARRAY_ROWS = 3,
	parameter ARRAY_COLS = 3,
	
	parameter ROW_W = (ARRAY_ROWS <= 1) ? 1 : $clog2(ARRAY_ROWS + 1),
	parameter COL_W = (ARRAY_COLS <= 1) ? 1 : $clog2(ARRAY_COLS + 1)
)(
	input wire i_clk,
	input wire i_reset,
	
	// =====================================================
	// TILE CONTROL
	// =====================================================
	input wire i_start,
	
	// K = Kw * Cin
	input wire [CFG_W-1:0] i_k_len,
	
	// stride * Cin
	input wire [ADDR_W-1:0] i_window_step,
	
	// Dynamic physical tile size
	input wire [ROW_W-1:0] i_active_rows,
	input wire [COL_W-1:0] i_active_cols,
	
	// Base address trong SRAM
	input wire [ADDR_W-1:0] i_ifmap_base_addr,
	input wire [ADDR_W-1:0] i_wgt_base_addr,
	
	// =====================================================
	// IFMAP SRAM LOAD PORT
	// CPU / TB ghi raw input tensor vào write bank
	// trước khi accelerator chạy
	// =====================================================
	input wire i_ifmap_mem_wr_en,
	input wire [ADDR_W-1:0] i_ifmap_mem_wr_addr,
	input wire signed [DATA_W-1:0] i_ifmap_mem_wr_data,
	input wire i_ifmap_mem_swap,
	
	// =====================================================
	// WEIGHT SRAM LOAD PORT
	// =====================================================
	input wire i_wgt_mem_wr_en,
	input wire [ADDR_W-1:0] i_wgt_mem_wr_addr,
	input wire signed [DATA_W-1:0] i_wgt_mem_wr_data,
	input wire i_wgt_mem_swap,
	
	// =====================================================
	// RESULT
	// =====================================================
	output wire [ARRAY_ROWS*ARRAY_COLS*ACC_W-1:0] o_tile,
	output wire o_tile_valid,
	
	// =====================================================
	// STATUS
	// =====================================================
	output wire o_busy,
	output wire o_done,
	
	output wire [2:0] o_state_dbg,
	output wire [2:0] o_core_state_dbg,
	
	output wire [2:0] o_ifmap_loader_state,
	output wire [2:0] o_wgt_loader_state,
	
	// =====================================================
	// MEMORY DEBUG
	// =====================================================
	output wire o_ifmap_read_bank,
	output wire o_ifmap_write_bank,
	
	output wire o_wgt_read_bank,
	output wire o_wgt_write_bank
);

	// =====================================================
	// TILE FSM STATES
	// =====================================================
	localparam S_IDLE       = 3'd0;
	localparam S_LOAD_START = 3'd1;
	localparam S_LOAD_WAIT  = 3'd2;
	localparam S_CORE_START = 3'd3;
	localparam S_COMPUTE    = 3'd4;
	localparam S_DONE       = 3'd5;
	
	reg [2:0] state_q;
	
	assign o_state_dbg = state_q;
	assign o_busy = (state_q != S_IDLE) && (state_q != S_DONE);
	assign o_done = (state_q == S_DONE);
	
	// =====================================================
	// LATCH TILE CONFIG
	// Freeze toàn bộ config khi nhận i_start
	// =====================================================
	reg [CFG_W-1:0] k_len_q;
	reg [ADDR_W-1:0] window_step_q;
	
	reg [ROW_W-1:0] active_rows_q;
	reg [COL_W-1:0] active_cols_q;
	
	reg [ADDR_W-1:0] ifmap_base_addr_q;
	reg [ADDR_W-1:0] wgt_base_addr_q;
	
	always @(posedge i_clk or posedge i_reset) begin
		if(i_reset) begin
			k_len_q <= {CFG_W{1'b0}};
			window_step_q <= {ADDR_W{1'b0}};
			
			active_rows_q <= {ROW_W{1'b0}};
			active_cols_q <= {COL_W{1'b0}};
			
			ifmap_base_addr_q <= {ADDR_W{1'b0}};
			wgt_base_addr_q <= {ADDR_W{1'b0}};
		end else begin
			if(i_start && state_q == S_IDLE) begin
				k_len_q <= i_k_len;
				window_step_q <= i_window_step;
				
				active_rows_q <= i_active_rows;
				active_cols_q <= i_active_cols;
				
				ifmap_base_addr_q <= i_ifmap_base_addr;
				wgt_base_addr_q <= i_wgt_base_addr;
			end
		end
	end
	
	// =====================================================
	// IFMAP SRAM
	// =====================================================
	wire ifmap_sram_rd_en;
	wire [ADDR_W-1:0] ifmap_sram_rd_addr;
	
	wire signed [DATA_W-1:0] ifmap_sram_rd_data;
	wire ifmap_sram_rd_valid;
	
	tensor_pingpong_buffer #(
		.DATA_W(DATA_W),
		.DEPTH(IFMAP_MEM_DEPTH),
		.ADDR_W(ADDR_W)
	) u_ifmap_mem(
		.i_clk(i_clk),
		.i_reset(i_reset),
		
		.i_swap(i_ifmap_mem_swap),
		
		.i_wr_en(i_ifmap_mem_wr_en),
		.i_wr_addr(i_ifmap_mem_wr_addr),
		.i_wr_data(i_ifmap_mem_wr_data),
		
		.i_rd_en(ifmap_sram_rd_en),
		.i_rd_addr(ifmap_sram_rd_addr),
		
		.o_rd_data(ifmap_sram_rd_data),
		.o_rd_valid(ifmap_sram_rd_valid),
		
		.o_read_bank(o_ifmap_read_bank),
		.o_write_bank(o_ifmap_write_bank)
	);
	
	// =====================================================
	// WEIGHT SRAM
	// =====================================================
	wire wgt_sram_rd_en;
	wire [ADDR_W-1:0] wgt_sram_rd_addr;
	
	wire signed [DATA_W-1:0] wgt_sram_rd_data;
	wire wgt_sram_rd_valid;
	
	tensor_pingpong_buffer #(
		.DATA_W(DATA_W),
		.DEPTH(WGT_MEM_DEPTH),
		.ADDR_W(ADDR_W)
	) u_wgt_mem(
		.i_clk(i_clk),
		.i_reset(i_reset),
		
		.i_swap(i_wgt_mem_swap),
		
		.i_wr_en(i_wgt_mem_wr_en),
		.i_wr_addr(i_wgt_mem_wr_addr),
		.i_wr_data(i_wgt_mem_wr_data),
		
		.i_rd_en(wgt_sram_rd_en),
		.i_rd_addr(wgt_sram_rd_addr),
		
		.o_rd_data(wgt_sram_rd_data),
		.o_rd_valid(wgt_sram_rd_valid),
		
		.o_read_bank(o_wgt_read_bank),
		.o_write_bank(o_wgt_write_bank)
	);
	
	// =====================================================
	// TILE CONTROL SIGNALS
	// =====================================================
	wire ifmap_load_start;
	wire wgt_load_start;
	wire core_start;
	
	assign ifmap_load_start = (state_q == S_LOAD_START);
	assign wgt_load_start = (state_q == S_LOAD_START);
	assign core_start = (state_q == S_CORE_START);
	
	// =====================================================
	// IFMAP FEEDER TOP SIGNALS
	// =====================================================
	wire ifmap_load_busy;
	wire ifmap_load_done;
	
	wire ifmap_ready;
	wire ifmap_feed_done;
	
	wire [ARRAY_ROWS*DATA_W-1:0] ifmap_data;
	
	wire [ARRAY_ROWS-1:0] ifmap_valid;
	wire [ARRAY_ROWS-1:0] ifmap_first;
	wire [ARRAY_ROWS-1:0] ifmap_last;
	
	wire [ARRAY_ROWS-1:0] ifmap_mem_empty;
	
	// =====================================================
	// WEIGHT FEEDER TOP SIGNALS
	// =====================================================
	wire wgt_load_busy;
	wire wgt_load_done;
	
	wire wgt_ready;
	wire wgt_feed_done;
	
	wire [ARRAY_COLS*DATA_W-1:0] wgt_data;
	
	wire [ARRAY_COLS-1:0] wgt_valid;
	wire [ARRAY_COLS-1:0] wgt_first;
	wire [ARRAY_COLS-1:0] wgt_last;
	
	wire [ARRAY_COLS-1:0] wgt_mem_empty;
	
	// =====================================================
	// GLOBAL FEED LOCKSTEP
	// Chỉ khi cả IFMAP và Weight cùng sẵn sàng mới advance k
	// =====================================================
	wire global_feed_step;
	
	assign global_feed_step = ifmap_ready & wgt_ready;
	
	// =====================================================
	// COMPUTE CORE FEEDER START
	// Core controller quyết định khi nào bắt đầu feed
	// =====================================================
	wire core_feeder_start;
	
	// =====================================================
	// CONV1D IFMAP FEEDER TOP
	// =====================================================
	conv1d_ifmap_feeder_top #(
		.DATA_W(DATA_W),
		.ADDR_W(ADDR_W),
		.CFG_W(CFG_W),
		
		.ARRAY_ROWS(ARRAY_ROWS),
		.ROW_W(ROW_W),
		
		.DEPTH(FEEDER_DEPTH),
		.K_W(K_W)
	) u_ifmap_feeder_top(
		.i_clk(i_clk),
		.i_reset(i_reset),
		
		// Conv1D load
		.i_load_start(ifmap_load_start),
		
		.i_base_addr(ifmap_base_addr_q),
		.i_k_len(k_len_q),
		.i_window_step(window_step_q),
		.i_active_rows(active_rows_q),
		
		.o_load_busy(ifmap_load_busy),
		.o_load_done(ifmap_load_done),
		
		// SRAM
		.o_sram_rd_en(ifmap_sram_rd_en),
		.o_sram_rd_addr(ifmap_sram_rd_addr),
		
		.i_sram_rd_data(ifmap_sram_rd_data),
		.i_sram_rd_valid(ifmap_sram_rd_valid),
		
		// Feed
		.i_feed_start(core_feeder_start),
		.i_feed_step(global_feed_step),
		
		.o_ready(ifmap_ready),
		.o_feed_done(ifmap_feed_done),
		
		// SA stream
		.o_data(ifmap_data),
		.o_valid(ifmap_valid),
		.o_first(ifmap_first),
		.o_last(ifmap_last),
		
		.o_mem_empty(ifmap_mem_empty),
		.o_loader_state(o_ifmap_loader_state)
	);
	
	// =====================================================
	// CONV1D WEIGHT FEEDER TOP
	// =====================================================
	conv1d_weight_feeder_top #(
		.DATA_W(DATA_W),
		.ADDR_W(ADDR_W),
		.CFG_W(CFG_W),
		
		.ARRAY_COLS(ARRAY_COLS),
		
		.DEPTH(FEEDER_DEPTH),
		.K_W(K_W),
		
		.COL_W(COL_W)
	) u_wgt_feeder_top(
		.i_clk(i_clk),
		.i_reset(i_reset),
		
		// Weight load
		.i_load_start(wgt_load_start),
		
		.i_base_addr(wgt_base_addr_q),
		.i_k_len(k_len_q),
		.i_active_cols(active_cols_q),
		
		.o_load_busy(wgt_load_busy),
		.o_load_done(wgt_load_done),
		
		// SRAM
		.o_sram_rd_en(wgt_sram_rd_en),
		.o_sram_rd_addr(wgt_sram_rd_addr),
		
		.i_sram_rd_data(wgt_sram_rd_data),
		.i_sram_rd_valid(wgt_sram_rd_valid),
		
		// Feed
		.i_feed_start(core_feeder_start),
		.i_feed_step(global_feed_step),
		
		.o_ready(wgt_ready),
		.o_feed_done(wgt_feed_done),
		
		// SA stream
		.o_data(wgt_data),
		.o_valid(wgt_valid),
		.o_first(wgt_first),
		.o_last(wgt_last),
		
		.o_mem_empty(wgt_mem_empty),
		.o_loader_state(o_wgt_loader_state)
	);
	
	// =====================================================
	// CNN COMPUTE CORE
	// =====================================================
	wire core_busy;
	wire core_done;
	
	cnn_compute_core #(
		.DATA_W(DATA_W),
		.ACC_W(ACC_W),
		
		.ARRAY_ROWS(ARRAY_ROWS),
		.ARRAY_COLS(ARRAY_COLS),
		
		.ROW_W(ROW_W),
		.COL_W(COL_W)
	) u_compute_core(
		.i_clk(i_clk),
		.i_reset(i_reset),
		
		.i_start(core_start),
		
		.i_active_rows(active_rows_q),
		.i_active_cols(active_cols_q),
		
		// IFMAP stream
		.i_ifm_data(ifmap_data),
		.i_ifm_valid(ifmap_valid),
		.i_ifm_first(ifmap_first),
		.i_ifm_last(ifmap_last),
		
		// Weight stream
		.i_wgt_data(wgt_data),
		.i_wgt_valid(wgt_valid),
		.i_wgt_first(wgt_first),
		.i_wgt_last(wgt_last),
		
		// Start external feeders
		.o_feeder_start(core_feeder_start),
		
		// Result
		.o_tile(o_tile),
		.o_tile_valid(o_tile_valid),
		
		// Status
		.o_busy(core_busy),
		.o_done(core_done),
		.o_state_dbg(o_core_state_dbg)
	);
	
	// =====================================================
	// LOAD DONE EVENT LATCH
	//
	// Hai loader không nhất thiết done cùng cycle.
	//
	// IFMAP load time phụ thuộc active_rows * K.
	// Weight load time phụ thuộc active_cols * K.
	// =====================================================
	reg ifmap_load_done_seen;
	reg wgt_load_done_seen;
	
	always @(posedge i_clk or posedge i_reset) begin
		if(i_reset) begin
			ifmap_load_done_seen <= 1'b0;
			wgt_load_done_seen <= 1'b0;
		end else begin
			if(state_q == S_LOAD_START) begin
				ifmap_load_done_seen <= 1'b0;
				wgt_load_done_seen <= 1'b0;
			end else begin
				if(ifmap_load_done) ifmap_load_done_seen <= 1'b1;
				if(wgt_load_done) wgt_load_done_seen <= 1'b1;
			end
		end
	end
	
	// =====================================================
	// TILE FSM
	// =====================================================
	always @(posedge i_clk or posedge i_reset) begin
		if(i_reset) begin
			state_q <= S_IDLE;
		end else begin
			case(state_q)
				
				// -----------------------------------------
				// IDLE
				// -----------------------------------------
				S_IDLE: begin
					if(i_start) begin
						state_q <= S_LOAD_START;
					end
				end
				
				// -----------------------------------------
				// Pulse load_start cho cả hai loader
				// -----------------------------------------
				S_LOAD_START: begin
					state_q <= S_LOAD_WAIT;
				end
				
				// -----------------------------------------
				// Wait BOTH load paths
				// -----------------------------------------
				S_LOAD_WAIT: begin
					if((ifmap_load_done_seen || ifmap_load_done) && (wgt_load_done_seen || wgt_load_done)) begin
						state_q <= S_CORE_START;
					end
				end
				
				// -----------------------------------------
				// Pulse start cho compute core
				// -----------------------------------------
				S_CORE_START: begin
					state_q <= S_COMPUTE;
				end
				
				// -----------------------------------------
				// Compute + drain
				// -----------------------------------------
				S_COMPUTE: begin
					if(core_done) begin
						state_q <= S_DONE;
					end
				end
				
				// -----------------------------------------
				// DONE
				// -----------------------------------------
				S_DONE: begin
					if(!i_start) begin
						state_q <= S_IDLE;
					end
				end
				
				default: begin
					state_q <= S_IDLE;
				end
				
			endcase
		end
	end

endmodule
