module conv1d_ifmap_feeder_top #(
	parameter DATA_W = 8,
	parameter ADDR_W = 16,
	parameter CFG_W = 16,
	parameter ARRAY_ROWS = 3,
	parameter ROW_W = (ARRAY_ROWS <= 1) ? 1 : $clog2(ARRAY_ROWS + 1),
	parameter DEPTH = 16,
	parameter K_W = 16
)(
	input wire i_clk,
	input wire i_reset,
	
	// Conv1D load control
	input wire i_load_start,
	input wire [ADDR_W-1:0] i_base_addr,
	input wire [CFG_W-1:0] i_k_len,
	input wire [ADDR_W-1:0] i_window_step,
	input wire [ROW_W-1:0] i_active_rows,
	
	output wire o_load_busy,
	output wire o_load_done,
	
	// SRAM read interface
	output wire o_sram_rd_en,
	output wire [ADDR_W-1:0] o_sram_rd_addr,
	input wire signed [DATA_W-1:0] i_sram_rd_data,
	input wire i_sram_rd_valid,
	
	// Feed control
	input wire i_feed_start,
	input wire i_feed_step,
	
	output wire o_ready,
	output wire o_feed_done,
	
	// Output to SA
	output wire [ARRAY_ROWS*DATA_W-1:0] o_data,
	output wire [ARRAY_ROWS-1:0] o_valid,
	output wire [ARRAY_ROWS-1:0] o_first,
	output wire [ARRAY_ROWS-1:0] o_last,
	
	// Debug
	output wire [ARRAY_ROWS-1:0] o_mem_empty,
	output wire [2:0] o_loader_state
);

	// Loader -> IFMAP feeder FIFO
	wire [ARRAY_ROWS-1:0] loader_wr_en;
	wire signed [ARRAY_ROWS*DATA_W-1:0] loader_wr_data;
	
	// Feeder status
	wire [ARRAY_ROWS-1:0] feeder_mem_full;
	wire feeder_busy;
	
	// K width adaptation
	wire [K_W-1:0] feeder_k_len;
	assign feeder_k_len = i_k_len;
	
	// CONV1D IFMAP LOADER
	conv1d_ifmap_loader #(
		.DATA_W(DATA_W),
		.ADDR_W(ADDR_W),
		.CFG_W(CFG_W),
		.ARRAY_ROWS(ARRAY_ROWS),
		.ROW_W(ROW_W)
	) u_loader(
		.i_clk(i_clk),
		.i_reset(i_reset),
		
		.i_start(i_load_start),
		
		.i_base_addr(i_base_addr),
		.i_k_len(i_k_len),
		.i_window_step(i_window_step),
		.i_active_rows(i_active_rows),
		
		.i_lane_full(feeder_mem_full),

		// SRAM read
		.o_sram_rd_en(o_sram_rd_en),
		.o_sram_rd_addr(o_sram_rd_addr),
		.i_sram_rd_data(i_sram_rd_data),
		.i_sram_rd_valid(i_sram_rd_valid),
		
		// Write into IFMAP feeder FIFOs
		.o_ifm_wr_en(loader_wr_en),
		.o_ifm_wr_data(loader_wr_data),
		
		.o_busy(o_load_busy),
		.o_done(o_load_done),
		.o_state_dbg(o_loader_state)
	);
	
	// Generic IFMAP feeder
	ifmap_feeder #(
		.DATA_W(DATA_W),
		.DEPTH(DEPTH),
		.K_W(K_W),
		.ARRAY_ROWS(ARRAY_ROWS),
		.ROW_W(ROW_W)
	) u_ifmap_feeder(
		.i_clk(i_clk),
		.i_reset(i_reset),
		
		// FIFO write from loader
		.i_wr_en(loader_wr_en),
		.i_wr_data(loader_wr_data),
		
		// Tile / feed control
		.i_start_tile(i_feed_start),
		.i_k_len(feeder_k_len),
		.i_active_rows(i_active_rows),
		.i_feed_step(i_feed_step),
		
		// SA stream
		.o_data(o_data),
		.o_valid(o_valid),
		.o_first(o_first),
		.o_last(o_last),
		
		// Status
		.o_ready(o_ready),
		.o_busy(feeder_busy),
		.o_done(o_feed_done),
		.o_mem_full(feeder_mem_full),
		.o_mem_empty(o_mem_empty)
	);

endmodule