module conv1d_weight_feeder_top#(
	parameter DATA_W = 8,
	parameter ADDR_W = 16,
	parameter CFG_W = 16,
	
	parameter ARRAY_COLS = 3,
	
	parameter DEPTH = 16,
	parameter CNT_W = 5,
	parameter K_W = 16,
	
	parameter COL_W = (ARRAY_COLS <= 1) ? 1 : $clog2(ARRAY_COLS+1)
)(
	input wire i_clk,
	input wire i_reset,
	
	// Weight load control
	//
	// weight SRAM
	// --> weight loader
	// --> weight feeders FIFO
	
	input wire i_load_start,
	
	input wire [ADDR_W-1:0] i_base_addr,
	
	// K = Kw * Cin (CONV1D)
	// K = Kw * Kh * Cin (CONV2D)
	input wire [CFG_W-1:0] i_k_len,
	
	input wire [COL_W-1:0] i_active_cols,
	
	output wire o_load_busy,
	output wire o_load_done,
	
	// Weight SRAM Read interface
	output wire o_sram_rd_en,
	output wire [ADDR_W-1:0] o_sram_rd_addr,
	
	input wire signed [DATA_W-1:0] i_sram_rd_data,
	input wire i_sram_rd_valid,
	
	// Feed control
	input wire i_feed_start,
	input wire i_feed_step, // global_feed_step = ifmap_ready & weight_ready
	
	output wire o_ready, 
	output wire o_feed_done,
	
	// output to SA
	output wire [ARRAY_COLS*DATA_W-1:0] o_data,
	output wire [ARRAY_COLS-1:0] o_valid,
	output wire [ARRAY_COLS-1:0] o_first,
	output wire [ARRAY_COLS-1:0] o_last,
	
	// debug
	output wire [ARRAY_COLS-1:0] o_mem_empty,
	output wire [2:0] o_loader_state
);

	// Loader --> weight feeder
	wire [ARRAY_COLS-1:0] loader_wr_en;
	wire signed [ARRAY_COLS*DATA_W-1:0] loader_wr_data;
	
	// K WIDTH ADAPTATION
	wire [K_W-1:0] feeder_k_len;
	
	assign feeder_k_len = i_k_len;
	
    // CONV1D WEIGHT LOADER
    //
    // Weight SRAM:
    //
    // Cout0 : K contiguous values
    // Cout1 : K contiguous values
    // Cout2 : K contiguous values
    //
    // Loader maps:
    //
    // Cout0 -> FIFO col0
    // Cout1 -> FIFO col1
    // Cout2 -> FIFO col2
	 
	conv1d_weight_loader #(
		.DATA_W(DATA_W),
      .ADDR_W(ADDR_W),
      .CFG_W(CFG_W),
      .ARRAY_COLS(ARRAY_COLS),
      .COL_W(COL_W)
    ) u_weight_loader (
      .i_clk(i_clk),
      .i_reset(i_reset),
      .i_start(i_load_start),
      .i_base_addr(i_base_addr),
      .i_k_len(i_k_len),
      .i_active_cols(i_active_cols),

		// SRAM read

      .o_sram_rd_en(o_sram_rd_en),
      .o_sram_rd_addr(o_sram_rd_addr),
      .i_sram_rd_data(i_sram_rd_data),
      .i_sram_rd_valid(i_sram_rd_valid),
		
		// WRITE INTO WEIGHT FEEDER FIFO

      .o_wgt_wr_en(loader_wr_en),
      .o_wgt_wr_data(loader_wr_data),
		
		//STATUS
      .o_busy(o_load_busy),
      .o_done(o_load_done),
      .o_state_dbg(o_loader_state)
    );
	
	// Weight Feeder: FIFO -> lockstep pop -> column skew -> valid/first/last -> SA
	weight_feeder #(
		.DATA_W(DATA_W),
      .DEPTH(DEPTH),
      .CNT_W(CNT_W),
      .K_W(K_W),

      .ARRAY_COLS(ARRAY_COLS),
      .COL_W(COL_W)
    ) u_weight_feeder (
      .i_clk(i_clk),
      .i_reset(i_reset),

		// FIFO WRITE FROM WEIGHT LOADER
      .i_wr_en(loader_wr_en),
      .i_wr_data(loader_wr_data),
		
		// TILE CONFIG
      .i_start_tile(i_feed_start),
      .i_k_len(feeder_k_len),
      .i_active_cols(i_active_cols),

		// GLOBAL LOCKSTEP
      .i_feed_step(i_feed_step),

		// SA STREAM
      .o_data(o_data),
		.o_valid(o_valid),
      .o_first(o_first),
      .o_last(o_last),

       // STATUS
       .o_ready(o_ready),
       .o_done(o_feed_done),
       .o_mem_empty(o_mem_empty)
    );


endmodule
