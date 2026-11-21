`default_nettype none

module layer_core #(
  parameter CLK_FREQ = 50000000,
  parameter BAUD_RATE = 115200,
  parameter MAX_PAYLOAD = 256,
  parameter ARRAY_ROWS = 3,
  parameter ARRAY_COLS = 3,
  parameter IFMAP_MEM_DEPTH = 4096,
  parameter WGT_MEM_DEPTH = 16384,
  parameter FEEDER_DEPTH = 1024,
  parameter COUT_MAX = 64
)(
  input wire i_clk, i_reset,

  input wire [7:0] i_rx_cmd, i_rx_seq,
  input wire [15:0] i_rx_length,
  input wire [MAX_PAYLOAD*8-1:0] i_rx_payload,
  input wire i_rx_valid, i_rx_error,

  input wire i_tx_busy,

  output wire o_rx_enable,
  output wire o_tx_start,
  output wire [7:0] o_tx_cmd, o_tx_seq,
  output wire [15:0] o_tx_length,
  output wire [MAX_PAYLOAD*8-1:0] o_tx_payload
);

  localparam ADDR_W = 16, CFG_W = 16;

	wire x_wr_en, w_wr_en, x_swap, w_swap, bias_wr_en, shift_wr_en, layer_start;
	wire [ADDR_W-1:0] x_wr_addr, w_wr_addr, bias_wr_addr, shift_wr_addr, window_step;
	wire [7:0] x_wr_data, w_wr_data, shift_wr_data;
	wire [31:0] bias_wr_data;
	wire [CFG_W-1:0] out_len, cout, k_len, pad_next, gap_rd_idx;
	wire pool_en, gap_en;
	wire [3:0] layer_flags;
	wire [ADDR_W-1:0] layer_w_base;
	wire [CFG_W-1:0] layer_b_base;
	wire [15:0] gap_rd_data;
	wire dbg_rd_en, dbg_rd_valid, layer_busy, layer_done, layer_err;
	wire [ADDR_W-1:0] dbg_rd_addr;
	wire [7:0] dbg_rd_data;

  layer_cmd_controller #(
    .ARRAY_ROWS(ARRAY_ROWS), .ARRAY_COLS(ARRAY_COLS),
    .CLK_FREQ(CLK_FREQ), .BAUD_RATE(BAUD_RATE),
    .MAX_PAYLOAD(MAX_PAYLOAD),
    .IFMAP_MEM_DEPTH(IFMAP_MEM_DEPTH),
    .WGT_MEM_DEPTH(WGT_MEM_DEPTH),
    .FEEDER_DEPTH(FEEDER_DEPTH),
    .COUT_MAX(COUT_MAX),
    .ADDR_W(ADDR_W),
    .CFG_W(CFG_W)
  )u_cmd(
    .i_clk(i_clk), .i_reset(i_reset),

    .o_rx_enable(o_rx_enable), .i_rx_cmd(i_rx_cmd), .i_rx_seq(i_rx_seq), .i_rx_length(i_rx_length),
    .i_rx_payload(i_rx_payload), .i_rx_valid(i_rx_valid), .i_rx_error(i_rx_error),
    .o_tx_start(o_tx_start), .o_tx_cmd(o_tx_cmd), .o_tx_seq(o_tx_seq), .o_tx_length(o_tx_length),
    .o_tx_payload(o_tx_payload), .i_tx_busy(i_tx_busy),

    .o_x_wr_en(x_wr_en), .o_x_wr_addr(x_wr_addr), .o_x_wr_data(x_wr_data), .o_x_swap(x_swap),
    .o_w_wr_en(w_wr_en), .o_w_wr_addr(w_wr_addr), .o_w_wr_data(w_wr_data), .o_w_swap(w_swap),
    .o_bias_wr_en(bias_wr_en), .o_bias_wr_addr(bias_wr_addr), .o_bias_wr_data(bias_wr_data),
    .o_shift_wr_en(shift_wr_en), .o_shift_wr_addr(shift_wr_addr), .o_shift_wr_data(shift_wr_data),
    .o_layer_start(layer_start), .o_out_len(out_len), .o_cout(cout), .o_k_len(k_len),
    .o_window_step(window_step), .o_pool_en(pool_en), .o_gap_en(gap_en), .o_pad_next(pad_next),
    .o_flags(layer_flags), .o_w_base(layer_w_base), .o_b_base(layer_b_base),
    .o_gap_rd_idx(gap_rd_idx), .i_gap_rd_data(gap_rd_data),
    .o_dbg_rd_en(dbg_rd_en), .o_dbg_rd_addr(dbg_rd_addr), .i_dbg_rd_data(dbg_rd_data), .i_dbg_rd_valid(dbg_rd_valid),
    .i_layer_busy(layer_busy), .i_layer_done(layer_done), .i_layer_err(layer_err)
  );

  conv1d_layer_top#(
		.ARRAY_ROWS(ARRAY_ROWS), .ARRAY_COLS(ARRAY_COLS),
		.IFMAP_MEM_DEPTH(IFMAP_MEM_DEPTH), .WGT_MEM_DEPTH(WGT_MEM_DEPTH),
		.FEEDER_DEPTH(FEEDER_DEPTH), .COUT_MAX(COUT_MAX)
  )u_layer(
		.i_clk(i_clk), .i_reset(i_reset),
		.i_x_wr_en(x_wr_en), .i_x_wr_addr(x_wr_addr), .i_x_wr_data(x_wr_data), .i_x_swap(x_swap),
		.i_w_wr_en(w_wr_en), .i_w_wr_addr(w_wr_addr), .i_w_wr_data(w_wr_data), .i_w_swap(w_swap),
		.i_bias_wr_en(bias_wr_en), .i_bias_wr_addr(bias_wr_addr), .i_bias_wr_data(bias_wr_data),
		.i_shift_wr_en(shift_wr_en), .i_shift_wr_addr(shift_wr_addr), .i_shift_wr_data(shift_wr_data),
		.i_layer_start(layer_start), .i_out_len(out_len), .i_cout(cout), .i_k_len(k_len),
		.i_window_step(window_step), .i_pool_en(pool_en), .i_gap_en(gap_en), .i_pad_next(pad_next),
		.i_flags(layer_flags), .i_w_base(layer_w_base), .i_b_base(layer_b_base),
		.i_gap_rd_idx(gap_rd_idx), .o_gap_rd_data(gap_rd_data),
		.i_dbg_rd_en(dbg_rd_en), .i_dbg_rd_addr(dbg_rd_addr),
		.o_dbg_rd_data(dbg_rd_data), .o_dbg_rd_valid(dbg_rd_valid),
		.o_busy(layer_busy), .o_done(layer_done), .o_err(layer_err)
  );

endmodule
`default_nettype wire