`default_nettype none
// ecg_layer_top: top-level cho board  chay TRON MOT LAYER.
//   uart_crc_top  -> layer_core (layer_cmd_controller + conv1d_layer_top)
// reset tich cuc cao.
module ecg_accel_top #(
	parameter CLK_FREQ = 50000000, BAUD_RATE = 115200,
	parameter MAX_PAYLOAD = 256,
	parameter RX_TIMEOUT_CLKS = CLK_FREQ/10,
	parameter ARRAY_ROWS = 3, ARRAY_COLS = 3,
	parameter IFMAP_MEM_DEPTH = 4096, WGT_MEM_DEPTH = 16384,
	parameter FEEDER_DEPTH = 1024, COUT_MAX = 64
)(input wire clk, reset, uart_rx, output wire uart_tx);

`ifndef SYNTHESIS
	initial begin
		if (CLK_FREQ/BAUD_RATE < 8 || RX_TIMEOUT_CLKS < 1 || MAX_PAYLOAD > 256 || COUT_MAX*2 > MAX_PAYLOAD)
			$fatal(1, "ecg_layer_top: tham so UART/payload khong hop le");
	end
`endif

	wire rx_enable, rx_valid, rx_error, tx_start, tx_busy;
	wire [7:0] rx_cmd, rx_seq, tx_cmd, tx_seq;
	wire [15:0] rx_length, tx_length;
	wire [MAX_PAYLOAD*8-1:0] rx_payload, tx_payload;

	uart_crc_top #(.CLK_FREQ(CLK_FREQ), .BAUD_RATE(BAUD_RATE),
		.MAX_PAYLOAD(MAX_PAYLOAD), .RX_TIMEOUT_CLKS(RX_TIMEOUT_CLKS)
	) u_transport (
		.clk(clk), .reset(reset), .uart_rx(uart_rx), .uart_tx(uart_tx),
		.rx_enable(rx_enable), .rx_cmd(rx_cmd), .rx_seq(rx_seq), .rx_length(rx_length),
		.rx_payload(rx_payload), .rx_valid(rx_valid), .rx_error(rx_error),
		.tx_start(tx_start), .tx_cmd(tx_cmd), .tx_seq(tx_seq), .tx_length(tx_length),
		.tx_payload(tx_payload), .tx_busy(tx_busy)
	);

	layer_core #(
		.CLK_FREQ(CLK_FREQ), .BAUD_RATE(BAUD_RATE),
		.MAX_PAYLOAD(MAX_PAYLOAD), .ARRAY_ROWS(ARRAY_ROWS), .ARRAY_COLS(ARRAY_COLS),
		.IFMAP_MEM_DEPTH(IFMAP_MEM_DEPTH), .WGT_MEM_DEPTH(WGT_MEM_DEPTH),
		.FEEDER_DEPTH(FEEDER_DEPTH), .COUT_MAX(COUT_MAX)
	) u_core(
		.i_clk(clk), .i_reset(reset),
		.o_rx_enable(rx_enable), .i_rx_cmd(rx_cmd), .i_rx_seq(rx_seq), .i_rx_length(rx_length),
		.i_rx_payload(rx_payload), .i_rx_valid(rx_valid), .i_rx_error(rx_error),
		.o_tx_start(tx_start), .o_tx_cmd(tx_cmd), .o_tx_seq(tx_seq), .o_tx_length(tx_length),
		.o_tx_payload(tx_payload), .i_tx_busy(tx_busy)
	);
endmodule
`default_nettype wire
