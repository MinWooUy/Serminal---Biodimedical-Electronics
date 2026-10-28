module de10_lite_top(
	input wire MAX10_CLK1_50,
	input wire KEY0,
	input wire uart_rx,
	output wire uart_tx
);

	wire reset;
	
	assign reset = ~KEY0;
	
	ecg_accel_top #(
		.CLK_FREQ(50000000),
		.BAUD_RATE(115200),
		.POST_ENABLE(1),
		.FEEDER_DEPTH(1024)
	) u_ecg_accel(
		.clk(MAX10_CLK1_50),
		.reset(reset),
		.uart_tx(uart_tx),
		.uart_rx(uart_rx)
	);

endmodule
