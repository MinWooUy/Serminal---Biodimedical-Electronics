`default_nettype none
module tensor_sram#(
	parameter DATA_W = 8,
	parameter DEPTH = 1024,
	parameter ADDR_W = (DEPTH <= 1) ? 1 : $clog2(DEPTH)
)(
	input wire i_clk,
	input wire i_reset,
	
	input wire i_wr_en,
	input wire [ADDR_W-1:0] i_wr_addr,
	input wire signed [DATA_W-1:0] i_wr_data,
	
	input wire i_rd_en,
	input wire [ADDR_W-1:0] i_rd_addr,

	// o_rd_valid để làm gì?
	// conv1d AGU yêu cầu:
	// cycle 10				cycle 11
	//
	// rd_en = 1			o_rd_valid = 1
	// rd_addr = 5			o_rd_data = X[5]
	//
	// khối Feeder_window_reader chỉ cần biết
	// --> o_rd_valid = 1 --> sample SRAM trả về đang hợp lệ --> metadata không bị lệch chu kỳ
	output reg o_rd_valid, 
	output reg signed [DATA_W-1:0] o_rd_data
);
	// Simple dual-port M9K template:
	// - port A: write-only
	// - port B: registered read, latency = 1 cycle
	//
	// Keep both accesses in one clocked process.  Quartus 18.1 otherwise
	// classifies this RAM as having asynchronous read logic and implements
	// every 1024x8 bank with registers/muxes instead of an M9K.
	// The two ports access separate ping-pong banks, so same-address
	// read-during-write is architecturally impossible.
	(* ramstyle = "M9K, no_rw_check" *)
	reg signed [DATA_W-1:0] mem [0:DEPTH-1];

	always @(posedge i_clk) begin
		if (i_wr_en)
			mem[i_wr_addr] <= i_wr_data;

		if (i_rd_en)
			o_rd_data <= mem[i_rd_addr];

		if (i_reset)
			o_rd_valid <= 1'b0;
		else
			o_rd_valid <= i_rd_en;
	end
endmodule
`default_nettype wire
