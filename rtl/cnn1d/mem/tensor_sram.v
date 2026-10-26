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
	// memory
	reg signed[DATA_W-1:0] mem[0:DEPTH-1];
	
	// Synchronous write
	always @(posedge i_clk) begin
		if(i_wr_en) begin
			mem[i_wr_addr] <= i_wr_data;
		end
	end
	
	// Synchronous read
	// Read latency = 1 clock
	
	always @(posedge i_clk) begin
		if(i_reset) begin
			o_rd_data <= {DATA_W{1'b0}};
			o_rd_valid <= 1'b0;
		end else begin
			o_rd_valid <= i_rd_en;
			if(i_rd_en) begin
				o_rd_data <= mem[i_rd_addr];
			end
		end
	end
endmodule
`default_nettype wire
