module feed_lane#(
	parameter DATA_W = 8,
	parameter CNT_W = 5,
	parameter DEPTH = 16,
	parameter LANE_N = 0
)(
	input clk, reset,
	
	input wire wr_en, rd_en,
	input wire i_first, i_last,
	
	input wire [DATA_W-1:0] wr_data,
	
	output wire mem_empty, mem_full,
	output wire o_valid, o_first, o_last,
	output wire [DATA_W-1:0] o_data
);
	wire [DATA_W-1:0] fifo_rd_data;
	wire rd_fire;

	assign rd_fire = rd_en & ~mem_empty;

	// FIFO
	FIFO_param #(
		.CNT_W(CNT_W),
		.DATA_W(DATA_W),
		.DEPTH(DEPTH)
	) u_fifo (
		.clk(clk), .reset(reset),
		.wr_en(wr_en), .rd_en(rd_fire),
		.wr_data(wr_data),
		.mem_full(mem_full), .mem_empty(mem_empty),
		.rd_data(fifo_rd_data)
	);

	// pipeline control metadata
	reg fifo_rd_valid_q;
	reg fifo_first_q;
	reg fifo_last_q;
	
	always @(posedge clk or posedge reset) begin
		if(reset) begin
			fifo_rd_valid_q <= 1'b0;
			fifo_first_q <= 1'b0;
			fifo_last_q <= 1'b0;
		end else begin
			fifo_rd_valid_q <= rd_fire;
			fifo_first_q <= rd_fire & i_first;
			fifo_last_q <= rd_fire & i_last;
		end
	end
	
	// SKEWING
	feed_delay #(
		.DATA_W(DATA_W),
		.LANE_N(LANE_N)
	) u_skew (
		.clk(clk), .reset(reset),
		.i_valid(fifo_rd_valid_q), .i_first(fifo_first_q), .i_last(fifo_last_q),
		.i_data(fifo_rd_data),
		.o_valid(o_valid), .o_first(o_first), .o_last(o_last),
		.o_data(o_data)
	);
endmodule
