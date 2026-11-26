`default_nettype none
module feed_delay#(
	parameter DATA_W = 8,
	parameter LANE_N = 0
)(
	input wire clk, reset,
	
	input wire i_valid,
	input wire i_first,
	input wire i_last,
	input wire [DATA_W-1:0] i_data,
	
	output wire o_valid, o_first, o_last,
	output wire [DATA_W-1:0] o_data
);

generate
	if(LANE_N == 0) begin
		assign o_data = i_data;
		assign o_first = i_first;
		assign o_last = i_last;
		assign o_valid = i_valid;
	end else begin 
		reg [DATA_W - 1:0] d_pipe [0:LANE_N - 1];
		reg v_pipe [0:LANE_N - 1];
		reg f_pipe [0:LANE_N - 1];
		reg l_pipe [0:LANE_N - 1];
		
		integer i;
		
		always @(posedge clk or posedge reset) begin
			if(reset) begin
				for(i = 0; i < LANE_N; i = i + 1) begin
					v_pipe[i] <= 1'b0;
					f_pipe[i] <= 1'b0;
					l_pipe[i] <= 1'b0;
					d_pipe[i] <= 0;
				end
			end else begin
				d_pipe[0] <= i_data;
				v_pipe[0] <= i_valid;
				f_pipe[0] <= i_first;
				l_pipe[0] <= i_last;
				for(i = 1; i < LANE_N; i = i + 1) begin
					v_pipe[i] <= v_pipe[i-1];
					f_pipe[i] <= f_pipe[i-1];
					l_pipe[i] <= l_pipe[i-1];
					d_pipe[i] <= d_pipe[i-1];
				end
			end
		end
		
		assign o_data = d_pipe[LANE_N - 1];
		assign o_first = f_pipe[LANE_N-1];
		assign o_last = l_pipe[LANE_N-1];
		assign o_valid = v_pipe[LANE_N - 1];
	end
endgenerate

endmodule
`default_nettype wire
