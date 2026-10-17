module Accumulator(
	input i_clk, i_reset, 
	input i_clear_acc, i_compute_en,
	
	input wire[31:0] i_psum,
	
	output reg[31:0] o_mac
);

	always @(posedge i_clk) begin
		if(i_reset)
			o_mac <= 32'b0;
		else if(i_clear_acc) 
			o_mac <= 32'b0;
		else if(i_compute_en) 
			o_mac <= i_psum;
	end

endmodule
