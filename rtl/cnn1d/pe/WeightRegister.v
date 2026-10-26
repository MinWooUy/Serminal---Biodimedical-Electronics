`default_nettype none
module WeightRegister(
	input wire clk, reset,
	input wire[7:0] inWeight,
	output reg[7:0] outWeight
);

	always @(posedge clk or posedge reset) begin
		if(reset) outWeight <= 0;
		else outWeight <= inWeight;
	end

endmodule
`default_nettype wire
