`default_nettype none
module ImageRegister(
	input wire clk, reset,
	input wire[7:0] inImage,
	output reg[7:0] outImage
); 
	always @(posedge clk or posedge reset) begin
		if(reset) outImage <= 0;
		else outImage <= inImage;
	end

endmodule
`default_nettype wire
