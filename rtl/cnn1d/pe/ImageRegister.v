module ImageRegister(
	input clk, reset,
	input wire[7:0] inImage,
	output reg[7:0] outImage
); 
	always @(posedge clk or posedge reset) begin
		if(reset) outImage <= 0;
		else outImage <= inImage;
	end

endmodule
