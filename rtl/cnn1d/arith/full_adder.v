module full_adder(
	input wire A, B, CarryIn,
	output wire Y, CarryOut
);
	assign Y = A ^ B ^ CarryIn;
	assign CarryOut = B & CarryIn | A & CarryIn | A & B;
endmodule
