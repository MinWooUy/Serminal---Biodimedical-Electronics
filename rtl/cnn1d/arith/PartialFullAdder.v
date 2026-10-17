module PartialFullAdder(
	input wire A, B, CarryIn,
	output wire S, P, G
);

	assign P = A ^ B;
	assign G = A & B;
	assign S = P ^ CarryIn;

endmodule
