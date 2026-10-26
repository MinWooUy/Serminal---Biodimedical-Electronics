`default_nettype none
module half_adder(
	input wire A, B,
	output wire S, Cout
);

	assign S = A ^ B;
	assign Cout= A & B;

endmodule
`default_nettype wire
