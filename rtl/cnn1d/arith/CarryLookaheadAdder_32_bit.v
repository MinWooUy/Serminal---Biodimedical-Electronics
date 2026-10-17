module CarryLookaheadAdder_32_bit(
	input wire[31:0] A, B,
	input Cin,
	
	output wire[31:0] S,
	output Cout, G_out, P_out
);
	wire P0, G0;
	wire C_next;

	CarryLookaheadAdder_16_bit CLA_16_1(
		.A(A[15:0]), .B(B[15:0]),
		.Cin(Cin),
	
		.S(S[15:0]),
		.Cout(), .G_out(G0), .P_out(P0)
	);
	
	CarryLookaheadAdder_16_bit CLA_16_2(
		.A(A[31:16]), .B(B[31:16]),
		.Cin(C_next),
	
		.S(S[31:16]),
		.Cout(), .G_out(G_out), .P_out(P_out)
	);

	assign C_next = Cin & P0 | G0;
	assign Cout = P0 & P_out & Cin | P_out & G0 | G_out;
	
endmodule
