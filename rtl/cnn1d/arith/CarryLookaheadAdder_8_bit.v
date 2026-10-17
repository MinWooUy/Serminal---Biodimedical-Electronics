module CarryLookaheadAdder_8_bit(
	input wire[7:0] A, B,
	input Cin,
	
	output wire[7:0] S,
	output Cout, G_out, P_out
);
	wire[1:0] P, G;
	wire C_next;
	
	CarryLookaheadAdder_4_bit CLA_4BIT_1(
		.A(A[3:0]), 
		.B(B[3:0]), 
		.Cin(Cin), 
		.S(S[3:0]), 
		.P_out(P[0]), 
		.G_out(G[0])
	);
	
	CarryLookaheadAdder_4_bit CLA_4BIT_2(
		.A(A[7:4]), 
		.B(B[7:4]), 
		.Cin(C_next),  
		.S(S[7:4]),
		.P_out(P[1]), 
		.G_out(G[1])
	);

	// Hierarchical Architecture
	assign C_next = Cin & P[0] | G[0]; // C[4]
	assign G_out = G[1] | (P[1] & G[0]); 
	assign P_out = P[0] & P[1];
	
	assign Cout = P_out & Cin | G_out;
endmodule
