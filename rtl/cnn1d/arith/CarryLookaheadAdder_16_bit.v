`default_nettype none
module CarryLookaheadAdder_16_bit(
	input wire[15:0] A, B,
	input wire Cin,
	
	output wire[15:0] S,
	output wire Cout, G_out, P_out
);
	
	wire[2:0] C_out;
	wire[3:0] P, G;
	wire[3:1] C_next;
	
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
		.Cin(C_next[1]),  
		.S(S[7:4]),
		.P_out(P[1]), 
		.G_out(G[1])
	);

	CarryLookaheadAdder_4_bit CLA_4BIT_3(
		.A(A[11:8]), 
		.B(B[11:8]), 
		.Cin(C_next[2]),  
		.S(S[11:8]),
		.P_out(P[2]), 
		.G_out(G[2])
	);

	CarryLookaheadAdder_4_bit CLA_4BIT_4(
		.A(A[15:12]), 
		.B(B[15:12]), 
		.Cin(C_next[3]),  
		.S(S[15:12]),
		.P_out(P[3]), 
		.G_out(G[3])
	);
	
	// Hierarchical Architecture
	assign C_next[1] = Cin & P[0] | G[0]; // C[4]
	assign C_next[2] = Cin & P[0] & P[1] | P[1] & G[0] | G[1]; // C[8]
	assign C_next[3] = Cin & P[0] & P[1] & P[2] | P[1] & P[2] & G[0] | P[2] & G[1] | G[2]; // C[12]
	
	// G,P đại diện cho khối CLA 16 bit (dùng nâng cấp 32 bit)
	assign G_out = P[1] & P[2] & P[3] & G[0] | P[2] & P[3] & G[1] | P[3] & G[2] | G[3]; 
	assign P_out = P[0] & P[1] & P[2] & P[3];
	
	assign Cout = P_out & Cin | G_out;
endmodule
`default_nettype wire
