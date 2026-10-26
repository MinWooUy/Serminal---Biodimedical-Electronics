`default_nettype none
module CarryLookaheadAdder_4_bit(
	input wire Cin,
	input wire[3:0] A, B,
	output wire[3:0] S,
	output wire P_out, G_out
	//output wire Cout // Cout = C_4
);	
	wire[3:0] P, G;
	wire[3:1] C_next;
	
	// Bit 0
	PartialFullAdder PFA0(.A(A[0]), .B(B[0]), .CarryIn(Cin), .S(S[0]), .P(P[0]), .G(G[0]));
	// C_1 = C_0P_0 + G_0
	assign C_next[1] = Cin & P[0] | G[0]; 		
	
	// Bit 1
	PartialFullAdder PFA1(.A(A[1]), .B(B[1]), .CarryIn(C_next[1]), .S(S[1]), .P(P[1]), .G(G[1]));
	// C_2 = P_0P_1C_0 + P_1G_0 + G_1
	assign C_next[2] = P[0] & P[1] & Cin | P[1] & G[0] | G[1]; 		
	
	// Bit 2
	PartialFullAdder PFA2(.A(A[2]), .B(B[2]), .CarryIn(C_next[2]), .S(S[2]), .P(P[2]), .G(G[2]));
	// C_3 = P_0P_1P_2C_0 + P_1P_2G_0 + P_2G_1 + G_2
	assign C_next[3] = P[0] & P[1] & P[2] & Cin | P[1] & P[2] & G[0] | P[2] & G[1] | G[2]; 
	
	// Bit 3
	PartialFullAdder PFA3(.A(A[3]), .B(B[3]), .CarryIn(C_next[3]), .S(S[3]), .P(P[3]), .G(G[3]));
	
	// C_4 = P_0P_1P_2P_3C_0 + P_1P_2P_3G_0 + P_2P_3G_1 + P_3G_2 + G3
	// assign Cout = P_out & Cin | G_out; (Nếu sài 4 bit thì mở comment) 
	
	// Mục đích: Ghép nâng câp lên nhiều bit (Ex: 8 - bit, 16 - bit, 32 - bit)
	// G_out: Đại diện của CLA 4 bit - giống như PFA 1 bit, G đại diện
	assign G_out = P[1] & P[2] & P[3] & G[0] | P[2] & P[3] & G[1] | P[3] & G[2] | G[3]; 
	// P_out: Đại diện của CLA 4 bit - giống như PFA 1 bit, P đại diện
	assign P_out = P[0] & P[1] & P[2] & P[3];
	 
endmodule
`default_nettype wire
