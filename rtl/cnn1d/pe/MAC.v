module MAC(
	input i_clk, i_reset, 
	input i_clear_acc, i_compute_en, i_acc_init,
	
	input wire[7:0] i_a, i_b,
	
	output wire[31:0] o_mac
);
	wire acc_init_q, compute_en_q;
	wire[15:0] ResultMul, outResultMul_pipe;
	wire[31:0] FinalOutput, expand_value, acc_value;

//	WallaceTreeMultiplier_8x8 WLT_Mul(
//		.A(A), .B(B),
//		.M(ResultMul)
//	 );

	// Multiplier
	wallace_tree_multiplier #(
		.DATA_W(8)
	) wlt_mul(
		.i_a(i_a), .i_b(i_b),
		.o_p(ResultMul)
	);
	 
	// pipeline mul
	Mul_Add_pipe pipe1(
		.clk(i_clk), .reset(i_reset),
		.inResultMul(ResultMul), .i_acc_init(i_acc_init), .i_compute_en(i_compute_en),
		.ResultMul_pipe(outResultMul_pipe), .acc_init_q(acc_init_q), .compute_en_q(compute_en_q)
	);
	
	// Signed Expand
	assign expand_value = {{16{outResultMul_pipe[15]}}, outResultMul_pipe};
	
	// Mux 2:1
	assign acc_value = acc_init_q ? 32'b0 : o_mac;
	
	// Adder
	CarryLookaheadAdder_32_bit CLA_32bit(
		.A(acc_value), .B(expand_value),
		.Cin(1'b0),
		.S(FinalOutput),
		.Cout(), .P_out(), .G_out()
	);	
	
	// Accumulator
	Accumulator Acc(
		.i_clk(i_clk), .i_reset(i_reset), 
		.i_clear_acc(i_clear_acc), .i_compute_en(compute_en_q),
		.i_psum(FinalOutput),
		.o_mac(o_mac)
	);
endmodule
