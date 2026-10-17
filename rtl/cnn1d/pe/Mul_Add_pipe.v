module Mul_Add_pipe(
	input clk, reset,
	
	input i_acc_init, i_compute_en,
	input wire[15:0] inResultMul,
	
	output reg acc_init_q, compute_en_q,
	output reg[15:0] ResultMul_pipe
); 
	always @(posedge clk) begin
		if(reset) begin
			ResultMul_pipe <= 0;
			acc_init_q <= 0;
			compute_en_q <= 0;
		end else begin
			ResultMul_pipe <= inResultMul;
			acc_init_q <= i_acc_init;
			compute_en_q <= i_compute_en;
		end
	end

endmodule
