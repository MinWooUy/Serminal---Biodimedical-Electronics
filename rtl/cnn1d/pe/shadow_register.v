`default_nettype none
module shadow_register(
	input wire i_clk, i_reset,
	
	input wire i_scan_en, 		// lan truyen output qua SA tới PSM 
	input wire i_capture_en,	// Chuyển psum từ acc vào shadow
	input wire[31:0] i_psum,
	input wire[31:0] i_scan_in,
	
	output reg[31:0] o_scan_out
);

	always @(posedge i_clk) begin
		if(i_reset) begin
			o_scan_out <= 32'b0;
		end else if(i_capture_en) begin
			o_scan_out <= i_psum;
		end else if(i_scan_en) begin
			o_scan_out <= i_scan_in;
		end
	end
	
endmodule
`default_nettype wire
