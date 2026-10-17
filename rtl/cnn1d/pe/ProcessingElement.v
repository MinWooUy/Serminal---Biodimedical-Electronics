module ProcessingElement #(
	parameter DATA_W = 8,
	parameter ACC_W = 32
)
(
	input i_clk, i_reset, 
	
	input i_clear_acc, 
	input i_scan_en, 
	input i_capture_en, 
	input i_compute_en,
	input i_acc_init,
	
	input wire[DATA_W-1:0] i_image, i_weight,
	input wire[ACC_W-1:0] i_pe,
	
	output wire[ACC_W-1:0] o_pe,
	output wire[DATA_W-1:0] o_image, o_weight
);		
	wire[ACC_W-1:0] mac_value;
	
	// Propagation weight
	WeightRegister W_reg(
		.clk(i_clk), .reset(i_reset),
		.inWeight(i_weight),
		.outWeight(o_weight)
	);
	
	// Propagation image
	ImageRegister I_reg(
		.clk(i_clk), .reset(i_reset),
		.inImage(i_image),
		.outImage(o_image)
	);
	
	// MAC engine
	MAC mac(
		.i_clk(i_clk), .i_reset(i_reset), 
		.i_clear_acc(i_clear_acc), .i_acc_init(i_acc_init), .i_compute_en(i_compute_en),
		.i_a(i_weight), .i_b(i_image),
		.o_mac(mac_value)
	);
	
	// Shadow register
	shadow_register shadow_reg(
		.i_clk(i_clk), .i_reset(i_reset),
		.i_scan_en(i_scan_en), .i_capture_en(i_capture_en),
		.i_psum(mac_value),
		.i_scan_in(i_pe),
		.o_scan_out(o_pe)
	);
endmodule
