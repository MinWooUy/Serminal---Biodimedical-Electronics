`default_nettype none
module pe_top #(
	parameter DATA_W = 8,
	parameter ACC_W = 32
)(
	input wire i_clk, i_reset,
	
	input wire i_clear_acc,
	input wire i_scan_en,
	
	input wire i_act_valid, i_wgt_valid,
	input wire i_act_first, i_wgt_first,
	input wire i_act_last, i_wgt_last,
	
	input wire[ACC_W-1:0] i_scan_in,
	input wire[DATA_W-1:0] i_a, i_b,
	
	output wire o_act_valid, o_wgt_valid,
	output wire o_act_first, o_wgt_first,
	output wire o_act_last, o_wgt_last,
	output wire o_capture_done,
	
	output wire[ACC_W-1:0] o_scan_out,
	output wire[DATA_W-1:0] o_image, o_weight
);

	wire compute_en, acc_init, capture_en, capture_en_d;
	reg act_valid_q1, wgt_valid_q1;
	reg act_first_q1, wgt_first_q1;
	reg act_last_q1, wgt_last_q1;
	reg capture_en_q1, capture_done_q1;
	
	// Logic local control signal
	assign compute_en = i_act_valid & i_wgt_valid;
	assign acc_init = i_act_first & i_wgt_first;
	
	// Propagation data 1cycle/hop
	always @(posedge i_clk) begin
		if(i_reset) begin
			act_valid_q1 <= 1'b0;
			wgt_valid_q1 <= 1'b0;
			act_first_q1 <= 1'b0;
			wgt_first_q1 <= 1'b0;
			act_last_q1 <= 1'b0;
			wgt_last_q1 <= 1'b0;
		end else begin
			act_valid_q1 <= i_act_valid;
			wgt_valid_q1 <= i_wgt_valid;
			act_first_q1 <= i_act_first;
			wgt_first_q1 <= i_wgt_first;
			act_last_q1 <= i_act_last;
			wgt_last_q1 <= i_wgt_last;
		end
	end
	
	// Output signals
	assign o_act_valid = act_valid_q1;
	assign o_wgt_valid = wgt_valid_q1;
	
	assign o_act_first = act_first_q1;
	assign o_wgt_first = wgt_first_q1;
	
	assign o_act_last = act_last_q1;
	assign o_wgt_last = wgt_last_q1;	
	
	// delay capture_en total 2 cycle
	always @(posedge i_clk) begin
		if(i_reset) begin
			capture_en_q1 <= 1'b0;
		end else begin
			capture_en_q1 <= capture_en_d; // delay 1 cycle
		end
	end
	
	assign capture_en_d = act_last_q1 & wgt_last_q1;
	assign capture_en = capture_en_q1; // add delay 1 cycle
	
	// PE
	ProcessingElement #(
		.DATA_W(DATA_W),
		.ACC_W(ACC_W)
	) pe
	(
		.i_clk(i_clk), .i_reset(i_reset), 
	
		.i_clear_acc(i_clear_acc), 
		.i_scan_en(i_scan_en), 
		.i_capture_en(capture_en), 
		.i_compute_en(compute_en),
		.i_acc_init(acc_init),
		.i_image(i_a), .i_weight(i_b),
		.i_pe(i_scan_in),
	
		.o_pe(o_scan_out),
		.o_image(o_image), .o_weight(o_weight)
	);	
	
	// When shadow capture completed
	always @(posedge i_clk) begin
		if(i_reset) begin
			capture_done_q1 <= 1'b0;
		end else begin
			// capture_en = lenh shadow capture tại canh CLK
			capture_done_q1 <= capture_en; 
		end
	end
	// o_capture_done = bao sau canh CLK la capture xong
	assign o_capture_done = capture_done_q1;// Total delay 3 cycles
endmodule
`default_nettype wire
