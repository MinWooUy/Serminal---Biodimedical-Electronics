module pe_array#(
  parameter DATA_W = 8,
  parameter ACC_W = 32,
  parameter ARRAY_ROWS = 3,
  parameter ARRAY_COLS = 3
)(
  input wire i_clk,
  input wire i_reset,
  
  input wire i_clear_acc,
  input wire i_scan_en,
  
  input wire[ARRAY_ROWS-1:0] i_act_valid, i_act_first, i_act_last,
  input wire[ARRAY_COLS-1:0] i_wgt_valid, i_wgt_first, i_wgt_last,
  
  input wire [ARRAY_ROWS*DATA_W-1:0] i_image,
  input wire [ARRAY_COLS*DATA_W-1:0] i_weight,

  // scan output of all PEs
  output wire[ARRAY_ROWS*ARRAY_COLS*ACC_W-1:0] o_scan_out,
  
  // capture done of all PEs
  output wire[ARRAY_ROWS*ARRAY_COLS-1:0] o_capture_done,
  
  // Optional boundary outputs
  output wire[ARRAY_ROWS*DATA_W-1:0] o_image,
  output wire[ARRAY_COLS*DATA_W-1:0] o_weight
);

	// Internal intercorrect
	wire [ARRAY_ROWS*(ARRAY_COLS+1)*DATA_W-1:0] image_link;
	wire [(ARRAY_ROWS+1)*ARRAY_COLS*DATA_W-1:0] weight_link;
  
	// Activation metadata - horizontal
	wire [ARRAY_ROWS*(ARRAY_COLS+1)-1:0] act_valid_link;
	wire [ARRAY_ROWS*(ARRAY_COLS+1)-1:0] act_first_link;
	wire [ARRAY_ROWS*(ARRAY_COLS+1)-1:0] act_last_link;
	
	// Weight metadata - vertical
	wire [(ARRAY_ROWS+1)*ARRAY_COLS-1:0] wgt_valid_link;
	wire [(ARRAY_ROWS+1)*ARRAY_COLS-1:0] wgt_first_link;
	wire [(ARRAY_ROWS+1)*ARRAY_COLS-1:0] wgt_last_link;
	
	// Shadow output of each PE
	wire[ARRAY_ROWS*ARRAY_COLS*ACC_W-1:0] scan_link;
	wire[ARRAY_ROWS*ARRAY_COLS-1:0] capture_done_link;
	
	
	// Feeder --> row input
	genvar r;
	generate
		for(r = 0; r < ARRAY_ROWS; r = r + 1) begin: GEN_IMAGE_INPUT
			assign image_link[r*(ARRAY_COLS+1)*DATA_W +: DATA_W] = i_image[r*DATA_W +: DATA_W];
			assign act_valid_link[r*(ARRAY_COLS+1)] = i_act_valid[r];
			assign act_first_link[r*(ARRAY_COLS+1)] = i_act_first[r];
			assign act_last_link[r*(ARRAY_COLS+1)] = i_act_last[r];
		end
	endgenerate

	// Feeder --> column inputs
	genvar c;
	generate
		for(c = 0; c < ARRAY_COLS; c = c + 1)begin: GEN_WEIGHT_INPUT
			assign weight_link[c*DATA_W +: DATA_W] = i_weight[c*DATA_W +: DATA_W];
			assign wgt_valid_link[c] = i_wgt_valid[c];
			assign wgt_first_link[c] = i_wgt_first[c];
			assign wgt_last_link[c] = i_wgt_last[c];	
		end
	endgenerate

	// Gen PE array
	genvar row,col;
	generate
		for(row = 0; row < ARRAY_ROWS; row = row + 1) begin: GEN_ROW
			for(col = 0; col < ARRAY_COLS; col = col + 1) begin: GEN_COL
				wire[ACC_W-1:0] local_scan_in;
				
				if(col == 0) begin: GEN_SCAN_FIRST
					assign local_scan_in = {ACC_W{1'b0}};
				end else begin: GEN_SCAN_NEXT
					assign local_scan_in = scan_link[(row*ARRAY_COLS+col-1)*ACC_W +: ACC_W];
			end
      
			//PE
			pe_top #(
				.DATA_W(DATA_W), .ACC_W(ACC_W)
			)u_pe(
				.i_clk(i_clk), .i_reset(i_reset),
				.i_clear_acc(i_clear_acc), .i_scan_en(i_scan_en),
				
				.i_act_valid(act_valid_link[row*(ARRAY_COLS+1)+col]),
				.i_act_first(act_first_link[row*(ARRAY_COLS+1)+col]),
				.i_act_last(act_last_link[row*(ARRAY_COLS+1)+col]),
				
				.i_wgt_valid(wgt_valid_link[row*ARRAY_COLS+col]),
				.i_wgt_first(wgt_first_link[row*ARRAY_COLS+col]),
				.i_wgt_last(wgt_last_link[row*ARRAY_COLS+col]),
				
				.i_a(image_link[(row*(ARRAY_COLS+1)+col)*DATA_W +: DATA_W]),
				.i_b(weight_link[(row*ARRAY_COLS+col)*DATA_W +: DATA_W]),
				.i_scan_in(local_scan_in),
				
				.o_act_valid(act_valid_link[row*(ARRAY_COLS+1)+col + 1]),
				.o_act_first(act_first_link[row*(ARRAY_COLS+1)+col + 1]),
				.o_act_last(act_last_link[row*(ARRAY_COLS+1)+col + 1]),
				
				.o_wgt_valid(wgt_valid_link[(row+1)*ARRAY_COLS+col]),
				.o_wgt_first(wgt_first_link[(row+1)*ARRAY_COLS+col]),
				.o_wgt_last(wgt_last_link[(row+1)*ARRAY_COLS+col]),
				
				.o_scan_out(scan_link[(row*ARRAY_COLS+col)*ACC_W+:ACC_W]),
				.o_capture_done(capture_done_link[row*ARRAY_COLS+col]),
				
				.o_image(image_link[(row*(ARRAY_COLS+1)+col+1)*DATA_W+:DATA_W]),
				.o_weight(weight_link[((row+1)*ARRAY_COLS+col)*DATA_W+:DATA_W])
			);
			end
		end
	endgenerate

	// EXPORT all shadow TÁP
	assign o_scan_out = scan_link;
	assign o_capture_done = capture_done_link;
	
   // Cột cuối --> PSM
   generate
		for(r = 0; r < ARRAY_ROWS; r = r + 1) begin: GEN_RIGHT_OUTPUT
			assign o_image[r*DATA_W+:DATA_W] = image_link[(r*(ARRAY_COLS+1)+ARRAY_COLS)*DATA_W+:DATA_W];
		end
   endgenerate
   // hàng cuối
   generate
     for(c = 0; c < ARRAY_COLS; c = c + 1)begin: GEN_BOTTOM_OUTPUT
			assign o_weight[c*DATA_W+:DATA_W] = weight_link[(ARRAY_ROWS*ARRAY_COLS+c)*DATA_W+:DATA_W];
     end
   endgenerate
endmodule
