`default_nettype none
module scan_endpoint_mux #(
	parameter ACC_W = 32,
	parameter ARRAY_ROWS = 3,
	parameter ARRAY_COLS = 3,
	parameter COL_W = (ARRAY_COLS <= 1) ? 1 : $clog2(ARRAY_COLS + 1)
)(
	// Toàn bộ shadow tạo từ PE array
	input wire[ARRAY_ROWS*ARRAY_COLS*ACC_W-1:0] i_scan_taps,
	input wire[COL_W-1:0] i_active_cols, // Số column đang activeƯ
	output reg[ARRAY_ROWS*ACC_W-1:0] o_scan_pe
);
	integer r;
	
	always @(*) begin
		o_scan_pe = {(ARRAY_ROWS*ACC_W){1'b0}};
		
		if((i_active_cols != 0) && (i_active_cols <= ARRAY_COLS)) begin
			for(r = 0; r < ARRAY_ROWS; r = r + 1) begin
				o_scan_pe[r*ACC_W +: ACC_W] = i_scan_taps[((r*ARRAY_COLS) + (i_active_cols - 1'b1))*ACC_W +: ACC_W];
			end
		end
	end

endmodule
`default_nettype wire
