// Note với phiên bản hiện tại:
// 
// Controller phải đảm bảo: 1 <= i_active_cols <= ARRAY_COLS và T_capture = active_cols
//
// drain: col_idx = active_cols_q - 1 - cap_cnt
//
// Khi:
// cap_cnt = active_cols_q - 1
// --> o_tile_valid = 1
// --> o_capture_done = pulse 1 cycle
//
//
// i_clr:
// --> clear PSM
// --> latch active_cols
//
// i_capture_en = 1 trong active_cols cycle
// --> capture từng cột
//
// capture cuối
// --> o_tile_valid = 1
// --> o_capture_done pulse

module partial_sum_manager #(
	parameter ACC_W = 32,
	parameter ARRAY_ROWS = 3,
	parameter ARRAY_COLS = 3,
	parameter ROW_W = (ARRAY_ROWS <= 1) ? 1 : $clog2(ARRAY_ROWS + 1),
	parameter COL_W = (ARRAY_COLS <= 1) ? 1 : $clog2(ARRAY_COLS + 1)
)(
	input wire i_clk,
	input wire i_reset,
	
	// Control
	input wire i_clr,
	input wire i_capture_en,
	
	// Dynamic active tile size
	input wire [ROW_W-1:0] i_active_rows,
	input wire [COL_W-1:0] i_active_cols,
	
	// Mỗi hàng đưa vào 1 scan endpoint: scan_tap[row][active_cols-1]
	input wire [ARRAY_ROWS*ARRAY_COLS*ACC_W-1:0] i_scan_taps,
	
	// Tile result
	output reg [ARRAY_ROWS*ARRAY_COLS*ACC_W-1:0] o_tile,
	output reg o_tile_valid,
	output reg o_capture_done
);

	// Đếm số cột đã capture
	reg [COL_W-1:0] cap_cnt;
	
	// Config đã latch của tile hiện tại
	reg [ROW_W-1:0] active_rows_q;
	reg [COL_W-1:0] active_cols_q;
	
	// MUX scan endpoint
	wire [ARRAY_ROWS*ACC_W-1:0] scan_pe;
	
	scan_endpoint_mux #(
		.ACC_W(ACC_W),
		.ARRAY_ROWS(ARRAY_ROWS),
		.ARRAY_COLS(ARRAY_COLS),
		.COL_W(COL_W)
	) u_scan_endpoint_mux(
		.i_scan_taps(i_scan_taps),
		.i_active_cols(active_cols_q),
		.o_scan_pe(scan_pe)
	);
	
	// Capture logic
	// scan order:
	// cycle 0: col = active_cols - 1
	// cycle 1: col = active_cols - 2
	// ...
	// cycle cuối: col = 0
	
	integer r, base;
	
	always @(posedge i_clk or posedge i_reset) begin
		if(i_reset) begin
			cap_cnt <= {COL_W{1'b0}};
			active_rows_q <= {ROW_W{1'b0}};
			active_cols_q <= {COL_W{1'b0}};
			
			o_tile <= {(ARRAY_ROWS*ARRAY_COLS*ACC_W){1'b0}};
			o_tile_valid <= 1'b0;
			o_capture_done <= 1'b0;
		end else begin
			o_capture_done <= 1'b0;
			
			// Chuẩn bị tile mới
			if(i_clr) begin
				cap_cnt <= {COL_W{1'b0}};
				active_rows_q <= i_active_rows;
				active_cols_q <= i_active_cols;
				
				o_tile <= {(ARRAY_ROWS*ARRAY_COLS*ACC_W){1'b0}};
				o_tile_valid <= 1'b0;
			end
			
			// Capture
			else if(i_capture_en && !o_tile_valid && (active_rows_q != {ROW_W{1'b0}}) && (active_cols_q != {COL_W{1'b0}})) begin
				// Tất cả active row capture song song
				for(r = 0; r < ARRAY_ROWS; r = r + 1) begin
					if(r < active_rows_q) begin
						base = (r*ARRAY_COLS + (active_cols_q - 1'b1 - cap_cnt)) * ACC_W;
						o_tile[base +: ACC_W] <= scan_pe[r*ACC_W +: ACC_W];
					end
				end
				
				// Capture cột cuối cùng
				if(cap_cnt == active_cols_q - 1'b1) begin
					o_tile_valid <= 1'b1;
					o_capture_done <= 1'b1;
				end else begin
					cap_cnt <= cap_cnt + 1'b1;
				end
			end
		end
	end

endmodule
