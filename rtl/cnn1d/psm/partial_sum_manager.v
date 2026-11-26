`default_nettype none
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
	input wire i_preload_start, // Xung sau khi i_clr
	input wire i_preload_mode,	// 0: ZERO; 1: BIAS (phát chung cho mọi hàng)
	
	// Có ở mỗi chu kỳ sau o_bias_rd_en
	input wire [ACC_W-1:0] i_bias_rd_data,

	// Dynamic active tile size
	input wire [ROW_W-1:0] i_active_rows,
	input wire [COL_W-1:0] i_active_cols,
	
	// Mỗi hàng đưa vào 1 scan endpoint: scan_tap[row][active_cols-1]
	input wire [ARRAY_ROWS*ARRAY_COLS*ACC_W-1:0] i_scan_taps,

	// preload control
	output wire o_bias_rd_en, // Đọc bias RAM
	output wire o_preload_shift, // Shift preload data vào tile
	output wire [COL_W-1:0] o_bias_rd_idx, // Cột trong tile: ac-1...0
	output wire [ARRAY_ROWS*ACC_W-1:0] o_preload_vec, 

	// Tile result
	output reg [ARRAY_ROWS*ARRAY_COLS*ACC_W-1:0] o_tile,
	output reg o_tile_valid,
	output reg o_capture_done,
	output reg o_preload_done, // xung, sau lần dịch cuối

	// DRAIN (chiều đọc): stream mỗi lần dịch = 1 kênh x ARRAY_ROWS vị trí
	// Valid/ready: chuỗi shadow chỉ dịch khi được nhận (backpressure)
	input wire i_drain_ready,
	output wire o_drain_valid,
	output wire [COL_W-1:0] o_drain_idx,
	output wire [ARRAY_ROWS*ACC_W-1:0] o_drain_vec,
	output wire o_drain_shift	
);

	// Đếm số cột đã capture
	reg [COL_W-1:0] cap_cnt;
	
	// Config đã latch của tile hiện tại
	reg [ROW_W-1:0] active_rows_q;
	reg [COL_W-1:0] active_cols_q;
	reg pre_busy, pre_shift_q, pre_last_q;
	reg [COL_W-1:0] pre_idx;

	assign o_bias_rd_en = pre_busy & i_preload_mode;
	assign o_bias_rd_idx = active_cols_q - 1'b1 - pre_idx;
	assign o_preload_shift = pre_shift_q;

	// Ngoài lúc preload luôn bơm 0
	assign o_preload_vec = (pre_shift_q && i_preload_mode) ? {ARRAY_ROWS{i_bias_rd_data}} : {(ARRAY_ROWS*ACC_W){1'b0}};

	always @(posedge i_clk or posedge i_reset) begin
		if(i_reset) begin
			pre_busy <= 1'b0;
			pre_idx <= {COL_W{1'b0}};
			pre_shift_q <= 1'b0;
			pre_last_q <= 1'b0;
			o_preload_done <= 1'b0;
		end else begin
			pre_shift_q  <= pre_busy;
			pre_last_q <= pre_busy && (pre_idx == active_cols_q - 1'b1);
			o_preload_done <= pre_last_q;
			if(i_preload_start && !pre_busy) begin
				pre_busy <= 1'b1;
				pre_idx <= {COL_W{1'b0}};
			end else if(pre_busy) begin
				if(pre_idx == active_cols_q - 1'b1) pre_busy <= 1'b0;
				else pre_idx <= pre_idx + 1'b1;
			end
		end
	end

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
	
	// DRAIN: FSM bật i_capture_en suốt S_DRAIN (cho phép drain)
	// Mỗi lần bắt tay = chụp vào o_tile + đẩy ra stream + dịch chuỗi đúng 1 lần.
	wire drain_active = i_capture_en && !o_tile_valid && (active_rows_q != {ROW_W{1'b0}}) && (active_cols_q != {COL_W{1'b0}});
	wire drain_fire = drain_active && i_drain_ready;
	
	assign o_drain_valid = drain_active;
	assign o_drain_idx = active_cols_q - 1'b1 - cap_cnt;
	assign o_drain_vec = scan_pe;
	assign o_drain_shift = drain_fire;

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
			else if(drain_fire) begin
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
`default_nettype wire
