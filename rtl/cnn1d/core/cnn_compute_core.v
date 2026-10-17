// Dùng tích hợp conv1d
module cnn_compute_core #(
	parameter DATA_W = 8,
	parameter ACC_W = 32,
	parameter ARRAY_ROWS = 3,
	parameter ARRAY_COLS = 3,
	parameter ROW_W = (ARRAY_ROWS <= 1) ? 1 : $clog2(ARRAY_ROWS + 1),
	parameter COL_W = (ARRAY_COLS <= 1) ? 1 : $clog2(ARRAY_COLS + 1)
)(
	input wire i_clk,
	input wire i_reset,
	
	// Điều khiển và cấu hình
	input wire i_start,
	input wire [ROW_W-1:0] i_active_rows,
	input wire [COL_W-1:0] i_active_cols,
	
	// IFMAP stream đã được skew từ IFMAP feeder
	input wire [ARRAY_ROWS*DATA_W-1:0] i_ifm_data,
	input wire [ARRAY_ROWS-1:0] i_ifm_valid,
	input wire [ARRAY_ROWS-1:0] i_ifm_first,
	input wire [ARRAY_ROWS-1:0] i_ifm_last,
	
	// WEIGHT stream đã được skew từ Weight feeder
	input wire [ARRAY_COLS*DATA_W-1:0] i_wgt_data,
	input wire [ARRAY_COLS-1:0] i_wgt_valid,
	input wire [ARRAY_COLS-1:0] i_wgt_first,
	input wire [ARRAY_COLS-1:0] i_wgt_last,
	
	// Start cho các feeder bên ngoài
	output wire o_feeder_start,
	
	// Kết quả tile
	output wire [ARRAY_ROWS*ARRAY_COLS*ACC_W-1:0] o_tile,
	output wire o_tile_valid,
	
	// Trạng thái
	output wire o_busy,
	output wire o_done,
	output wire [2:0] o_state_dbg
);

	// ---------
	// Latch tile config
	// ---------
	reg [ROW_W-1:0] active_rows_q;
	reg [COL_W-1:0] active_cols_q;
	
	always @(posedge i_clk or posedge i_reset) begin
		if(i_reset) begin
			active_rows_q <= {ROW_W{1'b0}};
			active_cols_q <= {COL_W{1'b0}};
		end else begin
			// Freeze config tại lúc nhận start
			if(i_start && !o_busy && !o_done) begin
				active_rows_q <= i_active_rows;
				active_cols_q <= i_active_cols;
			end
		end
	end
	
	// ---------
	// Controller signal
	// ---------
	wire feeder_start;
	wire scan_en;
	wire psm_clr;
	wire psm_capture_en;
	wire sa_tile_done;
	wire psm_capture_done;
	
	assign o_feeder_start = feeder_start;
	
	// ---------
	// Controller FSM
	// ---------
	cnn_control_fsm u_controller(
		.i_clk(i_clk),
		.i_reset(i_reset),
		
		.i_start(i_start),
		.i_sa_tile_done(sa_tile_done),
		.i_psm_capture_done(psm_capture_done),
		
		.o_feeder_start(feeder_start),
		.o_scan_en(scan_en),
		.o_psm_clr(psm_clr),
		.o_psm_capture_en(psm_capture_en),
		
		.busy(o_busy),
		.done(o_done),
		.state_dbg(o_state_dbg)
	);
	
	// ---------
	// Systolic array
	// ---------
	wire [ARRAY_ROWS*ARRAY_COLS*ACC_W-1:0] sa_scan_taps;
	wire [ARRAY_ROWS*ARRAY_COLS-1:0] sa_capture_done;
	wire [ARRAY_ROWS*DATA_W-1:0] sa_out_image;
	wire [ARRAY_COLS*DATA_W-1:0] sa_out_weight;
	
	pe_array #(
		.DATA_W(DATA_W),
		.ACC_W(ACC_W),
		.ARRAY_ROWS(ARRAY_ROWS),
		.ARRAY_COLS(ARRAY_COLS)
	) u_sa(
		.i_clk(i_clk),
		.i_reset(i_reset),
		
		.i_clear_acc(1'b0),
		.i_scan_en(scan_en),
		
		// IFMAP stream
		.i_act_valid(i_ifm_valid),
		.i_act_first(i_ifm_first),
		.i_act_last(i_ifm_last),
		
		// Weight stream
		.i_wgt_valid(i_wgt_valid),
		.i_wgt_first(i_wgt_first),
		.i_wgt_last(i_wgt_last),
		
		.i_image(i_ifm_data),
		.i_weight(i_wgt_data),
		
		.o_scan_out(sa_scan_taps),
		.o_capture_done(sa_capture_done),
		
		.o_image(sa_out_image),
		.o_weight(sa_out_weight)
	);
	
	// ---------
	// Select SA Tile done
	//
	// last active PE:
	// PE[active_rows-1][active_cols-1]
	//
	// index:
	// (active_rows-1)*ARRAY_COLS + (active_cols-1)
	// ---------
	reg sa_tile_done_sel;
	
	always @(*) begin
		sa_tile_done_sel = 1'b0;
		
		if((active_rows_q != {ROW_W{1'b0}}) && (active_rows_q <= ARRAY_ROWS) && (active_cols_q != {COL_W{1'b0}}) && (active_cols_q <= ARRAY_COLS)) begin
			sa_tile_done_sel = sa_capture_done[((active_rows_q - 1'b1)*ARRAY_COLS) + (active_cols_q - 1'b1)];
		end
	end
	
	assign sa_tile_done = sa_tile_done_sel;
	
	// ---------
	// Partial Sum Manager
	// ---------
	partial_sum_manager #(
		.ACC_W(ACC_W),
		.ARRAY_ROWS(ARRAY_ROWS),
		.ARRAY_COLS(ARRAY_COLS),
		.ROW_W(ROW_W),
		.COL_W(COL_W)
	) u_psm(
		.i_clk(i_clk),
		.i_reset(i_reset),
		
		.i_clr(psm_clr),
		.i_capture_en(psm_capture_en),
		
		.i_active_rows(active_rows_q),
		.i_active_cols(active_cols_q),
		
		.i_scan_taps(sa_scan_taps),
		
		.o_tile(o_tile),
		.o_tile_valid(o_tile_valid),
		.o_capture_done(psm_capture_done)
	);

endmodule