`default_nettype none
module conv1d_weight_loader #(
    parameter DATA_W = 8,
    parameter ADDR_W = 16,
    parameter CFG_W = 16,

    parameter ARRAY_COLS = 3,

    parameter COL_W = (ARRAY_COLS <= 1) ? 1 : $clog2(ARRAY_COLS + 1)
)(
    input wire i_clk,
    input wire i_reset,

	// control
    input wire i_start,

    // =====================================================
    // WEIGHT CONFIG
    //
    // Weight memory layout:
    //
    // col0: K contiguous weights
    // col1: K contiguous weights
    // col2: K contiguous weights
    // ...
    //
    // Therefore:
    //
    // k_len       = Kw * Cin
    // window_step = k_len
    // num_windows = active_cols
    // =====================================================

    input wire [ADDR_W-1:0] i_base_addr,

    input wire [CFG_W-1:0] i_k_len,

    input wire [COL_W-1:0] i_active_cols,
	
    // Backpressure: cờ full của từng lane FIFO (từ weight_feeder.o_mem_full)
    input wire [ARRAY_COLS-1:0] i_lane_full,

	// WEIGHT SRAM READ INTERFACE

    output wire o_sram_rd_en,

    output wire [ADDR_W-1:0] o_sram_rd_addr,

    input wire signed [DATA_W-1:0] i_sram_rd_data,

    input wire i_sram_rd_valid,
	 
	 // WRITE INTO EXISTING WEIGHT FEEDER FIFOS

    output reg [ARRAY_COLS-1:0] o_wgt_wr_en,

    output reg signed [ARRAY_COLS*DATA_W-1:0] o_wgt_wr_data,

	// Status
    output wire o_busy,
    output reg o_done,
    output wire [2:0] o_state_dbg
);
	// =====================================================
	// Pipeline 1 byte / chu ky (giong conv1d_ifmap_loader):
	//   chu ky n  : phat doc SRAM tai dia chi AGU, AGU tien, nho cot dich (col)
	//   chu ky n+1: du lieu ve -> ghi vao FIFO cua cot; FIFO day -> giu vao skid, ngung phat doc.
	// Bo cuc trong so: cot c (kenh col_base + c) la K byte lien tiep tai base + c*K.
	// =====================================================
	localparam S_IDLE  = 3'd0;
	localparam S_RUN   = 3'd1;
	localparam S_DRAIN = 3'd2;
	localparam S_DONE  = 3'd4;

	reg [2:0] state_q;

	assign o_state_dbg = state_q;
	assign o_busy = (state_q != S_IDLE) && (state_q != S_DONE);

	// Cot (lan FIFO) cua dia chi AGU hien tai: AGU window w -> cot w
	reg [COL_W-1:0] col_cnt_q;

	reg pend_q;
	reg [COL_W-1:0] pend_col_q;

	reg skid_q;
	reg [COL_W-1:0] skid_col_q;
	reg signed [DATA_W-1:0] skid_data_q;

	wire pend_block = pend_q && i_lane_full[pend_col_q];
	wire pend_ok    = pend_q && !i_lane_full[pend_col_q];
	wire skid_ok    = skid_q && !i_lane_full[skid_col_q];

	// AGU
	wire [ADDR_W-1:0] agu_addr;
	wire agu_valid;
	wire agu_busy;
	wire agu_done;
	wire agu_k_first;
	wire agu_k_last;
	wire agu_window_first;
	wire agu_window_last;

	wire agu_start = (state_q == S_IDLE) && i_start;
	wire issue = (state_q == S_RUN) && agu_valid && !skid_q && !pend_block;
	wire agu_advance = issue;

	// Moi kenh ra = K byte lien tiep: window_step = K, so window = so cot dang dung
	wire [ADDR_W-1:0] weight_step = i_k_len;
	wire [CFG_W-1:0] agu_num_windows = i_active_cols;

	conv1d_address_generating #(
		.ADDR_W(ADDR_W),
		.CFG_W(CFG_W)
	) u_agu (
		.i_clk(i_clk),
		.i_reset(i_reset),

		.i_start(agu_start),
		.i_advance(agu_advance),

		.i_base_addr(i_base_addr),
		.i_k_len(i_k_len),
		.i_window_step(weight_step),
		.i_num_windows(agu_num_windows),

		.o_addr(agu_addr),
		.o_valid(agu_valid),
		.o_k_first(agu_k_first),
		.o_k_last(agu_k_last),
		.o_window_first(agu_window_first),
		.o_window_last(agu_window_last),
		.o_busy(agu_busy),
		.o_done(agu_done)
	);

	// SRAM request
	assign o_sram_rd_en = issue;
	assign o_sram_rd_addr = agu_addr;

	// FIFO write
	always @(*) begin
		o_wgt_wr_en = {ARRAY_COLS{1'b0}};
		o_wgt_wr_data = {ARRAY_COLS*DATA_W{1'b0}};

		if(skid_ok) begin
			o_wgt_wr_en[skid_col_q] = 1'b1;
			o_wgt_wr_data[skid_col_q*DATA_W +: DATA_W] = skid_data_q;
		end else if(pend_ok) begin
			o_wgt_wr_en[pend_col_q] = 1'b1;
			o_wgt_wr_data[pend_col_q*DATA_W +: DATA_W] = i_sram_rd_data;
		end
	end

	// FSM + pipeline
	always @(posedge i_clk) begin
		if(i_reset) begin
			state_q <= S_IDLE;
			col_cnt_q <= {COL_W{1'b0}};
			pend_q <= 1'b0;
			pend_col_q <= {COL_W{1'b0}};
			skid_q <= 1'b0;
			skid_col_q <= {COL_W{1'b0}};
			skid_data_q <= {DATA_W{1'b0}};
			o_done <= 1'b0;
		end else begin
			o_done <= 1'b0;

			pend_q <= issue;
			if(issue) pend_col_q <= col_cnt_q;

			if(pend_block) begin
				skid_q <= 1'b1;
				skid_col_q <= pend_col_q;
				skid_data_q <= i_sram_rd_data;
			end else if(skid_ok) begin
				skid_q <= 1'b0;
			end

			case(state_q)
				S_IDLE: begin
					col_cnt_q <= {COL_W{1'b0}};
					if(i_start) state_q <= S_RUN;
				end

				S_RUN: begin
					if(issue && agu_k_last) begin
						if(agu_window_last) state_q <= S_DRAIN;
						else col_cnt_q <= col_cnt_q + 1'b1;
					end
				end

				S_DRAIN: begin
					if(!pend_q && !skid_q) state_q <= S_DONE;
				end

				S_DONE: begin
					o_done <= 1'b1;
					if(!i_start) state_q <= S_IDLE;
				end

				default: begin
					state_q <= S_IDLE;
					col_cnt_q <= {COL_W{1'b0}};
				end
			endcase
		end
	end
endmodule
`default_nettype wire
