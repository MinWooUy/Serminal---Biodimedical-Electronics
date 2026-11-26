`default_nettype none
module conv1d_ifmap_loader #(
	parameter DATA_W = 8,
	parameter ADDR_W = 16,
	parameter CFG_W = 16,
	parameter ARRAY_ROWS = 3,
	parameter ROW_W = (ARRAY_ROWS <= 1) ? 1 : $clog2(ARRAY_ROWS + 1)
)(
	input wire i_clk,
	input wire i_reset,
	
	// control
	input wire i_start,
	
	// CONV1D config
	// k_len = Kw * Cin
	// window_step = stride * Cin
	input wire [ADDR_W-1:0] i_base_addr,
	input wire [CFG_W-1:0] i_k_len,
	input wire [ADDR_W-1:0] i_window_step,
	input wire [ROW_W-1:0] i_active_rows,
	
	// SRAM read interface
	output wire o_sram_rd_en,
	output wire [ADDR_W-1:0] o_sram_rd_addr,
	input wire signed [DATA_W-1:0] i_sram_rd_data,
	input wire i_sram_rd_valid,
	
	// Backpressure: cờ full của từng lane FIFO (từ ifmap_feeder.o_mem_full)
	input wire [ARRAY_ROWS-1:0] i_lane_full,
	// Write into IFMAP feeder FIFOs
	output reg [ARRAY_ROWS-1:0] o_ifm_wr_en,
	output reg signed [ARRAY_ROWS*DATA_W-1:0] o_ifm_wr_data,
	
	// status
	output wire o_busy,
	output reg o_done,
	output wire [2:0] o_state_dbg
);

	// =====================================================
	// Pipeline 1 byte / chu ky (L04 buoc 1):
	//   chu ky n  : AGU co dia chi va duoc phep -> phat doc SRAM, AGU tien, nho lan dich (row)
	//   chu ky n+1: SRAM tra du lieu -> ghi thang vao FIFO cua lan do.
	//               Neu FIFO lan do day: giu byte vao thanh ghi dem (skid) va ngung phat doc
	//               cho toi khi byte do ghi duoc -> khong mat du lieu, thu tu k giu nguyen.
	// Tai mot thoi diem co toi da 1 byte tren duong (pend) HOAC 1 byte trong skid, khong ca hai.
	// =====================================================
	localparam S_IDLE  = 3'd0;
	localparam S_RUN   = 3'd1; // dang phat lenh doc
	localparam S_DRAIN = 3'd2; // da phat het, cho byte cuoi ghi xong
	localparam S_DONE  = 3'd4;

	reg [2:0] state_q;

	assign o_state_dbg = state_q;
	assign o_busy = (state_q != S_IDLE) && (state_q != S_DONE);

	// Lan (hang vat ly) cua dia chi AGU hien tai
	reg [ROW_W-1:0] row_cnt_q;

	// Byte dang tren duong tu SRAM (ve o chu ky nay)
	reg pend_q;
	reg [ROW_W-1:0] pend_row_q;

	// Thanh ghi dem khi FIFO dich day
	reg skid_q;
	reg [ROW_W-1:0] skid_row_q;
	reg signed [DATA_W-1:0] skid_data_q;

	wire pend_block = pend_q && i_lane_full[pend_row_q];
	wire pend_ok    = pend_q && !i_lane_full[pend_row_q];
	wire skid_ok    = skid_q && !i_lane_full[skid_row_q];

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

	// So window can tao = so active row
	wire [CFG_W-1:0] agu_num_windows;
	assign agu_num_windows = i_active_rows;

	conv1d_address_generating #(
		.ADDR_W(ADDR_W),
		.CFG_W(CFG_W)
	) u_agu(
		.i_clk(i_clk),
		.i_reset(i_reset),

		.i_start(agu_start),
		.i_advance(agu_advance),

		.i_base_addr(i_base_addr),
		.i_k_len(i_k_len),
		.i_window_step(i_window_step),
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

	// FIFO write: uu tien byte trong skid (khi co skid thi khong co byte tren duong)
	always @(*) begin
		o_ifm_wr_en = {ARRAY_ROWS{1'b0}};
		o_ifm_wr_data = {ARRAY_ROWS*DATA_W{1'b0}};

		if(skid_ok) begin
			o_ifm_wr_en[skid_row_q] = 1'b1;
			o_ifm_wr_data[skid_row_q*DATA_W +: DATA_W] = skid_data_q;
		end else if(pend_ok) begin
			o_ifm_wr_en[pend_row_q] = 1'b1;
			o_ifm_wr_data[pend_row_q*DATA_W +: DATA_W] = i_sram_rd_data;
		end
	end

	// FSM + pipeline
	always @(posedge i_clk) begin
		if(i_reset) begin
			state_q <= S_IDLE;
			row_cnt_q <= {ROW_W{1'b0}};
			pend_q <= 1'b0;
			pend_row_q <= {ROW_W{1'b0}};
			skid_q <= 1'b0;
			skid_row_q <= {ROW_W{1'b0}};
			skid_data_q <= {DATA_W{1'b0}};
			o_done <= 1'b0;
		end else begin
			o_done <= 1'b0;

			pend_q <= issue;
			if(issue) pend_row_q <= row_cnt_q;

			if(pend_block) begin
				skid_q <= 1'b1;
				skid_row_q <= pend_row_q;
				skid_data_q <= i_sram_rd_data;
			end else if(skid_ok) begin
				skid_q <= 1'b0;
			end

			case(state_q)
				S_IDLE: begin
					row_cnt_q <= {ROW_W{1'b0}};
					if(i_start && (i_active_rows != {ROW_W{1'b0}}) && (i_active_rows <= ARRAY_ROWS)) begin
						state_q <= S_RUN;
					end
				end

				S_RUN: begin
					if(issue && agu_k_last) begin
						if(agu_window_last) state_q <= S_DRAIN;
						else row_cnt_q <= row_cnt_q + 1'b1;
					end
				end

				// Xong khi khong con byte tren duong va skid rong (sau canh clock nay)
				S_DRAIN: begin
					if(!pend_q && !skid_q) state_q <= S_DONE;
				end

				S_DONE: begin
					o_done <= 1'b1;
					if(!i_start) begin
						state_q <= S_IDLE;
					end
				end

				default: begin
					state_q <= S_IDLE;
				end
			endcase
		end
	end

endmodule
`default_nettype wire
