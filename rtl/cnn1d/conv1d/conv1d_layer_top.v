`default_nettype none
// conv1d_layer_top: tu chay TRON MOT LAYER Conv1D (bias + requant + ReLU [+ MaxPool2] [+ GAP]).
//
//   layer_controller --tile_start--> conv1d_tile_top (feeder + PE + PSM + post)
//          ^                                  | o_post_valid
//          | tile_commit                     v
//          +------------------------- fmap_writer --> ghi vao bank GHI cua ifmap ping-pong
//                                              \--> (GAP) cong don vao gap_mem
//
// Quy uoc bo nho (ca vao lan ra deu "channel-last"):  X[t][c]  -> dia chi = (t + pad)*C + c
//   - Layer doc input tu bank DOC cua ifmap ping-pong, ghi output vao bank GHI.
//   - Cuoi layer (khong phai GAP) tu dong swap bank: output thanh input cua layer ke tiep.
//   - Truoc khi chay tile dau, ghi 0 vao 2*pad_next*Cout o padding cua output.
//
// Host/tb nap: input lop dau (x_wr + x_swap), weight [Cout][K] (w_wr + w_swap), bias, shift.
module conv1d_layer_top #(
  parameter DATA_W = 8,
  parameter ACC_W = 32,
  parameter ARRAY_ROWS = 3,
  parameter ARRAY_COLS = 3,
  parameter IFMAP_MEM_DEPTH = 4096,
  parameter WGT_MEM_DEPTH = 16384,
  parameter FEEDER_DEPTH = 1024,
  parameter COUT_MAX = 64,
  parameter ADDR_W = 16,
  parameter CFG_W = 16,
  parameter ROW_W = (ARRAY_ROWS <= 1) ? 1 : $clog2(ARRAY_ROWS + 1),
  parameter COL_W = (ARRAY_COLS <= 1) ? 1 : $clog2(ARRAY_COLS + 1)
)(
  input wire i_clk, i_reset,

  // Nạp dữ liệu (host/tb)
  input wire i_x_wr_en,
  input wire [ADDR_W-1:0] i_x_wr_addr,
  input wire signed [DATA_W-1:0] i_x_wr_data,
  input wire i_x_swap,

  input wire i_w_wr_en,
  input wire [ADDR_W-1:0] i_w_wr_addr,
  input wire signed [DATA_W-1:0] i_w_wr_data,
  input wire i_w_swap,

  input wire i_bias_wr_en,
  input wire [ADDR_W-1:0] i_bias_wr_addr,
  input wire [ACC_W-1:0] i_bias_wr_data,

  input wire i_shift_wr_en,
  input wire [ADDR_W-1:0] i_shift_wr_addr,
  input wire [DATA_W-1:0] i_shift_wr_data,

  // Cấu hình layer
  input wire i_layer_start,
  input wire [CFG_W-1:0] i_out_len, // Số vị trí output trước pooling
  input wire [CFG_W-1:0] i_cout,
  input wire [CFG_W-1:0] i_k_len, // K = Kw * Cin
  input wire [ADDR_W-1:0] i_window_step, // stride * Cin
  input wire i_pool_en,
  input wire i_gap_en,
  input wire [CFG_W-1:0] i_pad_next, // Padding của layer sau (số vị trí 0 mỗi đầu)

  // Đọc kết quả GAP
  input wire [CFG_W-1:0] i_gap_rd_idx,
  
  output wire [15:0] o_gap_rd_data,

  output wire o_busy, 
  output wire o_done,  // xung 1 chu kỳ khi cả layer đã ghi xong
  output wire o_err // Overrun của fmap writer

);
  localparam OUT_ADDR_W = (IFMAP_MEM_DEPTH <= 1) ? 1 : $clog2(IFMAP_MEM_DEPTH);

	// =====================================================
	// Cau hinh layer duoc chot
	// =====================================================
	reg [CFG_W-1:0] out_len_q, cout_q, k_len_q, pad_q;
	reg [ADDR_W-1:0] window_step_q;
	reg pool_q, gap_q;

	localparam L_IDLE = 2'd0, L_PAD = 2'd1, L_RUN = 2'd2, L_FIN = 2'd3;
	reg [1:0] lstate;

	wire [CFG_W-1:0] len_post = pool_q ? (out_len_q >> 1) : out_len_q;
	wire [ROW_W-1:0] rows_per_tile = pool_q ? ((ARRAY_ROWS >= 2) ? 2 : 1) : ARRAY_ROWS;

	// =====================================================
	// Layer controller
	// =====================================================
	wire ctrl_start = (lstate == L_RUN);
	wire ctrl_tile_start, ctrl_busy, ctrl_done;
	wire [ADDR_W-1:0] ctrl_x_base, ctrl_w_base;
	wire [ROW_W-1:0] ctrl_rows;
	wire [COL_W-1:0] ctrl_cols;
	wire [CFG_W-1:0] ctrl_row_base, ctrl_col_base;
	wire [CFG_W-1:0] ctrl_k_len;
	wire [ADDR_W-1:0] ctrl_window_step;
	wire [2:0] ctrl_dbg;

	// tile da commit (ghi xong) VA tile_top da ve IDLE -> controller moi duoc phat tile ke tiep
	wire tile_busy;
	wire writer_done;
	reg commit_pending;
	wire tile_commit = commit_pending && !tile_busy;

	conv1d_layer_controller #(
		.ADDR_W(ADDR_W), .CFG_W(CFG_W), .ARRAY_ROWS(ARRAY_ROWS), .ARRAY_COLS(ARRAY_COLS)
	) u_ctrl (
		.i_clk(i_clk), .i_reset(i_reset),
		.i_start(ctrl_start),
		.i_out_len(out_len_q), .i_out_channels(cout_q),
		.i_k_len(k_len_q), .i_window_step(window_step_q),
		.i_rows_per_tile(rows_per_tile),
		.i_ifmap_base_addr({ADDR_W{1'b0}}), .i_wgt_base_addr({ADDR_W{1'b0}}),
		.i_tile_done(tile_commit),
		.o_tile_start(ctrl_tile_start),
		.o_ifmap_base_addr(ctrl_x_base), .o_wgt_base_addr(ctrl_w_base),
		.o_active_rows(ctrl_rows), .o_active_cols(ctrl_cols),
		.o_out_row_base(ctrl_row_base), .o_out_col_base(ctrl_col_base),
		.o_k_len(ctrl_k_len), .o_window_step(ctrl_window_step),
		.o_busy(ctrl_busy), .o_done(ctrl_done), .o_state_dbg(ctrl_dbg)
	);

	always @(posedge i_clk or posedge i_reset) begin
		if(i_reset) commit_pending <= 1'b0;
		else if(ctrl_tile_start) commit_pending <= 1'b0;
		else if(writer_done) commit_pending <= 1'b1;
		else if(tile_commit) commit_pending <= 1'b0;
	end

	// =====================================================
	// Bias / shift RAM theo kenh toan cuc
	// =====================================================
	reg [ACC_W-1:0] bias_mem [0:COUT_MAX-1];
	reg [DATA_W-1:0] shift_mem [0:COUT_MAX-1];
	reg [ACC_W-1:0] bias_rd_data;

	wire tile_bias_rd_en;
	wire [COL_W-1:0] tile_bias_rd_idx;

	always @(posedge i_clk) begin
		if(i_bias_wr_en)  bias_mem[i_bias_wr_addr[$clog2(COUT_MAX)-1:0]] <= i_bias_wr_data;
		if(i_shift_wr_en) shift_mem[i_shift_wr_addr[$clog2(COUT_MAX)-1:0]] <= i_shift_wr_data;
		if(tile_bias_rd_en) bias_rd_data <= bias_mem[ctrl_col_base + tile_bias_rd_idx];
	end

	// Shift cua cac cot trong tile: kenh (col_base + c). Ngoai pham vi -> 0 (cot khong dung, post che).
	reg [ARRAY_COLS*8-1:0] post_shift;
	integer sc;
	always @* begin
		post_shift = {(ARRAY_COLS*8){1'b0}};
		for(sc = 0; sc < ARRAY_COLS; sc = sc + 1)
			if((ctrl_col_base + sc) < COUT_MAX)
				post_shift[sc*8 +: 8] = shift_mem[ctrl_col_base + sc];
	end

	// =====================================================
	// Tile top
	// =====================================================
	wire [ARRAY_ROWS*ARRAY_COLS*8-1:0] post_tile;
	wire post_valid;

	// Ghi vao bank GHI cua ifmap: pad (dau layer) | writer (trong layer) | host
	wire pad_wr_en;
	wire [ADDR_W-1:0] pad_wr_addr;
	wire wr_en_w;
	wire [OUT_ADDR_W-1:0] wr_addr_w;
	wire [DATA_W-1:0] wr_data_w;

	wire ifmap_wr_en = pad_wr_en | wr_en_w | i_x_wr_en;
	wire [ADDR_W-1:0] ifmap_wr_addr = pad_wr_en ? pad_wr_addr : wr_en_w ? {{(ADDR_W-OUT_ADDR_W){1'b0}}, wr_addr_w} : i_x_wr_addr;
	wire signed [DATA_W-1:0] ifmap_wr_data = pad_wr_en ? 8'sd0 : wr_en_w ? wr_data_w : i_x_wr_data;

	wire auto_swap = (lstate == L_FIN) && !gap_q;

	conv1d_tile_top #(
		.POST_ENABLE(1),
		.ADDR_W(ADDR_W), .CFG_W(CFG_W),
		.IFMAP_MEM_DEPTH(IFMAP_MEM_DEPTH), .WGT_MEM_DEPTH(WGT_MEM_DEPTH),
		.FEEDER_DEPTH(FEEDER_DEPTH),
		.ARRAY_ROWS(ARRAY_ROWS), .ARRAY_COLS(ARRAY_COLS)
	) u_tile (
		.i_clk(i_clk), .i_reset(i_reset),
		.i_start(ctrl_tile_start),
		.i_k_len(ctrl_k_len), .i_window_step(ctrl_window_step),
		.i_active_rows(ctrl_rows), .i_active_cols(ctrl_cols),
		.i_ifmap_base_addr(ctrl_x_base), .i_wgt_base_addr(ctrl_w_base),

		.i_preload_mode(1'b1),
		.i_bias_rd_data(bias_rd_data),
		.i_drain_ready(1'b1),
		.o_drain_valid(), .o_drain_idx(), .o_drain_vec(),

		.i_ifmap_mem_wr_en(ifmap_wr_en),
		.i_ifmap_mem_wr_addr(ifmap_wr_addr),
		.i_ifmap_mem_wr_data(ifmap_wr_data),
		.i_ifmap_mem_swap(i_x_swap | auto_swap),

		.i_wgt_mem_wr_en(i_w_wr_en),
		.i_wgt_mem_wr_addr(i_w_wr_addr),
		.i_wgt_mem_wr_data(i_w_wr_data),
		.i_wgt_mem_swap(i_w_swap),

		.o_bias_rd_en(tile_bias_rd_en),
		.o_bias_rd_idx(tile_bias_rd_idx),
		.o_tile(), .o_tile_valid(),

		.o_busy(tile_busy), .o_done(),
		.o_state_dbg(), .o_core_state_dbg(),
		.o_ifmap_loader_state(), .o_wgt_loader_state(),
		.o_ifmap_read_bank(), .o_ifmap_write_bank(),
		.o_wgt_read_bank(), .o_wgt_write_bank(),

		.i_post_shift(post_shift),
		.i_post_relu(1'b1),
		.o_post_tile(post_tile),
		.o_post_valid(post_valid),
		.o_cfg_err()
	);

	// =====================================================
	// fmap_writer
	// =====================================================
	wire gap_en_w;
	wire [CFG_W-1:0] gap_idx_w;
	wire [7:0] gap_data_w;
	wire writer_overrun;

	fmap_writer #(
		.DATA_W(8), .ARRAY_ROWS(ARRAY_ROWS), .ARRAY_COLS(ARRAY_COLS),
		.CFG_W(CFG_W), .DEPTH(IFMAP_MEM_DEPTH)
	) u_writer (
		.i_clk(i_clk), .i_reset(i_reset),
		.i_start(post_valid), .i_clear_overrun(i_layer_start && lstate == L_IDLE),
		.i_tile(post_tile),
		.i_row_base(ctrl_row_base), .i_col_base(ctrl_col_base),
		.i_active_rows(ctrl_rows), .i_active_cols(ctrl_cols),
		.i_out_channels(cout_q), .i_output_base({OUT_ADDR_W{1'b0}}),
		.i_row_pad(pad_q),
		.i_pool_en(pool_q), .i_gap_en(gap_q),
		.o_wr_en(wr_en_w), .o_wr_addr(wr_addr_w), .o_wr_data(wr_data_w),
		.o_gap_en(gap_en_w), .o_gap_idx(gap_idx_w), .o_gap_data(gap_data_w),
		.o_ready(), .o_overrun(writer_overrun), .o_done(writer_done)
	);
	assign o_err = writer_overrun;

	// =====================================================
	// GAP: cong don tong 45 vi tri moi kenh (gia tri sau ReLU: 0..127)
	// =====================================================
	reg [15:0] gap_mem [0:COUT_MAX-1];
	integer gi;
	wire gap_clear = (lstate == L_IDLE) && i_layer_start;
	always @(posedge i_clk) begin
		if(gap_clear) begin
			for(gi = 0; gi < COUT_MAX; gi = gi + 1) gap_mem[gi] <= 16'd0;
		end else if(gap_en_w) begin
			gap_mem[gap_idx_w[$clog2(COUT_MAX)-1:0]] <= gap_mem[gap_idx_w[$clog2(COUT_MAX)-1:0]] + {8'd0, gap_data_w};
		end
	end
	assign o_gap_rd_data = gap_mem[i_gap_rd_idx[$clog2(COUT_MAX)-1:0]];

	// =====================================================
	// Ghi 0 vao padding cua output (2 * pad * Cout o)
	//   vung dau : [0, pad*Cout)
	//   vung cuoi: [(len_post+pad)*Cout, (len_post+2*pad)*Cout)
	// =====================================================
	reg [31:0] pad_i;
	wire [31:0] pad_half = pad_q * cout_q;
	wire [31:0] pad_total = pad_half * 2;
	assign pad_wr_en = (lstate == L_PAD);
	assign pad_wr_addr = (pad_i < pad_half) ? pad_i[ADDR_W-1:0]
	                                        : ((len_post + pad_q)*cout_q + (pad_i - pad_half));

	// =====================================================
	// FSM layer
	// =====================================================
	reg done_q;
	assign o_done = done_q;
	assign o_busy = (lstate != L_IDLE);

	always @(posedge i_clk or posedge i_reset) begin
		if(i_reset) begin
			lstate <= L_IDLE;
			out_len_q <= 0; cout_q <= 0; k_len_q <= 0; pad_q <= 0; window_step_q <= 0;
			pool_q <= 0; gap_q <= 0;
			pad_i <= 0;
			done_q <= 0;
		end else begin
			done_q <= 1'b0;
			case(lstate)
				L_IDLE: if(i_layer_start) begin
					out_len_q <= i_out_len; cout_q <= i_cout; k_len_q <= i_k_len;
					window_step_q <= i_window_step;
					pool_q <= i_pool_en; gap_q <= i_gap_en;
					pad_q <= i_gap_en ? {CFG_W{1'b0}} : i_pad_next;
					pad_i <= 0;
					lstate <= (!i_gap_en && i_pad_next != 0) ? L_PAD : L_RUN;
				end

				L_PAD: begin
					if(pad_i == pad_total - 1) lstate <= L_RUN;
					else pad_i <= pad_i + 1;
				end

				L_RUN: if(ctrl_done) lstate <= L_FIN;

				L_FIN: begin
					done_q <= 1'b1;
					lstate <= L_IDLE;
				end
			endcase
		end
	end
endmodule
`default_nettype wire