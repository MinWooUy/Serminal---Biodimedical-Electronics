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
//
// Co layer i_flags (chot khi i_layer_start). FC = conv voi Kw = 1 va out_len nho; cac co
// chi bat them buoc o sau mang PE, duong tinh van la mot:
//   bit0 out_int32   : bo requant; ghi gia tri int32 tho cua hang 0 (little-endian, 4 byte/kenh)
//                      vao bank GHI tai dia chi 4*kenh. Can out_len = 1.
//   bit1 argmax      : (cung bit0) sau layer ghi chi so kenh lon nhat (u8) tai dia chi 4*cout.
//                      So sanh '>' chat theo thu tu kenh tang -> hoa thi chon chi so nho nhat.
//   bit2 row_combine : 3 hang PE cua cung mot kenh duoc ghep lam mot (FC1 cua mo hinh ECG):
//                        acc = rsr((P1 << 7) + P0, SG) + P2 + bias ; h = clamp(rsr(acc, SO), 0, 127)
//                      P_r = tong cua hang r (preload bias da cong vao moi hang, nen tru 129*bias).
//                      SO = shift kenh (dia chi shift 0..COUT_MAX-1), SG = shift "truoc"
//                      (dia chi shift COUT_MAX + kenh). Can out_len = 3, ARRAY_ROWS >= 3.
//   bit3 gap_in      : (cung bit2) truoc khi chay, dong goi gap_mem vao bank GHI roi swap:
//                        hang 0 = gap & 127, hang 1 = gap >> 7 (bao hoa 127), moi hang dai k_len,
//                        buoc hang = window_step; j >= so kenh GAP thi ghi 0. Hang 2: j < so kenh
//                        GAP ghi 0, phan j >= so kenh GAP do host ghi truoc (vd 4 dac trung RR).
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
  input wire [3:0] i_flags, // xem dau file: out_int32, argmax, row_combine, gap_in

  // Đọc kết quả GAP
  input wire [CFG_W-1:0] i_gap_rd_idx,  
  output wire [15:0] o_gap_rd_data,

	input wire i_dbg_rd_en,
	input wire [ADDR_W-1:0] i_dbg_rd_addr,
	output wire signed [7:0] o_dbg_rd_data,
	output wire o_dbg_rd_valid,

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
	reg [3:0] flags_q;
	wire f_i32 = flags_q[0];
	wire f_argmax = flags_q[1];
	wire f_comb = flags_q[2];
	reg [CFG_W-1:0] gap_cout_q; // so kenh cua layer GAP gan nhat (cho gap_in)

	localparam CH_W = (COUT_MAX <= 1) ? 1 : $clog2(COUT_MAX);
	localparam R1 = (ARRAY_ROWS >= 2) ? 1 : 0;
	localparam R2 = (ARRAY_ROWS >= 3) ? 2 : 0;

	localparam L_IDLE = 3'd0, L_PAD = 3'd1, L_RUN = 3'd2, L_FIN = 3'd3,
	           L_GPACK = 3'd4, L_GSWAP = 3'd5, L_PRED = 3'd6;
	reg [2:0] lstate;

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
	reg xs_done; // buoc sau PE (out_int32) da ghi xong tile
	wire tile_written = writer_done | xs_done;
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
		else if(tile_written) commit_pending <= 1'b1;
		else if(tile_commit) commit_pending <= 1'b0;
	end

	// =====================================================
	// Bias / shift RAM theo kenh toan cuc
	//   shift dia chi 0..COUT_MAX-1        : shift requant cua kenh (SO)
	//   shift dia chi COUT_MAX..2*COUT_MAX-1: shift "truoc" cua kenh (SG, chi dung voi row_combine)
	// =====================================================
	reg [ACC_W-1:0] bias_mem [0:COUT_MAX-1];
	reg [DATA_W-1:0] shift_mem [0:COUT_MAX-1];
	reg [DATA_W-1:0] preshift_mem [0:COUT_MAX-1];
	reg [ACC_W-1:0] bias_rd_data;

	wire tile_bias_rd_en;
	wire [COL_W-1:0] tile_bias_rd_idx;

	// Cong doc bias dung chung: PSM (preload dau tile) va buoc row_combine (sau tile) khong trung nhau
	wire xs_rd_en;
	wire [CFG_W-1:0] xs_ch;
	wire bias_rd_en_any = tile_bias_rd_en | xs_rd_en;
	wire [CFG_W-1:0] bias_rd_addr_any = xs_rd_en ? xs_ch : (ctrl_col_base + tile_bias_rd_idx);
	wire shift_hi = i_shift_wr_addr[CH_W];
	reg [DATA_W-1:0] preshift_rd, so_rd;

	always @(posedge i_clk) begin
		if(i_bias_wr_en)  bias_mem[i_bias_wr_addr[CH_W-1:0]] <= i_bias_wr_data;
		if(i_shift_wr_en && !shift_hi) shift_mem[i_shift_wr_addr[CH_W-1:0]] <= i_shift_wr_data;
		if(i_shift_wr_en && shift_hi) preshift_mem[i_shift_wr_addr[CH_W-1:0]] <= i_shift_wr_data;
		if(bias_rd_en_any) bias_rd_data <= bias_mem[bias_rd_addr_any[CH_W-1:0]];
		if(xs_rd_en) begin
			preshift_rd <= preshift_mem[xs_ch[CH_W-1:0]];
			so_rd <= shift_mem[xs_ch[CH_W-1:0]];
		end
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
	wire [ARRAY_ROWS*ARRAY_COLS*ACC_W-1:0] raw_tile; // int32 tho cua tile (giu den tile sau)
	wire post_valid;

	// Ghi vao bank GHI cua ifmap: pad (dau layer) | gap_in | writer | int32 | argmax | host
	wire pad_wr_en;
	wire [ADDR_W-1:0] pad_wr_addr;
	wire pack_wr_en;
	wire [ADDR_W-1:0] pack_wr_addr;
	wire [DATA_W-1:0] pack_wr_data;
	wire wr_en_w;
	wire [OUT_ADDR_W-1:0] wr_addr_w;
	wire [DATA_W-1:0] wr_data_w;
	wire i32_wr_en;
	wire [ADDR_W-1:0] i32_wr_addr;
	wire [DATA_W-1:0] i32_wr_data;
	wire pred_wr_en;
	wire [ADDR_W-1:0] pred_wr_addr;
	wire [DATA_W-1:0] pred_wr_data;

	wire ifmap_wr_en = pad_wr_en | pack_wr_en | wr_en_w | i32_wr_en | pred_wr_en | i_x_wr_en;
	wire [ADDR_W-1:0] ifmap_wr_addr = pad_wr_en  ? pad_wr_addr :
	                                  pack_wr_en ? pack_wr_addr :
	                                  wr_en_w    ? {{(ADDR_W-OUT_ADDR_W){1'b0}}, wr_addr_w} :
	                                  i32_wr_en  ? i32_wr_addr :
	                                  pred_wr_en ? pred_wr_addr : i_x_wr_addr;
	wire signed [DATA_W-1:0] ifmap_wr_data = pad_wr_en  ? 8'sd0 :
	                                         pack_wr_en ? pack_wr_data :
	                                         wr_en_w    ? wr_data_w :
	                                         i32_wr_en  ? i32_wr_data :
	                                         pred_wr_en ? pred_wr_data : i_x_wr_data;

	wire pack_swap = (lstate == L_GSWAP);
	wire auto_swap = ((lstate == L_FIN) && !gap_q) || pack_swap;

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
		.o_tile(raw_tile), .o_tile_valid(),

		.o_busy(tile_busy), .o_done(),
		.o_state_dbg(), .o_core_state_dbg(),
		.o_ifmap_loader_state(), .o_wgt_loader_state(),
		.o_ifmap_read_bank(), .o_ifmap_write_bank(),
		.o_wgt_read_bank(), .o_wgt_write_bank(),

		.i_post_shift(post_shift),
		.i_post_relu(1'b1),
		.o_post_tile(post_tile),
		.o_post_valid(post_valid),
		.o_cfg_err(),

		.i_dbg_rd_en(i_dbg_rd_en), .i_dbg_rd_addr(i_dbg_rd_addr),
		.o_dbg_rd_data(o_dbg_rd_data), .o_dbg_rd_valid(o_dbg_rd_valid)
	);

	// =====================================================
	// Buoc sau PE cho FC (chi khi bat co; conv thuong di thang post -> writer)
	//   row_combine: moi kenh 2 chu ky (doc bias/SG/SO -> tinh h), xong ca tile thi phat writer
	//   out_int32  : moi kenh 4 chu ky (ghi 4 byte), cap nhat argmax o byte 0
	// =====================================================
	function signed [39:0] rsr40; // (a + 2^(s-1)) >>> s, giong rshift_round; s <= 0 -> giu nguyen
		input signed [39:0] a;
		input [7:0] s;
		reg [4:0] amt;
		begin
			if (s == 8'd0 || s[7]) rsr40 = a;
			else if (s >= 8'd32) rsr40 = 40'sd0;
			else begin
				amt = s[4:0] - 5'd1;
				rsr40 = ((a >>> amt) + 40'sd1) >>> 1;
			end
		end
	endfunction

	localparam XS_IDLE = 2'd0, XS_RD = 2'd1, XS_COMB = 2'd2, XS_I32 = 2'd3;
	reg [1:0] xs;
	reg [COL_W-1:0] xs_c;
	reg [1:0] xs_b;
	reg comb_fire;
	reg [ARRAY_ROWS*ARRAY_COLS*8-1:0] comb_tile;
	reg signed [ACC_W-1:0] best_q;
	reg [7:0] best_idx_q;
	reg am_first;

	assign xs_ch = ctrl_col_base + xs_c;
	assign xs_rd_en = (xs == XS_RD);

	wire signed [ACC_W-1:0] t0 = raw_tile[(0 *ARRAY_COLS + xs_c)*ACC_W +: ACC_W];
	wire signed [ACC_W-1:0] t1 = raw_tile[(R1*ARRAY_COLS + xs_c)*ACC_W +: ACC_W];
	wire signed [ACC_W-1:0] t2 = raw_tile[(R2*ARRAY_COLS + xs_c)*ACC_W +: ACC_W];
	wire signed [39:0] b40 = {{8{bias_rd_data[ACC_W-1]}}, bias_rd_data};
	// Moi hang da duoc preload +bias, nen 128*T1 + T0 - 129*bias = 128*P1 + P0
	wire signed [39:0] comb_x = ({{8{t1[ACC_W-1]}}, t1} <<< 7) + {{8{t0[ACC_W-1]}}, t0} - (b40 <<< 7) - b40;
	// T2 = P2 + bias: dung la "+ P2 + bias" cua cong thuc
	wire signed [39:0] comb_y = rsr40(comb_x, preshift_rd) + {{8{t2[ACC_W-1]}}, t2};
	wire signed [39:0] comb_q = rsr40(comb_y, so_rd);
	wire [7:0] comb_h = comb_q[39] ? 8'd0 : (comb_q > 40'sd127) ? 8'd127 : comb_q[7:0];

	wire tile_ready = post_valid && (lstate == L_RUN);
	assign i32_wr_en = (xs == XS_I32);
	assign i32_wr_addr = {xs_ch[ADDR_W-3:0], xs_b};
	assign i32_wr_data = t0[xs_b*8 +: 8];

	always @(posedge i_clk or posedge i_reset) begin
		if(i_reset) begin
			xs <= XS_IDLE; xs_c <= 0; xs_b <= 0;
			comb_fire <= 1'b0; xs_done <= 1'b0; comb_tile <= 0;
			best_q <= 0; best_idx_q <= 0; am_first <= 1'b1;
		end else begin
			comb_fire <= 1'b0;
			xs_done <= 1'b0;
			if(lstate == L_IDLE && i_layer_start) am_first <= 1'b1;
			case(xs)
				XS_IDLE: if(tile_ready && (f_comb || f_i32)) begin
					xs_c <= 0; xs_b <= 0;
					comb_tile <= 0;
					xs <= f_comb ? XS_RD : XS_I32;
				end
				XS_RD: xs <= XS_COMB; // bias_rd_data, preshift_rd, so_rd co o chu ky sau
				XS_COMB: begin
					comb_tile[xs_c*8 +: 8] <= comb_h; // hang 0 cua tile
					if(xs_c == ctrl_cols - 1'b1) begin
						comb_fire <= 1'b1;
						xs <= XS_IDLE;
					end else begin
						xs_c <= xs_c + 1'b1;
						xs <= XS_RD;
					end
				end
				XS_I32: begin
					if(xs_b == 2'd0 && (am_first || t0 > best_q)) begin
						best_q <= t0;
						best_idx_q <= xs_ch[7:0];
						am_first <= 1'b0;
					end
					if(xs_b == 2'd3) begin
						xs_b <= 0;
						if(xs_c == ctrl_cols - 1'b1) begin
							xs_done <= 1'b1;
							xs <= XS_IDLE;
						end else xs_c <= xs_c + 1'b1;
					end else xs_b <= xs_b + 1'b1;
				end
			endcase
		end
	end

	assign pred_wr_en = (lstate == L_PRED);
	assign pred_wr_addr = {cout_q[ADDR_W-3:0], 2'b00};
	assign pred_wr_data = best_idx_q;

	// =====================================================
	// fmap_writer
	// =====================================================
	wire gap_en_w;
	wire [CFG_W-1:0] gap_idx_w;
	wire [7:0] gap_data_w;
	wire writer_overrun;
	wire writer_start = (tile_ready && !f_comb && !f_i32) || comb_fire;

	fmap_writer #(
		.DATA_W(8), .ARRAY_ROWS(ARRAY_ROWS), .ARRAY_COLS(ARRAY_COLS),
		.CFG_W(CFG_W), .DEPTH(IFMAP_MEM_DEPTH)
	) u_writer (
		.i_clk(i_clk), .i_reset(i_reset),
		.i_start(writer_start), .i_clear_overrun(i_layer_start && lstate == L_IDLE),
		.i_tile(f_comb ? comb_tile : post_tile),
		.i_row_base(ctrl_row_base), .i_col_base(ctrl_col_base),
		.i_active_rows(f_comb ? {{(ROW_W-1){1'b0}}, 1'b1} : ctrl_rows), .i_active_cols(ctrl_cols),
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
	// Xoa khi bat dau moi layer (nhu truoc), TRU layer doc GAP (gap_in): FC1 can GAP cua conv3
	wire gap_clear = (lstate == L_IDLE) && i_layer_start && !i_flags[3];
	always @(posedge i_clk) begin
		if(gap_clear) begin
			for(gi = 0; gi < COUT_MAX; gi = gi + 1) gap_mem[gi] <= 16'd0;
		end else if(gap_en_w) begin
			gap_mem[gap_idx_w[CH_W-1:0]] <= gap_mem[gap_idx_w[CH_W-1:0]] + {8'd0, gap_data_w};
		end
	end
	// Mot cong doc: luc dong goi (gap_in) dung chi so cua bo dem, con lai cua host (READ_GAP)
	reg [1:0] pk_r;
	reg [CFG_W-1:0] pk_j;
	wire [CFG_W-1:0] gap_sel = (lstate == L_GPACK) ? pk_j : i_gap_rd_idx;
	wire [15:0] gap_val = gap_mem[gap_sel[CH_W-1:0]];
	assign o_gap_rd_data = gap_val;

	// =====================================================
	// gap_in: dong goi gap_mem thanh 3 hang (lo, hi, aux) vao bank GHI
	// =====================================================
	wire pk_in = (pk_j < gap_cout_q);
	wire [7:0] pk_lo = {1'b0, gap_val[6:0]};
	wire [7:0] pk_hi = (gap_val[15:14] != 2'b00) ? 8'd127 : {1'b0, gap_val[13:7]};
	assign pack_wr_en = (lstate == L_GPACK) && !(pk_r == 2'd2 && !pk_in);
	assign pack_wr_addr = pk_r * window_step_q + pk_j;
	assign pack_wr_data = !pk_in ? 8'd0 : (pk_r == 2'd0) ? pk_lo : (pk_r == 2'd1) ? pk_hi : 8'd0;

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
			pool_q <= 0; gap_q <= 0; flags_q <= 0; gap_cout_q <= 0;
			pad_i <= 0; pk_r <= 0; pk_j <= 0;
			done_q <= 0;
		end else begin
			done_q <= 1'b0;
			case(lstate)
				L_IDLE: if(i_layer_start) begin
					out_len_q <= i_out_len; cout_q <= i_cout; k_len_q <= i_k_len;
					window_step_q <= i_window_step;
					pool_q <= i_pool_en; gap_q <= i_gap_en;
					flags_q <= i_flags;
					if(i_gap_en) gap_cout_q <= i_cout;
					pad_q <= i_gap_en ? {CFG_W{1'b0}} : i_pad_next;
					pad_i <= 0; pk_r <= 0; pk_j <= 0;
					if(i_flags[3]) lstate <= L_GPACK;
					else lstate <= (!i_gap_en && i_pad_next != 0) ? L_PAD : L_RUN;
				end

				// gap_in: 3 hang x k_len byte vao bank GHI, roi swap thanh bank DOC
				L_GPACK: begin
					if(pk_j == k_len_q - 1'b1) begin
						pk_j <= 0;
						if(pk_r == 2'd2) lstate <= L_GSWAP;
						else pk_r <= pk_r + 1'b1;
					end else pk_j <= pk_j + 1'b1;
				end

				L_GSWAP: lstate <= (pad_q != 0) ? L_PAD : L_RUN;

				L_PAD: begin
					if(pad_i == pad_total - 1) lstate <= L_RUN;
					else pad_i <= pad_i + 1;
				end

				L_RUN: if(ctrl_done) lstate <= (f_i32 && f_argmax) ? L_PRED : L_FIN;

				L_PRED: lstate <= L_FIN; // ghi nhan argmax tai dia chi 4*cout

				L_FIN: begin
					done_q <= 1'b1;
					lstate <= L_IDLE;
				end
			endcase
		end
	end
endmodule
`default_nettype wire