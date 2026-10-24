module weight_feeder #(
	parameter DATA_W = 8,
	parameter DEPTH = 16,
	parameter K_W = 16,
	parameter ARRAY_COLS = 3,
	parameter COL_W = (ARRAY_COLS <= 1) ? 1 : $clog2(ARRAY_COLS + 1)
)(
	input wire i_clk,
	input wire i_reset,
	
	// input lockstep
	input wire i_feed_step,
	
	// preload FIFO
	input wire [ARRAY_COLS-1:0] i_wr_en,
	input wire [ARRAY_COLS*DATA_W-1:0] i_wr_data,
	
	// control
	input wire i_start_tile,
	input wire [COL_W-1:0] i_active_cols,
	input wire [K_W-1:0] i_k_len,
	
	// output lockstep
	output wire o_ready,
	
	// status
	output wire[ARRAY_COLS-1:0] o_mem_full,
	output wire[ARRAY_COLS-1:0] o_mem_empty,
	output wire o_busy,
	output wire o_done,
	
	// SA
	output wire [ARRAY_COLS*DATA_W-1:0] o_data,
	output wire [ARRAY_COLS-1:0] o_valid,
	output wire [ARRAY_COLS-1:0] o_first,
	output wire [ARRAY_COLS-1:0] o_last
);

	// K -counter
	//
	// 1 bộ dùng chung cho các lane
	reg [K_W-1:0] k_cnt, k_len_q;
	reg busy_q;
	reg done_q;
	
	reg all_ready;
	integer c;
	always @(*)begin
		all_ready = 1'b1;
		for(c = 0; c < ARRAY_COLS; c = c + 1) begin
			if(c < i_active_cols) begin
				if(o_mem_empty[c]) all_ready = 1'b0;
			end
		end
	end
	
//	wire all_ready;
	wire rd_fire_all;
	
	//assign all_ready = &(~o_mem_empty); // Chỉ cần 1 col không có data, stall tránh lệch k
	
	assign o_ready = busy_q & all_ready;
	assign rd_fire_all = busy_q & all_ready & i_feed_step;
	
	assign o_busy = busy_q;
	assign o_done = done_q;
	
	always @(posedge i_clk or posedge i_reset) begin
		if(i_reset) begin
			busy_q <= 1'b0;
			done_q <= 1'b0;
			k_cnt <= {K_W{1'b0}};
			k_len_q <= {K_W{1'b0}};
		end else begin
			// done chỉ pulse 1 lần
			done_q <= 1'b0;
			
			// Bắt đầu 1 tile mới
			if(i_start_tile && !busy_q && (i_k_len != {K_W{1'b0}})) begin
				busy_q <= 1'b1;
				k_cnt <= {K_W{1'b0}};
				k_len_q <= i_k_len;
			end
			
			// Chỉ tăng k khi tất cả FIFO thực sự đọc
			else if(rd_fire_all) begin
				if(k_cnt == k_len_q - 1'b1) begin
					busy_q <= 1'b0;
					done_q <= 1'b1;
				end else begin
					k_cnt <= k_cnt + 1'b1;
				end
			end
		end
	end
	
	// metadata
	wire first_raw, last_raw;
	
	assign first_raw = rd_fire_all && (k_cnt == 0);
	assign last_raw = rd_fire_all && (k_cnt == k_len_q-1'b1);
	
	wire [ARRAY_COLS-1:0] lane_rd_en;

	genvar j;
	generate
		for(j = 0; j < ARRAY_COLS; j = j + 1) begin : GEN_LANE_RD
			assign lane_rd_en[j] = rd_fire_all && (j < i_active_cols);
		end
	endgenerate
	
	// FEED LANE
	genvar i;
	generate 
		for(i = 0; i < ARRAY_COLS; i=i+1) begin: g_lane
			feed_lane#(
				.DATA_W(DATA_W), .DEPTH(DEPTH), .LANE_N(i)
			) u_lane(
				.clk(i_clk), .reset(i_reset),
				
				.i_first(first_raw), .i_last(last_raw),
				
				.wr_en(i_wr_en[i]), .rd_en(lane_rd_en[i]),
				.wr_data(i_wr_data[i*DATA_W +: DATA_W]),
				
				.mem_empty(o_mem_empty[i]), .mem_full(o_mem_full[i]),
				
				.o_valid(o_valid[i]), .o_first(o_first[i]), .o_last(o_last[i]),
				.o_data(o_data[i*DATA_W +: DATA_W])
			);
		end
	endgenerate
endmodule
