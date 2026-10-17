module ifmap_feeder #(
	parameter DATA_W = 8,
	parameter DEPTH = 16,
	parameter CNT_W = 5,
	parameter K_W = 16,
	parameter ARRAY_ROWS = 3,
	parameter ROW_W = (ARRAY_ROWS <= 1) ? 1 : $clog2(ARRAY_ROWS + 1)
)(
	input wire i_clk,
	input wire i_reset,
	
	// preload FIFO
	input wire [ARRAY_ROWS-1:0] i_wr_en,
	input wire [ARRAY_ROWS*DATA_W-1:0] i_wr_data,
	
	// input lockstep
	input wire i_feed_step,
	
	// control
	input wire i_start_tile,
	input wire [K_W-1:0] i_k_len,
	input wire [ROW_W-1:0] i_active_rows,
	
	// output lockstep
	output wire o_ready,
	
	// status
	output wire [ARRAY_ROWS-1:0] o_mem_full,
	output wire [ARRAY_ROWS-1:0] o_mem_empty,
	output wire o_busy,
	output wire o_done,
	
	// SA
	output wire [ARRAY_ROWS*DATA_W-1:0] o_data,
	output wire [ARRAY_ROWS-1:0] o_valid,
	output wire [ARRAY_ROWS-1:0] o_first,
	output wire [ARRAY_ROWS-1:0] o_last
);

	// K-counter
	// 1 bộ dùng chung cho tất cả active row
	reg [K_W-1:0] k_cnt, k_len_q;
	reg [ROW_W-1:0] active_rows_q;
	reg busy_q;
	reg done_q;
	
	// Lockstep
	reg all_ready;
	wire rd_fire_all;
	
	integer r;
	
	always @(*) begin
		all_ready = 1'b1;
		
		for(r = 0; r < ARRAY_ROWS; r = r + 1) begin
			if((r < active_rows_q) && o_mem_empty[r]) begin
				all_ready = 1'b0;
			end
		end
	end
	
	assign o_ready = busy_q & all_ready;
	assign rd_fire_all = busy_q & all_ready & i_feed_step;
	
	assign o_busy = busy_q;
	assign o_done = done_q;
	
	// Tile control
	always @(posedge i_clk or posedge i_reset) begin
		if(i_reset) begin
			busy_q <= 1'b0;
			done_q <= 1'b0;
			k_cnt <= {K_W{1'b0}};
			k_len_q <= {K_W{1'b0}};
			active_rows_q <= {ROW_W{1'b0}};
		end else begin
			// done chỉ pulse 1 cycle
			done_q <= 1'b0;
			
			// Bắt đầu tile mới
			if(i_start_tile && !busy_q && (i_k_len != {K_W{1'b0}}) && (i_active_rows != {ROW_W{1'b0}}) && (i_active_rows <= ARRAY_ROWS)) begin
				busy_q <= 1'b1;
				k_cnt <= {K_W{1'b0}};
				k_len_q <= i_k_len;
				active_rows_q <= i_active_rows;
			end
			
			// Chỉ tăng k khi tất cả active FIFO thực sự đọc cùng nhau
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
	
	assign first_raw = rd_fire_all && (k_cnt == {K_W{1'b0}});
	assign last_raw = rd_fire_all && (k_cnt == k_len_q - 1'b1);
	
	// FEED LANE
	genvar i;
	generate
		for(i = 0; i < ARRAY_ROWS; i = i + 1) begin: g_lane
			wire lane_rd_en;
			
			assign lane_rd_en = rd_fire_all && (i < active_rows_q);
			
			feed_lane #(
				.DATA_W(DATA_W),
				.CNT_W(CNT_W),
				.DEPTH(DEPTH),
				.LANE_N(i)
			) u_lane(
				.clk(i_clk),
				.reset(i_reset),
				
				.i_first(first_raw),
				.i_last(last_raw),
				
				.wr_en(i_wr_en[i]),
				.rd_en(lane_rd_en),
				.wr_data(i_wr_data[i*DATA_W +: DATA_W]),
				
				.mem_empty(o_mem_empty[i]),
				.mem_full(o_mem_full[i]),
				
				.o_valid(o_valid[i]),
				.o_first(o_first[i]),
				.o_last(o_last[i]),
				.o_data(o_data[i*DATA_W +: DATA_W])
			);
		end
	endgenerate

endmodule
