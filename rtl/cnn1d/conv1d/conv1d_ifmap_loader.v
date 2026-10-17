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

	localparam S_IDLE = 3'd0;
	localparam S_REQ = 3'd1;
	localparam S_WAIT = 3'd2;
	localparam S_WRITE = 3'd3;
	localparam S_DONE = 3'd4;
	
	reg [2:0] state_q;
	
	assign o_state_dbg = state_q;
	assign o_busy = (state_q != S_IDLE) && (state_q != S_DONE);
	
	// Current physical row (khai báo trước wr_ok vì wr_ok dùng nó)
	reg [ROW_W-1:0] row_cnt_q;

	// Lane đích còn chỗ thì mới ghi
	wire wr_ok = ~i_lane_full[row_cnt_q];

	// AGU
	wire [ADDR_W-1:0] agu_addr;
	wire agu_valid;
	wire agu_busy;
	wire agu_done;
	wire agu_k_first;
	wire agu_k_last;
	wire agu_window_first;
	wire agu_window_last;
	
	// AGU control
	wire agu_start;
	wire agu_advance;
	
	assign agu_start = (state_q == S_IDLE) && i_start;
	assign agu_advance = (state_q == S_WRITE) && wr_ok; // Chỉ tăng địa chỉ khi ghi được
	
	// Số window cần tạo = số active row
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
	
	// SRAM response hold register
	reg signed [DATA_W-1:0] rd_data_q;
	
	// SRAM request
	assign o_sram_rd_en = (state_q == S_REQ) && agu_valid;
	assign o_sram_rd_addr = agu_addr;
	
	// FIFO write
	always @(*) begin
		o_ifm_wr_en = {ARRAY_ROWS{1'b0}};
		o_ifm_wr_data = {ARRAY_ROWS*DATA_W{1'b0}};
		
		if(state_q == S_WRITE && wr_ok) begin
			o_ifm_wr_en[row_cnt_q] = 1'b1;
			o_ifm_wr_data[row_cnt_q*DATA_W +: DATA_W] = rd_data_q;
		end
	end
	
	// FSM
	always @(posedge i_clk) begin
		if(i_reset) begin
			state_q <= S_IDLE;
			row_cnt_q <= {ROW_W{1'b0}};
			rd_data_q <= {DATA_W{1'b0}};
			o_done <= 1'b0;
		end else begin
			o_done <= 1'b0;
			
			case(state_q)
				S_IDLE: begin
					row_cnt_q <= {ROW_W{1'b0}};
					
					if(i_start && (i_active_rows != {ROW_W{1'b0}}) && (i_active_rows <= ARRAY_ROWS)) begin
						state_q <= S_REQ;
					end
				end
				
				S_REQ: begin
					if(agu_valid) begin
						state_q <= S_WAIT;
					end
				end
				
				S_WAIT: begin
					if(i_sram_rd_valid) begin
						rd_data_q <= i_sram_rd_data;
						state_q <= S_WRITE;
					end
				end
				
				S_WRITE: begin
					if(wr_ok) begin
						if(agu_k_last) begin
							if(agu_window_last) begin
								state_q <= S_DONE;
							end else begin
								row_cnt_q <= row_cnt_q + 1'b1;
								state_q <= S_REQ;
							end
						end else begin
							state_q <= S_REQ;
						end
					end
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