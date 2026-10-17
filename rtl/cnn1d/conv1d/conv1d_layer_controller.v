//Tile 0:
//IFMAP base = X_base
//W base     = W_base
//
//Tile 1:
//IFMAP base = X_base
//W base     = W_base + 3*K
//
//Tile 2:
//IFMAP base = X_base + 3*window_step
//W base     = W_base
//
//Tile 3:
//IFMAP base = X_base + 3*window_step
//W base     = W_base + 3*K


module conv1d_layer_controller #(
	parameter ADDR_W = 16,
	parameter CFG_W = 16,
	
	parameter ARRAY_ROWS = 3,
	parameter ARRAY_COLS = 3,
	
	parameter ROW_W = (ARRAY_ROWS <= 1) ? 1 : $clog2(ARRAY_ROWS + 1),
	parameter COL_W = (ARRAY_COLS <= 1) ? 1 : $clog2(ARRAY_COLS + 1)
)(
	input wire i_clk,
	input wire i_reset,
	
	// =====================================================
	// LAYER CONTROL
	// =====================================================
	input wire i_start,
	
	// Conv1D layer config
	input wire [CFG_W-1:0] i_out_len,
	input wire [CFG_W-1:0] i_out_channels,
	input wire [CFG_W-1:0] i_k_len,
	input wire [ADDR_W-1:0] i_window_step,
	
	input wire [ADDR_W-1:0] i_ifmap_base_addr,
	input wire [ADDR_W-1:0] i_wgt_base_addr,
	
	// =====================================================
	// TILE TOP STATUS
	// =====================================================
	input wire i_tile_done,
	
	// =====================================================
	// TILE TOP CONTROL
	// =====================================================
	output wire o_tile_start,
	
	output wire [ADDR_W-1:0] o_ifmap_base_addr,
	output wire [ADDR_W-1:0] o_wgt_base_addr,
	
	output wire [ROW_W-1:0] o_active_rows,
	output wire [COL_W-1:0] o_active_cols,
	
	// =====================================================
	// OUTPUT TILE POSITION
	//
	// Dùng sau này cho output_writer
	// =====================================================
	output wire [CFG_W-1:0] o_out_row_base,
	output wire [CFG_W-1:0] o_out_col_base,
	
	// =====================================================
	// LAYER CONFIG TO TILE TOP
	// =====================================================
	output wire [CFG_W-1:0] o_k_len,
	output wire [ADDR_W-1:0] o_window_step,
	
	// =====================================================
	// STATUS
	// =====================================================
	output wire o_busy,
	output wire o_done,
	output wire [2:0] o_state_dbg
);

	// =====================================================
	// STATES
	// =====================================================
	localparam S_IDLE       = 3'd0;
	localparam S_TILE_START = 3'd1;
	localparam S_TILE_WAIT  = 3'd2;
	localparam S_DONE       = 3'd3;
	
	reg [2:0] state_q;
	
	assign o_state_dbg = state_q;
	assign o_busy = (state_q != S_IDLE) && (state_q != S_DONE);
	assign o_done = (state_q == S_DONE);
	assign o_tile_start = (state_q == S_TILE_START);
	
	
	// =====================================================
	// LATCHED LAYER CONFIG
	// =====================================================
	reg [CFG_W-1:0] out_len_q;
	reg [CFG_W-1:0] out_channels_q;
	reg [CFG_W-1:0] k_len_q;
	reg [ADDR_W-1:0] window_step_q;
	
	reg [ADDR_W-1:0] layer_ifmap_base_q;
	reg [ADDR_W-1:0] layer_wgt_base_q;
	
	
	// =====================================================
	// CURRENT TILE CONFIG
	// =====================================================
	reg [ADDR_W-1:0] tile_ifmap_base_q;
	reg [ADDR_W-1:0] tile_wgt_base_q;
	
	reg [ROW_W-1:0] active_rows_q;
	reg [COL_W-1:0] active_cols_q;
	
	reg [CFG_W-1:0] out_row_base_q;
	reg [CFG_W-1:0] out_col_base_q;
	
	
	// =====================================================
	// REMAINING OUTPUT SIZE
	// =====================================================
	reg [CFG_W-1:0] rows_remaining_q;
	reg [CFG_W-1:0] cols_remaining_q;
	
	
	// =====================================================
	// OUTPUT ASSIGNMENT
	// =====================================================
	assign o_ifmap_base_addr = tile_ifmap_base_q;
	assign o_wgt_base_addr = tile_wgt_base_q;
	
	assign o_active_rows = active_rows_q;
	assign o_active_cols = active_cols_q;
	
	assign o_out_row_base = out_row_base_q;
	assign o_out_col_base = out_col_base_q;
	
	assign o_k_len = k_len_q;
	assign o_window_step = window_step_q;
	
	
	// =====================================================
	// CONTROLLER
	// =====================================================
	always @(posedge i_clk or posedge i_reset) begin
		if(i_reset) begin
			state_q <= S_IDLE;
			
			out_len_q <= {CFG_W{1'b0}};
			out_channels_q <= {CFG_W{1'b0}};
			k_len_q <= {CFG_W{1'b0}};
			window_step_q <= {ADDR_W{1'b0}};
			
			layer_ifmap_base_q <= {ADDR_W{1'b0}};
			layer_wgt_base_q <= {ADDR_W{1'b0}};
			
			tile_ifmap_base_q <= {ADDR_W{1'b0}};
			tile_wgt_base_q <= {ADDR_W{1'b0}};
			
			active_rows_q <= {ROW_W{1'b0}};
			active_cols_q <= {COL_W{1'b0}};
			
			out_row_base_q <= {CFG_W{1'b0}};
			out_col_base_q <= {CFG_W{1'b0}};
			
			rows_remaining_q <= {CFG_W{1'b0}};
			cols_remaining_q <= {CFG_W{1'b0}};
		end else begin
			case(state_q)
				
				// =================================================
				// IDLE
				// =================================================
				S_IDLE: begin
					if(i_start && (i_out_len != {CFG_W{1'b0}}) && (i_out_channels != {CFG_W{1'b0}}) && (i_k_len != {CFG_W{1'b0}})) begin
						out_len_q <= i_out_len;
						out_channels_q <= i_out_channels;
						k_len_q <= i_k_len;
						window_step_q <= i_window_step;
						
						layer_ifmap_base_q <= i_ifmap_base_addr;
						layer_wgt_base_q <= i_wgt_base_addr;
						
						tile_ifmap_base_q <= i_ifmap_base_addr;
						tile_wgt_base_q <= i_wgt_base_addr;
						
						out_row_base_q <= {CFG_W{1'b0}};
						out_col_base_q <= {CFG_W{1'b0}};
						
						rows_remaining_q <= i_out_len;
						cols_remaining_q <= i_out_channels;
						
						if(i_out_len > ARRAY_ROWS) active_rows_q <= ARRAY_ROWS;
						else active_rows_q <= i_out_len;
						
						if(i_out_channels > ARRAY_COLS) active_cols_q <= ARRAY_COLS;
						else active_cols_q <= i_out_channels;
						
						state_q <= S_TILE_START;
					end
				end
				
				
				// =================================================
				// TILE START
				//
				// Pulse o_tile_start 1 cycle.
				// Config đã ổn định trước khi start.
				// =================================================
				S_TILE_START: begin
					state_q <= S_TILE_WAIT;
				end
				
				
				// =================================================
				// TILE WAIT
				// =================================================
				S_TILE_WAIT: begin
					if(i_tile_done) begin
						
						// -----------------------------------------
						// Còn output channel tile tiếp theo
						// trong cùng output-position group
						// -----------------------------------------
						if(cols_remaining_q > active_cols_q) begin
							cols_remaining_q <= cols_remaining_q - ARRAY_COLS;
							out_col_base_q <= out_col_base_q + ARRAY_COLS;
							
							// W layout:
							// W[Cout][K]
							//
							// next Cout tile:
							// base += ARRAY_COLS * K
							tile_wgt_base_q <= tile_wgt_base_q + ARRAY_COLS*k_len_q;
							
							if((cols_remaining_q - ARRAY_COLS) > ARRAY_COLS) active_cols_q <= ARRAY_COLS;
							else active_cols_q <= cols_remaining_q - ARRAY_COLS;
							
							state_q <= S_TILE_START;
						end
						
						
						// -----------------------------------------
						// Hết column tile nhưng còn row tile
						// -----------------------------------------
						else if(rows_remaining_q > active_rows_q) begin
							rows_remaining_q <= rows_remaining_q - ARRAY_ROWS;
							cols_remaining_q <= out_channels_q;
							
							out_row_base_q <= out_row_base_q + ARRAY_ROWS;
							out_col_base_q <= {CFG_W{1'b0}};
							
							// IFMAP:
							// next output position group:
							//
							// base += ARRAY_ROWS * window_step
							tile_ifmap_base_q <= tile_ifmap_base_q + ARRAY_ROWS*window_step_q;
							
							// Weight quay lại Cout = 0
							tile_wgt_base_q <= layer_wgt_base_q;
							
							if((rows_remaining_q - ARRAY_ROWS) > ARRAY_ROWS) active_rows_q <= ARRAY_ROWS;
							else active_rows_q <= rows_remaining_q - ARRAY_ROWS;
							
							if(out_channels_q > ARRAY_COLS) active_cols_q <= ARRAY_COLS;
							else active_cols_q <= out_channels_q;
							
							state_q <= S_TILE_START;
						end
						
						
						// -----------------------------------------
						// Không còn tile nào
						// -----------------------------------------
						else begin
							state_q <= S_DONE;
						end
					end
				end
				
				
				// =================================================
				// DONE
				// =================================================
				S_DONE: begin
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