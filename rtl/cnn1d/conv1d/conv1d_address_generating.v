module conv1d_address_generating #(
	parameter ADDR_W = 16,
	parameter CFG_W = 16
)(
	input wire i_clk,
	input wire i_reset,
	
	input wire i_start, // Start one address-generation job
	
	input wire i_advance, // Địa chỉ hiện tại đã được dùng
	
	input wire [ADDR_W-1:0] i_base_addr, // Tensor bắt đầu ở đâu trong SRAM
	input wire [CFG_W-1:0] i_k_len,		 // Kw x Cin
	input wire [ADDR_W-1:0] i_window_step,// stride x Cin
	input wire [CFG_W-1:0] i_num_windows, // Số output positions cần generate
	
	// Address output
	output wire [ADDR_W-1:0] o_addr,
	output wire o_valid,
	
	// Metadata
	output wire o_k_first,
	output wire o_k_last,
	output wire o_window_first,
	output wire o_window_last,
	
	// Status
	output wire o_busy,
	output reg o_done
);
	// CONV1D flattened address
	// 
	// Tensor layout: X[time][channel]
	// 
	// Memory layout:	addr = time * Cin + channel
	// 
	// For one sliding window:
	//
	// 	--> window_base = out_pos * stride * Cin
	// 
	// k_idx already represents flattened
	//
	// 	--> k_idx = kw_idx * Cin + cin_idx
	// 	--> addr = base_addr + out_pos * stride * Cin + k_idx
	
	reg [ADDR_W-1:0] window_base_q;
	
	reg [CFG_W-1:0] k_cnt_q;
	reg [CFG_W-1:0] window_cnt_q;
	
	reg [CFG_W-1:0] k_len_q;
	reg [CFG_W-1:0] num_windows_q;
	
	reg [ADDR_W-1:0] window_step_q;
	
	reg busy_q;
	
	// Outputs
	assign o_busy = busy_q;
	assign o_valid = busy_q;
	
	// Current SRAM address
	//
	// addr = window_base + k
	// 
	// Only ONE address addition
	assign o_addr = window_base_q + k_cnt_q;
	
	// Metadata
	assign o_k_first = busy_q && (k_cnt_q == 0);
	assign o_k_last = busy_q && (k_cnt_q == k_len_q - 1'b1);
	assign o_window_first = busy_q && (window_cnt_q == 0);
	assign o_window_last = busy_q && (window_cnt_q == num_windows_q - 1'b1);
	
	// AGU control
	always @(posedge i_clk) begin
		if(i_reset) begin
			window_base_q <= 0;
			k_cnt_q <= 0;
			window_cnt_q <= 0;
			k_len_q <= 0;
			num_windows_q <= 0;
			window_step_q <= 0;
			busy_q <= 1'b0;
			o_done <= 1'b0;
		end else begin
			// done is a pulse
			o_done <= 1'b0;
			
			// Start
			if(i_start && !busy_q) begin
				window_base_q <= i_base_addr;
				
				k_cnt_q <= 0;
				window_cnt_q <= 0;
				
				k_len_q <= i_k_len;
				num_windows_q <= i_num_windows;
				
				window_step_q <= i_window_step;
				
				busy_q <= 1'b1;
			end else if(busy_q && i_advance) begin
				// Last K of current window
				if(k_cnt_q == k_len_q - 1'b1) begin
					// last window
					if(window_cnt_q == num_windows_q - 1'b1) begin
						busy_q <= 1'b0;
						o_done <= 1'b1;
					end
					
					// Next window
					else begin
						window_cnt_q <= window_cnt_q + 1'b1;
						k_cnt_q <= 0;
						
						// Next sliding-window base
						//
						// base_next = base_current + stride * Cin
						//
						// stride*Cin đã được tính trước đó: window_step
						
						window_base_q <= window_base_q + window_step_q;
					end
				end
				
				// Next K within same window
				else begin
					k_cnt_q <= k_cnt_q + 1'b1;
				end
			end
		end
	end
endmodule
