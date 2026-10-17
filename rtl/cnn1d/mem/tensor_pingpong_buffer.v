// Khối này sẽ là SRAM n bank buffer
// 
// Bọc 2 khối tensor_sram --> double buffer
// 
// read_bank = 0
// Bank0 --> Accelerator đọc
// Bank1 <-- Loader ghi dữ liệu mới
//
//	Swap
// read_bank = 1
// Bank1 --> Accelerator đọc
// Bank0 <-- loader ghi dữ liệu mới vào
//
// Logic: 		write_bank = ~read_bank
module tensor_pingpong_buffer#(
	parameter DATA_W = 8,
	parameter DEPTH = 1024,
	parameter ADDR_W = (DEPTH <= 1) ? 1 : $clog2(DEPTH)
)(
	input wire i_clk,
	input wire i_reset,
	
	// bank control
	input wire i_swap,
	
	// write side
	input wire i_wr_en,
	input wire [ADDR_W-1:0] i_wr_addr,
	input wire signed [DATA_W-1:0] i_wr_data,
	
	// read side
	input wire i_rd_en,
	input wire [ADDR_W-1:0] i_rd_addr,
	
	output wire signed[DATA_W-1:0] o_rd_data,
	output wire o_rd_valid,

	// Debug / STATUS
	output wire o_read_bank,
	output wire o_write_bank
);

	// Bank ownership
	//
	// 0: 
	// 	READ bank0
	// 	WRITE bank1
	// 1:
	// 	READ bank1
	//		WRITE bank0
	reg read_bank_q;
	
	always @(posedge i_clk) begin
		if(i_reset) begin
			read_bank_q <= 1'b0;
		end else begin
			if(i_swap) begin
				read_bank_q <= ~read_bank_q;
			end
		end
	end
	
	assign o_read_bank = read_bank_q;
	assign o_write_bank = ~read_bank_q;
	
	// Bank 0 control
	wire bank0_wr_en;
	wire bank0_rd_en;
	
	assign bank0_wr_en = i_wr_en && (read_bank_q == 1'b1);
	assign bank0_rd_en = i_rd_en && (read_bank_q == 1'b0);
	
	// Bank 1 control
	wire bank1_wr_en;
	wire bank1_rd_en;
	
	assign bank1_wr_en = i_wr_en && (read_bank_q == 1'b0);
	assign bank1_rd_en = i_rd_en && (read_bank_q == 1'b1);
	
	// Bank 0 output
	wire signed [DATA_W-1:0] bank0_rd_data;
	wire bank0_rd_valid;
	
	// Bank 1 output
	wire signed [DATA_W-1:0] bank1_rd_data;
	wire bank1_rd_valid;
	
	// SRAM bank 0
	tensor_sram #(
		.DATA_W(DATA_W),
		.DEPTH(DEPTH),
		.ADDR_W(ADDR_W)
	) u_bank0(
		.i_clk(i_clk), .i_reset(i_reset),
		.i_wr_en(bank0_wr_en),
		.i_wr_addr(i_wr_addr),
		.i_wr_data(i_wr_data),
		.i_rd_en(bank0_rd_en),
		.i_rd_addr(i_rd_addr),
		
		.o_rd_data(bank0_rd_data),
		.o_rd_valid(bank0_rd_valid)
	);
	
	// SRAM bank 1
	tensor_sram #(
		.DATA_W(DATA_W),
		.DEPTH(DEPTH),
		.ADDR_W(ADDR_W)
	) u_bank1(
		.i_clk(i_clk), 
		.i_reset(i_reset),
		.i_wr_en(bank1_wr_en),
		.i_wr_addr(i_wr_addr),
		.i_wr_data(i_wr_data),
		.i_rd_en(bank1_rd_en),
		.i_rd_addr(i_rd_addr),
		
		.o_rd_data(bank1_rd_data),
		.o_rd_valid(bank1_rd_valid)
	);
	
	// READ RESPONSE MUX
	// 
	// Không mux theo read_bank_q hiện tại.
	// 
	// SRAM có read latency, nên bank ownership có thể đã swap trước khi response cũ quay về
	//
	// Dùng rd_valid của từng SRAM để biết response đến từ bank nào
	
	assign o_rd_valid = bank0_rd_valid | bank1_rd_valid;
	assign o_rd_data = bank0_rd_valid ? bank0_rd_data : bank1_rd_valid ? bank1_rd_data : {DATA_W{1'b0}};
endmodule
