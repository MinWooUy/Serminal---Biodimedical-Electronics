module FIFO_param #(
	parameter CNT_W = 5,
	parameter DATA_W = 8,
	parameter DEPTH = 16
)(
	input wire clk, reset,

	input wire wr_en,	rd_en,
	input wire [DATA_W - 1:0] wr_data, 
	
	output wire mem_full, mem_empty,
	output reg [DATA_W - 1:0]rd_data
);

	reg [DATA_W-1:0] mem[0:DEPTH-1]; 	// memory
	reg [CNT_W-1:0] count;					// elements counter
	
	integer i;
	
	assign mem_full = (count == DEPTH);
	assign mem_empty = (count == 0);
	
	always @(posedge clk or posedge reset) begin
		if(reset) begin
			count <= 0;
			rd_data <= 0;
			for(i = 0; i < DEPTH; i = i + 1) 
				mem[i] <= 0;
		end else begin
			if(wr_en & ~mem_full) begin
				mem[count] <= wr_data;
				count <= count + 1'b1;
			end else if(rd_en & ~mem_empty) begin
				rd_data <= mem[0];
				for(i = 0; i < DEPTH-1; i = i+1)
					mem[i] <= mem[i+1];				// Shift data
				count <= count - 1'b1;
			end
		end
	end

endmodule
