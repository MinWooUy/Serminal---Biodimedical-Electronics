// FIFO vòng tròn: dữ liệu nằm yên trong RAM, hai con trỏ chạy vòng (map vào M9K).
// DEPTH phải là lũy thừa của 2. Đọc trễ 1 chu kỳ: rd_en ở chu kỳ n -> rd_data ở n+1.
// Giải thích: docs/issue/L01-ring-buffer.md
module ring_buffer #(
	parameter DATA_W = 8,
	parameter DEPTH = 16
)(
	input wire clk, reset,

	input wire wr_en,	rd_en,
	input wire [DATA_W - 1:0] wr_data,

	output wire mem_full, mem_empty,
	output reg [DATA_W - 1:0]rd_data,
	output wire [$clog2(DEPTH):0] count
);
	// Độ rộng địa chỉ suy ra từ DEPTH, không truyền từ ngoài vào.
	// (CNT_W của FIFO_param là độ rộng bộ đếm = log2 + 1, truyền nhầm sẽ hỏng địa chỉ.)
	localparam AW = $clog2(DEPTH);

	// synthesis translate_off
	initial begin
		if ((1 << AW) != DEPTH)
			$fatal(1, "ring_buffer: DEPTH=%0d khong phai luy thua cua 2", DEPTH);
	end
	// synthesis translate_on

	reg [DATA_W-1:0] mem[0:DEPTH-1]; 	// memory
  reg [AW:0] wr_ptr, rd_ptr; // write and read pointers (thêm 1 bit đánh dấu vòng)
	
  wire do_wr, do_rd;

  assign do_wr = wr_en & ~mem_full; // Đầy thì chặn ghi, kể cả khi đang đọc
  assign do_rd = rd_en & ~mem_empty;

  assign count = wr_ptr - rd_ptr; // số lượng phần tử trong bộ đệm
  assign mem_full = (wr_ptr[AW] != rd_ptr[AW]) && (wr_ptr[AW-1:0] == rd_ptr[AW-1:0]);
	assign mem_empty = (wr_ptr == rd_ptr);
	
  // Đường dữ liệu không reset
  always @(posedge clk) begin
    if(do_wr) begin
      mem[wr_ptr[AW-1:0]] <= wr_data;
    end 
    if(do_rd) begin
      rd_data <= mem[rd_ptr[AW-1:0]];
    end
  end

  always @(posedge clk or posedge reset) begin
    if(reset) begin
      wr_ptr <= 0;
      rd_ptr <= 0;
    end else begin
      if(do_wr) wr_ptr <= wr_ptr + 1'b1; 
      if(do_rd) rd_ptr <= rd_ptr + 1'b1;
    end
  end

endmodule
