`timescale 1ns/1ps
module tb_FIFO_param();

  parameter CNT_W  = 5;
  parameter DATA_W = 8;
  parameter DEPTH  = 16;

  reg                  clk, reset;
  reg                  wr_en, rd_en;
  reg  [DATA_W-1:0]    wr_data;
  wire                 mem_full, mem_empty;
  wire [DATA_W-1:0]    rd_data;

  integer k;   // biến dùng cho test đầy/rỗng

  // DUT
  FIFO_param #(.CNT_W(CNT_W), .DATA_W(DATA_W), .DEPTH(DEPTH)) uut (
    .clk(clk), .reset(reset),
    .wr_en(wr_en), .rd_en(rd_en),
    .wr_data(wr_data),
    .mem_full(mem_full), .mem_empty(mem_empty),
    .rd_data(rd_data)
  );

  // Clock 10ns
  initial clk = 0;
  always #5 clk = ~clk;

  // ghi 1 phần tử
  task push(input [DATA_W-1:0] d);
    begin
      @(negedge clk);
      wr_en   = 1'b1;
      rd_en   = 1'b0;
      wr_data = d;
      @(negedge clk);
      wr_en   = 1'b0;
    end
  endtask

  // đọc 1 phần tử, kiểm tra giá trị ra
  task pop(input [DATA_W-1:0] expected);
    begin
      @(negedge clk);
      // rd_data là tổ hợp = mem[0], kiểm tra TRƯỚC khi dịch
      if (rd_data !== expected) begin
        $display("FAILED: doc ra %h, mong doi %h (luc t=%0t)", rd_data, expected, $time);
        $stop;
      end
      rd_en = 1'b1;
      wr_en = 1'b0;
      @(negedge clk);
      rd_en = 1'b0;
    end
  endtask

  initial begin
    // Khoi tao + reset
    wr_en = 0; rd_en = 0; wr_data = 0;
    reset = 1;
    @(negedge clk);
    @(negedge clk);
    reset = 0;

    // --- Kiem tra trang thai sau reset: phai rong ---
    if (mem_empty !== 1'b1) begin
      $display("FAILED: sau reset mem_empty phai = 1"); $stop;
    end
    if (mem_full !== 1'b0) begin
      $display("FAILED: sau reset mem_full phai = 0"); $stop;
    end

    // --- Kich ban co ban: A1, B2, doc, C3, doc, doc ---
    push(8'hA1);
    push(8'hB2);
    pop (8'hA1);          // FIFO: ra dung thu tu vao
    push(8'hC3);
    pop (8'hB2);
    pop (8'hC3);

    // Sau khi doc het: phai rong
    @(negedge clk);
    if (mem_empty !== 1'b1) begin
      $display("FAILED: doc het roi ma mem_empty != 1"); $stop;
    end

    // --- Kiem tra DAY: ghi DEPTH phan tu lien tiep ---
    for (k = 0; k < DEPTH; k = k + 1)
      push(k[DATA_W-1:0]);          // ghi 0,1,2,...,DEPTH-1

    @(negedge clk);
    if (mem_full !== 1'b1) begin
      $display("FAILED: ghi day %0d phan tu ma mem_full != 1", DEPTH); $stop;
    end

    // Ghi them khi day: phai bi chan, count khong doi
    push(8'hFF);
    @(negedge clk);
    if (mem_full !== 1'b1) begin
      $display("FAILED: ghi khi day phai bi chan, van phai full"); $stop;
    end
	
    // --- Doc het, kiem tra dung thu tu (FIFO) ---
    for (k = 0; k < DEPTH; k = k + 1)
      pop(k[DATA_W-1:0]);          // ra dung thu tu da ghi

    @(negedge clk);
    if (mem_empty !== 1'b1) begin
      $display("FAILED: doc het ma mem_empty != 1"); $stop;
    end

    $display("FIFO PASSED!");
    $stop;
  end

endmodule