`timescale 1ns/1ps
module tb_feed_lane();

  parameter DATA_W = 8;
  parameter CNT_W  = 5;
  parameter DEPTH  = 16;
  parameter LANE_N = 2;          // test trễ 2 nhịp

  reg                  clk, reset;
  reg                  wr_en, rd_en;
  reg  [DATA_W-1:0]    wr_data;
  wire                 mem_empty, mem_full;
  wire                 o_valid;
  wire [DATA_W-1:0]    o_data;

  feed_lane #(
    .DATA_W(DATA_W), .CNT_W(CNT_W), .DEPTH(DEPTH), .LANE_N(LANE_N)
  ) uut (
    .clk(clk), .reset(reset),
    .wr_en(wr_en), .rd_en(rd_en), .wr_data(wr_data),
    .mem_empty(mem_empty), .mem_full(mem_full),
    .o_valid(o_valid), .o_data(o_data)
  );

  initial clk = 0;
  always #5 clk = ~clk;

  // Ghi 1 phần tử vào FIFO
  task push(input [DATA_W-1:0] d);
    begin
      @(negedge clk);
      wr_en = 1'b1; rd_en = 1'b0; wr_data = d;
      @(negedge clk);
      wr_en = 1'b0;
    end
  endtask

  integer cyc;
  reg [DATA_W-1:0] expected;
  reg got;

  initial begin
    wr_en = 0; rd_en = 0; wr_data = 0;
    reset = 1;
    @(negedge clk); @(negedge clk);
    reset = 0;

    // Nạp 3 giá trị vào FIFO
    push(8'hA1);
    push(8'hB2);
    push(8'hC3);

    // Bắt đầu đọc liên tục, theo dõi o_valid và o_data
    @(negedge clk);
    rd_en = 1'b1;

    // Kiểm tra: giá trị đầu tiên (A1) phải xuất hiện ở o_data
    // sau đúng LANE_N nhịp kể từ khi o_valid lên.
    // Ta chờ tới khi o_valid=1 rồi so giá trị theo thứ tự FIFO.
    expected = 8'hA1;
    got = 0;
    for (cyc = 0; cyc < 20 && !got; cyc = cyc + 1) begin
      @(posedge clk);
      #1;                         // chờ tín hiệu ổn định sau cạnh
      if (o_valid === 1'b1) begin
        if (o_data !== 8'hA1) begin
          $display("FAILED: gia tri dau ra %h, mong doi A1 (cyc=%0d)", o_data, cyc);
          $stop;
        end
        got = 1;
      end
    end
    if (!got) begin
      $display("FAILED: o_valid khong bao gio len 1"); $stop;
    end

    // Hai nhịp kế tiếp phải ra B2 rồi C3 (FIFO đúng thứ tự)
    @(posedge clk); #1;
    if (o_data !== 8'hB2) begin
      $display("FAILED: gia tri thu 2 %h, mong doi B2", o_data); $stop;
    end
    @(posedge clk); #1;
    if (o_data !== 8'hC3) begin
      $display("FAILED: gia tri thu 3 %h, mong doi C3", o_data); $stop;
    end

    rd_en = 1'b0;
    $display("FEED_LANE PASSED!");
    $stop;
  end

endmodule