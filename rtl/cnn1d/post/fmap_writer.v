// PE array -> PSM -> post -> | fmap_writer |-> fmap sram INT8 (output)
// SRAM output chứa: Conv + bias + requant + ReLU
// Khối hậu xử lý (post_processing): dịch, làm tròn, ReLU, clamp -> INT8 -> OUTPUT sram --> MaxPool hoặc conv layer tiếp theo và so Golden

// Ý tưởng: nhận kết quả 1 tile, rồi lần lượt ghi từng phần tử hợp lệ vào đúng vị trí trong output SRAM
//  -> Tile output: nhiều giá trị nằm song song trên bus.
//  -> fmap_writer: chọn từng giá trị + tính địa chỉ
//  -> SRAM: ghi một giá trị int8 mỗi chu kỳ

// Giả sử PE ARRAY là 3x3. Sau post, một tile có dạng:
//            Kênh
//            c0  c1  c2
// vị trí t0: 11  12  13
// vị trí t0: 21  22  23
// vị trí t0: 31  32  33

// hiện tại: tile[(row * ARRAY_COLS + col)*8 +: 8]
// Output sram dự định: 1 cổng ghi 8 bit
//  -> Writer cần tách 9 giá trị đó thành 9 lần ghi và tính địa chỉ cho mỗi làn.

// Phía post_process chỉ được gửi i_start khi o_ready = 1
// o_overrun một cờ báo lỗi. Lên 1 khi i_start = 1 mà state == S_WRITE (đang ghi tile trước đó). Khi o_overrun = 1, post_process cần dừng lại và reset lại khối này.

`default_nettype none

module fmap_writer #(
  parameter DATA_W = 8,
  parameter ARRAY_ROWS = 3,
  parameter ARRAY_COLS = 3,
  parameter CFG_W = 16,
  parameter ROW_W = (ARRAY_ROWS <= 1) ? 1 : $clog2(ARRAY_ROWS + 1),
  parameter COL_W = (ARRAY_COLS <= 1) ? 1 : $clog2(ARRAY_COLS + 1),
	parameter DEPTH = 1024,
  parameter ADDR_W = (DEPTH <= 1) ? 1 : $clog2(DEPTH)
)(
  input wire i_clk, i_reset,

  // tín hiệu điều khiển
  input wire i_start, i_clear_overrun,

  // Dữ liệu của 1 tile
  input wire [ARRAY_ROWS*ARRAY_COLS*DATA_W-1:0] i_tile,

  // Tile nằm ở đâu trong output của layer?
  input wire [CFG_W-1:0] i_row_base,
  input wire [CFG_W-1:0] i_col_base,

  // Bao nhiêu hàng, cột trong tile là hợp lệ?
  input wire [ROW_W-1:0] i_active_rows,
  input wire [COL_W-1:0] i_active_cols,

  // Số kênh output của cả layer và địa chỉ gốc
  input wire [CFG_W-1:0] i_out_channels,
  input wire [ADDR_W-1:0] i_output_base,

  output wire o_wr_en,
  output wire[ADDR_W-1:0] o_wr_addr,
  output wire[DATA_W-1:0] o_wr_data,

  output wire o_ready, 
  output wire o_overrun,
  output reg o_done
);
  localparam S_IDLE = 1'd0;
  localparam S_WRITE = 1'd1;

  reg state, overrun_q;

  reg [ROW_W-1:0] row;
  reg [COL_W-1:0] col;

  reg [CFG_W -1:0] row_base_q, col_base_q, out_channels_q;
  reg [ROW_W-1:0] active_rows_q;
  reg [COL_W-1:0] active_cols_q;
  reg [ADDR_W-1:0] output_base_q;
  reg [ARRAY_ROWS*ARRAY_COLS*DATA_W-1:0] tile_q;
  always @(posedge i_clk or posedge i_reset) begin
    if(i_reset) begin
      state <= S_IDLE;

      row <= 0; col <= 0;

      overrun_q <= 1'b0;
      row_base_q <= 0;
      col_base_q <= 0;
      out_channels_q <= 0;

      active_rows_q <= 0;
      active_cols_q <= 0;

      output_base_q <= 0;

      tile_q <= 0;

      o_done <= 1'b0;
    end else begin
      o_done <= 1'b0; // Mặc định: chỉ bật một chu kỳ khi ghi xong
      if(i_clear_overrun) overrun_q <= 1'b0;
      if(state == S_WRITE && i_start) begin
        overrun_q <= 1'b1; // Nếu đang ghi mà post_process lại gửi i_start, báo lỗi
      end
      case(state)
        S_IDLE: begin
          row_base_q <= i_row_base;
          col_base_q <= i_col_base;
          out_channels_q <= i_out_channels;

          active_rows_q <= i_active_rows;
          active_cols_q <= i_active_cols;

          output_base_q <= i_output_base;

          tile_q <= i_tile;
          if(i_start) begin
            if(i_active_rows != 0 && i_active_cols != 0) begin
              state <= S_WRITE;
              row <= 0; col <= 0;
            end else begin
              // Tile rỗng không có gì để ghi -> Báo done để không bị treo
              o_done <= 1'b1;
            end
          end
        end

        S_WRITE:begin
          // Counter 2-D: [Vị trí][Kênh]
          if(col == active_cols_q-1'b1) begin
            col <= 0;
            if(row == active_rows_q-1'b1) begin
              // Ghi phần tử cuối tại cạnh clock này
              state <= S_IDLE;
              o_done <= 1'b1;
            end else begin
              row <= row + 1'b1;
            end
          end else begin
            col <= col + 1'b1;
          end
        end
      endcase
    end
  end

  assign o_ready = (state == S_IDLE); // o_busy = !o_ready: nên không cần o_busy
  assign o_overrun = overrun_q;
  assign o_wr_en = (state == S_WRITE);
  assign o_wr_addr = output_base_q + (row_base_q + row)*out_channels_q + (col_base_q + col);
  assign o_wr_data = tile_q[(row*ARRAY_COLS + col)*DATA_W +: DATA_W];

endmodule

`default_nettype wire