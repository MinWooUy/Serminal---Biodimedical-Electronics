`default_nettype none
/*
  Layer command controller: Bộ giải mã lệnh uart cho conv1d_layer_top (Chạy trọn vẹn 1 layer)
  Stop-and-Wait: Interface (rx_* / tx_*)

  Response:
                            LSB     MSB      
      0x80 ACK    payload = [cmd, 0]
      0x81 NACK   payload = [cmd, code] code: 1-Lỗi khung/CRC, 2-Tham số sai
                                            3-Sai trạng thái, 3 lệnh lạ
      0x82 RESULT payload tùy lệnh

  Command (payload, little-endian):
    0x08 WRITE_X    addr:u16, data[n]                 -> ACK        Ghi vào bank ghi của IFMap
    0x09 WRITE_W    addr:u16, data[n]                 -> ACK        Ghi vào bank ghi của Weight
    0x0A WRITE_BIAS base:u16, int32[n]                -> ACK        bias theo kênh toàn cục
    0x0B WRITE_SHIFT base:u16, int8[n]                -> ACK        shift theo kênh toàn cục
                                                                    base 0..COUT_MAX-1: shift requant (SO)
                                                                    base COUT_MAX..: shift "truoc" SG (row_combine)
    0x0C SWAP        flags:u8 (bit0 x, bit1 w)        -> ACK
    0x0D CONFIG      out_len, cout, k_len, window_step:u16, pool, gap, pad_next:u8 [, flags:u8] (11/12 byte) -> ACK
                       flags (thieu = 0): bit0 out_int32, bit1 argmax, bit2 row_combine, bit3 gap_in
                       (y nghia: xem conv1d_layer_top.v). Rang buoc, sai -> NACK 2:
                         argmax can out_int32; out_int32: out_len = 1; row_combine: out_len = 3,
                         ARRAY_ROWS >= 3; gap_in can row_combine; out_int32/row_combine: pool = gap = pad = 0
    0x0E RUN         Rỗng                             -> RESULT [status:u8, cycles:u32] (Khi layer xong)
    0x0F READ_GAP    Rỗng                             -> RESULT COUT_MAX x u16
    0x10 READ_FMAP   offset:u16, len:u8               -> RESULT len byte (bank đọc của ifmap)

  Lenh danh rieng cho pipeline nhieu layer (da danh so, chua xu ly trong FSM nay):
    0x11 CONFIG_ALL  layer_count:u8 + 4 cau hinh      -> ACK khi du config
    0x12 LOAD_COMMIT checksum/flags                   -> ACK khi du data
    0x13 RUN_ALL     Rỗng                             -> RESULT sau khi chay het pipeline
    0x14 GET_RESULT  result_type:u8                   -> RESULT ket qua cuoi

    0x15 INFO        Rỗng                             -> RESULT 22 byte (phần cứng tự khai báo cấu hình)
                       [0] version=2  [1] ARRAY_ROWS  [2] ARRAY_COLS  [3] features (bit0 out_int32,
                       bit1 argmax, bit2 row_combine, bit3 gap_in)
                       [4..7] CLK_FREQ u32  [8..11] BAUD_RATE u32
                       [12..13] IFMAP_MEM_DEPTH  [14..15] WGT_MEM_DEPTH  [16..17] FEEDER_DEPTH
                       [18..19] COUT_MAX         [20..21] MAX_PAYLOAD      (u16, little-endian)

  Cac ma 0x11..0x14 chi la hop dong du kien o giai doan nay. Vi chua co nhanh
  xu ly tuong ung trong case, bitstream hien tai van tra NACK lenh la.
*/
module layer_cmd_controller#(
  parameter ARRAY_ROWS = 3,
  parameter ARRAY_COLS = 3,
  parameter CLK_FREQ = 50000000,
  parameter BAUD_RATE = 115200,
  parameter MAX_PAYLOAD = 256,
  parameter IFMAP_MEM_DEPTH = 4096,
  parameter WGT_MEM_DEPTH = 16384,
  parameter FEEDER_DEPTH = 1024,
  parameter COUT_MAX = 64,
  parameter ADDR_W = 16,
  parameter CFG_W = 16
)(  
  input wire i_clk, i_reset,

  // Transpost (uart_crc_top)
  input wire [7:0] i_rx_cmd, i_rx_seq,
  input wire [15:0] i_rx_length,
  input wire [MAX_PAYLOAD*8-1:0] i_rx_payload,
  input wire i_rx_valid, i_rx_error,

  output wire o_rx_enable,

  input wire i_tx_busy,

  output reg o_tx_start,
  output reg [7:0] o_tx_cmd, o_tx_seq,
  output reg [15:0] o_tx_length,
  output reg [MAX_PAYLOAD*8-1:0] o_tx_payload,

  // Conv1d layer top
  output wire o_x_wr_en,
  output wire [ADDR_W-1:0] o_x_wr_addr,
  output wire [7:0] o_x_wr_data,
  output wire o_x_swap,

  output wire o_w_wr_en,
  output wire [ADDR_W-1:0] o_w_wr_addr,
  output wire [7:0] o_w_wr_data,
  output wire o_w_swap,

	output wire o_bias_wr_en,
	output wire [ADDR_W-1:0] o_bias_wr_addr,
	output wire [31:0] o_bias_wr_data,

  output wire o_shift_wr_en,
  output wire [ADDR_W-1:0] o_shift_wr_addr,
  output wire [7:0] o_shift_wr_data,

  output wire o_layer_start,
  output reg[CFG_W-1:0] o_out_len, o_cout, o_k_len,
  output reg [ADDR_W-1:0] o_window_step,
  output reg o_pool_en, o_gap_en,
  output reg [CFG_W-1:0] o_pad_next,
  output reg [3:0] o_flags,

  input wire [15:0] i_gap_rd_data,
  output wire [CFG_W-1:0] o_gap_rd_idx,

  input wire [7:0] i_dbg_rd_data,
  input wire i_dbg_rd_valid,
  output wire o_dbg_rd_en,
  output wire [ADDR_W-1:0] o_dbg_rd_addr,

  input wire i_layer_busy, i_layer_done, i_layer_err
);
  localparam [7:0] C_WRITE_X = 8'h08;
  localparam [7:0] C_WRITE_W = 8'h09;
  localparam [7:0] C_WRITE_BIAS = 8'h0A;
  localparam [7:0] C_WRITE_SHIFT = 8'h0B;
  localparam [7:0] C_SWAP = 8'h0C;
  localparam [7:0] C_CONFIG = 8'h0D;
  localparam [7:0] C_RUN = 8'h0E;
  localparam [7:0] C_READ_GAP = 8'h0F;
  localparam [7:0] C_READ_FMAP = 8'h10;

  // CMD danh cho pipeline nhieu layer; chua duoc noi vao FSM.
  localparam [7:0] C_CONFIG_ALL = 8'h11;
  localparam [7:0] C_LOAD_COMMIT = 8'h12;
  localparam [7:0] C_RUN_ALL = 8'h13;
  localparam [7:0] C_GET_RESULT = 8'h14;

  localparam [7:0] C_INFO = 8'h15;

  // Thông tin phần cứng (do tham số synthesis quyết định) cho lệnh INFO
  localparam [7:0]  INFO_VERSION = 8'd2;
  localparam [7:0]  INFO_FEATURES = 8'h0F; // out_int32 | argmax | row_combine | gap_in
  localparam [7:0]  INFO_ROWS = ARRAY_ROWS;
  localparam [7:0]  INFO_COLS = ARRAY_COLS;
  localparam [31:0] INFO_CLK = CLK_FREQ;
  localparam [31:0] INFO_BAUD = BAUD_RATE;
  localparam [15:0] INFO_IFMAP = IFMAP_MEM_DEPTH;
  localparam [15:0] INFO_WGT = WGT_MEM_DEPTH;
  localparam [15:0] INFO_FEEDER = FEEDER_DEPTH;
  localparam [15:0] INFO_COUT = COUT_MAX;
  localparam [15:0] INFO_PAYLOAD = MAX_PAYLOAD;
  localparam [175:0] HW_INFO = {INFO_PAYLOAD, INFO_COUT, INFO_FEEDER, INFO_WGT, INFO_IFMAP,
                                INFO_BAUD, INFO_CLK, INFO_FEATURES, INFO_COLS, INFO_ROWS, INFO_VERSION};

  localparam IDLE = 0;
  localparam WRITE = 1;
  localparam WRITE_BIAS = 2;
  localparam SWAP = 3;
  localparam RUN_START = 4;
  localparam RUN_WAIT = 5;
  localparam RD_GAP = 6;
  localparam RD_FMAP_ISSUE = 7;
  localparam RD_FMAP_WAIT = 8;
  localparam SEND = 9;
  localparam TX_LAUNCH = 10;
  localparam TX_WAIT = 11;

  reg [3:0] state;
  reg [MAX_PAYLOAD*8-1:0] request;
  reg [15:0] address, count, index;
  reg [1:0] target;
  reg swap_x, swap_w, configured;
  reg [31:0] cycle_cnt;

  wire [15:0] req_addr = i_rx_payload[15:0];
  wire [15:0] cfg_out_len = i_rx_payload[15:0];
  wire [15:0] cfg_cout = i_rx_payload[31:16];
  wire [15:0] cfg_k = i_rx_payload[47:32];
  wire [15:0] cfg_step = i_rx_payload[63:48];
  wire [7:0] cfg_pool = i_rx_payload[71:64];
  wire [7:0] cfg_gap = i_rx_payload[79:72];
  wire [7:0] cfg_pad = i_rx_payload[87:80];
  wire [7:0] req_len8 = i_rx_payload[23:16];

  // Byte co thu 12 cua CONFIG (khong gui = 0, giu tuong thich host cu)
  wire [7:0] cfg_flags = (i_rx_length == 12) ? i_rx_payload[95:88] : 8'd0;
  wire cf_i32 = cfg_flags[0], cf_am = cfg_flags[1], cf_rc = cfg_flags[2], cf_gin = cfg_flags[3];
  wire cfg_flags_bad = (cfg_flags[7:4] != 0) || (cf_am && !cf_i32) || (cf_i32 && cf_rc) || (cf_gin && !cf_rc)
                    || ((cf_i32 || cf_rc) && (cfg_pool != 0 || cfg_gap != 0 || cfg_pad != 0))
                    || (cf_i32 && (cfg_out_len != 1 || ({2'b00, cfg_cout, 2'b00} + 18'd1) > IFMAP_MEM_DEPTH))
                    || (cf_rc && (cfg_out_len != 3 || ARRAY_ROWS < 3));

  assign o_rx_enable = (state == IDLE);

  wire [7:0] wr_data_b = request[(index + 2)*8 +: 8];
  wire [ADDR_W-1:0] wr_addr_b = address + index;

  assign o_x_wr_en = (state == WRITE) && (target == 2'd0);
  assign o_x_wr_addr = wr_addr_b;
  assign o_x_wr_data = wr_data_b;
  assign o_w_wr_en = (state == WRITE) && (target == 2'd1);
  assign o_w_wr_addr = wr_addr_b;
  assign o_w_wr_data = wr_data_b;
  assign o_shift_wr_en = (state == WRITE) && (target == 2'd2);
  assign o_shift_wr_addr = wr_addr_b;
  assign o_shift_wr_data = wr_data_b;

  assign o_bias_wr_en = (state == WRITE_BIAS);
  assign o_bias_wr_addr = address + (index >> 2);
  assign o_bias_wr_data = request[(index + 2)*8 +: 32];

  assign o_x_swap = (state == SWAP) && swap_x;
  assign o_w_swap = (state == SWAP) && swap_w;

  assign o_layer_start = (state == RUN_START);

  assign o_gap_rd_idx = index;
  assign o_dbg_rd_en = (state == RD_FMAP_ISSUE);
  assign o_dbg_rd_addr = address + index;

  always @(posedge i_clk or posedge i_reset) begin
    if(i_reset) begin
      state <= IDLE;
      
      o_tx_start <= 0;
      o_tx_cmd <= 0;
      o_tx_seq <= 0;
      o_tx_length <= 0;
      o_tx_payload <= 0;

      request <= 0;
      address <= 0;
      count <= 0;
      target <= 0;

      swap_x <= 0;
      swap_w <= 0;
      configured <= 0;
      cycle_cnt <= 0;

      o_out_len <= 0;
      o_cout <= 0;
      o_k_len <= 0;
      o_window_step <= 0;

      o_pool_en <= 0;
      o_gap_en <= 0;
      o_pad_next <= 0;
      o_flags <= 0;
    end else begin
      o_tx_start <= 0;

      case(state)
        IDLE:
          if(i_rx_valid || i_rx_error) begin
            o_tx_seq <= i_rx_seq;
            o_tx_length <= 2;
            o_tx_payload <= 0;
            o_tx_payload[7:0] <= i_rx_cmd;
            o_tx_cmd <= 8'h80;

            state <= SEND;
            
            if(i_rx_error) begin
              o_tx_cmd <= 8'h81;
              o_tx_payload[15:8] <= 1;
            end else case(i_rx_cmd)
              C_WRITE_X, C_WRITE_W: begin
                if(i_rx_length < 3 || ({1'b0, req_addr} + i_rx_length - 2) > ((i_rx_cmd == C_WRITE_X) ? IFMAP_MEM_DEPTH : WGT_MEM_DEPTH)) begin
                  o_tx_cmd <= 8'h81;
                  o_tx_payload[15:8] <= 2;
                end else begin
                  request <= i_rx_payload;
                  address <= req_addr;
                  count <= i_rx_length - 2;

                  index <= 0;
                  target <= (i_rx_cmd == C_WRITE_X) ? 2'd0 : 2'd1;
                  
                  state <= WRITE;
                end
              end

              C_WRITE_SHIFT: begin
                if(i_rx_length < 3 || ({1'b0, req_addr} + i_rx_length - 2) > 2*COUT_MAX)begin
                  o_tx_cmd <= 8'h81;
                  o_tx_payload[15:8] <= 2;
                end else begin
                  request <= i_rx_payload;
                  address <= req_addr;
                  count <= i_rx_length - 2;
                  index <= 0;
                  target <= 2'd2;
                  state <= WRITE;
                end
              end

              C_WRITE_BIAS: begin
                if(i_rx_length < 6 || i_rx_length[1:0] != 2'b10 || ({1'b0, req_addr} + ((i_rx_length - 2) >> 2)) > COUT_MAX) begin
                  o_tx_cmd <= 8'h81;
                  o_tx_payload[15:8] <= 2;
                end else begin
                  request <= i_rx_payload;
                  address <= req_addr;
                  count <= i_rx_length - 2;
                  index <= 0;
                  state <= WRITE_BIAS;
                end
              end

              C_SWAP: begin
                if(i_rx_length != 1 || i_rx_payload[7:2] != 0) begin
                  o_tx_cmd <= 8'h81;
                  o_tx_payload[15:8] <= 2;
                end else if(i_layer_busy) begin
                  o_tx_cmd <= 8'h81;
                  o_tx_payload[15:8] <= 3;
                end else begin
                  swap_x <= i_rx_payload[0];
                  swap_w <= i_rx_payload[1];
                  state <= SWAP;
                end
              end

              C_CONFIG: begin
                if((i_rx_length != 11 && i_rx_length != 12) || cfg_out_len == 0 || cfg_cout == 0 || cfg_cout > COUT_MAX || cfg_k == 0 || cfg_k > FEEDER_DEPTH || cfg_step == 0 || cfg_pool > 1 || cfg_gap > 1 || (cfg_pool == 1 && cfg_out_len[0]) || cfg_flags_bad) begin
                  o_tx_cmd <= 8'h81;
                  o_tx_payload[15:8] <= 2;
                end else if(i_layer_busy) begin
                  o_tx_cmd <= 8'h81;
                  o_tx_payload[15:8] <= 3;
                end else begin
                  o_out_len <= cfg_out_len;
                  o_cout <= cfg_cout;
                  o_k_len <= cfg_k;
                  o_window_step <= cfg_step;
                  o_pool_en <= cfg_pool[0];
                  o_gap_en <= cfg_gap[0];
                  o_pad_next <= {8'd0, cfg_pad};
                  o_flags <= cfg_flags[3:0];
                  configured <= 1;
                end
              end
              
              C_RUN: begin
                if(i_rx_length != 0) begin
                  o_tx_cmd <= 8'h81;
                  o_tx_payload[15:8] <= 2;
                end else if(!configured || i_layer_busy) begin
                  o_tx_cmd <= 8'h81;
                  o_tx_payload[15:8] <= 3;
                end else begin
                  cycle_cnt <= 0;
                  state <= RUN_START;
                end
              end

              C_READ_GAP: begin
                if(i_rx_length != 0) begin
                  o_tx_cmd <= 8'h81;
                  o_tx_payload[15:8] <= 2;
                end else if(i_layer_busy) begin
                  o_tx_cmd <= 8'h81;
                  o_tx_payload[15:8] <= 3;
                end else begin
                  o_tx_cmd <= 8'h82;
                  o_tx_length <= COUT_MAX * 2;
                  o_tx_payload <= 0;
                  index <= 0;
                  state <= RD_GAP;
                end
              end

              C_READ_FMAP: begin
                if(i_rx_length != 3 || req_len8 == 0 || ({1'b0, req_addr} + req_len8) > IFMAP_MEM_DEPTH) begin
                  o_tx_cmd <= 8'h81;
                  o_tx_payload[15:8] <= 3;
                end else if(i_layer_busy) begin
                  o_tx_cmd <= 8'h81;
                  o_tx_payload[15:8] <= 3;
                end else begin
                  o_tx_cmd <= 8'h82;
                  o_tx_length <= req_len8;
                  o_tx_payload <= 0;
                  address <= req_addr;
                  count <= req_len8;
                  index <= 0;
                  state <= RD_FMAP_ISSUE;
                end
              end

              C_INFO: begin
                if(i_rx_length != 0) begin
                  o_tx_cmd <= 8'h81;
                  o_tx_payload[15:8] <= 2;
                end else begin
                  o_tx_cmd <= 8'h82;
                  o_tx_length <= 22;
                  o_tx_payload[175:0] <= HW_INFO;
                end
              end

              default: begin
                o_tx_cmd <= 8'h81;
                o_tx_payload[15:8] <= 4;
              end
            endcase
          end

        WRITE: begin
          if(index == count - 1) state <= SEND;
          else index <= index + 1'b1;
        end

        WRITE_BIAS: begin
          if(index == count - 4) state <= SEND;
          else index <= index + 4;
        end

        SWAP: state <= SEND;

        RUN_START: begin
          cycle_cnt <= cycle_cnt + 1;
          state <= RUN_WAIT;
        end

        RUN_WAIT: begin
          cycle_cnt <= cycle_cnt + 1;
          if(i_layer_done) begin
            o_tx_cmd <= 8'h82;
            o_tx_length <= 5;
            o_tx_payload <= 0;
            o_tx_payload[7:0] <= {7'b0, i_layer_err};
            o_tx_payload[39:8] <= cycle_cnt + 1;
            state <= SEND;
          end
        end

        RD_GAP: begin
          o_tx_payload[index*16 +: 16] <= i_gap_rd_data;
          if(index == COUT_MAX - 1) state <= SEND;
          else index <= index+1'b1;  
        end

        RD_FMAP_ISSUE: state <= RD_FMAP_WAIT;

        RD_FMAP_WAIT: 
          if(i_dbg_rd_valid) begin
            o_tx_payload[index*8 +: 8] <= i_dbg_rd_data;
            if(index == count-1) state <= SEND;
            else begin 
              index <= index + 1'b1;
              state <= RD_FMAP_ISSUE;
            end
          end
        
        SEND:
          if(!i_tx_busy) begin
            o_tx_start <= 1;
            state <= TX_LAUNCH;
          end

        TX_LAUNCH: state <= TX_WAIT;
        TX_WAIT:
          if(!i_tx_busy) state <= IDLE;

        default: state <= IDLE;
      endcase
    end
  end

endmodule
`default_nettype wire