// Stop-and-wait host protocol. No pipelined requests or implicit RUN retries.
// Fixed accelerator geometry: INT8, INT32, 3x3, 16-bit addresses/config.
module host_command_controller #(
    parameter MAX_PAYLOAD=256, MEM_DEPTH=1024, FEEDER_DEPTH=16
)(
    input wire clk, reset,
    output wire rx_enable,
    input wire [7:0] rx_cmd, rx_seq,
    input wire [15:0] rx_length,
    input wire [MAX_PAYLOAD*8-1:0] rx_payload,
    input wire rx_valid, rx_error,
    output reg tx_start,
    output reg [7:0] tx_cmd, tx_seq,
    output reg [15:0] tx_length,
    output reg [MAX_PAYLOAD*8-1:0] tx_payload,
    input wire tx_busy,
    output wire x_wr_en, w_wr_en,
    output wire [15:0] wr_addr,
    output wire [7:0] wr_data,
    output wire x_swap, w_swap, tile_start,
    output reg [15:0] k_len, window_step, x_base, w_base,
    output reg [1:0] active_rows, active_cols,
    input wire tile_busy, tile_done,
    input wire [287:0] tile_data
);
    localparam IDLE=0, WRITE=1, SWAP=2, START=3, WAIT_TILE=4,
               SEND=5, TX_LAUNCH=6, TX_WAIT=7, CHECK_X=8, CHECK_W=9;
    reg [3:0] state;
    reg [MAX_PAYLOAD*8-1:0] request;
    reg [15:0] address, count, index;
    reg write_weight, configured, x_dirty, w_dirty;
    reg [MEM_DEPTH-1:0] x_stage_map,w_stage_map,x_active_map,w_active_map;
    wire [15:0] req_addr=rx_payload[15:0];
    wire [15:0] cfg_k=rx_payload[15:0];
    wire [15:0] cfg_step=rx_payload[31:16];
    wire [7:0] cfg_rows=rx_payload[39:32], cfg_cols=rx_payload[47:40];
    wire [15:0] cfg_x=rx_payload[63:48], cfg_w=rx_payload[79:64];
    reg [1:0] check_lane;
    wire [31:0] check_x_addr={16'b0,x_base}+check_lane*{16'b0,window_step}+index;
    wire [31:0] check_w_addr={16'b0,w_base}+check_lane*{16'b0,k_len}+index;
    // Validate one address per clock to avoid dozens of parallel bitmap muxes.
    assign rx_enable=(state==IDLE);
    assign x_wr_en=(state==WRITE)&&!write_weight;
    assign w_wr_en=(state==WRITE)&&write_weight;
    assign wr_addr=address+index;
    assign wr_data=request[(index+2)*8 +: 8];
    assign x_swap=(state==SWAP)&&x_dirty;
    assign w_swap=(state==SWAP)&&w_dirty;
    assign tile_start=(state==START);
    always @(posedge clk or posedge reset) begin
        if(reset) begin
            state<=IDLE;tx_start<=0;tx_cmd<=0;tx_seq<=0;tx_length<=0;tx_payload<=0;
            request<=0;address<=0;count<=0;index<=0;write_weight<=0;check_lane<=0;
            configured<=0;x_dirty<=0;w_dirty<=0;
            x_stage_map<=0;w_stage_map<=0;x_active_map<=0;w_active_map<=0;
            k_len<=0;window_step<=0;x_base<=0;w_base<=0;active_rows<=0;active_cols<=0;
        end else begin
            tx_start<=0;
            case(state)
                IDLE: if(rx_valid || rx_error) begin
                    tx_seq<=rx_seq;tx_length<=2;tx_payload<=0;
                    tx_payload[7:0]<=rx_cmd;
                    tx_cmd<=8'h80;state<=SEND;
                    if(rx_error) begin tx_cmd<=8'h81;tx_payload[15:8]<=1; end
                    else case(rx_cmd)
                        8'h01,8'h02: begin
                            if(rx_length<3 || ({1'b0,req_addr}+rx_length-2)>MEM_DEPTH) begin
                                tx_cmd<=8'h81;tx_payload[15:8]<=2;
                            end else begin
                                request<=rx_payload;address<=req_addr;count<=rx_length-2;
                                index<=0;write_weight<=(rx_cmd==2);state<=WRITE;
                            end
                        end
                        8'h03: begin
                            if(rx_length!=10 || cfg_k==0 || cfg_k>FEEDER_DEPTH ||
                               cfg_step==0 || cfg_rows==0 || cfg_rows>3 || cfg_cols==0 || cfg_cols>3 ||
                               ({16'b0,cfg_x}+(cfg_rows-1)*{16'b0,cfg_step}+cfg_k)>MEM_DEPTH ||
                               ({16'b0,cfg_w}+cfg_cols*{16'b0,cfg_k})>MEM_DEPTH) begin
                                tx_cmd<=8'h81;tx_payload[15:8]<=2;
                            end else begin
                                k_len<=cfg_k;window_step<=cfg_step;
                                active_rows<=cfg_rows[1:0];active_cols<=cfg_cols[1:0];
                                x_base<=cfg_x;w_base<=cfg_w;configured<=1;
                            end
                        end
                        8'h04: begin
                            if(rx_length!=0) begin tx_cmd<=8'h81;tx_payload[15:8]<=2; end
                            else if(!configured || tile_busy || tile_done) begin
                                tx_cmd<=8'h81;tx_payload[15:8]<=3;
                            end else begin state<=CHECK_X;index<=0;check_lane<=0;end
                        end
                        8'h05: begin
                            if(rx_length!=0) begin tx_cmd<=8'h81;tx_payload[15:8]<=2; end
                            else begin
                                tx_length<=3;
                                tx_payload[23:16]<={5'b0,w_dirty,x_dirty,configured};
                            end
                        end
                        default: begin tx_cmd<=8'h81;tx_payload[15:8]<=4; end
                    endcase
                end
                WRITE: begin
                    if(write_weight) begin w_stage_map[address+index]<=1;w_dirty<=1;end
                    else begin x_stage_map[address+index]<=1;x_dirty<=1;end
                    if(index==count-1) state<=SEND;
                    else index<=index+1'b1;
                end
                CHECK_X: begin
                    if(check_x_addr>=MEM_DEPTH ||
                       !(x_dirty ? x_stage_map[check_x_addr] : x_active_map[check_x_addr])) begin
                        tx_cmd<=8'h81;tx_payload[15:8]<=3;state<=SEND;
                    end else if(index==k_len-1) begin
                        index<=0;
                        if(check_lane==active_rows-1) begin check_lane<=0;state<=CHECK_W;end
                        else check_lane<=check_lane+1'b1;
                    end else index<=index+1'b1;
                end
                CHECK_W: begin
                    if(check_w_addr>=MEM_DEPTH ||
                       !(w_dirty ? w_stage_map[check_w_addr] : w_active_map[check_w_addr])) begin
                        tx_cmd<=8'h81;tx_payload[15:8]<=3;state<=SEND;
                    end else if(index==k_len-1) begin
                        index<=0;
                        if(check_lane==active_cols-1) state<=SWAP;
                        else check_lane<=check_lane+1'b1;
                    end else index<=index+1'b1;
                end
                SWAP: begin
                    if(x_dirty) begin x_active_map<=x_stage_map;x_stage_map<=0;x_dirty<=0;end
                    if(w_dirty) begin w_active_map<=w_stage_map;w_stage_map<=0;w_dirty<=0;end
                    state<=START;
                end
                START: state<=WAIT_TILE;
                WAIT_TILE: if(tile_done) begin
                    tx_cmd<=8'h82;tx_length<=36;tx_payload<=0;
                    tx_payload[287:0]<=tile_data;state<=SEND;
                end
                SEND: if(!tx_busy) begin tx_start<=1;state<=TX_LAUNCH;end
                TX_LAUNCH: state<=TX_WAIT;
                TX_WAIT: if(!tx_busy) state<=IDLE;
                default: state<=IDLE;
            endcase
        end
    end
endmodule
