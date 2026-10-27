// Host-facing wrapper for one Conv1D tile. reset is active high.
module ecg_accel_top #(
    parameter CLK_FREQ=50000000, BAUD_RATE=115200,
    parameter MEM_DEPTH=1024, FEEDER_DEPTH=1024, MAX_PAYLOAD=256,
    parameter RX_TIMEOUT_CLKS=CLK_FREQ/10, POST_ENABLE=0
)(input wire clk, reset, uart_rx, output wire uart_tx);

`ifndef SYNTHESIS
    initial begin
        if(CLK_FREQ/BAUD_RATE < 8 || RX_TIMEOUT_CLKS < 1)
            $fatal(1,"Invalid UART timing parameters");
        if(MAX_PAYLOAD < (POST_ENABLE ? 45 : 36) || MAX_PAYLOAD > 256 || MEM_DEPTH < 1 || MEM_DEPTH > 65536 ||
           FEEDER_DEPTH < 1 || FEEDER_DEPTH > 65535)
            $fatal(1,"Invalid accelerator transport dimensions");
    end
`endif
    
    wire rx_enable,rx_valid,rx_error,tx_start,tx_busy;
    wire [7:0] rx_cmd,rx_seq,tx_cmd,tx_seq;
    wire [15:0] rx_length,tx_length;
    wire [MAX_PAYLOAD*8-1:0] rx_payload,tx_payload;
    wire x_wr_en,w_wr_en,x_swap,w_swap,tile_start,tile_busy,tile_done;
    wire bias_wr_en,preload_bias_en,tile_bias_rd_en;
    wire [15:0] wr_addr,k_len,window_step,x_base,w_base;
    wire [15:0] bias_wr_addr;
    wire [7:0] wr_data;
    wire [31:0] bias_wr_data;
    wire [1:0] tile_bias_rd_idx;
    wire [1:0] active_rows,active_cols;
    wire [287:0] tile_data;
    wire [71:0] post_tile;
    wire post_valid;
    wire [23:0] post_shift;
    wire post_relu;
    wire post_configured;
   
    reg [31:0] bias_mem [0:2];
    reg [31:0] bias_rd_data;
    reg [71:0] post_tile_q;
    reg post_ready_q;

    always @(posedge clk or posedge reset) begin
        if(reset) begin
            post_tile_q <= 0;
            post_ready_q <= 0;
        end else begin
            if(tile_start)
                post_ready_q <=1'b0;
            if(post_valid) begin
                post_tile_q <= post_tile;
                post_ready_q <= 1'b1;
            end
        end
    end

    integer bias_i;
    // Bias is one INT32 per output column. The controller serializes writes and
    // execution, so one small active bank is sufficient for the single-tile path.
    always @(posedge clk or posedge reset) begin
        if(reset) begin
            bias_rd_data<=0;
            for(bias_i=0;bias_i<3;bias_i=bias_i+1) bias_mem[bias_i]<=0;
        end else begin
            if(bias_wr_en) bias_mem[bias_wr_addr[1:0]]<=bias_wr_data;
            if(tile_bias_rd_en) bias_rd_data<=bias_mem[tile_bias_rd_idx];
        end
    end

    uart_crc_top #(.CLK_FREQ(CLK_FREQ),.BAUD_RATE(BAUD_RATE),
        .MAX_PAYLOAD(MAX_PAYLOAD),.RX_TIMEOUT_CLKS(RX_TIMEOUT_CLKS)) u_transport(
        .clk(clk),.reset(reset),.uart_rx(uart_rx),.uart_tx(uart_tx),
        .rx_enable(rx_enable),.rx_cmd(rx_cmd),.rx_seq(rx_seq),.rx_length(rx_length),
        .rx_payload(rx_payload),.rx_valid(rx_valid),.rx_error(rx_error),
        .tx_start(tx_start),.tx_cmd(tx_cmd),.tx_seq(tx_seq),.tx_length(tx_length),
        .tx_payload(tx_payload),.tx_busy(tx_busy));
    
    host_command_controller #(.MAX_PAYLOAD(MAX_PAYLOAD),.MEM_DEPTH(MEM_DEPTH),
        .FEEDER_DEPTH(FEEDER_DEPTH),.POST_ENABLE(POST_ENABLE)) u_commands(
        .clk(clk),.reset(reset),.rx_enable(rx_enable),.rx_cmd(rx_cmd),.rx_seq(rx_seq),
        .rx_length(rx_length),.rx_payload(rx_payload),.rx_valid(rx_valid),.rx_error(rx_error),
        .tx_start(tx_start),.tx_cmd(tx_cmd),.tx_seq(tx_seq),.tx_length(tx_length),
        .tx_payload(tx_payload),.tx_busy(tx_busy),
        .x_wr_en(x_wr_en),.w_wr_en(w_wr_en),.wr_addr(wr_addr),.wr_data(wr_data),
        .bias_wr_en(bias_wr_en),.bias_wr_addr(bias_wr_addr),.bias_wr_data(bias_wr_data),
        .preload_bias_en(preload_bias_en),
        .x_swap(x_swap),.w_swap(w_swap),.tile_start(tile_start),.k_len(k_len),
        .window_step(window_step),.x_base(x_base),.w_base(w_base),
        .active_rows(active_rows),.active_cols(active_cols),
        .tile_busy(tile_busy),.tile_done(tile_done),.tile_data(tile_data),
        .post_ready(post_ready_q),
        .post_data(post_tile_q),.post_shift(post_shift),.post_relu(post_relu),
        .post_configured(post_configured)
        );

    conv1d_tile_top #(
        .POST_ENABLE(POST_ENABLE),
        .IFMAP_MEM_DEPTH(MEM_DEPTH),
        .WGT_MEM_DEPTH(MEM_DEPTH),
        .FEEDER_DEPTH(FEEDER_DEPTH)
        ) u_tile(
        .i_clk(clk),
        .i_reset(reset),
        .i_start(tile_start),
        .i_k_len(k_len),
        .i_window_step(window_step),
        .i_active_rows(active_rows),
        .i_active_cols(active_cols),
        .i_ifmap_base_addr(x_base),
        .i_wgt_base_addr(w_base),

        .i_ifmap_mem_wr_en(x_wr_en),
        .i_ifmap_mem_wr_addr(wr_addr),
        .i_ifmap_mem_wr_data(wr_data),
        .i_ifmap_mem_swap(x_swap),

        .i_wgt_mem_wr_en(w_wr_en),
        .i_wgt_mem_wr_addr(wr_addr),
        .i_wgt_mem_wr_data(wr_data),
        .i_wgt_mem_swap(w_swap),

        .i_preload_mode(preload_bias_en),
        .i_bias_rd_data(bias_rd_data),
        .o_bias_rd_en(tile_bias_rd_en),
        .o_bias_rd_idx(tile_bias_rd_idx),
        .i_drain_ready(1'b1),

        .i_post_shift(post_shift),
        .i_post_relu(post_relu),
        .o_post_tile(post_tile),
        .o_post_valid(post_valid),

        .o_tile(tile_data),
        .o_busy(tile_busy),
        .o_done(tile_done)
    );
endmodule
