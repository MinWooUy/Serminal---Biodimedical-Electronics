`default_nettype none

// Luu cau hinh da dong bang cua moi layer trong mot lan chay tu dong.
// RAM nay doc lap voi cac bank ping-pong cua feature-map.
module layer_config_ram #(
    parameter LAYERS = 4,
    parameter CFG_W = 16,
    parameter ADDR_W = 16,
    parameter LAYER_W = (LAYERS <= 1) ? 1 : $clog2(LAYERS)
)(
    input wire i_clk,
    input wire i_reset,

    input wire i_wr_en,
    input wire [LAYER_W-1:0] i_wr_layer,
    input wire [CFG_W-1:0] i_wr_out_len,
    input wire [CFG_W-1:0] i_wr_cout,
    input wire [CFG_W-1:0] i_wr_k_len,
    input wire [ADDR_W-1:0] i_wr_window_step,
    input wire i_wr_pool_en,
    input wire i_wr_gap_en,
    input wire [CFG_W-1:0] i_wr_pad_next,
    input wire i_wr_valid,

    input wire [LAYER_W-1:0] i_rd_layer,
    output reg [CFG_W-1:0] o_out_len,
    output reg [CFG_W-1:0] o_cout,
    output reg [CFG_W-1:0] o_k_len,
    output reg [ADDR_W-1:0] o_window_step,
    output reg o_pool_en,
    output reg o_gap_en,
    output reg [CFG_W-1:0] o_pad_next,
    output reg o_valid,
    output reg o_all_valid
);
    reg [CFG_W-1:0] out_len_mem [0:LAYERS-1];
    reg [CFG_W-1:0] cout_mem [0:LAYERS-1];
    reg [CFG_W-1:0] k_len_mem [0:LAYERS-1];
    reg [ADDR_W-1:0] step_mem [0:LAYERS-1];
    reg pool_mem [0:LAYERS-1];
    reg gap_mem [0:LAYERS-1];
    reg [CFG_W-1:0] pad_mem [0:LAYERS-1];
    reg valid_mem [0:LAYERS-1];

    integer i;
    always @(posedge i_clk or posedge i_reset) begin
        if (i_reset) begin
            o_all_valid <= 1'b0;
            for (i = 0; i < LAYERS; i = i + 1)
                valid_mem[i] <= 1'b0;
        end else if (i_wr_en && (i_wr_layer < LAYERS)) begin
            out_len_mem[i_wr_layer] <= i_wr_out_len;
            cout_mem[i_wr_layer] <= i_wr_cout;
            k_len_mem[i_wr_layer] <= i_wr_k_len;
            step_mem[i_wr_layer] <= i_wr_window_step;
            pool_mem[i_wr_layer] <= i_wr_pool_en;
            gap_mem[i_wr_layer] <= i_wr_gap_en;
            pad_mem[i_wr_layer] <= i_wr_pad_next;
            valid_mem[i_wr_layer] <= i_wr_valid;
            if (i_wr_valid) begin
                o_all_valid <= 1'b1;
                for (i = 0; i < LAYERS; i = i + 1)
                    if (i != i_wr_layer && !valid_mem[i])
                        o_all_valid <= 1'b0;
            end
        end
    end

    always @* begin
        if (i_rd_layer < LAYERS) begin
            o_out_len = out_len_mem[i_rd_layer];
            o_cout = cout_mem[i_rd_layer];
            o_k_len = k_len_mem[i_rd_layer];
            o_window_step = step_mem[i_rd_layer];
            o_pool_en = pool_mem[i_rd_layer];
            o_gap_en = gap_mem[i_rd_layer];
            o_pad_next = pad_mem[i_rd_layer];
            o_valid = valid_mem[i_rd_layer];
        end else begin
            o_out_len = 0;
            o_cout = 0;
            o_k_len = 0;
            o_window_step = 0;
            o_pool_en = 0;
            o_gap_en = 0;
            o_pad_next = 0;
            o_valid = 1'b0;
        end
    end
endmodule

`default_nettype wire
