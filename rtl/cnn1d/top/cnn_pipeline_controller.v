`default_nettype none

// Bo tuan tu layer tu dong cho mo hinh feature-map hai bank.
// Module nay khong quan ly frame UART hay bo nho tensor; cac khoi do nam ben ngoai.
module cnn_pipeline_controller #(
    parameter LAYERS = 4,
    parameter CFG_W = 16,
    parameter ADDR_W = 16,
    parameter ARRAY_ROWS = 3,
    parameter LAYER_W = (LAYERS <= 1) ? 1 : $clog2(LAYERS)
)(
    input wire i_clk,
    input wire i_reset,

    input wire i_cfg_wr_en,
    input wire [LAYER_W-1:0] i_cfg_wr_layer,
    input wire [CFG_W-1:0] i_cfg_wr_out_len,
    input wire [CFG_W-1:0] i_cfg_wr_cout,
    input wire [CFG_W-1:0] i_cfg_wr_k_len,
    input wire [ADDR_W-1:0] i_cfg_wr_window_step,
    input wire i_cfg_wr_pool_en,
    input wire i_cfg_wr_gap_en,
    input wire [CFG_W-1:0] i_cfg_wr_pad_next,
    input wire i_cfg_wr_valid,

    input wire i_run,
    input wire i_layer_done,

    output wire [LAYER_W-1:0] o_layer_index,
    output wire o_layer_start,
    output wire o_busy,
    output wire o_done,
    output wire o_error,
    output wire o_input_bank,
    output wire o_output_bank,

    output wire [CFG_W-1:0] o_out_len,
    output wire [CFG_W-1:0] o_cout,
    output wire [CFG_W-1:0] o_k_len,
    output wire [ADDR_W-1:0] o_window_step,
    output wire o_pool_en,
    output wire o_gap_en,
    output wire [CFG_W-1:0] o_pad_next,
    output wire o_config_valid,
    output wire o_all_config_valid
);
    localparam S_IDLE = 2'd0;
    localparam S_START = 2'd1;
    localparam S_WAIT = 2'd2;
    localparam S_DONE = 2'd3;

    reg [1:0] state_q;
    reg [LAYER_W-1:0] layer_q;
    reg input_bank_q;
    reg error_q;

    wire all_config_valid;
    layer_config_ram #(
        .LAYERS(LAYERS), .CFG_W(CFG_W), .ADDR_W(ADDR_W), .LAYER_W(LAYER_W)
    ) u_config (
        .i_clk(i_clk), .i_reset(i_reset),
        .i_wr_en(i_cfg_wr_en), .i_wr_layer(i_cfg_wr_layer),
        .i_wr_out_len(i_cfg_wr_out_len), .i_wr_cout(i_cfg_wr_cout),
        .i_wr_k_len(i_cfg_wr_k_len), .i_wr_window_step(i_cfg_wr_window_step),
        .i_wr_pool_en(i_cfg_wr_pool_en), .i_wr_gap_en(i_cfg_wr_gap_en),
        .i_wr_pad_next(i_cfg_wr_pad_next), .i_wr_valid(i_cfg_wr_valid),
        .i_rd_layer(layer_q), .o_out_len(o_out_len), .o_cout(o_cout),
        .o_k_len(o_k_len), .o_window_step(o_window_step),
        .o_pool_en(o_pool_en), .o_gap_en(o_gap_en), .o_pad_next(o_pad_next),
        .o_valid(o_config_valid), .o_all_valid(all_config_valid)
    );

    assign o_layer_index = layer_q;
    assign o_layer_start = (state_q == S_START);
    assign o_busy = (state_q == S_START) || (state_q == S_WAIT);
    assign o_done = (state_q == S_DONE);
    assign o_error = error_q;
    assign o_input_bank = input_bank_q;
    assign o_output_bank = ~input_bank_q;
    assign o_all_config_valid = all_config_valid;

    always @(posedge i_clk or posedge i_reset) begin
        if (i_reset) begin
            state_q <= S_IDLE;
            layer_q <= 0;
            input_bank_q <= 1'b0;
            error_q <= 1'b0;
        end else begin
            case (state_q)
                S_IDLE: begin
                    error_q <= 1'b0;
                    if (i_run) begin
                        if (all_config_valid) begin
                            layer_q <= 0;
                            input_bank_q <= 1'b0;
                            state_q <= S_START;
                        end else begin
                            error_q <= 1'b1;
                        end
                    end
                end
                S_START: state_q <= S_WAIT;
                S_WAIT: begin
                    if (i_layer_done) begin
                        if (layer_q == LAYERS - 1) begin
                            state_q <= S_DONE;
                        end else begin
                            layer_q <= layer_q + 1'b1;
                            input_bank_q <= ~input_bank_q;
                            state_q <= S_START;
                        end
                    end
                end
                S_DONE: begin
                    if (i_run) begin
                        if (all_config_valid) begin
                            layer_q <= 0;
                            input_bank_q <= 1'b0;
                            state_q <= S_START;
                        end else begin
                            error_q <= 1'b1;
                        end
                    end
                end
                default: state_q <= S_IDLE;
            endcase
        end
    end
endmodule

`default_nettype wire
