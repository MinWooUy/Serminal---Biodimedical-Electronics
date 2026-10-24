// One registered tile: bias -> round/shift -> optional ReLU -> symmetric INT8.
// Rows are positions, columns are output channels. No backpressure.
module conv1d_post_process #(
    parameter ROWS=3, COLS=3,
    parameter ROW_W=$clog2(ROWS+1), COL_W=$clog2(COLS+1)
)(
    input wire i_clk, i_reset, i_valid,
    input wire [ROWS*COLS*32-1:0] i_tile,
    input wire [COLS*32-1:0] i_bias,
    input wire [COLS*8-1:0] i_shift, // signed: k + a_in - a_out
    input wire i_relu,
    input wire [ROW_W-1:0] i_active_rows,
    input wire [COL_W-1:0] i_active_cols,
    output reg [ROWS*COLS*8-1:0] o_tile,
    output reg o_valid
);
    function [7:0] quantize;
        input signed [31:0] acc, bias;
        input signed [7:0] shift;
        input relu;
        reg signed [63:0] value;
        begin
            // Widen BEFORE addition: no wrap at the INT32 boundary.
            value = {{32{acc[31]}},acc};
            value = value + {{32{bias[31]}},bias};
            if (shift > 0) begin
                if (shift >= 34) value = 0;
                else value = (value + (64'sd1 <<< (shift-1))) >>> shift;
            end else if (shift < 0) begin
                // Any nonzero integer shifted >= 8 already saturates INT8.
                if (shift <= -8) begin
                    if (value > 0) value = 127;
                    else if (value < 0) value = -127;
                end else value = value <<< (-shift);
            end
            if (relu && value < 0) value = 0;
            if (value > 127) value = 127;
            if (value < -127) value = -127;
            quantize = value[7:0];
        end
    endfunction
    integer r,c;
    always @(posedge i_clk or posedge i_reset) begin
        if (i_reset) begin o_valid<=0; o_tile<=0; end
        else begin
            o_valid <= i_valid;
            if (i_valid) begin
                for (r=0;r<ROWS;r=r+1)
                    for (c=0;c<COLS;c=c+1)
                        if (r<i_active_rows && c<i_active_cols)
                            o_tile[(r*COLS+c)*8 +: 8] <= quantize(
                                i_tile[(r*COLS+c)*32 +: 32],
                                i_bias[c*32 +: 32],i_shift[c*8 +: 8],i_relu);
                        else o_tile[(r*COLS+c)*8 +: 8] <= 0;
            end
        end
    end
endmodule
