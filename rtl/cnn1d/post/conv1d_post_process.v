// One tile: acc -> round/shift -> optional ReLU -> symmetric INT8.
// Rows are positions, columns are output channels. One quantizer per column
// walks the rows one per cycle, so cost grows with COLS, not ROWS*COLS.
// i_tile must stay stable from the i_valid cycle until o_valid. i_valid is
// ignored while o_busy is high. Latency: o_valid is high ROWS+1 cycles after
// the i_valid edge, for one cycle. Config inputs are latched on i_valid.
module conv1d_post_process #(
    parameter ROWS=3, COLS=3,
    parameter ROW_W=$clog2(ROWS+1), COL_W=$clog2(COLS+1)
)(
    input wire i_clk, i_reset, i_valid,
    input wire [ROWS*COLS*32-1:0] i_tile,
    input wire [COLS*8-1:0] i_shift, // signed: k + a_in - a_out
    input wire i_relu,
    input wire [ROW_W-1:0] i_active_rows,
    input wire [COL_W-1:0] i_active_cols,
    output reg [ROWS*COLS*8-1:0] o_tile,
    output reg o_valid,
    output wire o_busy
);
    // (acc + 2^(s-1)) >>> s on a 64-bit value, built from
    // floor((floor(acc / 2^(s-1)) + 1) / 2), so no wide rounding constant.
    function [7:0] quantize;
        input signed [31:0] acc;
        input signed [7:0] shift;
        input relu;
        reg signed [32:0] half;
        reg [4:0] amt;
        reg [2:0] left;
        reg signed [38:0] wide;
        reg signed [38:0] value;
        begin
            if (shift > 0) begin
                // Any 32-bit acc rounds to 0 once shift >= 32.
                if (shift >= 32) value = 0;
                else begin
                    amt = shift[4:0] - 5'd1;
                    half = acc >>> amt;
                    value = (half + 33'sd1) >>> 1;
                end
            end else if (shift < 0) begin
                // Any nonzero integer shifted >= 8 already saturates INT8.
                if (shift <= -8) begin
                    if (acc > 0) value = 127;
                    else if (acc < 0) value = -127;
                    else value = 0;
                end else begin
                    left = 3'd0 - shift[2:0];
                    wide = acc;
                    value = wide <<< left;
                end
            end else value = acc;
            if (relu && value < 0) value = 0;
            if (value > 39'sd127) value = 127;
            if (value < -39'sd127) value = -127;
            quantize = value[7:0];
        end
    endfunction

    localparam IDX_W = (ROWS <= 1) ? 1 : $clog2(ROWS);

    reg busy;
    reg [IDX_W-1:0] row_q;
    reg [COLS*8-1:0] shift_q;
    reg relu_q;
    reg [ROW_W-1:0] rows_q;
    reg [COL_W-1:0] cols_q;
    wire [COLS*8-1:0] q_bus;
    assign o_busy = busy;

    genvar gc;
    generate for (gc = 0; gc < COLS; gc = gc + 1) begin : g_col
        reg [31:0] acc_sel;
        integer rr;
        always @* begin
            acc_sel = 32'd0;
            for (rr = 0; rr < ROWS; rr = rr + 1)
                if (row_q == rr) acc_sel = i_tile[(rr*COLS+gc)*32 +: 32];
        end
        assign q_bus[gc*8 +: 8] = quantize(acc_sel, shift_q[gc*8 +: 8], relu_q);
    end endgenerate

    integer r, c;
    always @(posedge i_clk or posedge i_reset) begin
        if (i_reset) begin
            busy <= 0; 
			row_q <= 0; 
            o_valid <= 0; 
            o_tile <= 0;
            
            shift_q <= 0; 
            relu_q <= 0; 
            rows_q <= 0; 
            cols_q <= 0;
        end else begin
            o_valid <= 0;
            if (!busy) begin
                if (i_valid) begin
                    busy <= 1; 
					row_q <= 0;
                    
					shift_q <= i_shift; 
					relu_q <= i_relu;
                    
					rows_q <= i_active_rows; 
					cols_q <= i_active_cols;
                end
            end else begin
                for (r = 0; r < ROWS; r = r + 1)
                    if (row_q == r)
                        for (c = 0; c < COLS; c = c + 1)
                            o_tile[(r*COLS+c)*8 +: 8] <= (r < rows_q && c < cols_q) ? q_bus[c*8 +: 8] : 8'd0;
                if (row_q == ROWS-1) begin
                    busy <= 0; 
					o_valid <= 1;
					row_q <= 0;
                end else row_q <= row_q + 1'b1;
            end
        end
    end
endmodule
