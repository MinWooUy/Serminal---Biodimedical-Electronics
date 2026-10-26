`default_nettype none
module conv1d_weight_loader #(
    parameter DATA_W = 8,
    parameter ADDR_W = 16,
    parameter CFG_W = 16,

    parameter ARRAY_COLS = 3,

    parameter COL_W = (ARRAY_COLS <= 1) ? 1 : $clog2(ARRAY_COLS + 1)
)(
    input wire i_clk,
    input wire i_reset,

	// control
    input wire i_start,

    // =====================================================
    // WEIGHT CONFIG
    //
    // Weight memory layout:
    //
    // col0: K contiguous weights
    // col1: K contiguous weights
    // col2: K contiguous weights
    // ...
    //
    // Therefore:
    //
    // k_len       = Kw * Cin
    // window_step = k_len
    // num_windows = active_cols
    // =====================================================

    input wire [ADDR_W-1:0] i_base_addr,

    input wire [CFG_W-1:0] i_k_len,

    input wire [COL_W-1:0] i_active_cols,
	
    // Backpressure: cờ full của từng lane FIFO (từ weight_feeder.o_mem_full)
    input wire [ARRAY_COLS-1:0] i_lane_full,

	// WEIGHT SRAM READ INTERFACE

    output wire o_sram_rd_en,

    output wire [ADDR_W-1:0] o_sram_rd_addr,

    input wire signed [DATA_W-1:0] i_sram_rd_data,

    input wire i_sram_rd_valid,
	 
	 // WRITE INTO EXISTING WEIGHT FEEDER FIFOS

    output reg [ARRAY_COLS-1:0] o_wgt_wr_en,

    output reg signed [ARRAY_COLS*DATA_W-1:0] o_wgt_wr_data,

	// Status
    output wire o_busy,
    output reg o_done,
    output wire [2:0] o_state_dbg
);
	// FSM

    localparam S_IDLE  = 3'd0;
    localparam S_REQ   = 3'd1;
    localparam S_WAIT  = 3'd2;
    localparam S_WRITE = 3'd3;
    localparam S_DONE  = 3'd4;

    reg [2:0] state_q;

    assign o_state_dbg = state_q;

    assign o_busy = (state_q != S_IDLE) && (state_q != S_DONE);
	
    // Current physical column (khai báo trước wr_ok vì wr_ok dùng nó).
    // AGU sequence: window0 -> column0 FIFO, window1 -> column1 FIFO, ...
    // For Weight, one AGU window = one output channel.
    reg [COL_W-1:0] col_cnt_q;

    // Lane đích còn chỗ thì mới ghi
    wire wr_ok = ~i_lane_full[col_cnt_q];

	// AGU
    wire [ADDR_W-1:0] agu_addr;

    wire agu_valid;
    wire agu_busy;
    wire agu_done;

    wire agu_k_first;
    wire agu_k_last;

    wire agu_window_first;
    wire agu_window_last;


    // =====================================================
    // AGU CONTROL
    //
    // Start when loader accepts start.
    //
    // Advance only when current SRAM data has been
    // committed into the corresponding Weight FIFO.
    // =====================================================

    wire agu_start;
    wire agu_advance;

    assign agu_start = (state_q == S_IDLE) && i_start;
    assign agu_advance = (state_q == S_WRITE) && wr_ok;


    // =====================================================
    // AGU CONFIG ADAPTATION
    // =====================================================

    // Each output channel contains K contiguous values.
    //
    // Therefore:
    //
    // col0 base = base
    // col1 base = base + K
    // col2 base = base + 2K
    // ...

    wire [ADDR_W-1:0] weight_step;

    assign weight_step = i_k_len;

    // Number of AGU "windows" = number of active SA columns.

    wire [CFG_W-1:0] agu_num_windows;

    assign agu_num_windows = i_active_cols;


    // =====================================================
    // AGU INSTANCE
    //
    // Reuse the SAME AGU already verified for IFMAP.
    // =====================================================

    conv1d_address_generating #(
        .ADDR_W(ADDR_W),
        .CFG_W(CFG_W)
    ) u_agu (
        .i_clk(i_clk),
        .i_reset(i_reset),

        .i_start(agu_start),
        .i_advance(agu_advance),

        .i_base_addr(i_base_addr),

        .i_k_len(i_k_len),

        .i_window_step(weight_step),

        .i_num_windows(agu_num_windows),

        .o_addr(agu_addr),
        .o_valid(agu_valid),

        .o_k_first(agu_k_first),
        .o_k_last(agu_k_last),

        .o_window_first(agu_window_first),
        .o_window_last(agu_window_last),

        .o_busy(agu_busy),
        .o_done(agu_done)
    );


    // =====================================================
    // SRAM RESPONSE REGISTER
    // =====================================================

    reg signed [DATA_W-1:0] rd_data_q;

    // =====================================================
    // SRAM REQUEST
    // =====================================================

    assign o_sram_rd_en = (state_q == S_REQ) && agu_valid;
    assign o_sram_rd_addr = agu_addr;


    // =====================================================
    // WEIGHT FIFO WRITE CONTROL
    // =====================================================
    always @(*) begin
        o_wgt_wr_en = {ARRAY_COLS{1'b0}};
        o_wgt_wr_data = {ARRAY_COLS*DATA_W{1'b0}};

        if(state_q == S_WRITE && wr_ok) begin
            o_wgt_wr_en[col_cnt_q] = 1'b1;
            o_wgt_wr_data[col_cnt_q*DATA_W +: DATA_W] = rd_data_q;

        end
    end
	
	// FSM
    always @(posedge i_clk) begin
        if(i_reset) begin
            state_q <= S_IDLE;
            col_cnt_q <= 0;
            rd_data_q <= 0;
            o_done <= 1'b0;
        end
        else begin
            // DONE is a one-cycle pulse
            o_done <= 1'b0;
            case(state_q)
                S_IDLE: begin
                    col_cnt_q <= 0;
                    if(i_start) begin
                        state_q <= S_REQ;
                    end
                end

                // =========================================
                // ISSUE WEIGHT SRAM READ
                // =========================================

                S_REQ: begin
                    if(agu_valid) begin
                        // During this state:
                        //
                        // o_sram_rd_en   = 1
                        // o_sram_rd_addr = agu_addr
                        state_q <= S_WAIT;
                    end
                end

                // =========================================
                // WAIT FOR SRAM RESPONSE
                // =========================================

                S_WAIT: begin
                    if(i_sram_rd_valid) begin
                        rd_data_q <= i_sram_rd_data;
                        state_q <= S_WRITE;
                    end
                end

                // =========================================
                // WRITE TO CURRENT COLUMN FIFO
                //
                // During S_WRITE:
                //
                // o_wgt_wr_en[col] = 1
                // agu_advance      = 1
                //
                // At the edge:
                //
                // FIFO captures weight
                // AGU advances
                // =========================================

                S_WRITE: begin
                    if(wr_ok) begin
                        // -------------------------------------
                        // Last K element of current output
                        // channel?
                        // -------------------------------------
                        if(agu_k_last) begin
                            // ---------------------------------
                            // Last active output channel?
                            // ---------------------------------
                            if(agu_window_last) begin
                                state_q <= S_DONE;
                            end
                            else begin
                                col_cnt_q <= col_cnt_q + 1'b1;
                                state_q <= S_REQ;
                            end
                        end
                        else begin
                            state_q <= S_REQ;
                        end
                    end
                end

                S_DONE: begin
                    o_done <= 1'b1;
                    if(!i_start) begin
                        state_q <= S_IDLE;
                    end
                end

                default: begin
                    state_q <= S_IDLE;
                    col_cnt_q <= 0;
                end
            endcase
        end
    end
endmodule
`default_nettype wire
