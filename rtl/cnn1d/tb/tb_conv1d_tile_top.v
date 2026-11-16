`timescale 1ns/1ps

// Geometry-scalable smoke test for conv1d_tile_top.
// Examples:
//   sim/run_tb.sh tb_conv1d_tile_top
//   sim/run_tb.sh --param ARRAY_ROWS=7 --param ARRAY_COLS=7 tb_conv1d_tile_top
module tb_conv1d_tile_top #(
    parameter ARRAY_ROWS = 3,
    parameter ARRAY_COLS = 3
);
    localparam DATA_W = 8;
    localparam ACC_W = 32;
    localparam ADDR_W = 6;
    localparam CFG_W = 16;
    localparam IFMAP_MEM_DEPTH = 64;
    localparam WGT_MEM_DEPTH = 64;
    localparam FEEDER_DEPTH = 16;
    localparam K_W = 16;
    localparam K_LEN = 3;
    localparam WINDOW_STEP = 1;
    localparam ROW_W = (ARRAY_ROWS <= 1) ? 1 : $clog2(ARRAY_ROWS + 1);
    localparam COL_W = (ARRAY_COLS <= 1) ? 1 : $clog2(ARRAY_COLS + 1);
    localparam N_OUT = ARRAY_ROWS * ARRAY_COLS;
    localparam PARTIAL_ROWS = (ARRAY_ROWS > 1) ? ARRAY_ROWS - 1 : 1;
    localparam PARTIAL_COLS = (ARRAY_COLS > 1) ? ARRAY_COLS - 1 : 1;

    reg i_clk = 1'b0;
    always #5 i_clk = ~i_clk;

    reg i_reset, i_start;
    reg [CFG_W-1:0] i_k_len;
    reg [ADDR_W-1:0] i_window_step, i_ifmap_base_addr, i_wgt_base_addr;
    reg [ROW_W-1:0] i_active_rows;
    reg [COL_W-1:0] i_active_cols;
    reg i_ifmap_mem_wr_en, i_ifmap_mem_swap;
    reg [ADDR_W-1:0] i_ifmap_mem_wr_addr;
    reg signed [DATA_W-1:0] i_ifmap_mem_wr_data;
    reg i_wgt_mem_wr_en, i_wgt_mem_swap;
    reg [ADDR_W-1:0] i_wgt_mem_wr_addr;
    reg signed [DATA_W-1:0] i_wgt_mem_wr_data;

    wire [N_OUT*ACC_W-1:0] o_tile;
    wire o_tile_valid, o_busy, o_done, o_cfg_err;
    wire [N_OUT*8-1:0] post_tile;
    wire post_valid;
    wire o_bias_rd_en;
    wire [COL_W-1:0] o_bias_rd_idx;
    wire o_drain_valid;
    wire [COL_W-1:0] o_drain_idx;
    wire [ARRAY_ROWS*ACC_W-1:0] o_drain_vec;
    wire [2:0] o_state_dbg, o_core_state_dbg, o_ifmap_loader_state, o_wgt_loader_state;
    wire o_ifmap_read_bank, o_ifmap_write_bank, o_wgt_read_bank, o_wgt_write_bank;

    integer errors, timeout_cnt, post_checks;
    integer current_rows, current_cols;
    integer r, c, idx;
    reg signed [ACC_W-1:0] expected [0:N_OUT-1];
    reg signed [63:0] post_expected;

    conv1d_tile_top #(
        .POST_ENABLE(1),
        .DATA_W(DATA_W), .ACC_W(ACC_W),
        .ADDR_W(ADDR_W), .CFG_W(CFG_W),
        .IFMAP_MEM_DEPTH(IFMAP_MEM_DEPTH), .WGT_MEM_DEPTH(WGT_MEM_DEPTH),
        .FEEDER_DEPTH(FEEDER_DEPTH), .K_W(K_W),
        .ARRAY_ROWS(ARRAY_ROWS), .ARRAY_COLS(ARRAY_COLS),
        .ROW_W(ROW_W), .COL_W(COL_W)
    ) dut (
        .i_clk(i_clk), .i_reset(i_reset),
        .i_start(i_start), .i_k_len(i_k_len), .i_window_step(i_window_step),
        .i_active_rows(i_active_rows), .i_active_cols(i_active_cols),
        .i_ifmap_base_addr(i_ifmap_base_addr), .i_wgt_base_addr(i_wgt_base_addr),
        .i_ifmap_mem_wr_en(i_ifmap_mem_wr_en),
        .i_ifmap_mem_wr_addr(i_ifmap_mem_wr_addr),
        .i_ifmap_mem_wr_data(i_ifmap_mem_wr_data), .i_ifmap_mem_swap(i_ifmap_mem_swap),
        .i_wgt_mem_wr_en(i_wgt_mem_wr_en),
        .i_wgt_mem_wr_addr(i_wgt_mem_wr_addr),
        .i_wgt_mem_wr_data(i_wgt_mem_wr_data), .i_wgt_mem_swap(i_wgt_mem_swap),
        .i_preload_mode(1'b0), .i_bias_rd_data({ACC_W{1'b0}}),
        .o_bias_rd_en(o_bias_rd_en), .o_bias_rd_idx(o_bias_rd_idx),
        .i_drain_ready(1'b1), .o_drain_valid(o_drain_valid),
        .o_drain_idx(o_drain_idx), .o_drain_vec(o_drain_vec),
        .i_post_shift({ARRAY_COLS{8'd1}}), .i_post_relu(1'b1),
        .o_post_tile(post_tile), .o_post_valid(post_valid),
        .o_tile(o_tile), .o_tile_valid(o_tile_valid), .o_busy(o_busy), .o_done(o_done),
        .o_state_dbg(o_state_dbg), .o_core_state_dbg(o_core_state_dbg),
        .o_ifmap_loader_state(o_ifmap_loader_state), .o_wgt_loader_state(o_wgt_loader_state),
        .o_ifmap_read_bank(o_ifmap_read_bank), .o_ifmap_write_bank(o_ifmap_write_bank),
        .o_wgt_read_bank(o_wgt_read_bank), .o_wgt_write_bank(o_wgt_write_bank),
        .o_cfg_err(o_cfg_err),
        .i_dbg_rd_en(1'b0),
        .i_dbg_rd_addr({ADDR_W{1'b0}}),
        .o_dbg_rd_data(),
        .o_dbg_rd_valid()
    );

    function integer sample_value;
        input integer address;
        begin sample_value = ((address * 5 + 3) % 17) - 8; end
    endfunction

    function integer weight_value;
        input integer column;
        input integer tap;
        begin weight_value = ((column * 7 + tap * 3 + 1) % 13) - 6; end
    endfunction

    task reset_dut;
        begin
            i_reset = 1'b1; i_start = 1'b0;
            i_k_len = 0; i_window_step = 0; i_active_rows = 0; i_active_cols = 0;
            i_ifmap_base_addr = 0; i_wgt_base_addr = 0;
            i_ifmap_mem_wr_en = 1'b0; i_ifmap_mem_wr_addr = 0; i_ifmap_mem_wr_data = 0;
            i_ifmap_mem_swap = 1'b0;
            i_wgt_mem_wr_en = 1'b0; i_wgt_mem_wr_addr = 0; i_wgt_mem_wr_data = 0;
            i_wgt_mem_swap = 1'b0;
            repeat (4) @(posedge i_clk);
            @(negedge i_clk); i_reset = 1'b0;
        end
    endtask

    task write_ifmap;
        input integer address;
        input integer value;
        begin
            @(negedge i_clk);
            i_ifmap_mem_wr_en = 1'b1;
            i_ifmap_mem_wr_addr = address;
            i_ifmap_mem_wr_data = value;
            @(posedge i_clk);
            @(negedge i_clk);
            i_ifmap_mem_wr_en = 1'b0;
        end
    endtask

    task write_weight;
        input integer address;
        input integer value;
        begin
            @(negedge i_clk);
            i_wgt_mem_wr_en = 1'b1;
            i_wgt_mem_wr_addr = address;
            i_wgt_mem_wr_data = value;
            @(posedge i_clk);
            @(negedge i_clk);
            i_wgt_mem_wr_en = 1'b0;
        end
    endtask

    task load_data;
        integer address;
        begin
            for (address = 0; address < K_LEN + (ARRAY_ROWS - 1) * WINDOW_STEP; address = address + 1)
                write_ifmap(address, sample_value(address));
            for (address = 0; address < ARRAY_COLS * K_LEN; address = address + 1)
                write_weight(address, weight_value(address / K_LEN, address % K_LEN));
            @(negedge i_clk);
            i_ifmap_mem_swap = 1'b1; i_wgt_mem_swap = 1'b1;
            @(posedge i_clk);
            @(negedge i_clk);
            i_ifmap_mem_swap = 1'b0; i_wgt_mem_swap = 1'b0;
        end
    endtask

    task build_reference;
        input integer active_rows;
        input integer active_cols;
        integer row, col, tap, acc;
        begin
            for (row = 0; row < ARRAY_ROWS; row = row + 1)
                for (col = 0; col < ARRAY_COLS; col = col + 1) begin
                    acc = 0;
                    if (row < active_rows && col < active_cols)
                        for (tap = 0; tap < K_LEN; tap = tap + 1)
                            acc = acc + sample_value(row * WINDOW_STEP + tap) * weight_value(col, tap);
                    expected[row * ARRAY_COLS + col] = acc;
                end
        end
    endtask

    task wait_tile_valid;
        begin
            timeout_cnt = 0;
            while (o_tile_valid !== 1'b1 && timeout_cnt < 5000) begin
                @(posedge i_clk);
                timeout_cnt = timeout_cnt + 1;
            end
            if (timeout_cnt >= 5000) begin
                $display("[FAIL] Timeout waiting for o_tile_valid");
                errors = errors + 1;
            end
        end
    endtask

    task wait_done;
        begin
            timeout_cnt = 0;
            while (o_done !== 1'b1 && timeout_cnt < 5000) begin
                @(posedge i_clk);
                timeout_cnt = timeout_cnt + 1;
            end
            if (timeout_cnt >= 5000) begin
                $display("[FAIL] Timeout waiting for o_done");
                errors = errors + 1;
            end
        end
    endtask

    task check_raw_tile;
        integer row, col, element;
        reg signed [ACC_W-1:0] value;
        begin
            for (row = 0; row < ARRAY_ROWS; row = row + 1)
                for (col = 0; col < ARRAY_COLS; col = col + 1) begin
                    element = row * ARRAY_COLS + col;
                    value = $signed(o_tile[element * ACC_W +: ACC_W]);
                    if (value !== expected[element]) begin
                        $display("[FAIL] raw C[%0d][%0d]=%0d expected=%0d", row, col, value, expected[element]);
                        errors = errors + 1;
                    end
                end
        end
    endtask

    task run_case;
        input [8*64-1:0] case_name;
        input integer active_rows;
        input integer active_cols;
        begin
            $display("=== %0s: array=%0dx%0d active=%0dx%0d ===",
                     case_name, ARRAY_ROWS, ARRAY_COLS, active_rows, active_cols);
            reset_dut();
            load_data();
            build_reference(active_rows, active_cols);
            current_rows = active_rows; current_cols = active_cols;
            @(negedge i_clk);
            i_k_len = K_LEN; i_window_step = WINDOW_STEP;
            i_active_rows = active_rows; i_active_cols = active_cols;
            i_ifmap_base_addr = 0; i_wgt_base_addr = 0; i_start = 1'b1;
            @(posedge i_clk);
            @(negedge i_clk); i_start = 1'b0;
            wait_tile_valid();
            if (o_tile_valid === 1'b1) check_raw_tile();
            wait_done();
            // Post processing walks the rows one per cycle: o_post_valid
            // comes up to ARRAY_ROWS+1 cycles after o_tile_valid.
            repeat (ARRAY_ROWS + 6) @(posedge i_clk);
        end
    endtask

    // Post output uses the same physical row-major tile layout as o_tile.
    always @(negedge i_clk) begin
        if (post_valid) begin
            post_checks = post_checks + 1;
            for (r = 0; r < ARRAY_ROWS; r = r + 1)
                for (c = 0; c < ARRAY_COLS; c = c + 1) begin
                    idx = r * ARRAY_COLS + c;
                    post_expected = $signed(o_tile[idx * ACC_W +: ACC_W]);
                    post_expected = (post_expected + 1) >>> 1;
                    if (post_expected < 0) post_expected = 0;
                    if (post_expected > 127) post_expected = 127;
                    if (r >= current_rows || c >= current_cols) post_expected = 0;
                    if ($signed(post_tile[idx * 8 +: 8]) !== post_expected) begin
                        $display("[FAIL] post C[%0d][%0d]=%0d expected=%0d", r, c,
                                 $signed(post_tile[idx * 8 +: 8]), post_expected);
                        errors = errors + 1;
                    end
                end
        end
    end

    initial begin
        errors = 0; post_checks = 0; current_rows = 0; current_cols = 0;
        run_case("full", ARRAY_ROWS, ARRAY_COLS);
        run_case("partial", PARTIAL_ROWS, PARTIAL_COLS);
        run_case("corner", 1, 1);
        $display("==============================================");
        $display(" CONV1D TILE SCALABLE ARRAY TEST SUMMARY");
        $display(" ARRAY = %0dx%0d", ARRAY_ROWS, ARRAY_COLS);
        if (errors == 0 && post_checks == 3)
            $display(" RESULT : ALL TESTS PASSED (raw + post)");
        else begin
            $display(" RESULT : FAILED (%0d errors, %0d post results)", errors, post_checks);
            $fatal(1, "tb_conv1d_tile_top failed");
        end
        $display("==============================================");
        $finish;
    end

    initial begin
        #2000000;
        $display("[TIMEOUT] tb_conv1d_tile_top");
        $fatal(1, "tb_conv1d_tile_top timeout");
    end
endmodule
