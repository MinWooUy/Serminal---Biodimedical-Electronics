`timescale 1ns/1ps
// Testbench cho fmap_writer (ban co pool / pad / gap).
// Chay them 7x7:  sim/run_tb.sh --param ARRAY_ROWS=7 --param ARRAY_COLS=7 tb_fmap_writer
module tb_fmap_writer;
    parameter ARRAY_ROWS = 3;
    parameter ARRAY_COLS = 3;
    parameter DEPTH      = 8192;

    localparam DATA_W  = 8;
    localparam CFG_W   = 16;
    localparam ROW_W   = (ARRAY_ROWS <= 1) ? 1 : $clog2(ARRAY_ROWS + 1);
    localparam COL_W   = (ARRAY_COLS <= 1) ? 1 : $clog2(ARRAY_COLS + 1);
    localparam ADDR_W  = (DEPTH <= 1) ? 1 : $clog2(DEPTH);
    localparam TILE_W  = ARRAY_ROWS * ARRAY_COLS * DATA_W;

    localparam L_OUT      = 360;
    localparam C_OUT      = 16;
    localparam ROWS_LAST  = (L_OUT % ARRAY_ROWS == 0) ? ARRAY_ROWS : (L_OUT % ARRAY_ROWS);
    localparam COLS_LAST  = (C_OUT % ARRAY_COLS == 0) ? ARRAY_COLS : (C_OUT % ARRAY_COLS);

    reg clk = 0, reset = 1;
    always #5 clk = ~clk;

    reg                  i_start = 0, i_clear_overrun = 0;
    reg [TILE_W-1:0]     i_tile = 0;
    reg [CFG_W-1:0]      i_row_base = 0, i_col_base = 0, i_out_channels = 0, i_row_pad = 0;
    reg [ROW_W-1:0]      i_active_rows = 0;
    reg [COL_W-1:0]      i_active_cols = 0;
    reg [ADDR_W-1:0]     i_output_base = 0;
    reg                  i_pool_en = 0, i_gap_en = 0;
    wire                 o_wr_en, o_ready, o_overrun, o_done, o_gap_en;
    wire [ADDR_W-1:0]    o_wr_addr;
    wire [DATA_W-1:0]    o_wr_data, o_gap_data;
    wire [CFG_W-1:0]     o_gap_idx;

    fmap_writer #(
        .DATA_W(DATA_W), .ARRAY_ROWS(ARRAY_ROWS), .ARRAY_COLS(ARRAY_COLS),
        .CFG_W(CFG_W), .DEPTH(DEPTH)
    ) dut (
        .i_clk(clk), .i_reset(reset),
        .i_start(i_start), .i_clear_overrun(i_clear_overrun),
        .i_tile(i_tile),
        .i_row_base(i_row_base), .i_col_base(i_col_base),
        .i_active_rows(i_active_rows), .i_active_cols(i_active_cols),
        .i_out_channels(i_out_channels), .i_output_base(i_output_base),
        .i_row_pad(i_row_pad), .i_pool_en(i_pool_en), .i_gap_en(i_gap_en),
        .o_wr_en(o_wr_en), .o_wr_addr(o_wr_addr), .o_wr_data(o_wr_data),
        .o_gap_en(o_gap_en), .o_gap_idx(o_gap_idx), .o_gap_data(o_gap_data),
        .o_ready(o_ready), .o_overrun(o_overrun), .o_done(o_done)
    );

    integer errors = 0;
    integer seed = 32'h1234;
    integer total_wr = 0,  exp_wr = 0;      // so lan ghi SRAM
    integer total_gap = 0, exp_gap = 0;     // so lan phat GAP
    reg     exp_overrun = 0;
    reg [TILE_W-1:0] tile_ref;

    always @(posedge clk) if (!reset) begin
        if (o_wr_en  === 1'b1) total_wr  = total_wr + 1;
        if (o_gap_en === 1'b1) total_gap = total_gap + 1;
    end

    task check;
        input [31:0] got, want;
        input [8*24-1:0] what;
        begin
            if (got !== want) begin
                errors = errors + 1;
                $display("[FAIL] t=%0t %0s: got=%0d want=%0d", $time, what, got, want);
            end
        end
    endtask

    task idle;
        input integer n;
        integer i;
        begin
            for (i = 0; i < n; i = i + 1) begin
                @(negedge clk);
                check(o_wr_en, 0, "idle wr_en");
                check(o_gap_en, 0, "idle gap_en");
                check(o_done,  0, "idle done");
                check(o_ready, 1, "idle ready");
                check(o_overrun, exp_overrun, "idle overrun");
            end
        end
    endtask

    task clear_overrun;
        begin
            i_clear_overrun = 1;
            @(negedge clk);
            i_clear_overrun = 0;
            exp_overrun = 0;
            check(o_overrun, 0, "overrun cleared");
        end
    endtask

    // pool: nhom cap hang (2p, 2p+1) -> max; rb phai chan khi pool.
    // pad : so vi tri 0 o dau output. gap: phat o_gap_* thay vi ghi SRAM.
    task run_tile;
        input integer rb, cb, ar, ac, cout, obase;
        input inject;
        input pool;
        input integer pad;
        input gap;
        integer r, c, k, n, nrows, pos;
        reg signed [7:0] ta, tb2, d;
        begin
            for (r = 0; r < ARRAY_ROWS; r = r + 1)
                for (c = 0; c < ARRAY_COLS; c = c + 1)
                    tile_ref[(r*ARRAY_COLS + c)*DATA_W +: DATA_W] = $random(seed);
            nrows = pool ? (ar / 2) : ar;          // so hang OUTPUT
            n = nrows * ac;
            if (gap) exp_gap = exp_gap + n; else exp_wr = exp_wr + n;

            check(o_ready, 1, "ready before start");
            i_tile = tile_ref;
            i_row_base = rb;  i_col_base = cb;
            i_active_rows = ar; i_active_cols = ac;
            i_out_channels = cout; i_output_base = obase;
            i_row_pad = pad; i_pool_en = pool; i_gap_en = gap;
            i_start = 1;
            @(negedge clk);

            // doi het dau vao thanh rac: DUT phai dung ban da chot
            i_tile = ~tile_ref;
            i_row_base = 16'hFFFF; i_col_base = 16'hFFFF; i_out_channels = 16'hFFFF; i_row_pad = 16'hFFFF;
            i_active_rows = {ROW_W{1'b1}}; i_active_cols = {COL_W{1'b1}};
            i_output_base = {ADDR_W{1'b1}};
            i_pool_en = ~pool; i_gap_en = ~gap;
            // inject chi co nghia khi DUT dang ghi (n > 0); tile rong da ve IDLE ngay
            i_start = (n > 0) ? inject : 1'b0;
            if (inject && n > 0) exp_overrun = 1;

            for (k = 0; k < n; k = k + 1) begin
                r = k / ac;  c = k % ac;            // r: hang output
                if (pool) begin
                    ta  = tile_ref[((2*r)*ARRAY_COLS + c)*DATA_W +: DATA_W];
                    tb2 = tile_ref[((2*r+1)*ARRAY_COLS + c)*DATA_W +: DATA_W];
                    d   = (tb2 > ta) ? tb2 : ta;
                    pos = rb/2 + r;
                end else begin
                    d   = tile_ref[(r*ARRAY_COLS + c)*DATA_W +: DATA_W];
                    pos = rb + r;
                end
                if (gap) begin
                    check(o_wr_en,   0,      "gap: wr_en must be 0");
                    check(o_gap_en,  1,      "gap_en");
                    check(o_gap_idx, cb + c, "gap_idx");
                    check(o_gap_data, {24'd0, d}, "gap_data");
                end else begin
                    check(o_gap_en,  0,      "wr: gap_en must be 0");
                    check(o_wr_en,   1,      "wr_en");
                    check(o_wr_addr, obase + (pos + pad)*cout + (cb + c), "wr_addr");
                    check(o_wr_data, {24'd0, d}, "wr_data");
                end
                check(o_done,  0, "done too early");
                check(o_ready, 0, "ready while busy");
                @(negedge clk);
                i_start = 0;
            end

            check(o_done,  1, "done pulse");
            check(o_ready, 1, "ready after done");
            check(o_wr_en, 0, "wr_en after done");
            check(o_gap_en, 0, "gap_en after done");
            check(o_overrun, exp_overrun, "overrun flag");
        end
    endtask

    integer it, ar, ac, co, cb, rb, ob, pl, pd, gp;

    initial begin
        @(negedge clk); @(negedge clk);
        reset = 0;
        idle(2);

        // ---- A. Che do cu (pool=0 pad=0 gap=0): phai giong het ban truoc ----
        run_tile(0, 0, ARRAY_ROWS, ARRAY_COLS, ARRAY_COLS, 0, 0, 0, 0, 0);   idle(2);
        run_tile(0, 0, 1, ARRAY_COLS, ARRAY_COLS, 0, 0, 0, 0, 0);            idle(1);
        run_tile(0, 0, ARRAY_ROWS, 1, ARRAY_COLS, 0, 0, 0, 0, 0);            idle(1);
        run_tile(ARRAY_ROWS, ARRAY_COLS, ARRAY_ROWS, ARRAY_COLS, C_OUT, 100, 0, 0, 0, 0);  idle(1);
        run_tile(L_OUT - ROWS_LAST, C_OUT - COLS_LAST, ROWS_LAST, COLS_LAST, C_OUT, 0, 0, 0, 0, 0); idle(1);
        run_tile(0, 0, ARRAY_ROWS, ARRAY_COLS, C_OUT, 0, 0, 0, 0, 0);
        run_tile(0, ARRAY_COLS, ARRAY_ROWS, ARRAY_COLS, C_OUT, 0, 0, 0, 0, 0);   // lien tiep
        idle(2);

        // ---- B. Padding (chi doi dia chi) ----
        run_tile(0, 0, ARRAY_ROWS, ARRAY_COLS, C_OUT, 0, 0, 0, 2, 0);   idle(1);
        run_tile(ARRAY_ROWS, 0, ARRAY_ROWS, ARRAY_COLS, C_OUT, 50, 0, 0, 1, 0);  idle(1);

        // ---- C. Pool: tile 2 hang (nhu layer_top), pool 1 cap trong tile ----
        run_tile(0, 0, 2, ARRAY_COLS, C_OUT, 0, 0, 1, 0, 0);            idle(1);
        run_tile(4, ARRAY_COLS, 2, ARRAY_COLS, C_OUT, 0, 0, 1, 2, 0);   idle(1);   // row_base chan, co pad
        run_tile(358, C_OUT - COLS_LAST, 2, COLS_LAST, C_OUT, 0, 0, 1, 2, 0); idle(1); // cap cuoi cua conv0
        // pool nhung chi con 1 hang (le): bi bo, khong ghi gi, van done
        run_tile(0, 0, 1, ARRAY_COLS, C_OUT, 0, 0, 1, 0, 0);            idle(1);
        // ARRAY_ROWS hang + pool: (neu >= 4) co nhieu cap trong mot tile
        if (ARRAY_ROWS >= 4) begin
            run_tile(0, 0, ARRAY_ROWS, ARRAY_COLS, C_OUT, 0, 0, 1, 1, 0); idle(1);
        end

        // ---- D. GAP: khong ghi SRAM, phat (kenh, gia tri) ----
        run_tile(0, 0, ARRAY_ROWS, ARRAY_COLS, 64, 0, 0, 0, 0, 1);      idle(1);
        run_tile(ARRAY_ROWS, 60, ARRAY_ROWS, 4 <= ARRAY_COLS ? 4 : ARRAY_COLS, 64, 0, 0, 0, 0, 1); idle(1);

        // ---- E. Overrun (o moi che do) ----
        run_tile(0, 0, ARRAY_ROWS, ARRAY_COLS, C_OUT, 0, 1, 0, 0, 0);   idle(3);
        clear_overrun;
        run_tile(0, 0, 2, ARRAY_COLS, C_OUT, 0, 1, 1, 1, 0);            idle(3);
        clear_overrun;

        // ---- F. Tile rong ----
        run_tile(0, 0, 0, ARRAY_COLS, C_OUT, 0, 0, 0, 0, 0);            idle(1);
        run_tile(0, 0, ARRAY_ROWS, 0, C_OUT, 0, 0, 0, 0, 0);            idle(1);

        // ---- G. Reset giua luc dang ghi ----
        i_pool_en = 0; i_gap_en = 0; i_row_pad = 0;   // dau vao con "rac" tu ca truoc: dat lai ro rang
        for (it = 0; it < TILE_W/DATA_W; it = it + 1)
            tile_ref[it*DATA_W +: DATA_W] = $random(seed);
        i_tile = tile_ref; i_row_base = 0; i_col_base = 0;
        i_active_rows = ARRAY_ROWS; i_active_cols = ARRAY_COLS;
        i_out_channels = C_OUT; i_output_base = 0;
        i_start = 1;
        @(negedge clk);  i_start = 0;
        @(negedge clk);
        check(o_wr_en, 1, "wr_en before reset");
        exp_wr = exp_wr + 1;
        reset = 1;  #1;
        check(o_wr_en, 0, "reset: wr_en");
        check(o_ready, 1, "reset: ready");
        @(negedge clk);  reset = 0;
        idle(2);
        run_tile(0, 0, ARRAY_ROWS, ARRAY_COLS, C_OUT, 0, 0, 0, 0, 0);   idle(1);

        // ---- H. Ngau nhien, tron moi che do ----
        for (it = 0; it < 400; it = it + 1) begin
            pl = ({$random(seed)} % 3) == 0;
            gp = ({$random(seed)} % 5) == 0;
            pd = {$random(seed)} % 4;
            ar = 1 + ({$random(seed)} % ARRAY_ROWS);
            ac = 1 + ({$random(seed)} % ARRAY_COLS);
            co = ac + ({$random(seed)} % 24);
            cb = {$random(seed)} % (co - ac + 1);
            rb = ({$random(seed)} % 100) * 2;             // luon chan
            ob = {$random(seed)} % 1000;
            run_tile(rb, cb, ar, ac, co, ob, ({$random(seed)} % 8) == 0, pl, pd, gp);
            if (exp_overrun) begin idle(1); clear_overrun; end
            if ({$random(seed)} % 2) idle(1 + ({$random(seed)} % 3));
        end
        idle(3);

        check(total_wr,  exp_wr,  "total SRAM writes");
        check(total_gap, exp_gap, "total gap emits");
        if (errors == 0)
            $display("PASS tb_fmap_writer %0dx%0d: %0d writes, %0d gap emits checked", ARRAY_ROWS, ARRAY_COLS, total_wr, total_gap);
        else
            $display("[FAIL] tb_fmap_writer: %0d mismatches", errors);
        $finish;
    end

    initial begin
        #50000000;
        $display("[FAIL] TIMEOUT");
        $finish;
    end
endmodule
