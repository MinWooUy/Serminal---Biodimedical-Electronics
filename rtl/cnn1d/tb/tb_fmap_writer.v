`timescale 1ns/1ps
// Testbench cho fmap_writer.
// Chạy thêm cấu hình 7x7:
//   sim/run_tb.sh --param ARRAY_ROWS=7 --param ARRAY_COLS=7 tb_fmap_writer
module tb_fmap_writer;
    parameter ARRAY_ROWS = 3;
    parameter ARRAY_COLS = 3;
    parameter DEPTH      = 8192;     // đủ cho output Conv0: 360*16 = 5760 byte

    localparam DATA_W  = 8;
    localparam CFG_W   = 16;
    localparam ROW_W   = (ARRAY_ROWS <= 1) ? 1 : $clog2(ARRAY_ROWS + 1);
    localparam COL_W   = (ARRAY_COLS <= 1) ? 1 : $clog2(ARRAY_COLS + 1);
    localparam ADDR_W  = (DEPTH <= 1) ? 1 : $clog2(DEPTH);
    localparam TILE_W  = ARRAY_ROWS * ARRAY_COLS * DATA_W;

    // Kích thước Conv0 để làm các ca "giống thật"
    localparam L_OUT      = 360;
    localparam C_OUT      = 16;
    localparam ROWS_LAST  = (L_OUT % ARRAY_ROWS == 0) ? ARRAY_ROWS : (L_OUT % ARRAY_ROWS);
    localparam COLS_LAST  = (C_OUT % ARRAY_COLS == 0) ? ARRAY_COLS : (C_OUT % ARRAY_COLS);

    reg clk = 0, reset = 1;
    always #5 clk = ~clk;

    reg                  i_start = 0, i_clear_overrun = 0;
    reg [TILE_W-1:0]     i_tile = 0;
    reg [CFG_W-1:0]      i_row_base = 0, i_col_base = 0, i_out_channels = 0;
    reg [ROW_W-1:0]      i_active_rows = 0;
    reg [COL_W-1:0]      i_active_cols = 0;
    reg [ADDR_W-1:0]     i_output_base = 0;
    wire                 o_wr_en, o_ready, o_overrun, o_done;
    wire [ADDR_W-1:0]    o_wr_addr;
    wire [DATA_W-1:0]    o_wr_data;

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
        .o_wr_en(o_wr_en), .o_wr_addr(o_wr_addr), .o_wr_data(o_wr_data),
        .o_ready(o_ready), .o_overrun(o_overrun), .o_done(o_done)
    );

    integer errors = 0;
    integer seed = 32'h1234;
    integer total_wr = 0;          // số chu kỳ o_wr_en thực tế (monitor đếm)
    integer exp_total = 0;         // số lần ghi mà testbench mong đợi
    reg     exp_overrun = 0;
    reg [TILE_W-1:0] tile_ref;     // bản gốc của tile đang test

    // Monitor: đếm mọi lần ghi, kể cả lần ghi "lạ" ngoài dự kiến.
    always @(posedge clk)
        if (!reset && o_wr_en === 1'b1) total_wr = total_wr + 1;

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

    // Chờ n chu kỳ rảnh, kiểm tra DUT không ghi gì.
    task idle;
        input integer n;
        integer i;
        begin
            for (i = 0; i < n; i = i + 1) begin
                @(negedge clk);
                check(o_wr_en, 0, "idle wr_en");
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

    // Gọi khi đang đứng ở một cạnh xuống clock.
    // Trả về cũng ở cạnh xuống, đúng chu kỳ o_done = 1
    // (nên gọi run_tile tiếp ngay = bắn i_start liên tiếp, không có khe hở).
    task run_tile;
        input integer rb, cb, ar, ac, cout, obase;
        input inject;               // 1 = bắn thêm i_start khi DUT đang ghi
        integer r, c, k, n;
        begin
            for (r = 0; r < ARRAY_ROWS; r = r + 1)
                for (c = 0; c < ARRAY_COLS; c = c + 1)
                    tile_ref[(r*ARRAY_COLS + c)*DATA_W +: DATA_W] = $random(seed);
            n = ar * ac;
            exp_total = exp_total + n;

            check(o_ready, 1, "ready before start");
            i_tile = tile_ref;
            i_row_base = rb;  i_col_base = cb;
            i_active_rows = ar; i_active_cols = ac;
            i_out_channels = cout; i_output_base = obase;
            i_start = 1;
            @(negedge clk);        // đã qua cạnh lên nhận start

            // Đổi hết đầu vào thành rác: DUT phải dùng bản đã chốt.
            i_tile = ~tile_ref;
            i_row_base = 16'hFFFF; i_col_base = 16'hFFFF; i_out_channels = 16'hFFFF;
            i_active_rows = {ROW_W{1'b1}}; i_active_cols = {COL_W{1'b1}};
            i_output_base = {ADDR_W{1'b1}};
            i_start = inject;
            if (inject) exp_overrun = 1;

            for (k = 0; k < n; k = k + 1) begin
                r = k / ac;  c = k % ac;       // thứ tự: hàng ngoài, cột trong
                check(o_wr_en, 1, "wr_en");
                check(o_wr_addr, obase + (rb + r)*cout + (cb + c), "wr_addr");
                check(o_wr_data, tile_ref[(r*ARRAY_COLS + c)*DATA_W +: DATA_W], "wr_data");
                check(o_done,  0, "done too early");
                check(o_ready, 0, "ready while busy");
                @(negedge clk);
                i_start = 0;
            end

            check(o_done,  1, "done pulse");
            check(o_ready, 1, "ready after done");
            check(o_wr_en, 0, "wr_en after done");
            check(o_overrun, exp_overrun, "overrun flag");
        end
    endtask

    integer it, ar, ac, co, cb, rb, ob;

    initial begin
        @(negedge clk); @(negedge clk);
        reset = 0;
        idle(2);

        // 1. Tile đầy đủ
        run_tile(0, 0, ARRAY_ROWS, ARRAY_COLS, ARRAY_COLS, 0, 0);   idle(2);

        // 2. Tile thiếu hàng / cột / cả hai
        run_tile(0, 0, 1, ARRAY_COLS, ARRAY_COLS, 0, 0);            idle(1);
        run_tile(0, 0, ARRAY_ROWS, 1, ARRAY_COLS, 0, 0);            idle(1);
        run_tile(0, 0, 1, 1, ARRAY_COLS, 0, 0);                     idle(1);

        // 3. Tile giữa lớp Conv0, có output_base và col_base khác 0
        run_tile(ARRAY_ROWS, ARRAY_COLS, ARRAY_ROWS, ARRAY_COLS, C_OUT, 100, 0);   idle(1);

        // 4. Tile cuối Conv0 (thiếu hàng, thiếu cột): chạm địa chỉ cuối cùng
        run_tile(L_OUT - ROWS_LAST, C_OUT - COLS_LAST, ROWS_LAST, COLS_LAST, C_OUT, 0, 0);
        idle(1);

        // 5. Hai tile liên tiếp, i_start ngay chu kỳ o_done
        run_tile(0, 0,          ARRAY_ROWS, ARRAY_COLS, C_OUT, 0, 0);
        run_tile(0, ARRAY_COLS, ARRAY_ROWS, ARRAY_COLS, C_OUT, 0, 0);
        idle(2);

        // 6. Overrun: i_start khi đang ghi. Tile thứ hai bị bỏ, tile đầu vẫn đúng,
        //    cờ lỗi giữ nguyên đến khi xóa.
        run_tile(0, 0, ARRAY_ROWS, ARRAY_COLS, C_OUT, 0, 1);
        idle(3);
        clear_overrun;
        run_tile(0, 0, ARRAY_ROWS, ARRAY_COLS, C_OUT, 0, 0);        idle(1);

        // 7. Tile rỗng: không ghi gì, vẫn báo done
        run_tile(0, 0, 0, ARRAY_COLS, C_OUT, 0, 0);                 idle(1);
        run_tile(0, 0, ARRAY_ROWS, 0, C_OUT, 0, 0);                 idle(1);

        // 8. Reset giữa lúc đang ghi
        for (it = 0; it < TILE_W/DATA_W; it = it + 1)
            tile_ref[it*DATA_W +: DATA_W] = $random(seed);
        i_tile = tile_ref; i_row_base = 0; i_col_base = 0;
        i_active_rows = ARRAY_ROWS; i_active_cols = ARRAY_COLS;
        i_out_channels = C_OUT; i_output_base = 0;
        i_start = 1;
        @(negedge clk);  i_start = 0;      // chu kỳ ghi 1
        @(negedge clk);                    // chu kỳ ghi 2
        check(o_wr_en, 1, "wr_en before reset");
        exp_total = exp_total + 1;         // chỉ chu kỳ ghi 1 được monitor đếm
        reset = 1;  #1;
        check(o_wr_en, 0, "reset: wr_en");
        check(o_ready, 1, "reset: ready");
        @(negedge clk);  reset = 0;
        idle(2);
        run_tile(0, 0, ARRAY_ROWS, ARRAY_COLS, C_OUT, 0, 0);        idle(1);

        // 9. Ngẫu nhiên
        for (it = 0; it < 300; it = it + 1) begin
            ar = 1 + ({$random(seed)} % ARRAY_ROWS);
            ac = 1 + ({$random(seed)} % ARRAY_COLS);
            co = ac + ({$random(seed)} % 24);
            cb = {$random(seed)} % (co - ac + 1);
            rb = {$random(seed)} % 200;
            ob = {$random(seed)} % 1000;
            run_tile(rb, cb, ar, ac, co, ob, ({$random(seed)} % 8) == 0);
            if (exp_overrun) begin idle(1); clear_overrun; end
            if ({$random(seed)} % 2) idle(1 + ({$random(seed)} % 3));
        end
        idle(3);

        check(total_wr, exp_total, "total writes");
        if (errors == 0)
            $display("PASS tb_fmap_writer %0dx%0d: %0d writes checked", ARRAY_ROWS, ARRAY_COLS, total_wr);
        else
            $display("[FAIL] tb_fmap_writer: %0d mismatches", errors);
        $finish;
    end

    initial begin
        #20000000;
        $display("[FAIL] TIMEOUT");
        $finish;
    end
endmodule