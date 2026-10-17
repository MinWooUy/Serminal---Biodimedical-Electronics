`timescale 1ns/1ps

module tb_conv1d_ifmap_feeder_top;

    // =========================================================
    // PARAMETERS
    // =========================================================

    localparam DATA_W     = 8;
    localparam ADDR_W     = 5;
    localparam CFG_W      = 16;

    localparam ARRAY_ROWS = 3;

    localparam MEM_DEPTH  = 32;

    localparam DEPTH      = 16;
    localparam K_W        = 16;


    // =========================================================
    // CLOCK / RESET
    // =========================================================

    reg i_clk;
    reg i_reset;


    // =========================================================
    // CONV1D IFMAP TOP CONTROL
    // =========================================================

    reg i_load_start;

    reg [ADDR_W-1:0] i_base_addr;
    reg [CFG_W-1:0]  i_k_len;
    reg [ADDR_W-1:0] i_window_step;

    wire o_load_busy;
    wire o_load_done;


    reg i_feed_start;

    wire o_ready;
    wire o_feed_done;


    // =========================================================
    // FEED STEP
    //
    // Standalone IFMAP test:
    //
    // if feeder is ready -> allow one global feed step
    //
    // Later:
    //
    // global_feed_step = ifm_ready & wgt_ready
    // =========================================================

    wire i_feed_step;

    assign i_feed_step = o_ready;


    // =========================================================
    // SRAM INTERFACE
    // =========================================================

    wire o_sram_rd_en;

    wire [ADDR_W-1:0]
        o_sram_rd_addr;

    wire signed [DATA_W-1:0]
        sram_rd_data;

    wire sram_rd_valid;


    // =========================================================
    // PING-PONG WRITE SIDE
    // =========================================================

    reg mem_swap;

    reg mem_wr_en;

    reg [ADDR_W-1:0]
        mem_wr_addr;

    reg signed [DATA_W-1:0]
        mem_wr_data;


    wire mem_read_bank;
    wire mem_write_bank;


    // =========================================================
    // OUTPUT TO SA
    // =========================================================

    wire [ARRAY_ROWS*DATA_W-1:0]
        o_data;

    wire [ARRAY_ROWS-1:0]
        o_valid;

    wire [ARRAY_ROWS-1:0]
        o_first;

    wire [ARRAY_ROWS-1:0]
        o_last;


    wire [ARRAY_ROWS-1:0]
        o_mem_empty;

    wire [2:0]
        o_loader_state;


    // =========================================================
    // TEST VARIABLES
    // =========================================================

    integer errors;

    integer expected
        [0:ARRAY_ROWS-1][0:DEPTH-1];

    integer recv_count
        [0:ARRAY_ROWS-1];

    integer first_cycle
        [0:ARRAY_ROWS-1];

    integer monitor_cycle;

    integer expected_len;

    integer monitor_enable;

    integer feed_done_seen;

    integer r;
    integer k;


    // =========================================================
    // MEMORY
    // =========================================================

    tensor_pingpong_buffer #(
        .DATA_W(DATA_W),
        .DEPTH(MEM_DEPTH),
        .ADDR_W(ADDR_W)
    ) u_mem (
        .i_clk(i_clk),
        .i_reset(i_reset),

        .i_swap(mem_swap),

        .i_wr_en(mem_wr_en),
        .i_wr_addr(mem_wr_addr),
        .i_wr_data(mem_wr_data),

        .i_rd_en(o_sram_rd_en),
        .i_rd_addr(o_sram_rd_addr),

        .o_rd_data(sram_rd_data),
        .o_rd_valid(sram_rd_valid),

        .o_read_bank(mem_read_bank),
        .o_write_bank(mem_write_bank)
    );


    // =========================================================
    // DUT
    // =========================================================

    conv1d_ifmap_feeder_top #(
        .DATA_W(DATA_W),
        .ADDR_W(ADDR_W),
        .CFG_W(CFG_W),

        .ARRAY_ROWS(ARRAY_ROWS),

        .DEPTH(DEPTH),
        .K_W(K_W)
    ) dut (
        .i_clk(i_clk),
        .i_reset(i_reset),

        // ---------------------------------------------
        // LOAD
        // ---------------------------------------------

        .i_load_start(i_load_start),

        .i_base_addr(i_base_addr),
        .i_k_len(i_k_len),
        .i_window_step(i_window_step),

        .o_load_busy(o_load_busy),
        .o_load_done(o_load_done),

        // ---------------------------------------------
        // SRAM
        // ---------------------------------------------

        .o_sram_rd_en(o_sram_rd_en),
        .o_sram_rd_addr(o_sram_rd_addr),

        .i_sram_rd_data(sram_rd_data),
        .i_sram_rd_valid(sram_rd_valid),

        // ---------------------------------------------
        // FEED
        // ---------------------------------------------

        .i_feed_start(i_feed_start),
        .i_feed_step(i_feed_step),

        .o_ready(o_ready),
        .o_feed_done(o_feed_done),

        // ---------------------------------------------
        // OUTPUT
        // ---------------------------------------------

        .o_data(o_data),

        .o_valid(o_valid),
        .o_first(o_first),
        .o_last(o_last),

        .o_mem_empty(o_mem_empty),

        .o_loader_state(o_loader_state)
    );


    // =========================================================
    // CLOCK
    // =========================================================

    initial begin

        i_clk = 1'b0;

        forever #5
            i_clk = ~i_clk;

    end


    // =========================================================
    // RESET
    // =========================================================

    task reset_dut;

        integer rr;
        integer kk;

    begin

        i_reset = 1'b1;

        i_load_start = 1'b0;
        i_feed_start = 1'b0;

        i_base_addr   = 0;
        i_k_len       = 0;
        i_window_step = 0;

        mem_swap = 1'b0;

        mem_wr_en   = 1'b0;
        mem_wr_addr = 0;
        mem_wr_data = 0;

        monitor_enable = 0;

        feed_done_seen = 0;

        monitor_cycle = 0;

        expected_len = 0;


        for(rr = 0;
            rr < ARRAY_ROWS;
            rr = rr + 1) begin

            recv_count[rr] = 0;
            first_cycle[rr] = -1;

            for(kk = 0;
                kk < DEPTH;
                kk = kk + 1) begin

                expected[rr][kk] = 0;

            end

        end


        repeat(4)
            @(posedge i_clk);


        @(negedge i_clk);

        i_reset = 1'b0;

    end
    endtask


    // =========================================================
    // WRITE ONE WORD INTO CURRENT WRITE BANK
    // =========================================================

    task write_mem;

        input integer addr;
        input integer value;

    begin

        @(negedge i_clk);

        mem_wr_en   = 1'b1;
        mem_wr_addr = addr;
        mem_wr_data = value;


        @(posedge i_clk);


        @(negedge i_clk);

        mem_wr_en = 1'b0;

    end
    endtask


    // =========================================================
    // SWAP PING-PONG BANK
    // =========================================================

    task swap_mem;

    begin

        @(negedge i_clk);

        mem_wr_en = 1'b0;
        mem_swap  = 1'b1;


        @(posedge i_clk);


        @(negedge i_clk);

        mem_swap = 1'b0;

    end
    endtask


    // =========================================================
    // START WINDOW LOADER
    // =========================================================

    task start_load;

        input integer base_addr;
        input integer k_len;
        input integer step;

    begin

        @(negedge i_clk);

        i_base_addr   = base_addr;
        i_k_len       = k_len;
        i_window_step = step;

        i_load_start = 1'b1;


        @(posedge i_clk);


        @(negedge i_clk);

        i_load_start = 1'b0;

    end
    endtask


    // =========================================================
    // START EXISTING IFMAP FEEDER
    // =========================================================

    task start_feed;

    begin

        feed_done_seen = 0;

        @(negedge i_clk);

        i_feed_start = 1'b1;


        @(posedge i_clk);


        @(negedge i_clk);

        i_feed_start = 1'b0;

    end
    endtask


    // =========================================================
    // RESET OUTPUT MONITOR
    // =========================================================

    task clear_monitor;

        input integer k_len;

        integer rr;
        integer kk;

    begin

        expected_len = k_len;

        monitor_cycle = 0;

        feed_done_seen = 0;


        for(rr = 0;
            rr < ARRAY_ROWS;
            rr = rr + 1) begin

            recv_count[rr]  = 0;
            first_cycle[rr] = -1;

        end

    end
    endtask


    // =========================================================
    // OUTPUT MONITOR
    //
    // Sample at negedge:
    //
    // feeder outputs have already settled after previous
    // positive clock edge.
    // =========================================================

    always @(negedge i_clk) begin

        if(!i_reset) begin

            if(o_feed_done)
                feed_done_seen = 1;


            if(monitor_enable) begin

                for(r = 0;
                    r < ARRAY_ROWS;
                    r = r + 1) begin

                    if(o_valid[r]) begin

                        // ---------------------------------
                        // Protect against extra output
                        // ---------------------------------

                        if(recv_count[r] >=
                           expected_len) begin

                            $display(
                                "[FAIL] row%0d produced extra data=%0d",
                                r,
                                $signed(
                                    o_data[
                                        r*DATA_W
                                        +: DATA_W
                                    ]
                                )
                            );

                            errors = errors + 1;

                        end

                        else begin

                            // -----------------------------
                            // First valid cycle
                            // -----------------------------

                            if(recv_count[r] == 0)
                                first_cycle[r] =
                                    monitor_cycle;


                            // -----------------------------
                            // DATA CHECK
                            // -----------------------------

                            if(
                                $signed(
                                    o_data[
                                        r*DATA_W
                                        +: DATA_W
                                    ]
                                )
                                !==
                                expected[r][recv_count[r]]
                            ) begin

                                $display(
                                    "[FAIL] row%0d k=%0d data=%0d expected=%0d",
                                    r,
                                    recv_count[r],
                                    $signed(
                                        o_data[
                                            r*DATA_W
                                            +: DATA_W
                                        ]
                                    ),
                                    expected[r][recv_count[r]]
                                );

                                errors = errors + 1;

                            end
                            else begin

                                $display(
                                    "[PASS] cycle=%0d row%0d k=%0d data=%0d first=%b last=%b",
                                    monitor_cycle,
                                    r,
                                    recv_count[r],
                                    $signed(
                                        o_data[
                                            r*DATA_W
                                            +: DATA_W
                                        ]
                                    ),
                                    o_first[r],
                                    o_last[r]
                                );

                            end


                            // -----------------------------
                            // FIRST CHECK
                            // -----------------------------

                            if(
                                o_first[r]
                                !==
                                (recv_count[r] == 0)
                            ) begin

                                $display(
                                    "[FAIL] row%0d k=%0d FIRST mismatch",
                                    r,
                                    recv_count[r]
                                );

                                errors = errors + 1;

                            end


                            // -----------------------------
                            // LAST CHECK
                            // -----------------------------

                            if(
                                o_last[r]
                                !==
                                (recv_count[r]
                                 == expected_len-1)
                            ) begin

                                $display(
                                    "[FAIL] row%0d k=%0d LAST mismatch",
                                    r,
                                    recv_count[r]
                                );

                                errors = errors + 1;

                            end


                            recv_count[r] =
                                recv_count[r] + 1;

                        end

                    end

                end


                monitor_cycle =
                    monitor_cycle + 1;

            end

        end

    end


    // =========================================================
    // CHECK FINAL STREAM
    // =========================================================

    task check_stream_done;

        integer rr;

    begin

        // -------------------------------------------------
        // Every row must produce exactly K values
        // -------------------------------------------------

        for(rr = 0;
            rr < ARRAY_ROWS;
            rr = rr + 1) begin

            if(recv_count[rr] !== expected_len) begin

                $display(
                    "[FAIL] row%0d count=%0d expected=%0d",
                    rr,
                    recv_count[rr],
                    expected_len
                );

                errors = errors + 1;

            end

        end


        // -------------------------------------------------
        // Verify skew:
        //
        // row1 starts 1 cycle after row0
        // row2 starts 2 cycles after row0
        // -------------------------------------------------

        if(first_cycle[1] !==
           first_cycle[0] + 1) begin

            $display(
                "[FAIL] row1 skew wrong: row0=%0d row1=%0d",
                first_cycle[0],
                first_cycle[1]
            );

            errors = errors + 1;

        end
        else begin

            $display(
                "[PASS] row1 skew = +1 cycle"
            );

        end


        if(first_cycle[2] !==
           first_cycle[0] + 2) begin

            $display(
                "[FAIL] row2 skew wrong: row0=%0d row2=%0d",
                first_cycle[0],
                first_cycle[2]
            );

            errors = errors + 1;

        end
        else begin

            $display(
                "[PASS] row2 skew = +2 cycles"
            );

        end

    end
    endtask


    // =========================================================
    // TC1
    //
    // Cin=1
    // Kw=3
    // stride=1
    //
    // X = [1 2 3 4 5 6 7 8]
    //
    // Loader generates:
    //
    // row0 FIFO = 1 2 3
    // row1 FIFO = 2 3 4
    // row2 FIFO = 3 4 5
    //
    // Feeder should output:
    //
    // cycle    row0    row1    row2
    //
    //           1
    //           2       2
    //           3       3       3
    //                   4       4
    //                           5
    // =========================================================

    task test_cin1;

    begin

        $display("");
        $display("======================================");
        $display(" TC1 : IFMAP FULL PATH Cin=1");
        $display("======================================");


        reset_dut();


        // -------------------------------------------------
        // Load input tensor into WRITE bank
        // -------------------------------------------------

        write_mem(0,1);
        write_mem(1,2);
        write_mem(2,3);
        write_mem(3,4);

        write_mem(4,5);
        write_mem(5,6);
        write_mem(6,7);
        write_mem(7,8);


        // -------------------------------------------------
        // Make loaded bank the READ bank
        // -------------------------------------------------

        swap_mem();


        // -------------------------------------------------
        // Expected skewed streams
        // -------------------------------------------------

        expected[0][0] = 1;
        expected[0][1] = 2;
        expected[0][2] = 3;

        expected[1][0] = 2;
        expected[1][1] = 3;
        expected[1][2] = 4;

        expected[2][0] = 3;
        expected[2][1] = 4;
        expected[2][2] = 5;


        clear_monitor(3);


        // -------------------------------------------------
        // Sliding-window loader
        // -------------------------------------------------

        start_load(
            0,      // base
            3,      // K = Kw * Cin
            1       // stride * Cin
        );


        wait(o_load_done == 1'b1);

        #1;


        // Every feeder FIFO must contain data now
        if(o_mem_empty !==
           {ARRAY_ROWS{1'b0}}) begin

            $display(
                "[FAIL] FIFO empty after load: %b",
                o_mem_empty
            );

            errors = errors + 1;

        end
        else begin

            $display(
                "[PASS] all IFMAP FIFOs loaded"
            );

        end


        // -------------------------------------------------
        // Feed to SA
        // -------------------------------------------------

        monitor_enable = 1;

        start_feed();


        wait(
            recv_count[0] == 3 &&
            recv_count[1] == 3 &&
            recv_count[2] == 3
        );


        // Allow any final metadata/state transition
        repeat(2)
            @(negedge i_clk);


        monitor_enable = 0;


        check_stream_done();


        $display(
            "TC1 first cycles = %0d %0d %0d",
            first_cycle[0],
            first_cycle[1],
            first_cycle[2]
        );

    end
    endtask


    // =========================================================
    // TC2
    //
    // Cin=2
    // Kw=3
    // stride=1
    //
    // Tensor:
    //
    // t0 = [1  10]
    // t1 = [2  20]
    // t2 = [3  30]
    // t3 = [4  40]
    // t4 = [5  50]
    //
    // Memory:
    //
    // 1 10 2 20 3 30 4 40 5 50
    //
    // K = 6
    //
    // Expected:
    //
    // row0 = 1 10 2 20 3 30
    // row1 = 2 20 3 30 4 40
    // row2 = 3 30 4 40 5 50
    // =========================================================

    task test_cin2;

    begin

        $display("");
        $display("======================================");
        $display(" TC2 : IFMAP FULL PATH Cin=2");
        $display("======================================");


        reset_dut();


        // -------------------------------------------------
        // Input tensor
        // -------------------------------------------------

        write_mem(0,1);
        write_mem(1,10);

        write_mem(2,2);
        write_mem(3,20);

        write_mem(4,3);
        write_mem(5,30);

        write_mem(6,4);
        write_mem(7,40);

        write_mem(8,5);
        write_mem(9,50);


        swap_mem();


        // -------------------------------------------------
        // Expected row 0
        // -------------------------------------------------

        expected[0][0] = 1;
        expected[0][1] = 10;
        expected[0][2] = 2;
        expected[0][3] = 20;
        expected[0][4] = 3;
        expected[0][5] = 30;


        // -------------------------------------------------
        // Expected row 1
        // -------------------------------------------------

        expected[1][0] = 2;
        expected[1][1] = 20;
        expected[1][2] = 3;
        expected[1][3] = 30;
        expected[1][4] = 4;
        expected[1][5] = 40;


        // -------------------------------------------------
        // Expected row 2
        // -------------------------------------------------

        expected[2][0] = 3;
        expected[2][1] = 30;
        expected[2][2] = 4;
        expected[2][3] = 40;
        expected[2][4] = 5;
        expected[2][5] = 50;


        clear_monitor(6);


        // -------------------------------------------------
        // Load windows into FIFOs
        // -------------------------------------------------

        start_load(
            0,
            6,      // Kw*Cin = 3*2
            2       // stride*Cin = 1*2
        );


        wait(o_load_done == 1'b1);

        #1;


        if(o_mem_empty !==
           {ARRAY_ROWS{1'b0}}) begin

            $display(
                "[FAIL] FIFO empty after Cin=2 load: %b",
                o_mem_empty
            );

            errors = errors + 1;

        end
        else begin

            $display(
                "[PASS] Cin=2 windows loaded"
            );

        end


        // -------------------------------------------------
        // Start skewed feed
        // -------------------------------------------------

        monitor_enable = 1;

        start_feed();


        wait(
            recv_count[0] == 6 &&
            recv_count[1] == 6 &&
            recv_count[2] == 6
        );


        repeat(2)
            @(negedge i_clk);


        monitor_enable = 0;


        check_stream_done();


        $display(
            "TC2 first cycles = %0d %0d %0d",
            first_cycle[0],
            first_cycle[1],
            first_cycle[2]
        );

    end
    endtask


    // =========================================================
    // DEBUG MONITOR
    // =========================================================

    always @(posedge i_clk) begin

        if(!i_reset) begin

            $display(
                "T=%0t load_state=%0d load_busy=%b load_done=%b | mem_empty=%b | feed_ready=%b feed_step=%b feed_done=%b | valid=%b",
                $time,

                o_loader_state,
                o_load_busy,
                o_load_done,

                o_mem_empty,

                o_ready,
                i_feed_step,
                o_feed_done,

                o_valid
            );

        end

    end


    // =========================================================
    // MAIN
    // =========================================================

    initial begin

        errors = 0;

        i_reset = 0;

        i_load_start = 0;
        i_feed_start = 0;

        i_base_addr   = 0;
        i_k_len       = 0;
        i_window_step = 0;

        mem_swap = 0;

        mem_wr_en   = 0;
        mem_wr_addr = 0;
        mem_wr_data = 0;

        monitor_enable = 0;


        // -------------------------------------------------
        // RUN
        // -------------------------------------------------

        test_cin1();

        test_cin2();


        // -------------------------------------------------
        // SUMMARY
        // -------------------------------------------------

        $display("");
        $display("======================================");
        $display(" CONV1D IFMAP FULL-PATH SUMMARY");
        $display("======================================");

        if(errors == 0) begin

            $display(
                " RESULT : ALL TESTS PASSED"
            );

        end
        else begin

            $display(
                " RESULT : FAILED (%0d errors)",
                errors
            );

        end

        $display("======================================");


        repeat(5)
            @(posedge i_clk);

        $finish;

    end


    // =========================================================
    // GLOBAL TIMEOUT
    // =========================================================

    initial begin

        #200000;

        $display("");
        $display(
            "[TIMEOUT] tb_conv1d_ifmap_feeder_top"
        );

        $finish;

    end


endmodule