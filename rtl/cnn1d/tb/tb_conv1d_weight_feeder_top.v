`timescale 1ns/1ps

module tb_conv1d_weight_feeder_top;

    // =========================================================
    // PARAMETERS
    // =========================================================

    localparam DATA_W     = 8;
    localparam ADDR_W     = 6;
    localparam CFG_W      = 16;

    localparam ARRAY_COLS = 3;

    localparam DEPTH      = 16;
    localparam K_W        = 16;

    localparam MEM_DEPTH  = 64;

    localparam COL_W =
        (ARRAY_COLS <= 1) ?
        1 : $clog2(ARRAY_COLS + 1);


    // =========================================================
    // CLOCK / RESET
    // =========================================================

    reg i_clk;
    reg i_reset;


    // =========================================================
    // WEIGHT LOAD CONTROL
    // =========================================================

    reg i_load_start;

    reg [ADDR_W-1:0]
        i_base_addr;

    reg [CFG_W-1:0]
        i_k_len;

    reg [COL_W-1:0]
        i_active_cols;


    wire o_load_busy;
    wire o_load_done;


    // =========================================================
    // FEED CONTROL
    // =========================================================

    reg i_feed_start;

    wire o_ready;

    wire o_feed_done;


    // Standalone Weight test:
    //
    // whenever Weight feeder is ready,
    // allow it to advance.
    //
    // Later in full system:
    //
    // global_feed_step =
    //     ifm_ready & wgt_ready;

    wire i_feed_step;

    assign i_feed_step =
        o_ready;


    // =========================================================
    // WEIGHT SRAM INTERFACE
    // =========================================================

    wire o_sram_rd_en;

    wire [ADDR_W-1:0]
        o_sram_rd_addr;

    wire signed [DATA_W-1:0]
        sram_rd_data;

    wire sram_rd_valid;


    // =========================================================
    // PING-PONG BUFFER WRITE SIDE
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

    wire [ARRAY_COLS*DATA_W-1:0]
        o_data;

    wire [ARRAY_COLS-1:0]
        o_valid;

    wire [ARRAY_COLS-1:0]
        o_first;

    wire [ARRAY_COLS-1:0]
        o_last;


    // =========================================================
    // DEBUG
    // =========================================================

    wire [ARRAY_COLS-1:0]
        o_mem_empty;

    wire [2:0]
        o_loader_state;


    // =========================================================
    // SCOREBOARD
    // =========================================================

    integer errors;

    integer expected_len;
    integer active_cols_test;

    integer recv_col0;
    integer recv_col1;
    integer recv_col2;

    integer first_cycle_col0;
    integer first_cycle_col1;
    integer first_cycle_col2;

    integer monitor_cycle;

    reg monitor_enable;


    integer exp_col0 [0:DEPTH-1];
    integer exp_col1 [0:DEPTH-1];
    integer exp_col2 [0:DEPTH-1];

    integer i;


    // =========================================================
    // CLOCK
    // =========================================================

    initial begin

        i_clk = 1'b0;

        forever #5
            i_clk = ~i_clk;

    end


    // =========================================================
    // PING-PONG SRAM
    // =========================================================

    tensor_pingpong_buffer #(
        .DATA_W(DATA_W),
        .DEPTH(MEM_DEPTH),
        .ADDR_W(ADDR_W)
    ) u_weight_mem (
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

    conv1d_weight_feeder_top #(
        .DATA_W(DATA_W),
        .ADDR_W(ADDR_W),
        .CFG_W(CFG_W),

        .ARRAY_COLS(ARRAY_COLS),

        .DEPTH(DEPTH),
        .K_W(K_W),

        .COL_W(COL_W)
    ) dut (
        .i_clk(i_clk),
        .i_reset(i_reset),

        // ---------------------------------------------
        // LOAD CONTROL
        // ---------------------------------------------

        .i_load_start(i_load_start),

        .i_base_addr(i_base_addr),

        .i_k_len(i_k_len),

        .i_active_cols(i_active_cols),

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
        // SA STREAM
        // ---------------------------------------------

        .o_data(o_data),

        .o_valid(o_valid),

        .o_first(o_first),

        .o_last(o_last),

        // ---------------------------------------------
        // DEBUG
        // ---------------------------------------------

        .o_mem_empty(o_mem_empty),

        .o_loader_state(o_loader_state)
    );


    // =========================================================
    // RESET
    // =========================================================

    task reset_dut;

    begin

        i_reset = 1'b1;

        i_load_start = 1'b0;
        i_feed_start = 1'b0;

        i_base_addr   = 0;
        i_k_len       = 0;
        i_active_cols = 0;

        mem_swap = 1'b0;

        mem_wr_en   = 1'b0;
        mem_wr_addr = 0;
        mem_wr_data = 0;

        monitor_enable = 1'b0;

        recv_col0 = 0;
        recv_col1 = 0;
        recv_col2 = 0;

        first_cycle_col0 = -1;
        first_cycle_col1 = -1;
        first_cycle_col2 = -1;

        monitor_cycle = 0;

        expected_len     = 0;
        active_cols_test = 0;


        for(i = 0;
            i < DEPTH;
            i = i + 1) begin

            exp_col0[i] = 0;
            exp_col1[i] = 0;
            exp_col2[i] = 0;

        end


        repeat(4)
            @(posedge i_clk);


        @(negedge i_clk);

        i_reset = 1'b0;

    end
    endtask


    // =========================================================
    // WRITE WEIGHT SRAM
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

        mem_swap = 1'b1;


        @(posedge i_clk);


        @(negedge i_clk);

        mem_swap = 1'b0;

    end
    endtask


    // =========================================================
    // START WEIGHT LOAD
    // =========================================================

    task start_load;

        input integer base_addr;
        input integer k_len;
        input integer active_cols;

    begin

        @(negedge i_clk);

        i_base_addr   = base_addr;
        i_k_len       = k_len;
        i_active_cols = active_cols;

        i_load_start = 1'b1;


        @(posedge i_clk);


        @(negedge i_clk);

        i_load_start = 1'b0;

    end
    endtask


    // =========================================================
    // START WEIGHT FEED
    // =========================================================

    task start_feed;

    begin

        @(negedge i_clk);

        i_feed_start = 1'b1;


        @(posedge i_clk);


        @(negedge i_clk);

        i_feed_start = 1'b0;

    end
    endtask


    // =========================================================
    // CLEAR STREAM MONITOR
    // =========================================================

    task clear_monitor;

        input integer k_len;
        input integer active_cols;

    begin

        expected_len     = k_len;
        active_cols_test = active_cols;

        recv_col0 = 0;
        recv_col1 = 0;
        recv_col2 = 0;

        first_cycle_col0 = -1;
        first_cycle_col1 = -1;
        first_cycle_col2 = -1;

        monitor_cycle = 0;

    end
    endtask


    // =========================================================
    // OUTPUT MONITOR
    //
    // Sample at negedge so feeder outputs have settled
    // after the previous posedge.
    // =========================================================

    always @(negedge i_clk) begin

        if(!i_reset && monitor_enable) begin

            // =================================================
            // COLUMN 0
            // =================================================

            if(o_valid[0]) begin

                if(recv_col0 >= expected_len) begin

                    $display(
                        "[FAIL] COL0 produced extra data=%0d",
                        $signed(
                            o_data[
                                0*DATA_W
                                +: DATA_W
                            ]
                        )
                    );

                    errors = errors + 1;

                end
                else begin

                    if(recv_col0 == 0)
                        first_cycle_col0 =
                            monitor_cycle;


                    // DATA

                    if(
                        $signed(
                            o_data[
                                0*DATA_W
                                +: DATA_W
                            ]
                        )
                        !==
                        exp_col0[recv_col0]
                    ) begin

                        $display(
                            "[FAIL] COL0 k=%0d data=%0d expected=%0d",
                            recv_col0,
                            $signed(
                                o_data[
                                    0*DATA_W
                                    +: DATA_W
                                ]
                            ),
                            exp_col0[recv_col0]
                        );

                        errors = errors + 1;

                    end
                    else begin

                        $display(
                            "[PASS] cycle=%0d COL0 k=%0d data=%0d first=%b last=%b",
                            monitor_cycle,
                            recv_col0,
                            $signed(
                                o_data[
                                    0*DATA_W
                                    +: DATA_W
                                ]
                            ),
                            o_first[0],
                            o_last[0]
                        );

                    end


                    // FIRST

                    if(
                        o_first[0]
                        !==
                        (recv_col0 == 0)
                    ) begin

                        $display(
                            "[FAIL] COL0 k=%0d FIRST mismatch",
                            recv_col0
                        );

                        errors = errors + 1;

                    end


                    // LAST

                    if(
                        o_last[0]
                        !==
                        (recv_col0 ==
                         expected_len-1)
                    ) begin

                        $display(
                            "[FAIL] COL0 k=%0d LAST mismatch",
                            recv_col0
                        );

                        errors = errors + 1;

                    end


                    recv_col0 =
                        recv_col0 + 1;

                end

            end


            // =================================================
            // COLUMN 1
            // =================================================

            if(o_valid[1]) begin

                if(active_cols_test < 2) begin

                    $display(
                        "[FAIL] inactive COL1 produced valid"
                    );

                    errors = errors + 1;

                end
                else if(recv_col1 >= expected_len) begin

                    $display(
                        "[FAIL] COL1 produced extra data"
                    );

                    errors = errors + 1;

                end
                else begin

                    if(recv_col1 == 0)
                        first_cycle_col1 =
                            monitor_cycle;


                    if(
                        $signed(
                            o_data[
                                1*DATA_W
                                +: DATA_W
                            ]
                        )
                        !==
                        exp_col1[recv_col1]
                    ) begin

                        $display(
                            "[FAIL] COL1 k=%0d data=%0d expected=%0d",
                            recv_col1,
                            $signed(
                                o_data[
                                    1*DATA_W
                                    +: DATA_W
                                ]
                            ),
                            exp_col1[recv_col1]
                        );

                        errors = errors + 1;

                    end
                    else begin

                        $display(
                            "[PASS] cycle=%0d COL1 k=%0d data=%0d first=%b last=%b",
                            monitor_cycle,
                            recv_col1,
                            $signed(
                                o_data[
                                    1*DATA_W
                                    +: DATA_W
                                ]
                            ),
                            o_first[1],
                            o_last[1]
                        );

                    end


                    if(
                        o_first[1]
                        !==
                        (recv_col1 == 0)
                    ) begin

                        $display(
                            "[FAIL] COL1 FIRST mismatch"
                        );

                        errors = errors + 1;

                    end


                    if(
                        o_last[1]
                        !==
                        (recv_col1 ==
                         expected_len-1)
                    ) begin

                        $display(
                            "[FAIL] COL1 LAST mismatch"
                        );

                        errors = errors + 1;

                    end


                    recv_col1 =
                        recv_col1 + 1;

                end

            end


            // =================================================
            // COLUMN 2
            // =================================================

            if(o_valid[2]) begin

                if(active_cols_test < 3) begin

                    $display(
                        "[FAIL] INACTIVE COL2 produced valid data=%0d",
                        $signed(
                            o_data[
                                2*DATA_W
                                +: DATA_W
                            ]
                        )
                    );

                    errors = errors + 1;

                end
                else if(recv_col2 >= expected_len) begin

                    $display(
                        "[FAIL] COL2 produced extra data"
                    );

                    errors = errors + 1;

                end
                else begin

                    if(recv_col2 == 0)
                        first_cycle_col2 =
                            monitor_cycle;


                    if(
                        $signed(
                            o_data[
                                2*DATA_W
                                +: DATA_W
                            ]
                        )
                        !==
                        exp_col2[recv_col2]
                    ) begin

                        $display(
                            "[FAIL] COL2 k=%0d data=%0d expected=%0d",
                            recv_col2,
                            $signed(
                                o_data[
                                    2*DATA_W
                                    +: DATA_W
                                ]
                            ),
                            exp_col2[recv_col2]
                        );

                        errors = errors + 1;

                    end
                    else begin

                        $display(
                            "[PASS] cycle=%0d COL2 k=%0d data=%0d first=%b last=%b",
                            monitor_cycle,
                            recv_col2,
                            $signed(
                                o_data[
                                    2*DATA_W
                                    +: DATA_W
                                ]
                            ),
                            o_first[2],
                            o_last[2]
                        );

                    end


                    if(
                        o_first[2]
                        !==
                        (recv_col2 == 0)
                    ) begin

                        $display(
                            "[FAIL] COL2 FIRST mismatch"
                        );

                        errors = errors + 1;

                    end


                    if(
                        o_last[2]
                        !==
                        (recv_col2 ==
                         expected_len-1)
                    ) begin

                        $display(
                            "[FAIL] COL2 LAST mismatch"
                        );

                        errors = errors + 1;

                    end


                    recv_col2 =
                        recv_col2 + 1;

                end

            end


            monitor_cycle =
                monitor_cycle + 1;

        end

    end


    // =========================================================
    // CHECK FINAL STREAM
    // =========================================================

    task check_stream_done;

    begin

        // -------------------------------------------------
        // Active column counts
        // -------------------------------------------------

        if(recv_col0 !== expected_len) begin

            $display(
                "[FAIL] COL0 count=%0d expected=%0d",
                recv_col0,
                expected_len
            );

            errors = errors + 1;

        end


        if(active_cols_test >= 2) begin

            if(recv_col1 !== expected_len) begin

                $display(
                    "[FAIL] COL1 count=%0d expected=%0d",
                    recv_col1,
                    expected_len
                );

                errors = errors + 1;

            end

        end


        if(active_cols_test >= 3) begin

            if(recv_col2 !== expected_len) begin

                $display(
                    "[FAIL] COL2 count=%0d expected=%0d",
                    recv_col2,
                    expected_len
                );

                errors = errors + 1;

            end

        end
        else begin

            if(recv_col2 !== 0) begin

                $display(
                    "[FAIL] inactive COL2 count=%0d expected=0",
                    recv_col2
                );

                errors = errors + 1;

            end
            else begin

                $display(
                    "[PASS] inactive COL2 produced no stream"
                );

            end

        end


        // -------------------------------------------------
        // COLUMN SKEW
        //
        // col0 delay = 0
        // col1 delay = 1
        // col2 delay = 2
        // -------------------------------------------------

        if(active_cols_test >= 2) begin

            if(
                first_cycle_col1
                !==
                first_cycle_col0 + 1
            ) begin

                $display(
                    "[FAIL] COL1 skew wrong: col0=%0d col1=%0d",
                    first_cycle_col0,
                    first_cycle_col1
                );

                errors = errors + 1;

            end
            else begin

                $display(
                    "[PASS] COL1 skew = +1 cycle"
                );

            end

        end


        if(active_cols_test >= 3) begin

            if(
                first_cycle_col2
                !==
                first_cycle_col0 + 2
            ) begin

                $display(
                    "[FAIL] COL2 skew wrong: col0=%0d col2=%0d",
                    first_cycle_col0,
                    first_cycle_col2
                );

                errors = errors + 1;

            end
            else begin

                $display(
                    "[PASS] COL2 skew = +2 cycles"
                );

            end

        end

    end
    endtask


    // =========================================================
    // TEST 1
    //
    // K = 3
    // active_cols = 3
    //
    // W0 = [ 1, 0,-1]
    // W1 = [ 1, 1, 1]
    // W2 = [-1, 2,-1]
    //
    // SRAM:
    //
    // addr0..2 -> col0
    // addr3..5 -> col1
    // addr6..8 -> col2
    // =========================================================

    task test_active_cols_3;

    begin

        $display("");
        $display("======================================");
        $display(" TC1 : WEIGHT FULL PATH");
        $display(" K=3 ACTIVE_COLS=3");
        $display("======================================");


        reset_dut();


        // -------------------------------------------------
        // Weight tensor / packed accelerator layout
        // -------------------------------------------------

        write_mem(0,  1);
        write_mem(1,  0);
        write_mem(2, -1);

        write_mem(3,  1);
        write_mem(4,  1);
        write_mem(5,  1);

        write_mem(6, -1);
        write_mem(7,  2);
        write_mem(8, -1);


        swap_mem();


        // -------------------------------------------------
        // Expected stream per SA column
        // -------------------------------------------------

        exp_col0[0] =  1;
        exp_col0[1] =  0;
        exp_col0[2] = -1;

        exp_col1[0] =  1;
        exp_col1[1] =  1;
        exp_col1[2] =  1;

        exp_col2[0] = -1;
        exp_col2[1] =  2;
        exp_col2[2] = -1;


        clear_monitor(
            3,
            3
        );


        // -------------------------------------------------
        // SRAM -> Weight FIFOs
        // -------------------------------------------------

        start_load(
            0,
            3,
            3
        );


        wait(o_load_done == 1'b1);

        #1;


        // All three active FIFOs should contain data.

        if(o_mem_empty !== 3'b000) begin

            $display(
                "[FAIL] FIFO status after load = %b expected 000",
                o_mem_empty
            );

            errors = errors + 1;

        end
        else begin

            $display(
                "[PASS] all Weight FIFOs loaded"
            );

        end


        // -------------------------------------------------
        // Feed
        // -------------------------------------------------

        monitor_enable = 1'b1;

        start_feed();


        wait(
            recv_col0 == 3 &&
            recv_col1 == 3 &&
            recv_col2 == 3
        );


        // Need to wait enough for any final delayed output.
        repeat(2)
            @(negedge i_clk);


        monitor_enable = 1'b0;


        check_stream_done();


        $display(
            "TC1 first cycles: col0=%0d col1=%0d col2=%0d",
            first_cycle_col0,
            first_cycle_col1,
            first_cycle_col2
        );

    end
    endtask


    // =========================================================
    // TEST 2
    //
    // active_cols = 2
    //
    // base = 4
    //
    // col0 = 11 12 13
    // col1 = 21 22 23
    //
    // col2 SRAM contains garbage:
    // 99 98 97
    //
    // Loader must NOT write it.
    // Feeder must NOT wait for col2.
    // Feeder must NEVER assert valid[2].
    // =========================================================

    task test_active_cols_2;

    begin

        $display("");
        $display("======================================");
        $display(" TC2 : PARTIAL OUTPUT TILE");
        $display(" K=3 ACTIVE_COLS=2");
        $display("======================================");


        reset_dut();


        // -------------------------------------------------
        // Garbage before base
        // -------------------------------------------------

        write_mem(0, -51);
        write_mem(1, -52);
        write_mem(2, -53);
        write_mem(3, -54);


        // -------------------------------------------------
        // Active col0
        // -------------------------------------------------

        write_mem(4, 11);
        write_mem(5, 12);
        write_mem(6, 13);


        // -------------------------------------------------
        // Active col1
        // -------------------------------------------------

        write_mem(7, 21);
        write_mem(8, 22);
        write_mem(9, 23);


        // -------------------------------------------------
        // Inactive col2 garbage
        // -------------------------------------------------

        write_mem(10, 99);
        write_mem(11, 98);
        write_mem(12, 97);


        swap_mem();


        // -------------------------------------------------
        // Expected active streams
        // -------------------------------------------------

        exp_col0[0] = 11;
        exp_col0[1] = 12;
        exp_col0[2] = 13;

        exp_col1[0] = 21;
        exp_col1[1] = 22;
        exp_col1[2] = 23;


        clear_monitor(
            3,
            2
        );


        // -------------------------------------------------
        // LOAD
        // -------------------------------------------------

        start_load(
            4,
            3,
            2
        );


        wait(o_load_done == 1'b1);

        #1;


        // Expected:
        //
        // col0 nonempty
        // col1 nonempty
        // col2 empty
        //
        // Bit order:
        //
        // [2] col2
        // [1] col1
        // [0] col0

        if(o_mem_empty !== 3'b100) begin

            $display(
                "[FAIL] partial FIFO status=%b expected=100",
                o_mem_empty
            );

            errors = errors + 1;

        end
        else begin

            $display(
                "[PASS] active col0/1 loaded, inactive col2 empty"
            );

        end


        // -------------------------------------------------
        // FEED
        // -------------------------------------------------

        monitor_enable = 1'b1;

        start_feed();


        wait(
            recv_col0 == 3 &&
            recv_col1 == 3
        );


        // Keep monitoring after active streams end so an
        // accidental delayed col2 valid is also detected.

        repeat(4)
            @(negedge i_clk);


        monitor_enable = 1'b0;


        check_stream_done();


        $display(
            "TC2 first cycles: col0=%0d col1=%0d",
            first_cycle_col0,
            first_cycle_col1
        );

    end
    endtask


    // =========================================================
    // DEBUG MONITOR
    // =========================================================

    always @(posedge i_clk) begin

        if(!i_reset) begin

            $display(
                "T=%0t load_state=%0d load_busy=%b load_done=%b | mem_empty=%b | ready=%b step=%b feed_done=%b | valid=%b data={%0d,%0d,%0d}",
                $time,

                o_loader_state,
                o_load_busy,
                o_load_done,

                o_mem_empty,

                o_ready,
                i_feed_step,
                o_feed_done,

                o_valid,

                $signed(
                    o_data[
                        2*DATA_W
                        +: DATA_W
                    ]
                ),

                $signed(
                    o_data[
                        1*DATA_W
                        +: DATA_W
                    ]
                ),

                $signed(
                    o_data[
                        0*DATA_W
                        +: DATA_W
                    ]
                )
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
        i_active_cols = 0;

        mem_swap = 0;

        mem_wr_en   = 0;
        mem_wr_addr = 0;
        mem_wr_data = 0;

        monitor_enable = 0;


        // =================================================
        // RUN TESTS
        // =================================================

        test_active_cols_3();

        test_active_cols_2();


        // =================================================
        // SUMMARY
        // =================================================

        $display("");
        $display("======================================");
        $display(" CONV1D WEIGHT FEEDER TOP SUMMARY");
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
            "[TIMEOUT] tb_conv1d_weight_feeder_top"
        );

        $finish;

    end


endmodule