`timescale 1ns/1ps

module tb_conv1d_weight_loader;

    // =========================================================
    // PARAMETERS
    // =========================================================

    localparam DATA_W     = 8;
    localparam ADDR_W     = 6;
    localparam CFG_W      = 16;

    localparam ARRAY_COLS = 3;

    localparam COL_W =
        (ARRAY_COLS <= 1) ?
        1 : $clog2(ARRAY_COLS + 1);

    localparam MEM_DEPTH = 64;
    localparam CAP_DEPTH = 16;


    // =========================================================
    // CLOCK / RESET
    // =========================================================

    reg i_clk;
    reg i_reset;


    // =========================================================
    // WEIGHT LOADER CONTROL
    // =========================================================

    reg i_start;

    reg [ADDR_W-1:0]
        i_base_addr;

    reg [CFG_W-1:0]
        i_k_len;

    reg [COL_W-1:0]
        i_active_cols;


    // =========================================================
    // SRAM READ INTERFACE
    // =========================================================

    wire o_sram_rd_en;

    wire [ADDR_W-1:0]
        o_sram_rd_addr;

    wire signed [DATA_W-1:0]
        sram_rd_data;

    wire sram_rd_valid;


    // =========================================================
    // WEIGHT LOADER OUTPUT
    // =========================================================

    wire [ARRAY_COLS-1:0]
        o_wgt_wr_en;

    wire signed
        [ARRAY_COLS*DATA_W-1:0]
        o_wgt_wr_data;


    // =========================================================
    // STATUS
    // =========================================================

    wire o_busy;
    wire o_done;

    wire [2:0]
        o_state_dbg;


    // =========================================================
    // PING-PONG MEMORY WRITE SIDE
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
    // CAPTURED WEIGHT FIFO CONTENTS
    //
    // These memories emulate:
    //
    // Weight FIFO col0
    // Weight FIFO col1
    // Weight FIFO col2
    // =========================================================

    reg signed [DATA_W-1:0]
        cap_col0 [0:CAP_DEPTH-1];

    reg signed [DATA_W-1:0]
        cap_col1 [0:CAP_DEPTH-1];

    reg signed [DATA_W-1:0]
        cap_col2 [0:CAP_DEPTH-1];


    integer cnt_col0;
    integer cnt_col1;
    integer cnt_col2;

    integer errors;

    integer i;


    // =========================================================
    // OPTIONAL ADDRESS RANGE CHECK
    // =========================================================

    reg check_addr_enable;

    integer expected_addr_min;
    integer expected_addr_max;


    // =========================================================
    // CLOCK
    // =========================================================

    initial begin

        i_clk = 1'b0;

        forever #5
            i_clk = ~i_clk;

    end


    // =========================================================
    // PING-PONG MEMORY
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

    conv1d_weight_loader #(
        .DATA_W(DATA_W),
        .ADDR_W(ADDR_W),
        .CFG_W(CFG_W),

        .ARRAY_COLS(ARRAY_COLS),
        .COL_W(COL_W)
    ) dut (
        .i_clk(i_clk),
        .i_reset(i_reset),

        .i_start(i_start),

        .i_base_addr(i_base_addr),
        .i_k_len(i_k_len),

        .i_active_cols(i_active_cols),

        // Không có FIFO thật phía sau: lane luôn còn chỗ
        .i_lane_full({ARRAY_COLS{1'b0}}),

        // SRAM
        .o_sram_rd_en(o_sram_rd_en),
        .o_sram_rd_addr(o_sram_rd_addr),

        .i_sram_rd_data(sram_rd_data),
        .i_sram_rd_valid(sram_rd_valid),

        // Weight FIFO write
        .o_wgt_wr_en(o_wgt_wr_en),
        .o_wgt_wr_data(o_wgt_wr_data),

        // Status
        .o_busy(o_busy),
        .o_done(o_done),

        .o_state_dbg(o_state_dbg)
    );


    // =========================================================
    // RESET
    // =========================================================

    task reset_dut;

    begin

        i_reset = 1'b1;

        i_start = 1'b0;

        i_base_addr   = 0;
        i_k_len       = 0;
        i_active_cols = 0;

        mem_swap = 1'b0;

        mem_wr_en   = 1'b0;
        mem_wr_addr = 0;
        mem_wr_data = 0;

        cnt_col0 = 0;
        cnt_col1 = 0;
        cnt_col2 = 0;

        check_addr_enable = 1'b0;

        expected_addr_min = 0;
        expected_addr_max = 0;


        for(i = 0;
            i < CAP_DEPTH;
            i = i + 1) begin

            cap_col0[i] = 0;
            cap_col1[i] = 0;
            cap_col2[i] = 0;

        end


        repeat(4)
            @(posedge i_clk);


        @(negedge i_clk);

        i_reset = 1'b0;

    end
    endtask


    // =========================================================
    // WRITE ONE WORD INTO PING-PONG WRITE BANK
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
    // SWAP MEMORY BANK
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
    // START WEIGHT LOADER
    // =========================================================

    task start_loader;

        input integer base_addr;
        input integer k_len;
        input integer active_cols;

    begin

        @(negedge i_clk);

        i_base_addr   = base_addr;
        i_k_len       = k_len;
        i_active_cols = active_cols;

        i_start = 1'b1;


        @(posedge i_clk);


        @(negedge i_clk);

        i_start = 1'b0;

    end
    endtask


    // =========================================================
    // CAPTURE WRITES
    //
    // This models the WRITE side of the existing
    // weight_feeder FIFOs.
    //
    // Sampling at posedge matches FIFO write semantics.
    // =========================================================

    always @(posedge i_clk) begin

        if(!i_reset) begin

            // -------------------------------------------------
            // Check illegal multi-column write
            // Loader should write exactly one column at a time.
            // -------------------------------------------------

            if(
                (o_wgt_wr_en != 3'b000) &&
                (o_wgt_wr_en != 3'b001) &&
                (o_wgt_wr_en != 3'b010) &&
                (o_wgt_wr_en != 3'b100)
            ) begin

                $display(
                    "[FAIL] Multiple weight columns written simultaneously: wr_en=%b",
                    o_wgt_wr_en
                );

                errors = errors + 1;

            end


            // -------------------------------------------------
            // COLUMN 0
            // -------------------------------------------------

            if(o_wgt_wr_en[0]) begin

                cap_col0[cnt_col0] =
                    $signed(
                        o_wgt_wr_data[
                            0*DATA_W
                            +: DATA_W
                        ]
                    );


                $display(
                    "[CAPTURE] COL0 k=%0d data=%0d",
                    cnt_col0,
                    $signed(
                        o_wgt_wr_data[
                            0*DATA_W
                            +: DATA_W
                        ]
                    )
                );


                cnt_col0 = cnt_col0 + 1;

            end


            // -------------------------------------------------
            // COLUMN 1
            // -------------------------------------------------

            if(o_wgt_wr_en[1]) begin

                cap_col1[cnt_col1] =
                    $signed(
                        o_wgt_wr_data[
                            1*DATA_W
                            +: DATA_W
                        ]
                    );


                $display(
                    "[CAPTURE] COL1 k=%0d data=%0d",
                    cnt_col1,
                    $signed(
                        o_wgt_wr_data[
                            1*DATA_W
                            +: DATA_W
                        ]
                    )
                );


                cnt_col1 = cnt_col1 + 1;

            end


            // -------------------------------------------------
            // COLUMN 2
            // -------------------------------------------------

            if(o_wgt_wr_en[2]) begin

                cap_col2[cnt_col2] =
                    $signed(
                        o_wgt_wr_data[
                            2*DATA_W
                            +: DATA_W
                        ]
                    );


                $display(
                    "[CAPTURE] COL2 k=%0d data=%0d",
                    cnt_col2,
                    $signed(
                        o_wgt_wr_data[
                            2*DATA_W
                            +: DATA_W
                        ]
                    )
                );


                cnt_col2 = cnt_col2 + 1;

            end

        end

    end


    // =========================================================
    // CHECK SRAM ADDRESS RANGE
    // =========================================================

    always @(posedge i_clk) begin

        if(!i_reset &&
           check_addr_enable &&
           o_sram_rd_en) begin

            if(
                (o_sram_rd_addr < expected_addr_min) ||
                (o_sram_rd_addr > expected_addr_max)
            ) begin

                $display(
                    "[FAIL] SRAM address out of expected range: addr=%0d range=[%0d..%0d]",
                    o_sram_rd_addr,
                    expected_addr_min,
                    expected_addr_max
                );

                errors = errors + 1;

            end

        end

    end


    // =========================================================
    // CHECK VALUE
    // =========================================================

    task check_value;

        input integer col;
        input integer idx;
        input integer expected;

        integer actual;

    begin

        actual = 0;


        case(col)

            0:
                actual = $signed(cap_col0[idx]);

            1:
                actual = $signed(cap_col1[idx]);

            2:
                actual = $signed(cap_col2[idx]);

        endcase


        if(actual !== expected) begin

            $display(
                "[FAIL] col%0d[%0d] got=%0d expected=%0d",
                col,
                idx,
                actual,
                expected
            );

            errors = errors + 1;

        end
        else begin

            $display(
                "[PASS] col%0d[%0d] = %0d",
                col,
                idx,
                actual
            );

        end

    end
    endtask


    // =========================================================
    // TC1
    //
    // K = 3
    // active_cols = 3
    //
    // W0 = [ 1, 0,-1]
    // W1 = [ 1, 1, 1]
    // W2 = [-1, 2,-1]
    //
    // SRAM layout:
    //
    // addr0 =  1
    // addr1 =  0
    // addr2 = -1
    //
    // addr3 =  1
    // addr4 =  1
    // addr5 =  1
    //
    // addr6 = -1
    // addr7 =  2
    // addr8 = -1
    // =========================================================

    task test_active_cols_3;

    begin

        $display("");
        $display("======================================");
        $display(" TC1 : K=3 ACTIVE_COLS=3");
        $display("======================================");


        reset_dut();


        // -------------------------------------------------
        // Load Weight SRAM
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


        // Make loaded bank readable
        swap_mem();


        // -------------------------------------------------
        // Expected SRAM access range
        // -------------------------------------------------

        expected_addr_min = 0;
        expected_addr_max = 8;

        check_addr_enable = 1'b1;


        // -------------------------------------------------
        // Start loader
        // -------------------------------------------------

        start_loader(
            0,      // base_addr
            3,      // K
            3       // active_cols
        );


        wait(o_done == 1'b1);

        #1;

        check_addr_enable = 1'b0;


        // -------------------------------------------------
        // Count checks
        // -------------------------------------------------

        if(cnt_col0 !== 3) begin

            $display(
                "[FAIL] COL0 count=%0d expected=3",
                cnt_col0
            );

            errors = errors + 1;

        end


        if(cnt_col1 !== 3) begin

            $display(
                "[FAIL] COL1 count=%0d expected=3",
                cnt_col1
            );

            errors = errors + 1;

        end


        if(cnt_col2 !== 3) begin

            $display(
                "[FAIL] COL2 count=%0d expected=3",
                cnt_col2
            );

            errors = errors + 1;

        end


        // -------------------------------------------------
        // Data checks
        // -------------------------------------------------

        check_value(0,0, 1);
        check_value(0,1, 0);
        check_value(0,2,-1);

        check_value(1,0, 1);
        check_value(1,1, 1);
        check_value(1,2, 1);

        check_value(2,0,-1);
        check_value(2,1, 2);
        check_value(2,2,-1);

    end
    endtask


    // =========================================================
    // TC2
    //
    // K = 3
    // active_cols = 2
    //
    // base_addr = 4
    //
    // SRAM:
    //
    // addr4  = 11
    // addr5  = 12
    // addr6  = 13
    //
    // addr7  = 21
    // addr8  = 22
    // addr9  = 23
    //
    // addr10 = 99
    // addr11 = 98
    // addr12 = 97
    //
    // active_cols=2 means:
    //
    // col0 = 11 12 13
    // col1 = 21 22 23
    // col2 MUST NOT be written.
    //
    // AGU must only read addr4..addr9.
    // =========================================================

    task test_active_cols_2;

    begin

        $display("");
        $display("======================================");
        $display(" TC2 : K=3 ACTIVE_COLS=2");
        $display("======================================");


        reset_dut();


        // -------------------------------------------------
        // Some garbage before base
        // -------------------------------------------------

        write_mem(0, -50);
        write_mem(1, -51);
        write_mem(2, -52);
        write_mem(3, -53);


        // -------------------------------------------------
        // Active column 0
        // -------------------------------------------------

        write_mem(4, 11);
        write_mem(5, 12);
        write_mem(6, 13);


        // -------------------------------------------------
        // Active column 1
        // -------------------------------------------------

        write_mem(7, 21);
        write_mem(8, 22);
        write_mem(9, 23);


        // -------------------------------------------------
        // Inactive column 2 garbage
        //
        // Loader MUST NOT consume these.
        // -------------------------------------------------

        write_mem(10, 99);
        write_mem(11, 98);
        write_mem(12, 97);


        swap_mem();


        // Only addresses 4..9 are legal
        expected_addr_min = 4;
        expected_addr_max = 9;

        check_addr_enable = 1'b1;


        start_loader(
            4,      // base_addr
            3,      // K
            2       // active_cols
        );


        wait(o_done == 1'b1);

        #1;

        check_addr_enable = 1'b0;


        // -------------------------------------------------
        // Counts
        // -------------------------------------------------

        if(cnt_col0 !== 3) begin

            $display(
                "[FAIL] COL0 count=%0d expected=3",
                cnt_col0
            );

            errors = errors + 1;

        end
        else begin

            $display(
                "[PASS] COL0 count = 3"
            );

        end


        if(cnt_col1 !== 3) begin

            $display(
                "[FAIL] COL1 count=%0d expected=3",
                cnt_col1
            );

            errors = errors + 1;

        end
        else begin

            $display(
                "[PASS] COL1 count = 3"
            );

        end


        // -------------------------------------------------
        // CRITICAL:
        // inactive col2 must NEVER receive a write
        // -------------------------------------------------

        if(cnt_col2 !== 0) begin

            $display(
                "[FAIL] INACTIVE COL2 was written %0d times",
                cnt_col2
            );

            errors = errors + 1;

        end
        else begin

            $display(
                "[PASS] inactive COL2 untouched"
            );

        end


        // -------------------------------------------------
        // Data
        // -------------------------------------------------

        check_value(0,0,11);
        check_value(0,1,12);
        check_value(0,2,13);

        check_value(1,0,21);
        check_value(1,1,22);
        check_value(1,2,23);

    end
    endtask


    // =========================================================
    // DEBUG
    // =========================================================

    always @(posedge i_clk) begin

        if(!i_reset) begin

            $display(
                "T=%0t state=%0d busy=%b done=%b rd_en=%b addr=%0d rd_valid=%b wr_en=%b",
                $time,

                o_state_dbg,

                o_busy,
                o_done,

                o_sram_rd_en,
                o_sram_rd_addr,
                sram_rd_valid,

                o_wgt_wr_en
            );

        end

    end


    // =========================================================
    // MAIN
    // =========================================================

    initial begin

        errors = 0;

        i_clk   = 0;
        i_reset = 0;

        i_start = 0;

        i_base_addr   = 0;
        i_k_len       = 0;
        i_active_cols = 0;

        mem_swap = 0;

        mem_wr_en   = 0;
        mem_wr_addr = 0;
        mem_wr_data = 0;

        check_addr_enable = 0;


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
        $display(" CONV1D WEIGHT LOADER TEST SUMMARY");
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
    // TIMEOUT
    // =========================================================

    initial begin

        #200000;

        $display("");
        $display(
            "[TIMEOUT] tb_conv1d_weight_loader"
        );

        $finish;

    end


endmodule