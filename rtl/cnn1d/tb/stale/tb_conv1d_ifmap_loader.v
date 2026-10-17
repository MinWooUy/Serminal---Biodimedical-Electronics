`timescale 1ns/1ps

module tb_conv1d_ifmap_loader;

    localparam DATA_W     = 8;
    localparam ADDR_W     = 4;
    localparam CFG_W      = 16;

    localparam ARRAY_ROWS = 3;

    localparam SRAM_DEPTH = 16;


    // =====================================================
    // CLOCK / RESET
    // =====================================================

    reg i_clk;
    reg i_reset;


    // =====================================================
    // LOADER CONTROL
    // =====================================================

    reg i_start;

    reg [ADDR_W-1:0] i_base_addr;
    reg [CFG_W-1:0]  i_k_len;

    reg [ADDR_W-1:0] i_window_step;


    // =====================================================
    // PING-PONG BUFFER WRITE SIDE
    // =====================================================

    reg                      mem_swap;

    reg                      mem_wr_en;
    reg [ADDR_W-1:0]         mem_wr_addr;

    reg signed [DATA_W-1:0]  mem_wr_data;


    // =====================================================
    // SRAM READ CONNECTION
    // =====================================================

    wire                     sram_rd_en;

    wire [ADDR_W-1:0]        sram_rd_addr;

    wire signed [DATA_W-1:0] sram_rd_data;

    wire                     sram_rd_valid;


    // =====================================================
    // IFMAP WRITE STREAM
    // =====================================================

    wire [ARRAY_ROWS-1:0]
        ifm_wr_en;

    wire signed
        [ARRAY_ROWS*DATA_W-1:0]
        ifm_wr_data;


    // =====================================================
    // STATUS
    // =====================================================

    wire loader_busy;
    wire loader_done;

    wire [2:0] loader_state;

    wire mem_read_bank;
    wire mem_write_bank;


    integer errors;


    // =====================================================
    // CAPTURE GENERATED WINDOWS
    //
    // Enough for K up to 16.
    // =====================================================

    integer captured
        [0:ARRAY_ROWS-1][0:15];

    integer row_count
        [0:ARRAY_ROWS-1];


    integer r;
    integer k;


    // =====================================================
    // MEMORY
    // =====================================================

    tensor_pingpong_buffer #(
        .DATA_W(DATA_W),
        .DEPTH(SRAM_DEPTH),
        .ADDR_W(ADDR_W)
    ) u_mem (
        .i_clk(i_clk),
        .i_reset(i_reset),

        .i_swap(mem_swap),

        .i_wr_en(mem_wr_en),
        .i_wr_addr(mem_wr_addr),
        .i_wr_data(mem_wr_data),

        .i_rd_en(sram_rd_en),
        .i_rd_addr(sram_rd_addr),

        .o_rd_data(sram_rd_data),
        .o_rd_valid(sram_rd_valid),

        .o_read_bank(mem_read_bank),
        .o_write_bank(mem_write_bank)
    );


    // =====================================================
    // DUT
    // =====================================================

    conv1d_ifmap_loader #(
        .DATA_W(DATA_W),
        .ADDR_W(ADDR_W),
        .CFG_W(CFG_W),
        .ARRAY_ROWS(ARRAY_ROWS)
    ) dut (
        .i_clk(i_clk),
        .i_reset(i_reset),

        .i_start(i_start),

        .i_base_addr(i_base_addr),
        .i_k_len(i_k_len),
        .i_window_step(i_window_step),

        .o_sram_rd_en(sram_rd_en),
        .o_sram_rd_addr(sram_rd_addr),

        .i_sram_rd_data(sram_rd_data),
        .i_sram_rd_valid(sram_rd_valid),

        .o_ifm_wr_en(ifm_wr_en),
        .o_ifm_wr_data(ifm_wr_data),

        .o_busy(loader_busy),
        .o_done(loader_done),

        .o_state_dbg(loader_state)
    );


    // =====================================================
    // CLOCK
    // =====================================================

    initial begin

        i_clk = 1'b0;

        forever #5
            i_clk = ~i_clk;

    end


    // =====================================================
    // RESET
    // =====================================================

    task reset_dut;

        integer rr;
        integer kk;

    begin

        i_reset = 1'b1;

        i_start = 1'b0;

        i_base_addr   = 0;
        i_k_len       = 0;
        i_window_step = 0;

        mem_swap = 1'b0;

        mem_wr_en   = 1'b0;
        mem_wr_addr = 0;
        mem_wr_data = 0;


        for(rr = 0;
            rr < ARRAY_ROWS;
            rr = rr + 1) begin

            row_count[rr] = 0;

            for(kk = 0;
                kk < 16;
                kk = kk + 1) begin

                captured[rr][kk] = 0;

            end

        end


        repeat(3)
            @(posedge i_clk);


        @(negedge i_clk);

        i_reset = 1'b0;

    end
    endtask


    // =====================================================
    // MEMORY WRITE
    //
    // Writes into ping-pong WRITE bank.
    // =====================================================

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


    // =====================================================
    // SWAP
    // =====================================================

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


    // =====================================================
    // START LOADER
    // =====================================================

    task start_loader;

        input integer base_addr;
        input integer k_len;
        input integer step;

    begin

        @(negedge i_clk);

        i_base_addr   = base_addr;
        i_k_len       = k_len;
        i_window_step = step;

        i_start = 1'b1;


        @(posedge i_clk);


        @(negedge i_clk);

        i_start = 1'b0;

    end
    endtask


    // =====================================================
    // CAPTURE OUTPUT WRITES
    //
    // This models the write side of the existing
    // IFMAP feeder FIFOs.
    // =====================================================

    always @(posedge i_clk) begin

        for(r = 0;
            r < ARRAY_ROWS;
            r = r + 1) begin

            if(ifm_wr_en[r]) begin

                captured[r][row_count[r]] =
                    $signed(
                        ifm_wr_data[
                            r*DATA_W
                            +: DATA_W
                        ]
                    );


                $display(
                    "[CAPTURE] row=%0d k=%0d data=%0d addr=%0d",
                    r,
                    row_count[r],
                    $signed(
                        ifm_wr_data[
                            r*DATA_W
                            +: DATA_W
                        ]
                    ),
                    sram_rd_addr
                );


                row_count[r] =
                    row_count[r] + 1;

            end

        end

    end


    // =====================================================
    // CHECK ONE CAPTURED VALUE
    // =====================================================

    task check_value;

        input integer row;
        input integer idx;
        input integer expected;

    begin

        if(captured[row][idx] !== expected) begin

            $display(
                "[FAIL] row%0d[%0d] got=%0d expected=%0d",
                row,
                idx,
                captured[row][idx],
                expected
            );

            errors = errors + 1;

        end
        else begin

            $display(
                "[PASS] row%0d[%0d] = %0d",
                row,
                idx,
                captured[row][idx]
            );

        end

    end
    endtask


    // =====================================================
    // MAIN
    // =====================================================

    initial begin

        errors = 0;


        // =================================================
        // TC1
        //
        // X = [1 2 3 4 5 6 7 8]
        //
        // Cin    = 1
        // Kw     = 3
        // stride = 1
        //
        // K           = 3
        // window_step = 1
        //
        // EXPECT:
        //
        // row0 = [1 2 3]
        // row1 = [2 3 4]
        // row2 = [3 4 5]
        // =================================================

        reset_dut();


        $display("");
        $display("======================================");
        $display(" TC1 : Cin=1 Kw=3 Stride=1");
        $display("======================================");


        write_mem(0, 1);
        write_mem(1, 2);
        write_mem(2, 3);
        write_mem(3, 4);
        write_mem(4, 5);
        write_mem(5, 6);
        write_mem(6, 7);
        write_mem(7, 8);


        // Written data currently sits in WRITE bank.
        // Make it the READ bank.

        swap_mem();


        start_loader(
            0,      // base
            3,      // K = Kw*Cin
            1       // stride*Cin
        );


        wait(loader_done == 1'b1);

        @(posedge i_clk);
        #1;


        if(row_count[0] !== 3 ||
           row_count[1] !== 3 ||
           row_count[2] !== 3) begin

            $display(
                "[FAIL] wrong row counts: %0d %0d %0d",
                row_count[0],
                row_count[1],
                row_count[2]
            );

            errors = errors + 1;

        end


        check_value(0,0,1);
        check_value(0,1,2);
        check_value(0,2,3);

        check_value(1,0,2);
        check_value(1,1,3);
        check_value(1,2,4);

        check_value(2,0,3);
        check_value(2,1,4);
        check_value(2,2,5);


        // =================================================
        // TC2
        //
        // Cin=2
        //
        // Tensor:
        //
        // t0 [1  10]
        // t1 [2  20]
        // t2 [3  30]
        // t3 [4  40]
        // t4 [5  50]
        //
        // Memory:
        //
        // 1,10,2,20,3,30,4,40,5,50
        //
        // Kw=3
        // K=6
        // step=2
        //
        // EXPECT:
        //
        // row0 = 1 10 2 20 3 30
        // row1 = 2 20 3 30 4 40
        // row2 = 3 30 4 40 5 50
        // =================================================

        reset_dut();


        $display("");
        $display("======================================");
        $display(" TC2 : Cin=2 Kw=3 Stride=1");
        $display("======================================");


        write_mem(0, 1);
        write_mem(1, 10);

        write_mem(2, 2);
        write_mem(3, 20);

        write_mem(4, 3);
        write_mem(5, 30);

        write_mem(6, 4);
        write_mem(7, 40);

        write_mem(8, 5);
        write_mem(9, 50);


        swap_mem();


        start_loader(
            0,
            6,
            2
        );


        wait(loader_done == 1'b1);

        @(posedge i_clk);
        #1;


        // ROW0

        check_value(0,0,1);
        check_value(0,1,10);
        check_value(0,2,2);
        check_value(0,3,20);
        check_value(0,4,3);
        check_value(0,5,30);


        // ROW1

        check_value(1,0,2);
        check_value(1,1,20);
        check_value(1,2,3);
        check_value(1,3,30);
        check_value(1,4,4);
        check_value(1,5,40);


        // ROW2

        check_value(2,0,3);
        check_value(2,1,30);
        check_value(2,2,4);
        check_value(2,3,40);
        check_value(2,4,5);
        check_value(2,5,50);


        // =================================================
        // SUMMARY
        // =================================================

        $display("");
        $display("======================================");
        $display(" CONV1D IFMAP FEEDER TEST SUMMARY");
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


    // =====================================================
    // DEBUG
    // =====================================================

    always @(posedge i_clk) begin

        if(!i_reset) begin

            $display(
                "T=%0t state=%0d busy=%b done=%b rd_en=%b addr=%0d rd_valid=%b",
                $time,
                loader_state,
                loader_busy,
                loader_done,
                sram_rd_en,
                sram_rd_addr,
                sram_rd_valid
            );

        end

    end


    // =====================================================
    // TIMEOUT
    // =====================================================

    initial begin

        #100000;

        $display("");
        $display(
            "[TIMEOUT] conv1d_ifmap_feeder"
        );

        $finish;

    end


endmodule