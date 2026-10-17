`timescale 1ns/1ps

module tb_conv1d_addr_gen;

    localparam ADDR_W = 16;
    localparam CFG_W  = 16;


    reg i_clk;
    reg i_reset;

    reg i_start;
    reg i_advance;

    reg [ADDR_W-1:0] i_base_addr;
    reg [CFG_W-1:0]  i_k_len;
    reg [ADDR_W-1:0] i_window_step;
    reg [CFG_W-1:0]  i_num_windows;


    wire [ADDR_W-1:0] o_addr;
    wire o_valid;

    wire o_k_first;
    wire o_k_last;

    wire o_window_first;
    wire o_window_last;

    wire o_busy;
    wire o_done;


    integer errors;


    // =====================================================
    // DUT
    // =====================================================

    conv1d_address_generating #(
        .ADDR_W(ADDR_W),
        .CFG_W(CFG_W)
    ) dut (
        .i_clk(i_clk),
        .i_reset(i_reset),

        .i_start(i_start),
        .i_advance(i_advance),

        .i_base_addr(i_base_addr),
        .i_k_len(i_k_len),
        .i_window_step(i_window_step),
        .i_num_windows(i_num_windows),

        .o_addr(o_addr),
        .o_valid(o_valid),

        .o_k_first(o_k_first),
        .o_k_last(o_k_last),

        .o_window_first(o_window_first),
        .o_window_last(o_window_last),

        .o_busy(o_busy),
        .o_done(o_done)
    );


    // =====================================================
    // CLOCK
    // =====================================================

    initial begin

        i_clk = 0;

        forever #5
            i_clk = ~i_clk;

    end


    // =====================================================
    // RESET
    // =====================================================

    task reset_dut;

    begin

        i_reset = 1'b1;

        i_start   = 1'b0;
        i_advance = 1'b0;

        i_base_addr   = 0;
        i_k_len       = 0;
        i_window_step = 0;
        i_num_windows = 0;


        repeat(3)
            @(posedge i_clk);


        @(negedge i_clk);

        i_reset = 1'b0;

    end
    endtask


    // =====================================================
    // START AGU
    // =====================================================

    task start_agu;

        input integer base_addr;
        input integer k_len;
        input integer step;
        input integer num_windows;

    begin

        @(negedge i_clk);

        i_base_addr   = base_addr;
        i_k_len       = k_len;
        i_window_step = step;
        i_num_windows = num_windows;

        i_start = 1'b1;


        @(posedge i_clk);


        @(negedge i_clk);

        i_start = 1'b0;

    end
    endtask


    // =====================================================
    // CHECK CURRENT ADDRESS THEN ADVANCE
    // =====================================================

    task check_and_advance;

        input integer expected_addr;

    begin

        #1;

        if(o_valid !== 1'b1) begin

            $display(
                "[FAIL] valid=0 expected addr=%0d",
                expected_addr
            );

            errors = errors + 1;

        end


        if(o_addr !== expected_addr) begin

            $display(
                "[FAIL] addr=%0d expected=%0d",
                o_addr,
                expected_addr
            );

            errors = errors + 1;

        end
        else begin

            $display(
                "[PASS] addr=%0d",
                o_addr
            );

        end


        @(negedge i_clk);

        i_advance = 1'b1;


        @(posedge i_clk);


        @(negedge i_clk);

        i_advance = 1'b0;

    end
    endtask


    // =====================================================
    // MAIN
    // =====================================================

    initial begin

        errors = 0;

        reset_dut();


        // =================================================
        // TC1
        //
        // Cin=1
        // Kw=3
        // stride=1
        //
        // K=3
        // step=1
        //
        // windows:
        // 0 1 2
        // 1 2 3
        // 2 3 4
        // =================================================

        $display("");
        $display("======================================");
        $display(" TC1 : Cin=1 Kw=3 Stride=1");
        $display("======================================");


        start_agu(
            0,
            3,
            1,
            3
        );


        check_and_advance(0);
        check_and_advance(1);
        check_and_advance(2);

        check_and_advance(1);
        check_and_advance(2);
        check_and_advance(3);

        check_and_advance(2);
        check_and_advance(3);
        check_and_advance(4);


        // =================================================
        // TC2
        //
        // Cin=2
        // Kw=3
        //
        // K=6
        // step=2
        //
        // windows:
        //
        // 0 1 2 3 4 5
        // 2 3 4 5 6 7
        // 4 5 6 7 8 9
        // =================================================

        reset_dut();


        $display("");
        $display("======================================");
        $display(" TC2 : Cin=2 Kw=3 Stride=1");
        $display("======================================");


        start_agu(
            0,
            6,
            2,
            3
        );


        check_and_advance(0);
        check_and_advance(1);
        check_and_advance(2);
        check_and_advance(3);
        check_and_advance(4);
        check_and_advance(5);

        check_and_advance(2);
        check_and_advance(3);
        check_and_advance(4);
        check_and_advance(5);
        check_and_advance(6);
        check_and_advance(7);

        check_and_advance(4);
        check_and_advance(5);
        check_and_advance(6);
        check_and_advance(7);
        check_and_advance(8);
        check_and_advance(9);


        // =================================================
        // TC3
        //
        // stride=2
        // Cin=2
        //
        // step = 4
        // =================================================

        reset_dut();


        $display("");
        $display("======================================");
        $display(" TC3 : STRIDE=2 Cin=2");
        $display("======================================");


        start_agu(
            0,
            6,
            4,
            3
        );


        // window0
        check_and_advance(0);
        check_and_advance(1);
        check_and_advance(2);
        check_and_advance(3);
        check_and_advance(4);
        check_and_advance(5);


        // window1
        check_and_advance(4);
        check_and_advance(5);
        check_and_advance(6);
        check_and_advance(7);
        check_and_advance(8);
        check_and_advance(9);


        // window2
        check_and_advance(8);
        check_and_advance(9);
        check_and_advance(10);
        check_and_advance(11);
        check_and_advance(12);
        check_and_advance(13);


        // =================================================
        // TC4
        // BASE ADDRESS
        // =================================================

        reset_dut();


        $display("");
        $display("======================================");
        $display(" TC4 : BASE ADDRESS = 100");
        $display("======================================");


        start_agu(
            100,
            3,
            1,
            2
        );


        check_and_advance(100);
        check_and_advance(101);
        check_and_advance(102);

        check_and_advance(101);
        check_and_advance(102);
        check_and_advance(103);


        // =================================================
        // TC5
        // STALL TEST
        // =================================================

        reset_dut();


        $display("");
        $display("======================================");
        $display(" TC5 : STALL / HOLD ADDRESS");
        $display("======================================");


        start_agu(
            0,
            3,
            1,
            1
        );


        #1;

        if(o_addr !== 0) begin
            $display("[FAIL] initial addr");
            errors = errors + 1;
        end


        // Do NOT advance for 3 clocks
        repeat(3) begin

            @(posedge i_clk);

            #1;

            if(o_addr !== 0) begin

                $display(
                    "[FAIL] AGU moved during stall"
                );

                errors = errors + 1;

            end

            else begin

                $display(
                    "[PASS] stall holds addr=0"
                );

            end

        end


        check_and_advance(0);
        check_and_advance(1);
        check_and_advance(2);


        // =================================================
        // SUMMARY
        // =================================================

        $display("");
        $display("======================================");
        $display(" CONV1D AGU TEST SUMMARY");
        $display("======================================");


        if(errors == 0)
            $display(" RESULT : ALL TESTS PASSED");
        else
            $display(
                " RESULT : FAILED (%0d errors)",
                errors
            );


        $display("======================================");


        repeat(3)
            @(posedge i_clk);

        $finish;

    end


    // =====================================================
    // TIMEOUT
    // =====================================================

    initial begin

        #50000;

        $display(
            "[TIMEOUT] conv1d_agu timeout"
        );

        $finish;

    end


endmodule