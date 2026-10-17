`timescale 1ns/1ps

module tb_tensor_sram;

    localparam DATA_W = 8;
    localparam DEPTH  = 16;
    localparam ADDR_W = 4;


    reg i_clk;
    reg i_reset;


    reg                     i_wr_en;
    reg [ADDR_W-1:0]        i_wr_addr;
    reg signed [DATA_W-1:0] i_wr_data;


    reg                     i_rd_en;
    reg [ADDR_W-1:0]        i_rd_addr;

    wire signed [DATA_W-1:0] o_rd_data;
    wire                     o_rd_valid;


    integer errors;


    // =====================================================
    // DUT
    // =====================================================

    tensor_sram #(
        .DATA_W(DATA_W),
        .DEPTH(DEPTH),
        .ADDR_W(ADDR_W)
    ) dut (
        .i_clk(i_clk),
        .i_reset(i_reset),

        .i_wr_en(i_wr_en),
        .i_wr_addr(i_wr_addr),
        .i_wr_data(i_wr_data),

        .i_rd_en(i_rd_en),
        .i_rd_addr(i_rd_addr),

        .o_rd_data(o_rd_data),
        .o_rd_valid(o_rd_valid)
    );


    // =====================================================
    // CLOCK
    // =====================================================

    initial begin
        i_clk = 1'b0;
        forever #5 i_clk = ~i_clk;
    end


    // =====================================================
    // RESET
    // =====================================================

    task reset_dut;
    begin

        i_reset = 1'b1;

        i_wr_en   = 1'b0;
        i_wr_addr = 0;
        i_wr_data = 0;

        i_rd_en   = 1'b0;
        i_rd_addr = 0;

        repeat(3)
            @(posedge i_clk);

        @(negedge i_clk);

        i_reset = 1'b0;

    end
    endtask


    // =====================================================
    // WRITE ONE WORD
    // =====================================================

    task write_word;

        input integer addr;
        input integer value;

    begin

        @(negedge i_clk);

        i_wr_en   = 1'b1;
        i_wr_addr = addr;
        i_wr_data = value;

        @(posedge i_clk);

        @(negedge i_clk);

        i_wr_en = 1'b0;

    end
    endtask


    // =====================================================
    // READ + CHECK
    // =====================================================

    task read_check;

        input integer addr;
        input integer expected;

    begin

        // Request read
        @(negedge i_clk);

        i_rd_en   = 1'b1;
        i_rd_addr = addr;


        // SRAM samples address here
        @(posedge i_clk);

        #1;


        if(o_rd_valid !== 1'b1) begin

            $display(
                "[FAIL] addr=%0d : rd_valid=0",
                addr
            );

            errors = errors + 1;

        end


        if($signed(o_rd_data) !== expected) begin

            $display(
                "[FAIL] addr=%0d got=%0d expected=%0d",
                addr,
                $signed(o_rd_data),
                expected
            );

            errors = errors + 1;

        end
        else begin

            $display(
                "[PASS] addr=%0d data=%0d",
                addr,
                $signed(o_rd_data)
            );

        end


        @(negedge i_clk);

        i_rd_en = 1'b0;


        // Check valid drops
        @(posedge i_clk);

        #1;

        if(o_rd_valid !== 1'b0) begin

            $display(
                "[FAIL] rd_valid should return to 0"
            );

            errors = errors + 1;

        end

    end
    endtask


    // =====================================================
    // MAIN
    // =====================================================

    initial begin

        errors = 0;

        i_reset = 0;

        i_wr_en   = 0;
        i_wr_addr = 0;
        i_wr_data = 0;

        i_rd_en   = 0;
        i_rd_addr = 0;


        reset_dut();


        // =================================================
        // TC1 - Positive value
        // =================================================

        $display("");
        $display("======================================");
        $display(" TC1 : POSITIVE VALUE");
        $display("======================================");

        write_word(0, 12);

        read_check(0, 12);


        // =================================================
        // TC2 - Signed negative
        // =================================================

        $display("");
        $display("======================================");
        $display(" TC2 : SIGNED NEGATIVE");
        $display("======================================");

        write_word(1, -7);

        read_check(1, -7);


        // =================================================
        // TC3 - INT8 limits
        // =================================================

        $display("");
        $display("======================================");
        $display(" TC3 : INT8 BOUNDARY");
        $display("======================================");

        write_word(2, 127);
        write_word(3, -128);

        read_check(2, 127);
        read_check(3, -128);


        // =================================================
        // TC4 - Different addresses
        // =================================================

        $display("");
        $display("======================================");
        $display(" TC4 : MULTIPLE ADDRESSES");
        $display("======================================");

        write_word(5, 33);
        write_word(9, -44);
        write_word(15, 88);

        read_check(5,  33);
        read_check(9, -44);
        read_check(15, 88);


        // =================================================
        // FINAL
        // =================================================

        $display("");
        $display("======================================");
        $display(" TENSOR SRAM TEST SUMMARY");
        $display("======================================");

        if(errors == 0) begin

            $display(" RESULT : ALL TESTS PASSED");

        end
        else begin

            $display(
                " RESULT : FAILED (%0d errors)",
                errors
            );

        end

        $display("======================================");


        repeat(3)
            @(posedge i_clk);

        $finish;

    end


    // =====================================================
    // TIMEOUT
    // =====================================================

    initial begin

        #10000;

        $display(
            "[TIMEOUT] tensor_sram test timeout"
        );

        $finish;

    end


endmodule