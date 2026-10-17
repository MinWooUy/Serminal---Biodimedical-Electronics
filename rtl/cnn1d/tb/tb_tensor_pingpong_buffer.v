`timescale 1ns/1ps

module tb_tensor_pingpong_buffer;

    localparam DATA_W = 8;
    localparam DEPTH  = 16;
    localparam ADDR_W = 4;


    // =====================================================
    // SIGNALS
    // =====================================================

    reg i_clk;
    reg i_reset;

    reg i_swap;

    reg                     i_wr_en;
    reg [ADDR_W-1:0]        i_wr_addr;
    reg signed [DATA_W-1:0] i_wr_data;

    reg                     i_rd_en;
    reg [ADDR_W-1:0]        i_rd_addr;

    wire signed [DATA_W-1:0] o_rd_data;
    wire                     o_rd_valid;

    wire o_read_bank;
    wire o_write_bank;


    integer errors;


    // =====================================================
    // DUT
    // =====================================================

    tensor_pingpong_buffer #(
        .DATA_W(DATA_W),
        .DEPTH(DEPTH),
        .ADDR_W(ADDR_W)
    ) dut (
        .i_clk(i_clk),
        .i_reset(i_reset),

        .i_swap(i_swap),

        .i_wr_en(i_wr_en),
        .i_wr_addr(i_wr_addr),
        .i_wr_data(i_wr_data),

        .i_rd_en(i_rd_en),
        .i_rd_addr(i_rd_addr),

        .o_rd_data(o_rd_data),
        .o_rd_valid(o_rd_valid),

        .o_read_bank(o_read_bank),
        .o_write_bank(o_write_bank)
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
    begin

        i_reset = 1'b1;

        i_swap = 1'b0;

        i_wr_en   = 1'b0;
        i_wr_addr = 0;
        i_wr_data = 0;

        i_rd_en   = 1'b0;
        i_rd_addr = 0;


        repeat(3)
            @(posedge i_clk);


        @(negedge i_clk);

        i_reset = 1'b0;


        @(posedge i_clk);
        #1;


        if(o_read_bank !== 1'b0) begin

            $display(
                "[FAIL] reset: read_bank expected 0"
            );

            errors = errors + 1;

        end


        if(o_write_bank !== 1'b1) begin

            $display(
                "[FAIL] reset: write_bank expected 1"
            );

            errors = errors + 1;

        end

    end
    endtask


    // =====================================================
    // WRITE WORD
    //
    // Writes into CURRENT WRITE BANK
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
    // READ CHECK
    //
    // Reads from CURRENT READ BANK
    // =====================================================

    task read_check;

        input integer addr;
        input integer expected;

    begin

        @(negedge i_clk);

        i_rd_en   = 1'b1;
        i_rd_addr = addr;


        @(posedge i_clk);

        #1;


        if(o_rd_valid !== 1'b1) begin

            $display(
                "[FAIL] addr=%0d rd_valid=0",
                addr
            );

            errors = errors + 1;

        end


        if($signed(o_rd_data) !== expected) begin

            $display(
                "[FAIL] bank=%0d addr=%0d got=%0d expected=%0d",
                o_read_bank,
                addr,
                $signed(o_rd_data),
                expected
            );

            errors = errors + 1;

        end
        else begin

            $display(
                "[PASS] bank=%0d addr=%0d data=%0d",
                o_read_bank,
                addr,
                $signed(o_rd_data)
            );

        end


        @(negedge i_clk);

        i_rd_en = 1'b0;


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
    // SWAP BANK
    // =====================================================

    task swap_bank;

        reg old_bank;

    begin

        old_bank = o_read_bank;


        // No read/write during swap
        @(negedge i_clk);

        i_rd_en = 1'b0;
        i_wr_en = 1'b0;

        i_swap = 1'b1;


        @(posedge i_clk);

        #1;


        if(o_read_bank !== ~old_bank) begin

            $display(
                "[FAIL] swap: read_bank did not toggle"
            );

            errors = errors + 1;

        end
        else begin

            $display(
                "[PASS] swap: read_bank %0d -> %0d",
                old_bank,
                o_read_bank
            );

        end


        @(negedge i_clk);

        i_swap = 1'b0;

    end
    endtask


    // =====================================================
    // SIMULTANEOUS READ + WRITE
    //
    // Read current read bank
    // Write opposite bank
    // =====================================================

    task read_write_check;

        input integer read_addr;
        input integer expected_read;

        input integer write_addr;
        input integer write_value;

    begin

        @(negedge i_clk);

        i_rd_en   = 1'b1;
        i_rd_addr = read_addr;

        i_wr_en   = 1'b1;
        i_wr_addr = write_addr;
        i_wr_data = write_value;


        @(posedge i_clk);

        #1;


        if(o_rd_valid !== 1'b1) begin

            $display(
                "[FAIL] simultaneous R/W: rd_valid=0"
            );

            errors = errors + 1;

        end


        if($signed(o_rd_data) !== expected_read) begin

            $display(
                "[FAIL] simultaneous R/W read got=%0d expected=%0d",
                $signed(o_rd_data),
                expected_read
            );

            errors = errors + 1;

        end
        else begin

            $display(
                "[PASS] simultaneous R/W read=%0d, write=%0d",
                $signed(o_rd_data),
                write_value
            );

        end


        @(negedge i_clk);

        i_rd_en = 1'b0;
        i_wr_en = 1'b0;

    end
    endtask


    // =====================================================
    // MAIN
    // =====================================================

    initial begin

        errors = 0;

        i_reset = 0;

        i_swap = 0;

        i_wr_en   = 0;
        i_wr_addr = 0;
        i_wr_data = 0;

        i_rd_en   = 0;
        i_rd_addr = 0;


        // =================================================
        // RESET
        //
        // read = bank0
        // write = bank1
        // =================================================

        reset_dut();


        // =================================================
        // PP1
        // FILL BANK1
        // =================================================

        $display("");
        $display("======================================");
        $display(" PP1 : FILL WRITE BANK1");
        $display("======================================");

        if(
            (o_read_bank !== 1'b0) ||
            (o_write_bank !== 1'b1)
        ) begin

            $display(
                "[FAIL] initial ownership wrong"
            );

            errors = errors + 1;

        end
        else begin

            $display(
                "[PASS] READ=bank0 WRITE=bank1"
            );

        end


        write_word(0, 11);
        write_word(1, 22);
        write_word(2, 33);


        // =================================================
        // PP2
        // SWAP AND READ BANK1
        // =================================================

        $display("");
        $display("======================================");
        $display(" PP2 : SWAP -> READ BANK1");
        $display("======================================");


        swap_bank();


        read_check(0, 11);
        read_check(1, 22);
        read_check(2, 33);


        // =================================================
        // PP3
        // READ BANK1 WHILE WRITE BANK0
        // =================================================

        $display("");
        $display("======================================");
        $display(" PP3 : SIMULTANEOUS READ/WRITE");
        $display("======================================");


        // Read bank1[0] = 11
        // Write bank0[0] = 44

        read_write_check(
            0,
            11,
            0,
            44
        );


        // Read bank1[1] = 22
        // Write bank0[1] = 55

        read_write_check(
            1,
            22,
            1,
            55
        );


        // Read bank1[2] = 33
        // Write bank0[2] = -66

        read_write_check(
            2,
            33,
            2,
            -66
        );


        // =================================================
        // PP4
        // SWAP -> READ BANK0
        // =================================================

        $display("");
        $display("======================================");
        $display(" PP4 : SWAP -> READ BANK0");
        $display("======================================");


        swap_bank();


        read_check(0, 44);
        read_check(1, 55);
        read_check(2, -66);


        // =================================================
        // PP5
        // DATA PERSISTENCE
        //
        // Swap back to bank1.
        // Old bank1 contents must still exist.
        // =================================================

        $display("");
        $display("======================================");
        $display(" PP5 : BANK DATA PERSISTENCE");
        $display("======================================");


        swap_bank();


        read_check(0, 11);
        read_check(1, 22);
        read_check(2, 33);


        // =================================================
        // FINAL
        // =================================================

        $display("");
        $display("======================================");
        $display(" PING-PONG BUFFER TEST SUMMARY");
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
    // TIMEOUT
    // =====================================================

    initial begin

        #20000;

        $display("");
        $display(
            "[TIMEOUT] tensor_pingpong_buffer timeout"
        );

        $finish;

    end


endmodule