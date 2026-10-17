`timescale 1ns/1ps

module tb_wallace_tree_multiplier;

    // ============================================================
    // Parameters
    // ============================================================

    parameter DATA_W = 8;
    localparam PROD_W = 2 * DATA_W;

    localparam MIN_VAL = -(1 << (DATA_W-1));
    localparam MAX_VAL =  (1 << (DATA_W-1)) - 1;


    // ============================================================
    // DUT signals
    // ============================================================

    reg  [DATA_W-1:0] i_a;
    reg  [DATA_W-1:0] i_b;

    wire [PROD_W-1:0] o_p;


    // ============================================================
    // Test control
    // ============================================================

    integer test_count;
    integer pass_count;
    integer fail_count;

    integer i;
    integer j;

    reg signed [PROD_W-1:0] expected;


    // ============================================================
    // DUT
    // ============================================================

    wallace_tree_multiplier #(
        .DATA_W(DATA_W)
    ) DUT (
        .i_a(i_a),
        .i_b(i_b),
        .o_p(o_p)
    );


    // ============================================================
    // Task: check one test case
    // ============================================================

    task check_case;

        input integer a_value;
        input integer b_value;

        begin

            // Drive two's-complement bit patterns to DUT
            i_a = a_value;
            i_b = b_value;

            // Reference model
            expected = a_value * b_value;

            // DUT is combinational
            #1;

            test_count = test_count + 1;


            // ----------------------------------------------------
            // Detect X/Z first
            // ----------------------------------------------------

            if (^o_p === 1'bx) begin

                fail_count = fail_count + 1;

                $display(
                    "[FAIL-X] Test=%0d A=%0d B=%0d DUT=%b",
                    test_count,
                    a_value,
                    b_value,
                    o_p
                );

            end


            // ----------------------------------------------------
            // Functional comparison
            // ----------------------------------------------------

            else if (o_p !== expected) begin

                fail_count = fail_count + 1;

                $display(
                    "[FAIL] Test=%0d A=%0d B=%0d | DUT=%0d (0x%h) | EXP=%0d (0x%h)",
                    test_count,
                    a_value,
                    b_value,
                    $signed(o_p),
                    o_p,
                    expected,
                    expected
                );

            end


            // ----------------------------------------------------
            // PASS
            // ----------------------------------------------------

            else begin

                pass_count = pass_count + 1;

            end

        end

    endtask


    // ============================================================
    // Main test sequence
    // ============================================================

    initial begin

        // --------------------------------------------------------
        // Initialization
        // --------------------------------------------------------

        i_a = 0;
        i_b = 0;

        test_count = 0;
        pass_count = 0;
        fail_count = 0;

        #1;


        $display("");
        $display("============================================================");
        $display(" Baugh-Wooley Wallace Tree Multiplier Verification");
        $display(" DATA_W      = %0d", DATA_W);
        $display(" PRODUCT_W   = %0d", PROD_W);
        $display(" SIGNED RANGE = %0d to %0d", MIN_VAL, MAX_VAL);
        $display("============================================================");
        $display("");


        // ========================================================
        // Phase 1: Directed corner cases
        // ========================================================

        $display("[INFO] Running directed corner cases...");


        // Zero
        check_case(0, 0);
        check_case(0, 1);
        check_case(1, 0);
        check_case(0, -1);
        check_case(-1, 0);


        // Basic sign combinations
        check_case(1, 1);
        check_case(1, -1);
        check_case(-1, 1);
        check_case(-1, -1);

        check_case(2, 3);
        check_case(-2, 3);
        check_case(2, -3);
        check_case(-2, -3);


        // Maximum positive
        check_case(MAX_VAL, MAX_VAL);
        check_case(MAX_VAL, 1);
        check_case(1, MAX_VAL);
        check_case(MAX_VAL, -1);
        check_case(-1, MAX_VAL);


        // Most-negative number
        check_case(MIN_VAL, MIN_VAL);
        check_case(MIN_VAL, 1);
        check_case(1, MIN_VAL);
        check_case(MIN_VAL, -1);
        check_case(-1, MIN_VAL);


        // Opposite extremes
        check_case(MIN_VAL, MAX_VAL);
        check_case(MAX_VAL, MIN_VAL);


        $display(
            "[INFO] Directed tests complete. Current FAIL=%0d",
            fail_count
        );


        // ========================================================
        // Phase 2: Exhaustive verification
        // ========================================================

        $display("");
        $display(
            "[INFO] Starting exhaustive verification: %0d x %0d cases...",
            (MAX_VAL-MIN_VAL+1),
            (MAX_VAL-MIN_VAL+1)
        );


        for (i = MIN_VAL; i <= MAX_VAL; i = i + 1) begin

            for (j = MIN_VAL; j <= MAX_VAL; j = j + 1) begin

                check_case(i, j);

            end

        end


        // ========================================================
        // Final report
        // ========================================================

        $display("");
        $display("============================================================");
        $display(" FINAL VERIFICATION REPORT");
        $display("============================================================");

        $display(" Total tests : %0d", test_count);
        $display(" Passed      : %0d", pass_count);
        $display(" Failed      : %0d", fail_count);


        if (fail_count == 0) begin

            $display("");
            $display(" >>> RESULT: PASS <<<");
            $display(" Multiplier functional verification successful.");

        end

        else begin

            $display("");
            $display(" >>> RESULT: FAIL <<<");
            $display(" Functional errors detected.");

        end


        $display("============================================================");
        $display("");

        $finish;

    end


    // ============================================================
    // Optional waveform dump
    // ============================================================

    // initial begin

    //     $dumpfile("wallace_tree_multiplier.vcd");
    //     $dumpvars(0, tb_wallace_tree_multiplier);

    // end


endmodule