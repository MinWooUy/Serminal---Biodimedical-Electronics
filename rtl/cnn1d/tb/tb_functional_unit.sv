`timescale 1ns / 1ps

module tb_functional_unit;
    localparam int OUT_WIDTH = 8;

    logic clk = 1'b0;
    logic rst_n = 1'b0;
    logic [3:0] alu_op = '0;
    logic [1:0] lut_mode = '0;
    logic [$clog2(72)-1:0] shift_amount = '0;
    logic valid_in = 1'b0;
    logic valid_out;
    logic [47:0] in_data_a0 = '0;
    logic [39:0] in_data_b0 = '0;
    logic [31:0] in_data_a1 = '0;
    logic [31:0] in_data_b1 = '0;
    logic [63:0] psum_in = '0;
    logic [OUT_WIDTH-1:0] sram_c_data;
    logic sram_c_we;

    int pass_count = 0;
    int fail_count = 0;

    always #5 clk = ~clk;

    functional_unit #(
        .OUT_WIDTH(OUT_WIDTH)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .alu_op(alu_op),
        .lut_mode(lut_mode),
        .shift_amount(shift_amount),
        .valid_in(valid_in),
        .valid_out(valid_out),
        .in_data_a0(in_data_a0),
        .in_data_b0(in_data_b0),
        .in_data_a1(in_data_a1),
        .in_data_b1(in_data_b1),
        .psum_in(psum_in),
        .sram_c_data(sram_c_data),
        .sram_c_we(sram_c_we)
    );

    task automatic run_case(
        input integer a,
        input integer b,
        input integer psum,
        input integer shift,
        input logic [3:0] operation,
        input logic [1:0] mode,
        input integer expected,
        input string name
    );
        integer actual;
        integer cycles;
        begin
            @(negedge clk);
            in_data_a0 = a;
            in_data_b0 = b;
            psum_in = psum;
            shift_amount = shift;
            alu_op = operation;
            lut_mode = mode;
            valid_in = 1'b1;

            @(negedge clk);
            valid_in = 1'b0;

            cycles = 0;
            while (!valid_out && cycles < 8) begin
                @(negedge clk);
                cycles = cycles + 1;
            end

            if (!valid_out) begin
                $display("FAIL %-20s: valid_out timeout", name);
                fail_count = fail_count + 1;
            end else begin
                actual = $signed(sram_c_data);
                if (actual !== expected || sram_c_we !== 1'b1) begin
                    $display("FAIL %-20s: expected=%0d actual=%0d data=%h we=%b", name, expected, actual, sram_c_data, sram_c_we);
                    fail_count = fail_count + 1;
                end else begin
                    $display("PASS %-20s: result=%0d latency=%0d cycles", name, actual, cycles + 1);
                    pass_count = pass_count + 1;
                end
            end
        end
    endtask

    initial begin
        repeat (2) @(negedge clk);
        rst_n = 1'b1;

        // Fake LUT contents: address 12 returns 20.
        dut.lut_table[12] = 16'd20;

        // Path 0: multiply, add PSUM, round-shift, then clamp.
        run_case(3, 4, 2, 1, 4'b0000, 2'b00, 7, "mac_round_shift");
        run_case(-3, 4, 0, 1, 4'b0000, 2'b00, -6, "signed_round_shift");
        run_case(100, 2, 0, 0, 4'b0000, 2'b00, 127, "positive_clamp");
        run_case(-100, 2, 0, 0, 4'b0000, 2'b00, -128, "negative_clamp");

        // LUT path: 3 * 4 = 12 selects the fake LUT entry.
        run_case(3, 4, 0, 0, 4'b1111, 2'b00, 20, "lut_direct");
        run_case(3, 4, 0, 0, 4'b1111, 2'b01, 10, "lut_shift_right");

        @(negedge clk);
        if (valid_out !== 1'b0 || sram_c_we !== 1'b0) begin
            $display("FAIL output_valid_clear: valid_out=%b we=%b", valid_out, sram_c_we);
            fail_count = fail_count + 1;
        end else begin
            $display("PASS output_valid_clear");
            pass_count = pass_count + 1;
        end

        $display("SUMMARY: %0d passed, %0d failed", pass_count, fail_count);
        if (fail_count != 0) $fatal(1, "functional_unit testbench failed");
        $finish;
    end
endmodule
